from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..kernels.pack import Pack
from ..kernels.quant_utils import activation_tile_absmax_scales, quantize_activation_tile_absmax, quantize_activation_tile_with_scales
from ..kernels.triton.gemm_fp16_64x64 import gemm_fp16_tiles_add
from ..kernels.triton.gemm_int8_64x64 import gemm_int8_tiles_add


DEFAULT_ACTIVATION_ROW_CHUNK_SIZE = 4096


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MixedLinear(nn.Module):
    def __init__(self, pack: Pack, bias: Optional[torch.Tensor], cfg: Any):
        super().__init__()
        self.cfg = cfg

        self.M_orig = int(pack.M_orig)
        self.K_orig = int(pack.K_orig)
        self.M = int(pack.M)
        self.K = int(pack.K)
        self.tile_m = int(pack.tile_m)
        self.tile_n = int(pack.tile_n)
        self.n_tr = int(pack.n_tr)
        self.n_tc = int(pack.n_tc)

        self.register_buffer("int8_tiles", pack.int8_tiles, persistent=True)
        self.register_buffer("int8_scales", pack.int8_scales, persistent=True)
        self.register_buffer("int8_tile_ids", pack.int8_tile_ids, persistent=True)
        self.register_buffer("fp16_tiles", pack.fp16_tiles, persistent=True)
        self.register_buffer("fp16_tile_ids", pack.fp16_tile_ids, persistent=True)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _out_dtype(self) -> torch.dtype:
        dtype_out = _cfg_get(self.cfg, "dtype_out", "fp16")
        if dtype_out in ("bf16", torch.bfloat16):
            return torch.bfloat16
        return torch.float16

    def _activation_row_chunk_size(self) -> int:
        value = int(_cfg_get(self.cfg, "activation_row_chunk_size", DEFAULT_ACTIVATION_ROW_CHUNK_SIZE))
        return max(1, value)

    def _forward_chunk(self, x_chunk: torch.Tensor, sx: torch.Tensor) -> torch.Tensor:
        x_int8 = quantize_activation_tile_with_scales(x_chunk, sx=sx, tile_n=self.tile_n)
        y_acc = torch.zeros((x_chunk.shape[0], self.M), device=x_chunk.device, dtype=torch.float32)
        gemm_int8_tiles_add(
            x_int8=x_int8,
            sx=sx,
            int8_tiles=self.int8_tiles,
            int8_scales=self.int8_scales,
            int8_tile_ids=self.int8_tile_ids,
            M=self.M,
            K=self.K,
            n_tc=self.n_tc,
            y=y_acc,
        )
        gemm_fp16_tiles_add(
            x_fp=x_chunk,
            fp16_tiles=self.fp16_tiles,
            fp16_tile_ids=self.fp16_tile_ids,
            M=self.M,
            K=self.K,
            n_tc=self.n_tc,
            y=y_acc,
        )

        y = y_acc[:, : self.M_orig].to(self._out_dtype())
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = x.shape
        if x.dim() == 3:
            x2d = x.reshape(-1, x.shape[-1])
            restore_3d = True
        elif x.dim() == 2:
            x2d = x
            restore_3d = False
        else:
            raise ValueError(f"Expected input rank 2 or 3, got shape={tuple(x.shape)}")

        n_rows, k_in = x2d.shape
        if k_in != self.K_orig:
            raise ValueError(f"Input K mismatch: got {k_in}, expected {self.K_orig}")

        if self.K != self.K_orig:
            x_work = torch.zeros((n_rows, self.K), device=x2d.device, dtype=x2d.dtype)
            x_work[:, : self.K_orig] = x2d
        else:
            x_work = x2d

        x_work = x_work.contiguous()
        row_chunk_size = self._activation_row_chunk_size()

        if n_rows <= row_chunk_size:
            x_int8, sx = quantize_activation_tile_absmax(x_work, tile_n=self.tile_n)
            y_acc = torch.zeros((n_rows, self.M), device=x2d.device, dtype=torch.float32)
            gemm_int8_tiles_add(
                x_int8=x_int8,
                sx=sx,
                int8_tiles=self.int8_tiles,
                int8_scales=self.int8_scales,
                int8_tile_ids=self.int8_tile_ids,
                M=self.M,
                K=self.K,
                n_tc=self.n_tc,
                y=y_acc,
            )
            gemm_fp16_tiles_add(
                x_fp=x_work,
                fp16_tiles=self.fp16_tiles,
                fp16_tile_ids=self.fp16_tile_ids,
                M=self.M,
                K=self.K,
                n_tc=self.n_tc,
                y=y_acc,
            )

            y = y_acc[:, : self.M_orig].to(self._out_dtype())
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)
        else:
            sx = activation_tile_absmax_scales(x_work, tile_n=self.tile_n)
            y = torch.cat(
                [
                    self._forward_chunk(x_work[start : start + row_chunk_size].contiguous(), sx)
                    for start in range(0, n_rows, row_chunk_size)
                ],
                dim=0,
            )

        if restore_3d:
            y = y.reshape(in_shape[0], in_shape[1], self.M_orig)
        return y
