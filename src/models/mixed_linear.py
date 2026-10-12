from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..kernels.pack import Pack, concat_packs_along_rows
from ..kernels.quant_utils import (
    activation_tile_absmax_scales,
    quantize_activation_tile_absmax,
    quantize_activation_tile_with_scales,
)
from ..kernels.triton.gemm_rows_splitk_64x64 import gemm_mixed_rows_splitk


DEFAULT_ACTIVATION_ROW_CHUNK_SIZE = 4096
# Sibling projections reuse one activation quantization for prefill-sized inputs.
ACTIVATION_QUANT_CACHE_MIN_ROWS = 32


@dataclass
class _ActivationQuantCacheEntry:
    source_ref: weakref.ReferenceType[torch.Tensor]
    signature: Tuple[Any, ...]
    consumers: set[int]
    x_int8: torch.Tensor
    sx: torch.Tensor


# Q/K/V and gate/up project the same activation independently.  Keep one
# inference-only entry per CUDA stream so those sibling projections can reuse
# the exact same activation quantization without changing the quantization
# formula or the mixed GEMM.
_ACTIVATION_QUANT_CACHE: Dict[Tuple[int, int], _ActivationQuantCacheEntry] = {}


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _activation_version(x: torch.Tensor) -> Optional[int]:
    try:
        return int(x._version)
    except RuntimeError:
        # Inference tensors intentionally do not expose a version counter.
        return None


def _quantize_shared_activation(
    source: torch.Tensor,
    x_work: torch.Tensor,
    tile_n: int,
    consumer_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        int(x_work.shape[0]) < ACTIVATION_QUANT_CACHE_MIN_ROWS
        or not (torch.is_inference_mode_enabled() and x_work.is_cuda)
    ):
        return quantize_activation_tile_absmax(x_work, tile_n=tile_n)

    stream_id = int(torch.cuda.current_stream(device=x_work.device).cuda_stream)
    slot = (int(x_work.device.index or 0), stream_id)
    signature = (
        int(x_work.data_ptr()),
        int(x_work.storage_offset()),
        tuple(x_work.shape),
        tuple(x_work.stride()),
        x_work.dtype,
        int(tile_n),
        _activation_version(source),
    )
    cached = _ACTIVATION_QUANT_CACHE.get(slot)
    if (
        cached is not None
        and cached.source_ref() is source
        and cached.signature == signature
        and consumer_id not in cached.consumers
    ):
        cached.consumers.add(consumer_id)
        return cached.x_int8, cached.sx

    # Drop the previous entry before allocating the replacement so an
    # unrelated activation is not kept alive across layers or iterations.
    _ACTIVATION_QUANT_CACHE.pop(slot, None)
    x_int8, sx = quantize_activation_tile_absmax(x_work, tile_n=tile_n)

    try:
        source_ref: weakref.ReferenceType[torch.Tensor]

        def evict(dead_ref: weakref.ReferenceType[torch.Tensor]) -> None:
            current = _ACTIVATION_QUANT_CACHE.get(slot)
            if current is not None and current.source_ref is dead_ref:
                _ACTIVATION_QUANT_CACHE.pop(slot, None)

        source_ref = weakref.ref(source, evict)
    except TypeError:
        return x_int8, sx

    _ACTIVATION_QUANT_CACHE[slot] = _ActivationQuantCacheEntry(
        source_ref=source_ref,
        signature=signature,
        consumers={consumer_id},
        x_int8=x_int8,
        sx=sx,
    )
    return x_int8, sx


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
        self.register_buffer("int8_row_ptr", pack.int8_row_ptr, persistent=True)
        self.register_buffer("int8_row_indices", pack.int8_row_indices, persistent=True)
        self.register_buffer("fp16_tiles", pack.fp16_tiles, persistent=True)
        self.register_buffer("fp16_tile_ids", pack.fp16_tile_ids, persistent=True)
        self.register_buffer("fp16_row_ptr", pack.fp16_row_ptr, persistent=True)
        self.register_buffer("fp16_row_indices", pack.fp16_row_indices, persistent=True)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def to_pack(self) -> Pack:
        return Pack(
            M_orig=self.M_orig,
            K_orig=self.K_orig,
            M=self.M,
            K=self.K,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            n_tr=self.n_tr,
            n_tc=self.n_tc,
            int8_tiles=self.int8_tiles,
            int8_scales=self.int8_scales,
            int8_tile_ids=self.int8_tile_ids,
            int8_row_ptr=self.int8_row_ptr,
            int8_row_indices=self.int8_row_indices,
            fp16_tiles=self.fp16_tiles,
            fp16_tile_ids=self.fp16_tile_ids,
            fp16_row_ptr=self.fp16_row_ptr,
            fp16_row_indices=self.fp16_row_indices,
        )

    def _out_dtype(self) -> torch.dtype:
        dtype_out = _cfg_get(self.cfg, "dtype_out", "fp16")
        if dtype_out in ("bf16", torch.bfloat16):
            return torch.bfloat16
        return torch.float16

    def _activation_row_chunk_size(self) -> int:
        value = int(_cfg_get(self.cfg, "activation_row_chunk_size", DEFAULT_ACTIVATION_ROW_CHUNK_SIZE))
        return max(1, value)

    def _out_bias(self, out_dtype: torch.dtype) -> Optional[torch.Tensor]:
        bias = self.bias
        if bias is None:
            return None
        if bias.dtype != out_dtype or not bias.is_contiguous():
            bias = bias.to(dtype=out_dtype).contiguous()
        return bias

    def _gemm_bias(
        self,
        x_fp: torch.Tensor,
        x_int8: torch.Tensor,
        sx: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_dtype = self._out_dtype()
        return gemm_mixed_rows_splitk(
            x_fp=x_fp,
            x_int8=x_int8,
            sx=sx,
            int8_tiles=self.int8_tiles,
            int8_scales=self.int8_scales,
            int8_tile_ids=self.int8_tile_ids,
            int8_row_ptr=self.int8_row_ptr,
            fp16_tiles=self.fp16_tiles,
            fp16_tile_ids=self.fp16_tile_ids,
            fp16_row_ptr=self.fp16_row_ptr,
            M=self.M_orig,
            K=self.K_orig,
            n_tr=self.n_tr,
            n_tc=self.n_tc,
            out_dtype=out_dtype,
            bias=self._out_bias(out_dtype),
            out=out,
        )

    def _forward_chunk(self, x_chunk: torch.Tensor, sx: torch.Tensor, out: torch.Tensor) -> None:
        x_int8 = quantize_activation_tile_with_scales(x_chunk, sx=sx, tile_n=self.tile_n)
        self._gemm_bias(x_fp=x_chunk, x_int8=x_int8, sx=sx, out=out)

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

        x_work = x2d.contiguous()
        row_chunk_size = self._activation_row_chunk_size()

        if n_rows <= row_chunk_size:
            x_int8, sx = _quantize_shared_activation(
                source=x,
                x_work=x_work,
                tile_n=self.tile_n,
                consumer_id=id(self),
            )
            y = self._gemm_bias(x_fp=x_work, x_int8=x_int8, sx=sx)
        else:
            sx = activation_tile_absmax_scales(x_work, tile_n=self.tile_n)
            # Chunks write straight into their row slice of one output (no
            # concatenation copy, no per-chunk outputs fragmenting the cache).
            y = torch.empty((n_rows, self.M_orig), device=x_work.device, dtype=self._out_dtype())
            for start in range(0, n_rows, row_chunk_size):
                self._forward_chunk(
                    x_work[start : start + row_chunk_size],
                    sx,
                    y[start : start + row_chunk_size],
                )

        if restore_3d:
            y = y.reshape(in_shape[0], in_shape[1], self.M_orig)
        return y


@dataclass
class _PendingGroupOutput:
    source_ref: weakref.ReferenceType[torch.Tensor]
    signature: Tuple[Any, ...]
    consumed: set[int]
    output: torch.Tensor


class MixedLinearGroup(nn.Module):
    """Sibling projections of one activation executed as one stacked GEMM.

    Q/K/V and Gate/Up read the same input.  Stacking their packs along the
    output rows quantizes the activation once
    and launches one mixed GEMM; every output element is produced by the same
    tile sequence and arithmetic as the member's own MixedLinear.
    """

    def __init__(self, members: "list[MixedLinear]"):
        super().__init__()
        if len(members) < 2:
            raise ValueError("A projection group needs at least two members")
        has_bias = [member.bias is not None for member in members]
        if any(has_bias) and not all(has_bias):
            raise ValueError("Projection group members must all have or all lack bias")
        packs = [member.to_pack() for member in members]
        bias = torch.cat([member.bias.detach() for member in members]) if all(has_bias) else None
        self.linear = MixedLinear(concat_packs_along_rows(packs), bias, members[0].cfg)
        bounds = []
        start = 0
        for member in members:
            bounds.append((start, start + member.M_orig))
            start += member.M_orig
        self.bounds = tuple(bounds)
        self._pending: Optional[_PendingGroupOutput] = None

    def _signature(self, x: torch.Tensor) -> Tuple[Any, ...]:
        return (
            int(x.data_ptr()),
            int(x.storage_offset()),
            tuple(x.shape),
            tuple(x.stride()),
            x.dtype,
            x.device,
            _activation_version(x),
        )

    def output_for(self, x: torch.Tensor, index: int) -> torch.Tensor:
        pending = self._pending
        signature = self._signature(x)
        if (
            pending is not None
            and pending.source_ref() is x
            and pending.signature == signature
            and index not in pending.consumed
        ):
            pending.consumed.add(index)
            output = pending.output
            if len(pending.consumed) == len(self.bounds):
                self._pending = None
            return output

        self._pending = None
        output = self.linear(x)
        try:
            source_ref = weakref.ref(x)
        except TypeError:
            return output
        self._pending = _PendingGroupOutput(
            source_ref=source_ref,
            signature=signature,
            consumed={index},
            output=output,
        )
        return output

    def reset_pending(self) -> None:
        self._pending = None


class MixedLinearSlice(nn.Module):
    """One member projection served from its MixedLinearGroup output."""

    def __init__(self, group: MixedLinearGroup, index: int):
        super().__init__()
        # Not registered as a submodule: the group is owned (and its buffers
        # counted once) by the parent module.
        object.__setattr__(self, "group", group)
        self.index = int(index)
        self.start, self.stop = group.bounds[self.index]
        self.out_features = self.stop - self.start
        self.in_features = group.linear.K_orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.output_for(x, self.index)[..., self.start : self.stop]


PROJECTION_GROUPS = (
    ("tilequant_qkv", ("q_proj", "k_proj", "v_proj")),
    ("tilequant_gate_up", ("gate_proj", "up_proj")),
)


def fuse_projection_groups(model: nn.Module) -> int:
    """Replace Q/K/V and Gate/Up MixedLinear siblings by stacked groups."""

    fused = 0
    for parent in list(model.modules()):
        for group_name, member_names in PROJECTION_GROUPS:
            members = [getattr(parent, name, None) for name in member_names]
            if not all(isinstance(member, MixedLinear) for member in members):
                continue
            first = members[0]
            if any(
                member.K_orig != first.K_orig
                or member.M_orig % member.tile_m != 0
                or member.tile_m != first.tile_m
                or member.tile_n != first.tile_n
                or member.int8_tiles.device != first.int8_tiles.device
                for member in members
            ):
                continue
            if len({member.bias is None for member in members}) != 1:
                continue
            group = MixedLinearGroup(members)
            parent.add_module(group_name, group)
            for index, name in enumerate(member_names):
                setattr(parent, name, MixedLinearSlice(group, index))
            del members, first
            fused += 1
    return fused


def reset_projection_groups(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MixedLinearGroup):
            module.reset_pending()
