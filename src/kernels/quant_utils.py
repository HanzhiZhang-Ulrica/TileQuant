from __future__ import annotations

from typing import Tuple

import torch


INT8_MAX = 127.0


def quantize_activation_absmax(
    x_fp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_fp.numel() == 0:
        sx = torch.tensor(1.0, device=x_fp.device, dtype=torch.float32)
        return torch.zeros_like(x_fp, dtype=torch.int8), sx

    max_abs = x_fp.abs().max()
    if float(max_abs.item()) == 0.0:
        sx = torch.tensor(1.0, device=x_fp.device, dtype=torch.float32)
        return torch.zeros_like(x_fp, dtype=torch.int8), sx

    sx = (max_abs / INT8_MAX).to(torch.float32)
    x_q = torch.clamp(torch.round(x_fp / sx), -127, 127).to(torch.int8)
    return x_q, sx


def quantize_activation_per_tensor_absmax(
    x_fp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_activation_absmax(x_fp)



def activation_tile_absmax_scales(
    x_fp: torch.Tensor,
    tile_n: int = 64,
) -> torch.Tensor:
    if x_fp.dim() != 2:
        raise ValueError(f"Expected 2D activation tensor, got shape={tuple(x_fp.shape)}")
    if tile_n <= 0:
        raise ValueError(f"tile_n must be > 0, got {tile_n}")

    n, k = x_fp.shape
    if k == 0:
        return torch.empty((0,), device=x_fp.device, dtype=torch.float32)

    n_tc = (k + tile_n - 1) // tile_n
    if (k % tile_n) != 0:
        x_work = x_fp.new_zeros((n, n_tc * tile_n))
        x_work[:, :k] = x_fp
    else:
        x_work = x_fp

    x_tiles = x_work.view(n, n_tc, tile_n)
    max_abs = x_tiles.abs().amax(dim=(0, 2))
    return torch.where(
        max_abs > 0,
        max_abs / INT8_MAX,
        torch.ones_like(max_abs, dtype=torch.float32),
    ).to(torch.float32).contiguous()



def quantize_activation_tile_with_scales(
    x_fp: torch.Tensor,
    sx: torch.Tensor,
    tile_n: int = 64,
) -> torch.Tensor:
    if x_fp.dim() != 2:
        raise ValueError(f"Expected 2D activation tensor, got shape={tuple(x_fp.shape)}")
    if tile_n <= 0:
        raise ValueError(f"tile_n must be > 0, got {tile_n}")

    n, k = x_fp.shape
    if k == 0:
        return torch.zeros_like(x_fp, dtype=torch.int8)

    n_tc = (k + tile_n - 1) // tile_n
    if sx.numel() != n_tc:
        raise ValueError(f"Scale length mismatch: got {sx.numel()}, expected {n_tc}")

    if (k % tile_n) != 0:
        x_work = x_fp.new_zeros((n, n_tc * tile_n))
        x_work[:, :k] = x_fp
    else:
        x_work = x_fp

    x_tiles = x_work.view(n, n_tc, tile_n)
    x_q_tiles = torch.clamp(
        torch.round(x_tiles / sx.view(1, n_tc, 1)),
        -127,
        127,
    ).to(torch.int8)
    return x_q_tiles.view(n, n_tc * tile_n)[:, :k].contiguous()



def quantize_activation_tile_absmax(
    x_fp: torch.Tensor,
    tile_n: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sx = activation_tile_absmax_scales(x_fp, tile_n=tile_n)
    x_q = quantize_activation_tile_with_scales(x_fp, sx=sx, tile_n=tile_n)
    return x_q, sx



def quantize_weight_tile_absmax(
    w_fp_tile_64x64: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs = w_fp_tile_64x64.abs().max()
    if float(max_abs.item()) == 0.0:
        sw = torch.tensor(1.0, device=w_fp_tile_64x64.device, dtype=torch.float32)
        return torch.zeros_like(w_fp_tile_64x64, dtype=torch.int8), sw

    sw = (max_abs / INT8_MAX).to(torch.float32)
    w_q = torch.clamp(torch.round(w_fp_tile_64x64 / sw), -127, 127).to(torch.int8)
    return w_q, sw



def pad_to_multiple(W: torch.Tensor, multiple: int = 64) -> Tuple[torch.Tensor, int, int]:
    if W.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape={tuple(W.shape)}")

    M, K = W.shape
    M_pad = ((M + multiple - 1) // multiple) * multiple
    K_pad = ((K + multiple - 1) // multiple) * multiple
    if M_pad == M and K_pad == K:
        return W, M, K

    W_pad = W.new_zeros((M_pad, K_pad))
    W_pad[:M, :K] = W
    return W_pad, M_pad, K_pad
