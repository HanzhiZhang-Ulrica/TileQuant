from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _gemm_fp16_tiles_add_kernel(
    x_ptr,
    x_stride_n,
    x_stride_k,
    tiles_ptr,
    tiles_stride_t,
    tiles_stride_r,
    tiles_stride_c,
    tile_ids_ptr,
    y_ptr,
    y_stride_n,
    y_stride_m,
    N,
    M,
    K,
    n_tc,
    BLOCK_N: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_n = tl.program_id(1)

    tile_id = tl.load(tile_ids_ptr + pid_tile)
    r_tile = tile_id // n_tc
    c_tile = tile_id % n_tc

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 64)
    offs_r = tl.arange(0, 64)

    c_base = c_tile * 64
    r_base = r_tile * 64

    x_ptrs = x_ptr + offs_n[:, None] * x_stride_n + (c_base + offs_k[None, :]) * x_stride_k
    x_mask = (offs_n[:, None] < N) & ((c_base + offs_k[None, :]) < K)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)

    w_ptrs = (
        tiles_ptr
        + pid_tile * tiles_stride_t
        + offs_r[:, None] * tiles_stride_r
        + offs_k[None, :] * tiles_stride_c
    )
    w = tl.load(w_ptrs)

    acc = tl.dot(x, tl.trans(w))

    y_ptrs = y_ptr + offs_n[:, None] * y_stride_n + (r_base + offs_r[None, :]) * y_stride_m
    y_mask = (offs_n[:, None] < N) & ((r_base + offs_r[None, :]) < M)
    tl.atomic_add(y_ptrs, acc, mask=y_mask)


def gemm_fp16_tiles_add(
    x_fp: torch.Tensor,
    fp16_tiles: torch.Tensor,
    fp16_tile_ids: torch.Tensor,
    M: int,
    K: int,
    n_tc: int,
    y: torch.Tensor,
) -> None:
    if fp16_tiles.numel() == 0:
        return
    if x_fp.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x_fp must be fp16/bf16, got {x_fp.dtype}")
    if y.dtype != torch.float32:
        raise TypeError(f"y must be float32 accumulator, got {y.dtype}")

    N = x_fp.shape[0]
    T16 = fp16_tiles.shape[0]
    BLOCK_N = 64
    grid = (T16, triton.cdiv(N, BLOCK_N))

    _gemm_fp16_tiles_add_kernel[grid](
        x_fp,
        x_fp.stride(0),
        x_fp.stride(1),
        fp16_tiles,
        fp16_tiles.stride(0),
        fp16_tiles.stride(1),
        fp16_tiles.stride(2),
        fp16_tile_ids,
        y,
        y.stride(0),
        y.stride(1),
        N,
        M,
        K,
        n_tc,
        BLOCK_N=BLOCK_N,
    )
