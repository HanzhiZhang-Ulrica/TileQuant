from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _gemm_int8_tiles_add_kernel(
    x_ptr,
    x_stride_n,
    x_stride_k,
    sx_ptr,
    tiles_ptr,
    tiles_stride_t,
    tiles_stride_r,
    tiles_stride_c,
    scales_ptr,
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
    x_i8 = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.int8)

    w_ptrs = (
        tiles_ptr
        + pid_tile * tiles_stride_t
        + offs_r[:, None] * tiles_stride_r
        + offs_k[None, :] * tiles_stride_c
    )
    w_i8 = tl.load(w_ptrs).to(tl.int8)

    acc_i32 = tl.dot(x_i8, tl.trans(w_i8))

    sx = tl.load(sx_ptr + c_tile).to(tl.float32)
    sw = tl.load(scales_ptr + pid_tile).to(tl.float32)
    scale = sx * sw
    acc_f = acc_i32.to(tl.float32) * scale

    y_ptrs = y_ptr + offs_n[:, None] * y_stride_n + (r_base + offs_r[None, :]) * y_stride_m
    y_mask = (offs_n[:, None] < N) & ((r_base + offs_r[None, :]) < M)
    tl.atomic_add(y_ptrs, acc_f, mask=y_mask)


def gemm_int8_tiles_add(
    x_int8: torch.Tensor,
    sx: torch.Tensor,
    int8_tiles: torch.Tensor,
    int8_scales: torch.Tensor,
    int8_tile_ids: torch.Tensor,
    M: int,
    K: int,
    n_tc: int,
    y: torch.Tensor,
) -> None:
    if int8_tiles.numel() == 0:
        return
    if x_int8.dtype != torch.int8:
        raise TypeError(f"x_int8 must be int8, got {x_int8.dtype}")
    if y.dtype != torch.float32:
        raise TypeError(f"y must be float32 accumulator, got {y.dtype}")

    if sx.dim() == 0:
        sx_t = sx.reshape(1).expand(int(n_tc)).contiguous()
    else:
        sx_t = sx.reshape(-1)
        if int(sx_t.numel()) != int(n_tc):
            raise ValueError(f"sx must be scalar or length n_tc={n_tc}, got shape={tuple(sx.shape)}")
    sx_t = sx_t.to(device=x_int8.device, dtype=torch.float32).contiguous()

    N = x_int8.shape[0]
    T8 = int8_tiles.shape[0]
    BLOCK_N = 64
    grid = (T8, triton.cdiv(N, BLOCK_N))

    _gemm_int8_tiles_add_kernel[grid](
        x_int8,
        x_int8.stride(0),
        x_int8.stride(1),
        sx_t,
        int8_tiles,
        int8_tiles.stride(0),
        int8_tiles.stride(1),
        int8_tiles.stride(2),
        int8_scales,
        int8_tile_ids,
        y,
        y.stride(0),
        y.stride(1),
        N,
        M,
        K,
        n_tc,
        BLOCK_N=BLOCK_N,
    )
