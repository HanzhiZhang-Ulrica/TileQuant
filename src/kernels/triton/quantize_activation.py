from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .launch import CachedLauncher, all_aligned


@triton.jit
def _activation_tile_absmax_scales_64_kernel(
    x_ptr,
    x_stride_n,
    x_stride_k,
    sx_ptr,
    N,
    K,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 64)
    cols = pid_c * 64 + offs_k

    max_abs = 0.0
    for row_start in tl.range(0, N, BLOCK_N):
        rows = row_start + offs_n
        ptrs = x_ptr + rows[:, None] * x_stride_n + cols[None, :] * x_stride_k
        mask = (rows[:, None] < N) & (cols[None, :] < K)
        x = tl.load(ptrs, mask=mask, other=0.0)
        block_max = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
        max_abs = tl.maximum(max_abs, block_max)

    scale_native = max_abs.to(tl.float32) / 127.0
    if IS_BF16:
        scale_native = scale_native.to(tl.bfloat16)
    else:
        scale_native = scale_native.to(tl.float16)
    scale = tl.where(max_abs > 0.0, scale_native.to(tl.float32), 1.0)
    tl.store(sx_ptr + pid_c, scale)


@triton.jit
def _quantize_activation_with_scales_64_kernel(
    x_ptr,
    x_stride_n,
    x_stride_k,
    sx_ptr,
    q_ptr,
    q_stride_n,
    q_stride_k,
    N,
    K,
    BLOCK_N: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 64)
    cols = pid_c * 64 + offs_k

    ptrs = x_ptr + offs_n[:, None] * x_stride_n + cols[None, :] * x_stride_k
    mask = (offs_n[:, None] < N) & (cols[None, :] < K)
    x = tl.load(ptrs, mask=mask, other=0.0)
    scale = tl.load(sx_ptr + pid_c).to(tl.float32)
    q = libdevice.rint(tl.div_rn(x.to(tl.float32), scale))
    q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)

    q_ptrs = q_ptr + offs_n[:, None] * q_stride_n + cols[None, :] * q_stride_k
    tl.store(q_ptrs, q, mask=mask)


@triton.jit
def _quantize_activation_64_kernel(
    x_ptr,
    x_stride_n,
    x_stride_k,
    q_ptr,
    q_stride_n,
    q_stride_k,
    sx_ptr,
    N,
    K,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 64)
    cols = pid_c * 64 + offs_k

    ptrs = x_ptr + offs_n[:, None] * x_stride_n + cols[None, :] * x_stride_k
    mask = (offs_n[:, None] < N) & (cols[None, :] < K)
    x = tl.load(ptrs, mask=mask, other=0.0)

    max_abs = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale_native = max_abs.to(tl.float32) / 127.0
    if IS_BF16:
        scale_native = scale_native.to(tl.bfloat16)
    else:
        scale_native = scale_native.to(tl.float16)
    scale = tl.where(max_abs > 0.0, scale_native.to(tl.float32), 1.0)

    q = libdevice.rint(tl.div_rn(x.to(tl.float32), scale))
    q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)
    q_ptrs = q_ptr + offs_n[:, None] * q_stride_n + cols[None, :] * q_stride_k
    tl.store(q_ptrs, q, mask=mask)
    tl.store(sx_ptr + pid_c, scale)


_QUANTIZE_LAUNCHER = CachedLauncher(_quantize_activation_64_kernel, ("BLOCK_N", "IS_BF16"))


def quantize_activation_tile_absmax_fused(
    x_fp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, k = x_fp.shape
    if k == 0:
        return torch.empty_like(x_fp, dtype=torch.int8), torch.empty(
            (0,), device=x_fp.device, dtype=torch.float32
        )

    n_tc = (int(k) + 63) // 64
    block_n = triton.next_power_of_2(max(1, int(n)))
    num_warps = 1 if block_n <= 4 else 4
    x_q = torch.empty_like(x_fp, dtype=torch.int8, memory_format=torch.contiguous_format)
    sx = torch.empty((n_tc,), device=x_fp.device, dtype=torch.float32)
    is_bf16 = x_fp.dtype == torch.bfloat16
    key = None
    if int(n) <= 64 and all_aligned(x_fp, x_q, sx):
        key = (x_fp.dtype, x_fp.stride(0), x_fp.stride(1), x_q.stride(0), int(n), int(k), block_n)
    _QUANTIZE_LAUNCHER(
        (n_tc, 1, 1),
        (
            x_fp,
            x_fp.stride(0),
            x_fp.stride(1),
            x_q,
            x_q.stride(0),
            x_q.stride(1),
            sx,
            int(n),
            int(k),
            block_n,
            is_bf16,
        ),
        num_warps=num_warps,
        key=key,
        device_index=x_fp.get_device(),
    )
    return x_q, sx


def activation_tile_absmax_scales_64_fused(x_fp: torch.Tensor) -> torch.Tensor:
    n, k = x_fp.shape
    if k == 0:
        return torch.empty((0,), device=x_fp.device, dtype=torch.float32)

    n_tc = (int(k) + 63) // 64
    sx = torch.empty((n_tc,), device=x_fp.device, dtype=torch.float32)
    _activation_tile_absmax_scales_64_kernel[(n_tc,)](
        x_fp,
        x_fp.stride(0),
        x_fp.stride(1),
        sx,
        int(n),
        int(k),
        BLOCK_N=16,
        IS_BF16=x_fp.dtype == torch.bfloat16,
        num_warps=4,
    )
    return sx


def quantize_activation_tile_with_scales_64_fused(
    x_fp: torch.Tensor,
    sx: torch.Tensor,
) -> torch.Tensor:
    n, k = x_fp.shape
    if k == 0:
        return torch.empty_like(x_fp, dtype=torch.int8)

    n_tc = (int(k) + 63) // 64
    block_n = 16
    x_q = torch.empty_like(x_fp, dtype=torch.int8, memory_format=torch.contiguous_format)
    _quantize_activation_with_scales_64_kernel[(n_tc, triton.cdiv(n, block_n))](
        x_fp,
        x_fp.stride(0),
        x_fp.stride(1),
        sx,
        x_q,
        x_q.stride(0),
        x_q.stride(1),
        int(n),
        int(k),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return x_q
