from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from .launch import CachedLauncher, all_aligned
from .utils import select_block_n


# Row-owned mixed GEMM over a row-major compact tile arena.
#
# Pack invariant: INT8 and FP16 tiles are stored in ascending tile-id order,
# so the tiles owned by output-row tile r occupy the contiguous positions
# [row_ptr[r], row_ptr[r + 1]) in ascending input-column order.  The weight
# address of each iteration therefore depends only on the loop index, which
# lets Triton prefetch the next tiles while the current tile is in the tensor
# cores.
#
# Numerics: an exact INT32 64x64 tile dot,
# one FP32 product with (sx * sw), FP32 accumulation in ascending column order,
# then FP16/BF16 tiles accumulated by the FP32 tensor-core dot.  SPLIT_K > 1
# partitions each row's tile list into contiguous ranges and the last split adds
# the FP32 partials in fixed split order inside the same kernel, so results
# are deterministic.


# Split-K targets 4 CTAs per SM, at most 16 splits and at least 4 tiles each.
SPLITK_CTAS_PER_SM = 4
SPLITK_MAX = 16
SPLITK_MIN_TILES = 4
ROWS_NUM_STAGES = 3

_SM_COUNT: Dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    index = int(device.index if device.index is not None else torch.cuda.current_device())
    count = _SM_COUNT.get(index)
    if count is None:
        count = int(torch.cuda.get_device_properties(index).multi_processor_count)
        _SM_COUNT[index] = count
    return count


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


def select_split_k(
    *,
    n_tr: int,
    n_blocks: int,
    n_tc: int,
    device: torch.device,
) -> int:
    target = SPLITK_CTAS_PER_SM * _sm_count(device)
    ctas = max(1, int(n_tr) * int(n_blocks))
    if ctas >= target:
        return 1
    split = _next_power_of_2((target + ctas - 1) // ctas)
    split = min(split, SPLITK_MAX)
    while split > 1 and (int(n_tc) // split) < SPLITK_MIN_TILES:
        split //= 2
    return max(1, split)


def _rows_config(n_rows: int, n_tr: int, n_tc: int) -> Tuple[int, int]:
    # Larger row blocks reuse each weight tile across more activation rows; on
    # A100 this pays off when the output is much wider than the input.
    if int(n_rows) > 64 and int(n_tr) >= 4 * int(n_tc):
        return 128, 4
    return select_block_n(n_rows), 4


@triton.jit
def _gemm_mixed_rows_splitk_kernel(
    x_int8_ptr,
    x_int8_stride_n,
    sx_ptr,
    w8_ptr,
    w8_scales_ptr,
    w8_ids_ptr,
    w8_row_ptr,
    x_fp_ptr,
    x_fp_stride_n,
    w16_ptr,
    w16_ids_ptr,
    w16_row_ptr,
    out_ptr,
    out_stride_n,
    bias_ptr,
    ws_ptr,
    ws_stride_s,
    counters_ptr,
    N,
    M,
    K,
    n_tc: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_FP16: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    r_tile = tl.program_id(0)
    pid_n = tl.program_id(1)
    split = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 64)
    offs_r = tl.arange(0, 64)
    n_mask = offs_n < N
    tile_offsets = offs_r[:, None] * 64 + offs_k[None, :]
    x_int8_rows = x_int8_ptr + offs_n.to(tl.int64)[:, None] * x_int8_stride_n
    x_fp_rows = x_fp_ptr + offs_n.to(tl.int64)[:, None] * x_fp_stride_n

    acc = tl.zeros((BLOCK_N, 64), dtype=tl.float32)

    begin = tl.load(w8_row_ptr + r_tile)
    count = tl.load(w8_row_ptr + r_tile + 1) - begin
    lo = begin + (count * split) // SPLIT_K
    hi = begin + (count * (split + 1)) // SPLIT_K
    for pos in tl.range(lo, hi, num_stages=NUM_STAGES):
        c_tile = tl.load(w8_ids_ptr + pos) % n_tc
        cols = c_tile * 64 + offs_k
        x_i8 = tl.load(
            x_int8_rows + cols[None, :],
            mask=n_mask[:, None] & (cols[None, :] < K),
            other=0,
        )
        w_i8 = tl.load(w8_ptr + pos.to(tl.int64) * 4096 + tile_offsets)
        tile_acc = tl.dot(x_i8, tl.trans(w_i8)).to(tl.float32)
        sx = tl.load(sx_ptr + c_tile).to(tl.float32)
        sw = tl.load(w8_scales_ptr + pos).to(tl.float32)
        acc += tile_acc * (sx * sw)

    if HAS_FP16:
        begin = tl.load(w16_row_ptr + r_tile)
        count = tl.load(w16_row_ptr + r_tile + 1) - begin
        lo = begin + (count * split) // SPLIT_K
        hi = begin + (count * (split + 1)) // SPLIT_K
        for pos in tl.range(lo, hi, num_stages=NUM_STAGES):
            c_tile = tl.load(w16_ids_ptr + pos) % n_tc
            cols = c_tile * 64 + offs_k
            x = tl.load(
                x_fp_rows + cols[None, :],
                mask=n_mask[:, None] & (cols[None, :] < K),
                other=0.0,
            )
            w = tl.load(w16_ptr + pos.to(tl.int64) * 4096 + tile_offsets)
            acc += tl.dot(x, tl.trans(w))

    out_cols = r_tile * 64 + offs_r
    out_mask = n_mask[:, None] & (out_cols[None, :] < M)
    tile_rows = offs_n.to(tl.int64)[:, None]
    out_ptrs = out_ptr + tile_rows * out_stride_n + out_cols[None, :]
    if SPLIT_K > 1:
        # Serial split-K reduction (CUTLASS semaphore pattern): every split
        # publishes its FP32 partial, and the last split to arrive adds the
        # partials in fixed split order 0..SPLIT_K-1.  Deterministic, and no
        # second kernel launch.
        ws_ptrs = ws_ptr + tile_rows * M + out_cols[None, :]
        tl.store(ws_ptrs + split * ws_stride_s, acc, mask=out_mask)
        tl.debug_barrier()
        counter = counters_ptr + r_tile * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            acc = tl.load(ws_ptrs, mask=out_mask, other=0.0, cache_modifier=".cg")
            for other in tl.static_range(1, SPLIT_K):
                acc += tl.load(
                    ws_ptrs + other * ws_stride_s,
                    mask=out_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            y = acc.to(out_ptr.dtype.element_ty)
            if HAS_BIAS:
                bias = tl.load(bias_ptr + out_cols, mask=out_cols < M, other=0.0)
                y = (y.to(tl.float32) + bias.to(tl.float32)[None, :]).to(out_ptr.dtype.element_ty)
            tl.store(out_ptrs, y, mask=out_mask)
            tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")
    else:
        y = acc.to(out_ptr.dtype.element_ty)
        if HAS_BIAS:
            # Same rounding as `y.to(out_dtype) + bias.to(out_dtype)` in PyTorch.
            bias = tl.load(bias_ptr + out_cols, mask=out_cols < M, other=0.0)
            y = (y.to(tl.float32) + bias.to(tl.float32)[None, :]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptrs, y, mask=out_mask)


class _SplitKWorkspace:
    """Per (device, stream) FP32 partial buffer and zeroed arrival counters.

    Kernels on one stream run in order and every kernel returns the counters
    it used to zero, so the buffers are reused without re-initialization
    (also inside CUDA-graph capture/replay).
    """

    def __init__(self) -> None:
        self.partials: Dict[Tuple[int, int], torch.Tensor] = {}
        self.counters: Dict[Tuple[int, int], torch.Tensor] = {}

    def get(self, device: torch.device, device_index: int, partial_numel: int, counter_numel: int):
        key = (device_index, torch._C._cuda_getCurrentRawStream(device_index))
        partials = self.partials.get(key)
        if partials is None or partials.numel() < partial_numel:
            partials = torch.empty((max(partial_numel, 1 << 20),), device=device, dtype=torch.float32)
            self.partials[key] = partials
        counters = self.counters.get(key)
        if counters is None or counters.numel() < counter_numel:
            counters = torch.zeros((max(counter_numel, 4096),), device=device, dtype=torch.int32)
            self.counters[key] = counters
        return partials, counters


_WORKSPACE = _SplitKWorkspace()
_ROWS_LAUNCHER = CachedLauncher(
    _gemm_mixed_rows_splitk_kernel,
    ("n_tc", "BLOCK_N", "SPLIT_K", "HAS_FP16", "HAS_BIAS", "NUM_STAGES"),
)


def gemm_mixed_rows_splitk(
    *,
    x_fp: torch.Tensor,
    x_int8: torch.Tensor,
    sx: torch.Tensor,
    int8_tiles: torch.Tensor,
    int8_scales: torch.Tensor,
    int8_tile_ids: torch.Tensor,
    int8_row_ptr: torch.Tensor,
    fp16_tiles: torch.Tensor,
    fp16_tile_ids: torch.Tensor,
    fp16_row_ptr: torch.Tensor,
    M: int,
    K: int,
    n_tr: int,
    n_tc: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    split_k: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return x @ W^T (+ bias) for a row-major packed mixed INT8/FP16 weight.

    ``bias`` must already have dtype ``out_dtype`` and be contiguous.  ``out``
    (optional) is an (N, M) tensor with unit column stride to write into.
    """

    n = int(x_fp.shape[0])
    block_n, num_warps = _rows_config(n, n_tr, n_tc)
    n_blocks = triton.cdiv(n, block_n)
    if split_k is None:
        split_k = select_split_k(n_tr=n_tr, n_blocks=n_blocks, n_tc=n_tc, device=x_fp.device)
    device_index = x_fp.get_device()
    if out is None:
        out = torch.empty((n, M), device=x_fp.device, dtype=out_dtype)
    elif tuple(out.shape) != (n, int(M)) or out.dtype != out_dtype or out.stride(1) != 1:
        raise ValueError("out must be an (N, M) tensor of out_dtype with unit column stride")
    if split_k > 1:
        ws, counters = _WORKSPACE.get(
            x_fp.device, device_index, split_k * n * int(M), int(n_tr) * n_blocks
        )
    else:
        ws = counters = out
    if bias is None:
        bias = out
        has_bias = False
    else:
        has_bias = True
    has_fp16 = fp16_tiles.numel() != 0

    key = None
    # Decode-sized calls repeat with identical integers; prefill lengths vary.
    if n_blocks == 1 and all_aligned(
        x_int8, sx, int8_tiles, int8_scales, int8_tile_ids, int8_row_ptr, x_fp,
        fp16_tiles, fp16_tile_ids, fp16_row_ptr, out, bias, ws, counters,
    ):
        key = (
            x_fp.dtype, fp16_tiles.dtype, out_dtype, sx.dtype, int8_scales.dtype,
            int8_tile_ids.dtype, fp16_tile_ids.dtype, x_int8.stride(0), x_fp.stride(0), out.stride(0),
            n, int(M), int(K), int(n_tc), block_n, int(split_k), has_fp16, has_bias,
            ROWS_NUM_STAGES,
        )

    _ROWS_LAUNCHER(
        (int(n_tr), n_blocks, int(split_k)),
        (
            x_int8,
            x_int8.stride(0),
            sx,
            int8_tiles,
            int8_scales,
            int8_tile_ids,
            int8_row_ptr,
            x_fp,
            x_fp.stride(0),
            fp16_tiles,
            fp16_tile_ids,
            fp16_row_ptr,
            out,
            out.stride(0),
            bias,
            ws,
            n * int(M),
            counters,
            n,
            int(M),
            int(K),
            int(n_tc),
            block_n,
            int(split_k),
            has_fp16,
            has_bias,
            ROWS_NUM_STAGES,
        ),
        num_warps=num_warps,
        key=key,
        device_index=device_index,
    )
    return out
