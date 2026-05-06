from __future__ import annotations

from typing import Tuple

import triton
import triton.language as tl
import torch


TILE_M = 64
TILE_N = 64


@triton.jit
def enforce_tile_layout(ptr, stride0, stride1):
    tl.multiple_of(stride0, TILE_M)
    tl.multiple_of(stride1, TILE_N)
    tl.max_contiguous(ptr, TILE_N)


def grid_2d_tiles(n_tiles: int, n_blocks: int) -> Tuple[int, int]:
    return (n_tiles, n_blocks)


def normalize_out_dtype(dtype_like) -> torch.dtype:
    if dtype_like in ("bf16", torch.bfloat16):
        return torch.bfloat16
    return torch.float16
