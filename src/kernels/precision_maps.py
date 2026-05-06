from __future__ import annotations

import math
from typing import Dict

import torch


def build_precision_map(
    tile_scores: torch.Tensor,
    fp16_ratio: float,
) -> Dict[str, torch.Tensor | int]:
    if tile_scores.dim() != 2:
        raise ValueError(f"tile_scores must be [n_tr, n_tc], got shape={tuple(tile_scores.shape)}")

    fp16_ratio = min(max(fp16_ratio, 0.0), 1.0)
    n_tr, n_tc = int(tile_scores.shape[0]), int(tile_scores.shape[1])
    total_tiles = n_tr * n_tc

    sorted_ids = torch.argsort(tile_scores.reshape(-1).to(torch.float32), descending=True)
    n_fp16 = int(math.floor(fp16_ratio * total_tiles))
    fp16_tile_ids = sorted_ids[:n_fp16].to(torch.int32).contiguous()
    int8_tile_ids = sorted_ids[n_fp16:].to(torch.int32).contiguous()

    return {
        "int8_tile_ids": int8_tile_ids,
        "fp16_tile_ids": fp16_tile_ids,
        "n_tr": n_tr,
        "n_tc": n_tc,
    }


def build_precision_map_from_mask_keep(
    mask_keep: torch.Tensor,
) -> Dict[str, torch.Tensor | int]:
    if mask_keep.dim() != 2:
        raise ValueError(f"mask_keep must be [n_tr, n_tc], got shape={tuple(mask_keep.shape)}")

    n_tr, n_tc = int(mask_keep.shape[0]), int(mask_keep.shape[1])
    mk = mask_keep.to(torch.int32).reshape(-1)
    fp16_tile_ids = torch.nonzero(mk == 1, as_tuple=False).reshape(-1).to(torch.int32).contiguous()
    int8_tile_ids = torch.nonzero(mk == 0, as_tuple=False).reshape(-1).to(torch.int32).contiguous()

    if (fp16_tile_ids.numel() + int8_tile_ids.numel()) != (n_tr * n_tc):
        raise RuntimeError(
            f"Invalid mask_keep values: expected binary values in {{0,1}}, got shape={tuple(mask_keep.shape)}"
        )

    return {
        "int8_tile_ids": int8_tile_ids,
        "fp16_tile_ids": fp16_tile_ids,
        "n_tr": n_tr,
        "n_tc": n_tc,
    }
