from __future__ import annotations

from typing import Any

import torch

from .pack import Pack, pack_weight
from .precision_maps import build_precision_map, build_precision_map_from_mask_keep


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_pack(W_fp: torch.Tensor, tile_scores: torch.Tensor, cfg: Any) -> Pack:
    pmap = build_precision_map(
        tile_scores=tile_scores,
        fp16_ratio=float(_cfg_get(cfg, "fp16_ratio", 0.0)),
    )
    return pack_weight(W_fp, pmap, cfg)


def build_pack_from_mask_keep(W_fp: torch.Tensor, mask_keep: torch.Tensor, cfg: Any) -> Pack:
    pmap = build_precision_map_from_mask_keep(mask_keep=mask_keep)
    return pack_weight(W_fp, pmap, cfg)


def build_pack_auto(W_fp: torch.Tensor, tile_data: torch.Tensor, cfg: Any) -> Pack:
    source = str(_cfg_get(cfg, "precision_map_source", "scores")).lower()
    if source == "mask_keep":
        return build_pack_from_mask_keep(W_fp=W_fp, mask_keep=tile_data, cfg=cfg)
    return build_pack(W_fp=W_fp, tile_scores=tile_data, cfg=cfg)
