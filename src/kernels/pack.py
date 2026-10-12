from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from .quant_utils import pad_to_multiple


@dataclass
class Pack:
    M_orig: int
    K_orig: int
    M: int
    K: int
    tile_m: int
    tile_n: int
    n_tr: int
    n_tc: int
    int8_tiles: torch.Tensor
    int8_scales: torch.Tensor
    int8_tile_ids: torch.Tensor
    int8_row_ptr: torch.Tensor
    int8_row_indices: torch.Tensor
    fp16_tiles: torch.Tensor
    fp16_tile_ids: torch.Tensor
    fp16_row_ptr: torch.Tensor
    fp16_row_indices: torch.Tensor


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _order_tile_ids_row_major(tile_ids: torch.Tensor) -> torch.Tensor:
    # Row-major compact arena: the tiles of output-row tile r are contiguous
    # and in ascending input-column order, so row-owned kernels address the
    # payload directly from the loop position (no index indirection).  The
    # per-tile quantization does not depend on the storage order.
    if tile_ids.numel() <= 1:
        return tile_ids.contiguous()
    return torch.sort(tile_ids).values.contiguous()


def _build_row_schedule(
    tile_ids: torch.Tensor,
    *,
    n_tr: int,
    n_tc: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_ptr = torch.zeros((n_tr + 1,), device=tile_ids.device, dtype=torch.int32)
    if tile_ids.numel() == 0:
        return row_ptr, torch.empty((0,), device=tile_ids.device, dtype=torch.int32)

    rows = torch.div(tile_ids, n_tc, rounding_mode="floor")
    row_indices = torch.argsort(tile_ids).to(torch.int32).contiguous()
    counts = torch.bincount(rows, minlength=n_tr).to(torch.int32)
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    return row_ptr, row_indices


def _pack_weight_with_tile_ids(
    W_fp: torch.Tensor,
    *,
    n_tr: int,
    n_tc: int,
    int8_tile_ids: torch.Tensor,
    fp16_tile_ids: torch.Tensor,
    iaa_col_importance: Optional[torch.Tensor],
    cfg: Any,
) -> Pack:
    if W_fp.dim() != 2:
        raise ValueError(f"Expected W to be 2D, got shape={tuple(W_fp.shape)}")

    tile_m = int(_cfg_get(cfg, "tile_m", 64))
    tile_n = int(_cfg_get(cfg, "tile_n", 64))
    use_pad = bool(_cfg_get(cfg, "pad_to_tile", True))
    quant_style = str(_cfg_get(cfg, "quant_style", "gi")).lower()
    if quant_style not in ("gi", "iaa"):
        raise ValueError(f"Unsupported quant_style={quant_style}. Expected one of: gi, iaa")

    M_orig, K_orig = W_fp.shape
    if use_pad:
        W_work, M, K = pad_to_multiple(W_fp, multiple=64)
    else:
        if (M_orig % tile_m) != 0 or (K_orig % tile_n) != 0:
            raise ValueError("pad_to_tile=False but weight shape is not multiple of tile size")
        W_work = W_fp
        M, K = M_orig, K_orig

    if n_tr * tile_m != M or n_tc * tile_n != K:
        raise ValueError(
            f"PrecisionMap dimensions mismatch: map(n_tr={n_tr}, n_tc={n_tc}) "
            f"vs weight(M={M}, K={K}, tile_m={tile_m}, tile_n={tile_n})"
        )
    int8_tile_ids = _order_tile_ids_row_major(
        int8_tile_ids.to(device=W_work.device, dtype=torch.int64)
    )
    fp16_tile_ids = _order_tile_ids_row_major(
        fp16_tile_ids.to(device=W_work.device, dtype=torch.int64)
    )
    int8_row_ptr, int8_row_indices = _build_row_schedule(
        int8_tile_ids,
        n_tr=n_tr,
        n_tc=n_tc,
    )
    fp16_row_ptr, fp16_row_indices = _build_row_schedule(
        fp16_tile_ids,
        n_tr=n_tr,
        n_tc=n_tc,
    )
    tile_view = W_work.view(n_tr, tile_m, n_tc, tile_n).permute(0, 2, 1, 3)

    device = W_fp.device
    w_dtype = W_fp.dtype
    if int8_tile_ids.numel() > 0:
        int8_r = torch.div(int8_tile_ids, n_tc, rounding_mode="floor")
        int8_c = int8_tile_ids.remainder(n_tc)
        int8_src = tile_view[int8_r, int8_c]
        # BF16/FP16 abs and max preserve the selected source value exactly.
        # Cast only the reduced tensor instead of materializing every tile in FP32.
        max_abs = int8_src.abs().amax(dim=(1, 2)).to(torch.float32)
        base_scales = torch.where(
            max_abs == 0,
            torch.ones_like(max_abs, dtype=torch.float32),
            (max_abs / 127.0).to(torch.float32),
        )

        if quant_style == "iaa" and iaa_col_importance is not None:
            col_imp = iaa_col_importance.to(device=W_work.device, dtype=torch.float32).reshape(-1)
            if int(col_imp.numel()) < int(W_work.shape[1]):
                tmp = torch.zeros((int(W_work.shape[1]),), device=W_work.device, dtype=torch.float32)
                tmp[: int(col_imp.numel())] = col_imp
                col_imp = tmp
            else:
                col_imp = col_imp[: int(W_work.shape[1])]
            col_imp_tiles = col_imp.view(n_tc, tile_n)
            int8_col_imp = col_imp_tiles[int8_c].clamp_min(1e-8)
            wm = int8_col_imp[:, None, :]

            cand_mult = torch.tensor([0.75, 1.00, 1.25], device=W_work.device, dtype=torch.float32)
            best_err = None
            best_scales = None
            src_f = int8_src.to(torch.float32)
            for m in cand_mult:
                s = (base_scales * m).clamp_min(1e-12)
                dq = torch.clamp(torch.round(src_f / s[:, None, None]), -127, 127) * s[:, None, None]
                err = ((dq - src_f) ** 2 * wm).mean(dim=(1, 2))
                if best_err is None:
                    best_err = err
                    best_scales = s
                else:
                    better = err < best_err
                    best_err = torch.where(better, err, best_err)
                    best_scales = torch.where(better, s, best_scales)
            int8_scales_t = best_scales.to(torch.float32)
        else:
            int8_scales_t = base_scales.to(torch.float32)

        int8_tiles_t = torch.clamp(
            torch.round(int8_src / int8_scales_t[:, None, None]),
            -127,
            127,
        ).to(torch.int8)
    else:
        int8_tiles_t = torch.empty((0, tile_m, tile_n), device=device, dtype=torch.int8)
        int8_scales_t = torch.empty((0,), device=device, dtype=torch.float32)

    if fp16_tile_ids.numel() > 0:
        fp16_r = torch.div(fp16_tile_ids, n_tc, rounding_mode="floor")
        fp16_c = fp16_tile_ids.remainder(n_tc)
        fp16_tiles_t = tile_view[fp16_r, fp16_c].to(w_dtype).contiguous()
    else:
        fp16_tiles_t = torch.empty((0, tile_m, tile_n), device=device, dtype=w_dtype)

    if int8_tiles_t.shape[0] + fp16_tiles_t.shape[0] != (n_tr * n_tc):
        raise RuntimeError(
            "Packed tile count mismatch: "
            f"T8={int8_tiles_t.shape[0]}, T16={fp16_tiles_t.shape[0]}, total={n_tr * n_tc}"
        )

    return Pack(
        M_orig=M_orig,
        K_orig=K_orig,
        M=M,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        n_tr=n_tr,
        n_tc=n_tc,
        int8_tiles=int8_tiles_t,
        int8_scales=int8_scales_t,
        int8_tile_ids=int8_tile_ids.to(device=device, dtype=torch.int32).contiguous(),
        int8_row_ptr=int8_row_ptr.to(device=device, dtype=torch.int32).contiguous(),
        int8_row_indices=int8_row_indices.to(device=device, dtype=torch.int32).contiguous(),
        fp16_tiles=fp16_tiles_t,
        fp16_tile_ids=fp16_tile_ids.to(device=device, dtype=torch.int32).contiguous(),
        fp16_row_ptr=fp16_row_ptr.to(device=device, dtype=torch.int32).contiguous(),
        fp16_row_indices=fp16_row_indices.to(device=device, dtype=torch.int32).contiguous(),
    )


def pack_weight_from_tile_ids(
    W_fp: torch.Tensor,
    *,
    n_tr: int,
    n_tc: int,
    int8_tile_ids: torch.Tensor,
    fp16_tile_ids: torch.Tensor,
    iaa_col_importance: Optional[torch.Tensor] = None,
    cfg: Any,
) -> Pack:
    return _pack_weight_with_tile_ids(
        W_fp=W_fp,
        n_tr=int(n_tr),
        n_tc=int(n_tc),
        int8_tile_ids=int8_tile_ids,
        fp16_tile_ids=fp16_tile_ids,
        iaa_col_importance=iaa_col_importance,
        cfg=cfg,
    )


def pack_weight(W_fp: torch.Tensor, pmap: Dict[str, torch.Tensor | int], cfg: Any) -> Pack:
    return _pack_weight_with_tile_ids(
        W_fp=W_fp,
        n_tr=int(pmap["n_tr"]),
        n_tc=int(pmap["n_tc"]),
        int8_tile_ids=pmap["int8_tile_ids"],
        fp16_tile_ids=pmap["fp16_tile_ids"],
        iaa_col_importance=None,
        cfg=cfg,
    )


def is_row_major_pack(pack: Pack) -> bool:
    for ids in (pack.int8_tile_ids, pack.fp16_tile_ids):
        if ids.numel() > 1 and not bool((ids[1:] > ids[:-1]).all()):
            return False
    return True


def concat_packs_along_rows(packs: Sequence[Pack]) -> Pack:
    """Stack packs of projections that share one input along the output rows.

    Every member must have the same input geometry and an output width that is
    a multiple of tile_m, so member row tiles map one-to-one onto the stacked
    row tiles and each member's output occupies an exact column slice.  Tile
    payloads and scales are copied unchanged.
    """

    if not packs:
        raise ValueError("Expected at least one pack")
    first = packs[0]
    for pack in packs:
        if (pack.K_orig, pack.K, pack.tile_m, pack.tile_n, pack.n_tc) != (
            first.K_orig,
            first.K,
            first.tile_m,
            first.tile_n,
            first.n_tc,
        ):
            raise ValueError("Packs must share input geometry to be stacked")
        if pack.M_orig != pack.M or pack.M_orig % pack.tile_m != 0:
            raise ValueError("Stacked packs require tile-aligned output widths")
        if not is_row_major_pack(pack):
            raise ValueError("Stacked packs require row-major tile order")

    n_tc = first.n_tc

    def stack(kind: str):
        tiles = []
        scales = []
        ids = []
        row_ptrs = [torch.zeros((1,), dtype=torch.int32, device=first.int8_tiles.device)]
        row_indices = []
        tile_base = 0
        row_base = 0
        for pack in packs:
            pack_ids = getattr(pack, f"{kind}_tile_ids").to(torch.int64)
            tiles.append(getattr(pack, f"{kind}_tiles"))
            if kind == "int8":
                scales.append(pack.int8_scales)
            ids.append(pack_ids + row_base * n_tc)
            row_ptrs.append(getattr(pack, f"{kind}_row_ptr")[1:].to(torch.int32) + tile_base)
            row_indices.append(getattr(pack, f"{kind}_row_indices").to(torch.int32) + tile_base)
            tile_base += int(pack_ids.numel())
            row_base += pack.n_tr
        return (
            torch.cat(tiles, dim=0).contiguous(),
            torch.cat(scales, dim=0).contiguous() if scales else None,
            torch.cat(ids, dim=0).to(torch.int32).contiguous(),
            torch.cat(row_ptrs, dim=0).contiguous(),
            torch.cat(row_indices, dim=0).contiguous(),
        )

    int8_tiles, int8_scales, int8_ids, int8_row_ptr, int8_row_indices = stack("int8")
    fp16_tiles, _, fp16_ids, fp16_row_ptr, fp16_row_indices = stack("fp16")
    M = sum(pack.M for pack in packs)
    return Pack(
        M_orig=M,
        K_orig=first.K_orig,
        M=M,
        K=first.K,
        tile_m=first.tile_m,
        tile_n=first.tile_n,
        n_tr=sum(pack.n_tr for pack in packs),
        n_tc=n_tc,
        int8_tiles=int8_tiles,
        int8_scales=int8_scales,
        int8_tile_ids=int8_ids,
        int8_row_ptr=int8_row_ptr,
        int8_row_indices=int8_row_indices,
        fp16_tiles=fp16_tiles,
        fp16_tile_ids=fp16_ids,
        fp16_row_ptr=fp16_row_ptr,
        fp16_row_indices=fp16_row_indices,
    )
