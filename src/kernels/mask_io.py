from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

_ARCHIVE_BASENAME = "fp16_keep_ids"
_ARCHIVE_CACHE: Dict[str, Dict[str, Tuple[Tuple[int, int], np.ndarray]]] = {}


def layer_name_to_stem(layer_name: str) -> str:
    return layer_name.replace(".", "_")


def _to_binary_uint8(mask_keep: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask_keep)
    if arr.ndim != 2:
        raise ValueError(f"mask_keep must be 2D, got shape={tuple(arr.shape)}")
    arr_u8 = arr.astype(np.uint8, copy=False)
    vals = np.unique(arr_u8)
    if not set(int(v) for v in vals.tolist()).issubset({0, 1}):
        raise ValueError(f"mask_keep must be binary {{0,1}}, got values={vals.tolist()}")
    return arr_u8


def _archive_name(tile_m: int, tile_n: int) -> str:
    return f"{_ARCHIVE_BASENAME}_tile{tile_m}x{tile_n}.npz"


def _archive_path(keep_dir: Path, tile_m: int, tile_n: int) -> Path:
    return Path(keep_dir) / _archive_name(tile_m=tile_m, tile_n=tile_n)


def _fp16_ids_from_mask(mask_keep: np.ndarray) -> np.ndarray:
    arr_u8 = _to_binary_uint8(mask_keep)
    flat = arr_u8.reshape(-1)
    ids = np.flatnonzero(flat == 1)
    if flat.size <= np.iinfo(np.uint16).max:
        return ids.astype(np.uint16, copy=False)
    return ids.astype(np.uint32, copy=False)


def save_mask_keep_archive(
    keep_dir: Path,
    masks_by_stem: Dict[str, np.ndarray],
    tile_m: int = 64,
    tile_n: int = 64,
) -> Path:
    keep_dir = Path(keep_dir)
    keep_dir.mkdir(parents=True, exist_ok=True)
    if not masks_by_stem:
        raise ValueError("masks_by_stem is empty")

    stems = sorted(str(k) for k in masks_by_stem.keys())
    save_dict: Dict[str, np.ndarray] = {
        "stems": np.asarray(stems, dtype=np.str_),
    }
    for idx, stem in enumerate(stems):
        arr_u8 = _to_binary_uint8(masks_by_stem[stem])
        save_dict[f"shape_{idx}"] = np.asarray(arr_u8.shape, dtype=np.int32)
        save_dict[f"ids_{idx}"] = _fp16_ids_from_mask(arr_u8)

    archive = _archive_path(keep_dir=keep_dir, tile_m=tile_m, tile_n=tile_n)
    np.savez_compressed(archive, **save_dict)
    _ARCHIVE_CACHE.pop(str(archive.resolve()), None)
    return archive


def _load_archive_map(keep_dir: Path, tile_m: int, tile_n: int) -> Dict[str, Tuple[Tuple[int, int], np.ndarray]]:
    archive = _archive_path(keep_dir=keep_dir, tile_m=tile_m, tile_n=tile_n)
    if not archive.exists():
        raise FileNotFoundError(f"FP16 keep-id archive not found: {archive}")
    cache_key = str(archive.resolve())
    if cache_key in _ARCHIVE_CACHE:
        return _ARCHIVE_CACHE[cache_key]

    with np.load(archive, allow_pickle=False) as data:
        stems = [str(x) for x in np.asarray(data["stems"]).tolist()]
        out: Dict[str, Tuple[Tuple[int, int], np.ndarray]] = {}
        for idx, stem in enumerate(stems):
            shape_arr = np.asarray(data[f"shape_{idx}"], dtype=np.int64)
            if shape_arr.shape != (2,):
                raise RuntimeError(f"Invalid shape metadata for stem={stem} in {archive}")
            n_tr, n_tc = int(shape_arr[0]), int(shape_arr[1])
            ids = np.asarray(data[f"ids_{idx}"])
            if ids.dtype not in (np.uint16, np.uint32):
                ids = ids.astype(np.uint32, copy=False)
            total_tiles = n_tr * n_tc
            if ids.size > 0 and int(ids.max()) >= total_tiles:
                raise RuntimeError(f"Out-of-bound fp16 tile id for stem={stem} in {archive}")
            out[stem] = ((n_tr, n_tc), np.asarray(ids))
    _ARCHIVE_CACHE[cache_key] = out
    return out


def load_mask_keep_for_layer(
    keep_dir: Path,
    layer_name: str,
    tile_m: int = 64,
    tile_n: int = 64,
) -> torch.Tensor:
    stem = layer_name_to_stem(layer_name)
    amap = _load_archive_map(keep_dir=Path(keep_dir), tile_m=tile_m, tile_n=tile_n)
    if stem not in amap:
        raise FileNotFoundError(f"Layer stem not found in archive for layer '{layer_name}': stem={stem}")
    (n_tr, n_tc), ids = amap[stem]
    arr = np.zeros((n_tr * n_tc,), dtype=np.uint8)
    arr[ids.astype(np.int64, copy=False)] = 1
    arr = arr.reshape((n_tr, n_tc))
    return torch.from_numpy(arr).to(torch.uint8).contiguous()


def load_fp16_keep_ids_for_layer(
    keep_dir: Path,
    layer_name: str,
    tile_m: int = 64,
    tile_n: int = 64,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    stem = layer_name_to_stem(layer_name)
    amap = _load_archive_map(keep_dir=Path(keep_dir), tile_m=tile_m, tile_n=tile_n)
    if stem not in amap:
        raise FileNotFoundError(f"Layer stem not found in archive for layer '{layer_name}': stem={stem}")
    (n_tr, n_tc), ids = amap[stem]
    ids_t = torch.from_numpy(ids.astype(np.int64, copy=False)).to(torch.int32).contiguous()
    return ids_t, (n_tr, n_tc)


def load_mask_keep_stem_dict(
    keep_dir: Path,
    tile_m: int = 64,
    tile_n: int = 64,
) -> Dict[str, torch.Tensor]:
    amap = _load_archive_map(keep_dir=Path(keep_dir), tile_m=tile_m, tile_n=tile_n)
    out: Dict[str, torch.Tensor] = {}
    for stem, ((n_tr, n_tc), ids) in amap.items():
        arr = np.zeros((n_tr * n_tc,), dtype=np.uint8)
        arr[ids.astype(np.int64, copy=False)] = 1
        arr = arr.reshape((n_tr, n_tc))
        out[stem] = torch.from_numpy(arr).to(torch.uint8).contiguous()
    return out
