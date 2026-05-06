from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ..explore import workflow_common as wc


def load_local_model_dir(
    model_name: str,
    download_if_missing: bool = False,
    revision: Optional[str] = None,
) -> Path:
    return wc.ensure_model_present(
        model_root=Path(wc.DEFAULT_MODEL_ROOT).resolve(),
        model_name=model_name,
        download_if_missing=download_if_missing,
        revision=revision,
    )


def resolve_decision_dir(local_model_dir: Path, quant_style: str) -> Path:
    decision_dir = Path(wc.DEFAULT_DECISION_ROOT).resolve() / Path(local_model_dir).name / str(quant_style).lower()
    if not decision_dir.exists():
        raise RuntimeError(f"Decision directory not found: {decision_dir}")
    return decision_dir


def resolve_keep_dir(decision_dir: Path, keep_tag: Optional[str] = None) -> Path:
    mask_root = Path(decision_dir) / "tile_masks"
    if not mask_root.exists():
        raise RuntimeError(f"Mask root not found: {mask_root}")
    keep_dirs = [path for path in mask_root.iterdir() if path.is_dir() and path.name.startswith("keep")]
    if not keep_dirs:
        raise RuntimeError(f"No keepXX folders found under: {mask_root}")
    if keep_tag is not None:
        picked = mask_root / keep_tag
        if not picked.exists():
            raise RuntimeError(f"Requested keep dir not found: {picked}")
        return picked
    return sorted(keep_dirs, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]


def resolve_decision_and_keep_dir(
    local_model_dir: Path,
    quant_style: str,
    keep_tag: Optional[str] = None,
) -> Tuple[Path, Path]:
    decision_dir = resolve_decision_dir(local_model_dir=local_model_dir, quant_style=quant_style)
    keep_dir = resolve_keep_dir(decision_dir=decision_dir, keep_tag=keep_tag)
    return decision_dir, keep_dir
