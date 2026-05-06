#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = (REPO_ROOT.parent / "model_path").resolve()
DEFAULT_DATA_ROOT = (REPO_ROOT / "datasets").resolve()
DEFAULT_OUTPUT_ROOT = (REPO_ROOT / "outputs").resolve()
DEFAULT_DECISION_ROOT = (REPO_ROOT / "Decision").resolve()


def ddp_is_active() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank0() -> bool:
    return ddp_rank() == 0


def ddp_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def safe_folder_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def resolve_local_model_dir(model_root: Path, model_name: str) -> Path:
    return (model_root / safe_folder_name(model_name)).resolve()


def _resolve_existing_model_dir(model_name: str) -> Optional[Path]:
    raw = Path(str(model_name)).expanduser()
    try:
        candidate = raw.resolve(strict=False)
    except Exception:
        candidate = raw
    if candidate.is_dir() and (candidate / "config.json").exists():
        return candidate.resolve()
    return None


def ensure_model_present(
    model_root: Path,
    model_name: str,
    download_if_missing: bool,
    revision: Optional[str] = None,
) -> Path:
    existing_dir = _resolve_existing_model_dir(model_name)
    if existing_dir is not None:
        return existing_dir

    local_dir = resolve_local_model_dir(model_root, model_name)
    if (local_dir / "config.json").exists():
        return local_dir

    raw_model_name = str(model_name)
    raw_path = Path(raw_model_name).expanduser()
    resolved_raw_path = raw_path.resolve(strict=False)
    if raw_path.is_absolute() or raw_model_name.startswith(".") or raw_model_name.startswith("~"):
        if not resolved_raw_path.exists():
            raise RuntimeError(
                "Local model path does not exist: "
                f"{resolved_raw_path}"
            )
        if not resolved_raw_path.is_dir():
            raise RuntimeError(
                "Local model path is not a directory: "
                f"{resolved_raw_path}"
            )
        raise RuntimeError(
            "Local model directory exists but is not directly loadable by Hugging Face. "
            "Expected config.json at the directory root: "
            f"{resolved_raw_path}"
        )
    if not download_if_missing:
        raise RuntimeError(f"Local model not found at {local_dir}.")

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", None)

    if is_rank0():
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            revision=revision,
            token=token,
        )
    ddp_barrier()

    if not (local_dir / "config.json").exists():
        raise RuntimeError(f"Download finished but config.json not found in {local_dir}.")
    return local_dir


def resolve_local_dataset_split_dir(data_root: Path, dataset_id: str, split: str) -> Path:
    return (data_root / safe_folder_name(dataset_id) / split).resolve()


def dataset_id_candidates(dataset_id: str) -> List[str]:
    cands = [dataset_id]
    if "/" in dataset_id:
        tail = dataset_id.split("/")[-1]
        if tail and tail not in cands:
            cands.append(tail)
    out: List[str] = []
    for x in cands:
        if x not in out:
            out.append(x)
    return out


def _try_load_dataset_and_save(data_root: Path, dataset_id: str, split: str) -> Optional[Tuple[Path, str]]:
    try:
        from datasets import load_dataset
    except Exception:
        return None

    for did in dataset_id_candidates(dataset_id):
        try:
            ds = load_dataset(did, split=split)
            out_dir = resolve_local_dataset_split_dir(data_root, did, split)
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out_dir))
            return out_dir, did
        except Exception:
            continue
    return None


def _download_dataset_via_raw_text_fallback(data_root: Path, dataset_id: str, split: str) -> Tuple[Path, str]:
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    split_lower = split.lower()
    if split_lower in ("train", "training"):
        base = "train"
    elif split_lower in ("validation", "valid", "val", "dev"):
        base = "val"
    elif split_lower in ("test", "testing"):
        base = "test"
    else:
        raise ValueError(f"Unsupported split for raw fallback: {split}")

    src_path_in_repo = f"data/{base}.src.cleaned"
    tgt_path_in_repo = f"data/{base}.tgt"
    token = os.environ.get("HF_TOKEN", None)
    last_err: Optional[Exception] = None

    for did in dataset_id_candidates(dataset_id):
        try:
            raw_dir = (data_root / safe_folder_name(did) / "_raw").resolve()
            raw_dir.mkdir(parents=True, exist_ok=True)
            out_dir = resolve_local_dataset_split_dir(data_root, did, split)

            if is_rank0():
                hf_hub_download(
                    repo_id=did,
                    filename=src_path_in_repo,
                    repo_type="dataset",
                    local_dir=str(raw_dir),
                    token=token,
                )
                hf_hub_download(
                    repo_id=did,
                    filename=tgt_path_in_repo,
                    repo_type="dataset",
                    local_dir=str(raw_dir),
                    token=token,
                )
            ddp_barrier()

            src_local = raw_dir / src_path_in_repo
            tgt_local = raw_dir / tgt_path_in_repo
            if not (src_local.exists() and tgt_local.exists()):
                raise RuntimeError("Raw fallback files not found after download.")

            sources = src_local.read_text(encoding="utf-8", errors="ignore").splitlines()
            targets = tgt_local.read_text(encoding="utf-8", errors="ignore").splitlines()
            n = min(len(sources), len(targets))
            docs: List[str] = []
            sums: List[str] = []
            for s, t in zip(sources[:n], targets[:n]):
                s = s.strip()
                t = t.strip()
                if s and t:
                    docs.append(s)
                    sums.append(t)
            if not docs:
                raise RuntimeError("Raw fallback produced empty dataset.")

            ds = Dataset.from_dict({"document": docs, "summary": sums})
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out_dir))
            return out_dir, did
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Raw fallback failed. Last error: {type(last_err).__name__}: {last_err}")


def ensure_dataset_local(data_root: Path, dataset_id: str, split: str, download_if_missing: bool) -> Tuple[Path, str]:
    from datasets import load_from_disk

    for did in dataset_id_candidates(dataset_id):
        local_dir = resolve_local_dataset_split_dir(data_root, did, split)
        if local_dir.exists() and (local_dir / "dataset_info.json").exists():
            return local_dir, did

    if not download_if_missing:
        raise RuntimeError(
            "Local dataset not found. Tried:\n  "
            + "\n  ".join(str(resolve_local_dataset_split_dir(data_root, did, split)) for did in dataset_id_candidates(dataset_id))
        )

    if is_rank0():
        saved = _try_load_dataset_and_save(data_root, dataset_id, split)
        if saved is None:
            _download_dataset_via_raw_text_fallback(data_root, dataset_id, split)
    ddp_barrier()

    for did in dataset_id_candidates(dataset_id):
        local_dir = resolve_local_dataset_split_dir(data_root, did, split)
        if local_dir.exists() and (local_dir / "dataset_info.json").exists():
            _ = load_from_disk(str(local_dir))
            return local_dir, did
    raise RuntimeError("Dataset download/cache failed: dataset_info.json still not found.")


def pad_to_multiple(x: torch.Tensor, tile_m: int, tile_n: int) -> torch.Tensor:
    m, k = x.shape
    mp = ((m + tile_m - 1) // tile_m) * tile_m
    kp = ((k + tile_n - 1) // tile_n) * tile_n
    if mp == m and kp == k:
        return x
    return F.pad(x, (0, kp - k, 0, mp - m), mode="constant", value=0.0)


def tile_reduce_sum(x: torch.Tensor, tile_m: int, tile_n: int) -> torch.Tensor:
    x = pad_to_multiple(x, tile_m, tile_n)
    m, k = x.shape
    n_tr = m // tile_m
    n_tc = k // tile_n
    return x.view(n_tr, tile_m, n_tc, tile_n).sum(dim=(1, 3))


def save_heatmap(
    tile_map: np.ndarray,
    out_path: Path,
    cmap: str = "viridis",
    norm_mode: str = "log1p",
    vmax_q: float = 0.99,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = tile_map.astype(np.float64, copy=False)
    x = np.maximum(x, 0.0)
    if norm_mode == "log1p":
        x = np.log1p(x)
        pos = x[x > 0]
        if pos.size > 0:
            scale = float(np.quantile(pos, 0.99))
            if np.isfinite(scale) and scale > 0:
                x = x / scale

    flat = x.reshape(-1)
    vmax = float(np.quantile(np.abs(flat), vmax_q)) if flat.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(flat))) if flat.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    _ = ax.imshow(x, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.tick_params(labelsize=7)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)


def build_prompt_simple(document: str) -> str:
    return "Summarize the following text.\n\nText:\n" + document.strip() + "\n\nSummary:\n"


def encode_supervised_example(
    tokenizer,
    document: str,
    summary: str,
    max_input_tokens: int,
    max_summary_tokens: int,
) -> Tuple[List[int], List[int]]:
    prompt = build_prompt_simple(document)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_input_tokens)
    sum_ids = tokenizer.encode(summary.strip(), add_special_tokens=False, truncation=True, max_length=max_summary_tokens)
    if tokenizer.eos_token_id is not None:
        sum_ids = sum_ids + [tokenizer.eos_token_id]

    input_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(int(tokenizer.bos_token_id))
    input_ids.extend(prompt_ids)
    prompt_len = len(input_ids)
    input_ids.extend(sum_ids)
    labels = [-100] * prompt_len + list(sum_ids)
    return input_ids, labels


class Tee:
    def __init__(self, file_path: Path, mode: str = "w"):
        self.file_path = file_path
        self.f = open(file_path, mode, encoding="utf-8")
        self.closed = False

    def write(self, s: str = "") -> None:
        if self.closed:
            return
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self) -> None:
        if not self.closed:
            self.f.close()
            self.closed = True
