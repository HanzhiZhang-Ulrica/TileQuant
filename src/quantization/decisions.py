from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..explore import workflow_common as wc
from ..kernels.calibrate import collect_iaa_tile_scores_with_channel_stats, collect_gi_tile_scores
from ..kernels.mask_io import save_mask_keep_archive
from ..models.rules import iter_target_linears
from .resolve import load_local_model_dir, resolve_decision_dir, resolve_keep_dir


DEFAULT_KEEP_FRACS = (0.01, 0.02, 0.05, 0.10, 0.20)


def infer_model_type(local_model_dir: Path) -> str:
    config = AutoConfig.from_pretrained(str(local_model_dir))
    model_type = str(getattr(config, "model_type", "")).lower()
    if "qwen" in model_type:
        return "qwen"
    if "llama" in model_type:
        return "llama"
    raise RuntimeError(f"Unsupported model_type for decision build: {model_type!r}")


def _score_npz_path(decision_dir: Path, quant_style: str, tile_m: int, tile_n: int) -> Path:
    return decision_dir / f"{quant_style}_tile_scores_tile{tile_m}x{tile_n}.npz"



def _atomic_savez(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    tmp_path = path.parent / f"{path.name}.tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp_path.replace(path)


def _atomic_save_json(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.parent / f"{path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.parent / f"{path.name}.tmp"
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _prune_unused_decision_files(decision_dir: Path) -> None:
    for pattern in ("*_iaa_sum.npy", "*_gi_sum.npy"):
        for path in Path(decision_dir).glob(pattern):
            if path.is_file():
                path.unlink()

    for rel_path in (
        "iaa_activation_progress.pt",
        "gi_tile_scores_progress.pt",
        "iaa_layer_stats.json",
        "iaa_meta.json",
        "done_iaa.flag",
        "tile_masks/eval_results.json",
    ):
        path = Path(decision_dir) / rel_path
        if path.exists():
            path.unlink()



def _load_npz_map(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}


def _load_existing_meta(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def _build_cfg(tile_m: int, tile_n: int, decision_dir: Path, num_batches: int, checkpoint_every_batches: int) -> SimpleNamespace:
    num_samples = None if int(num_batches) <= 0 else int(num_batches)
    return SimpleNamespace(
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        module_filters=[],
        calib=SimpleNamespace(
            num_samples=num_samples,
            max_tokens=None,
            checkpoint_dir=str(decision_dir),
            checkpoint_every_batches=int(checkpoint_every_batches),
            resume_from_checkpoint=True,
        ),
    )


def _build_examples(
    ds,
    tokenizer,
    max_examples: int,
    max_input_tokens: int,
    max_summary_tokens: int,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    out: List[np.ndarray] = []
    lengths: List[int] = []
    skipped = 0
    target_examples = len(ds) if max_examples <= 0 else max_examples
    for i in range(len(ds)):
        ex = ds[int(i)]
        doc = ex.get("document", None)
        summ = ex.get("summary", None)
        if doc is None or summ is None:
            skipped += 1
            continue
        ids, _ = wc.encode_supervised_example(
            tokenizer=tokenizer,
            document=doc,
            summary=summ,
            max_input_tokens=max_input_tokens,
            max_summary_tokens=max_summary_tokens,
        )
        arr = np.asarray(ids, dtype=np.int64)
        out.append(arr)
        lengths.append(int(arr.shape[0]))
        if len(out) >= target_examples:
            break
    stats = {
        "requested_examples": float(max_examples),
        "selected_examples": float(len(out)),
        "dataset_size": float(len(ds)),
        "skipped_examples": float(skipped),
        "mean_seq_len": float(np.mean(lengths) if lengths else 0.0),
        "max_seq_len": float(np.max(lengths) if lengths else 0.0),
        "min_seq_len": float(np.min(lengths) if lengths else 0.0),
    }
    return out, stats


def _make_loader(examples: List[np.ndarray], pad_id: int, batch_size: int) -> DataLoader:
    items = [{"input_ids": x} for x in examples]

    def _collate(batch):
        max_len = max(int(x["input_ids"].shape[0]) for x in batch)
        input_ids = np.full((len(batch), max_len), int(pad_id), dtype=np.int64)
        attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            length = int(ids.shape[0])
            input_ids[i, :length] = ids
            attention_mask[i, :length] = 1
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
        }

    return DataLoader(items, batch_size=batch_size, shuffle=False, collate_fn=_collate)


def _topk_mask(importance: np.ndarray, keep_frac: float) -> np.ndarray:
    values = importance.reshape(-1)
    k = max(1, int(math.ceil(float(keep_frac) * int(values.size))))
    if k >= int(values.size):
        idx = np.arange(int(values.size))
    else:
        idx = np.argpartition(values, -k)[-k:]
    mask = np.zeros_like(values, dtype=np.uint8)
    mask[idx] = 1
    return mask.reshape(importance.shape)


def _save_masks(
    decision_dir: Path,
    scores_by_stem: Dict[str, np.ndarray],
    keep_fracs: Sequence[float],
    model_name_or_path: str,
    dataset_id: str,
    dataset_split: str,
    quant_style: str,
    tile_m: int,
    tile_n: int,
) -> None:
    mask_root = decision_dir / "tile_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    for keep_frac in keep_fracs:
        out_dir = mask_root / f"keep{int(float(keep_frac) * 100):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        masks_by_stem = {
            stem: _topk_mask(score, keep_frac=float(keep_frac))
            for stem, score in scores_by_stem.items()
        }
        save_mask_keep_archive(
            keep_dir=out_dir,
            masks_by_stem=masks_by_stem,
            tile_m=tile_m,
            tile_n=tile_n,
        )
        _atomic_save_json(
            out_dir / "meta.json",
            {
                "model_name_or_path": model_name_or_path,
                "dataset_id": dataset_id,
                "dataset_split": dataset_split,
                "quant_style": quant_style,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "keep_frac": float(keep_frac),
                "mask_rule": "topk_by_importance",
            },
        )


def build_decisions(
    model_name_or_path: str,
    quant_style: str = "gi",
    dataset_id: str = "alexfabbri/multi_news",
    dataset_split: str = "train",
    num_examples: int = 8192,
    num_batches: int = -1,
    batch_size: int = 1,
    tile_m: int = 64,
    tile_n: int = 64,
    dtype: str = "bf16",
    max_input_tokens: int = 4096,
    max_summary_tokens: int = 256,
    checkpoint_every_batches: int = 100,
    download_if_missing: bool = False,
    keep_fracs: Sequence[float] = DEFAULT_KEEP_FRACS,
) -> Path:
    quant_style = str(quant_style).lower()
    if quant_style not in {"gi", "iaa"}:
        raise ValueError(f"Unsupported quant_style={quant_style}. Expected one of: gi, iaa")

    local_model_dir = load_local_model_dir(
        model_name=model_name_or_path,
        download_if_missing=download_if_missing,
        revision=None,
    )
    decision_dir = Path(wc.DEFAULT_DECISION_ROOT).resolve() / local_model_dir.name / quant_style
    decision_dir.mkdir(parents=True, exist_ok=True)

    tile_m = int(tile_m)
    tile_n = int(tile_n)
    keep_fracs = [float(x) for x in keep_fracs]
    meta_path = decision_dir / "decision_meta.json"
    done_flag = decision_dir / "decision_ready.flag"
    score_npz_path = _score_npz_path(decision_dir, quant_style, tile_m, tile_n)
    channel_stats_path = decision_dir / "iaa_channel_mean_abs.npz"
    existing_meta = _load_existing_meta(meta_path)

    resolved_dataset_id = str(existing_meta.get("dataset_id", dataset_id))
    resolved_dataset_split = str(existing_meta.get("dataset_split", dataset_split))
    example_stats = existing_meta.get("example_stats", None)
    model_type = infer_model_type(local_model_dir)

    score_outputs_ready = score_npz_path.exists() and (quant_style != "iaa" or channel_stats_path.exists())
    if score_outputs_ready:
        scores_by_stem = _load_npz_map(score_npz_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has no pad/eos token id")

        local_dataset_dir, resolved_dataset_id = wc.ensure_dataset_local(
            data_root=Path(wc.DEFAULT_DATA_ROOT).resolve(),
            dataset_id=dataset_id,
            split=dataset_split,
            download_if_missing=download_if_missing,
        )
        resolved_dataset_split = dataset_split
        ds = load_from_disk(str(local_dataset_dir))
        examples, example_stats = _build_examples(
            ds=ds,
            tokenizer=tokenizer,
            max_examples=int(num_examples),
            max_input_tokens=int(max_input_tokens),
            max_summary_tokens=int(max_summary_tokens),
        )
        if not examples:
            raise RuntimeError("No usable calibration examples")
        loader = _make_loader(examples=examples, pad_id=int(pad_id), batch_size=int(batch_size))
        cfg = _build_cfg(
            tile_m=tile_m,
            tile_n=tile_n,
            decision_dir=decision_dir,
            num_batches=int(num_batches),
            checkpoint_every_batches=int(checkpoint_every_batches),
        )
        torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_dir),
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        model.eval()
        model.config.use_cache = False
        targets = iter_target_linears(model, cfg=cfg, model_type=model_type)
        target_names = {name for _, _, name, _ in targets}
        if not target_names:
            raise RuntimeError(f"No target linears found for model_type={model_type}")

        if quant_style == "iaa":
            scores, channel_stats = collect_iaa_tile_scores_with_channel_stats(
                model=model,
                dataloader=loader,
                target_filter_fn=lambda name: name in target_names,
                cfg=cfg,
            )
            channel_stats_by_stem = {
                name.replace(".", "_"): arr.cpu().numpy().astype(np.float32, copy=False)
                for name, arr in channel_stats.items()
            }
            _atomic_savez(channel_stats_path, channel_stats_by_stem)
        else:
            scores = collect_gi_tile_scores(
                model=model,
                dataloader=loader,
                target_filter_fn=lambda name: name in target_names,
                cfg=cfg,
            )

        scores_by_stem = {
            name.replace(".", "_"): arr.cpu().numpy().astype(np.float32, copy=False)
            for name, arr in scores.items()
        }
        _atomic_savez(score_npz_path, scores_by_stem)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_masks(
        decision_dir=decision_dir,
        scores_by_stem=scores_by_stem,
        keep_fracs=keep_fracs,
        model_name_or_path=model_name_or_path,
        dataset_id=resolved_dataset_id,
        dataset_split=resolved_dataset_split,
        quant_style=quant_style,
        tile_m=tile_m,
        tile_n=tile_n,
    )

    meta = {
        "model_name_or_path": model_name_or_path,
        "local_model_dir": str(local_model_dir),
        "model_type": model_type,
        "dataset_id": resolved_dataset_id,
        "dataset_split": resolved_dataset_split,
        "quant_style": quant_style,
        "num_examples": int(num_examples),
        "num_batches": int(num_batches),
        "batch_size": int(batch_size),
        "tile_m": tile_m,
        "tile_n": tile_n,
        "dtype": dtype,
        "checkpoint_every_batches": int(checkpoint_every_batches),
        "keep_fracs": keep_fracs,
        "num_targets": int(len(scores_by_stem)),
        "example_stats": example_stats,
        "score_rule": (
            "iaa_like_absW_times_meanAbsX_tile_sum"
            if quant_style == "iaa"
            else "gi_absW_times_grad_tile_sum_accumulated_per_batch"
        ),
        "score_file": score_npz_path.name,
        "channel_stats_file": channel_stats_path.name if quant_style == "iaa" else None,
    }
    _atomic_save_json(meta_path, meta)
    _atomic_write_text(done_flag, "done\n")
    _prune_unused_decision_files(decision_dir)
    return decision_dir


def ensure_decision_artifacts(
    model_name_or_path: str,
    quant_style: str = "gi",
    keep_tag: Optional[str] = "keep01",
    dataset_id: str = "alexfabbri/multi_news",
    dataset_split: str = "train",
    num_examples: int = 8192,
    num_batches: int = -1,
    batch_size: int = 1,
    tile_m: int = 64,
    tile_n: int = 64,
    dtype: str = "bf16",
    max_input_tokens: int = 4096,
    max_summary_tokens: int = 256,
    checkpoint_every_batches: int = 100,
    download_if_missing: bool = False,
    keep_fracs: Sequence[float] = DEFAULT_KEEP_FRACS,
) -> Tuple[Path, Path, Path]:
    quant_style = str(quant_style).lower()
    local_model_dir = load_local_model_dir(
        model_name=model_name_or_path,
        download_if_missing=download_if_missing,
        revision=None,
    )
    should_build = False
    try:
        decision_dir = resolve_decision_dir(local_model_dir=local_model_dir, quant_style=quant_style)
    except RuntimeError:
        should_build = True
        decision_dir = Path(wc.DEFAULT_DECISION_ROOT).resolve() / local_model_dir.name / quant_style

    channel_stats_path = decision_dir / "iaa_channel_mean_abs.npz"
    if quant_style == "iaa" and not channel_stats_path.exists():
        should_build = True
    if not should_build:
        try:
            _ = resolve_keep_dir(decision_dir=decision_dir, keep_tag=keep_tag)
        except RuntimeError:
            should_build = True

    if should_build:
        decision_dir = build_decisions(
            model_name_or_path=model_name_or_path,
            quant_style=quant_style,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            num_examples=int(num_examples),
            num_batches=int(num_batches),
            batch_size=int(batch_size),
            tile_m=int(tile_m),
            tile_n=int(tile_n),
            dtype=dtype,
            max_input_tokens=int(max_input_tokens),
            max_summary_tokens=int(max_summary_tokens),
            checkpoint_every_batches=int(checkpoint_every_batches),
            download_if_missing=download_if_missing,
            keep_fracs=keep_fracs,
        )

    keep_dir = resolve_keep_dir(decision_dir=decision_dir, keep_tag=keep_tag)
    return local_model_dir, decision_dir, keep_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build resumable decision scores and keep masks.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--quant_style", type=str, choices=["gi", "iaa"], default="gi")
    parser.add_argument("--dataset_id", type=str, default="alexfabbri/multi_news")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--num_examples", type=int, default=8192)
    parser.add_argument("--num_batches", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tile_m", type=int, default=64)
    parser.add_argument("--tile_n", type=int, default=64)
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_summary_tokens", type=int, default=256)
    parser.add_argument("--checkpoint_every_batches", type=int, default=100)
    parser.add_argument("--download_if_missing", action="store_true")
    parser.add_argument("--keep_fracs", type=float, nargs="*", default=list(DEFAULT_KEEP_FRACS))
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    decision_dir = build_decisions(
        model_name_or_path=args.model_name_or_path,
        quant_style=args.quant_style,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        num_examples=int(args.num_examples),
        num_batches=int(args.num_batches),
        batch_size=int(args.batch_size),
        tile_m=int(args.tile_m),
        tile_n=int(args.tile_n),
        dtype=args.dtype,
        max_input_tokens=int(args.max_input_tokens),
        max_summary_tokens=int(args.max_summary_tokens),
        checkpoint_every_batches=int(args.checkpoint_every_batches),
        download_if_missing=bool(args.download_if_missing),
        keep_fracs=[float(x) for x in args.keep_fracs],
    )
    print(f"[decision_dir] {decision_dir}")


if __name__ == "__main__":
    main()
