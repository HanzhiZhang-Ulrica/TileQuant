from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .api import load_mixed_precision_model_from_masks
from .decisions import ensure_decision_artifacts, infer_model_type
from .types import ApplyConfig, TileConfig


FLASH_ATTN_IMPL = "flash_attention_2"


def _resolve_iaa_channel_stats_path(quant_style: str, decision_dir: Path, cli_path: Optional[str]) -> Optional[str]:
    if quant_style != "iaa":
        return cli_path
    if cli_path:
        return cli_path
    auto_path = decision_dir / "iaa_channel_mean_abs.npz"
    if not auto_path.exists():
        raise RuntimeError(
            "IAA quant_style requires channel stats, but auto path is missing: "
            f"{auto_path}. Run build_decisions.py --quant_style iaa first or let apply_model.py build decisions automatically."
        )
    return str(auto_path)



def _resolve_target_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    device_name = str(device or "auto").strip().lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)



def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None



def _resolve_attn_implementation(device: str | torch.device) -> Optional[str]:
    device_obj = _resolve_target_device(device)
    if device_obj.type != "cuda":
        return None
    if not _flash_attn_available():
        return None
    return FLASH_ATTN_IMPL



def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    quant_style: str = "gi",
    keep_tag: Optional[str] = "keep01",
    dtype: str = "bf16",
    device: str | torch.device = "auto",
    iaa_channel_stats_path: Optional[str] = None,
    dataset_id: str = "alexfabbri/multi_news",
    dataset_split: str = "train",
    num_examples: int = 8192,
    num_batches: int = -1,
    batch_size: int = 1,
    tile_m: int = 64,
    tile_n: int = 64,
    max_input_tokens: int = 4096,
    max_summary_tokens: int = 256,
    checkpoint_every_batches: int = 100,
    download_if_missing: bool = False,
) -> Tuple[torch.nn.Module, Any, Dict[str, object]]:
    device_obj = _resolve_target_device(device)
    attn_implementation = _resolve_attn_implementation(device_obj)
    local_model_dir, decision_dir, keep_dir = ensure_decision_artifacts(
        model_name_or_path=model_name_or_path,
        quant_style=quant_style,
        keep_tag=keep_tag,
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
        download_if_missing=bool(download_if_missing),
    )
    model_type = infer_model_type(local_model_dir)
    quant_style = str(quant_style).lower()
    iaa_channel_stats_path = _resolve_iaa_channel_stats_path(
        quant_style=quant_style,
        decision_dir=decision_dir,
        cli_path=iaa_channel_stats_path,
    )

    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, stats = load_mixed_precision_model_from_masks(
        local_model_dir=local_model_dir,
        keep_dir=keep_dir,
        apply_cfg=ApplyConfig(
            model_type=model_type,
            dtype_out=dtype,
            tile=TileConfig(tile_m=int(tile_m), tile_n=int(tile_n)),
            module_filters=[],
            quant_style=quant_style,
            iaa_channel_stats_path=iaa_channel_stats_path,
        ),
        torch_dtype=torch_dtype,
        device=device_obj,
        attn_implementation=attn_implementation,
    )
    info: Dict[str, object] = {
        "device": device_obj,
        "local_model_dir": local_model_dir,
        "decision_dir": decision_dir,
        "keep_dir": keep_dir,
        "model_type": model_type,
        "quant_style": quant_style,
        "tile_m": int(tile_m),
        "tile_n": int(tile_n),
        "iaa_channel_stats_path": iaa_channel_stats_path,
        "attn_implementation": attn_implementation,
        "stats": stats,
    }
    return model, tokenizer, info
