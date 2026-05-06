from __future__ import annotations

import fnmatch
from typing import Any, List, Tuple

import torch.nn as nn


LLAMA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)

QWEN_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _suffixes(model_type: str):
    m = model_type.lower()
    if "qwen" in m:
        return QWEN_TARGET_SUFFIXES
    return LLAMA_TARGET_SUFFIXES


def should_quantize_name(full_name: str, cfg: Any) -> bool:
    patterns = _cfg_get(cfg, "module_filters", None)
    if patterns:
        if not any(fnmatch.fnmatch(full_name, p) for p in patterns):
            return False
    return True


def iter_target_linears(
    model: nn.Module,
    cfg: Any,
    model_type: str = "llama",
) -> List[Tuple[nn.Module, str, str, nn.Linear]]:
    suffixes = _suffixes(model_type)
    module_map = dict(model.named_modules())
    result: List[Tuple[nn.Module, str, str, nn.Linear]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not full_name.endswith(suffixes):
            continue
        if not should_quantize_name(full_name, cfg):
            continue

        if "." in full_name:
            parent_name, attr_name = full_name.rsplit(".", 1)
            parent = module_map[parent_name]
        else:
            parent = model
            attr_name = full_name
        result.append((parent, attr_name, full_name, module))

    return result
