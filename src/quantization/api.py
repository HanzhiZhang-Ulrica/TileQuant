from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers.modeling_utils import init_empty_weights as _init_empty_weights
except ImportError:
    try:
        from accelerate import init_empty_weights as _init_empty_weights
    except ImportError:
        _init_empty_weights = None

try:
    from transformers.modeling_utils import set_module_tensor_to_device
except ImportError:
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "set_module_tensor_to_device is required but not found in transformers or accelerate."
        ) from exc

from ..kernels.mask_io import load_fp16_keep_ids_for_layer
from ..kernels.pack import pack_weight_from_tile_ids
from ..models.mixed_linear import MixedLinear, fuse_projection_groups
from ..models.rules import iter_target_linears
from .types import ApplyConfig


def _expected_tile_grid(shape: Iterable[int], tile_m: int, tile_n: int) -> Tuple[int, int]:
    m, k = int(shape[0]), int(shape[1])
    n_tr = (m + tile_m - 1) // tile_m
    n_tc = (k + tile_n - 1) // tile_n
    return n_tr, n_tc


def _make_cfg(apply_cfg: ApplyConfig):
    return SimpleNamespace(
        tile_m=int(apply_cfg.tile.tile_m),
        tile_n=int(apply_cfg.tile.tile_n),
        precision_map_source="mask_keep",
        act_quant="per_tensor_absmax",
        weight_quant="per_tile_absmax",
        quant_style=str(getattr(apply_cfg, "quant_style", "gi")),
        dtype_out=str(apply_cfg.dtype_out),
        pad_to_tile=True,
        module_filters=list(apply_cfg.module_filters),
    )


def _load_iaa_channel_stats(keep_dir: Path, apply_cfg: ApplyConfig) -> Dict[str, torch.Tensor]:
    override_path = getattr(apply_cfg, "iaa_channel_stats_path", None)
    if override_path:
        path = Path(str(override_path))
    else:
        path = Path(keep_dir).parent.parent / "iaa_channel_mean_abs.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"IAA channel stats not found: {path}. "
            "Generate via build_decisions.py --quant_style iaa or pass --iaa_channel_stats_path."
        )
    with np.load(path, allow_pickle=False) as data:
        return {
            key: torch.from_numpy(np.asarray(data[key], dtype=np.float32)).to(torch.float32).contiguous()
            for key in data.files
        }


def validate_fp16_ids_coverage_and_shapes(
    model: nn.Module,
    keep_dir,
    apply_cfg: ApplyConfig,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int, int]]:
    cfg = _make_cfg(apply_cfg)
    targets = iter_target_linears(model, cfg=cfg, model_type=apply_cfg.model_type)
    if not targets:
        raise RuntimeError(f"No target linears found for model_type={apply_cfg.model_type}.")

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor, int, int]] = {}
    for _, _, full_name, linear in targets:
        fp16_ids, (n_tr, n_tc) = load_fp16_keep_ids_for_layer(
            keep_dir=keep_dir,
            layer_name=full_name,
            tile_m=cfg.tile_m,
            tile_n=cfg.tile_n,
        )
        expected_shape = _expected_tile_grid(linear.weight.shape, cfg.tile_m, cfg.tile_n)
        if (n_tr, n_tc) != expected_shape:
            raise RuntimeError(
                f"FP16 keep-id shape mismatch for {full_name}: "
                f"archive={(n_tr, n_tc)} expected={expected_shape}"
            )
        total = n_tr * n_tc
        if fp16_ids.numel() > 0:
            ids64 = fp16_ids.to(torch.int64)
            if int(ids64.min().item()) < 0 or int(ids64.max().item()) >= total:
                raise RuntimeError(
                    f"FP16 keep-id out of range for {full_name}: total_tiles={total} "
                    f"min={int(ids64.min().item())} max={int(ids64.max().item())}"
                )
        fp16_ids_i32 = fp16_ids.to(torch.int32).contiguous()
        fp_mask = torch.zeros((total,), dtype=torch.bool)
        if fp16_ids_i32.numel() > 0:
            fp_mask[fp16_ids_i32.to(torch.int64)] = True
        int8_ids_i32 = torch.nonzero(~fp_mask, as_tuple=False).reshape(-1).to(torch.int32).contiguous()
        out[full_name] = (fp16_ids_i32, int8_ids_i32, int(n_tr), int(n_tc))
    return out


def _resolve_safetensor_shards(local_model_dir: Path) -> Iterable[Path]:
    local_model_dir = Path(local_model_dir)
    index_path = local_model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid or empty weight_map in index: {index_path}")
        shard_files = sorted(set(weight_map.values()))
        for rel in shard_files:
            shard_path = local_model_dir / rel
            if not shard_path.exists():
                raise RuntimeError(f"Missing shard file listed in index: {shard_path}")
            yield shard_path
        return

    single_path = local_model_dir / "model.safetensors"
    if single_path.exists():
        yield single_path
        return
    raise RuntimeError(
        f"No safetensors checkpoint found in {local_model_dir}. "
        "Expected model.safetensors or model.safetensors.index.json."
    )


def _cast_loaded_tensor(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(dtype=torch_dtype)
    return tensor


def load_mixed_precision_model_from_masks(
    local_model_dir,
    keep_dir,
    apply_cfg: ApplyConfig,
    torch_dtype: torch.dtype,
    device,
    attn_implementation: Optional[str] = None,
):
    """
    Load checkpoint shards and quantize target linears during load.

    This avoids loading dense target linear weights into the final model only to
    quantize them in a second pass.
    """
    local_model_dir = Path(local_model_dir)
    load_started_at = time.perf_counter()
    cfg = _make_cfg(apply_cfg)
    device = torch.device(device)

    config_kwargs = {}
    if attn_implementation:
        config_kwargs["attn_implementation"] = attn_implementation
    config = AutoConfig.from_pretrained(str(local_model_dir), **config_kwargs)

    init_ctx = _init_empty_weights if _init_empty_weights is not None else nullcontext
    with init_ctx():
        try:
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch_dtype,
                **config_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                **config_kwargs,
            )

    targets = iter_target_linears(model, cfg=cfg, model_type=apply_cfg.model_type)
    if not targets:
        raise RuntimeError(f"No target linears found for model_type={apply_cfg.model_type}.")

    target_meta: Dict[str, Tuple[nn.Module, str]] = {
        full_name: (parent, attr_name) for parent, attr_name, full_name, _ in targets
    }
    fp16_data = validate_fp16_ids_coverage_and_shapes(model=model, keep_dir=keep_dir, apply_cfg=apply_cfg)
    target_names = set(fp16_data.keys())
    quant_style = str(getattr(apply_cfg, "quant_style", "gi")).lower()
    iaa_stats_by_stem: Dict[str, torch.Tensor] = {}
    if quant_style == "iaa":
        iaa_stats_by_stem = _load_iaa_channel_stats(keep_dir=Path(keep_dir), apply_cfg=apply_cfg)

    pending_target_bias: Dict[str, torch.Tensor] = {}
    loaded_non_target = 0
    replaced_targets = 0
    quant_pack_total = 0.0
    non_target_load_total = 0.0

    for shard_path in _resolve_safetensor_shards(local_model_dir):
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)

                if key.endswith(".weight"):
                    mod_name = key[: -len(".weight")]
                    if mod_name in target_names:
                        pack_started_at = time.perf_counter()
                        parent, attr_name = target_meta[mod_name]
                        w_fp = _cast_loaded_tensor(tensor, torch_dtype=torch_dtype)
                        fp16_ids, int8_ids, n_tr, n_tc = fp16_data[mod_name]
                        iaa_col_importance = None
                        if quant_style == "iaa":
                            stem = mod_name.replace(".", "_")
                            iaa_col_importance = iaa_stats_by_stem.get(stem, None)
                        pack = pack_weight_from_tile_ids(
                            W_fp=w_fp,
                            int8_tile_ids=int8_ids,
                            fp16_tile_ids=fp16_ids,
                            n_tr=n_tr,
                            n_tc=n_tc,
                            iaa_col_importance=iaa_col_importance,
                            cfg=cfg,
                        )
                        new_mod = MixedLinear(pack=pack, bias=None, cfg=cfg).to(device)
                        setattr(parent, attr_name, new_mod)
                        replaced_targets += 1

                        if mod_name in pending_target_bias:
                            bias = _cast_loaded_tensor(pending_target_bias.pop(mod_name), torch_dtype=torch_dtype)
                            new_mod.bias = nn.Parameter(bias.to(device=device).detach().clone(), requires_grad=False)
                        quant_pack_total += time.perf_counter() - pack_started_at
                        continue

                if key.endswith(".bias"):
                    mod_name = key[: -len(".bias")]
                    if mod_name in target_names:
                        mod = dict(model.named_modules()).get(mod_name, None)
                        if isinstance(mod, MixedLinear):
                            bias = _cast_loaded_tensor(tensor, torch_dtype=torch_dtype)
                            mod.bias = nn.Parameter(bias.to(device=device).detach().clone(), requires_grad=False)
                        else:
                            pending_target_bias[mod_name] = tensor
                        continue

                value = _cast_loaded_tensor(tensor, torch_dtype=torch_dtype)
                non_target_started_at = time.perf_counter()
                set_module_tensor_to_device(model, key, device, value=value)
                non_target_load_total += time.perf_counter() - non_target_started_at
                loaded_non_target += 1

    if pending_target_bias:
        raise RuntimeError(
            f"Found target bias tensors without corresponding target weight loads, e.g. {list(pending_target_bias)[:3]}"
        )

    module_map = dict(model.named_modules())
    final_replaced = 0
    for name in target_names:
        mod = module_map.get(name, None)
        if not isinstance(mod, MixedLinear):
            raise RuntimeError(f"Target module is not MixedLinear after load-time apply: {name}")
        final_replaced += 1

    if final_replaced != len(target_names):
        raise RuntimeError(f"Replacement mismatch after load: {final_replaced}/{len(target_names)}")

    # Buffers that are not in the checkpoint (e.g. RoPE inv_freq) are created on
    # the host by from_config; keep every buffer on the model device so a forward
    # never issues an implicit host-to-device copy (which also blocks CUDA-graph
    # capture).
    for module in model.modules():
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device != device and buffer.device.type != "meta":
                module._buffers[buffer_name] = buffer.to(device)
        for attr in ("original_inv_freq",):
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor) and value.device != device:
                setattr(module, attr, value.to(device))

    fused_groups = fuse_projection_groups(model)

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    load_total = time.perf_counter() - load_started_at
    return model, {
        "num_targets": len(target_names),
        "num_replaced": final_replaced,
        "num_replaced_during_load": replaced_targets,
        "num_loaded_non_target_tensors": loaded_non_target,
        "load_total": float(load_total),
        "quant_pack_total": float(quant_pack_total),
        "non_target_load_total": float(non_target_load_total),
        "attn_implementation": attn_implementation,
        "fused_projection_groups": int(fused_groups),
    }
