from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _next_token_ce_loss(model_outputs, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )


def _model_input_device(model: nn.Module) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and len(device_map) > 0:
        first = next(iter(device_map.values()))
        if isinstance(first, str) and first not in {"cpu", "disk"}:
            return torch.device(first)
    return next(model.parameters()).device


def collect_gi_tile_scores(
    model: nn.Module,
    dataloader: Iterable,
    target_filter_fn: Callable[[str], bool],
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    tile_m = int(_cfg_get(cfg, "tile_m", 64))
    tile_n = int(_cfg_get(cfg, "tile_n", 64))
    calib_cfg = _cfg_get(cfg, "calib", {})
    num_samples = _cfg_get(calib_cfg, "num_samples", None)
    max_tokens = _cfg_get(calib_cfg, "max_tokens", None)
    checkpoint_dir_raw = _cfg_get(calib_cfg, "checkpoint_dir", None)
    checkpoint_every = int(_cfg_get(calib_cfg, "checkpoint_every_batches", 0))
    resume_from_checkpoint = bool(_cfg_get(calib_cfg, "resume_from_checkpoint", False))
    checkpoint_path: Path | None = None
    if checkpoint_dir_raw:
        checkpoint_dir = Path(str(checkpoint_dir_raw)).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "gi_tile_scores_progress.pt"

    targets: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target_filter_fn(name):
            targets[name] = module
    if not targets:
        raise RuntimeError("No target linear layers matched for GI calibration")

    def _empty_tile_scores(module: nn.Linear) -> torch.Tensor:
        m, k = module.weight.shape
        n_tr = (int(m) + tile_m - 1) // tile_m
        n_tc = (int(k) + tile_n - 1) // tile_n
        return torch.zeros((n_tr, n_tc), dtype=torch.float32)

    processed_batches = 0
    tokens_seen = 0
    tile_scores_dict: Dict[str, torch.Tensor] = {name: _empty_tile_scores(module) for name, module in targets.items()}

    if checkpoint_path is not None and resume_from_checkpoint and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        processed_batches = int(ckpt.get("processed_batches", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        saved_scores = ckpt.get("tile_scores", {})
        for name, module in targets.items():
            if name not in saved_scores:
                continue
            saved = torch.as_tensor(saved_scores[name], dtype=torch.float32)
            empty = _empty_tile_scores(module)
            if saved.numel() == empty.numel():
                tile_scores_dict[name] = saved.reshape_as(empty).contiguous()

    def _save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        payload = {
            "processed_batches": int(processed_batches),
            "tokens_seen": int(tokens_seen),
            "tile_scores": {
                name: arr.detach().cpu().numpy().astype(np.float32, copy=False)
                for name, arr in tile_scores_dict.items()
            },
        }
        tmp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(payload, tmp_path)
        tmp_path.replace(checkpoint_path)

    input_device = _model_input_device(model)
    model_was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    total_batches = None
    try:
        total_batches = len(dataloader)  # type: ignore[arg-type]
    except Exception:
        total_batches = None
    if num_samples is not None:
        total_batches = min(int(num_samples), int(total_batches)) if total_batches is not None else int(num_samples)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_batches,
            initial=min(processed_batches, total_batches) if total_batches is not None else processed_batches,
            desc="GI calib",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < processed_batches:
                continue
            if not isinstance(batch, dict):
                raise TypeError("Calibration dataloader must yield dict batches with input_ids")

            model.zero_grad(set_to_none=True)
            model_inputs = {k: (v.to(input_device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            input_ids = model_inputs.get("input_ids", None)
            if input_ids is None:
                raise ValueError("Calibration batch dict must include input_ids")

            outputs = model(**model_inputs)
            loss = _next_token_ce_loss(outputs, input_ids)
            loss.backward()

            with torch.no_grad():
                for name, module in targets.items():
                    grad = module.weight.grad
                    if grad is None:
                        raise RuntimeError(f"Missing gradient for target linear: {name}")

                    score = (module.weight.detach().to(torch.float32) * grad.detach().to(torch.float32)).abs()
                    m, k = score.shape
                    m_pad = ((m + tile_m - 1) // tile_m) * tile_m
                    k_pad = ((k + tile_n - 1) // tile_n) * tile_n
                    if m_pad != m or k_pad != k:
                        score_pad = score.new_zeros((m_pad, k_pad))
                        score_pad[:m, :k] = score
                        score = score_pad
                    n_tr = score.shape[0] // tile_m
                    n_tc = score.shape[1] // tile_n
                    tile_scores = score.reshape(n_tr, tile_m, n_tc, tile_n).sum(dim=(1, 3))
                    tile_scores_dict[name] += tile_scores.detach().cpu().to(torch.float32)

            processed_batches += 1
            tokens_seen += int(input_ids.numel())
            if pbar is not None:
                pbar.update(1)
            if checkpoint_path is not None and checkpoint_every > 0 and processed_batches % checkpoint_every == 0:
                _save_checkpoint()
            if num_samples is not None and processed_batches >= num_samples:
                break
            if max_tokens is not None and tokens_seen >= max_tokens:
                break
        _save_checkpoint()
    finally:
        if pbar is not None:
            pbar.close()
        model.zero_grad(set_to_none=True)
        if model_was_training:
            model.train()

    return {name: score.contiguous() for name, score in tile_scores_dict.items()}


def collect_iaa_tile_scores(
    model: nn.Module,
    dataloader: Iterable,
    target_filter_fn: Callable[[str], bool],
    cfg: Any,
) -> Dict[str, torch.Tensor]:
    scores, _ = collect_iaa_tile_scores_with_channel_stats(
        model=model,
        dataloader=dataloader,
        target_filter_fn=target_filter_fn,
        cfg=cfg,
    )
    return scores


def collect_iaa_tile_scores_with_channel_stats(
    model: nn.Module,
    dataloader: Iterable,
    target_filter_fn: Callable[[str], bool],
    cfg: Any,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    IAA-like tile importance:
    - Collect activation magnitude statistics per input channel for each target linear
    - Score each weight entry by |W| * E[|x_channel|]
    - Reduce to tile scores by sum over each tile
    """
    tile_m = int(_cfg_get(cfg, "tile_m", 64))
    tile_n = int(_cfg_get(cfg, "tile_n", 64))
    calib_cfg = _cfg_get(cfg, "calib", {})
    num_samples = _cfg_get(calib_cfg, "num_samples", None)
    max_tokens = _cfg_get(calib_cfg, "max_tokens", None)
    checkpoint_dir_raw = _cfg_get(calib_cfg, "checkpoint_dir", None)
    checkpoint_every = int(_cfg_get(calib_cfg, "checkpoint_every_batches", 0))
    resume_from_checkpoint = bool(_cfg_get(calib_cfg, "resume_from_checkpoint", False))
    checkpoint_path: Path | None = None
    if checkpoint_dir_raw:
        checkpoint_dir = Path(str(checkpoint_dir_raw)).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "iaa_activation_progress.pt"

    targets: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target_filter_fn(name):
            targets[name] = module
    if not targets:
        raise RuntimeError("No target linear layers matched for IAA calibration")

    act_sum: Dict[str, torch.Tensor] = {}
    act_count: Dict[str, int] = {}
    hooks = []

    for name, module in targets.items():
        layer_device = module.weight.device
        act_sum[name] = torch.zeros((module.in_features,), device=layer_device, dtype=torch.float64)
        act_count[name] = 0

        def _pre_hook(mod, inputs, layer_name=name):
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
            if x2.numel() == 0:
                return
            dst = act_sum[layer_name].device
            act_sum[layer_name] += x2.abs().sum(dim=0).to(device=dst, dtype=torch.float64)
            act_count[layer_name] += int(x2.shape[0])

        hooks.append(module.register_forward_pre_hook(_pre_hook))

    processed_batches = 0
    tokens_seen = 0
    if checkpoint_path is not None and resume_from_checkpoint and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        act_count_ckpt = ckpt.get("act_count", {})
        act_sum_ckpt = ckpt.get("act_sum", {})
        for name in targets.keys():
            if name in act_count_ckpt:
                act_count[name] = int(act_count_ckpt[name])
            if name in act_sum_ckpt:
                t = torch.as_tensor(act_sum_ckpt[name], dtype=torch.float64, device=act_sum[name].device)
                if t.numel() == act_sum[name].numel():
                    act_sum[name] = t.reshape_as(act_sum[name])
        processed_batches = int(ckpt.get("processed_batches", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))

    def _save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        payload = {
            "processed_batches": int(processed_batches),
            "tokens_seen": int(tokens_seen),
            "act_count": {k: int(v) for k, v in act_count.items()},
            "act_sum": {k: v.detach().cpu().numpy().astype(np.float64, copy=False) for k, v in act_sum.items()},
        }
        tmp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(payload, tmp_path)
        tmp_path.replace(checkpoint_path)

    model_was_training = model.training
    model.eval()
    input_device = _model_input_device(model)

    total_batches = None
    try:
        total_batches = len(dataloader)  # type: ignore[arg-type]
    except Exception:
        total_batches = None
    if num_samples is not None:
        total_batches = min(int(num_samples), int(total_batches)) if total_batches is not None else int(num_samples)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total_batches,
            initial=min(processed_batches, total_batches) if total_batches is not None else processed_batches,
            desc="IAA calib",
            dynamic_ncols=True,
            leave=True,
        )
    last_printed = processed_batches

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx < processed_batches:
                    continue
                if not isinstance(batch, dict):
                    raise TypeError("Calibration dataloader must yield dict batches with input_ids")
                model_inputs = {k: (v.to(input_device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                input_ids = model_inputs.get("input_ids", None)
                if input_ids is None:
                    raise ValueError("Calibration batch dict must include input_ids")
                _ = model(**model_inputs)

                processed_batches += 1
                tokens_seen += int(input_ids.numel())
                if pbar is not None:
                    pbar.update(1)
                elif processed_batches - last_printed >= 50:
                    print(f"[IAA calib] processed_batches={processed_batches} tokens_seen={tokens_seen}", flush=True)
                    last_printed = processed_batches
                if checkpoint_path is not None and checkpoint_every > 0 and processed_batches % checkpoint_every == 0:
                    _save_checkpoint()
                if num_samples is not None and processed_batches >= num_samples:
                    break
                if max_tokens is not None and tokens_seen >= max_tokens:
                    break
            _save_checkpoint()
    finally:
        if pbar is not None:
            pbar.close()
        for h in hooks:
            h.remove()
        if model_was_training:
            model.train()

    tile_scores_dict: Dict[str, torch.Tensor] = {}
    channel_stats_dict: Dict[str, torch.Tensor] = {}
    for name, module in targets.items():
        cnt = int(act_count[name])
        if cnt <= 0:
            raise RuntimeError(f"No activation samples captured for target linear: {name}")

        mean_abs_x = (act_sum[name] / float(cnt)).to(torch.float32)
        channel_stats_dict[name] = mean_abs_x.detach().cpu().to(torch.float32)
        W = module.weight.detach().to(torch.float32)
        S = W.abs() * mean_abs_x.view(1, -1)

        m, k = S.shape
        m_pad = ((m + tile_m - 1) // tile_m) * tile_m
        k_pad = ((k + tile_n - 1) // tile_n) * tile_n
        if m_pad != m or k_pad != k:
            s_pad = S.new_zeros((m_pad, k_pad))
            s_pad[:m, :k] = S
            S = s_pad

        n_tr = S.shape[0] // tile_m
        n_tc = S.shape[1] // tile_n
        tile_scores = S.reshape(n_tr, tile_m, n_tc, tile_n).sum(dim=(1, 3))
        tile_scores_dict[name] = tile_scores.detach().cpu().to(torch.float32)

    return tile_scores_dict, channel_stats_dict
