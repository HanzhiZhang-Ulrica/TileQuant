"""Greedy-generation latency, throughput and energy for one (prompt, output, batch) shape.

Protocol: a fixed token prompt of length P repeated B times; two warm-up and
five timed requests; median TTFT (request through first token), TPOT
(following tokens), output tokens/s and NVML energy per 1K generated tokens;
maximum peak allocated memory.  TileQuant models decode with CUDA graphs
(``src/runtime/graph_decode.py``); ``--method bf16`` runs the dense model with
the same generation loop for reference.

Example:
    python examples/benchmark_generation.py --model_dir /path/to/Llama-3.2-3B-Instruct \
        --method tilequant --quant_style gi --keep_tag keep01 --prompt 2048 --output 256 --batch 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantization import ApplyConfig, TileConfig, infer_model_type, load_mixed_precision_model_from_masks  # noqa: E402
from src.runtime.graph_decode import CudaGraphGreedyDecoder  # noqa: E402

WARMUPS, REPEATS = 2, 5
PROMPT_TEXT = "Tile-level mixed-precision quantization benchmark. "


def load_model(args) -> torch.nn.Module:
    if args.method == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}
        )
    else:
        decision = REPO_ROOT / "Decision" / Path(args.model_dir).name / args.quant_style
        apply_cfg = ApplyConfig(
            model_type=infer_model_type(Path(args.model_dir)),
            dtype_out="bf16",
            tile=TileConfig(tile_m=64, tile_n=64),
            module_filters=[],
            quant_style=args.quant_style,
            iaa_channel_stats_path=str(decision / "iaa_channel_mean_abs.npz") if args.quant_style == "iaa" else None,
        )
        model, _ = load_mixed_precision_model_from_masks(
            local_model_dir=Path(args.model_dir),
            keep_dir=decision / "tile_masks" / args.keep_tag,
            apply_cfg=apply_cfg,
            torch_dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
            attn_implementation="sdpa",
        )
    model.eval()
    model.config.use_cache = True
    return model


def energy_mj(handle) -> int | None:
    if handle is None:
        return None
    import pynvml

    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)


@torch.inference_mode()
def eager_request(model, tokens, batch, output_tokens, handle):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    energy0 = energy_mj(handle)
    start = time.perf_counter()
    inputs = torch.tensor([tokens] * batch, device="cuda:0")
    out = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
    current, cache = out.logits[:, -1:].argmax(-1), out.past_key_values
    torch.cuda.synchronize()
    first = time.perf_counter()
    for _ in range(output_tokens - 1):
        out = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        current, cache = out.logits[:, -1:].argmax(-1), out.past_key_values
    torch.cuda.synchronize()
    return start, first, time.perf_counter(), energy0, energy_mj(handle)


@torch.inference_mode()
def graph_request(decoder, tokens, batch, output_tokens, handle):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    energy0 = energy_mj(handle)
    start = time.perf_counter()
    decoder.prefill([tokens] * batch)
    torch.cuda.synchronize()
    first = time.perf_counter()
    decoder.decode(output_tokens - 1)
    torch.cuda.synchronize()
    return start, first, time.perf_counter(), energy0, energy_mj(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--method", choices=["tilequant", "bf16"], default="tilequant")
    parser.add_argument("--quant_style", choices=["gi", "iaa"], default="gi")
    parser.add_argument("--keep_tag", default="keep01")
    parser.add_argument("--prompt", type=int, default=2048)
    parser.add_argument("--output", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except Exception:
        handle = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    pattern = tokenizer(PROMPT_TEXT, add_special_tokens=False).input_ids
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    tokens = bos + (pattern * (args.prompt // len(pattern) + 1))[: args.prompt - len(bos)]

    model = load_model(args)
    if args.method == "tilequant":
        decoder = CudaGraphGreedyDecoder.prepare(
            model, prompt_tokens=args.prompt, max_new_tokens=args.output, batch_size=args.batch
        )
        run = lambda: graph_request(decoder, tokens, args.batch, args.output, handle)  # noqa: E731
    else:
        run = lambda: eager_request(model, tokens, args.batch, args.output, handle)  # noqa: E731

    for _ in range(WARMUPS):
        run()
    rows = []
    for _ in range(REPEATS):
        start, first, end, e0, e1 = run()
        rows.append({
            "ttft_ms": (first - start) * 1000,
            "tpot_ms": (end - first) * 1000 / max(args.output - 1, 1),
            "output_tok_s": args.batch * args.output / (end - start),
            "energy_j_per_1k_tokens": (e1 - e0) / (args.batch * args.output) if e0 is not None else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
    summary = {key: statistics.median(r[key] for r in rows) for key in ("ttft_ms", "tpot_ms", "output_tok_s")}
    if all(r["energy_j_per_1k_tokens"] is not None for r in rows):
        summary["energy_j_per_1k_tokens"] = statistics.median(r["energy_j_per_1k_tokens"] for r in rows)
    summary["peak_allocated_gib"] = max(r["peak_allocated_gib"] for r in rows)
    print(json.dumps({**vars(args), **summary}, indent=1))


if __name__ == "__main__":
    main()
