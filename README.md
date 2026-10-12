# TileQuant

TileQuant assigns precision at the granularity of 64x64 weight tiles. A small
fraction of important tiles in every decoder linear layer (Q/K/V/O,
Gate/Up/Down) stays in BF16, and all other tiles are stored as INT8 with one
scale per tile. Activations are quantized to INT8 per 64-column tile. A mixed
INT8/BF16 GEMM computes each output tile from its INT8 tiles (exact INT32 tile
products, scaled in FP32) and its BF16 tiles.

Tile importance comes from one of two calibration criteria:

- **GI** (gradient informed): |W| x |dL/dW| summed per tile over calibration data.
- **IAA** (input activation aware): |W| x mean |X| per tile, plus
  channel-aware INT8 scale selection.

## Installation

Requires Python 3.10+, CUDA 12.x and an NVIDIA GPU with INT8 tensor cores
(tested on A100).

```bash
pip install -r requirements.txt
```

## Repository layout

```text
build_decisions.py          calibration entry point (GI / IAA tile maps)
Decision/<model>/<gi|iaa>/  released tile maps, keep01/02/05/10/20
src/kernels/                tile packing, activation quantization, calibration
src/kernels/triton/         mixed INT8/BF16 tile GEMM (Triton)
src/models/mixed_linear.py  MixedLinear replacement module
src/quantization/           load-time packing and model construction
src/runtime/graph_decode.py CUDA-graph greedy decoding
examples/                   inference and benchmark scripts
```

Tile maps for Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct and
Qwen2.5-14B-Instruct are included, so calibration is optional.

## Build tile maps

```bash
python build_decisions.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --quant_style gi \
  --dataset_id alexfabbri/multi_news \
  --dataset_split train \
  --keep_fracs 0.01 0.02 0.05 0.10 0.20 \
  --download_if_missing
```

Use `--quant_style iaa` for IAA. Artifacts are written to
`Decision/<model_folder>/<gi|iaa>/`; `tile_masks/keepXX` holds the BF16 tile
ids for a retention of XX%.

## Run a quantized model

```bash
python examples/apply_model.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --quant_style gi \
  --keep_tag keep01
```

Target weights are packed while the checkpoint is loaded; no quantized
checkpoint is stored.

## Efficiency benchmark

```bash
python examples/benchmark_generation.py \
  --model_dir /path/to/Llama-3.2-3B-Instruct \
  --method tilequant --quant_style gi --keep_tag keep01 \
  --prompt 2048 --output 256 --batch 4
```

Reports median TTFT, TPOT, output tokens/s, energy per 1K generated tokens
(NVML) and peak memory over five timed requests after two warm-ups.
`--method bf16` runs the dense BF16 model with the same protocol.
