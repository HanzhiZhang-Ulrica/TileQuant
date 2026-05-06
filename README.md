# TileQuant

TileQuant builds tile-level mixed-precision masks for LLM linear layers and loads a model with selected tiles kept in FP16/BF16 while the rest are packed as INT8.

## Build GI

GI (gradient informed) uses calibration gradients to score important weight tiles.

```bash
python build_decisions.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --quant_style gi \
  --dataset_id alexfabbri/multi_news \
  --dataset_split train \
  --keep_fracs 0.01 0.02 0.05 0.10 0.20 \
  --download_if_missing
```

This writes decision artifacts under:

```text
Decision/<model_folder>/gi/
```

## Build IAA

IAA (input activation aware) uses input activation statistics to score important tiles and channel-aware INT8 packing.

```bash
python build_decisions.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --quant_style iaa \
  --dataset_id alexfabbri/multi_news \
  --dataset_split train \
  --keep_fracs 0.01 0.02 0.05 0.10 0.20 \
  --download_if_missing
```

This writes decision artifacts under:

```text
Decision/<model_folder>/iaa/
```

## Use

Run the example loader with an existing keep ratio:

```bash
python examples/apply_model.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --quant_style gi \
  --keep_tag keep01
```

Use `--quant_style iaa` to load IAA artifacts. `keep_tag` selects the retained FP tile ratio, such as `keep01`, `keep05`, or `keep20`.
