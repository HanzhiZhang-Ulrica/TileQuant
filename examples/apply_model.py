from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantization import load_model_and_tokenizer


KEEP_TAG = "keep01"
QUANT_STYLE = "gi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 512
GEN_MAX_NEW_TOKENS = 64
MODEL_NAME_OR_PATH = str(REPO_ROOT.parent / "model_path" / "meta-llama_Llama-3.2-3B-Instruct")

PROMPT = """
[INST]Summarize the following paragraphs:[/INST]

Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had
peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought
Alice "without pictures or conversation?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the
pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit
with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself,
"Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered
at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and
looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit
with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and
fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

Answer:
"""


def _clean_completion(text: str) -> str:
    out = text
    for marker in ("[INST]", "[/INST]", "[ENDINST]"):
        idx = out.find(marker)
        if idx >= 0:
            out = out[:idx]
    return out.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a quantized model and run the built-in summarization prompt.")
    parser.add_argument("--model_name_or_path", type=str, default=MODEL_NAME_OR_PATH)
    parser.add_argument("--keep_tag", type=str, default=KEEP_TAG)
    parser.add_argument("--quant_style", type=str, choices=["gi", "iaa"], default=QUANT_STYLE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--gen_max_new_tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    model, tokenizer, _ = load_model_and_tokenizer(
        args.model_name_or_path,
        quant_style=args.quant_style,
        keep_tag=args.keep_tag,
        device=args.device,
        download_if_missing=True,
    )

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=int(args.max_seq_len),
    )
    inputs = {key: value.to(args.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(args.gen_max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    response_ids = output_ids[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(_clean_completion(response_text))


if __name__ == "__main__":
    main()
