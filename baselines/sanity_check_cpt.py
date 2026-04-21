"""Sanity-check a continued-pretrained model against its base.

Compares per-passage NLL/perplexity on the CPT corpus, shows a prefix->continuation
completion for one passage, and runs a free-form prompt to confirm the model
hasn't collapsed. All in a single forward/generation pass per model.

Usage:
    python sanity_check_cpt.py \
        --base_model allenai/OLMo-2-1124-7B-Instruct \
        --cpt_model out/cpt_wiki_20 \
        --data_path data/wiki_20/data.json
"""

import argparse
import json
import math
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def passage_nll(model, tokenizer, text: str, max_length: int = 2048) -> float:
    """Average per-token NLL on `text` (lower = model assigns higher prob)."""
    ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).input_ids.to(model.device)
    if ids.size(1) < 2:
        return float("nan")
    out = model(input_ids=ids, labels=ids)
    return out.loss.item()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def load_passages(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def eval_model(tag: str, model_path: str, passages: List[dict], prefix_tokens: int):
    print(f"\n=== {tag}: {model_path} ===")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    nlls = []
    for p in passages:
        nll = passage_nll(model, tok, p["text"])
        nlls.append(nll)
        print(f"  [{p['id']:>10}] NLL={nll:.3f}  PPL={math.exp(nll):.2f}  {p['title'][:60]}")
    mean_nll = sum(nlls) / len(nlls)
    print(f"  mean NLL: {mean_nll:.3f}   mean PPL: {math.exp(mean_nll):.2f}")

    # Prefix-continuation demo on the first passage.
    first_text = passages[0]["text"]
    prefix_ids = tok(first_text, add_special_tokens=False).input_ids[:prefix_tokens]
    prefix = tok.decode(prefix_ids)
    cont = generate(model, tok, prefix, max_new_tokens=120)
    print("\n  -- prefix completion (passage 0) --")
    print(f"  PREFIX : ...{prefix[-200:]}")
    print(f"  MODEL  : {cont[:400]}")
    true_cont = tok.decode(
        tok(first_text, add_special_tokens=False).input_ids[prefix_tokens:prefix_tokens + 60]
    )
    print(f"  TRUTH  : {true_cont[:400]}")

    # Coherence check on an unrelated prompt.
    print("\n  -- free-form coherence probe --")
    probe = "The three most important inventions of the 20th century were"
    print(f"  PROMPT : {probe}")
    print(f"  MODEL  : {generate(model, tok, probe, max_new_tokens=80)[:400]}")

    del model
    torch.cuda.empty_cache()
    return nlls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--cpt_model", default="out/cpt_wiki_20")
    parser.add_argument("--data_path", default="data/wiki_20/data.json")
    parser.add_argument("--prefix_tokens", type=int, default=80)
    args = parser.parse_args()

    passages = load_passages(args.data_path)
    base_nlls = eval_model("BASE", args.base_model, passages, args.prefix_tokens)
    cpt_nlls = eval_model("CPT ", args.cpt_model, passages, args.prefix_tokens)

    print("\n=== per-passage NLL delta (CPT - BASE; negative = improved) ===")
    for p, b, c in zip(passages, base_nlls, cpt_nlls):
        print(f"  [{p['id']:>10}]  base={b:.3f}  cpt={c:.3f}  delta={c - b:+.3f}")
    mean_b = sum(base_nlls) / len(base_nlls)
    mean_c = sum(cpt_nlls) / len(cpt_nlls)
    print(
        f"\n  MEAN  base NLL={mean_b:.3f} (PPL {math.exp(mean_b):.2f})  "
        f"cpt NLL={mean_c:.3f} (PPL {math.exp(mean_c):.2f})  "
        f"delta={mean_c - mean_b:+.3f}"
    )


if __name__ == "__main__":
    main()
