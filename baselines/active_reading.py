"""Active-reading baseline for continued pretraining (CPT).

For each document in a wiki-style JSON dataset:
1. Generate learning strategies via ``build_learning_strategy_prompt``.
2. Apply each strategy to the document via ``build_active_reading_prompt``.
3. Save the resulting enriched documents as a JSON file compatible with cpt.py.
4. Optionally launch cpt.py on the generated data.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from inference import format_instruct_user_prompt
from generate_questions import load_dataset_json


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_active_reading_prompt(strategy: str, chunk: str) -> str:
    return f"""
Here's a learning strategy:
{ strategy }
Apply this strategy to the following document:

<document>
{ chunk }
</document>
"""


def build_learning_strategy_prompt(chunk: str) -> str:
    return f"""
Consider the following document. What are some strategies specific to this document that I can use to help me learn and remember all of the information contained? Use markdown and prefix each strategy with ##.

<document>
{ chunk }
</document>
"""


# ---------------------------------------------------------------------------
# Strategy parsing
# ---------------------------------------------------------------------------

def parse_strategies(text: str) -> List[str]:
    """Split LLM output on ``## `` headings, returning each strategy block."""
    blocks = re.split(r"(?m)^##\s+", text)
    strategies = []
    for block in blocks:
        block = block.strip()
        if block:
            strategies.append(block)
    return strategies


# ---------------------------------------------------------------------------
# vLLM backend (mirrors data/generate_questions.py)
# ---------------------------------------------------------------------------

def _generate_vllm(
    prompts: List[str],
    model_name: str,
    tokenizer_name: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> List[str]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    formatted = [format_instruct_user_prompt(tokenizer, p) for p in prompts]

    free_mem, total_mem = torch.cuda.mem_get_info()
    free_frac = free_mem / total_mem
    target_utilization = min(0.6, free_frac - 0.05)
    target_utilization = max(target_utilization, 0.2)
    print(f"  vLLM: {free_mem/1e9:.1f}/{total_mem/1e9:.1f} GiB free "
          f"({free_frac:.1%}), gpu_memory_utilization={target_utilization:.2f}")

    llm_kwargs = dict(
        model=model_name,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=target_utilization,
    )
    if tokenizer_name != model_name:
        llm_kwargs["tokenizer"] = tokenizer_name

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(formatted, sampling_params)

    results = [output.outputs[0].text for output in outputs]

    del llm
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# OpenAI backend (mirrors eval/eval_questions.py)
# ---------------------------------------------------------------------------

def _openai_client():
    from openai import OpenAI
    import dotenv
    dotenv.load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _sanitize(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def _generate_openai(
    prompts: List[str],
    client,
    model: str,
    temperature: float = 0.7,
    max_workers: int = 8,
) -> List[str]:
    results = [None] * len(prompts)

    def _call(idx):
        messages = [{"role": "user", "content": _sanitize(prompts[idx])}]
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
        )
        return idx, resp.choices[0].message.content.strip()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, i) for i in range(len(prompts))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  LLM calls"):
            idx, text = fut.result()
            results[idx] = text
    return results


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def save_dataset_json(docs: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(docs)} documents -> {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_active_reading_docs(
    dataset: List[dict],
    generate_fn,
) -> List[dict]:
    """Generate active-reading documents for every passage in *dataset*.

    *generate_fn* takes a ``List[str]`` of prompts and returns a ``List[str]``
    of completions (either the vLLM or OpenAI wrapper).
    """

    # Step 1: generate learning strategies for each document
    print(f"Step 1/2: Generating learning strategies for {len(dataset)} documents …")
    strategy_prompts = [build_learning_strategy_prompt(doc["text"]) for doc in dataset]
    strategy_responses = generate_fn(strategy_prompts)

    all_strategies: List[List[str]] = []
    for resp in strategy_responses:
        parsed = parse_strategies(resp)
        if not parsed:
            parsed = [resp]
        all_strategies.append(parsed)

    # Step 2: for each document, apply all strategies and concatenate results
    print("Step 2/2: Applying strategies to generate active-reading documents …")
    ar_prompts = []
    ar_index = []
    for doc_idx, (doc, strategies) in enumerate(zip(dataset, all_strategies)):
        for strat_idx, strat in enumerate(strategies):
            ar_prompts.append(build_active_reading_prompt(strat, doc["text"]))
            ar_index.append((doc_idx, strat_idx))

    ar_responses = generate_fn(ar_prompts)

    doc_parts: Dict[int, List[str]] = {i: [] for i in range(len(dataset))}
    for (doc_idx, _), resp in zip(ar_index, ar_responses):
        doc_parts[doc_idx].append(resp)

    output_docs = []
    for doc_idx, doc in enumerate(dataset):
        combined = "\n\n".join(doc_parts[doc_idx])
        output_docs.append({
            "id": doc["id"],
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "text": combined,
        })

    return output_docs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate active-reading documents for CPT")
    p.add_argument("--data_path", type=str, default="data/wiki_20/data.json")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default: data dir / active_reading.json)")
    p.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--tokenizer", type=str, default=None,
                    help="Tokenizer (defaults to --model)")
    p.add_argument("--backend", choices=["hf", "openai"], default="hf")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=2048,
                    help="Max new tokens per generation (HF only)")
    p.add_argument("--max_workers", type=int, default=8,
                    help="Concurrent API calls (OpenAI only)")
    p.add_argument("--run_cpt", action="store_true",
                    help="Launch cpt.py on the generated data after generation")
    p.add_argument("--cpt_output_dir", type=str, default="out/cpt_active_reading")
    p.add_argument("--cpt_epochs", type=float, default=10.0)
    p.add_argument("--cpt_lr", type=float, default=5e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_name = args.tokenizer or args.model

    raw_dataset = load_dataset_json(args.data_path)
    dataset = [{"id": id, "text": text} for id, text in raw_dataset.items()]
    print(f"Loaded {len(dataset)} documents from {args.data_path}")

    if args.backend == "openai":
        client = _openai_client()
        generate_fn = lambda prompts: _generate_openai(
            prompts, client, args.model,
            temperature=args.temperature, max_workers=args.max_workers,
        )
    else:
        generate_fn = lambda prompts: _generate_vllm(
            prompts, args.model, tokenizer_name,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )

    output_docs = generate_active_reading_docs(dataset, generate_fn)

    if args.output is None:
        data_dir = os.path.dirname(os.path.abspath(args.data_path))
        args.output = os.path.join(data_dir, "active_reading.json")

    save_dataset_json(output_docs, args.output)

    if args.run_cpt:
        cpt_script = os.path.join(os.path.dirname(__file__), "cpt.py")
        cmd = [
            sys.executable, cpt_script,
            "--data_path", args.output,
            "--output_dir", args.cpt_output_dir,
            "--num_train_epochs", str(args.cpt_epochs),
            "--learning_rate", str(args.cpt_lr),
        ]
        print(f"\nLaunching CPT: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
