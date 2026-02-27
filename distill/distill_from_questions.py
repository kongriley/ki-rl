from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_config import DistilConfig
from distil_trainer import DistilTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight question-only distillation from a JSONL prompt file."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/scratch/rileyis/ki-rl/data/wiki_20/data.json",
        help="Path to document dataset.",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default="/data/scratch/rileyis/ki-rl/data/wiki_20/gpt-5-mini_questions.jsonl",
        help="Path to JSONL with instruction/input fields.",
    )
    parser.add_argument(
        "--student_model_name",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Student model to train.",
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Teacher model used for distillation targets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/scratch/rileyis/ki-rl/distill/out/questions_distill",
        help="Directory to write checkpoints and final model.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of JSONL examples to load (0 means all).",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Completions sampled per prompt in DistilTrainer.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt token length.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=256,
        help="Maximum generated completion length.",
    )
    parser.add_argument(
        "--top_entropy_quantile",
        type=float,
        default=1.0,
        help="Entropy masking quantile for policy loss.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Enable vLLM colocated generation.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory fraction for colocated vLLM mode.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="HF Trainer reporting backend (e.g., 'none', 'wandb').",
    )
    return parser.parse_args()

def _build_prompt(example: Dict) -> str:
    question = str(example.get("input", "")).strip()
    return f"<Question>\n{question}\nAnswer:"

def _build_teacher_prompt(example: Dict, document: str) -> str:
    return (
        "Read the following passage carefully, then answer the question.\n\n"
        f"<Passage>\n{document}\n\n"
    ) + _build_prompt(example)

def load_question_dataset(path: str, data_path: str, *, max_samples: int) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = {d["id"]: d["text"] for d in dataset}

    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            ex = json.loads(line)
            document = dataset[ex.get("id")]
            prompt_text = _build_prompt(ex)
            prompt = [{"role": "user", "content": prompt_text}]
            teacher_prompt_text = _build_teacher_prompt(ex, document)
            teacher_prompt = [{"role": "user", "content": teacher_prompt_text}]
            rows.append(
                {
                    "prompt": prompt,
                    "teacher_prompt": teacher_prompt,
                    "id": ex.get("id"),
                    "title": ex.get("title"),
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break

    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_question_dataset(
        args.questions_path,
        args.dataset_path,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataset)} prompts from {args.questions_path}")

    config = DistilConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=2,
        bf16=True,
        beta=0.0,
        alpha=0.0,
        full_logit_distillation=False,
        top_entropy_quantile=args.top_entropy_quantile,
        mask_truncated_completions=False,
        log_completions=False,
        report_to=args.report_to,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        use_liger_kernel=True,
    )

    trainer = DistilTrainer(
        model=args.student_model_name,
        ref_model=None if args.teacher_model_name == args.student_model_name else args.teacher_model_name,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    print(f"Saved final model to: {final_dir}")


if __name__ == "__main__":
    main()
