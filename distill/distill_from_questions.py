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
        description="Distillation from dataset and questions file."
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
        "--model_name",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Model to train.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/scratch/rileyis/ki-rl/distill/out/questions_distill",
        help="Directory to write checkpoints and final model.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=float, default=1, help="Training epochs.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Gradient accumulation steps.",
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
        default=1024,
        help="Maximum generated completion length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
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

def load_question_dataset(path: str, data_path: str) -> Dataset:
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

    return Dataset.from_list(rows)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_question_dataset(
        args.questions_path,
        args.dataset_path,
    )
    print(f"Loaded {len(dataset)} prompts from {args.questions_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        max_prompt_length = 1024,
        max_completion_length = 1024,
        num_train_epochs = args.num_train_epochs,
        save_steps = 100,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = 0.0,
        vllm_importance_sampling_correction = True,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    print(f"Saved final model to: {final_dir}")