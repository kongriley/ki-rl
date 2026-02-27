from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import torch
from datasets import Dataset
import argparse
from typing import List

from question_distill_pipeline import QuestionThenDistillConfig, build_question_distillation_dataset
from question_rl import QuestionGeneratorGRPOTrainer, QuestionGRPOTrainConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Two-loop question generation + prompt distillation")
    parser.add_argument("--dataset_path", type=str, default="/data/scratch/rileyis/ki-rl/data/wiki_20/data.json", help="Dataset path")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
        help="Base model (student init). Also default for question model and teacher if not provided.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations")
    parser.add_argument("--num_prompts_per_batch", type=int, default=8,
                        help="Number of prompts per batch for each GPU")
    parser.add_argument("--top_entropy_quantile", type=float, default=1.0,
                        help="Top entropy quantile")
    parser.add_argument("--output_dir", type=str, default="/data/scratch/rileyis/ki-rl/distill/out",
                        help="Output directory")

    # Question generation (GRPO)
    parser.add_argument("--question_model_name", type=str, default=None, help="Question generator model to optimize (defaults to base).")
    parser.add_argument("--teacher_model_name", type=str, default=None, help="Teacher model for student distillation reference (defaults to base).")
    parser.add_argument("--qg_max_steps", type=int, default=200, help="GRPO max steps for question generator.")
    parser.add_argument("--qg_learning_rate", type=float, default=5e-6, help="GRPO learning rate for question generator.")
    parser.add_argument("--qg_num_generations", type=int, default=4, help="GRPO num_generations for question generator.")
    parser.add_argument("--qg_per_device_batch_size", type=int, default=1, help="GRPO per-device batch size for question generator.")
    parser.add_argument("--qg_grad_accum_steps", type=int, default=1, help="GRPO gradient accumulation for question generator.")
    parser.add_argument("--qg_temperature", type=float, default=1.0, help="Sampling temperature for question generator.")
    parser.add_argument("--qg_top_p", type=float, default=0.95, help="Top-p for question generator.")
    parser.add_argument("--questions_per_text", type=int, default=1, help="How many questions to generate per text for distillation.")
    parser.add_argument("--max_question_new_tokens", type=int, default=64, help="Max tokens for generated questions.")
    parser.add_argument("--max_teacher_answer_new_tokens", type=int, default=512, help="Max tokens for teacher example answers.")
    parser.add_argument("--dataset_max_texts", type=int, default=256, help="Max number of texts to use for RL+dataset build.")

    # Two-loop settings
    parser.add_argument("--outer_iters", type=int, default=3, help="Number of outer/inner alternations.")
    parser.add_argument("--inner_epochs_per_iter", type=int, default=3, help="Student distillation epochs per iteration.")
    return parser.parse_args()

def load_dataset(path, test_size=0.1, tokenizer=None):
    """Load a text dataset and split into train/test."""

    print(f"Loading dataset from {path}")
    if os.path.isdir(path):
        dataset = Dataset.load_from_disk(path)
    else:
        dataset = Dataset.from_json(path)
    
    print(f"Loaded dataset with {len(dataset)} examples")
    
    if test_size == 0.0:
        return dataset, None
    
    # Split into train and test
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    
    print(f"Split into {len(train_dataset)} training and {len(test_dataset)} validation examples")

    return train_dataset, test_dataset

def _extract_texts_for_questioning(dataset: Dataset, *, max_items: int) -> List[str]:
    return [ex["text"] for ex in dataset][:max_items]

if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    model_name = args.base_model_name

    # Load tokenizer for distillation stage
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base dataset (used either directly for distillation, or as a source of texts)
    dataset_raw, _ = load_dataset(dataset_path, test_size=0.0, tokenizer=tokenizer)

    question_model_name = args.question_model_name or model_name
    teacher_model_name = args.teacher_model_name or model_name

    texts = _extract_texts_for_questioning(
        dataset_raw,
        max_items=args.dataset_max_texts,
    )
    if len(texts) == 0:
        raise RuntimeError("Could not extract any texts from dataset for question generation.")

    qd_cfg = QuestionThenDistillConfig(
        questions_per_text=args.questions_per_text,
        qg_max_question_new_tokens=args.max_question_new_tokens,
        qg_max_prompt_length=1024,
        qg_temperature=args.qg_temperature,
        qg_top_p=args.qg_top_p,
    )

    # Two-loop: alternate (outer) question optimization (GRPO) and (inner) student distillation.
    os.makedirs(args.output_dir, exist_ok=True)

    current_student_path = model_name
    current_question_model_path = question_model_name

    for it in range(args.outer_iters):
        # Outer loop: train question generator to minimize student success.
        qg_dir = os.path.join(args.output_dir, f"qg_iter_{it}")
        os.makedirs(qg_dir, exist_ok=True)

        qg_cfg = QuestionGRPOTrainConfig(
            output_dir=qg_dir,
            learning_rate=args.qg_learning_rate,
            max_steps=args.qg_max_steps,
            per_device_train_batch_size=args.qg_per_device_batch_size,
            gradient_accumulation_steps=args.qg_grad_accum_steps,
            num_generations=args.qg_num_generations,
            max_prompt_length=1024,
            max_question_new_tokens=args.max_question_new_tokens,
            temperature=args.qg_temperature,
            top_p=args.qg_top_p,
            logging_steps=10,
            seed=0,
            device=None,
            reward_max_answer_new_tokens=args.max_teacher_answer_new_tokens,
        )

        q_trainer = QuestionGeneratorGRPOTrainer(
            question_model=current_question_model_path,
            question_tokenizer=None,
            config=qg_cfg,
            reward_student_model=current_student_path,
            reward_teacher_model=teacher_model_name,
            reward_tokenizer=None,
        )
        q_trainer.train(texts)

        # Persist question model for the next iteration
        q_trainer.q_model.save_pretrained(qg_dir)
        q_trainer.q_tok.save_pretrained(qg_dir)
        current_question_model_path = qg_dir

        # Build distillation dataset for this iteration.
        dataset = build_question_distillation_dataset(
            texts=texts,
            question_model=q_trainer.q_model,
            question_tokenizer=q_trainer.q_tok,
            cfg=qd_cfg,
        )

        ds_dir = os.path.join(args.output_dir, f"distill_dataset_iter_{it}")
        os.makedirs(ds_dir, exist_ok=True)
        dataset.save_to_disk(ds_dir)

        # Inner loop: distill student (question-only) from teacher (doc+question).
        student_out_dir = os.path.join(args.output_dir, f"student_iter_{it}")
        os.makedirs(student_out_dir, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(current_student_path, torch_dtype=torch.bfloat16)
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(current_student_path)

        config = DistilConfig(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.3,
            learning_rate=args.learning_rate,
            beta=0.0,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            use_liger_loss=False,
            logging_steps=1,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.num_prompts_per_batch * args.num_generations,
            num_generations=args.num_generations,
            num_iterations=1,
            max_prompt_length=1024,
            max_completion_length=1024,
            num_train_epochs=args.inner_epochs_per_iter,
            save_steps=50000,
            save_only_model=True,
            max_grad_norm=1,
            report_to="wandb",
            output_dir=student_out_dir,
            mask_truncated_completions=False,
            log_completions=False,
            alpha=0.0,
            full_logit_distillation=True,
            top_entropy_quantile=args.top_entropy_quantile,
            sync_ref_model=True,
            ref_model_sync_steps=1,
            ref_model_mixup_alpha=0.01,
        )
        trainer = DistilTrainer(
            model=model,
            ref_model=teacher_model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        final_dir = os.path.join(student_out_dir, "final_model")
        trainer.save_model(final_dir)
        current_student_path = final_dir
