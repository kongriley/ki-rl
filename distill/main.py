import argparse
import json
import os
from typing import Dict
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from trl import GRPOConfig, GRPOTrainer

from distil_config import DistilConfig
from distil_trainer import DistilTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_generation_iterations", type=int, default=3, help="Number of times to run the generation loop")
    p.add_argument("--num_question_generations", type=int, default=10, help="Number of questions to generate for each passage")
    p.add_argument("--dataset_path", default="./data/wiki_20/data.json")
    p.add_argument("--model_name", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--output_dir", default="./distill/out/grpo_distill")
    p.add_argument("--question_model_path", type=str, default=None,
                   help="Path to a pre-trained question model. Skips question generator training and uses this model directly for question generation.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    return p.parse_args()


def load_dataset(data_path: str) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = {d["id"]: d["text"] for d in dataset}
    return dataset

"""
Question generation functions
"""

def _create_question_prompt(text: str) -> str:
    return f"""
    Using the following passage, generate one free-response question about the passage.
    This question will be used in a separate examination in two weeks, where the students are not given the passage, so be clear about the context, but do not explicitly reference the passage in the question.
    The answers should be answerable solely based on the information in the passage. However, if the student has never seen the passage before, the answer should not be answerable (i.e. it should not contain extraneous information that is not in the passage).

    <Passage>
    {text}

    <Question>
    """

def _build_prompt_conversation(prompt: str):
    return [{"role": "user", "content": prompt}]

def build_question_prompt_dataset(dataset: Dict) -> Dataset:
    question_prompt_dataset = []
    for id, text in dataset.items():
        prompt = _create_question_prompt(text)
        prompt = _build_prompt_conversation(prompt)
        question_prompt_dataset.append({"id": id, "prompt": prompt})
    return Dataset.from_list(question_prompt_dataset)

def generate_questions(model_name_or_path: str, tokenizer_name: str, dataset: Dict, num_question_generations: int, temperature: float = 1.2) -> list:
    llm = LLM(
        model=model_name_or_path,
        tokenizer=tokenizer_name,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=1024,
        n=num_question_generations,
    )

    prompts = []
    ids = []
    for id, text in dataset.items():
        prompts.append(_create_question_prompt(text))
        ids.append(id)

    outputs = llm.generate(prompts, sampling_params)

    questions = []
    for output, id in zip(outputs, ids):
        for completion in output.outputs:
            questions.append({"id": id, "question": completion.text})

    del llm
    torch.cuda.empty_cache()
    return questions

def _build_prompt(question: str) -> str:
    return f"<Question>\n{question}\n\n<Answer>"

def _build_teacher_prompt(question: str, document: str) -> str:
    return (
        "Read the following passage carefully, then answer the question.\n\n"
        f"<Passage>\n{document}\n\n"
    ) + _build_prompt(question)

def build_question_dataset(model_name_or_path: str, tokenizer_name: str, dataset: Dict, num_question_generations: int) -> Dataset:
    question_dataset = []
    questions = generate_questions(model_name_or_path, tokenizer_name, dataset, num_question_generations)
    for row in questions:   
        id, question = row["id"], row["question"]
        prompt = _build_prompt_conversation(_build_prompt(question))
        teacher_prompt = _build_prompt_conversation(_build_teacher_prompt(question, dataset[id]))
        question_dataset.append({"id": id, "prompt": prompt, "teacher_prompt": teacher_prompt, "question": question})
    return Dataset.from_list(question_dataset)

# TODO: add passage
def _build_judge_prompt(question: str, answer: str) -> str:
    return (
        "You are an impartial judge. Decide whether the student's answer is "
        "correct compared to the reference. The wording need not match exactly, "
        "but all key facts must be present and accurate. "
        "Respond with ONLY the single word 'correct' or 'incorrect'."
        "\n\n"
        f"<Question>\n{question}\n\n"
        f"<Student's Answer>\n{answer}\n\n"
        "<Verdict>"
    )

def _normalize_completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    if isinstance(completion, list):
        return " ".join(_normalize_completion_text(item) for item in completion)
    return str(completion)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    assert len(dataset) > 0, "No dataset loaded"

    skip_question_training = args.question_model_path is not None

    if not skip_question_training:
        question_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )

        question_config = GRPOConfig(
            seed=args.seed,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1, 
            vllm_gpu_memory_utilization=0.3,
            vllm_enable_sleep_mode=False,
            learning_rate=args.learning_rate,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            logging_steps = 1,
            bf16 = True,
            fp16 = False,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            num_generations=16,
            max_steps=100,
            max_prompt_length=3072,
            max_completion_length=1024,
            num_train_epochs = 2,
            report_to = "none",
        )

    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    distillation_config = DistilConfig(
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
        report_to = None,
        output_dir = args.output_dir,
        log_completions = True, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = 0.0,
        vllm_importance_sampling_correction = True,
    )

    for i in range(args.num_generation_iterations):
        teacher_model.to("cpu")
        torch.cuda.empty_cache()

        if not skip_question_training:
            question_prompt_dataset = build_question_prompt_dataset(dataset)
            print(f"Question prompt dataset length: {len(question_prompt_dataset)}")
            assert len(question_prompt_dataset) > 0, "No question prompts generated"

            @torch.no_grad()
            # TODO: judge currently is the student model itself. This should be changed to a separate model.
            def reward_question_difficulty(completions, **kwargs) -> float:
                questions = [_normalize_completion_text(c) for c in completions]
                rewards = []
                sub_batch = 4
                for start in range(0, len(questions), sub_batch):
                    batch_qs = questions[start:start + sub_batch]

                    student_inputs = tokenizer(
                        [_build_prompt(q) for q in batch_qs],
                        padding=True, truncation=True, max_length=1024, return_tensors="pt", padding_side="left",
                    ).to(student_model.device)
                    student_out = student_model.generate(**student_inputs, max_new_tokens=256)
                    student_answers = tokenizer.batch_decode(
                        student_out[:, student_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    )

                    judge_inputs = tokenizer(
                        [_build_judge_prompt(q, a) for q, a in zip(batch_qs, student_answers)],
                        padding=True, truncation=True, max_length=2048, return_tensors="pt", padding_side="left",
                    ).to(student_model.device)
                    judge_out = student_model.generate(**judge_inputs, max_new_tokens=16)
                    verdicts = tokenizer.batch_decode(
                        judge_out[:, judge_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                    )

                    rewards.extend(
                        -1.0 if v.strip().lower().startswith("correct") else 0.0
                        for v in verdicts
                    )
                return rewards

            question_trainer = GRPOTrainer(
                model=question_model,
                args=question_config,
                train_dataset=question_prompt_dataset,
                reward_funcs=reward_question_difficulty,
            )
            question_trainer.train()
            question_model_dir = os.path.join(args.output_dir, f"question_model_{i}")
            question_model.save_pretrained(question_model_dir)
            tokenizer.save_pretrained(question_model_dir)
            print(f"Saved question model to: {question_model_dir}")
            question_gen_path = question_model_dir
            question_model.to("cpu")
        else:
            question_gen_path = args.question_model_path

        student_model.to("cpu")
        torch.cuda.empty_cache()

        print(f"Building question dataset using vLLM from: {question_gen_path}")
        question_dataset = build_question_dataset(question_gen_path, args.model_name, dataset, args.num_question_generations)
        print(f"Question dataset length: {len(question_dataset)}")
        assert len(question_dataset) > 0, "No questions generated"

        student_model.to("cuda")

        student_trainer = DistilTrainer(
            model=student_model,
            ref_model=teacher_model,
            args=distillation_config,
            train_dataset=question_dataset,
        )
        student_trainer.train()
        student_model_dir = os.path.join(args.output_dir, f"student_model_{i}")
        student_model.save_pretrained(student_model_dir)
        print(f"Saved student model to: {student_model_dir}")

    final_student_model_dir = os.path.join(args.output_dir, "final_student_model")
    student_model.save_pretrained(final_student_model_dir)
    print(f"Saved final student model to: {final_student_model_dir}")