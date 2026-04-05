import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import re
from typing import Dict, List, Tuple
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
    p.add_argument("--num_questions_per_generation", type=int, default=5, help="Number of questions to request per LLM completion")
    p.add_argument("--dataset_path", default="./data/wiki_20/data.json")
    p.add_argument("--model_name", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--output_dir", default="./distill/out/grpo_distill")
    p.add_argument("--question_model_path", type=str, default=None,
                   help="Path to a pre-trained question model. Skips question generator training and uses this model directly for question generation.")
    p.add_argument("--skip_first_iteration", action="store_true", default=False,
                   help="Skip the first iteration of the question generation loop.")
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

def _create_question_prompt(text: str, num_questions: int = 1) -> str:
    return f"""
    Using the following passage, generate {num_questions} question{'' if num_questions == 1 else 's'} about the passage, along with their answers.
    This question will be used in a separate examination in two weeks, where the students are not given the passage. Therefore, be clear about the context, but do not explicitly reference the passage in the question.
    The answers should be answerable solely based on the information in the passage. However, if the student has never seen the passage before, the answer should not be answerable (i.e. it should not contain extraneous information that is not in the passage).

    Format your response as:
    Question 1: <your question>
    Answer 1: <the answer>
    ...
    Question {num_questions}: <your question>
    Answer {num_questions}: <the answer>

    <Passage>
    {text}

    <Response>
    """

def _build_prompt_conversation(prompt: str):
    return [{"role": "user", "content": prompt}]

def _parse_question_answer(text: str) -> Tuple[str, str]:
    """Extract a single (question, answer) from generated text with 'Question [N]:' / 'Answer [N]:' prefixes."""
    q_match = re.search(r"Question\s*\d*:\s*(.+?)(?=\n\s*Answer\s*\d*:|\Z)", text, re.DOTALL)
    a_match = re.search(r"Answer\s*\d*:\s*(.+?)(?=\n\s*Question\s*\d*:|\Z)", text, re.DOTALL)
    question = q_match.group(1).strip() if q_match else text.strip()
    answer = a_match.group(1).strip() if a_match else ""
    return question, answer

def _parse_question_answers(text: str) -> List[Tuple[str, str]]:
    """Extract all (question, answer) pairs from text with numbered 'Question N:' / 'Answer N:' format."""
    pattern = r"Question\s*\d*:\s*(.+?)\s*Answer\s*\d*:\s*(.+?)(?=Question\s*\d*:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [(q.strip(), a.strip()) for q, a in matches]
    return [_parse_question_answer(text)]

def build_question_prompt_dataset(dataset: Dict, num_questions_per_generation: int = 1) -> Dataset:
    question_prompt_dataset = []
    for id, text in dataset.items():
        prompt = _create_question_prompt(text, num_questions=num_questions_per_generation)
        prompt = _build_prompt_conversation(prompt)
        question_prompt_dataset.append({"id": id, "prompt": prompt})
    return Dataset.from_list(question_prompt_dataset)

def _is_valid_qa(question: str, answer: str) -> bool:
    return len(question) >= 10 and len(answer) >= 1

def generate_questions(model_name_or_path: str, tokenizer_name: str, dataset: Dict, num_question_generations: int, num_questions_per_generation: int = 5, temperature: float = 1.2, max_retries: int = 3) -> list:
    n = max(1, math.ceil(num_question_generations / num_questions_per_generation))

    llm = LLM(
        model=model_name_or_path,
        tokenizer=tokenizer_name,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.6,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=1024,
        n=n,
    )

    all_ids = list(dataset.keys())
    all_prompts = [_create_question_prompt(dataset[id], num_questions=num_questions_per_generation) for id in all_ids]

    questions_by_id: Dict[str, list] = {id: [] for id in all_ids}

    for attempt in range(max_retries):
        if attempt == 0:
            pending_ids = all_ids
            pending_prompts = all_prompts
        else:
            pending_ids = [id for id in all_ids if len(questions_by_id[id]) < num_question_generations]
            if not pending_ids:
                break
            pending_prompts = [_create_question_prompt(dataset[id], num_questions=num_questions_per_generation) for id in pending_ids]
            print(f"  Retry {attempt}/{max_retries}: regenerating for {len(pending_ids)} passages with insufficient questions")

        outputs = llm.generate(pending_prompts, sampling_params)

        for output, id in zip(outputs, pending_ids):
            for completion in output.outputs:
                for question, answer in _parse_question_answers(completion.text):
                    if _is_valid_qa(question, answer):
                        questions_by_id[id].append({"id": id, "question": question, "answer": answer})

    questions = []
    for id in all_ids:
        passage_qs = questions_by_id[id][:num_question_generations]
        if len(passage_qs) < num_question_generations:
            print(f"  Warning: passage {id} only got {len(passage_qs)}/{num_question_generations} valid questions after {max_retries} attempts")
        questions.extend(passage_qs)

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

def _run_generate_questions(args_and_queue):
    """Subprocess target: generates questions in a fresh CUDA context."""
    (model_name_or_path, tokenizer_name, dataset, num_question_generations,
     num_questions_per_generation, temperature, queue) = args_and_queue
    results = generate_questions(
        model_name_or_path, tokenizer_name, dataset,
        num_question_generations, num_questions_per_generation, temperature,
    )
    queue.put(results)

def build_question_dataset(model_name_or_path: str, tokenizer_name: str, dataset: Dict, num_question_generations: int, num_questions_per_generation: int = 5) -> Dataset:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_run_generate_questions,
        args=((model_name_or_path, tokenizer_name, dataset,
               num_question_generations, num_questions_per_generation, 1.2, queue),),
    )
    p.start()
    questions = queue.get()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Question generation subprocess exited with code {p.exitcode}")

    question_dataset = []
    for row in questions:
        id, question, answer = row["id"], row["question"], row["answer"]
        prompt = _build_prompt_conversation(_build_prompt(question))
        teacher_prompt = _build_prompt_conversation(_build_teacher_prompt(question, dataset[id]))
        question_dataset.append({"id": id, "prompt": prompt, "teacher_prompt": teacher_prompt, "question": question, "answer": answer})
    return Dataset.from_list(question_dataset)

def _build_judge_prompt(question: str, reference_answer: str, student_answer: str) -> str:
    return (
        "You are an impartial judge. Decide whether the student's answer agrees with "
        "the reference answer. The wording need not match exactly, "
        "but all key facts must be present and accurate. "
        "Respond with ONLY the single word 'correct' or 'incorrect'."
        "\n\n"
        f"<Question>\n{question}\n\n"
        f"<Reference Answer>\n{reference_answer}\n\n"
        f"<Student's Answer>\n{student_answer}\n\n"
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
            max_prompt_length=3072,
            max_completion_length=1024,
            num_train_epochs = 2,
            report_to = "none",
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    judge_model = student_model
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
        log_completions = True,
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = 0.0,
        vllm_importance_sampling_correction = True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )

    for i in range(args.num_generation_iterations):
        # ── Phase 1: Question model training (GRPO) ──
        # Student stays on GPU for reward fn inference. The GRPOTrainer's
        # colocated vLLM sleeps after generation, freeing ~60 GB before
        # the reward function runs, so there's plenty of room.
        if not (i == 0 and args.skip_first_iteration) and not skip_question_training:
            question_prompt_dataset = build_question_prompt_dataset(dataset)
            print(f"Question prompt dataset length: {len(question_prompt_dataset)}")
            assert len(question_prompt_dataset) > 0, "No question prompts generated"

            _reward_call_count = [0]

            FORMAT_PENALTY = -2.0

            @torch.no_grad()
            def reward_question_difficulty(completions, **kwargs) -> float:
                _reward_call_count[0] += 1
                raw_texts = [_normalize_completion_text(c) for c in completions]
                parsed = [_parse_question_answer(t) for t in raw_texts]
                questions = [q for q, _ in parsed]
                ref_answers = [a for _, a in parsed]

                valid_mask = [_is_valid_qa(q, a) for q, a in zip(questions, ref_answers)]
                valid_indices = [i for i, v in enumerate(valid_mask) if v]
                valid_qs = [questions[i] for i in valid_indices]
                valid_refs = [ref_answers[i] for i in valid_indices]

                rewards = [FORMAT_PENALTY] * len(questions)
                all_student_answers = [""] * len(questions)
                all_verdicts = ["[format_penalty]"] * len(questions)

                if valid_qs:
                    sub_batch = 4
                    batch_rewards = []
                    batch_student_answers = []
                    batch_verdicts = []
                    for start in range(0, len(valid_qs), sub_batch):
                        batch_qs = valid_qs[start:start + sub_batch]
                        batch_refs = valid_refs[start:start + sub_batch]

                        student_inputs = tokenizer(
                            [_build_prompt(q) for q in batch_qs],
                            padding=True, truncation=True, max_length=1024, return_tensors="pt", padding_side="left",
                        ).to(student_model.device)
                        student_out = student_model.generate(**student_inputs, max_new_tokens=256)
                        student_answers = tokenizer.batch_decode(
                            student_out[:, student_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                        )

                        judge_inputs = tokenizer(
                            [_build_judge_prompt(q, ref, sa) for q, ref, sa in zip(batch_qs, batch_refs, student_answers)],
                            padding=True, truncation=True, max_length=2048, return_tensors="pt", padding_side="left",
                        ).to(judge_model.device)
                        judge_out = judge_model.generate(**judge_inputs, max_new_tokens=16)
                        verdicts = tokenizer.batch_decode(
                            judge_out[:, judge_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                        )

                        batch_student_answers.extend(student_answers)
                        batch_verdicts.extend(verdicts)
                        batch_rewards.extend(
                            -1.0 if v.strip().lower().startswith("correct") else 0.0
                            for v in verdicts
                        )

                    for j, idx in enumerate(valid_indices):
                        rewards[idx] = batch_rewards[j]
                        all_student_answers[idx] = batch_student_answers[j]
                        all_verdicts[idx] = batch_verdicts[j]

                n_format_bad = sum(1 for v in valid_mask if not v)
                n_samples = min(3, len(questions))
                print(f"\n{'='*60}")
                print(f"[Reward call #{_reward_call_count[0]}] Showing {n_samples}/{len(questions)} completions ({n_format_bad} format penalties):")
                for j in range(n_samples):
                    print(f"  --- Sample {j+1} {'[BAD FORMAT]' if not valid_mask[j] else ''} ---")
                    print(f"  Q: {questions[j][:200]}")
                    print(f"  Ref A: {ref_answers[j][:200]}")
                    print(f"  Student A: {all_student_answers[j][:200]}")
                    print(f"  Verdict: {all_verdicts[j].strip()} -> reward={rewards[j]}")
                print(f"  Mean reward: {sum(rewards)/len(rewards):.3f}")
                print(f"{'='*60}\n")

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

            # Clean up the GRPOTrainer and its colocated vLLM before next phase
            del question_trainer
            question_model.to("cpu")
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            question_gen_path = args.question_model_path if args.question_model_path else model_name

        # ── Phase 2: Generate question dataset (standalone vLLM) ──
        # The standalone vLLM in generate_questions needs most of the GPU.
        # Move models off GPU for this phase only.
        student_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        print(f"Building question dataset using vLLM from: {question_gen_path}")
        question_dataset = build_question_dataset(question_gen_path, args.model_name, dataset, args.num_question_generations, args.num_questions_per_generation)
        print(f"Question dataset length: {len(question_dataset)}")
        assert len(question_dataset) > 0, "No questions generated"

        # ── Phase 3: Distillation (DistilTrainer) ──
        # Move student back to GPU; teacher stays on CPU and the trainer's
        # accelerator.prepare_model handles placing it for forward passes.
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

        # Clean up the DistilTrainer and its colocated vLLM before next iteration
        del student_trainer
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    final_student_model_dir = os.path.join(args.output_dir, "final_student_model")
    student_model.save_pretrained(final_student_model_dir)
    print(f"Saved final student model to: {final_student_model_dir}")
