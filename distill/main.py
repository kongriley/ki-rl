import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from inference import (
    build_student_prompt,
    build_teacher_prompt,
    build_judge_prompt,
    parse_verdict,
    batch_generate_answers,
    batch_generate_with_context,
    batch_judge_answers,
    evaluate_qa,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_generation_iterations", type=int, default=5, help="Number of times to run the generation loop")
    p.add_argument("--num_question_generations", type=int, default=10, help="Number of questions to generate for each passage. The total number of questions generated will be num_question_generations * len(dataset).")
    p.add_argument("--num_questions_per_generation", type=int, default=5, help="Number of questions to request per LLM completion. This does not affect the total number of questions generated.")
    p.add_argument("--dataset_path", default="./data/wiki_20/data.json")
    p.add_argument("--model_name", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--output_dir", default="./distill/out/grpo_distill")
    p.add_argument("--question_model_path", type=str, default=None,
                   help="Path to a pre-trained question model. Skips question generator training and uses this model directly for question generation.")
    p.add_argument("--skip_first_iteration", action="store_true", default=False,
                   help="Skip the first iteration of the question generation loop.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_question_model_train_epochs", type=float, default=1)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--num_grpo_generations", type=int, default=8,
                   help="Number of completions sampled per prompt during GRPO. Lower values reduce reward-function inference cost.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--report_student_performance", action=argparse.BooleanOptionalAction, default=False,
                   help="Report student performance on the question dataset after distillation.")
    p.add_argument("--log_student_completions", action="store_true", default=False)
    p.add_argument("--save_question_model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--debug", action="store_true", default=False, help="Changes max steps to 1 and does not report student performance for debugging")
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
    This question will be used in a separate examination in two weeks, where the students are not given the passage.

    Each question must be fully self-contained and understandable on its own, without needing the passage for context. Include specific names, dates, and topics directly in the question so a reader can understand exactly what is being asked. It should also not contain extraneous information that is not in the passage.
    - Bad: "Who was appointed after the resignation?" (unclear who or what)
    - Good: "Who was appointed CEO of OpenAI after Sam Altman's brief resignation in November 2023?" (self-contained)

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

    free_mem, total_mem = torch.cuda.mem_get_info()
    free_frac = free_mem / total_mem
    target_utilization = min(0.6, free_frac - 0.05)
    target_utilization = max(target_utilization, 0.2)
    print(f"  vLLM question gen: {free_mem/1e9:.1f}/{total_mem/1e9:.1f} GiB free "
          f"({free_frac:.1%}), using gpu_memory_utilization={target_utilization:.2f}")
    llm_kwargs = dict(
        model=model_name_or_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=target_utilization,
    )
    if tokenizer_name != model_name_or_path:
        llm_kwargs["tokenizer"] = tokenizer_name
    llm = LLM(**llm_kwargs)
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
    return build_student_prompt(question)

def _build_teacher_prompt(question: str, document: str) -> str:
    return build_teacher_prompt(question, document)

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
    return build_judge_prompt(question, reference_answer, student_answer)

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

    # Only load models needed in the main process (eval).
    # GRPO and distillation each run in their own subprocess with a fresh
    # CUDA context, so they load models independently.
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(args.num_generation_iterations):
        # ── Phase 1: Question model training (GRPO subprocess) ──
        # Runs in a subprocess so each iteration gets a clean CUDA context
        # (vLLM colocated MemPool is a per-process singleton).
        if not (i == 0 and args.skip_first_iteration) and not skip_question_training:
            question_model_input = (
                args.model_name if i == 0
                else os.path.join(args.output_dir, f"question_model_{i - 1}")
            )
            student_input = (
                args.model_name if i == 0
                else os.path.join(args.output_dir, f"student_model_{i - 1}")
            )
            question_model_output = os.path.join(args.output_dir, f"question_model_{i}")

            grpo_kw = dict(
                seed=args.seed,
                use_vllm=True,
                vllm_mode="colocate",
                vllm_tensor_parallel_size=1,
                vllm_gpu_memory_utilization=0.3,
                vllm_enable_sleep_mode=True,
                learning_rate=args.learning_rate,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                logging_steps=1,
                bf16=True,
                fp16=False,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_generations=args.num_grpo_generations,
                max_prompt_length=3072,
                max_completion_length=1024,
                num_train_epochs=args.num_question_model_train_epochs,
                report_to="none",
                gradient_checkpointing=True,
                optim="adamw_8bit",
                output_dir=question_model_output,
            )
            if args.debug:
                grpo_kw["max_steps"] = 1

            grpo_manifest = {
                "question_model_path": question_model_input,
                "student_model_path": student_input,
                "judge_model_path": args.model_name,
                "tokenizer_name": args.model_name,
                "dataset_path": args.dataset_path,
                "output_dir": question_model_output,
                "grpo_config": grpo_kw,
            }
            grpo_manifest_path = os.path.join(args.output_dir, f"_grpo_manifest_{i}.json")
            with open(grpo_manifest_path, "w") as f:
                json.dump(grpo_manifest, f, indent=2)

            grpo_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_phase_worker.py")
            print(f"Running GRPO training in subprocess ({grpo_worker})...")
            subprocess.run([sys.executable, grpo_worker, grpo_manifest_path], check=True)
            print(f"Question model saved to: {question_model_output}")

        if skip_question_training:
            question_gen_path = args.question_model_path
        else:
            question_gen_path = os.path.join(args.output_dir, f"question_model_{i}")

        # ── Phase 2: Generate question dataset (standalone vLLM) ──
        # Move eval models off GPU before spawning the subprocess.
        student_model.to("cpu")
        judge_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        print(f"Building question dataset using vLLM from: {question_gen_path}")
        question_dataset = build_question_dataset(question_gen_path, args.model_name, dataset, args.num_question_generations, args.num_questions_per_generation)
        print(f"Question dataset length: {len(question_dataset)}")
        assert len(question_dataset) > 0, "No questions generated"

        # ── Phase 3: Distillation (subprocess for clean CUDA context) ──
        # vLLM colocated mode leaks CUDA memory that empty_cache() can't
        # reclaim.  Running distillation in a fresh process prevents those
        # leaks from accumulating across iterations and causing OOM.
        student_model_dir = os.path.join(args.output_dir, f"student_model_{i}")

        # The student weights on disk: iteration 0 uses the base model;
        # later iterations use the previous iteration's saved checkpoint.
        if i == 0:
            student_input_path = args.model_name
        else:
            student_input_path = os.path.join(args.output_dir, f"student_model_{i - 1}")

        dataset_path = os.path.join(args.output_dir, f"_questions_{i}.jsonl")
        question_dataset.to_json(dataset_path)

        distil_kw = dict(
            seed=args.seed,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.3,
            vllm_enable_sleep_mode=True,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=1,
            bf16=True,
            fp16=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_prompt_length=1024,
            max_completion_length=1024,
            num_train_epochs=args.num_train_epochs,
            save_steps=100,
            max_grad_norm=1,
            report_to="none",
            output_dir=student_model_dir,
            log_completions=args.log_student_completions and not args.debug,
            sync_ref_model=False,
            vllm_importance_sampling_correction=True,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )
        if args.debug:
            distil_kw["max_steps"] = 1

        manifest = {
            "student_model_path": student_input_path,
            "teacher_model_path": args.model_name,
            "tokenizer_name": args.model_name,
            "dataset_path": dataset_path,
            "output_dir": student_model_dir,
            "distil_config": distil_kw,
        }
        manifest_path = os.path.join(args.output_dir, f"_distill_manifest_{i}.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill_phase_worker.py")
        print(f"Running student distillation in subprocess ({worker_py})...")
        subprocess.run([sys.executable, worker_py, manifest_path], check=True)

        # Reload the trained student from the subprocess output
        del student_model
        gc.collect()
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_dir, torch_dtype=torch.bfloat16,
        )
        print(f"Student model saved to: {student_model_dir}")

        if args.report_student_performance and not args.debug:
            print("Reporting student performance on question dataset from this iteration...")

            eval_question_dataset = question_dataset
            print(f"Evaluation question dataset length: {len(eval_question_dataset)}")

            student_model.eval()
            student_model.to("cuda")
            judge_model.to("cuda")

            eval_summary = evaluate_qa(
                student_model, judge_model, tokenizer,
                eval_question_dataset["question"],
                eval_question_dataset["answer"],
                ids=eval_question_dataset["id"],
            )
            print(f"[Iteration {i}] Student accuracy: "
                  f"{eval_summary['correct']}/{eval_summary['total']} "
                  f"({eval_summary['accuracy']:.1%})")

            results_path = os.path.join(args.output_dir, f"results_student_model_{i}.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"iteration": i, **eval_summary}, f, indent=2)
            print(f"Saved evaluation results to: {results_path}")

        student_model.to("cpu")
        judge_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    final_student_model_dir = os.path.join(args.output_dir, "final_student_model")
    student_model.save_pretrained(final_student_model_dir)
    print(f"Saved final student model to: {final_student_model_dir}")
