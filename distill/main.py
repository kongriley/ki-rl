import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import re
import sys
from typing import Dict, List, Tuple
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from trl import GRPOConfig, GRPOTrainer

from distil_config import DistilConfig
from distil_trainer import DistilTrainer

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
    p.add_argument("--report_student_performance", action=argparse.BooleanOptionalAction, default=True,
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
            num_generations=args.num_grpo_generations,
            max_prompt_length=3072,
            max_completion_length=1024,
            num_train_epochs = args.num_question_model_train_epochs,
            report_to = "none",
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )
        if args.debug:
            question_config.max_steps = 1

    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
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
        report_to = "none",
        output_dir = args.output_dir,
        log_completions = args.log_student_completions and not args.debug,
        sync_ref_model = False,
        vllm_importance_sampling_correction = True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )
    if args.debug:
        distillation_config.max_steps = 1

    for i in range(args.num_generation_iterations):
        # ── Phase 1: Question model training (GRPO) ──
        # Student + judge stay on CPU except during the reward function.
        # The reward fn moves them to GPU for inference, then back to CPU
        # before returning so the backward pass has enough memory.
        if not (i == 0 and args.skip_first_iteration) and not skip_question_training:
            question_prompt_dataset = build_question_prompt_dataset(dataset)
            print(f"Question prompt dataset length: {len(question_prompt_dataset)}")
            assert len(question_prompt_dataset) > 0, "No question prompts generated"

            _reward_call_count = [0]

            FORMAT_PENALTY = -2.0
            PRESIDIO_PENALTY = -1.5
            EASY_PENALTY = -1.0
            GARBAGE_PENALTY = -0.5
            GOOD_REWARD = 1.0
            LENGTH_PENALTY = -0.3
            MAX_QUESTION_LENGTH = 300

            def _swap_to_gpu(model):
                torch.cuda.empty_cache()
                model.to("cuda")

            def _swap_to_cpu(model):
                model.to("cpu")
                torch.cuda.empty_cache()

            @torch.no_grad()
            def reward_question_difficulty(completions, **kwargs) -> float:
                _reward_call_count[0] += 1
                raw_texts = [_normalize_completion_text(c) for c in completions]
                parsed = [_parse_question_answer(t) for t in raw_texts]
                questions = [q for q, _ in parsed]
                ref_answers = [a for _, a in parsed]

                passage_ids = kwargs.get("id", [None] * len(questions))

                valid_mask = [_is_valid_qa(q, a) for q, a in zip(questions, ref_answers)]
                valid_indices = [i for i, v in enumerate(valid_mask) if v]
                valid_qs = [questions[i] for i in valid_indices]
                valid_refs = [ref_answers[i] for i in valid_indices]
                valid_ids = [passage_ids[i] for i in valid_indices]

                rewards = [FORMAT_PENALTY] * len(questions)
                all_closed_answers = [""] * len(questions)
                all_open_answers = [""] * len(questions)
                all_closed_verdicts = ["[format_penalty]"] * len(questions)
                all_open_verdicts = ["[format_penalty]"] * len(questions)
                all_reasons = ["format_penalty"] * len(questions)

                if valid_qs:
                    presidio_mask = [
                        "PRESIDIO_ANONYMIZED" in q or "PRESIDIO_ANONYMIZED" in a
                        for q, a in zip(valid_qs, valid_refs)
                    ]
                    clean_indices = [j for j, is_p in enumerate(presidio_mask) if not is_p]
                    clean_qs = [valid_qs[j] for j in clean_indices]
                    clean_refs = [valid_refs[j] for j in clean_indices]
                    clean_ids = [valid_ids[j] for j in clean_indices]

                    for j, is_p in enumerate(presidio_mask):
                        if is_p:
                            idx = valid_indices[j]
                            rewards[idx] = PRESIDIO_PENALTY
                            all_reasons[idx] = "presidio"

                    if clean_qs:
                        _swap_to_gpu(student_model)
                        closed_answers = batch_generate_answers(
                            student_model, tokenizer, clean_qs, batch_size=4,
                        )
                        _swap_to_cpu(student_model)

                        _swap_to_gpu(judge_model)
                        closed_verdicts = batch_judge_answers(
                            judge_model, tokenizer, clean_qs, clean_refs,
                            closed_answers, batch_size=4,
                        )
                        _swap_to_cpu(judge_model)

                        needs_open = [k for k, (correct, _) in enumerate(closed_verdicts) if not correct]
                        open_qs = [clean_qs[k] for k in needs_open]
                        open_refs = [clean_refs[k] for k in needs_open]
                        open_docs = [dataset[clean_ids[k]] for k in needs_open]

                        if open_qs:
                            _swap_to_gpu(student_model)
                            open_answers_subset = batch_generate_with_context(
                                student_model, tokenizer, open_qs, open_docs, batch_size=4,
                            )
                            _swap_to_cpu(student_model)

                            _swap_to_gpu(judge_model)
                            open_verdicts_subset = batch_judge_answers(
                                judge_model, tokenizer, open_qs, open_refs,
                                open_answers_subset, batch_size=4,
                            )
                            _swap_to_cpu(judge_model)
                        else:
                            open_answers_subset = []
                            open_verdicts_subset = []

                        open_answers_full = [""] * len(clean_qs)
                        open_verdicts_full = [("", "[skipped_easy]")] * len(clean_qs)
                        for sub_k, orig_k in enumerate(needs_open):
                            open_answers_full[orig_k] = open_answers_subset[sub_k]
                            open_verdicts_full[orig_k] = open_verdicts_subset[sub_k]

                        for k, j in enumerate(clean_indices):
                            idx = valid_indices[j]
                            closed_correct, closed_text = closed_verdicts[k]
                            open_correct, open_text = open_verdicts_full[k]

                            all_closed_answers[idx] = closed_answers[k]
                            all_open_answers[idx] = open_answers_full[k]
                            all_closed_verdicts[idx] = closed_text
                            all_open_verdicts[idx] = open_text

                            if closed_correct:
                                rewards[idx] = EASY_PENALTY
                                all_reasons[idx] = "too_easy"
                            elif open_correct:
                                rewards[idx] = GOOD_REWARD
                                all_reasons[idx] = "good_question"
                            else:
                                rewards[idx] = GARBAGE_PENALTY
                                all_reasons[idx] = "garbage"

                            if len(questions[idx]) > MAX_QUESTION_LENGTH:
                                rewards[idx] += LENGTH_PENALTY
                                all_reasons[idx] += "+long"

                n_format_bad = sum(1 for v in valid_mask if not v)
                n_samples = min(3, len(questions))
                print(f"\n{'='*60}")
                print(f"[Reward call #{_reward_call_count[0]}] Showing {n_samples}/{len(questions)} completions ({n_format_bad} format penalties):")
                for j in range(n_samples):
                    tag = "[BAD FORMAT]" if not valid_mask[j] else ""
                    print(f"  --- Sample {j+1} {tag} ---")
                    print(f"  Q: {questions[j][:200]}")
                    print(f"  Ref A: {ref_answers[j][:200]}")
                    print(f"  Closed-book A: {all_closed_answers[j][:200]}")
                    print(f"  Closed verdict: {all_closed_verdicts[j]}")
                    print(f"  Open-book A: {all_open_answers[j][:200]}")
                    print(f"  Open verdict: {all_open_verdicts[j]}")
                    print(f"  Reason: {all_reasons[j]} -> reward={rewards[j]}")
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
            if args.save_question_model:
                question_model_dir = os.path.join(args.output_dir, f"question_model_{i}")
                question_model.save_pretrained(question_model_dir)
                tokenizer.save_pretrained(question_model_dir)
                print(f"Saved question model to: {question_model_dir}")

            # Clean up the GRPOTrainer and its colocated vLLM before next phase.
            # The pluggable allocator / CUDA graphs from vLLM colocate may not
            # fully release, so we free everything we can and let the subprocess
            # query actual free memory to set gpu_memory_utilization dynamically.
            question_model.zero_grad(set_to_none=True)
            del question_trainer
            question_model.to("cpu")
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if skip_question_training:
            question_gen_path = args.question_model_path
        elif args.save_question_model:
            question_gen_path = os.path.join(args.output_dir, f"question_model_{i}")
        else:
            question_gen_path = os.path.join(args.output_dir, f"_tmp_question_model_{i}")
            question_model.save_pretrained(question_gen_path)
            tokenizer.save_pretrained(question_gen_path)

        # ── Phase 2: Generate question dataset (standalone vLLM) ──
        # The standalone vLLM in generate_questions needs most of the GPU.
        # Move all models off GPU; teacher_model may be on CUDA if the
        # DistilTrainer moved it there as ref_model in a prior iteration.
        student_model.to("cpu")
        teacher_model.to("cpu")
        judge_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        print(f"Building question dataset using vLLM from: {question_gen_path}")
        question_dataset = build_question_dataset(question_gen_path, args.model_name, dataset, args.num_question_generations, args.num_questions_per_generation)
        print(f"Question dataset length: {len(question_dataset)}")
        assert len(question_dataset) > 0, "No questions generated"

        # ── Phase 3: Distillation (DistilTrainer) ──
        # Only student goes back to GPU; judge is not used here and would
        # compete with vLLM + ref_model for memory, causing OOM on wake_up().
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
        tokenizer.save_pretrained(student_model_dir)
        print(f"Saved student model to: {student_model_dir}")

        # Clean up the DistilTrainer and its colocated vLLM before next iteration
        student_model.zero_grad(set_to_none=True)
        del student_trainer
        teacher_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if args.report_student_performance and not args.debug:
            print("Reporting student performance on question dataset from this iteration...")

            # Reuse the question_dataset already generated in Phase 2 — no need
            # to spin up another vLLM subprocess.
            eval_question_dataset = question_dataset
            print(f"Evaluation question dataset length: {len(eval_question_dataset)}")

            # Move off GPU anything that's there (DistilTrainer may have placed
            # teacher_model on CUDA as ref_model), then bring student + judge up.
            teacher_model.to("cpu")
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            student_model.gradient_checkpointing_disable()
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

        student_model.zero_grad(set_to_none=True)
        student_model.to("cpu")
        teacher_model.to("cpu")
        judge_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    final_student_model_dir = os.path.join(args.output_dir, "final_student_model")
    student_model.save_pretrained(final_student_model_dir)
    print(f"Saved final student model to: {final_student_model_dir}")
