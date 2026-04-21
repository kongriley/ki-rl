import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
from typing import Dict
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from eval_questions import (
    _openai_client,
    judge_openai_batch,
    load_questions as load_questions_jsonl,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from generate_questions import (
    load_dataset_json,
    build_question_prompt_dataset,
    build_question_dataset,
    build_question_dataset_for_deficits,
    generate_questions,
    generate_questions_openai,
    _create_question_prompt,
    _build_prompt_conversation,
    _parse_question_answers,
    _is_valid_qa,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_generation_iterations", type=int, default=10, help="Number of times to run the generation loop")
    p.add_argument("--num_question_generations", type=int, default=10, help="Number of questions to generate for each passage. The total number of questions generated will be num_question_generations * len(dataset).")
    p.add_argument("--num_questions_per_generation", type=int, default=8, help="Number of questions to request per LLM completion. This does not affect the total number of questions generated.")
    p.add_argument("--dataset_path", default="./data/wiki_20/data.json")
    p.add_argument("--model_name", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--output_dir", default="./distill/out/grpo_distill")
    p.add_argument("--question_model_path", type=str, default=None,
                   help="Path to a pre-trained question model. Skips question generator training and uses this model directly for question generation.")
    p.add_argument("--skip_generator_training", action="store_true", default=False,
                   help="Skip GRPO question-generator training and use --model_name as the question generator.")
    p.add_argument("--skip_first_iteration", action="store_true", default=False,
                   help="Skip the first iteration of the question generation loop.")
    p.add_argument("--accumulate_questions", action="store_true", default=False,
                   help="In Phase 2, append the previous iteration's generated questions for each "
                        "passage to the question-generation prompt and instruct the model to produce "
                        "different ones. Has no effect in the first iteration.")
    p.add_argument("--num_accumulated_questions", type=int, default=None,
                   help="When --accumulate_questions is set, cap on how many of the previous "
                        "iteration's questions to include per passage in the prompt. "
                        "If unset, include all of the previous iteration's questions for the passage.")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_question_model_train_epochs", type=float, default=1)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--num_grpo_generations", type=int, default=4,
                   help="Number of completions sampled per prompt during GRPO. Lower values reduce reward-function inference cost.")
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--report_student_performance", action=argparse.BooleanOptionalAction, default=False,
                   help="Report student performance on the question dataset after distillation.")
    p.add_argument("--eval_questions_path", type=str, default=None,
                   help="Path to pre-generated questions JSONL for evaluation. "
                        "When set, uses these fixed questions instead of the iteration's generated questions.")
    p.add_argument("--eval_question_model", type=str, default=None,
                   help="Model name for on-the-fly eval question generation (e.g. gpt-5-mini). "
                        "Mutually exclusive with --eval_questions_path.")
    p.add_argument("--eval_question_backend", choices=["hf", "openai"], default="openai",
                   help="Backend for eval question generation (default: openai).")
    p.add_argument("--eval_num_questions", type=int, default=3,
                   help="Number of questions per passage when generating eval questions on-the-fly.")
    p.add_argument("--eval_judge_model", type=str, default=None,
                   help="Judge model name for evaluation (e.g. gpt-4o-mini). "
                        "When None, uses the local HF judge model.")
    p.add_argument("--eval_judge_backend", choices=["hf", "openai"], default="hf",
                   help="Backend for the evaluation judge. "
                        "When 'openai', skips loading the local judge model entirely.")
    p.add_argument("--log_student_completions", action="store_true", default=False)
    p.add_argument("--vllm_importance_sampling_correction", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_student_model", action=argparse.BooleanOptionalAction, default=False,
                   help="Persist student model checkpoints and the final model to the output directory.")
    p.add_argument("--save_question_model", action=argparse.BooleanOptionalAction, default=False,
                   help="Persist question model checkpoints to the output directory.")
    p.add_argument("--save_student_result_copy_dir", type=str, default=None,
                   help="Directory to copy the result JSONs to. If not set, the results will not be copied.")
    p.add_argument("--save_student_result_copy_name", type=str, default=None, help="Name of the result JSONs to copy, which will be inserted as results_(name)_(iteration_number).json. If not set, the name will default to student_model.")
    p.add_argument("--debug", action="store_true", default=False, help="Changes max steps to 1 for debugging")
    return p.parse_args()


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

    dataset = load_dataset_json(args.dataset_path)
    assert len(dataset) > 0, "No dataset loaded"

    skip_question_training = (
        args.question_model_path is not None or args.skip_generator_training
    )

    if args.eval_questions_path and args.eval_question_model:
        raise SystemExit("Specify --eval_questions_path or --eval_question_model, not both.")
    if args.question_model_path and args.skip_generator_training:
        raise SystemExit("Specify --question_model_path or --skip_generator_training, not both.")

    need_local_judge = (
        args.report_student_performance and args.eval_judge_backend == "hf"
    )

    # Only load models needed in the main process (eval).
    # GRPO and distillation each run in their own subprocess with a fresh
    # CUDA context, so they load models independently.
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    if need_local_judge:
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        judge_model = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prev_generated_by_id: Dict[str, list] = {}

    for i in range(args.num_generation_iterations):
        all_good_questions = []
        # ── Phase 1: Question model training (GRPO subprocess) ──
        # Runs in a subprocess so each iteration gets a clean CUDA context
        # (vLLM colocated MemPool is a per-process singleton).
        grpo_ran = False
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

            num_prompts = len(dataset)
            grpo_grad_accum = min(args.gradient_accumulation_steps, num_prompts)
            grpo_grad_accum = max(grpo_grad_accum, args.num_grpo_generations)
            total_samples = int(num_prompts * args.num_grpo_generations * args.num_question_model_train_epochs)
            computed_max_steps = max(1, math.ceil(total_samples / grpo_grad_accum))

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
                gradient_accumulation_steps=grpo_grad_accum,
                num_generations=args.num_grpo_generations,
                max_prompt_length=3072,
                max_completion_length=args.max_completion_length,
                max_steps=computed_max_steps,
                save_strategy="no",
                report_to="none",
                gradient_checkpointing=True,
                optim="adamw_8bit",
                output_dir=question_model_output,
            )
            if args.debug:
                grpo_kw["max_steps"] = 1

            good_questions_path = os.path.join(args.output_dir, f"_good_questions_{i}.jsonl")
            grpo_manifest = {
                "question_model_path": question_model_input,
                "student_model_path": student_input,
                "judge_model_path": args.model_name,
                "tokenizer_name": args.model_name,
                "dataset_path": args.dataset_path,
                "output_dir": question_model_output,
                "good_questions_path": good_questions_path,
                "grpo_config": grpo_kw,
            }
            grpo_manifest_path = os.path.join(args.output_dir, f"_grpo_manifest_{i}.json")
            with open(grpo_manifest_path, "w") as f:
                json.dump(grpo_manifest, f, indent=2)

            grpo_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo_phase_worker.py")
            print(f"Running GRPO training in subprocess ({grpo_worker})...")
            subprocess.run([sys.executable, grpo_worker, grpo_manifest_path], check=True)
            os.remove(grpo_manifest_path)
            print(f"Question model saved to: {question_model_output}")
            grpo_ran = True

        # ── Phase 2: Build question dataset for distillation ──
        # Move eval models off GPU before any potential subprocess.
        student_model.to("cpu")
        if judge_model is not None:
            judge_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Collect good questions discovered during this iteration's GRPO training.
        if grpo_ran:
            gq_path = os.path.join(args.output_dir, f"_good_questions_{i}.jsonl")
            if os.path.exists(gq_path):
                with open(gq_path) as f:
                    new_qs = [json.loads(line) for line in f if line.strip()]
                all_good_questions.extend(new_qs)
                os.remove(gq_path)
                print(f"Collected {len(new_qs)} good questions this iteration")

        if args.question_model_path is not None:
            question_gen_path = args.question_model_path
        elif args.skip_generator_training:
            question_gen_path = args.model_name
        else:
            question_gen_path = os.path.join(args.output_dir, f"question_model_{i}")

        # Compute per-passage deficits against the target.
        good_by_passage: Dict[str, list] = {}
        for q in all_good_questions:
            good_by_passage.setdefault(q["id"], []).append(q)

        deficit_counts: Dict[str, int] = {}
        for pid in dataset:
            need = args.num_question_generations - len(good_by_passage.get(pid, []))
            if need > 0:
                deficit_counts[pid] = need

        if deficit_counts:
            total_deficit = sum(deficit_counts.values())
            print(f"Generating questions for {len(deficit_counts)}/{len(dataset)} passages "
                  f"({total_deficit} total deficit, have {len(all_good_questions)} good)")
            if args.accumulate_questions and i > 0 and prev_generated_by_id:
                k = args.num_accumulated_questions
                previous_questions_by_id = {}
                for pid, qs in prev_generated_by_id.items():
                    if not qs:
                        continue
                    selected = qs if k is None else qs[-k:] if k > 0 else []
                    if selected:
                        previous_questions_by_id[pid] = [q["question"] for q in selected]
                if previous_questions_by_id:
                    cap_str = "all" if k is None else f"up to {k}"
                    print(f"Accumulating {cap_str} previous questions for "
                          f"{len(previous_questions_by_id)} passages")
            else:
                previous_questions_by_id = None
            generated_dataset = build_question_dataset_for_deficits(
                question_gen_path, args.model_name, dataset,
                deficit_counts, args.num_questions_per_generation,
                student_prompt_fn=build_student_prompt,
                teacher_prompt_fn=build_teacher_prompt,
                previous_questions_by_id=previous_questions_by_id,
            )
        else:
            generated_dataset = None
            print(f"All {len(dataset)} passages have >= {args.num_question_generations} "
                  f"good questions; skipping generation")

        rows = []
        for q in all_good_questions:
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "answer": q["answer"],
                "prompt": _build_prompt_conversation(build_student_prompt(q["question"])),
                "teacher_prompt": _build_prompt_conversation(
                    build_teacher_prompt(q["question"], dataset[q["id"]])),
            })
        if generated_dataset is not None:
            for idx in range(len(generated_dataset)):
                rows.append(dict(generated_dataset[idx]))
        question_dataset = Dataset.from_list(rows)

        if args.accumulate_questions:
            prev_generated_by_id = {}
            if generated_dataset is not None:
                for idx in range(len(generated_dataset)):
                    row = generated_dataset[idx]
                    prev_generated_by_id.setdefault(row["id"], []).append(
                        {"question": row["question"], "answer": row["answer"]}
                    )
        print(f"Distillation dataset: {len(all_good_questions)} good questions"
              + (f" + {len(generated_dataset)} generated" if generated_dataset else "")
              + f" = {len(question_dataset)} total")
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

        distil_grad_accum = min(args.gradient_accumulation_steps, len(question_dataset))
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
            gradient_accumulation_steps=distil_grad_accum,
            max_prompt_length=1024,
            max_completion_length=args.max_completion_length,
            num_train_epochs=args.num_train_epochs,
            save_strategy="no",
            max_grad_norm=1,
            report_to="none",
            output_dir=student_model_dir,
            log_completions=args.log_student_completions and not args.debug,
            sync_ref_model=False,
            vllm_importance_sampling_correction=args.vllm_importance_sampling_correction,
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
        os.remove(manifest_path)
        # os.remove(dataset_path)

        # Reload the trained student from the subprocess output
        del student_model
        gc.collect()
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_dir, torch_dtype=torch.bfloat16,
        )

        # Previous iteration's on-disk models are no longer needed by any
        # subprocess; clean them up unless the user asked to keep them.
        if i > 0:
            if not args.save_student_model:
                prev = os.path.join(args.output_dir, f"student_model_{i - 1}")
                if os.path.isdir(prev):
                    shutil.rmtree(prev)
            if not args.save_question_model and not skip_question_training:
                prev = os.path.join(args.output_dir, f"question_model_{i - 1}")
                if os.path.isdir(prev):
                    shutil.rmtree(prev)

        if args.report_student_performance:
            print("Reporting student performance...")

            # ---- Resolve evaluation questions ----
            if args.eval_questions_path:
                raw_qs = load_questions_jsonl(args.eval_questions_path)
                eval_questions = [q["input"] for q in raw_qs]
                eval_answers = [q["output"] for q in raw_qs]
                eval_ids = [q["id"] for q in raw_qs]
            elif args.eval_question_model:
                if args.eval_question_backend == "openai":
                    oai = _openai_client()
                    raw_qs = generate_questions_openai(
                        oai, args.eval_question_model, dataset,
                        args.eval_num_questions,
                        num_questions_per_generation=args.num_questions_per_generation,
                    )
                else:
                    raw_qs = generate_questions(
                        args.eval_question_model, args.eval_question_model,
                        dataset, args.eval_num_questions,
                        num_questions_per_generation=args.num_questions_per_generation,
                    )
                eval_questions = [q["question"] for q in raw_qs]
                eval_answers = [q["answer"] for q in raw_qs]
                eval_ids = [q["id"] for q in raw_qs]
            else:
                eval_questions = list(question_dataset["question"])
                eval_answers = list(question_dataset["answer"])
                eval_ids = list(question_dataset["id"])

            print(f"Evaluation questions: {len(eval_questions)}")

            # ---- Student answers (always local HF) ----
            student_model.eval()
            student_model.to("cuda")

            student_answers = batch_generate_answers(
                student_model, tokenizer, eval_questions,
            )

            # ---- Judging ----
            if args.eval_judge_backend == "openai":
                judge_name = args.eval_judge_model
                if judge_name is None:
                    raise SystemExit(
                        "--eval_judge_model is required when --eval_judge_backend=openai"
                    )
                oai_judge = _openai_client()
                verdicts = judge_openai_batch(
                    oai_judge, judge_name,
                    eval_questions, eval_answers, student_answers,
                )
            else:
                if judge_model is not None:
                    judge_model.to("cuda")
                verdicts = batch_judge_answers(
                    judge_model, tokenizer,
                    eval_questions, eval_answers, student_answers,
                )

            # ---- Assemble results ----
            results = []
            correct = 0
            for idx, (q, ref, sa, (is_correct, verdict)) in enumerate(
                zip(eval_questions, eval_answers, student_answers, verdicts)
            ):
                correct += int(is_correct)
                entry = {
                    "question": q,
                    "reference_answer": ref,
                    "student_answer": sa,
                    "verdict": verdict,
                    "is_correct": is_correct,
                    "id": eval_ids[idx],
                }
                results.append(entry)

            total = len(results)
            accuracy = correct / total if total else 0.0
            eval_summary = {
                "iteration": i,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results,
            }

            print(f"[Iteration {i}] Student accuracy: "
                  f"{eval_summary['correct']}/{eval_summary['total']} "
                  f"({eval_summary['accuracy']:.1%})")

            results_path = os.path.join(args.output_dir, f"results_student_model_{i}.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(eval_summary, f, indent=2)
            print(f"Saved evaluation results to: {results_path}")

            if args.save_student_result_copy_dir:
                copy_name = args.save_student_result_copy_name or "student_model"
                copy_name = f"results_{copy_name}_{i}.json"
                copy_path = os.path.join(args.save_student_result_copy_dir, copy_name)
                with open(copy_path, "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, indent=2)
                print(f"Copied evaluation results to: {copy_path}")

        student_model.to("cpu")
        if judge_model is not None:
            judge_model.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Clean up the last iteration's intermediate models.
    last = args.num_generation_iterations - 1
    if not args.save_student_model:
        path = os.path.join(args.output_dir, f"student_model_{last}")
        if os.path.isdir(path):
            shutil.rmtree(path)
    if not args.save_question_model and not skip_question_training:
        path = os.path.join(args.output_dir, f"question_model_{last}")
        if os.path.isdir(path):
            shutil.rmtree(path)

    # Save the final student model always
    final_student_model_dir = os.path.join(args.output_dir, "final_student_model")
    student_model.save_pretrained(final_student_model_dir)
    tokenizer.save_pretrained(final_student_model_dir)
    print(f"Saved final student model to: {final_student_model_dir}")
