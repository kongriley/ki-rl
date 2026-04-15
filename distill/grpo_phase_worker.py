"""GRPO phase worker — trains the question model in a subprocess.

Running GRPO in a fresh process avoids two problems:
1. vLLM colocated sleep-mode MemPool is a per-process singleton; creating a
   second colocated vLLM with sleep mode in the same process crashes.
2. CUDA memory leaked by the pluggable allocator / CUDA graphs doesn't
   accumulate across iterations.
"""

import gc
import json
import os
import re
import socket
import sys
from typing import Dict, List, Tuple


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_find_free_port())

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from inference import (  # noqa: E402
    batch_generate_answers,
    batch_generate_with_context,
    batch_judge_answers,
)
from main import _normalize_completion_text  # noqa: E402
from generate_questions import (  # noqa: E402
    _parse_question_answer,
    _is_valid_qa,
    build_question_prompt_dataset,
    load_dataset_json,
)


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


def main():
    manifest_path = sys.argv[1]
    with open(manifest_path) as f:
        manifest = json.load(f)

    question_model = AutoModelForCausalLM.from_pretrained(
        manifest["question_model_path"], torch_dtype=torch.bfloat16,
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        manifest["student_model_path"], torch_dtype=torch.bfloat16,
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        manifest["judge_model_path"], torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset_json(manifest["dataset_path"])
    question_prompt_dataset = build_question_prompt_dataset(dataset)

    config = GRPOConfig(**manifest["grpo_config"])

    _reward_call_count = [0]
    good_questions = []

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

        for idx in range(len(questions)):
            if "good_question" in all_reasons[idx]:
                good_questions.append({
                    "id": passage_ids[idx],
                    "question": questions[idx],
                    "answer": ref_answers[idx],
                })

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

    trainer = GRPOTrainer(
        model=question_model,
        processing_class=tokenizer,
        args=config,
        train_dataset=question_prompt_dataset,
        reward_funcs=reward_question_difficulty,
    )
    trainer.train()

    good_questions_path = manifest.get("good_questions_path")
    if good_questions_path:
        with open(good_questions_path, "w") as f:
            for q in good_questions:
                f.write(json.dumps(q) + "\n")
        print(f"[grpo_phase_worker] Saved {len(good_questions)} good questions to {good_questions_path}")

    output_dir = manifest["output_dir"]
    trainer.save_model(output_dir)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        tokenizer.save_pretrained(output_dir)
    print(f"[grpo_phase_worker] Saved question model to {output_dir}")

    del trainer, question_model, student_model, judge_model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
