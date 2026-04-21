import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from tqdm import tqdm


def load_dataset_json(data_path: str) -> Dict:
    """Load a wiki-style JSON file into ``{id: text}`` mapping."""
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = {d["id"]: d["text"] for d in dataset}
    return dataset


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _create_question_prompt(
    text: str,
    num_questions: int = 1,
    previous_questions: Optional[List[str]] = None,
) -> str:
    plural = '' if num_questions == 1 else 's'
    previous_block = ""
    if previous_questions:
        prev_list = "\n".join(f"- {q}" for q in previous_questions)
        previous_block = f"""
The following question{'' if len(previous_questions) == 1 else 's'} {'was' if len(previous_questions) == 1 else 'were'} already created for this passage:
{prev_list}

Generate {num_questions} *different* question{plural} covering different aspects of the passage. Do not repeat any of the above.
"""
    return f"""Read the passage below, then write exactly {num_questions} question{plural} with short answer{plural}.

Rules:
- Each question must include enough context (names, dates, topics) to be answerable without the passage.
- Each answer must be a short, factual response grounded in the passage.
- Do not copy text verbatim from the passage as your question.
{previous_block}
Use this exact format for every pair (do not add any other tags or headers):
Q: <your question>
A: <your answer>

Passage:
{text}

"""


def _build_prompt_conversation(prompt: str):
    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_question_answer(text: str) -> Tuple[str, str]:
    """Extract a single (question, answer) pair from ``Q: … A: …`` text."""
    q_match = re.search(r"Q:\s*(.+?)(?=\nA:|\Z)", text, re.DOTALL)
    a_match = re.search(r"A:\s*(.+?)(?=\nQ:|\Z)", text, re.DOTALL)
    if q_match and a_match:
        return q_match.group(1).strip(), a_match.group(1).strip()
    return text.strip(), ""


def _parse_question_answers(text: str) -> List[Tuple[str, str]]:
    """Extract all (question, answer) pairs from ``Q: … A: …`` text."""
    matches = re.findall(r"Q:\s*(.+?)\s*\nA:\s*(.+?)(?=\nQ:|\Z)", text, re.DOTALL)
    if matches:
        return [(q.strip(), a.strip()) for q, a in matches]
    return [_parse_question_answer(text)]


def _is_valid_qa(question: str, answer: str) -> bool:
    if len(question) < 10 or len(answer) < 3:
        return False
    if re.search(r"Q:|A:", question) or re.search(r"Q:|A:", answer):
        return False
    return True


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_question_prompt_dataset(dataset: Dict, num_questions_per_generation: int = 1) -> Dataset:
    question_prompt_dataset = []
    for id, text in dataset.items():
        prompt = _create_question_prompt(text, num_questions=num_questions_per_generation)
        prompt = _build_prompt_conversation(prompt)
        question_prompt_dataset.append({"id": id, "prompt": prompt})
    return Dataset.from_list(question_prompt_dataset)


# ---------------------------------------------------------------------------
# vLLM question generation
# ---------------------------------------------------------------------------

def generate_questions(
    model_name_or_path: str,
    tokenizer_name: str,
    dataset: Dict,
    num_question_generations: int,
    num_questions_per_generation: int = 5,
    temperature: float = 1.2,
    max_retries: int = 5,
    tensor_parallel_size: int = 1,
) -> list:
    from vllm import LLM, SamplingParams

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
        tensor_parallel_size=tensor_parallel_size,
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


def generate_questions_for_deficits(
    model_name_or_path: str,
    tokenizer_name: str,
    dataset: Dict,
    deficit_counts: Dict,
    num_questions_per_generation: int = 5,
    temperature: float = 1.2,
    max_retries: int = 3,
    previous_questions_by_id: Optional[Dict[str, List[str]]] = None,
) -> list:
    """Generate questions only for passages that still need them.

    ``deficit_counts`` maps passage id -> number of additional questions needed.
    Passages not present (or with count <= 0) are skipped entirely.

    ``previous_questions_by_id`` optionally maps passage id -> list of questions
    generated for that passage in a previous iteration. When provided, these
    are appended to the prompt with an instruction to generate different
    questions.
    """
    from vllm import LLM, SamplingParams

    deficit_counts = {pid: n for pid, n in deficit_counts.items() if n > 0}
    if not deficit_counts:
        return []

    prev_qs_by_id: Dict[str, List[str]] = previous_questions_by_id or {}

    def _prompt_for(pid: str) -> str:
        return _create_question_prompt(
            dataset[pid],
            num_questions=num_questions_per_generation,
            previous_questions=prev_qs_by_id.get(pid),
        )

    max_needed = max(deficit_counts.values())
    n = max(1, math.ceil(max_needed / num_questions_per_generation))

    free_mem, total_mem = torch.cuda.mem_get_info()
    free_frac = free_mem / total_mem
    target_utilization = min(0.6, free_frac - 0.05)
    target_utilization = max(target_utilization, 0.2)
    print(f"  vLLM deficit gen: {free_mem/1e9:.1f}/{total_mem/1e9:.1f} GiB free "
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

    all_ids = list(deficit_counts.keys())
    all_prompts = [_prompt_for(pid) for pid in all_ids]

    questions_by_id: Dict[str, list] = {pid: [] for pid in all_ids}

    for attempt in range(max_retries):
        if attempt == 0:
            pending_ids = all_ids
            pending_prompts = all_prompts
        else:
            pending_ids = [
                pid for pid in all_ids
                if len(questions_by_id[pid]) < deficit_counts[pid]
            ]
            if not pending_ids:
                break
            pending_prompts = [_prompt_for(pid) for pid in pending_ids]
            print(f"  Retry {attempt}/{max_retries}: regenerating for "
                  f"{len(pending_ids)} passages with insufficient questions")

        outputs = llm.generate(pending_prompts, sampling_params)

        for output, pid in zip(outputs, pending_ids):
            for completion in output.outputs:
                for question, answer in _parse_question_answers(completion.text):
                    if _is_valid_qa(question, answer):
                        questions_by_id[pid].append(
                            {"id": pid, "question": question, "answer": answer}
                        )

    questions = []
    for pid in all_ids:
        needed = deficit_counts[pid]
        passage_qs = questions_by_id[pid][:needed]
        if len(passage_qs) < needed:
            print(f"  Warning: passage {pid} only got {len(passage_qs)}/{needed} "
                  f"questions after {max_retries} attempts")
        questions.extend(passage_qs)

    del llm
    torch.cuda.empty_cache()
    return questions


# ---------------------------------------------------------------------------
# OpenAI question generation
# ---------------------------------------------------------------------------

def generate_questions_openai(
    client,
    model: str,
    dataset: Dict,
    num_questions: int,
    num_questions_per_generation: int = 5,
) -> list:
    """Generate questions via the OpenAI API using the same prompt format as the vLLM path."""
    all_questions = []
    for passage_id, text in tqdm(dataset.items(), desc="Generating questions (OpenAI)"):
        prompt = _create_question_prompt(text, num_questions=num_questions_per_generation)
        messages = [{"role": "user", "content": prompt}]

        collected = []
        while len(collected) < num_questions:
            resp = client.chat.completions.create(
                model=model, messages=messages
            )
            raw = resp.choices[0].message.content
            for question, answer in _parse_question_answers(raw):
                if _is_valid_qa(question, answer):
                    collected.append({"id": passage_id, "question": question, "answer": answer})

        if len(collected) < num_questions:
            print(f"  Warning: passage {passage_id} only got {len(collected)}/{num_questions} valid questions")
        all_questions.extend(collected[:num_questions])

    return all_questions


# ---------------------------------------------------------------------------
# Subprocess helpers (used by distill/main.py)
# ---------------------------------------------------------------------------

def _run_generate_questions(args_and_queue):
    """Subprocess target: generates questions in a fresh CUDA context."""
    (model_name_or_path, tokenizer_name, dataset, num_question_generations,
     num_questions_per_generation, temperature, tensor_parallel_size, queue) = args_and_queue
    results = generate_questions(
        model_name_or_path, tokenizer_name, dataset,
        num_question_generations, num_questions_per_generation, temperature,
        tensor_parallel_size=tensor_parallel_size,
    )
    queue.put(results)


def build_question_dataset(
    model_name_or_path: str,
    tokenizer_name: str,
    dataset: Dict,
    num_question_generations: int,
    num_questions_per_generation: int = 5,
    student_prompt_fn=None,
    teacher_prompt_fn=None,
    tensor_parallel_size: int = 1,
) -> Dataset:
    """Generate questions in a subprocess and build a HF Dataset.

    ``student_prompt_fn(question) -> str`` and
    ``teacher_prompt_fn(question, passage) -> str`` are optional prompt
    builders.  When *None* the corresponding ``prompt`` / ``teacher_prompt``
    columns are omitted from the returned dataset.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_run_generate_questions,
        args=((model_name_or_path, tokenizer_name, dataset,
               num_question_generations, num_questions_per_generation, 1.2,
               tensor_parallel_size, queue),),
    )
    p.start()
    questions = queue.get()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Question generation subprocess exited with code {p.exitcode}")

    question_dataset = []
    for row in questions:
        id, question, answer = row["id"], row["question"], row["answer"]
        entry: dict = {"id": id, "question": question, "answer": answer}
        if student_prompt_fn is not None:
            entry["prompt"] = _build_prompt_conversation(student_prompt_fn(question))
        if teacher_prompt_fn is not None:
            entry["teacher_prompt"] = _build_prompt_conversation(teacher_prompt_fn(question, dataset[id]))
        question_dataset.append(entry)
    return Dataset.from_list(question_dataset)


def _run_generate_questions_for_deficits(args_and_queue):
    """Subprocess target: generates deficit questions in a fresh CUDA context."""
    (model_name_or_path, tokenizer_name, dataset, deficit_counts,
     num_questions_per_generation, temperature, previous_questions_by_id,
     queue) = args_and_queue
    results = generate_questions_for_deficits(
        model_name_or_path, tokenizer_name, dataset,
        deficit_counts, num_questions_per_generation, temperature,
        previous_questions_by_id=previous_questions_by_id,
    )
    queue.put(results)


def build_question_dataset_for_deficits(
    model_name_or_path: str,
    tokenizer_name: str,
    dataset: Dict,
    deficit_counts: Dict,
    num_questions_per_generation: int = 5,
    student_prompt_fn=None,
    teacher_prompt_fn=None,
    previous_questions_by_id: Optional[Dict[str, List[str]]] = None,
) -> Dataset:
    """Generate questions for passages with deficits in a subprocess and build a HF Dataset.

    ``deficit_counts`` maps passage id -> number of additional questions needed.

    ``previous_questions_by_id`` optionally maps passage id -> list of previously
    generated questions. When provided, the generation prompt is augmented with
    these and asks for different questions (see ``_create_question_prompt``).
    """
    deficit_counts = {pid: n for pid, n in deficit_counts.items() if n > 0}
    if not deficit_counts:
        return Dataset.from_list([])

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_run_generate_questions_for_deficits,
        args=((model_name_or_path, tokenizer_name, dataset,
               deficit_counts, num_questions_per_generation, 1.2,
               previous_questions_by_id, queue),),
    )
    p.start()
    questions = queue.get()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Question generation subprocess exited with code {p.exitcode}")

    question_dataset = []
    for row in questions:
        id, question, answer = row["id"], row["question"], row["answer"]
        entry: dict = {"id": id, "question": question, "answer": answer}
        if student_prompt_fn is not None:
            entry["prompt"] = _build_prompt_conversation(student_prompt_fn(question))
        if teacher_prompt_fn is not None:
            entry["teacher_prompt"] = _build_prompt_conversation(teacher_prompt_fn(question, dataset[id]))
        question_dataset.append(entry)
    return Dataset.from_list(question_dataset)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _openai_client():
    from openai import OpenAI
    import dotenv
    dotenv.load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def parse_args():
    p = argparse.ArgumentParser(description="Generate questions from wiki passages using vLLM or OpenAI")
    p.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct",
                    help="HF hub id / local checkpoint / OpenAI model name")
    p.add_argument("--backend", choices=["hf", "openai"], default="hf",
                    help="Generation backend (default: hf/vLLM)")
    p.add_argument("--tokenizer", type=str, default=None,
                    help="Tokenizer name (defaults to --model)")
    p.add_argument("--dataset_path", type=str, default="data/wiki_20/data.json",
                    help="Path to wiki JSON file")
    p.add_argument("--num_questions", type=int, default=3,
                    help="Target number of valid questions per passage")
    p.add_argument("--num_questions_per_generation", type=int, default=5,
                    help="Questions requested per prompt")
    p.add_argument("--temperature", type=float, default=1.2,
                    help="Sampling temperature, will be overridden by backend")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSONL path (default: auto-generated in dataset dir)")
    p.add_argument("--max_passages", type=int, default=None,
                    help="Limit to first N passages (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = args.tokenizer or args.model

    dataset = load_dataset_json(args.dataset_path)
    if args.max_passages is not None:
        dataset = dict(list(dataset.items())[:args.max_passages])
    print(f"Loaded {len(dataset)} passages from {args.dataset_path}")

    if args.backend == "openai":
        if args.temperature != 1:
            print(f"Warning: temperature {args.temperature} is not supported for OpenAI")
        client = _openai_client()
        raw_qs = generate_questions_openai(
            client, args.model, dataset, args.num_questions,
            num_questions_per_generation=args.num_questions_per_generation
        )
    else:
        raw_qs = generate_questions(
            args.model, tokenizer, dataset, args.num_questions,
            num_questions_per_generation=args.num_questions_per_generation,
            temperature=args.temperature,
        )

    questions = []
    for q in raw_qs:
        questions.append({
            "id": q["id"],
            "instruction": "Respond to the following question.",
            "input": q["question"],
            "output": q["answer"],
        })

    if args.output is None:
        safe_model_name = args.model.replace("/", "_")
        out_dir = os.path.dirname(os.path.abspath(args.dataset_path))
        args.output = os.path.join(out_dir, f"{safe_model_name}_questions.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"Generated {len(questions)} questions -> {args.output}")


if __name__ == "__main__":
    main()
