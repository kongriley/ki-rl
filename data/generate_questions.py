import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
from typing import Dict, List, Tuple

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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

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


def _is_valid_qa(question: str, answer: str) -> bool:
    return len(question) >= 10 and len(answer) >= 1


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
    max_retries: int = 3,
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
        attempts = 0
        while len(collected) < num_questions and attempts < 3:
            resp = client.chat.completions.create(
                model=model, messages=messages
            )
            raw = resp.choices[0].message.content
            for question, answer in _parse_question_answers(raw):
                if _is_valid_qa(question, answer):
                    collected.append({"id": passage_id, "question": question, "answer": answer})
            attempts += 1

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
     num_questions_per_generation, temperature, queue) = args_and_queue
    results = generate_questions(
        model_name_or_path, tokenizer_name, dataset,
        num_question_generations, num_questions_per_generation, temperature,
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
