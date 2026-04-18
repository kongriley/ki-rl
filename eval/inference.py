"""Shared inference primitives for student evaluation and judge verdicts."""

import re
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_student_prompt(question: str) -> str:
    return f"<Question>\n{question}\n\n<Answer>"


def build_teacher_prompt(question: str, document: str) -> str:
    return (
        "Read the following passage carefully, then answer the question.\n\n"
        f"<Passage>\n{document}\n\n"
    ) + build_student_prompt(question)


def build_judge_prompt(question: str, reference_answer: str, student_answer: str) -> str:
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


def build_batched_judge_prompt(
    questions: List[str],
    references: List[str],
    student_answers: List[str],
) -> str:
    """Build a single prompt that asks the judge to evaluate N triples at once."""
    header = (
        "You are an impartial judge. For each numbered item below, decide whether "
        "the student's answer agrees with the reference answer. The wording need not "
        "match exactly, but all key facts must be present and accurate.\n"
        "Respond with ONLY a numbered list, one verdict per line, in the format:\n"
        "1. correct\n"
        "2. incorrect\n"
        "...and so on.\n\n"
    )
    items: List[str] = []
    for i, (q, ref, sa) in enumerate(zip(questions, references, student_answers), 1):
        items.append(
            f"{i}.\n"
            f"<Question>\n{q}\n\n"
            f"<Reference Answer>\n{ref}\n\n"
            f"<Student's Answer>\n{sa}\n"
        )
    return header + "\n".join(items)


def parse_batched_verdicts(
    text: str, expected_count: int
) -> List[Tuple[bool, str]]:
    """Parse numbered verdicts (``1. correct``, ``2. incorrect``, ...).

    Returns a list of ``(is_correct, raw_line)`` tuples of length
    ``expected_count``.  If a line cannot be matched, it is treated as
    incorrect and a warning is emitted.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    verdicts: List[Tuple[bool, str]] = []
    matched_by_number: Dict[int, str] = {}
    for ln in lines:
        m = re.match(r"(\d+)\.\s*(.*)", ln)
        if m:
            matched_by_number[int(m.group(1))] = m.group(2)

    for idx in range(1, expected_count + 1):
        raw = matched_by_number.get(idx, "")
        if raw:
            verdicts.append(parse_verdict(raw))
        else:
            warnings.warn(
                f"Could not parse verdict for item {idx} from batched judge "
                f"response; marking as incorrect. Full response:\n{text}"
            )
            verdicts.append((False, f"[parse_error] item {idx}"))
    return verdicts


def format_instruct_user_prompt(tokenizer, user_message: str) -> str:
    """Match DistilTrainer/TRL: one user message, then generation prompt.

    Training uses ``maybe_apply_chat_template({"prompt": messages}, tokenizer)``
    with ``add_generation_prompt=True`` when the last role is ``user``.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return user_message
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def parse_verdict(text: str) -> Tuple[bool, str]:
    v = text.lower().strip()
    is_correct = "correct" in v and "incorrect" not in v
    return is_correct, text.strip()


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    max_length: int = 1024,
    use_tqdm: bool = False,
) -> List[str]:
    """Generate completions for a list of pre-built prompt strings (after chat template if used)."""
    all_outputs: List[str] = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="Generating answers",
        disable=not use_tqdm
    ):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding_side="left",
            add_special_tokens=False,
        ).to(model.device)
        gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens)
        if tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        out = model.generate(**gen_kwargs)
        decoded = tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        all_outputs.extend(decoded)
    return all_outputs


def batch_generate_answers(
    model,
    tokenizer,
    questions: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    use_tqdm: bool = False,
) -> List[str]:
    """Generate closed-book answers (no passage context)."""
    prompts = [
        format_instruct_user_prompt(tokenizer, build_student_prompt(q)) for q in questions
    ]
    return batch_generate(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, use_tqdm=use_tqdm)


def batch_generate_with_context(
    model,
    tokenizer,
    questions: List[str],
    documents: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> List[str]:
    """Generate open-book answers (with passage context)."""
    prompts = [
        format_instruct_user_prompt(tokenizer, build_teacher_prompt(q, doc))
        for q, doc in zip(questions, documents)
    ]
    return batch_generate(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_length=4096,
    )


@torch.no_grad()
def batch_judge_answers(
    model,
    tokenizer,
    questions: List[str],
    references: List[str],
    student_answers: List[str],
    batch_size: int = 4,
) -> List[Tuple[bool, str]]:
    """Judge each (question, reference, student_answer) triple and return (is_correct, verdict_text)."""
    all_verdicts: List[Tuple[bool, str]] = []
    for start in range(0, len(questions), batch_size):
        batch_qs = questions[start : start + batch_size]
        batch_refs = references[start : start + batch_size]
        batch_sas = student_answers[start : start + batch_size]

        judge_texts = [
            format_instruct_user_prompt(tokenizer, build_judge_prompt(q, ref, sa))
            for q, ref, sa in zip(batch_qs, batch_refs, batch_sas)
        ]
        inputs = tokenizer(
            judge_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            padding_side="left",
            add_special_tokens=False,
        ).to(model.device)
        gen_kwargs = dict(**inputs, max_new_tokens=16)
        if tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        out = model.generate(**gen_kwargs)
        raw = tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        all_verdicts.extend(parse_verdict(v) for v in raw)
    return all_verdicts


@torch.no_grad()
def batch_judge_answers_multi_prompt(
    model,
    tokenizer,
    questions: List[str],
    references: List[str],
    student_answers: List[str],
    group_size: int = 5,
) -> List[Tuple[bool, str]]:
    """Judge triples by packing ``group_size`` items into each prompt.

    Unlike :func:`batch_judge_answers` (one prompt per triple, GPU-batched),
    this function reduces the total number of forward passes by asking the
    judge to evaluate multiple triples in a single generation.
    """
    all_verdicts: List[Tuple[bool, str]] = []
    for start in range(0, len(questions), group_size):
        grp_qs = questions[start : start + group_size]
        grp_refs = references[start : start + group_size]
        grp_sas = student_answers[start : start + group_size]

        prompt_text = build_batched_judge_prompt(grp_qs, grp_refs, grp_sas)
        prompt = format_instruct_user_prompt(tokenizer, prompt_text)

        inputs = tokenizer(
            [prompt],
            truncation=True,
            max_length=4096,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        max_tokens = 8 * len(grp_qs)
        gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False)
        if tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        out = model.generate(**gen_kwargs)
        raw = tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        all_verdicts.extend(parse_batched_verdicts(raw, len(grp_qs)))
    return all_verdicts


# ---------------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------------

def evaluate_qa(
    student_model,
    judge_model,
    tokenizer,
    questions: List[str],
    references: List[str],
    ids: Optional[List[str]] = None,
    batch_size: int = 4,
) -> Dict:
    """Run closed-book student evaluation and return an accuracy summary + per-question results."""
    student_answers = batch_generate_answers(
        student_model, tokenizer, questions, batch_size=batch_size
    )
    verdicts = batch_judge_answers(
        judge_model, tokenizer, questions, references, student_answers, batch_size=batch_size
    )

    results = []
    correct = 0
    for idx, (q, ref, sa, (is_correct, verdict)) in enumerate(
        zip(questions, references, student_answers, verdicts)
    ):
        correct += int(is_correct)
        entry: Dict = {
            "question": q,
            "reference_answer": ref,
            "student_answer": sa,
            "verdict": verdict,
            "is_correct": is_correct,
        }
        if ids is not None:
            entry["id"] = ids[idx]
        results.append(entry)

    total = len(results)
    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }
