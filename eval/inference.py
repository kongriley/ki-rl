"""Shared inference primitives for student evaluation and judge verdicts."""

from typing import Dict, List, Optional, Tuple

import torch


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
) -> List[str]:
    """Generate completions for a list of pre-built prompt strings."""
    all_outputs: List[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding_side="left",
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
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
) -> List[str]:
    """Generate closed-book answers (no passage context)."""
    prompts = [build_student_prompt(q) for q in questions]
    return batch_generate(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens)


def batch_generate_with_context(
    model,
    tokenizer,
    questions: List[str],
    documents: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> List[str]:
    """Generate open-book answers (with passage context)."""
    prompts = [build_teacher_prompt(q, doc) for q, doc in zip(questions, documents)]
    return batch_generate(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, max_length=4096)


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

        inputs = tokenizer(
            [
                build_judge_prompt(q, ref, sa)
                for q, ref, sa in zip(batch_qs, batch_refs, batch_sas)
            ],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            padding_side="left",
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=16)
        raw = tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        all_verdicts.extend(parse_verdict(v) for v in raw)
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
