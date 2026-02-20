from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Dict, List, Sequence

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from question_rl import sample_questions


@dataclass
class QuestionThenDistillConfig:
    # Question generation
    questions_per_text: int = 1
    qg_max_prompt_length: int = 1024
    qg_max_question_new_tokens: int = 64
    qg_temperature: float = 1.0
    qg_top_p: float = 0.95

    # Teacher answering (used to create "example answer" that goes into teacher_prompt)
    teacher_answer_max_prompt_length: int = 2048
    teacher_answer_max_new_tokens: int = 512
    teacher_answer_temperature: float = 0.2
    teacher_answer_top_p: float = 0.95

    # Prompt formatting
    include_system_prompt: bool = True


def _ensure_pad(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _make_student_prompt(question: str, *, include_system: bool) -> List[dict]:
    msgs: List[dict] = []
    if include_system:
        msgs.append(
            {
                "role": "system",
                "content": "Answer the user's question. Be concise and correct.",
            }
        )
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs


def _make_teacher_answer_prompt(text: str, question: str, *, include_system: bool) -> List[dict]:
    msgs: List[dict] = []
    if include_system:
        msgs.append(
            {
                "role": "system",
                "content": "Answer the user's question using ONLY the provided text. Be concise and correct.",
            }
        )
    msgs.append({"role": "user", "content": f"<TEXT>\n{text}\n</TEXT>\n\nQuestion: {question}"})
    return msgs


def _make_teacher_prompt(text: str, question: str, teacher_answer: str, *, include_system: bool) -> List[dict]:
    """
    DistilTrainer conditions the teacher on this prompt when scoring the student's completion tokens.

    We give the teacher a *worked example answer* (teacher_answer) and ask it to answer "on its own".
    That mirrors the existing wiki_1k behavior in `main.py` which provides an "example response".
    """
    msgs: List[dict] = []
    if include_system:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You will be given a text, a question, and an example answer. "
                    "Use the example to answer well."
                ),
            }
        )

    teacher_prompt = Template(
        """<TEXT>
$text
</TEXT>

Question: $question

This is an example answer:
$answer

Now answer with a response of your own."""
    )
    msgs.append({"role": "user", "content": teacher_prompt.substitute(text=text, question=question, answer=teacher_answer)})
    return msgs


@torch.no_grad()
def _teacher_answer_batch(
    teacher_model: PreTrainedModel,
    teacher_tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[List[dict]],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    prompt_texts = [
        teacher_tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        if hasattr(teacher_tokenizer, "apply_chat_template")
        else "\n".join(f"{m['role'].upper()}: {m['content']}" for m in p) + "\nASSISTANT:"
        for p in prompts
    ]
    enc = teacher_tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(teacher_model.device) for k, v in enc.items()}
    out = teacher_model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=teacher_tokenizer.pad_token_id,
        eos_token_id=teacher_tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].size(1) :]
    answers = teacher_tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [a.strip() for a in answers]


def build_question_distillation_dataset(
    *,
    texts: Sequence[str],
    question_model: PreTrainedModel,
    question_tokenizer: PreTrainedTokenizerBase,
    teacher_answer_model: PreTrainedModel,
    teacher_answer_tokenizer: PreTrainedTokenizerBase,
    cfg: QuestionThenDistillConfig,
) -> Dataset:
    _ensure_pad(question_tokenizer)
    _ensure_pad(teacher_answer_tokenizer)

    rows: List[Dict] = []
    for text in texts:
        for _ in range(cfg.questions_per_text):
            qs, _, _ = sample_questions(
                question_model,
                question_tokenizer,
                [text],
                max_prompt_length=cfg.qg_max_prompt_length,
                max_new_tokens=cfg.qg_max_question_new_tokens,
                temperature=cfg.qg_temperature,
                top_p=cfg.qg_top_p,
            )
            question = qs[0].strip()
            # Student sees QUESTION ONLY; teacher sees TEXT+QUESTION.
            student_prompt = _make_student_prompt(question, include_system=cfg.include_system_prompt)
            teacher_answer_prompt = _make_teacher_answer_prompt(text, question, include_system=cfg.include_system_prompt)
            answer = _teacher_answer_batch(
                teacher_answer_model,
                teacher_answer_tokenizer,
                [teacher_answer_prompt],
                max_prompt_length=cfg.teacher_answer_max_prompt_length,
                max_new_tokens=cfg.teacher_answer_max_new_tokens,
                temperature=cfg.teacher_answer_temperature,
                top_p=cfg.teacher_answer_top_p,
            )[0]

            teacher_prompt = _make_teacher_prompt(
                text,
                question,
                answer,
                include_system=cfg.include_system_prompt,
            )
            rows.append(
                {
                    "prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "seed_text": text,
                    "question": question,
                    "teacher_example_answer": answer,
                }
            )
    return Dataset.from_list(rows)

