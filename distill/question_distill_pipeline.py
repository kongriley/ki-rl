from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

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

    # Prompt formatting
    include_system_prompt: bool = True


def _ensure_pad(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _make_question_answer_prompt(question: str, *, include_system: bool) -> List[dict]:
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


def _make_teacher_question_answer_prompt(text: str, question: str, *, include_system: bool) -> List[dict]:
    msgs: List[dict] = []
    if include_system:
        msgs.append(
            {
                "role": "system",
                "content": "Answer the user's question using only the provided context.",
            }
        )
    msgs.append({"role": "user", "content": f"Context:\n{text}\n\nQuestion: {question}"})
    return msgs


def build_question_distillation_dataset(
    *,
    texts: Sequence[str],
    question_model: PreTrainedModel,
    question_tokenizer: PreTrainedTokenizerBase,
    cfg: QuestionThenDistillConfig,
) -> Dataset:
    _ensure_pad(question_tokenizer)

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
            prompt = _make_question_answer_prompt(question, include_system=cfg.include_system_prompt)
            teacher_prompt = _make_teacher_question_answer_prompt(
                text, question, include_system=cfg.include_system_prompt
            )
            rows.append(
                {
                    # DistilTrainer expects both keys.
                    # Student sees question-only; teacher sees context+question.
                    "prompt": prompt,
                    "teacher_prompt": teacher_prompt,
                    "seed_text": text,
                    "question": question,
                }
            )
    return Dataset.from_list(rows)

