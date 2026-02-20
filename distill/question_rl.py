import re
import inspect
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trl import GRPOConfig, GRPOTrainer


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_question_prompt(text: str) -> List[dict]:
    # Conversational prompt so it works with chat template tokenizers too.
    return [
        {
            "role": "system",
            "content": (
                "You generate exactly one high-quality, specific question about the given text. "
                "Only output the question; no preamble, no numbering."
            ),
        },
        {"role": "user", "content": f"<TEXT>\n{text}\n</TEXT>"},
    ]


def _make_judge_prompt(text: str, question: str) -> str:
    # We keep this as plain text and parse a float from the output.
    # Expected: a single line "SCORE: <float between 0 and 1>"
    return (
        "You are a strict evaluator of question quality.\n"
        "Given a TEXT and a QUESTION, output ONLY one line:\n"
        "SCORE: <number between 0 and 1>\n\n"
        "Criteria: the question is answerable from the text, specific, non-trivial, and unambiguous.\n\n"
        f"TEXT:\n{text}\n\n"
        f"QUESTION:\n{question}\n"
    )


_SCORE_RE = re.compile(r"score\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_score(text: str) -> float:
    m = _SCORE_RE.search(text)
    if not m:
        # Conservative default when judge doesn't follow format
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def _chat_to_text(tokenizer: PreTrainedTokenizerBase, messages: List[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: naive formatting
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def _make_teacher_answer_prompt(text: str, question: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "Answer the user's question using ONLY the provided text. Be concise and correct.",
        },
        {"role": "user", "content": f"<TEXT>\n{text}\n</TEXT>\n\nQuestion: {question}"},
    ]


def _make_student_answer_prompt(question: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "Answer the user's question. Be concise and correct.",
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


def _make_answer_judge_prompt(text: str, question: str, reference_answer: str, student_answer: str) -> str:
    return (
        "You are a strict evaluator of answer correctness.\n"
        "Given a TEXT, a QUESTION, a REFERENCE_ANSWER (correct), and a STUDENT_ANSWER, output ONLY one line:\n"
        "SCORE: <number between 0 and 1>\n\n"
        "Interpretation: 1.0 = student answer is fully correct and supported by the text; "
        "0.0 = incorrect / unsupported / refusal.\n\n"
        f"TEXT:\n{text}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE_ANSWER:\n{reference_answer}\n\n"
        f"STUDENT_ANSWER:\n{student_answer}\n"
    )


@torch.no_grad()
def _generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[List[dict]],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    texts = [_chat_to_text(tokenizer, p) for p in prompts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].size(1) :]
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [t.strip() for t in decoded]


@torch.no_grad()
def sample_questions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[torch.Tensor], List[int]]:
    """
    Returns:
      - questions: decoded questions (str)
      - sequences: full token ids (prompt+question) per item (unpadded)
      - prompt_lens: prompt length per item
    """
    prompts = [_chat_to_text(tokenizer, _make_question_prompt(t)) for t in texts]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

    sequences = []
    questions = []
    for i in range(out.size(0)):
        seq = out[i, :].detach()
        sequences.append(seq)
        q_ids = seq[prompt_lens[i] :]
        q_text = tokenizer.decode(q_ids, skip_special_tokens=True).strip()
        # keep only first line as "question"
        q_text = q_text.splitlines()[0].strip()
        questions.append(q_text)
    return questions, sequences, prompt_lens


def _batched_sequence_logprobs(
    model: PreTrainedModel,
    sequences: List[torch.Tensor],
    prompt_lens: List[int],
    pad_token_id: int,
) -> torch.Tensor:
    """
    Computes sum log p(question_tokens | prompt) for each sequence.
    Returns shape (B,) tensor.
    """
    device = model.device
    max_len = max(s.numel() for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
    for i, s in enumerate(sequences):
        input_ids[i, : s.numel()] = s.to(device)
        attention_mask[i, : s.numel()] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    # next-token prediction: logits[t] predicts token[t+1]
    logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]

    # Build a mask for "question" tokens only (excluding prompt tokens)
    mask = torch.zeros_like(target, dtype=torch.float32)  # (B, L-1)
    for i, p_len in enumerate(prompt_lens):
        start = max(p_len - 1, 0)  # shift because of next-token alignment
        end = (attention_mask[i].sum().item() - 1)  # last token has no target
        if end > start:
            mask[i, start:end] = 1.0

    selected = logps.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(dim=1)


@torch.no_grad()
def judge_question_rewards(
    judge_model: PreTrainedModel,
    judge_tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    questions: Sequence[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
) -> List[float]:
    prompts = [_make_judge_prompt(t, q) for t, q in zip(texts, questions)]
    enc = judge_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(judge_model.device) for k, v in enc.items()}
    out = judge_model.generate(
        **enc,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=judge_tokenizer.pad_token_id,
        eos_token_id=judge_tokenizer.eos_token_id,
    )
    # Decode only generated tail for parsing
    gen = out[:, enc["input_ids"].size(1) :]
    texts_out = judge_tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [_parse_score(t) for t in texts_out]


@torch.no_grad()
def judge_answer_correctness_rewards(
    judge_model: PreTrainedModel,
    judge_tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    questions: Sequence[str],
    reference_answers: Sequence[str],
    student_answers: Sequence[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
) -> List[float]:
    prompts = [
        _make_answer_judge_prompt(t, q, ra, sa)
        for t, q, ra, sa in zip(texts, questions, reference_answers, student_answers)
    ]
    enc = judge_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(judge_model.device) for k, v in enc.items()}
    out = judge_model.generate(
        **enc,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=judge_tokenizer.pad_token_id,
        eos_token_id=judge_tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].size(1) :]
    texts_out = judge_tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [_parse_score(t) for t in texts_out]


@dataclass
class QuestionGRPOTrainConfig:
    # Core GRPO training
    output_dir: str = "qg_grpo_out"
    learning_rate: float = 5e-6
    max_steps: int = 200
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    # TRL GRPOConfig expects generation_batch_size (total number of completions generated per step)
    # to be divisible by num_generations. If None, we set it to
    # per_device_train_batch_size * num_generations (i.e., num_generations completions per prompt).
    generation_batch_size: Optional[int] = None
    max_prompt_length: int = 1024
    max_question_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    logging_steps: int = 10
    seed: int = 0
    device: Optional[str] = None

    # Reward models/settings
    judge_model: str = ""
    student_model: Optional[str] = None
    teacher_answer_model: Optional[str] = None  # if None, defaults to judge_model
    judge_max_new_tokens: int = 64
    judge_temperature: float = 0.0
    student_answer_max_new_tokens: int = 256
    teacher_answer_max_new_tokens: int = 256
    answer_judge_max_new_tokens: int = 64
    answer_judge_temperature: float = 0.0
    use_quality_gate: bool = True
    quality_gate_min: float = 0.3
    correctness_threshold: float = 0.8  # logging only


def _completion_to_question_text(completion) -> str:
    # GRPOTrainer may pass completions either as plain strings, or as a conversational list-of-messages.
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        text = str(completion[-1].get("content", ""))
    else:
        text = str(completion)
    text = text.strip()
    # keep only first line as "question"
    return text.splitlines()[0].strip()


class QuestionGeneratorGRPOTrainer:
    """
    GRPO-based question generator trainer.

    The model generates a question given TEXT. Reward encourages:
      - question quality/answerability from the text
      - (optionally) student failure (teacher text+q vs student q-only)

    Reward used when student_model is set:
        reward = quality * (1 - correctness)
    Otherwise:
        reward = quality
    """

    def __init__(
        self,
        question_model: Union[str, PreTrainedModel],
        question_tokenizer: Optional[PreTrainedTokenizerBase],
        config: QuestionGRPOTrainConfig,
    ):
        self.cfg = config
        torch.manual_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device) if self.cfg.device else _default_device()

        if GRPOTrainer is None or GRPOConfig is None:  # pragma: no cover
            raise ImportError(
                "TRL GRPO is not available. Please install/upgrade `trl` to a version that provides "
                "`GRPOTrainer` and `GRPOConfig`."
            )

        # Question model (policy)
        if isinstance(question_model, str):
            self.q_model = AutoModelForCausalLM.from_pretrained(question_model, torch_dtype=torch.bfloat16)
        else:
            self.q_model = question_model
        self.q_model.to(self.device)
        self.q_model.train()

        self.q_tok = question_tokenizer or AutoTokenizer.from_pretrained(
            self.q_model.config._name_or_path, padding_side="left"
        )
        if self.q_tok.pad_token is None:
            self.q_tok.pad_token = self.q_tok.eos_token

        # Judge + optional student/teacher for correctness scoring
        if not self.cfg.judge_model:
            raise ValueError("QuestionGRPOTrainConfig.judge_model must be set to a model name/path.")
        self.j_model = AutoModelForCausalLM.from_pretrained(self.cfg.judge_model, torch_dtype=torch.bfloat16).to(self.device)
        self.j_model.eval()
        self.j_tok = AutoTokenizer.from_pretrained(self.cfg.judge_model, padding_side="left")
        if self.j_tok.pad_token is None:
            self.j_tok.pad_token = self.j_tok.eos_token

        self.s_model = None
        self.s_tok = None
        self.t_model = None
        self.t_tok = None
        if self.cfg.student_model:
            self.s_model = AutoModelForCausalLM.from_pretrained(self.cfg.student_model, torch_dtype=torch.bfloat16).to(
                self.device
            )
            self.s_model.eval()
            self.s_tok = AutoTokenizer.from_pretrained(self.cfg.student_model, padding_side="left")
            if self.s_tok.pad_token is None:
                self.s_tok.pad_token = self.s_tok.eos_token

            teacher_name = self.cfg.teacher_answer_model or self.cfg.judge_model
            self.t_model = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.bfloat16).to(self.device)
            self.t_model.eval()
            self.t_tok = AutoTokenizer.from_pretrained(teacher_name, padding_side="left")
            if self.t_tok.pad_token is None:
                self.t_tok.pad_token = self.t_tok.eos_token

    def train(self, texts: Sequence[str]) -> dict:
        if len(texts) == 0:
            raise ValueError("Need at least one training text for question GRPO.")

        prompts = [_make_question_prompt(t) for t in texts]
        ds = Dataset.from_list([{"prompt": p, "text": t} for p, t in zip(prompts, texts)])

        def reward_func(prompts, completions, text=None, **kwargs):
            # `text` comes from the dataset column (list[str])
            if text is None:
                # Fallback: if trainer doesn't pass columns as kwargs
                texts_local = ["" for _ in range(len(completions))]
            else:
                texts_local = list(text)

            questions = [_completion_to_question_text(c) for c in completions]

            quality = judge_question_rewards(
                self.j_model,
                self.j_tok,
                texts_local,
                questions,
                max_prompt_length=self.cfg.max_prompt_length,
                max_new_tokens=self.cfg.judge_max_new_tokens,
                temperature=self.cfg.judge_temperature,
            )

            if self.s_model is None or self.t_model is None:
                return [float(q) for q in quality]

            # Teacher answers using text+question; student answers using question-only
            teacher_prompts = [_make_teacher_answer_prompt(t, q) for t, q in zip(texts_local, questions)]
            teacher_answers = _generate_batch(
                self.t_model,
                self.t_tok,
                teacher_prompts,
                max_prompt_length=self.cfg.max_prompt_length,
                max_new_tokens=self.cfg.teacher_answer_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            student_prompts = [_make_student_answer_prompt(q) for q in questions]
            student_answers = _generate_batch(
                self.s_model,
                self.s_tok,
                student_prompts,
                max_prompt_length=self.cfg.max_prompt_length,
                max_new_tokens=self.cfg.student_answer_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )

            correctness: List[float] = [0.0 for _ in range(len(questions))]
            if self.cfg.use_quality_gate:
                gated = [
                    (t, q, ta, sa, i)
                    for i, (t, q, ta, sa, qual) in enumerate(
                        zip(texts_local, questions, teacher_answers, student_answers, quality)
                    )
                    if qual >= self.cfg.quality_gate_min
                ]
                if len(gated) > 0:
                    gt, gq, gta, gsa, gi = zip(*gated)
                    judged = judge_answer_correctness_rewards(
                        self.j_model,
                        self.j_tok,
                        list(gt),
                        list(gq),
                        list(gta),
                        list(gsa),
                        max_prompt_length=self.cfg.max_prompt_length,
                        max_new_tokens=self.cfg.answer_judge_max_new_tokens,
                        temperature=self.cfg.answer_judge_temperature,
                    )
                    for idx, val in zip(gi, judged):
                        correctness[idx] = float(val)
            else:
                correctness = [
                    float(x)
                    for x in judge_answer_correctness_rewards(
                        self.j_model,
                        self.j_tok,
                        texts_local,
                        questions,
                        teacher_answers,
                        student_answers,
                        max_prompt_length=self.cfg.max_prompt_length,
                        max_new_tokens=self.cfg.answer_judge_max_new_tokens,
                        temperature=self.cfg.answer_judge_temperature,
                    )
                ]

            # reward = quality * (1 - correctness)
            return [float(qual) * float(1.0 - corr) for qual, corr in zip(quality, correctness)]

        grpo_kwargs = dict(
            output_dir=self.cfg.output_dir,
            learning_rate=self.cfg.learning_rate,
            max_steps=self.cfg.max_steps,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            num_generations=self.cfg.num_generations,
            max_prompt_length=self.cfg.max_prompt_length,
            max_completion_length=self.cfg.max_question_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            logging_steps=self.cfg.logging_steps,
            remove_unused_columns=False,
            seed=self.cfg.seed,
            report_to="none",
        )

        # Some TRL versions require generation_batch_size (total completions generated per step)
        # to be divisible by num_generations. We set a safe default and auto-fix invalid values.
        try:
            grpo_sig = inspect.signature(GRPOConfig.__init__)
            supports_generation_batch_size = "generation_batch_size" in grpo_sig.parameters
        except Exception:  # pragma: no cover
            supports_generation_batch_size = False

        if supports_generation_batch_size:
            gen_bs = self.cfg.generation_batch_size
            if gen_bs is None:
                gen_bs = int(self.cfg.per_device_train_batch_size) * int(self.cfg.num_generations)
            if gen_bs <= 0:
                raise ValueError(f"generation_batch_size must be positive, got {gen_bs}.")
            if gen_bs % int(self.cfg.num_generations) != 0:
                fixed = int(math.ceil(gen_bs / int(self.cfg.num_generations)) * int(self.cfg.num_generations))
                warnings.warn(
                    f"generation_batch_size ({gen_bs}) must be divisible by num_generations "
                    f"({self.cfg.num_generations}); adjusting to {fixed}.",
                    RuntimeWarning,
                )
                gen_bs = fixed
            grpo_kwargs["generation_batch_size"] = int(gen_bs)

        grpo_args = GRPOConfig(**grpo_kwargs)

        trainer = GRPOTrainer(
            model=self.q_model,
            reward_funcs=reward_func,
            args=grpo_args,
            train_dataset=ds,
            processing_class=self.q_tok,
        )
        trainer.train()
        return {"status": "ok"}

