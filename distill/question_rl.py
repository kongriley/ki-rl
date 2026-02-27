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


def _chat_to_text(tokenizer: PreTrainedTokenizerBase, messages: List[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: naive formatting
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


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
    beta: float = 0.0
    use_ref_model: bool = False
    reward_max_answer_new_tokens: int = 128


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


def _make_student_answer_prompt(question: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "Answer the user's question. Be concise and correct.",
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


def _make_teacher_answer_prompt(text: str, question: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": "Answer the user's question using only the provided context.",
        },
        {"role": "user", "content": f"Context:\n{text}\n\nQuestion: {question}"},
    ]


class QuestionGeneratorGRPOTrainer:
    """
    GRPO-based question generator trainer.

    The model generates a question given TEXT.
    Reward is negative trainer accuracy on the generated question batch.
    """

    def __init__(
        self,
        question_model: Union[str, PreTrainedModel],
        question_tokenizer: Optional[PreTrainedTokenizerBase],
        config: QuestionGRPOTrainConfig,
        reward_student_model: Union[str, PreTrainedModel],
        reward_teacher_model: Union[str, PreTrainedModel],
        reward_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.cfg = config
        torch.manual_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device) if self.cfg.device else _default_device()

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

        # Reward models (frozen): student gets question-only prompt, teacher gets text+question prompt.
        if isinstance(reward_student_model, str):
            self.reward_student_model = AutoModelForCausalLM.from_pretrained(
                reward_student_model, torch_dtype=torch.bfloat16
            )
        else:
            self.reward_student_model = reward_student_model
        if isinstance(reward_teacher_model, str):
            self.reward_teacher_model = AutoModelForCausalLM.from_pretrained(
                reward_teacher_model, torch_dtype=torch.bfloat16
            )
        else:
            self.reward_teacher_model = reward_teacher_model

        self.reward_tokenizer = reward_tokenizer or AutoTokenizer.from_pretrained(
            self.reward_student_model.config._name_or_path, padding_side="left"
        )
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

        self.reward_student_model.to(self.device)
        self.reward_teacher_model.to(self.device)
        self.reward_student_model.eval()
        self.reward_teacher_model.eval()

    @torch.no_grad()
    def _generate_student_answer_ids(self, question: str) -> torch.Tensor:
        prompt_text = _chat_to_text(self.reward_tokenizer, _make_student_answer_prompt(question))
        enc = self.reward_tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_length,
            padding=True,
        )
        enc = {k: v.to(self.reward_student_model.device) for k, v in enc.items()}
        out = self.reward_student_model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=self.cfg.reward_max_answer_new_tokens,
            pad_token_id=self.reward_tokenizer.pad_token_id,
            eos_token_id=self.reward_tokenizer.eos_token_id,
        )
        prompt_len = int(enc["attention_mask"].sum(dim=1).item())
        completion_ids = out[0, prompt_len:].detach()
        if completion_ids.numel() == 0:
            eos_id = self.reward_tokenizer.eos_token_id
            if eos_id is None:
                eos_id = self.reward_tokenizer.pad_token_id
            return torch.tensor([eos_id], device=self.reward_student_model.device)
        return completion_ids

    @torch.no_grad()
    def _per_token_logps_for_prompt_and_completion(
        self,
        model: PreTrainedModel,
        prompt_text: str,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.reward_tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_length,
            padding=True,
        )
        prompt_ids = enc["input_ids"].to(model.device)
        input_ids = torch.cat([prompt_ids, completion_ids.unsqueeze(0).to(model.device)], dim=1)
        attention_mask = torch.ones_like(input_ids, device=model.device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        token_logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_tokens = input_ids[:, 1:]
        gathered = token_logps.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

        prompt_len = prompt_ids.size(1)
        start_idx = max(prompt_len - 1, 0)
        end_idx = start_idx + completion_ids.numel()
        return gathered[:, start_idx:end_idx].squeeze(0)

    @staticmethod
    def _kl_approx_from_logp_diff(student_logps: torch.Tensor, teacher_logps: torch.Tensor) -> torch.Tensor:
        # Matches DistilTrainer's approximation based only on token log-prob differences.
        diff = student_logps - teacher_logps
        return diff + torch.exp(-diff) - 1.0

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
            rewards: List[float] = []
            for source_text, question in zip(texts_local, questions):
                if not question:
                    rewards.append(0.0)
                    continue

                completion_ids = self._generate_student_answer_ids(question)
                student_prompt = _chat_to_text(self.reward_tokenizer, _make_student_answer_prompt(question))
                teacher_prompt = _chat_to_text(
                    self.reward_tokenizer, _make_teacher_answer_prompt(source_text, question)
                )
                student_logps = self._per_token_logps_for_prompt_and_completion(
                    self.reward_student_model, student_prompt, completion_ids
                )
                teacher_logps = self._per_token_logps_for_prompt_and_completion(
                    self.reward_teacher_model, teacher_prompt, completion_ids
                )
                reward = self._kl_approx_from_logp_diff(student_logps, teacher_logps).mean()
                rewards.append(float(reward.item()))
            return rewards

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
            beta=self.cfg.beta,
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

        trainer_kwargs = dict(
            model=self.q_model,
            reward_funcs=reward_func,
            args=grpo_args,
            train_dataset=ds,
            processing_class=self.q_tok,
        )
        if not self.cfg.use_ref_model:
            grpo_trainer_sig = inspect.signature(GRPOTrainer.__init__)
            if "ref_model" in grpo_trainer_sig.parameters:
                trainer_kwargs["ref_model"] = None

        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()
        return {"status": "ok"}

