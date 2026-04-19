"""Continued pretraining (next-token completion) for OLMo-2-1124-7B-Instruct
on the Wikipedia passages in data/wiki_20/data.json. No chat template is
applied -- each passage's raw text is tokenized, concatenated, and packed into
fixed-length blocks for standard causal-LM training.
"""

import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--data_path", default="data/wiki_20/data.json")
    parser.add_argument("--output_dir", default="out/cpt_wiki_20")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    text_column = "text"
    dataset = dataset.select_columns([text_column])

    # Append EOS so document boundaries are preserved after packing.
    eos = tokenizer.eos_token or ""

    def tokenize(batch):
        return tokenizer(
            [t + eos for t in batch[text_column]],
            add_special_tokens=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[text_column],
        num_proc=args.num_proc,
    )

    block_size = args.block_size

    def group_texts(batch):
        concatenated = {k: sum(batch[k], []) for k in batch.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [v[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, v in concatenated.items()
        }
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    packed = tokenized.map(group_texts, batched=True, num_proc=args.num_proc)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="bfloat16",
    )
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="wandb",
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
