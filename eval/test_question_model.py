import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/scratch/rileyis/ki-rl/distill")
from main import load_dataset, generate_questions, _build_prompt, _build_judge_prompt


@torch.no_grad()
def answer_questions(model, tokenizer, questions, max_new_tokens=256):
    answers = []
    for q in questions:
        inputs = tokenizer(
            _build_prompt(q["question"]),
            return_tensors="pt", truncation=True, max_length=1024,
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answers.append(answer)
    return answers


@torch.no_grad()
def judge_answers(model, tokenizer, questions, answers):
    verdicts = []
    for q, a in zip(questions, answers):
        inputs = tokenizer(
            _build_judge_prompt(q["question"], a),
            return_tensors="pt", truncation=True, max_length=2048,
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=16)
        verdict = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        verdicts.append(verdict)
    return verdicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/scratch/rileyis/ki-rl/distill/out/grpo_distill/question_model_0")
    p.add_argument("--student_model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--tokenizer", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--dataset", default="/data/scratch/rileyis/ki-rl/data/wiki_20/data.json")
    p.add_argument("--num_questions", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--max_passages", type=int, default=5, help="0=all")
    p.add_argument("--full_passage", action="store_true")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading question model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    dataset = load_dataset(args.dataset)
    if args.max_passages > 0:
        dataset = dict(list(dataset.items())[:args.max_passages])

    questions = generate_questions(model, tokenizer, dataset, args.num_questions, args.temperature)

    del model
    torch.cuda.empty_cache()

    print(f"Loading student model: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=torch.bfloat16, device_map="auto")
    student.eval()

    answers = answer_questions(student, tokenizer, questions)
    verdicts = judge_answers(student, tokenizer, questions, answers)

    num_correct = sum(1 for v in verdicts if v.lower().startswith("correct"))
    print(f"\nOverall: {num_correct}/{len(verdicts)} correct ({num_correct/len(verdicts):.0%})")

    for q, a, v in zip(questions, answers, verdicts):
        print(f"\n{'='*80}")
        if args.full_passage:
            print(f"PASSAGE [{q['id']}]\n{dataset[q['id']]}")
        else:
            passage_preview = dataset[q["id"]][:200]
            print(f"PASSAGE [{q['id']}]: {passage_preview}...")
        print(f"  Q: {q['question']}")
        print(f"  A: {a}")
        print(f"  Judge: {v}")


if __name__ == "__main__":
    main()
