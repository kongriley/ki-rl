"""Generate questions with a question model, then evaluate a student closed-book."""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "distill"))
from main import load_dataset, generate_questions

from inference import evaluate_qa


def main():
    p = argparse.ArgumentParser(
        description="Use the question model to generate Q&A from passages, then score the student "
        "closed-book against reference answers (same judge setup as test_question_model)."
    )
    p.add_argument("--student_model", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--question_model", default="distill/out/grpo_distill/question_model_0")
    p.add_argument("--tokenizer", default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--dataset", default="data/wiki_20/data.json")
    p.add_argument("--num_questions", type=int, default=3,
                   help="Target number of valid questions per passage (see generate_questions)")
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--max_passages", type=int, default=None,
                   help="Limit to first N passages (default: use all)")
    p.add_argument("--full_passage", action="store_true")
    p.add_argument("--output", type=str, default=None,
                   help="Save results to this JSON file")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset)
    if args.max_passages is not None:
        dataset = dict(list(dataset.items())[:args.max_passages])

    print(f"Generating questions with: {args.question_model}")
    questions = generate_questions(
        args.question_model,
        args.tokenizer,
        dataset,
        args.num_questions,
        temperature=args.temperature,
    )

    print(f"Loading student: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    student.eval()

    summary = evaluate_qa(
        student,
        student,
        tokenizer,
        [q["question"] for q in questions],
        [q["answer"] for q in questions],
        ids=[q["id"] for q in questions],
    )

    for r, q in zip(summary["results"], questions):
        print(f"\n{'='*80}")
        if args.full_passage:
            print(f"PASSAGE [{q['id']}]\n{dataset[q['id']]}")
        else:
            passage_preview = dataset[q["id"]][:200]
            print(f"PASSAGE [{q['id']}]: {passage_preview}...")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['student_answer']}")
        print(f"  Judge: {r['verdict']}")

    print(
        f"\nOverall: {summary['correct']}/{summary['total']} correct "
        f"({summary['accuracy']:.0%})"
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
