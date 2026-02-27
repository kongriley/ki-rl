import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate free-response questions (with or without context)")
    p.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--backend", choices=["hf", "openai"], default="hf",
                    help="Backend for the answering model")
    p.add_argument("--judge_model", type=str, default=None,
                    help="Judge model (defaults to --model)")
    p.add_argument("--judge_backend", choices=["hf", "openai"], default=None,
                    help="Backend for the judge (defaults to --backend)")
    p.add_argument("--data_path", type=str, default="data/wiki_20/data.json")
    p.add_argument("--questions_path", type=str,
                    default="data/wiki_20/gpt-5-mini_questions.jsonl")
    p.add_argument("--icl", action="store_true",
                    help="ICL evaluation: ask questions with providing the passage")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_data(data_path, questions_path):
    with open(data_path, "r") as f:
        articles = {item["id"]: item for item in json.load(f)}

    questions = []
    with open(questions_path, "r") as f:
        for line in f:
            q = json.loads(line)
            if q["output"].strip():
                questions.append(q)

    return articles, questions


def build_icl_prompt(article_text: str, question: str) -> str:
    return (
        "Read the following passage carefully, then answer the question.\n\n"
        f"--- Passage ---\n{article_text}\n--- End of Passage ---\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_blind_prompt(question: str) -> str:
    return f"Answer the following question.\n\nQuestion: {question}\n\nAnswer:"


JUDGE_SYSTEM = (
    "You are an impartial judge. Decide whether the student's answer is "
    "correct compared to the reference. The wording need not match exactly, "
    "but all key facts must be present and accurate. "
    "Respond with ONLY the single word 'correct' or 'incorrect'."
)


def build_judge_prompt(question: str, gold: str, answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer: {gold}\n\n"
        f"Student's answer: {answer}\n\n"
        "Verdict:"
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_hf(model, tokenizer, messages, max_new_tokens=256, temperature=0.0):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(temperature=temperature, do_sample=True)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def _openai_client():
    from openai import OpenAI
    return OpenAI()


def generate_openai(client, model, messages, max_tokens=256, temperature=0.0):
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def judge_openai(client, model, question, gold, answer):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_judge_prompt(question, gold, answer)},
    ]
    verdict = generate_openai(client, model, messages, max_tokens=8, temperature=0.0)
    return parse_verdict(verdict)


def judge_hf(model, tokenizer, question, gold, answer):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_judge_prompt(question, gold, answer)},
    ]
    verdict = generate_hf(model, tokenizer, messages, max_new_tokens=8, temperature=0.0)
    return parse_verdict(verdict)


def parse_verdict(verdict: str) -> tuple[bool, str]:
    v = verdict.lower().strip()
    is_correct = "correct" in v and "incorrect" not in v
    return is_correct, verdict


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    articles, questions = load_data(args.data_path, args.questions_path)
    print(f"Loaded {len(articles)} articles, {len(questions)} questions (non-empty answers)")

    # ---- answering model ----
    if args.backend == "openai":
        oai = _openai_client()
        answer_fn = lambda msgs: generate_openai(oai, args.model, msgs, max_tokens=args.max_new_tokens)
    else:
        print(f"Loading model: {args.model}")
        tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
        mdl.eval()
        answer_fn = lambda msgs: generate_hf(mdl, tok, msgs, max_new_tokens=args.max_new_tokens)

    # ---- judge ----
    judge_model_name = args.judge_model or args.model
    if args.judge_backend == "openai":
        oai_judge = oai if args.backend == "openai" else _openai_client()
        judge_fn = lambda q, g, a: judge_openai(oai_judge, judge_model_name, q, g, a)
    else:
        if args.backend == "hf" and judge_model_name == args.model:
            j_mdl, j_tok = mdl, tok
        else:
            print(f"Loading judge model: {judge_model_name}")
            j_tok = AutoTokenizer.from_pretrained(judge_model_name, padding_side="left")
            if j_tok.pad_token is None:
                j_tok.pad_token = j_tok.eos_token
            j_mdl = AutoModelForCausalLM.from_pretrained(judge_model_name, device_map="auto")
            j_mdl.eval()
        judge_fn = lambda q, g, a: judge_hf(j_mdl, j_tok, q, g, a)

    results = []
    correct = 0
    total = 0

    mode = "icl" if args.icl else "closed-book"
    print(f"Evaluation mode: {mode}")

    for q in tqdm(questions, desc=f"Evaluating ({mode})"):
        if args.icl:
            article = articles.get(q["id"])
            if article is None:
                print(f"Warning: no article for id={q['id']}, skipping")
                continue
            prompt = build_icl_prompt(article["text"], q["input"])
        else:
            prompt = build_blind_prompt(q["input"])

        messages = [{"role": "user", "content": prompt}]
        model_answer = answer_fn(messages)

        is_correct, verdict = judge_fn(q["input"], q["output"], model_answer)
        correct += int(is_correct)
        total += 1

        results.append({
            "id": q["id"],
            "title": q["title"],
            "question": q["input"],
            "gold_answer": q["output"],
            "model_answer": model_answer,
            "judge_verdict": verdict,
            "is_correct": is_correct,
        })

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"[{q['title']}]  {'CORRECT' if is_correct else 'WRONG'}")
            print(f"  Q: {q['input'][:150]}…")
            print(f"  Gold:  {q['output'][:150]}")
            print(f"  Model: {model_answer[:150]}")

    accuracy = correct / total if total else 0.0
    print(f"\n{'='*60}")
    print(f"[{mode}] Overall: {correct}/{total} correct  ({accuracy:.1%})")

    per_article = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_article[r["title"]]["correct"] += int(r["is_correct"])
        per_article[r["title"]]["total"] += 1

    print("\nPer-article breakdown:")
    for title, s in sorted(per_article.items()):
        a = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {title}: {s['correct']}/{s['total']} ({a:.0%})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"mode": mode, "accuracy": accuracy, "correct": correct, "total": total, "results": results}, f, indent=2)
        print(f"\nSaved to {args.output}")

    return accuracy, results


if __name__ == "__main__":
    args = parse_args()
    if args.judge_backend is None:
        args.judge_backend = args.backend
    run_evaluation(args)
