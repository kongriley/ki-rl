import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "distill"))

from inference import (
    build_student_prompt,
    build_teacher_prompt,
    build_judge_prompt as _shared_build_judge_prompt,
    format_instruct_user_prompt,
    parse_verdict,
    batch_generate,
    batch_judge_answers_multi_prompt,
)

from generate_questions import (
    generate_questions,
    generate_questions_openai,
    _create_question_prompt,
    _parse_question_answers,
    _is_valid_qa,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate free-response questions (with or without context)")
    p.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct")
    p.add_argument("--backend", choices=["hf", "openai"], default="hf",
                    help="Backend for the answering model")
    p.add_argument("--judge_model", type=str, default="gpt-5-mini",
                    help="Judge model (None becomes --model, default: gpt-5-mini)")
    p.add_argument("--judge_backend", choices=["hf", "openai"], default="openai",
                    help="Backend for the judge (None becomes --backend, default: openai)")
    p.add_argument("--data_path", type=str, default="data/wiki_20/data.json")
    p.add_argument("--questions_path", type=str, default=None,
                    help="Path to pre-generated questions JSONL (mutually exclusive with --question_model)")
    p.add_argument("--icl", action="store_true",
                    help="ICL evaluation: ask questions with providing the passage")
    p.add_argument("--rag", action="store_true",
                    help="RAG evaluation: retrieve the passage via embedding similarity and ask with it in context")
    p.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-8B",
                    help="Embedding model used for RAG retrieval")
    p.add_argument("--embedding_batch_size", type=int, default=2,
                    help="Batch size for embedding inference")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--answer_batch_size", type=int, default=8,
                    help="Batch size for answer generation (HF: GPU batch, OpenAI: concurrent calls)")
    p.add_argument("--batch_judge", action="store_true",
                    help="Batch judging: HF uses multi-prompt, OpenAI uses concurrent API calls")
    p.add_argument("--judge_batch_size", type=int, default=5,
                    help="Number of questions per judge batch (only with --batch_judge)")

    qg = p.add_argument_group("question generation",
                              "Generate questions on-the-fly instead of loading from --questions_path")
    qg.add_argument("--question_model", type=str, default=None,
                     help="HF hub id / local checkpoint / OpenAI model name for question generation")
    qg.add_argument("--question_backend", choices=["hf", "openai"], default="hf",
                     help="Backend for question generation (default: hf/vLLM)")
    qg.add_argument("--num_questions", type=int, default=3,
                     help="Target number of valid questions per passage")
    qg.add_argument("--num_questions_per_generation", type=int, default=5,
                     help="Questions requested per prompt")
    qg.add_argument("--temperature", type=float, default=1.2,
                     help="Sampling temperature for question generation. Will be overridden by backend.")
    qg.add_argument("--max_passages", type=int, default=None,
                     help="Limit to first N passages (default: all)")
    qg.add_argument("--save_questions", type=str, default=None,
                     help="Save generated questions to this JSONL path")

    return p.parse_args()


def load_articles(data_path):
    with open(data_path, "r") as f:
        return {item["id"]: item for item in json.load(f)}


def load_questions(questions_path):
    questions = []
    with open(questions_path, "r") as f:
        for line in f:
            q = json.loads(line)
            if q["output"].strip():
                questions.append(q)
    return questions


def load_data(data_path, questions_path):
    return load_articles(data_path), load_questions(questions_path)


# ---------------------------------------------------------------------------
# On-the-fly question generation
# ---------------------------------------------------------------------------

def _run_question_generation(args, articles):
    """Dispatch to HF/vLLM or OpenAI question generation and return normalised question list."""
    dataset = {aid: a["text"] for aid, a in articles.items()}
    if args.max_passages is not None:
        dataset = dict(list(dataset.items())[:args.max_passages])

    if args.question_backend == "openai":
        oai = _openai_client()
        raw_qs = generate_questions_openai(
            oai, args.question_model, dataset, args.num_questions,
            num_questions_per_generation=args.num_questions_per_generation,
            temperature=args.temperature,
        )
    else:
        print(f"Generating questions with: {args.question_model}")
        raw_qs = generate_questions(
            args.question_model, args.question_model, dataset,
            args.num_questions,
            num_questions_per_generation=args.num_questions_per_generation,
            temperature=args.temperature,
        )

    questions = []
    for q in raw_qs:
        questions.append({
            "id": q["id"],
            "input": q["question"],
            "output": q["answer"],
        })

    if args.save_questions:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_questions)), exist_ok=True)
        with open(args.save_questions, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        print(f"Saved {len(questions)} generated questions to {args.save_questions}")

    return questions


def build_icl_prompt(article_text: str, question: str) -> str:
    return build_teacher_prompt(question, article_text)


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

_RAG_TASK = "Given a query, retrieve relevant passages that answers the query"


def _last_token_pool(last_hidden_states, attention_mask):
    # Qwen3-Embedding uses the last non-pad token's hidden state as the embedding.
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    b = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(b, device=last_hidden_states.device), seq_lens]


def _format_query(q: str) -> str:
    return f"Instruct: {_RAG_TASK}\nQuery:{q}"


def _embed_texts(texts, model, tokenizer, batch_size, max_length=8192):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc)
        e = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        embs.append(e.float().cpu())
    return torch.cat(embs, dim=0)


def retrieve_passages(articles, questions, embedding_model_name, batch_size=2):
    """Embed every passage and question, return retrieved article id per question."""
    import gc
    print(f"Loading embedding model: {embedding_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, padding_side="left")
    model = AutoModel.from_pretrained(
        embedding_model_name, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    article_ids = list(articles.keys())
    passage_texts = [articles[aid]["text"] for aid in article_ids]
    print(f"Embedding {len(passage_texts)} passages...")
    passage_embs = _embed_texts(passage_texts, model, tokenizer, batch_size)

    query_texts = [_format_query(q["input"]) for q in questions]
    print(f"Embedding {len(query_texts)} questions...")
    query_embs = _embed_texts(query_texts, model, tokenizer, batch_size)

    sims = query_embs @ passage_embs.T  # cosine similarity (inputs are L2-normalised)
    best_idx = sims.argmax(dim=1).tolist()
    retrieved = [article_ids[i] for i in best_idx]

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return retrieved


def build_blind_prompt(question: str) -> str:
    return build_student_prompt(question)


def build_judge_prompt(question: str, gold: str, answer: str) -> str:
    return _shared_build_judge_prompt(question, gold, answer)


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
    import dotenv
    dotenv.load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _sanitize_for_openai(text: str) -> str:
    """Strip NUL bytes and other control characters that break OpenAI JSON payloads."""
    import re
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def generate_openai(client, model, messages, max_tokens=256, temperature=0.0):
    clean_messages = [
        {**m, "content": _sanitize_for_openai(m["content"])} for m in messages
    ]
    resp = client.chat.completions.create(
        model=model, messages=clean_messages,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Batched answer generation
# ---------------------------------------------------------------------------

def _build_all_prompts(questions, articles, args, mode):
    """Build prompts for all questions, filtering out missing articles for ICL.

    Returns (kept_questions, prompts) where kept_questions is the subset of
    questions that were not skipped.
    """
    kept = []
    prompts = []
    for q in questions:
        if mode == "icl":
            article = articles.get(q["id"])
            if article is None:
                print(f"Warning: no article for id={q['id']}, skipping")
                continue
            prompts.append(build_icl_prompt(article["text"], q["input"]))
        elif mode == "rag":
            article = articles.get(q["retrieved_id"])
            if article is None:
                print(f"Warning: retrieved id={q['retrieved_id']} missing, skipping")
                continue
            prompts.append(build_icl_prompt(article["text"], q["input"]))
        else:
            prompts.append(build_blind_prompt(q["input"]))
        kept.append(q)
    return kept, prompts


def batch_answer_hf(model, tokenizer, prompts, batch_size=4, max_new_tokens=256):
    """Generate answers for all prompts using GPU-batched inference."""
    formatted = [
        format_instruct_user_prompt(tokenizer, p) for p in prompts
    ]
    return batch_generate(
        model, tokenizer, formatted, use_tqdm=True,
        batch_size=batch_size, max_new_tokens=max_new_tokens, max_length=8192,
    )


def batch_answer_openai(client, model, prompts, max_tokens=256, max_workers=8):
    """Generate answers for all prompts concurrently via OpenAI API."""
    answers = [None] * len(prompts)

    def _gen(idx):
        msgs = [{"role": "user", "content": prompts[idx]}]
        return idx, generate_openai(client, model, msgs, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_gen, i) for i in range(len(prompts))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Answering (OpenAI concurrent)"):
            idx, ans = fut.result()
            answers[idx] = ans
    return answers


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def judge_openai(client, model, question, gold, answer):
    messages = [
        {"role": "user", "content": build_judge_prompt(question, gold, answer)},
    ]
    verdict = generate_openai(client, model, messages, temperature=0.0)
    return parse_verdict(verdict)


def judge_hf(model, tokenizer, question, gold, answer):
    messages = [
        {"role": "user", "content": build_judge_prompt(question, gold, answer)},
    ]
    verdict = generate_hf(model, tokenizer, messages, max_new_tokens=8, temperature=0.0)
    return parse_verdict(verdict)


# ---------------------------------------------------------------------------
# Batched judging
# ---------------------------------------------------------------------------

def judge_openai_batch(client, model, questions, golds, answers, batch_size=5):
    """Judge all triples concurrently via OpenAI API using a thread pool."""
    verdicts = [None] * len(questions)

    def _judge_single(idx):
        return idx, judge_openai(client, model, questions[idx], golds[idx], answers[idx])

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(_judge_single, i) for i in range(len(questions))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging (OpenAI concurrent)"):
            idx, result = fut.result()
            verdicts[idx] = result
    return verdicts


def judge_hf_batch(model, tokenizer, questions, golds, answers, batch_size=5):
    """Judge all triples using multi-prompt batching (multiple triples per prompt)."""
    return batch_judge_answers_multi_prompt(
        model, tokenizer, questions, golds, answers, group_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    articles = load_articles(args.data_path)

    if args.question_model:
        questions = _run_question_generation(args, articles)
    else:
        questions = load_questions(args.questions_path)

    print(f"Loaded {len(articles)} articles, {len(questions)} questions (non-empty answers)")

    if args.icl and args.rag:
        raise SystemExit("--icl and --rag are mutually exclusive")
    if args.rag:
        mode = "rag"
    elif args.icl:
        mode = "icl"
    else:
        mode = "closed-book"
    print(f"Evaluation mode: {mode}")

    # ---- RAG retrieval (embed passages + questions, pick best match) ----
    if mode == "rag":
        retrieved_ids = retrieve_passages(
            articles, questions, args.embedding_model,
            batch_size=args.embedding_batch_size,
        )
        retrieval_hits = 0
        for q, rid in zip(questions, retrieved_ids):
            q["retrieved_id"] = rid
            retrieval_hits += int(rid == q["id"])
        retrieval_acc = retrieval_hits / len(questions) if questions else 0.0
        print(f"Retrieval accuracy: {retrieval_hits}/{len(questions)} ({retrieval_acc:.1%})")

    # ---- build prompts ----
    kept_questions, prompts = _build_all_prompts(questions, articles, args, mode)

    # ---- generate answers (batched) ----
    if args.backend == "openai":
        oai = _openai_client()
        model_answers = batch_answer_openai(
            oai, args.model, prompts,
            max_tokens=args.max_new_tokens, max_workers=args.answer_batch_size,
        )
    else:
        print(f"Loading model: {args.model}")
        tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
        mdl.eval()
        print(f"Generating answers for {len(prompts)} questions in batches of {args.answer_batch_size}...")
        model_answers = batch_answer_hf(
            mdl, tok, prompts,
            batch_size=args.answer_batch_size, max_new_tokens=args.max_new_tokens,
        )

    # ---- judge ----
    judge_model_name = args.judge_model or args.model
    all_qs = [q["input"] for q in kept_questions]
    all_golds = [q["output"] for q in kept_questions]

    print(f"Judge model: {judge_model_name}")

    if args.batch_judge:
        if args.judge_backend == "openai":
            oai_judge = oai if args.backend == "openai" else _openai_client()
            verdicts = judge_openai_batch(
                oai_judge, judge_model_name, all_qs, all_golds, model_answers,
                batch_size=args.judge_batch_size,
            )
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
            print(f"Judging {len(all_qs)} answers in batches of {args.judge_batch_size}...")
            verdicts = judge_hf_batch(
                j_mdl, j_tok, all_qs, all_golds, model_answers,
                batch_size=args.judge_batch_size,
            )
    else:
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
        verdicts = []
        for q_text, gold, ma in tqdm(
            zip(all_qs, all_golds, model_answers), total=len(all_qs), desc="Judging",
        ):
            verdicts.append(judge_fn(q_text, gold, ma))

    # ---- assemble results ----
    results = []
    correct = 0
    retrieval_correct_total = 0
    for q, ma, (is_correct, verdict) in zip(kept_questions, model_answers, verdicts):
        correct += int(is_correct)
        row = {
            "id": q["id"],
            "question": q["input"],
            "gold_answer": q["output"],
            "model_answer": ma,
            "judge_verdict": verdict,
            "is_correct": is_correct,
        }
        if mode == "rag":
            row["retrieved_id"] = q["retrieved_id"]
            row["retrieval_correct"] = (q["retrieved_id"] == q["id"])
            retrieval_correct_total += int(row["retrieval_correct"])
        results.append(row)

        if args.verbose:
            print(f"\n{'='*60}")
            tag = f"[{q['id']}]"
            if mode == "rag":
                tag += f" (retrieved {q['retrieved_id']} {'OK' if row['retrieval_correct'] else 'MISS'})"
            print(f"{tag}  {'CORRECT' if is_correct else 'WRONG'}")
            print(f"  Q: {q['input'][:150]}…")
            print(f"  Gold:  {q['output'][:150]}")
            print(f"  Model: {ma[:150]}")

    total = len(results)
    accuracy = correct / total if total else 0.0
    print(f"\n{'='*60}")
    print(f"[{mode}] Overall: {correct}/{total} correct  ({accuracy:.1%})")
    if mode == "rag":
        ret_acc = retrieval_correct_total / total if total else 0.0
        print(f"[{mode}] Retrieval: {retrieval_correct_total}/{total} correct  ({ret_acc:.1%})")

    per_passage = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_passage[r["id"]]["correct"] += int(r["is_correct"])
        per_passage[r["id"]]["total"] += 1

    print("\nPer-passage breakdown:")
    for pid, s in sorted(per_passage.items()):
        a = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {pid}: {s['correct']}/{s['total']} ({a:.0%})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        payload = {"mode": mode, "accuracy": accuracy, "correct": correct, "total": total, "results": results}
        if mode == "rag":
            payload["retrieval_accuracy"] = retrieval_correct_total / total if total else 0.0
            payload["retrieval_correct"] = retrieval_correct_total
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {args.output}")

    return accuracy, results


if __name__ == "__main__":
    args = parse_args()
    if args.judge_backend is None:
        args.judge_backend = args.backend
    if (args.question_model is None and args.questions_path is None) or (args.question_model and args.questions_path):
        raise SystemExit("Specify either --question_model or --questions_path, not both.")
    run_evaluation(args)
