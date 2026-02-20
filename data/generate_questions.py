import argparse
import dotenv
import os
import re
import json
import time
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate questions from wiki data")
    parser.add_argument("--backend", choices=["openai", "olmo"], default="openai",
                        help="LLM backend to use (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: per-backend default)")
    parser.add_argument("--output-dir", default="data/wiki_20",
                        help="Output directory")
    parser.add_argument("--questions-per-example", type=int, default=5)
    parser.add_argument("--questions-per-prompt", type=int, default=1)
    return parser.parse_args()

def create_question_prompt(data_content: str, num_questions=10) -> str:
    """Create the prompt for question generation."""
    return f"""
    Using the following data, generate {num_questions} free-response question{'' if num_questions == 1 else 's'} about the data.
    These questions will be used in a separate examination in two weeks, where the students are not given the data, so be clear about the context, but do not explicitly reference the data in the questions.
    The answers should be answerable solely based on the information in the data.

    The questions should be in the following format:

    ```
    # Question 1
    Question: [question]
    Answer: [answer]

    ```

    # Data
    {data_content}
    """

def md_to_json(md_content):
    """Convert markdown quiz format to JSON."""
    questions = []
    blocks = re.split(r'^# Question \d+$', md_content, flags=re.MULTILINE)[1:]
    
    for i, block in enumerate(blocks, 1):
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2: continue
        
        # Find the question line and answer line
        q_line = next((l for l in lines if l.startswith('Question:')), None)
        a_line = next((l for l in lines if l.startswith('Answer:')), None)
        if not q_line or not a_line: continue
        
        # Extract the question text (everything after "Question: ")
        question_text = q_line[9:].strip()
        
        
        answer = a_line.split(':', 1)[1].strip()
        
        questions.append({
            "id": i, 
            "question": question_text, 
            "answer": answer, 
        })
    
    return questions

def format_litgpt_instruct(questions_jsonl, metadata=None, randomize_options=False):
    if metadata is None:
        metadata = {}
    return [
        dict({
            "instruction": "Respond to the following question.",
            "input": question["question"],
            "output": question["answer"]
        }, **metadata)
        for question in questions_jsonl
    ]

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "olmo": "allenai/OLMo-2-1124-13B-Instruct",
}

def run_openai(all_messages, model):
    from oai_batch import pipeline as oai_pipeline
    return oai_pipeline(all_messages, model=model)

def run_olmo(all_messages, model):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from olmo_batch import pipeline as olmo_pipeline
    print(f"Loading OLMo model {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    return olmo_pipeline(all_messages, hf_model, tokenizer)

if __name__ == "__main__":
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading wiki 20 data...")
    with open(os.path.join(output_dir, "data.json"), "r") as f:
        wiki_20 = json.load(f)

    all_messages = []
    all_metadata = []
    for example in wiki_20:
        prompt = create_question_prompt(example['text'], args.questions_per_prompt)
        num_prompts = args.questions_per_example // args.questions_per_prompt
        messages = [[{"role": "user", "content": prompt}]] * num_prompts
        metadata = [{'id': example['id'], 'title': example['title']}] * num_prompts
        all_metadata.extend(metadata)
        all_messages.extend(messages)
    print(f"Generated {len(all_messages)} prompts total")

    runners = {"openai": run_openai, "olmo": run_olmo}
    responses = runners[args.backend](all_messages, model)

    all_questions = []
    for response, metadata in zip(responses, all_metadata):
        questions = format_litgpt_instruct(md_to_json(response), metadata=metadata)
        all_questions.extend(questions)

    safe_model_name = model.replace("/", "_")
    out_path = os.path.join(output_dir, f"{safe_model_name}_questions.jsonl")
    with open(out_path, "w") as f:
        for question in all_questions:
            f.write(json.dumps(question) + "\n")

    print(f"Generated {len(all_questions)} questions -> {out_path}")