
import dotenv
import os
import re
import json
import time
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from oai_batch import pipeline as oai_pipeline
from olmo_batch import pipeline as olmo_pipeline

from tqdm import tqdm

def create_question_prompt(data_content: str, num_questions=10) -> str:
    """Create the prompt for question generation."""
    return f"""
    Using the following data, generate {num_questions} multiple choice question{'' if num_questions == 1 else 's'} about the data.
    These questions will be used in a separate examination in two weeks, where the students are not given the data, so be clear about the context.
    The questions should be about the data, and the answers should be in the data.

    The questions should be in the following format:

    ```
    # Question 1
    Question: [question] 
    A) [answer] 
    B) [answer] 
    C) [answer] 
    D) [answer]

    Answer: B

    # Question 2
    Question: [question] 
    A) [answer] 
    B) [answer] 
    C) [answer] 
    D) [answer]

    Answer: C
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
        
        # Find all option lines (A), B), C), D))
        option_lines = [l for l in lines if re.match(r'^[A-D]\)', l)]
        
        # Parse options
        options = {}
        for option_line in option_lines:
            match = re.match(r'^([A-D])\)\s*(.+)$', option_line)
            if match:
                letter, text = match.groups()
                options[letter] = text.strip()

        question_with_options = question_text + "\n" + "\n".join(option_lines)
        
        answer = a_line.split(':', 1)[1].strip()
        
        questions.append({
            "id": i, 
            "question_with_options": question_with_options,
            "question": question_text, 
            "options": options, 
            "correct_answer": answer, 
            "answer_text": options.get(answer, "")
        })
    
    return questions

def format_litgpt_instruct(questions_jsonl, metadata=None, randomize_options=False):
    if metadata is None:
        metadata = {}
    return [
        dict({
            "instruction": "Respond to the following question using one of the letters A, B, C, D only. Do not use any other text.",
            "input": question["question_with_options"],
            "output": question["correct_answer"]
        }, **metadata)
        for question in questions_jsonl
    ]

if __name__ == "__main__":

    print("Loading wiki 20 data...")
    with open("data/wiki_20/data.json", "r") as f:
        wiki_20 = json.load(f)

    questions_per_example = 5
    questions_per_prompt = 1
    
    # Prepare all message batches
    all_messages = []
    all_metadata = []
    for example in wiki_20:
        prompt = create_question_prompt(example['text'], questions_per_prompt)
        num_prompts = questions_per_example // questions_per_prompt
        messages = [[{"role": "user", "content": prompt}]] * num_prompts
        metadata = [{'id': example['id'], 'title': example['title']}] * num_prompts
        all_metadata.extend(metadata)
        all_messages.extend(messages)
    print(f"Generated {len(all_messages)} prompts total")

    # model_name = "allenai/OLMo-2-1124-7B-Instruct"
    # model_finetuned = AutoModelForCausalLM.from_pretrained("./out/finetune/lora/final", torch_dtype=torch.bfloat16, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # batch_size = 16  # Adjust based on your GPU memory

    # # Process in batches
    # all_questions = []
    # for i in tqdm(range(0, len(all_messages), batch_size), desc="Generating questions"):
    #     batch = all_messages[i:i+batch_size]
    #     batch_metadata = all_metadata[i:i+batch_size]
    #     generated_answers = olmo_pipeline(batch, model, tokenizer)
        
    #     for generated_answer, metadata in zip(generated_answers, batch_metadata):
    #         questions = format_litgpt_instruct(md_to_json(generated_answer), metadata=metadata)
    #         all_questions.extend(questions)
    
    # print(f"Generated {len(all_questions)} questions total")
    # with open("data/wiki_20/olmo_questions.jsonl", "w") as f:
    #     for q in all_questions:
    #         f.write(json.dumps(q) + "\n")

    responses = oai_pipeline(all_messages, model="gpt-5-mini")
    all_questions = []
    for response, metadata in zip(responses, all_metadata):
        questions = format_litgpt_instruct(md_to_json(response), metadata=metadata)
        all_questions.extend(questions)
    
    with open("data/wiki_20/gpt_5_mini_questions.jsonl", "w") as f:
        for question in all_questions:
            f.write(json.dumps(question) + "\n")

    print(f"Generated {len(all_questions)} questions total")