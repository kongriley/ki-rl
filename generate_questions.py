
import dotenv
import os
import re
import json
import time
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

import openai

dotenv.load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_question_prompt(data_content: str, num_questions=10) -> str:
    """Create the prompt for question generation."""
    return f"""
    Using the following data, generate {num_questions} multiple choice questions about the data.
    These questions will be used in a separate examination in two weeks, where the students are not given the data, so be clear about when the events happened.
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
    ```

    # Data
    {data_content}
    """

def prepare_batch_requests(wiki_data, num_questions=10):
    """Prepare batch requests for OpenAI API."""
    batch_requests = []
    
    for i, example in enumerate(wiki_data):
        text = example['text']
        prompt = create_question_prompt(text, num_questions)
        
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",  # Using a standard model instead of gpt-5-nano
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000
            }
        }
        batch_requests.append(request)
    
    return batch_requests

def create_batch_file(batch_requests, filename="batch_requests.jsonl"):
    """Create a JSONL file for batch processing."""
    with open(filename, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    return filename

def submit_batch(batch_file_path, max_retries=3):
    """Submit batch file to OpenAI and return batch job ID."""
    for attempt in range(max_retries):
        try:
            with open(batch_file_path, 'rb') as f:
                batch_input_file = client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            batch_job = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            return batch_job.id
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

def wait_for_batch_completion(batch_id, check_interval=30):
    """Wait for batch job to complete and return the result."""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        batch_job = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_job.status}")
        
        if batch_job.status == "completed":
            return batch_job
        elif batch_job.status in ["failed", "expired", "cancelled"]:
            raise Exception(f"Batch job failed with status: {batch_job.status}")
        
        time.sleep(check_interval)

def download_batch_results(batch_job, max_retries=3):
    """Download and parse batch results."""
    for attempt in range(max_retries):
        try:
            result_file_id = batch_job.output_file_id
            if not result_file_id:
                raise Exception("No output file ID found in batch job")
            
            result = client.files.content(result_file_id)
            
            results = []
            for line in result.text.strip().split('\n'):
                if line:
                    results.append(json.loads(line))
            
            return results
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

def process_batch_results(batch_results):
    """Process batch results and convert to the desired format."""
    all_questions = []
    failed_requests = []
    
    for result in batch_results:
        try:
            if result.get('response') and result['response'].get('body'):
                response_content = result['response']['body']['choices'][0]['message']['content']
                questions = format_litgpt_instruct(md_to_json(response_content))
                all_questions.extend(questions)
            else:
                failed_requests.append(result.get('custom_id', 'unknown'))
        except Exception as e:
            print(f"Error processing result {result.get('custom_id', 'unknown')}: {e}")
            failed_requests.append(result.get('custom_id', 'unknown'))
    
    if failed_requests:
        print(f"Warning: {len(failed_requests)} requests failed: {failed_requests}")
    
    return all_questions

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

def format_litgpt_instruct(questions_jsonl, randomize_options=False):
    return [
        {
            "instruction": "Respond to the following question using one of the letters A, B, C, D only. Do not use any other text.",
            "input": question["question_with_options"],
            "output": question["correct_answer"]
        }
        for question in questions_jsonl
    ]

if __name__ == "__main__":

    wiki_1k = load_from_disk("data/wiki_1k")
    
    # Prepare batch requests
    print("Preparing batch requests...")
    batch_requests = prepare_batch_requests(wiki_1k)
    
    # Create batch file
    batch_file = create_batch_file(batch_requests, "data/wiki_1k_batch_requests.jsonl")
    print(f"Created batch file: {batch_file}")
    
    # Submit batch job
    print("Submitting batch job...")
    batch_id = submit_batch(batch_file)
    print(f"Batch job submitted with ID: {batch_id}")
    
    # Wait for completion
    completed_batch = wait_for_batch_completion(batch_id)

    # batch_id = "batch_68e9291e8a708190a13efa53ee4ce549"
    # completed_batch = client.batches.retrieve(batch_id) 
    
    # Download and process results
    print("Downloading batch results...")
    batch_results = download_batch_results(completed_batch)
    
    print("Processing batch results...")
    question_ds = process_batch_results(batch_results)
    
    # Save results
    print(f"Generated {len(question_ds)} questions total")
    with open("data/wiki_1k_questions.jsonl", "w") as f:
        for q in question_ds:
            f.write(json.dumps(q) + "\n")
    
    print("Questions saved to data/wiki_1k_questions.jsonl")
    
    # Clean up batch file
    if 'batch_file' in locals() and os.path.exists(batch_file):
        os.remove(batch_file)
        print("Cleaned up temporary batch file")
