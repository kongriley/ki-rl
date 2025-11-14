from typing import List
import json
import time
import openai
import dotenv
import os

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def prepare_batch_requests(data: List, model="gpt-4o-mini"):
    """Prepare batch requests for OpenAI API."""
    batch_requests = []
    
    for i, messages in enumerate(data):
        
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,  # Using a standard model instead of gpt-5-nano
                "messages": messages,
                "max_completion_tokens": 4096
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
    responses = []
    failed_requests = []
    
    for result in batch_results:
        try:
            if result.get('response') and result['response'].get('body'):
                response_content = result['response']['body']['choices'][0]['message']['content']
                responses.append(response_content)
            else:
                failed_requests.append(result.get('custom_id', 'unknown'))
        except Exception as e:
            print(f"Error processing result {result.get('custom_id', 'unknown')}: {e}")
            failed_requests.append(result.get('custom_id', 'unknown'))
    
    if failed_requests:
        print(f"Warning: {len(failed_requests)} requests failed: {failed_requests}")
    
    return responses

def pipeline(data, model="gpt-5-mini"):
    batch_requests = prepare_batch_requests(data, model=model)
    batch_file_path = create_batch_file(batch_requests, filename=f"data/batch_requests_{model}.jsonl")
    batch_id = submit_batch(batch_file_path)
    batch_job = wait_for_batch_completion(batch_id)
    batch_results = download_batch_results(batch_job)
    responses = process_batch_results(batch_results)
    return responses