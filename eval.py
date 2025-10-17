from litgpt import LLM
import openai
import dotenv
import os
import json
from tqdm import tqdm

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = LLM.load("allenai/OLMo-2-1124-7B-Instruct")
llm_finetuned = LLM.load("./out/finetune/lora/final")

def eval_message(question_with_options, context_data=None) -> str:
    message = f"Respond to the following question using one of the letters A, B, C, D only. Do not use any other text. \n{question_with_options}"
    if context_data:
        message += f"\n\nYou can use the following context data to answer the question: {context_data}"
    return message

def eval_questions_with_litgpt(json_content, llm, context_data=None):
    results = []
    for question in tqdm(json_content):
        messages = [
            {"role": "user", "content": eval_message(question['input'], context_data)},
        ]
        generated_answer = llm.generate(messages)
        # Retrieve the first character
        generated_answer = generated_answer[0]

        print(f"Question: {question['input']}")
        print(f"Generated answer: {generated_answer}")
        print(f"Correct answer: {question['output']}")
        print("--------------------------------")

        results.append({"question": question["input"], "generated_answer": generated_answer, "correct_answer": question["output"], "is_correct": generated_answer == question["output"]})
    return results

def eval_questions_with_openai(json_content, model="gpt-4o", context_data=None):
    results = []
    for question in json_content:
        response = client.responses.create(
            model=model,
            input=eval_message(question['question_with_options'], context_data),
        )
        generated_answer = response.output_text[0]

        print(f"Question: {question['question']}")
        print(f"Generated answer: {generated_answer}")
        print(f"Correct answer: {question['correct_answer']}")
        print("--------------------------------")

        results.append({"question": question["question"], "generated_answer": generated_answer, "correct_answer": question["correct_answer"], "is_correct": generated_answer == question["correct_answer"]})
    return results

if __name__ == "__main__":
    with open("data/wiki_1k_questions.jsonl", "r") as f:
        json_content = [json.loads(l) for l in f]

    # Sample 100 questions
    json_content = json_content[:100]
    
    print("Evaluating with OLMo-2-1124-7B-Instruct")
    pipeline_results = eval_questions_with_litgpt(json_content, llm)
    print(f"OLMo Blind Accuracy: {sum(result['is_correct'] for result in pipeline_results) / len(pipeline_results)}")

    finetune_pipeline_results = eval_questions_with_litgpt(json_content, llm_finetuned)
    print(f"Finetuned Accuracy: {sum(result['is_correct'] for result in finetune_pipeline_results) / len(finetune_pipeline_results)}")