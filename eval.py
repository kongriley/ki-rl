from litgpt import LLM
import random
import os
import json
from tqdm import tqdm

def eval_message(question_with_options, context_data=None) -> str:
    message = f"Respond to the following question using one of the letters A, B, C, D only. Do not respond with any other text. \n{question_with_options}"
    if context_data:
        message += f"\n\nYou can use the following context data to answer the question: {context_data}"
    return message

def eval_questions_with_litgpt(json_content, llm, context_data=None):
    results = []
    for i, question in enumerate(tqdm(json_content)):
        if context_data is not None:
            content = eval_message(question['input'], context_data[i])
        else:
            content = eval_message(question['input'])
        messages = [
            {"role": "user", "content": content},
        ]
        generated_answer = llm.generate(messages)
        # Retrieve the last character
        final_answer = generated_answer[-1]

        print(f"Question: {question['input']}")
        print(f"Generated answer: {generated_answer}")
        print(f"Correct answer: {question['output']}")
        print("--------------------------------")

        results.append({"question": question["input"], "generated_answer": final_answer, "correct_answer": question["output"], "is_correct": final_answer == question["output"]})
    return results

if __name__ == "__main__":
    llm = LLM.load("allenai/OLMo-2-1124-7B-Instruct")
    # llm_finetuned = LLM.load("./out/sft_wiki_20/final")

    with open("data/wiki_20/gpt_5_mini_questions.jsonl", "r") as f:
        json_content = [json.loads(l) for l in f]

    with open("data/wiki_20/data.json", "r") as f:
        wiki_20 = {d['id']: d['text'] for d in json.load(f)}
    context_data = [wiki_20[q['id']] for q in json_content]

    # Shuffle and split into train and test
    random.shuffle(json_content)
    
    # print("Evaluating with OLMo-2-1124-7B-Instruct")
    pipeline_results = eval_questions_with_litgpt(json_content, llm_finetuned, context_data)
    print(f"OLMo Blind Accuracy: {sum(result['is_correct'] for result in pipeline_results) / len(pipeline_results)}")
