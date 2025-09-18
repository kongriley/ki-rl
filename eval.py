from transformers import pipeline
import re
import json

pipe = pipeline("text-generation", model="allenai/OLMo-2-1124-7B-Instruct")

def md_to_json(md_content):
    """Convert markdown quiz format to JSON."""
    questions = []
    blocks = re.split(r'^# Question \d+$', md_content, flags=re.MULTILINE)[1:]
    
    for i, block in enumerate(blocks, 1):
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2: continue
        
        q_line = next((l for l in lines if l.startswith('Question:')), None)
        a_line = next((l for l in lines if l.startswith('Answer:')), None)
        if not q_line or not a_line: continue
        
        text = q_line[9:].strip()
        first_opt = text.find('A)')
        question = text[:first_opt].strip()
        
        options = {}
        for match in re.findall(r'([A-D])\)\s*([^A-D]*?)(?=\s*[A-D]\)|$)', text):
            options[match[0]] = match[1].strip()
        
        answer = a_line.split(':', 1)[1].strip()
        questions.append({
            "id": i, "question_with_options": text, "question": question, "options": options, 
            "correct_answer": answer, "answer_text": options.get(answer, "")
        })
    
    return {"questions": questions}

with open("data/2025_nba_playoffs_questions.md", "r") as f:
    md_content = f.read()

json_content = md_to_json(md_content)

print(json_content)

for question in json_content["questions"]:
    messages = [
        {"role": "user", "content": f"Respond to the following question using the letters A, B, C, D only. \n{question['question_with_options']}"},
    ]
    response = pipe(messages)
    print(response.choices[0].message.content)
    print(question["correct_answer"])
    print("--------------------------------")

