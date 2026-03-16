from datasets import load_dataset, load_from_disk
dataset = load_from_disk("distill/out/grpo_distill/question_prompt_dataset.json")

for item in dataset:
    print(item)