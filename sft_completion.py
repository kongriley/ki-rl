from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/wiki_20/data.json", split="train")
print(dataset)

dataset = dataset.remove_columns(["url", "title"])

model_name = "allenai/OLMo-2-1124-7B-Instruct"
trainer = SFTTrainer(
    model=model_name,
    train_dataset=dataset,
)

trainer.train()

print("Training complete")

print("Saving model...")
trainer.save_model("out/sft_wiki_20_completion")
print("Model saved")