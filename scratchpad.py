from datasets import load_dataset, load_from_disk
import json
wiki_1k = load_from_disk("data/wiki_1k")

with open("data/wiki_1k/titles.txt", "w") as f:
    for item in wiki_1k:
        f.write(item['title'] + "\n")