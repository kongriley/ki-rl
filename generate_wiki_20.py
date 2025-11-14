from datasets import load_dataset, load_from_disk
import random
import urllib.parse 
from tqdm import tqdm
import json

wiki_1k = load_from_disk("data/wiki_1k")

good_titles = {
    "Dave Sands",
    "Nazime Sultan",
    "Vulcan Street Plant",
    "Oath of Allegiance (New Zealand)",
    "Baby colic",
    "WarCry Network",
    "Heisei Rider vs. Shōwa Rider: Kamen Rider Taisen feat. Super Sentai",
    "Item Idem",
    "Martin Crimp",
    "Wolf Warrior",
    "Dark Sun: Wake of the Ravager",
    "Siraj ud-Daulah",
    "Space Quest 6",
    "G-funk",
    "Hans Speckaert",
    "The Creative Gene",
    "51 Astor Place",
    "Google Brain",
    "Mrs. Munger's Class",
    "Acoustic quieting",
}

out_dataset = []
used_titles = set()

for i, example in enumerate(tqdm(wiki_1k)):
    if example['title'] in good_titles:
        out_dataset.append(example)
        used_titles.add(example['title'])

print(len(out_dataset))
print(good_titles - used_titles)

with open('data/wiki_20/data.json', 'w') as f:
    json.dump(out_dataset, f)