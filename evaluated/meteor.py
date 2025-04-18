import json
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize

nltk.download('wordnet')

with open('merged_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

meteor_doc = []
meteor_gpt = []
meteor_gpt4o = []

for item in data:
    ref = word_tokenize(item['en'].lower())
    doc = word_tokenize(item['en_doc'].lower())
    gpt = word_tokenize(item['en_gpt'].lower())
    gpt4o = word_tokenize(item.get('en_gpt4o', '').lower()) if 'en_gpt4o' in item else []

    meteor_doc.append(single_meteor_score(ref, doc))
    meteor_gpt.append(single_meteor_score(ref, gpt))
    if gpt4o:
        meteor_gpt4o.append(single_meteor_score(ref, gpt4o))

avg_meteor_doc = sum(meteor_doc) / len(meteor_doc)
avg_meteor_gpt = sum(meteor_gpt) / len(meteor_gpt)
avg_meteor_gpt4o = sum(meteor_gpt4o) / len(meteor_gpt4o) if meteor_gpt4o else 0

print(f"📏 Avg METEOR (Doc): {avg_meteor_doc:.4f}")
print(f"📏 Avg METEOR (GPT): {avg_meteor_gpt:.4f}")
print(f"📏 Avg METEOR (GPT-4o): {avg_meteor_gpt4o:.4f}")
