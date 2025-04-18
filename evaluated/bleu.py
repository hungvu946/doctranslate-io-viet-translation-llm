import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Tải punkt nếu chưa có
# nltk.download('punkt')

# Hàm tính điểm BLEU với smoothing
def calc_bleu(reference, hypothesis):
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)

# Đọc dữ liệu từ file JSON
with open('merged_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Tính điểm BLEU trung bình
bleu_doc_total = 0
bleu_gpt_total = 0
bleu_gpt4o_total = 0
count = 0

for item in data:
    ref = item["en"]
    doc = item["en_doc"]
    gpt = item["en_gpt"]
    gpt4o = item["en_gpt4o"]

    bleu_doc_total += calc_bleu(ref, doc)
    bleu_gpt_total += calc_bleu(ref, gpt)
    bleu_gpt4o_total += calc_bleu(ref, gpt4o)
    count += 1

# Trung bình BLEU
avg_bleu_doc = bleu_doc_total / count
avg_bleu_gpt = bleu_gpt_total / count
avg_bleu_gpt4o = bleu_gpt4o_total / count

# In kết quả
print(f"📊 Average BLEU Score:")
print(f" - en_doc  : {avg_bleu_doc:.4f}")
print(f" - en_gpt  : {avg_bleu_gpt:.4f}")
print(f" - en_gpt4o: {avg_bleu_gpt4o:.4f}")
