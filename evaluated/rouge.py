import json
from rouge_score import rouge_scorer

# Mở và đọc file merged_output.json
with open('merged_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Khởi tạo scorer để tính điểm ROUGE
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Khởi tạo danh sách để lưu các điểm ROUGE
rouge_doc_1 = []
rouge_doc_2 = []
rouge_doc_L = []

rouge_gpt_1 = []
rouge_gpt_2 = []
rouge_gpt_L = []

rouge_gpt4o_1 = []
rouge_gpt4o_2 = []
rouge_gpt4o_L = []

for item in data:
    ref = item['en']  # Kết quả tham chiếu
    doc = item['en_doc']  # Kết quả từ vi-en_doc
    gpt = item['en_gpt']  # Kết quả từ vi-en_gpt
    gpt4o = item.get('en_gpt4o', '')  # Kết quả từ vi-en_gpt4o nếu có

    # Tính điểm ROUGE cho từng phiên bản
    scores_doc = scorer.score(ref, doc)
    scores_gpt = scorer.score(ref, gpt)
    scores_gpt4o = scorer.score(ref, gpt4o) if gpt4o else None

    # Lưu điểm ROUGE-1, ROUGE-2, ROUGE-L vào các danh sách
    rouge_doc_1.append(scores_doc['rouge1'].fmeasure)
    rouge_doc_2.append(scores_doc['rouge2'].fmeasure)
    rouge_doc_L.append(scores_doc['rougeL'].fmeasure)

    rouge_gpt_1.append(scores_gpt['rouge1'].fmeasure)
    rouge_gpt_2.append(scores_gpt['rouge2'].fmeasure)
    rouge_gpt_L.append(scores_gpt['rougeL'].fmeasure)

    if scores_gpt4o:
        rouge_gpt4o_1.append(scores_gpt4o['rouge1'].fmeasure)
        rouge_gpt4o_2.append(scores_gpt4o['rouge2'].fmeasure)
        rouge_gpt4o_L.append(scores_gpt4o['rougeL'].fmeasure)

# Tính điểm trung bình cho mỗi loại ROUGE
avg_rouge_doc_1 = sum(rouge_doc_1) / len(rouge_doc_1)
avg_rouge_doc_2 = sum(rouge_doc_2) / len(rouge_doc_2)
avg_rouge_doc_L = sum(rouge_doc_L) / len(rouge_doc_L)

avg_rouge_gpt_1 = sum(rouge_gpt_1) / len(rouge_gpt_1)
avg_rouge_gpt_2 = sum(rouge_gpt_2) / len(rouge_gpt_2)
avg_rouge_gpt_L = sum(rouge_gpt_L) / len(rouge_gpt_L)

avg_rouge_gpt4o_1 = sum(rouge_gpt4o_1) / len(rouge_gpt4o_1) if rouge_gpt4o_1 else 0
avg_rouge_gpt4o_2 = sum(rouge_gpt4o_2) / len(rouge_gpt4o_2) if rouge_gpt4o_2 else 0
avg_rouge_gpt4o_L = sum(rouge_gpt4o_L) / len(rouge_gpt4o_L) if rouge_gpt4o_L else 0

# In ra các điểm ROUGE
print(f"📏 Avg ROUGE-1 (Doc): {avg_rouge_doc_1:.4f}")
print(f"📏 Avg ROUGE-2 (Doc): {avg_rouge_doc_2:.4f}")
print(f"📏 Avg ROUGE-L (Doc): {avg_rouge_doc_L:.4f}")

print(f"📏 Avg ROUGE-1 (GPT): {avg_rouge_gpt_1:.4f}")
print(f"📏 Avg ROUGE-2 (GPT): {avg_rouge_gpt_2:.4f}")
print(f"📏 Avg ROUGE-L (GPT): {avg_rouge_gpt_L:.4f}")

print(f"📏 Avg ROUGE-1 (GPT-4o): {avg_rouge_gpt4o_1:.4f}")
print(f"📏 Avg ROUGE-2 (GPT-4o): {avg_rouge_gpt4o_2:.4f}")
print(f"📏 Avg ROUGE-L (GPT-4o): {avg_rouge_gpt4o_L:.4f}")
