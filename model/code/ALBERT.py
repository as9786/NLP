# Library
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('albert/albert-base-v2')

# 모형
model = AutoModelForMaskedLM.from_pretrained(
    "albert/albert-base-v2",
    dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto"
)

# 입력
prompt = "Plants create energy through a process known as [MASK]."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    predictions = outputs.logits[0, mask_token_index]

# 결과
top_k = torch.topk(predictions, k=5).indices.tolist()
for token_id in top_k[0]:
    print(f"Prediction: {tokenizer.decode([token_id])}")
