# Library
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 모형
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# 입력 문장
text = 'He ate out with his friends on Monday'

# Tokens
tokens = tokenizer.encode_plus(text, padding='max_length', truncation=True, return_tensors='pt')

# 입력
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# 문장 분류
output = model(input_ids, attention_mask)
logits = output.logits
predicted_class_idx = torch.argmax(logits, dim=-1).item()

print("Predicted class:", predicted_class_name)
