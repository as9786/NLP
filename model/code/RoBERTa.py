# Library
import torch
from transformers import RobertaTokenizer, RobertaModel

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 모형
model = RobertaModel.from_pretrained('roberta-base')

# 입력
inp = 'I am studying now'

# Tokenize
inp_token = torch.tensor([tokenizer.encode(inp, add_special_tokens=True)])

# 결과
output = model(inp_token)
print(output)
