# Library

import os
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Data
data = [
    {
        "instruction": "Translate to English",
        "input": "안녕하세요",
        "output": "Hello"
    },
    {
        "instruction": "Answer the question",
        "input": "What is LoRA?",
        "output": "LoRA is a parameter-efficient fine-tuning method."
    },
    {
        "instruction": "Summarize the sentence",
        "input": "LoRA reduces trainable parameters by injecting low-rank adapters.",
        "output": "LoRA enables efficient fine-tuning with fewer trainable parameters."
    },
]

dataset = Dataset.from_list(data)

# Prompt
def format_example(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}

dataset = dataset.map(format_example)

# 모형 종류
model_name = 'gpt2'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 모형
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# 최대 길이
max_length = 256

# Tokenize function
def tokenize_function(data):
    tokens = tokenizer(data['text'], truncation=True, max_length=max_length, padding='max_length')
    tokens['label'] = tokens['input_ids'].copy()
    return tokens

# Example dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

# LoRA
lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r=9,# Rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    target_modules=['c_attn']
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 학습 매개변수
training_args = TrainingArguments(
    output_dir='./lora_output',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
)

# 학습
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

prompt = """### Instruction:
Answer the question

### Input:
What is LoRA?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
