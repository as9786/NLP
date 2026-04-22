# Library
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Data
data = [
    {
        "instruction": "Translate to English",
        "input": "안녕하세요",
        "output": "Hello"
    },
    {
        "instruction": "Answer the question",
        "input": "What is QLoRA?",
        "output": "QLoRA fine-tunes LoRA adapters on top of a 4-bit quantized base model."
    },
    {
        "instruction": "Summarize the sentence",
        "input": "QLoRA reduces memory usage by quantizing the pretrained model to 4-bit and training LoRA adapters.",
        "output": "QLoRA enables memory-efficient fine-tuning with 4-bit quantization and LoRA."
    },
]

dataset = Dataset.from_list(data)

def format_example(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}

dataset = dataset.map(format_example)

# 모형
model_name = 'gpt2'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 4-Bit Quantization
compute_dtype = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype
)

# 양자화 모형
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# 학습 준비
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

max_length = 256

def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    bf16=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

prompt = """### Instruction:
Answer the question

### Input:
What is QLoRA?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
