import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from transformers import default_data_collator

#load model
model_name = 'meta-llama/Llama-3.1-8B-Instruct'  
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# set padding tokenTokenizer 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

#set lora
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="./list.jsonl")
train_dataset = dataset["train"]

max_token_length = 0
longest_instance = None

# Step 1: Calculate token lengths across the dataset
def calculate_token_lengths(batch):
    lengths = []
    for messages in batch["messages"]:
        inputs = [
            msg["content"][0]["content"]
            for msg in messages
            if msg["role"] == "user"
        ]
        if len(inputs) > 1:
            inputs = [" ".join(inputs)]
        
        model_inputs = tokenizer(inputs, truncation=False, padding=False)
        lengths.extend([len(seq) for seq in model_inputs["input_ids"]])
    
    return {"lengths": lengths}

# Step 2: Apply the function to the dataset and collect lengths
lengths_dataset = train_dataset.map(calculate_token_lengths, batched=True)
all_lengths = lengths_dataset["lengths"]


def preprocess_function(examples):
    all_inputs = []
    all_targets = []
    
    for messages in examples["messages"]: # Loop through each "messages" entry in the batch
        # Extract user message content
        inputs = [
            msg["content"][0]["content"]
            for msg in messages
            if msg["role"] == "user"
        ]
        # Extract assistant message content (target responses)
        targets = [
            msg["content"][0]["content"]
            for msg in messages
            if msg["role"] == "assistant"
        ]
        # Add processed inputs and targets to the batch lists
        all_inputs.extend(inputs)
        all_targets.extend(targets)
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(all_inputs, max_length=2048, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(all_targets, max_length=2048, truncation=True, padding="max_length")

    # Assign labels and replace padding tokens with -100
    model_inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in labels["input_ids"]
    ]

    return model_inputs

# Apply the preprocessing function to the dataset
tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1, 
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    fp16=True,
    learning_rate=2e-5,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    max_steps=6000,  
    save_strategy="steps",
    save_steps=1000,  
    save_total_limit=6,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator, 
)

trainer.train()

trainer.save_model("trained_model")
