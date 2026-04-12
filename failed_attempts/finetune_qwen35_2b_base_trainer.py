#!/usr/bin/env python3
"""
Qwen3.5-2B Fine-tuning - CATEGORICAL GRADING
Uses base Trainer (most reliable approach)
"""

import os
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import pandas as pd
from datetime import datetime

print("\n" + "="*80)
print("  QWEN3.5-2B - CATEGORICAL GRADING".center(80))
print("  Grades: A=Excellent, B=Good, C=Fair, D=Poor".center(80))
print("="*80 + "\n")

QUESTION_MAP = {
    1: "Explain how colorimetric biosensors using nanomaterials enhance sensitivity and specificity compared to HRP-based biosensors.",
    2: "Describe biofunctionalization of gold nanoparticles and how specificity is achieved. How is interference minimized?",
    3: "How do physicochemical properties of nanomaterials affect biosensor performance? What criteria are used for selection?",
    4: "Define Limit of Detection (LOD), how it is calculated, and its significance."
}

print("[STEP 1] Loading data...")
df = pd.read_csv("synthetic_exam_data_categorical.csv")
print(f"  ✓ Loaded {len(df)} samples\n")

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

def create_prompt(row):
    q_num = int(row['question_number'])
    question_text = QUESTION_MAP[q_num]
    answer_text = row['answer_text'][:300]
    grade = row['grade']
    
    prompt = f"""Question: {question_text}

Answer: {answer_text}

Grade: {grade}"""
    return prompt

train_texts = [create_prompt(row) for _, row in train_df.iterrows()]
val_texts = [create_prompt(row) for _, row in val_df.iterrows()]

print(f"  Training: {len(train_texts)}")
print(f"  Validation: {len(val_texts)}\n")

print("[STEP 2] Loading Qwen3.5-2B...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-2B",
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
print("  ✓ Model loaded\n")

print("[STEP 3] Attaching LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)
print("  ✓ LoRA attached\n")

print("[STEP 4] Creating datasets...")
from datasets import Dataset

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

# Tokenize (text-only, no vision processing)
def tokenize_function(examples):
    # Use tokenizer.tokenizer to bypass vision processing
    return tokenizer.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("  ✓ Datasets tokenized\n")

print("[STEP 5] TRAINING...")
print("="*80)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Model: Qwen3.5-2B (bf16 LoRA)")
print(f"  Task: Categorical grading (A/B/C/D)")
print(f"  Batch: 4")
print(f"  Epochs: 3")
print(f"  Expected: ~15-20 minutes")
print("="*80 + "\n")

from transformers import TrainingArguments, Trainer
import torch

# Custom data collator for text-only (avoids processor issues)
class SimpleDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, features):
        # Get max length in batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            
            # Pad
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            labels = input_ids.copy()
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["labels"] = torch.tensor(batch["labels"])
        
        return batch

data_collator = SimpleDataCollator(tokenizer.tokenizer)

training_args = TrainingArguments(
    output_dir="qwen35_categorical_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_steps=5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="no",
    save_steps=0,
    save_strategy="no",
    optim="adamw_8bit",
    seed=3407,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer_stats = trainer.train()

print("\n" + "="*80)
print("  TRAINING COMPLETE!".center(80))
print("="*80)
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Final loss: {trainer_stats.training_loss:.4f}")

model.save_pretrained("qwen35_categorical_final")
tokenizer.save_pretrained("qwen35_categorical_final")

print(f"  Model saved: qwen35_categorical_final/\n")

print("="*80)
print("  NEXT: Test predictions!".center(80))
print("="*80)
print("\nRun: python test_qwen35_grader.py\n")
