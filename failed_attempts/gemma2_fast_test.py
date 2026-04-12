#!/usr/bin/env python3
"""
Gemma 2 9B - ULTRA FAST TEST
Just to see if it outputs ANY tokens (even garbage)
"""

import os
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import pandas as pd
from datetime import datetime

print("\n" + "="*80)
print("  GEMMA 2 - ULTRA FAST TEST".center(80))
print("  Just checking if it outputs ANYTHING".center(80))
print("="*80 + "\n")

# Minimal data
df = pd.read_csv("synthetic_exam_data_categorical.csv")
df = df.head(50)  # Only 50 samples for speed

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def create_prompt(row):
    q = int(row['question_number'])
    ans = row['answer_text'][:200]
    grade = row['grade']
    return f"Q{q}: {ans}\n\nGrade: {grade}"

train_texts = [create_prompt(row) for _, row in train_df.iterrows()]

print(f"[1] Data: {len(train_texts)} samples\n")

print("[2] Loading Gemma 2 9B...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length=256,  # SHORT
    dtype=None,
    load_in_4bit=True,
)
print("  ✓ Loaded\n")

print("[3] LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
print("  ✓ Ready\n")

print("[4] Dataset...")
from datasets import Dataset

train_dataset = Dataset.from_dict({"text": train_texts})

def tokenize_fn(ex):
    return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print("  ✓ Ready\n")

print("[5] TRAINING (1 epoch, 40 samples)...")
print("="*80)
print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*80 + "\n")

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="gemma2_fast_test",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        learning_rate=2e-4,
        logging_steps=2,
        eval_strategy="no",
        save_strategy="no",
        optim="paged_adamw_8bit",
        seed=3407,
    ),
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

print(f"\n✓ Finished at {datetime.now().strftime('%H:%M:%S')}\n")

model.save_pretrained("gemma2_fast_test_final")
tokenizer.save_pretrained("gemma2_fast_test_final")

print("="*80)
print("  Now test output...".center(80))
print("="*80)
print("\nRun: python debug_gemma2_output.py\n")
