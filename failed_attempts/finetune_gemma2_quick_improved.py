#!/usr/bin/env python3
"""
Gemma 2 9B Fine-tuning - QUICK RUN (IMPROVED)
- Full question text in prompts
- Explicit grading instructions
- Better training config
- Deterministic inference
"""

import os
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import pandas as pd
from datetime import datetime

print("\n" + "="*80)
print("  GEMMA 2 9B - QUICK RUN (IMPROVED)".center(80))
print("  Full questions + explicit grading instructions".center(80))
print("="*80 + "\n")

# Full question text (same as grading rubric)
QUESTION_MAP = {
    1: "Explain how colorimetric biosensors using nanomaterials enhance sensitivity and specificity compared to HRP-based biosensors.",
    2: "Describe biofunctionalization of gold nanoparticles and how specificity is achieved. How is interference minimized?",
    3: "How do physicochemical properties of nanomaterials affect biosensor performance? What criteria are used for selection?",
    4: "Define Limit of Detection (LOD), how it is calculated, and its significance."
}

# Load data
print("[STEP 1] Loading data...")
df = pd.read_csv("synthetic_exam_data_100students.csv")
print(f"  ✓ Loaded {len(df)} synthetic samples")
print(f"    Mark range: {df['marks_obtained'].min()}-{df['marks_obtained'].max()}")
print(f"    Mean mark: {df['marks_obtained'].mean():.2f}\n")

# Try to load real exam data too
try:
    real_df = pd.read_csv("exam_results_cleaned_final.csv")
    df = pd.concat([df, real_df], ignore_index=True)
    print(f"  ✓ Added {len(real_df)} real exam samples")
    print(f"    Total: {len(df)} samples\n")
except:
    print(f"  (No real exam data found, using synthetic only)\n")

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Create prompts with full question text and grading criteria
def format_prompt(row):
    q_num = int(row['question_number'])
    question_text = QUESTION_MAP[q_num]
    answer_text = row['answer_text']
    mark = row['marks_obtained']
    
    # Full prompt with context
    prompt = f"""Question: {question_text}

Student Answer: {answer_text}

Grading criteria: Correctness, completeness, clarity. Score out of 4.
Mark: {mark}"""
    return prompt

train_df = train_df.copy()
val_df = val_df.copy()
train_df['text'] = train_df.apply(format_prompt, axis=1)
val_df['text'] = val_df.apply(format_prompt, axis=1)

print(f"  Training: {len(train_df)} | Validation: {len(val_df)}\n")

# Load model
print("[STEP 2] Loading Gemma 2 9B...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length=512,  # Quick run: shorter sequences
    dtype=None,
    load_in_4bit=True,
)
print("  ✓ Model loaded\n")

# LoRA (improved)
print("[STEP 3] Attaching LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # INCREASED from 8
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
print("  ✓ LoRA attached\n")

# Tokenize
print("[STEP 4] Tokenizing...")
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df[['text']])
val_dataset = Dataset.from_pandas(val_df[['text']])

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # Quick run: shorter
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

print(f"  ✓ Tokenized\n")

# Training
print("[STEP 5] QUICK RUN - Training...")
print("="*80)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Samples: {len(train_df)} training")
print(f"  Batch: 8 (no accumulation for speed)")
print(f"  Seq length: 512 (quick validation)")
print(f"  Steps: ~50 (2 epochs)")
print(f"  Learning rate: 2e-4 (stable)")
print(f"  Expected: ~10-15 minutes, loss should drop to 2-4")
print("="*80 + "\n")

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="gemma2_quick_run_improved",
    num_train_epochs=2,  # 2 epochs
    per_device_train_batch_size=8,  # LARGER for speed
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,  # NO accumulation for speed
    warmup_steps=5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    learning_rate=2e-4,  # Consistent, stable
    logging_steps=5,
    eval_strategy="no",
    save_steps=0,
    save_strategy="no",
    optim="paged_adamw_8bit",
    seed=3407,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer_stats = trainer.train()

print("\n" + "="*80)
print("  QUICK RUN COMPLETE!".center(80))
print("="*80)
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Final loss: {trainer_stats.training_loss:.4f}")

model.save_pretrained("gemma2_quick_run_improved")
tokenizer.save_pretrained("gemma2_quick_run_improved")

print(f"  Model saved: gemma2_quick_run_improved/\n")

print("="*80)
print("  NEXT: Test predictions!".center(80))
print("="*80)
print("\nRun: python test_exam_grader_improved.py\n")
