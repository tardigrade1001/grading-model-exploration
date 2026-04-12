#!/usr/bin/env python3
"""
Gemma 2 9B - BASE TRAINER with proper label masking
Masks instruction tokens, trains only on mark prediction
"""

import os
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import pandas as pd
from datetime import datetime

print("\n" + "="*80)
print("  GEMMA 2 9B - MASKED TRAINING (Mark prediction only)".center(80))
print("="*80 + "\n")

QUESTION_MAP = {
    1: "Explain how colorimetric biosensors using nanomaterials enhance sensitivity and specificity compared to HRP-based biosensors.",
    2: "Describe biofunctionalization of gold nanoparticles and how specificity is achieved. How is interference minimized?",
    3: "How do physicochemical properties of nanomaterials affect biosensor performance? What criteria are used for selection?",
    4: "Define Limit of Detection (LOD), how it is calculated, and its significance."
}

print("[STEP 1] Loading data...")
df = pd.read_csv("synthetic_exam_data_100students.csv")
print(f"  ✓ Loaded {len(df)} samples\n")

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Create text WITH MARKER for where to split instruction/response
def create_prompt(row):
    q_num = int(row['question_number'])
    question_text = QUESTION_MAP[q_num]
    answer_text = row['answer_text'][:400]
    mark = int(row['marks_obtained']) if row['marks_obtained'] == int(row['marks_obtained']) else row['marks_obtained']
    
    # Format: instruction | RESPONSE_START | response
    instruction = f"[INST] Grade this exam answer out of 4 marks.\n\nQuestion: {question_text}\n\nStudent Answer: {answer_text}\n\nProvide only the numeric grade. [/INST]\n"
    response = f"{mark}"
    
    return {
        "instruction": instruction,
        "response": response,
        "full_text": instruction + response
    }

train_data = [create_prompt(row) for _, row in train_df.iterrows()]
val_data = [create_prompt(row) for _, row in val_df.iterrows()]

print(f"  Training: {len(train_data)}")
print(f"  Validation: {len(val_data)}\n")

print("[STEP 2] Loading model...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
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
)
print("  ✓ LoRA attached\n")

print("[STEP 4] Creating datasets with proper masking...")
from datasets import Dataset

train_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in train_data],
    "response": [d["response"] for d in train_data],
})

val_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in val_data],
    "response": [d["response"] for d in val_data],
})

# Tokenize and mask
def tokenize_with_mask(examples):
    # Tokenize full text
    full_texts = [inst + resp for inst, resp in zip(examples["instruction"], examples["response"])]
    full_tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)
    
    # Tokenize instruction only (to get mask positions)
    inst_tokenized = tokenizer(examples["instruction"], padding=False, truncation=True)
    
    # Create labels: -100 for instruction tokens, actual token_ids for response
    labels = []
    for i, inst_len in enumerate(inst_tokenized["input_ids"]):
        inst_length = len(inst_len)
        label = [-100] * inst_length + full_tokenized["input_ids"][i][inst_length:]
        # Pad to max_length
        label += [-100] * (512 - len(label))
        labels.append(label[:512])
    
    full_tokenized["labels"] = labels
    return full_tokenized

train_dataset = train_dataset.map(
    tokenize_with_mask,
    batched=True,
    remove_columns=["instruction", "response"]
)

val_dataset = val_dataset.map(
    tokenize_with_mask,
    batched=True,
    remove_columns=["instruction", "response"]
)

print("  ✓ Datasets tokenized with masking\n")

print("[STEP 5] MASKED TRAINING...")
print("="*80)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Samples: {len(train_dataset)} training")
print(f"  Training ONLY on mark tokens (instruction masked)")
print(f"  Expected: ~10-15 minutes, should learn mark prediction")
print("="*80 + "\n")

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="gemma2_masked_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    warmup_steps=5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    learning_rate=2e-4,
    logging_steps=10,
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
print("  MASKED TRAINING COMPLETE!".center(80))
print("="*80)
print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Final loss: {trainer_stats.training_loss:.4f}")

model.save_pretrained("gemma2_masked_final")
tokenizer.save_pretrained("gemma2_masked_final")

print(f"  Model saved: gemma2_masked_final/\n")

print("="*80)
print("  NEXT: Test predictions!".center(80))
print("="*80)
print("\nRun: python test_masked_grader.py\n")
