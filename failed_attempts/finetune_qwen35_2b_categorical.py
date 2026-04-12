#!/usr/bin/env python3
"""
Qwen3.5-2B Fine-tuning - CATEGORICAL GRADING (A/B/C/D)
Much simpler task than numeric prediction
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

# Question map
QUESTION_MAP = {
    1: "Explain how colorimetric biosensors using nanomaterials enhance sensitivity and specificity compared to HRP-based biosensors.",
    2: "Describe biofunctionalization of gold nanoparticles and how specificity is achieved. How is interference minimized?",
    3: "How do physicochemical properties of nanomaterials affect biosensor performance? What criteria are used for selection?",
    4: "Define Limit of Detection (LOD), how it is calculated, and its significance."
}

print("[STEP 1] Loading and converting data...")
df = pd.read_csv("synthetic_exam_data_categorical.csv")
print(f"  ✓ Loaded {len(df)} samples")
print(f"  Grades: {dict(df['grade'].value_counts())}\n")

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Create simple text format
def create_prompt(row):
    q_num = int(row['question_number'])
    question_text = QUESTION_MAP[q_num]
    answer_text = row['answer_text'][:300]  # Shorter for categorical task
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
    dtype=torch.bfloat16,  # bf16 (not QLoRA)
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
print("  ✓ Datasets ready\n")

print("[STEP 5] TRAINING...")
print("="*80)
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Model: Qwen3.5-2B (bf16 LoRA)")
print(f"  Task: Categorical grading (A/B/C/D)")
print(f"  Batch: 4")
print(f"  Epochs: 3")
print(f"  Expected: ~15-20 minutes")
print("="*80 + "\n")

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=0,
        save_strategy="no",
        optim="adamw_8bit",
        seed=3407,
        output_dir="qwen35_categorical_output",
        remove_unused_columns=False,  # FIX: Don't remove unused columns
    ),
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
