#!/usr/bin/env python3
"""
Debug: What does Gemma 2 actually output?
"""

import torch
import pandas as pd
from unsloth import FastLanguageModel

print("\n" + "="*80)
print("  GEMMA 2 - RAW OUTPUT DEBUG".center(80))
print("="*80 + "\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gemma2_fast_test_final",
    max_seq_length=256,
)
model = model.eval()

df = pd.read_csv("synthetic_exam_data_categorical.csv")
test = df.sample(3, random_state=99)

for idx, row in test.iterrows():
    q = int(row['question_number'])
    ans = row['answer_text'][:200]
    actual = row['grade']
    
    prompt = f"Q{q}: {ans}\n\nGrade: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)
    
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full[len(prompt):]
    
    print(f"[Test {idx}]")
    print(f"  Actual: {actual}")
    print(f"  Generated: '{generated}'")
    print(f"  Bytes: {repr(generated)}")
    print(f"  First char: '{generated[0] if generated else '(empty)'}'")
    print()
