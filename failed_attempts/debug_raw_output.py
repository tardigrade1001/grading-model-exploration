#!/usr/bin/env python3
"""
Debug - show raw model output
"""

import torch
from unsloth import FastLanguageModel

print("\n" + "="*80)
print("  DEBUG: Raw Model Output".center(80))
print("="*80 + "\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gemma2_sft_final",
    max_seq_length=512,
)
model = model.eval()

# Test prompts
test_prompts = [
    "[INST] Grade this out of 4. [/INST]\n",
    "[INST] What is 2+2? [/INST]\n",
    "The mark is ",
    "Grade: ",
]

print("Testing model outputs:\n")

for i, prompt in enumerate(test_prompts):
    print(f"[Test {i+1}]")
    print(f"  Prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(prompt):]
    
    print(f"  Generated: '{generated}'")
    print(f"  Length: {len(generated)}\n")

print("="*80)
print("\nKEY: If generated is empty/whitespace → model not learning task\n")
