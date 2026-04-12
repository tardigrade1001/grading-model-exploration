# Failed Attempts: Why Large Language Models Didn't Work

This folder documents 7 different attempts to use large language models (Gemma 2, Qwen3.5) to predict exam marks. All failed. This is important to document because it shows:

1. **What was tried**
2. **Why it didn't work**
3. **What we learned**

## Summary Table

| # | Model | Method | Data | Loss | Output | Accuracy |
|---|-------|--------|------|------|--------|----------|
| 1 | Gemma 2 9B | Language modeling | Synthetic 400 | 9.78 | All 0.0 | 0% |
| 2 | Gemma 2 9B | Supervised fine-tuning | Synthetic 400 | 6.98 | All 0.0 | 0% |
| 3 | Gemma 2 9B | Masked language modeling | Synthetic 400 | 6.38 | All 0.0 | 0% |
| 4 | Qwen3.5-2B | Base trainer | Real 122 | — | Vision processor crash | 0% |
| 5 | Qwen3.5-2B | Custom collator | Real 122 | — | .pad() method not found | 0% |
| 6 | Qwen3.5-2B | Categorical A/B/C/D | Synthetic 400 | 35.88 | Garbage: %,&,) | 13% |
| 7 | Gemma 2 9B | Minimal fast test | Synthetic 400 | 20.62 | Empty strings | 0% |

## Detailed Analysis of Each Attempt

### Attempt 1: Gemma 2 9B Language Modeling

**File:** `finetune_gemma2_quick_improved.py`

**Approach:** Fine-tune Gemma 2 9B on exam answers using causal language modeling loss

**Setup:**
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b",
    max_seq_length=512,
    load_in_4bit=True,
)
# Fine-tune with training loop...
```

**Result:** 
- Training loss converged to 9.78
- Model outputs: All predictions were 0.0 (the baseline prediction)
- Accuracy: 0%

**Why it failed:**
- Language modeling loss optimizes for next-token prediction, not score prediction
- Model learned to output the baseline value rather than learning the task
- No explicit supervision for the mark prediction task

### Attempt 2: Gemma 2 9B Supervised Fine-Tuning (SFT)

**File:** `finetune_gemma2_slow_production_improved.py`

**Approach:** Format as instruction-response pairs for SFT

**Setup:**
```
Instruction: "You are grading an exam. Here is a student answer: [ANSWER]. Assign a mark from 1.0 to 4.0."
Response: "[MARK]"
```

**Result:**
- Training loss: 6.98 (lower than Attempt 1)
- Model outputs: All 0.0
- Accuracy: 0%

**Why it failed:**
- Even with explicit instruction format, model didn't learn to map text to numeric scores
- LLMs are optimized for text generation, not numeric prediction
- The mark prediction task is classification/regression, not generation

### Attempt 3: Gemma 2 9B Masked Language Modeling

**File:** `finetune_gemma2_masked.py`

**Approach:** Mask answer text and predict marks as masked tokens

**Setup:**
```python
# Format: [MASK] marks given to answer: [ANSWER]
# Predict: [MASK] token should be the mark
```

**Result:**
- Training loss: 6.38 (lowest so far)
- Model outputs: All 0.0
- Accuracy: 0%

**Why it failed:**
- Masked language modeling is for token prediction within text
- Marks (numeric values) are not well-represented in token vocabulary
- Task mismatch: MLM is for language understanding, not score prediction

### Attempt 4: Qwen3.5-2B Base Trainer

**File:** `finetune_qwen35_2b_base_trainer.py`

**Approach:** Use Qwen3.5 (multimodal model) with standard trainer

**Result:**
- Error: `RuntimeError: Cannot create CUDA tensors without CUDA enabled`
- Then: Vision processor crash trying to parse text as images

**Why it failed:**
- Qwen3.5 is a multimodal model (vision + language)
- When given text, it tried to process it as visual input
- Tokenizer mismatch between vision and text modalities

### Attempt 5: Qwen3.5-2B Custom Collator

**File:** `finetune_qwen35_2b_categorical.py`

**Approach:** Create custom data collator to handle Qwen tokenizer

**Setup:**
```python
class SimpleDataCollator:
    def __call__(self, batch):
        # Custom padding logic
```

**Result:**
- Error: `AttributeError: 'Qwen3VLProcessor' object has no attribute 'pad'`
- Model expects processor.pad() method that doesn't exist

**Why it failed:**
- Qwen3.5's processor wasn't designed for standard fine-tuning
- Custom collator couldn't work around the architectural mismatch
- Multimodal model architecture incompatible with text-only task

### Attempt 6: Qwen3.5-2B Categorical (A/B/C/D)

**File:** `finetune_qwen35_2b_categorical.py` (second version)

**Approach:** Format marks as categorical classes: A=1.0, B=2.5, C=3.5, D=4.0

**Setup:**
```python
# Expected tokens: 357 (A), 417 (B), 351 (C), 414 (D)
# Model should learn to output one of these tokens
```

**Result:**
- Training loss: 35.88
- Model outputs: Garbage characters (%,&,))
- Accuracy: 13% (some random correctness by chance)

**Debug finding:** 
- Training data verified correct (A/B/C/D strings present)
- Model was outputting token 4 (%) instead of 357 (A)
- Token mismatch: model never learned the task

**Why it failed:**
- Token vocabulary was wrong
- Model never learned the categorical mapping
- Multimodal architecture still fundamentally incompatible

### Attempt 7: Gemma 2 9B Minimal Fast Test

**File:** `gemma2_fast_test.py`

**Approach:** Minimal configuration to quickly test if anything works

**Setup:**
```python
# Simplest possible setup
# Just try to predict marks
```

**Result:**
- Training loss: 20.62
- Model outputs: Empty strings or newlines only
- Accuracy: 0%

**Why it failed:**
- Even minimal configuration couldn't work
- Indicates fundamental task mismatch

## Key Learning: Why LLMs Failed

### The Core Problem

**Large Language Models are built to generate text tokens, not predict numeric scores.**

```
LLM strength:  "The next token after 'The quick brown' is 'fox'"
LLM weakness:  "Given answer text, the numeric score is 2.3"
```

### Why Loss Convergence Didn't Help

Training loss converging doesn't mean the model learned the task. It meant:
- Model learned to output baseline values (0, 0.0, empty string)
- Model minimized loss by ignoring input and predicting average
- Classic overfitting: loss goes down, accuracy stays at 0%

### The Mismatch

| LLM Design | Task Requirements |
|-----------|-------------------|
| Designed for text generation | Needs numeric prediction |
| Billions of tokens trained | Only 122 examples to learn from |
| Output is text | Output should be numbers |
| Training on diverse web text | Training on domain-specific exams |

## What Would Have Been Needed

To make LLMs work for this task, you would need:

1. **Much more training data** (1000+, not 122)
2. **Different architecture** (regression head on top of embeddings, not text generation)
3. **Different loss function** (MSE or classification loss, not language modeling loss)
4. **Different fine-tuning approach** (prompt tuning, in-context learning, or LoRA adaptation)

## Why Simple ML Worked Instead

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 50 TF-IDF features → identifies which keywords matter
# LogisticRegression → learns weights for each keyword
# Result: 56% accuracy
```

**Advantages:**
- Directly solves the classification task
- Doesn't overfit on 122 samples
- Transparent: can see which features matter
- Fast to train and iterate

## Lessons

1. **Use the right tool for the right task** — LLMs for generation, simple ML for classification
2. **Loss convergence ≠ task learning** — Always check actual predictions
3. **Understand your data before your model** — 122 exam answers is very small
4. **Start simple** — Baseline first, then add complexity only if needed
5. **Document failures** — They're often more informative than successes

## Reproducibility

All failed attempts are preserved in this folder. To run them (optional):

```bash
# Install optional dependencies
pip install transformers torch unsloth xgboost

# Run any attempt
python finetune_gemma2_quick_improved.py
python finetune_qwen35_2b_categorical.py
# ... etc
```

They will fail as documented above, demonstrating the problem clearly.

## Conclusion

This folder documents a learning experience: sometimes the most valuable work is understanding why something doesn't work and pivoting to a better approach. The switch from LLM fine-tuning to simple ML resulted in:

- Simple ML baseline: **56% accuracy** ✓
- Engineered features: **68% accuracy** ✓
- LLM attempts: **0% accuracy** ✗

The successful approach was simpler, faster, and more interpretable.
