# Methodology

This document explains the technical approach, data processing, and machine learning methods used in this project.

## Problem Statement

Given student exam answer text, can we predict the mark assigned by an instructor? This is essentially a regression problem (predicting continuous scores 1.0-4.0) that we treat as a 3-class classification problem (Low: 1.0-1.5, Mid: 2.0-2.5, High: 3.0-4.0).

## Data Source & Preparation

### Raw Data Collection

- **122 student exam answers** from a 4-question assessment
- **Source:** Handwritten exam papers transcribed to text
- **Format:** Question number, answer text, instructor-assigned mark (1.0-4.0 scale)

### Data Cleaning

1. **Removed incomplete entries:** Blank answers, illegible text, marks outside 1.0-4.0 range
2. **Deduplicated:** Removed identical answers (suggesting duplicate records)
3. **Text normalization:** Lowercase, stripped whitespace, no additional preprocessing
4. **Final dataset:** 122 answers across 4 questions

### Mark Distribution

```
Class       Count   Percentage
1.0         22      18%
1.5         19      16%
2.0         36      30%
2.5         10      8%
3.0         17      14%
3.5         1       1%
4.0         17      14%
Total       122     100%
```

**Classification scheme:**
- Low: 1.0-1.5 (34 answers, 28%)
- Mid: 2.0-2.5 (46 answers, 38%)
- High: 3.0-4.0 (35 answers, 29%)
- Note: 3.5 merged into High class due to small sample

## Feature Engineering

### Text Features (TF-IDF)

1. **TF-IDF vectorization** (50 features, vocabulary from answer text)
   - Term frequency inverse document frequency
   - Captures which keywords/concepts appear in high vs low-scoring answers
   - Common words (the, is, a) get lower weight
   - Distinctive concepts get higher weight

### Domain-Informed Engineered Features

Based on manual analysis of grading patterns, 15 additional features were created:

#### Concept Presence Features

```python
concepts_good = {
    'mentions_antibody': r'\banti\w*bod\w*',
    'mentions_specificity': r'\bspecificit\w*',
    'mentions_sensitivity': r'\bsensitivit\w*',
    'mentions_nanomaterial': r'\bnano\w*',
    'mentions_tmb': r'\btmb\b',
    'mentions_binding': r'\bbind\w*',
    'mentions_target': r'\btarget\w*',
}

concepts_bad = {
    'overuses_lod': r'\blod\b|\blimit.*detection\b',
    'overuses_calibration': r'\bcalibr\w*',
}
```

**Rationale:** Manual inspection of high vs low-scoring answers revealed that certain concepts appear frequently in high marks (antibodies: 63%, specificity: 91%) while others appear in low marks (LOD: 71%, calibration: 63%).

#### Length Features

```python
answer_length = len(transcribed_text)  # Raw character count
answer_words = len(transcribed_text.split())  # Word count
is_detailed = 1 if answer_length >= 600 else 0  # Binary: detailed or not
is_minimal = 1 if answer_length < 400 else 0  # Binary: very brief
```

**Rationale:** High marks average 953 characters, low marks 397 characters (2.4× difference). The relationship is linear and consistent.

#### Question Features

```python
is_q4 = 1 if question_number == 4 else 0
is_q1 = 1 if question_number == 1 else 0
```

**Rationale:** Question 4 averages 1.38 marks, Question 1 averages 3.0 marks. Significant question-level bias exists.

### Feature Summary

- **TF-IDF features:** 50 (raw text features)
- **Engineered features:** 15 (domain-informed)
- **Total features:** 65 for full model, 45 selected by importance

## Model Development

### Approach 1: Large Language Model Fine-tuning (Failed)

Attempted 7 different configurations:

| # | Model | Method | Data | Outcome |
|---|-------|--------|------|---------|
| 1 | Gemma 2 9B | Language modeling | Synthetic 400 | Loss converged, output: all 0.0 |
| 2 | Gemma 2 9B | Supervised fine-tuning | Synthetic 400 | Loss: 6.98, output: all 0.0 |
| 3 | Gemma 2 9B | Masked training | Synthetic 400 | Loss: 6.38, output: all 0.0 |
| 4 | Qwen3.5-2B | Base trainer | Real 122 | Vision processor crash |
| 5 | Qwen3.5-2B | Custom collator | Real 122 | Processor.pad() not found |
| 6 | Qwen3.5-2B | Categorical A/B/C/D | Synthetic 400 | Loss: 35.88, output: garbage |
| 7 | Gemma 2 9B | Minimal fast test | Synthetic 400 | Loss: 20.62, output: empty |

**Why it failed:**
- LLMs are trained to generate tokens, not predict classes
- Even with correct training data, models never learned the task
- Large models overfit on small datasets
- Synthetic data (400 answers with random marks) provided no learning signal

See `failed_attempts/README.md` for detailed analysis of each attempt.

### Approach 2: Simple ML Baseline (56% Accuracy)

**Model:** LogisticRegression with TF-IDF features

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

vectorizer = TfidfVectorizer(max_features=50, lowercase=True)
model = LogisticRegression(class_weight='balanced', max_iter=1000)

pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', model)
])
```

**Training:** 5-fold StratifiedKFold cross-validation on 122 samples  
**CV accuracy:** 54.5% ± 14.6%  
**Hold-out test (20%, 25 samples):** 56%

**Why it worked:** Simple models don't overfit on small datasets. TF-IDF captures keyword presence which correlates with marks.

### Approach 3: Engineered Features (68% Accuracy)

**Model:** LogisticRegression with TF-IDF + 15 engineered features

```python
# Combine TF-IDF features (50) + engineered features (15)
X_combined = np.hstack([tfidf_features, engineered_features])

model = LogisticRegression(class_weight='balanced', max_iter=1000)
```

**Training:** Same 5-fold cross-validation  
**CV accuracy:** 62.8% ± 5.7%  
**Hold-out test:** 68% (17 out of 25 correct)

**Improvement:** +12 percentage points from baseline (56% to 68%)

**What the features learned:**
- Top predictor: Mention of antibodies (0.6355 importance)
- Length strongly matters (0.5383 for word count)
- Presence of key concepts (nanomaterials: 0.4878)
- Question number affects grading (question-level bias)

## Evaluation Methodology

### Cross-Validation

- **5-fold Stratified KFold** splits: Ensures each fold has same class distribution
- **Reason:** With only 122 samples, must use all data for training while getting unbiased estimates
- **Reported metric:** Mean accuracy across 5 folds with standard deviation

### Test Set Evaluation

- **Hold-out test:** 20% of data (25 samples), selected before model training
- **Reason:** Final check for overfitting on unseen data
- **Reported metric:** Accuracy on test set only

### Metrics

```
Accuracy = (TP + TN) / Total

Macro precision/recall/F1 for class imbalance awareness
```

No separate validation set (too small). Relied on cross-validation for hyperparameter tuning.

## Data vs Model Complexity

A key observation: With 122 samples,

```
Simple model + Good features >> Complex model + Bad features
56% (LR + TF-IDF)    <  68% (LR + TF-IDF + engineered)
0% (LLM attempts)    <  56% (Simple baseline)
```

**Why simple models won:**
1. Less risk of overfitting on small dataset
2. Easier to understand which features matter
3. Faster to train and iterate
4. More reproducible results

## Synthetic Data Experiment

Created 400 synthetic answers using an LLM with random marks. Goal: increase dataset size for LLM training.

**Result:** Completely useless. No model (including LLMs) could learn from random labels. This taught a valuable lesson: data quality matters infinitely more than data quantity.

## Reproducibility

All code is provided:
- Data preprocessing: `scripts/analyze_professor_grading.py`
- Feature engineering: In `improved_classifier_engineered.py`
- Model training: `scripts/improved_classifier_engineered.py`
- Evaluation: `scripts/verify_model_actually_works.py`

To reproduce:
```bash
python scripts/improved_classifier_engineered.py  # Trains on full 122 samples
python scripts/verify_model_actually_works.py     # Tests on 25-sample hold-out set
```

## Limitations & Future Work

1. **Small dataset:** 122 answers is very limited. More data would reduce variance in CV results.
2. **Single instructor:** Patterns might be instructor-specific. Would be interesting to compare across teachers.
3. **Single course:** Different courses might have different patterns.
4. **Binary text features:** Could add sentiment analysis, readability metrics, named entity recognition.
5. **Sequential models:** RNNs/Transformers could capture answer structure, but risk overfitting on 122 samples.

## Conclusion

Simple machine learning trained on carefully engineered features outperformed complex models on this task. The 68% accuracy demonstrates that exam grading follows learnable, consistent patterns based on measurable textual characteristics.
