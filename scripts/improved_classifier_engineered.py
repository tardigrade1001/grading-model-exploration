#!/usr/bin/env python3
"""
IMPROVED MODEL: Engineer features from Professor's patterns
- Answer length (2.4x difference between high/low)
- Key concepts that boost marks
- Avoid bad patterns
- Question-specific difficulty
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("  IMPROVED MODEL: ENGINEERED FEATURES".center(80))
print("="*80 + "\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
df = pd.read_csv("exam_results_cleaned_final.csv")

def mark_to_class(mark):
    if mark <= 1.5: return "Low"
    elif mark <= 2.5: return "Mid"
    else: return "High"

df['class'] = df['normalized_mark'].apply(mark_to_class)

# ============================================================================
# 2. ENGINEER FEATURES (Domain-based, from professor's grading)
# ============================================================================
print("[1] Engineering domain-based features...\n")

# Feature 1: Answer length (we know this matters!)
df['answer_length'] = df['transcribed_text'].str.len()
df['answer_words'] = df['transcribed_text'].str.split().str.len()

# Feature 2: Key concepts that appear in HIGH-scoring answers
concepts_good = {
    'mentions_antibody': r'\banti\w*bod\w*',
    'mentions_specificity': r'\bspecificit\w*',
    'mentions_sensitivity': r'\bsensitivit\w*',
    'mentions_nanomaterial': r'\bnano\w*',
    'mentions_tmb': r'\btmb\b',
    'mentions_binding': r'\bbind\w*',
    'mentions_target': r'\btarget\w*',
}

for feature, pattern in concepts_good.items():
    df[feature] = df['transcribed_text'].str.contains(pattern, case=False, na=False).astype(int)

# Feature 3: Bad patterns (appear in LOW-scoring answers)
concepts_bad = {
    'overuses_lod': r'\blod\b|\blimit.*detection\b',
    'overuses_calibration': r'\bcalibr\w*',
}

for feature, pattern in concepts_bad.items():
    df[feature] = df['transcribed_text'].str.contains(pattern, case=False, na=False).astype(int)

# Feature 4: Answer length category (length is strongest signal!)
df['is_detailed'] = (df['answer_length'] >= 600).astype(int)  # High-scoring avg is 953
df['is_minimal'] = (df['answer_length'] < 400).astype(int)    # Low-scoring avg is 397

# Feature 5: Question difficulty (Q4 is much harder)
df['is_q4'] = (df['question_number'] == 4).astype(int)
df['is_q1'] = (df['question_number'] == 1).astype(int)

print("  Features created:")
print(f"    ✓ Length features: answer_length, answer_words, is_detailed, is_minimal")
print(f"    ✓ Good concept features: {len(concepts_good)} concepts")
print(f"    ✓ Bad concept features: {len(concepts_bad)} concepts")
print(f"    ✓ Question features: is_q4, is_q1\n")

# ============================================================================
# 3. SPLIT DATA (proper indexing)
# ============================================================================
print("[2] Splitting data (80/20)...\n")

# Create indices for split
all_indices = np.arange(len(df))
train_idx_array, test_idx_array = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=42,
    stratify=df['class'].values
)

# Get train and test data using indices
df_train = df.iloc[train_idx_array].reset_index(drop=True)
df_test = df.iloc[test_idx_array].reset_index(drop=True)

X_train_text = df_train['transcribed_text'].values
X_test_text = df_test['transcribed_text'].values
y_train = df_train['class'].values
y_test = df_test['class'].values

# Get engineered features using the same indices
feature_cols = list(concepts_good.keys()) + list(concepts_bad.keys()) + \
               ['answer_length', 'answer_words', 'is_detailed', 'is_minimal', 'is_q4', 'is_q1']

X_train_engineered = df_train[feature_cols].reset_index(drop=True)
X_test_engineered = df_test[feature_cols].reset_index(drop=True)

print(f"  Train: {len(X_train_text)} samples")
print(f"  Test: {len(X_test_text)} samples")
print(f"  Train engineered shape: {X_train_engineered.shape}")
print(f"  Test engineered shape: {X_test_engineered.shape}\n")

# ============================================================================
# 4. TF-IDF (on train data only)
# ============================================================================
print("[3] TF-IDF Vectorization (fit on train)...\n")

tfidf = TfidfVectorizer(max_features=30, min_df=2, max_df=0.8, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
X_test_tfidf = tfidf.transform(X_test_text).toarray()

tfidf_cols = [f"tfidf_{i}" for i in range(X_train_tfidf.shape[1])]

# ============================================================================
# 5. COMBINE: Engineered + TF-IDF
# ============================================================================
print("[4] Combining engineered features + TF-IDF...\n")

X_train_combined = np.hstack([
    X_train_engineered.values,
    X_train_tfidf
])

X_test_combined = np.hstack([
    X_test_engineered.values,
    X_test_tfidf
])

# Normalize (length features are much larger)
scaler = StandardScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_test_combined = scaler.transform(X_test_combined)

print(f"  Total features: {X_train_combined.shape[1]}")
print(f"    - Engineered: {len(feature_cols)}")
print(f"    - TF-IDF: {len(tfidf_cols)}\n")

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================
print("[5] 5-Fold Cross-Validation:\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_combined, y_train), 1):
    X_train_fold = X_train_combined[train_idx]
    X_val_fold = X_train_combined[val_idx]
    y_train_fold = y_train[train_idx]
    y_val_fold = y_train[val_idx]
    
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_fold, y_train_fold)
    
    val_acc = model.score(X_val_fold, y_val_fold)
    fold_scores.append(val_acc)
    
    print(f"  Fold {fold}: {val_acc:.1%}")

print(f"\n  Mean: {np.mean(fold_scores):.1%} ± {np.std(fold_scores):.1%}\n")

# ============================================================================
# 7. FINAL MODEL ON HOLD-OUT TEST
# ============================================================================
print("[6] Final Model on HOLD-OUT TEST SET:\n")

final_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
final_model.fit(X_train_combined, y_train)

y_test_pred = final_model.predict(X_test_combined)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"  HOLD-OUT TEST ACCURACY: {test_acc:.1%} ({int(test_acc * len(y_test))}/{len(y_test)} correct)\n")

print("  Classification Report:")
print("  " + "-" * 70)
print(classification_report(y_test, y_test_pred, target_names=['Low', 'Mid', 'High']))

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("[7] Feature Importance (Top 15):\n")

all_feature_names = feature_cols + tfidf_cols
coef_avg = np.abs(final_model.coef_).mean(axis=0)
top_indices = np.argsort(coef_avg)[-15:][::-1]

for idx in top_indices:
    fname = all_feature_names[idx]
    importance = coef_avg[idx]
    bar = "█" * int(importance * 50)
    print(f"  {fname:<30} {bar} {importance:.4f}")

print()

# ============================================================================
# 9. COMPARISON: BASELINE vs IMPROVED
# ============================================================================
print("[8] COMPARISON:\n")
print("="*80)

# Train baseline (TF-IDF only)
tfidf_only = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
tfidf_only.fit(X_train_tfidf, y_train)
baseline_acc = tfidf_only.score(X_test_tfidf, y_test)

print(f"  BASELINE (TF-IDF only):           {baseline_acc:.1%}")
print(f"  IMPROVED (Engineered + TF-IDF):  {test_acc:.1%}")
print(f"  IMPROVEMENT:                      +{(test_acc - baseline_acc)*100:.1f} percentage points")
print("="*80 + "\n")

# ============================================================================
# 10. SAMPLE PREDICTIONS
# ============================================================================
print("[9] Sample Predictions (Improved Model):\n")
print("="*80)

sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
correct = 0

for i, idx in enumerate(sample_indices, 1):
    actual = y_test[idx]
    pred = y_test_pred[idx]
    text = X_test_text[idx][:70]
    
    match = "✓" if actual == pred else "✗"
    if actual == pred:
        correct += 1
    
    print(f"{match} {i}. Actual: {actual:<5} | Pred: {pred:<5} | '{text}...'")

print(f"\n  Sample accuracy: {correct}/{len(sample_indices)} ({100*correct/len(sample_indices):.0f}%)")
print("="*80 + "\n")

# ============================================================================
# 11. KEY INSIGHTS
# ============================================================================
print("[10] INSIGHTS:\n")

# Which engineered features helped most?
engineered_importance = coef_avg[:len(feature_cols)]
top_engineered = np.argsort(engineered_importance)[-5:][::-1]

print("  Most important ENGINEERED features:")
for idx in top_engineered:
    if engineered_importance[idx] > 0:
        fname = feature_cols[idx]
        imp = engineered_importance[idx]
        print(f"    • {fname:<30} ({imp:.4f})")

print()
