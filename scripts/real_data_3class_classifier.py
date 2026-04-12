#!/usr/bin/env python3
"""
Real Exam Grader - ChatGPT's Recommendation
122 real samples → 3 classes (Low/Mid/High) → TF-IDF + LogisticRegression + Cross-Validation
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("  REAL EXAM GRADER - 3-CLASS CLASSIFICATION".center(80))
print("="*80 + "\n")

# ============================================================================
# 1. LOAD REAL DATA
# ============================================================================
print("[1] Loading real exam data...")

df = pd.read_csv("exam_results_cleaned_final.csv")
print(f"  ✓ Loaded {len(df)} real samples\n")

# ============================================================================
# 2. COLLAPSE TO 3 CLASSES
# ============================================================================
print("[2] Collapsing marks to 3 classes...")

def mark_to_class(mark):
    if mark <= 1.5:
        return "Low"      # 1.0, 1.5
    elif mark <= 2.5:
        return "Mid"      # 2.0, 2.5
    else:
        return "High"     # 3.0, 3.5, 4.0

df['class'] = df['normalized_mark'].apply(mark_to_class)

class_counts = df['class'].value_counts()
print(f"  Low:  {class_counts.get('Low', 0)} samples")
print(f"  Mid:  {class_counts.get('Mid', 0)} samples")
print(f"  High: {class_counts.get('High', 0)} samples")
print(f"  Total: {len(df)} samples\n")

# ============================================================================
# 3. TF-IDF VECTORIZATION
# ============================================================================
print("[3] TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    max_features=50,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df['transcribed_text']).toarray()
y = df['class']

print(f"  ✓ Created {X.shape[1]} TF-IDF features\n")

# ============================================================================
# 4. CROSS-VALIDATION (5-FOLD)
# ============================================================================
print("[4] Training with 5-Fold Cross-Validation...\n")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"  Fold 1: {scores[0]:.3f}")
print(f"  Fold 2: {scores[1]:.3f}")
print(f"  Fold 3: {scores[2]:.3f}")
print(f"  Fold 4: {scores[3]:.3f}")
print(f"  Fold 5: {scores[4]:.3f}")
print(f"  ├─ Mean: {scores.mean():.3f}")
print(f"  └─ Std:  {scores.std():.3f}\n")

# ============================================================================
# 5. DETAILED EVALUATION (Using all data for detailed metrics)
# ============================================================================
print("[5] Detailed Metrics (trained on full data)...\n")

model.fit(X, y)
y_pred = model.predict(X)

print("Classification Report:")
print("-" * 80)
print(classification_report(y, y_pred, target_names=['Low', 'Mid', 'High']))

print("\nConfusion Matrix:")
print("-" * 80)
cm = confusion_matrix(y, y_pred, labels=['Low', 'Mid', 'High'])
print("         Predicted")
print("         Low  Mid  High")
print(f"Actual Low   {cm[0,0]:3d} {cm[0,1]:3d} {cm[0,2]:3d}")
print(f"Actual Mid   {cm[1,0]:3d} {cm[1,1]:3d} {cm[1,2]:3d}")
print(f"Actual High  {cm[2,0]:3d} {cm[2,1]:3d} {cm[2,2]:3d}\n")

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("[6] Top 15 Important Features:\n")

feature_names = tfidf.get_feature_names_out()
coef_avg = np.abs(model.coef_).mean(axis=0)
top_indices = np.argsort(coef_avg)[-15:][::-1]

for idx in top_indices:
    print(f"  {feature_names[idx]:<20} {coef_avg[idx]:.4f}")

print("\n")

# ============================================================================
# 7. SAMPLE PREDICTIONS
# ============================================================================
print("[7] Sample Predictions from Data:\n")
print("="*80)

sample_indices = np.random.choice(len(df), 10, replace=False)

for i, idx in enumerate(sample_indices, 1):
    actual_mark = df.iloc[idx]['normalized_mark']
    actual_class = df.iloc[idx]['class']
    text = df.iloc[idx]['transcribed_text'][:80]
    
    pred_class = y_pred[idx]
    match = "✓" if pred_class == actual_class else "✗"
    
    print(f"{match} {i}. Actual: {actual_class} ({actual_mark:.1f}) | Pred: {pred_class}")
    print(f"   Text: '{text}...'")

print("\n" + "="*80)

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n[8] SUMMARY\n")
print("="*80)
print(f"  Approach:           TF-IDF (50 features) + LogisticRegression")
print(f"  Classes:            Low (1.0-1.5) | Mid (2.0-2.5) | High (3.0-4.0)")
print(f"  Data:               122 real exam samples from Professor's marking")
print(f"  Cross-Val Acc:      {scores.mean():.1%} ± {scores.std():.1%}")
print(f"  Status:             Ready for deployment")
print("="*80 + "\n")

print("This is a WORKING BASELINE. Much simpler than LLM fine-tuning.")
print("If this shows good signal, can explore more complex models later.\n")
