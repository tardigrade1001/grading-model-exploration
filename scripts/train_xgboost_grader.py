#!/usr/bin/env python3
"""
Exam Grader using XGBoost + TF-IDF Features
Non-LLM approach: Extract features → Train XGBoost → Predict marks
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("  EXAM GRADER - XGBoost + TF-IDF FEATURES".center(80))
print("="*80 + "\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("[1] Loading data...")
df = pd.read_csv("synthetic_exam_data_categorical.csv")
print(f"  ✓ Loaded {len(df)} samples\n")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("[2] Feature Engineering...")

# Convert marks to numeric if needed
if df['marks_obtained'].dtype == 'object':
    df['marks_obtained'] = df['marks_obtained'].astype(float)

# Basic text features
df['answer_length'] = df['answer_text'].str.len()
df['answer_words'] = df['answer_text'].str.split().str.len()
df['answer_sentences'] = df['answer_text'].str.count(r'[.!?]') + 1

# Convert grade back to numeric for training
grade_to_numeric = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0}
df['grade_numeric'] = df['grade'].map(grade_to_numeric)

print(f"  ├─ Answer length: {df['answer_length'].min()}-{df['answer_length'].max()} chars")
print(f"  ├─ Answer words: {df['answer_words'].min()}-{df['answer_words'].max()} words")
print(f"  ├─ Sentences: {df['answer_sentences'].min()}-{df['answer_sentences'].max()} sentences")
print(f"  └─ Marks range: {df['marks_obtained'].min()}-{df['marks_obtained'].max()}\n")

# ============================================================================
# 3. TF-IDF VECTORIZATION
# ============================================================================
print("[3] TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    max_features=50,  # Top 50 important words
    min_df=2,         # Word must appear in at least 2 docs
    max_df=0.8,       # Word can appear in max 80% of docs
    ngram_range=(1, 2)  # Unigrams and bigrams
)

tfidf_features = tfidf.fit_transform(df['answer_text']).toarray()
tfidf_df = pd.DataFrame(
    tfidf_features,
    columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
)

print(f"  ✓ Created {tfidf_features.shape[1]} TF-IDF features\n")

# ============================================================================
# 4. COMBINE ALL FEATURES
# ============================================================================
print("[4] Combining features...")

# Question one-hot encoding
question_dummies = pd.get_dummies(df['question_number'], prefix='question')

X = pd.concat([
    df[['answer_length', 'answer_words', 'answer_sentences']],
    tfidf_df,
    question_dummies
], axis=1)

y = df['marks_obtained']

print(f"  ✓ Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features\n")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("[5] Train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"  ├─ Training set: {len(X_train)} samples")
print(f"  ├─ Test set: {len(X_test)} samples")
print(f"  └─ Train/test ratio: {len(X_train)/len(X_test):.1f}:1\n")

# ============================================================================
# 6. TRAIN XGBOOST
# ============================================================================
print("[6] Training XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print("  ✓ Training complete\n")

# ============================================================================
# 7. EVALUATE
# ============================================================================
print("[7] Evaluation on Test Set...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  ├─ MSE:  {mse:.4f}")
print(f"  ├─ RMSE: {rmse:.4f}")
print(f"  ├─ MAE:  {mae:.4f}")
print(f"  └─ R²:   {r2:.4f}\n")

# ============================================================================
# 8. SAMPLE PREDICTIONS
# ============================================================================
print("[8] Sample Predictions on Test Set:\n")
print("="*80)

for i in range(min(10, len(X_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    
    # Grade conversion
    def mark_to_grade(mark):
        if mark >= 3.5: return "A"
        elif mark >= 2.5: return "B"
        elif mark >= 1.5: return "C"
        else: return "D"
    
    actual_grade = mark_to_grade(actual)
    pred_grade = mark_to_grade(predicted)
    
    match = "✓" if actual_grade == pred_grade else "✗"
    
    print(f"{match} Actual: {actual:.1f} ({actual_grade}) | Predicted: {predicted:.2f} ({pred_grade}) | Error: {error:.2f}")

print("="*80 + "\n")

# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
print("[9] Top 15 Important Features:\n")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"  {row['feature']:<25} {bar} {row['importance']:.4f}")

print("\n")

# ============================================================================
# 10. SAVE MODEL
# ============================================================================
print("[10] Saving model...")

model.save_model("xgboost_exam_grader.json")

# Save feature names
with open("xgboost_features.txt", "w") as f:
    for feat in X.columns:
        f.write(feat + "\n")

print("  ✓ Model saved: xgboost_exam_grader.json")
print("  ✓ Features saved: xgboost_features.txt\n")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("="*80)
print("  SUMMARY".center(80))
print("="*80)
print(f"  Model Type:        XGBoost Regressor")
print(f"  Features:          {X.shape[1]} (TF-IDF + text stats + question)")
print(f"  Training samples:  {len(X_train)}")
print(f"  Test samples:      {len(X_test)}")
print(f"  RMSE:              {rmse:.4f} (marks out of 4.0)")
print(f"  MAE:               {mae:.4f}")
print(f"  R²:                {r2:.4f}")
print(f"  Status:            ✓ Ready to use")
print("="*80 + "\n")

print("Next: python test_xgboost_grader.py\n")
