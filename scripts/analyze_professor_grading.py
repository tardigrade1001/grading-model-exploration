#!/usr/bin/env python3
"""
ANALYZE: How does your Professor grade?
Grading patterns, question difficulty, what correlates with marks
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

print("\n" + "="*80)
print("  PROFESSOR'S GRADING ANALYSIS".center(80))
print("="*80 + "\n")

# Load raw data
df = pd.read_csv("exam_results_cleaned_final.csv")

print(f"Total answers analyzed: {len(df)}\n")

# ============================================================================
# 1. MARK DISTRIBUTION
# ============================================================================
print("[1] MARK DISTRIBUTION:\n")

mark_dist = df['normalized_mark'].value_counts().sort_index()
print("  Mark | Count | Percentage")
print("  " + "-" * 30)
for mark, count in mark_dist.items():
    pct = 100 * count / len(df)
    bar = "█" * int(pct / 2)
    print(f"  {mark:.1f}  |  {count:2d}   | {pct:5.1f}% {bar}")

print(f"\n  Mean: {df['normalized_mark'].mean():.2f}")
print(f"  Median: {df['normalized_mark'].median():.2f}")
print(f"  Std: {df['normalized_mark'].std():.2f}")
print(f"  Min: {df['normalized_mark'].min():.1f}")
print(f"  Max: {df['normalized_mark'].max():.1f}\n")

# ============================================================================
# 2. QUESTION DIFFICULTY
# ============================================================================
print("[2] QUESTION DIFFICULTY (by average mark):\n")

q_means = df.groupby('question_number')['normalized_mark'].agg(['mean', 'count', 'std'])
q_means = q_means.sort_values('mean', ascending=False)

print("  Question | Avg Mark | Answers | Std Dev | Difficulty")
print("  " + "-" * 55)

difficulty_map = {1: "Q1: Colorimetric biosensors vs HRP", 
                  2: "Q2: Biofunctionalization of AuNPs",
                  3: "Q3: Physicochemical properties",
                  4: "Q4: LOD definition and calculation"}

for q_num in [1, 2, 3, 4]:
    if q_num in df['question_number'].values:
        row = q_means.loc[q_num]
        mean_mark = row['mean']
        count = int(row['count'])
        std = row['std']
        
        if mean_mark >= 3.0:
            diff = "EASY (students answer well)"
        elif mean_mark >= 2.0:
            diff = "MEDIUM"
        else:
            diff = "HARD (students struggle)"
        
        print(f"  Q{q_num}      | {mean_mark:5.2f}    | {count:2d}      | {std:5.2f}  | {diff}")

print()

# ============================================================================
# 3. ANSWER LENGTH VS MARKS
# ============================================================================
print("[3] ANSWER LENGTH ANALYSIS:\n")

df['answer_length'] = df['transcribed_text'].str.len()
df['answer_words'] = df['transcribed_text'].str.split().str.len()

print("  By Mark | Avg Chars | Avg Words | Pattern")
print("  " + "-" * 55)

for mark in sorted(df['normalized_mark'].unique()):
    subset = df[df['normalized_mark'] == mark]
    avg_chars = subset['answer_length'].mean()
    avg_words = subset['answer_words'].mean()
    count = len(subset)
    
    if avg_chars > 500:
        pattern = "Long, detailed answers"
    elif avg_chars > 300:
        pattern = "Medium length"
    else:
        pattern = "Short/minimal answers"
    
    print(f"  {mark:.1f}     | {avg_chars:6.0f}   | {avg_words:6.1f}   | {pattern} ({count} answers)")

print()

# ============================================================================
# 4. KEY VOCABULARY BY MARK LEVEL
# ============================================================================
print("[4] WHAT WORDS DO HIGH-SCORING ANSWERS USE?\n")

high_mark_answers = df[df['normalized_mark'] >= 3.0]['transcribed_text']
low_mark_answers = df[df['normalized_mark'] <= 1.5]['transcribed_text']

def extract_keywords(texts):
    # Combine all text, lowercase, split by non-alphanumeric
    combined = " ".join(texts).lower()
    words = re.findall(r'\b[a-z]+\b', combined)
    
    # Remove common words
    stopwords = {'the', 'of', 'and', 'to', 'is', 'in', 'a', 'for', 'with', 'on', 'by', 'are', 'be', 'or', 'at', 'this', 'that', 'from', 'it', 'as', 'was', 'has', 'have', 'their', 'an', 'through', 'can', 'which', 'when', 'where', 'how', 'than', 'these', 'those', 'so', 'also', 'such'}
    
    words = [w for w in words if len(w) > 2 and w not in stopwords]
    
    return words

high_words = extract_keywords(high_mark_answers)
low_words = extract_keywords(low_mark_answers)

high_counter = Counter(high_words)
low_counter = Counter(low_words)

print("  High-scoring answers (3.0+) use:")
for word, count in high_counter.most_common(12):
    print(f"    • {word}")

print("\n  Low-scoring answers (≤1.5) use:")
for word, count in low_counter.most_common(12):
    print(f"    • {word}")

print()

# ============================================================================
# 5. SPECIFIC PATTERNS
# ============================================================================
print("[5] GRADING PATTERNS - WHAT MATTERS?\n")

# Check if mentioning specific concepts helps
concepts = {
    'antibody/antibodies': r'antibod',
    'specificity': r'specificit',
    'sensitivity': r'sensitivit',
    'surface area': r'surface area',
    'nanoparticle/nanomaterial': r'nano',
    'enzyme': r'enzyme',
    'TMB': r'tmb',
    'LOD/limit of detection': r'lod|limit.*detection',
    'calibration': r'calibr',
    'interference': r'interfer',
}

print("  Concept presence in answers by mark level:\n")
print("  Concept                | ≤1.5 Mark | 2.0-2.5 Mark | ≥3.0 Mark")
print("  " + "-" * 60)

for concept, pattern in concepts.items():
    low = (low_mark_answers.str.contains(pattern, case=False).sum() / len(low_mark_answers) * 100)
    mid = (df[(df['normalized_mark'] >= 2.0) & (df['normalized_mark'] <= 2.5)]['transcribed_text'].str.contains(pattern, case=False).sum() / len(df[(df['normalized_mark'] >= 2.0) & (df['normalized_mark'] <= 2.5)]) * 100)
    high = (high_mark_answers.str.contains(pattern, case=False).sum() / len(high_mark_answers) * 100)
    
    print(f"  {concept:<23} | {low:>8.0f}% | {mid:>11.0f}% | {high:>8.0f}%")

print()

# ============================================================================
# 6. CONSISTENCY CHECK
# ============================================================================
print("[6] GRADING CONSISTENCY:\n")

# Check if same student gets similar marks across questions
if 'student_id' in df.columns:
    student_marks = df.groupby('student_id')['normalized_mark'].std()
    print(f"  Student consistency:")
    print(f"    Average std dev across questions: {student_marks.mean():.2f}")
    print(f"    (0.0 = perfectly consistent, higher = inconsistent)")
    
    if student_marks.mean() < 0.5:
        print(f"    → CONSISTENT: Professor grades same student similarly across Qs")
    else:
        print(f"    → VARIABLE: Same student gets different marks on different Qs")
else:
    print("  [Student ID not available]")

print()

# ============================================================================
# 7. OUTLIERS
# ============================================================================
print("[7] OUTLIER ANSWERS:\n")

print("  Highest mark (4.0):")
high_4 = df[df['normalized_mark'] == 4.0]
if len(high_4) > 0:
    sample = high_4.iloc[0]
    text_preview = sample['transcribed_text'][:100]
    print(f"    Q{int(sample['question_number'])}: '{text_preview}...'")
    print(f"    Length: {len(sample['transcribed_text'])} chars, {sample['transcribed_text'].count(' ')+1} words")

print("\n  Lowest mark (1.0):")
low_1 = df[df['normalized_mark'] == 1.0]
if len(low_1) > 0:
    sample = low_1.iloc[0]
    text_preview = sample['transcribed_text'][:100]
    print(f"    Q{int(sample['question_number'])}: '{text_preview}...'")
    print(f"    Length: {len(sample['transcribed_text'])} chars, {sample['transcribed_text'].count(' ')+1} words")

print()

# ============================================================================
# 8. PROFESSOR'S GRADING PROFILE
# ============================================================================
print("[8] YOUR PROFESSOR'S GRADING PROFILE:\n")
print("="*80)

harsh = (df['normalized_mark'] <= 1.5).sum() / len(df) * 100
generous = (df['normalized_mark'] >= 3.0).sum() / len(df) * 100

print(f"  Strictness: {harsh:.0f}% of answers marked as Low (≤1.5)")
print(f"  Generosity: {generous:.0f}% of answers marked as High (≥3.0)")

if harsh > 40:
    print(f"  → STRICT GRADER: High bar, fewer high marks")
elif generous > 30:
    print(f"  → GENEROUS GRADER: Gives high marks more often")
else:
    print(f"  → BALANCED GRADER: Fair distribution of marks")

print(f"\n  Preferred mark: {df['normalized_mark'].mode()[0]:.1f} (most common)")
print(f"  Rarely gives: {df['normalized_mark'].min():.1f} (least common) and {df['normalized_mark'].max():.1f}")

print("="*80 + "\n")

# ============================================================================
# 9. ACTIONABLE INSIGHTS
# ============================================================================
print("[9] FOR STUDENTS - HOW TO SCORE HIGH WITH THIS PROFESSOR:\n")

print("  1. LENGTH MATTERS:")
high_avg_len = df[df['normalized_mark'] >= 3.0]['answer_length'].mean()
low_avg_len = df[df['normalized_mark'] <= 1.5]['answer_length'].mean()
print(f"     High-scoring answers: {high_avg_len:.0f} chars avg")
print(f"     Low-scoring answers:  {low_avg_len:.0f} chars avg")
print(f"     → Write detailed, not brief\n")

print("  2. KEY CONCEPTS TO MENTION:")
for concept, pattern in list(concepts.items())[:5]:
    high_pct = high_mark_answers.str.contains(pattern, case=False).sum() / len(high_mark_answers) * 100
    if high_pct > 60:
        print(f"     ✓ {concept} (appears in {high_pct:.0f}% of high marks)\n")

print("  3. QUESTION DIFFICULTY:")
for q_num in sorted(df['question_number'].unique()):
    q_mean = df[df['question_number'] == q_num]['normalized_mark'].mean()
    if q_mean < 2.0:
        print(f"     Q{q_num} is HARD (avg {q_mean:.1f}) - spend more time here\n")
    elif q_mean > 3.0:
        print(f"     Q{q_num} is EASY (avg {q_mean:.1f}) - most students do well\n")

print()
