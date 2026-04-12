#!/usr/bin/env python3
"""
4 Clean Graphs: Professor's Grading Biases
No annotations - just the data speaks for itself
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

df = pd.read_csv("../data/exam_results_cleaned_final.csv")
df['answer_length'] = df['transcribed_text'].str.len()

fig = plt.figure(figsize=(15, 10))

# ============================================================================
# GRAPH 1: LENGTH BIAS
# ============================================================================
ax1 = plt.subplot(2, 2, 1)

marks = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
lengths = [df[df['normalized_mark'] == m]['answer_length'].mean() for m in marks]
colors = ['#d73027' if m <= 1.5 else '#fee090' if m <= 2.5 else '#1a9850' for m in marks]

bars = ax1.bar(range(len(marks)), lengths, color=colors, edgecolor='black', linewidth=1.5)

for i, (bar, length) in enumerate(zip(bars, lengths)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{int(length)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax1.set_xticks(range(len(marks)))
ax1.set_xticklabels([f'{m}' for m in marks])
ax1.set_xlabel('Student Mark', fontweight='bold')
ax1.set_ylabel('Average Answer Length (characters)', fontweight='bold')
ax1.set_title('Answer Length vs Marks Received', fontweight='bold', fontsize=12, color='#d73027')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1050)

# ============================================================================
# GRAPH 2: CONCEPT CHECKLIST
# ============================================================================
ax2 = plt.subplot(2, 2, 2)

concepts = ['Antibody', 'Specificity', 'Sensitivity', 'Nanomaterial', 'TMB', 'LOD', 'Calibration']
low_marks = [17, 15, 10, 27, 2, 71, 63]
high_marks = [63, 91, 69, 74, 54, 6, 0]

x = np.arange(len(concepts))
width = 0.35

bars1 = ax2.bar(x - width/2, low_marks, width, label='Low Marks (≤1.5)', 
               color='#d73027', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, high_marks, width, label='High Marks (≥3.0)', 
               color='#1a9850', alpha=0.8, edgecolor='black', linewidth=1.5)

for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}%', ha='center', va='bottom', fontsize=7)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{int(height)}%', ha='center', va='bottom', fontsize=7)

ax2.set_ylabel('% of answers mentioning concept', fontweight='bold')
ax2.set_xlabel('Concept', fontweight='bold')
ax2.set_title('Concept Frequency by Mark Level', fontweight='bold', fontsize=12, color='#d73027')
ax2.set_xticks(x)
ax2.set_xticklabels(concepts, rotation=45, ha='right')
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3)

# ============================================================================
# GRAPH 3: Q4 CRUELTY
# ============================================================================
ax3 = plt.subplot(2, 2, 3)

questions = ['Q1', 'Q2', 'Q3', 'Q4']
avg_marks = [3.00, 2.36, 2.07, 1.38]
colors_q = ['#1a9850', '#fee090', '#fee090', '#d73027']

bars = ax3.bar(questions, avg_marks, color=colors_q, edgecolor='black', linewidth=2)

for bar, mark in zip(bars, avg_marks):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.08,
            f'{mark:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax3.axhline(y=2.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Average')
ax3.set_ylabel('Average Mark Given', fontweight='bold')
ax3.set_title('Average Marks Received by Question', fontweight='bold', fontsize=12, color='#d73027')
ax3.set_ylim(0, 3.5)
ax3.grid(axis='y', alpha=0.3)
ax3.legend(loc='upper right', fontsize=9)

# ============================================================================
# GRAPH 4: ML MODEL FEATURE IMPORTANCE
# ============================================================================
ax4 = plt.subplot(2, 2, 4)

features = ['Mentions\nAntibodies', 'Answer\nWord Count', 'Mentions\nNanomaterial', 
            'Is Detailed', 'Overuses\nCalibration', 'Answer\nLength']
importances = [0.6355, 0.5383, 0.4878, 0.4714, 0.4493, 0.4144]
colors_imp = ['#d73027', '#fc8d59', '#fee090', '#fee090', '#91bfdb', '#4575b4']

bars = ax4.barh(features, importances, color=colors_imp, edgecolor='black', linewidth=1.5)

for bar, imp in zip(bars, importances):
    width = bar.get_width()
    ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{imp:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

ax4.set_xlabel('Feature Importance (ML Model)', fontweight='bold')
ax4.set_title('Feature Importance in ML Model', fontweight='bold', fontsize=12, color='#d73027')
ax4.set_xlim(0, 0.75)
ax4.grid(axis='x', alpha=0.3)

# ============================================================================
# Overall title and layout
fig.suptitle('Exam Grading Pattern Analysis\nReverse-Engineered from 122 Real Exam Answers', 
            fontsize=16, fontweight='bold', y=0.998)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('grading_patterns_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ PNG saved: grading_patterns_analysis.png")

plt.savefig('grading_patterns_analysis.pdf', bbox_inches='tight', facecolor='white')
print("✓ PDF saved: grading_patterns_analysis.pdf")

plt.savefig('grading_patterns_analysis.svg', bbox_inches='tight', facecolor='white')
print("✓ SVG saved: grading_patterns_analysis.svg")

plt.show()

print("\n" + "="*80)
print("Outputs: PNG, PDF, SVG formats")
print("="*80 + "\n")