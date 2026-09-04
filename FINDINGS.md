# Findings

Four patterns showed up in this dataset. Three of them are one pattern seen from
different angles. This document works through which is which.

All tables come from `scripts/explore.py`, which reads the 97 training rows
only, and from `results/metrics.json`. Regenerate both before trusting them.

## The confound

The four questions are not four samples from one population. They differ in
marking, in topic vocabulary, and in expected answer length, all at once.

| Question | n | mean mark | mean length (chars) |
|---|---|---|---|
| Q1 | 21 | 3.07 | 784 |
| Q2 | 30 | 2.43 | 770 |
| Q3 | 22 | 2.09 | 770 |
| Q4 | 24 | 1.38 | 322 |

Class composition, training split:

| Question | Low | Mid | High |
|---|---|---|---|
| Q1 | 1 | 7 | 13 |
| Q2 | 5 | 15 | 10 |
| Q3 | 5 | 13 | 4 |
| Q4 | 22 | 1 | 1 |

Q4 is a single class with two exceptions, 22 Low answers out of 24. Any feature
that identifies Q4 will look like a strong predictor of a Low mark, and every
one of the pooled patterns below has a Q4-shaped hole in it.

## Pattern 1: concept words

Pooled across questions, concept presence separates the mark bands sharply.
Fraction of answers containing each term:

| Concept | Low | Mid | High |
|---|---|---|---|
| antibody | 0.15 | 0.50 | 0.64 |
| specificity | 0.15 | 0.53 | 0.93 |
| sensitivity | 0.12 | 0.44 | 0.68 |
| nanomaterial | 0.21 | 0.86 | 0.79 |
| tmb | 0.00 | 0.19 | 0.50 |
| binding | 0.18 | 0.56 | 0.57 |
| target | 0.27 | 0.64 | 0.79 |
| lod | 0.76 | 0.08 | 0.07 |
| calibration | 0.70 | 0.11 | 0.00 |

The first version of this project read that table as the instructor's rubric,
and described the marking as a checklist that rewards the word "specificity" and
penalizes the word "LOD".

Here is the same data grouped by question instead:

| Question | antibody | specificity | sensitivity | nanomaterial | tmb | binding | target | lod | calibration |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | 0.33 | 0.90 | 0.81 | 0.86 | 0.95 | 0.38 | 0.57 | 0.19 | 0.10 |
| Q2 | 0.60 | 0.77 | 0.23 | 0.70 | 0.03 | 0.80 | 0.83 | 0.03 | 0.07 |
| Q3 | 0.73 | 0.36 | 0.68 | 0.95 | 0.00 | 0.45 | 0.73 | 0.05 | 0.09 |
| Q4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 1.00 | 0.88 |

`lod` appears in every Q4 answer and almost nowhere else. `calibration` is
close behind. Q4 is 92% Low marks. `tmb` appears in 95% of Q1 answers and
almost nowhere else, and Q1 has the highest mean mark. The concept flags encode
which question was asked. The pooled table is that fact expressed through the
marks.

Q2 and Q3 are the only questions carrying both a spread of marks and a mix of
concepts, so they are the only place a real concept signal could show. Within
Q2:

**Q2**, n = 30 (Low 5, Mid 15, High 10)

| Band | antibody | specificity | sensitivity | nanomaterial | tmb | binding | target | lod | calibration |
|---|---|---|---|---|---|---|---|---|---|
| Low | 0.40 | 0.80 | 0.20 | 0.60 | 0.00 | 0.80 | 0.80 | 0.20 | 0.20 |
| Mid | 0.47 | 0.67 | 0.13 | 0.80 | 0.00 | 0.80 | 0.80 | 0.00 | 0.07 |
| High | 0.90 | 0.90 | 0.40 | 0.60 | 0.10 | 0.80 | 0.90 | 0.00 | 0.00 |

`specificity` separates nothing here. `binding` and `target` are flat across all
three bands. `antibody` does trend upward, on 10 High answers.

**Q3**, n = 22 (Low 5, Mid 13, High 4)

| Band | antibody | specificity | sensitivity | nanomaterial | tmb | binding | target | lod | calibration |
|---|---|---|---|---|---|---|---|---|---|
| Low | 0.60 | 0.20 | 0.60 | 0.80 | 0.00 | 0.40 | 0.80 | 0.20 | 0.20 |
| Mid | 0.69 | 0.23 | 0.62 | 1.00 | 0.00 | 0.54 | 0.62 | 0.00 | 0.08 |
| High | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.25 | 1.00 | 0.00 | 0.00 |

Within Q3, `specificity` and `sensitivity` do separate Low from High. That rests
on 5 Low answers and 4 High answers. It is worth another look on more data. It
is not a result.

**The claim that survives:** on these 122 answers, a concept-presence vocabulary
mostly encodes which question was asked. What separation remains inside a single
question is small and sits on single-digit sample sizes.

## Pattern 2: answer length

Pooled across questions, length rises with mark. Correlation between character
count and mark is r = 0.62 on the training split.

Length also tracks the question. Q4 answers average 322 characters against
roughly 775 for the other three, and Q4 carries the lowest marks. Part of that
0.62 is the question difference.

The part that is not is the interesting one. Correlation between length and mark
computed inside each question, where question identity is held constant:

| Question | n | Pearson r | Spearman r |
|---|---|---|---|
| Q1 | 21 | 0.36 | 0.41 |
| Q2 | 30 | 0.59 | 0.54 |
| Q3 | 22 | 0.52 | 0.53 |
| Q4 | 24 | 0.65 | 0.62 |

The relationship holds within every question. It is weaker than the pooled 0.62
in three of the four, and the gap is the size of the confound.

This is the one pattern in the dataset that is not explained by question
identity. It also drives the whole model: `length_only` scores 64.3% in
cross-validation, matching the 47-feature `full` model at 63.2%.

**On what it means.** Two readings fit these numbers equally well. Longer
answers may cover more of the expected content, in which case length is a proxy
for the thing being marked. Or the marking may reward volume directly. This
dataset has one marker, one paper, and no independent measure of answer quality,
so it cannot separate them. Nothing here shows that writing more causes a higher
mark.

## Pattern 3: question difficulty

Mean marks run from 3.07 on Q1 down to 1.38 on Q4 in the training split, a gap
of 1.7 points on a 4-point scale. Over all 122 answers the means are 3.00 and
1.38. This is real and it is large.

What it means is undetermined. Q4 asks for the limit of detection and its
derivation from a calibration line, which is the most quantitative of the four
questions. The gap is consistent with the question being harder, with it being
marked more strictly, with it being taught less well, and with students running
short of time on it. Q4 answers are also the shortest, which fits several of
those at once.

Distinguishing them needs the rubric, or a second marker, or the same question
set with a different cohort. None of those is in this dataset.

## Pattern 4: what the model learned

The `full` model reaches 62.3% on out-of-fold predictions. The per-question
majority rule, which reads no text, reaches 63.1%.

| Question | n | `full` model | Question-majority rule |
|---|---|---|---|
| Q1 | 29 | 0.586 | 0.586 |
| Q2 | 35 | 0.514 | 0.486 |
| Q3 | 30 | 0.500 | 0.567 |
| Q4 | 28 | 0.929 | 0.929 |
| **All** | 122 | **0.623** | **0.631** |

Q4 supplies the accuracy and the majority rule supplies it just as well. Q1
ties. Q3 loses to the rule. Q2 wins by one answer in 35, which is noise.

Out-of-fold confusion for the `full` model:

| actual \ predicted | Low | Mid | High |
|---|---|---|---|
| Low | 28 | 13 | 0 |
| Mid | 8 | 24 | 14 |
| High | 1 | 10 | 24 |

Errors sit almost entirely on adjacent bands. One High answer out of 35 is
called Low, and no Low answer is called High. The model orders answers roughly
correctly and places the boundaries badly, which is what an ordinal target
squeezed into nominal classes tends to look like.

## What to conclude

On 122 answers to 4 questions marked by one instructor:

- Question identity accounts for essentially all of the predictable variation.
- Answer length correlates with mark inside each question, r between 0.36 and
  0.65, and it is the only pattern that survives the confound.
- Concept-presence features encode the question. They do not encode a rubric.
- No text model in the ladder beats the text-free per-question majority rule.

The question of whether grading patterns are learnable from answer text stays
open. This dataset is too small, too concentrated in one question, and too
confounded to answer it. Answering it needs several papers, more than one
marker, and enough answers per question to evaluate within each question.
