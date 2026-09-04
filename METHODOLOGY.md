# Methodology

How the evaluation in `results/` is set up, and why each choice was made.

## Task

Given the text of a student's exam answer and the number of the question it
answers, predict which of three bands the instructor's mark falls into.

| band | marks | count |
|---|---|---|
| Low | 1.0, 1.5 | 41 |
| Mid | 2.0, 2.5 | 46 |
| High | 3.0, 3.5, 4.0 | 35 |

The underlying mark is ordinal with seven levels. Banding into three throws
away ordering information and it keeps every class large enough to stratify a
5-fold split at n=122. Seven-level classification on 122 rows puts one answer in
the 3.5 class. That is not a class. The banding is defined once in
`src/grading/data.py` and imported everywhere, so no script can quietly use a
different one.

## The baseline ladder

A model on this data is only interesting if it beats what you get for free.
Four floors, in order of how much each concedes to the model:

**`majority_class`** predicts Mid for everything. 37.7%.

**`question_only`** sees the question number and nothing else. 60.9%. This is
the number that matters. The four questions differ sharply in marking, most of
all Q4, which holds 26 Low marks out of 28. A model that reaches 63% while
knowing the question has demonstrated very little.

**`length_only`** sees the character count, the word count, and two
length flags. 64.3%.

**`per_question_majority`** predicts each question's most common training
class. 63.1%. It reads no answer text. It is a fitted estimator like every
other entry in the ladder, so it learns its question-to-class map on the
training rows of each fold and is scored on the held-out rows of that fold.
That is what makes the comparison with the text models fair. On this data the
map is stable enough that the in-sample, out-of-fold and cross-validated
figures all land on 63.1%.

The remaining models add TF-IDF and concept flags on top. `results/RESULTS.md`
carries the full table.

## Preventing leakage

Every model is a single sklearn `Pipeline` that takes the raw dataframe. Fitting
a pipeline fits its vectoriser, its scaler, and its length thresholds on the
rows it is handed and nothing else. The same object then goes to
`cross_val_score`, to `cross_val_predict`, and to the held-out split, so the
boundary is enforced by construction and there is no manual bookkeeping to get
wrong.

Two specific leaks in the first version of this project are closed by this.

**Vectoriser fitted on everything.** The earlier notebooks called
`TfidfVectorizer.fit_transform` on all 122 answers, then split. The vocabulary
and the document frequencies had seen the test rows.

**Thresholds chosen by reading the whole dataset.** `is_detailed` was defined
as 600 characters or more, and `is_minimal` as under 400, after reading that
high marks averaged 953 characters and low marks 397 across all 122 answers.
`LengthFeatures` in `src/grading/features.py` now learns those cut-offs in
`fit`, as the 75th and 25th percentile of the training rows it is given.

One thing this does not close. The concept vocabulary in `CONCEPT_PATTERNS` was
written by hand after looking at the data. It is fixed before any model is
fitted, so it does not move between folds, and the effect it does have is
visible in the `concepts_only` row of the ladder. It is disclosed here because
disclosure is the only available fix. The vocabulary is small, it is drawn from
the exam questions, and `FINDINGS.md` shows that what it mostly encodes is
question identity.

## Separating discovery from evaluation

`scripts/explore.py` reads only the 97 training rows of the split that
`scripts/evaluate.py` holds out. It prints tables and correlations and it prints
no accuracy. Anything it turns up can become a feature without touching the
reported metrics.

`scripts/evaluate.py` prints every accuracy in the repository and discovers
nothing. It writes `results/metrics.json` and `results/RESULTS.md`, and every
table in `README.md`, `METHODOLOGY.md` and `FINDINGS.md` is copied from those.
`scripts/make_figures.py` draws from `results/metrics.json` for the same reason,
so a panel cannot disagree with a table.

## What is reported and why

**Repeated stratified 5-fold cross-validation, 10 repeats.** 50 fits per model.
The mean is stable enough to compare models. The standard deviation across the
50 fits shows how far apart two models have to be before the difference means
anything, and on this data almost none of them are.

**A single stratified 80/20 held-out split, with a Wilson interval.** Kept
because it is what the first version reported, and shown with its interval so
the reason to distrust it is visible. 25 test answers gives a 95% interval
roughly 35 percentage points wide for every model in the ladder. The Wilson
form is used because the normal approximation is unreliable at that size.

**A split-sensitivity sweep.** The `full` model refitted across 20 different
stratified 80/20 splits of the same data. The results run from 40% to 76%. Any
single held-out number from this dataset reports the split as much as it
reports the model.

**A paired comparison against the baseline.** `cross_val_score` visits the
same 50 folds in the same order for every model, so the score vectors are
aligned fold for fold. The `full` model and `per_question_majority` are
differenced on those pairs and tested with a Wilcoxon signed-rank test. Two
mean accuracies whose standard deviations overlap this much cannot be compared
directly. The paired test can.

**Out-of-fold predictions by question.** Predictions for all 122 answers, each
made by a model that did not see it, broken down by question and set against
the per-question majority rule. This is the evaluation that answers the actual
research question, and the first version of this project never ran it.

## Reading the coefficients

Logistic regression coefficients are not reported as feature importances
anywhere in this repository, and the earlier ranking of them has been removed.

The features are correlated with each other and with question identity.
`mentions_lod` and `is_q4` fire on nearly the same rows. TF-IDF terms overlap
with the concept flags. Under that correlation the coefficients split the shared
signal in a way that depends on the fold and on the regularization strength, so
a ranking of them is not stable and does not say what drives a prediction.

The ladder in `results/RESULTS.md` answers the same question better. Fit a model
on one feature group at a time and read what each group is worth on its own.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/evaluate.py
python scripts/make_figures.py
```

The default seed is 0, set in `src/grading/models.py`. `results/metrics.json`
records the seed and the versions of Python, scikit-learn, numpy and pandas it
ran under. Pass `--seed` to `evaluate.py` to see how much of the output moves.
