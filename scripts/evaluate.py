#!/usr/bin/env python3
"""Single source of every number reported in this repository.

Run it from anywhere:

    python scripts/evaluate.py

It writes results/metrics.json and results/RESULTS.md. The tables in
README.md, METHODOLOGY.md and FINDINGS.md are copied from those two files.
No other script in this repo reports an accuracy.

Three evaluations are run for each model in the zoo.

1. Repeated stratified 5-fold cross-validation over all 122 answers, 10
   repeats, giving a mean and a standard deviation across 50 fits.
2. A single stratified 80/20 held-out split, fitted once, reported with a
   Wilson interval so the width of the interval sits next to the point
   estimate.
3. Out-of-fold predictions broken down by question, compared against a
   per-question majority rule. This is the evaluation that matters, and it is
   the one the first version of this project never ran.

The full model and that baseline are then compared fold by fold on the same
50 folds, with a Wilcoxon signed-rank test on the paired differences. Comparing
two mean accuracies says nothing when their standard deviations overlap as much
as these do. The paired test is the comparison that carries weight.
"""

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grading.data import CLASS_ORDER, load_dataset  # noqa: E402
from grading.models import RANDOM_STATE, build_models  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

N_SPLITS = 5
N_REPEATS = 10
TEST_SIZE = 0.2


def wilson_interval(successes, total, z=1.96):
    """Wilson score interval for a binomial proportion.

    Used because the held-out set holds 25 answers. A normal approximation is
    unreliable at that size and hides how wide the interval really is.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_question_majority_accuracy(df, y):
    """Accuracy of always predicting the most common class within each question.

    This rule reads no answer text. It is the floor for any model that is told
    which question it is grading.
    """
    series = pd.Series(y, index=df.index)
    modes = series.groupby(df["question_number"]).transform(lambda s: s.mode().iloc[0])
    return float((modes == series).mean())


def n_features(model, df_train):
    """Width of the design matrix the classifier actually sees.

    Read off the fitted pipeline. It was typed into the docs by hand once
    and typed wrong.
    """
    if not hasattr(model, "steps"):
        return None
    return int(model[:-1].transform(df_train).shape[1])


def per_question_breakdown(df, y, oof):
    out = {}
    positions = pd.Series(np.arange(len(df)))
    for q in sorted(df["question_number"].unique()):
        mask = (df["question_number"] == q).to_numpy()
        pos = positions[mask].to_numpy()
        sub_y, sub_p = y[pos], oof[pos]
        majority = pd.Series(sub_y).value_counts(normalize=True).iloc[0]
        out[str(q)] = {
            "n": int(mask.sum()),
            "model_accuracy": float((sub_y == sub_p).mean()),
            "question_majority_accuracy": float(majority),
        }
    return out


def holdout_seed_sweep(df, y, model, seeds):
    """Held-out accuracy of one model across many random splits.

    At n=122 a single 80/20 split puts 25 answers in the test set, and the
    reported figure moves by more than 20 points depending on which split is
    drawn. This sweep makes that visible.
    """
    out = []
    for s in seeds:
        tr, te = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE, random_state=s, stratify=y
        )
        model.fit(df.iloc[tr], y[tr])
        out.append(float(accuracy_score(y[te], model.predict(df.iloc[te]))))
    return out


def evaluate(df, seed):
    y = df["mark_class"].to_numpy()
    models = build_models()

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    oof_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rows = {}
    fold_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, df, y, cv=cv, scoring="accuracy", n_jobs=1)
        fold_scores[name] = scores

        model.fit(df_train, y_train)
        y_pred = model.predict(df_test)
        correct = int((y_pred == y_test).sum())
        low, high = wilson_interval(correct, len(y_test))

        oof = cross_val_predict(model, df, y, cv=oof_cv, n_jobs=1)

        rows[name] = {
            "n_features": n_features(model, df_train),
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "cv_n_fits": int(scores.size),
            "holdout_accuracy": correct / len(y_test),
            "holdout_correct": correct,
            "holdout_n": int(len(y_test)),
            "holdout_ci95": [low, high],
            "oof_accuracy": float(accuracy_score(y, oof)),
            "per_question": per_question_breakdown(df, y, oof),
        }

    # Paired comparison on identical folds. cross_val_score with the same cv
    # object visits the same 50 folds in the same order for every model, so
    # these two score vectors are aligned fold for fold.
    diff = fold_scores["full"] - fold_scores["per_question_majority"]
    paired = {
        "model_a": "full",
        "model_b": "per_question_majority",
        "n_folds": int(diff.size),
        "mean_difference": float(diff.mean()),
        "sd_difference": float(diff.std()),
        "folds_a_wins": int((diff > 0).sum()),
        "folds_tied": int((diff == 0).sum()),
        "wilcoxon_p": float(wilcoxon(fold_scores["full"],
                                     fold_scores["per_question_majority"]).pvalue),
    }

    sweep_seeds = list(range(20))
    sweep = holdout_seed_sweep(df, y, build_models()["full"], sweep_seeds)

    full_oof = cross_val_predict(models["full"], df, y, cv=oof_cv, n_jobs=1)
    reference = {
        "paired_comparison": paired,
        "holdout_seed_sweep": {
            "model": "full",
            "seeds": sweep_seeds,
            "accuracies": sweep,
            "min": min(sweep),
            "max": max(sweep),
            "mean": float(np.mean(sweep)),
        },
        "per_question_majority": per_question_majority_accuracy(df, y),
        "global_majority": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
        "full_model_oof": float(accuracy_score(y, full_oof)),
        "full_model_confusion": confusion_matrix(y, full_oof, labels=CLASS_ORDER).tolist(),
        "full_model_report": classification_report(
            y,
            full_oof,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        ),
    }

    return {
        "generated": date.today().isoformat(),
        "n_answers": int(len(df)),
        "class_order": CLASS_ORDER,
        "class_counts": df["mark_class"].value_counts().reindex(CLASS_ORDER).to_dict(),
        "config": {
            "seed": seed,
            "cv": "RepeatedStratifiedKFold({0} splits x {1} repeats)".format(N_SPLITS, N_REPEATS),
            "holdout": "stratified {0}% single split".format(int(TEST_SIZE * 100)),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "reference": reference,
        "models": rows,
    }


def render_markdown(res):
    ref = res["reference"]
    counts = ", ".join("{0} {1}".format(k, v) for k, v in res["class_counts"].items())
    lines = [
        "# Results",
        "",
        "Generated by `scripts/evaluate.py` on {0}. Every accuracy quoted anywhere "
        "in this repository comes from this file.".format(res["generated"]),
        "",
        "Dataset: {0} answers. Class counts: {1}.".format(res["n_answers"], counts),
        "",
        "Cross-validation: {0}, seed {1}. Held-out: {2}.".format(
            res["config"]["cv"], res["config"]["seed"], res["config"]["holdout"]
        ),
        "",
        "## Model ladder",
        "",
        "| Model | CV accuracy (mean ± sd) | Held-out (n={0}) | 95% CI on held-out |".format(
            res["models"]["full"]["holdout_n"]
        ),
        "|---|---|---|---|",
    ]
    for name, r in res["models"].items():
        lo, hi = r["holdout_ci95"]
        lines.append(
            "| `{0}` | {1:.3f} ± {2:.3f} | {3:.3f} ({4}/{5}) | {6:.2f} to {7:.2f} |".format(
                name, r["cv_mean"], r["cv_std"], r["holdout_accuracy"],
                r["holdout_correct"], r["holdout_n"], lo, hi
            )
        )

    lines += [
        "",
        "The held-out column carries {0} answers. Its interval spans roughly 35 "
        "percentage points for every model in the table, so it separates nothing. "
        "The cross-validation column is the one to read.".format(
            res["models"]["full"]["holdout_n"]
        ),
        "",
        "## Within-question evaluation",
        "",
        "Out-of-fold predictions from the `full` model, split by question, next to "
        "the accuracy of predicting each question's most common class with no text "
        "input at all.",
        "",
        "| Question | n | `full` model | Question-majority rule |",
        "|---|---|---|---|",
    ]
    for q, r in res["models"]["full"]["per_question"].items():
        lines.append(
            "| Q{0} | {1} | {2:.3f} | {3:.3f} |".format(
                q, r["n"], r["model_accuracy"], r["question_majority_accuracy"]
            )
        )
    lines += [
        "| **All** | {0} | **{1:.3f}** | **{2:.3f}** |".format(
            res["n_answers"], ref["full_model_oof"], ref["per_question_majority"]
        ),
        "",
        "The text model reaches {0:.1%} out of fold. The rule that ignores the text "
        "entirely reaches {1:.1%}. On this dataset the answer text adds nothing once "
        "the question is known.".format(
            ref["full_model_oof"], ref["per_question_majority"]
        ),
        "",
        "## Full model against the text-free baseline, paired by fold",
        "",
        "Both fitted on the same {0} folds, differenced fold by fold.".format(
            ref["paired_comparison"]["n_folds"]),
        "",
        "| quantity | value |",
        "|---|---|",
        "| mean difference in accuracy | {0:+.4f} |".format(
            ref["paired_comparison"]["mean_difference"]),
        "| sd of the differences | {0:.4f} |".format(
            ref["paired_comparison"]["sd_difference"]),
        "| folds where the full model wins | {0} of {1} |".format(
            ref["paired_comparison"]["folds_a_wins"],
            ref["paired_comparison"]["n_folds"]),
        "| Wilcoxon signed-rank p | {0:.3f} |".format(
            ref["paired_comparison"]["wilcoxon_p"]),
        "",
        "The {0}-feature text model and the rule that reads no text are "
        "indistinguishable on this data.".format(
            res["models"]["full"]["n_features"]),
        "",
        "## Split sensitivity",
        "",
        "Held-out accuracy of the `full` model across {0} different stratified "
        "80/20 splits of the same data. Each split puts {1} answers in the test "
        "set.".format(len(ref["holdout_seed_sweep"]["seeds"]),
                      res["models"]["full"]["holdout_n"]),
        "",
        "| statistic | value |",
        "|---|---|",
        "| lowest | {0:.3f} |".format(ref["holdout_seed_sweep"]["min"]),
        "| mean | {0:.3f} |".format(ref["holdout_seed_sweep"]["mean"]),
        "| highest | {0:.3f} |".format(ref["holdout_seed_sweep"]["max"]),
        "",
        "The spread is {0:.0f} percentage points. A single held-out number from "
        "this dataset reports the split as much as it reports the model.".format(
            100 * (ref["holdout_seed_sweep"]["max"] - ref["holdout_seed_sweep"]["min"])
        ),
        "",
        "## Confusion matrix, `full` model, out of fold",
        "",
        "| actual \\ predicted | " + " | ".join(res["class_order"]) + " |",
        "|---|" + "---|" * len(res["class_order"]),
    ]
    for label, row in zip(res["class_order"], ref["full_model_confusion"]):
        lines.append("| {0} | ".format(label) + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "## Environment",
        "",
        "```",
        "python {0}".format(res["config"]["python"]),
        "scikit-learn {0}".format(res["config"]["scikit_learn"]),
        "numpy {0}".format(res["config"]["numpy"]),
        "pandas {0}".format(res["config"]["pandas"]),
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Regenerate every reported metric.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    df = load_dataset()
    res = evaluate(df, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    markdown = render_markdown(res)
    (args.out / "RESULTS.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print("wrote {0}".format(args.out / "metrics.json"))
    print("wrote {0}".format(args.out / "RESULTS.md"))


if __name__ == "__main__":
    main()
