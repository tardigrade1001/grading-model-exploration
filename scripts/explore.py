#!/usr/bin/env python3
"""Exploratory analysis of the training split.

    python scripts/explore.py

Writes results/exploration.json and results/EXPLORATION.md, and prints the
same tables. The tables in FINDINGS.md are copied from that output and
scripts/check_docs.py verifies them against the JSON.

This script exists to keep feature discovery separate from evaluation. It
reads only the 97 answers in the training half of the split that
scripts/evaluate.py holds out, so any pattern it surfaces can be turned into
a feature without contaminating the reported metrics. It prints no accuracy.
Accuracies come from scripts/evaluate.py alone.

The first version of this project discovered its length thresholds and its
concept vocabulary by reading all 122 answers, then reported a held-out score
on 25 of those same answers. The thresholds had already seen them.

Every relationship recorded here is also broken down by question, because
question identity confounds all four of the raw patterns. FINDINGS.md carries
the interpretation.
"""

import os

# Single-threaded BLAS before numpy loads. Threaded reductions sum in whatever
# order the threads finish, so the lbfgs solver in LogisticRegression lands on a
# slightly different optimum each run. Pinning one thread makes every reported
# number reproduce exactly on one machine.
#
# Across operating systems and library versions the last few decimals still
# move. scripts/check_docs.py compares within a tolerance for that, and every
# value is quoted to three decimals, well above the drift.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grading.data import CLASS_ORDER, load_dataset  # noqa: E402
from grading.features import concept_table  # noqa: E402
from grading.models import RANDOM_STATE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TEST_SIZE = 0.2

# Questions carrying both a spread of marks and a mix of concepts. These are
# the only ones where a concept signal could show itself independently of
# question identity.
MIXED_QUESTIONS = (2, 3)


def short(name):
    """`mentions_antibody` -> `antibody`, for table headers."""
    return name.replace("mentions_", "")


def training_split(df, seed=RANDOM_STATE):
    """The same training rows scripts/evaluate.py fits on."""
    y = df["mark_class"].to_numpy()
    train_idx, _ = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    return df.iloc[train_idx].copy()


def explore(full, seed):
    df = training_split(full, seed)
    df["answer_chars"] = df["transcribed_text"].str.len()
    concepts = concept_table(df["transcribed_text"])

    by_question = {}
    for q, g in df.groupby("question_number"):
        counts = g["mark_class"].value_counts().reindex(CLASS_ORDER).fillna(0)
        by_question[str(q)] = {
            "n": int(len(g)),
            "mean_mark": float(g["normalized_mark"].mean()),
            "std_mark": float(g["normalized_mark"].std()),
            "mean_chars": float(g["answer_chars"].mean()),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "pearson_r": float(g["answer_chars"].corr(g["normalized_mark"])),
            "spearman_r": float(
                g["answer_chars"].corr(g["normalized_mark"], method="spearman")
            ),
            "concepts": {short(c): float(v)
                         for c, v in concepts[df["question_number"].eq(q).to_numpy()]
                         .mean().items()},
        }

    length_by_mark = {}
    for mark, g in df.groupby("normalized_mark"):
        length_by_mark["{0:.1f}".format(mark)] = {
            "n": int(len(g)),
            "mean_chars": float(g["answer_chars"].mean()),
        }

    concepts_by_class = {
        cls: {short(c): float(v) for c, v in
              concepts[df["mark_class"].eq(cls).to_numpy()].mean().items()}
        for cls in CLASS_ORDER
    }

    within = {}
    for q in MIXED_QUESTIONS:
        mask = df["question_number"].eq(q).to_numpy()
        sub_concepts, sub_class = concepts[mask], df.loc[mask, "mark_class"]
        within[str(q)] = {
            "n": int(mask.sum()),
            "class_counts": {
                cls: int((sub_class == cls).sum()) for cls in CLASS_ORDER
            },
            "by_class": {
                cls: {short(c): float(v) for c, v in
                      sub_concepts[(sub_class == cls).to_numpy()].mean().items()}
                for cls in CLASS_ORDER if (sub_class == cls).any()
            },
        }

    return {
        "generated": date.today().isoformat(),
        "seed": seed,
        "n_total": int(len(full)),
        "n_train": int(len(df)),
        "n_holdout": int(len(full) - len(df)),
        "class_order": CLASS_ORDER,
        "concept_names": [short(c) for c in concepts.columns],
        "pooled_length_mark_pearson_r": float(
            df["answer_chars"].corr(df["normalized_mark"])
        ),
        "by_question": by_question,
        "length_by_mark": length_by_mark,
        "concepts_by_class": concepts_by_class,
        "concepts_within_question": within,
    }


def render_markdown(res):
    qs = sorted(res["by_question"])
    lines = [
        "# Exploration",
        "",
        "Generated by `scripts/explore.py` on {0}. Training split only: {1} of "
        "{2} answers, the other {3} held out. No accuracy appears in this file. "
        "Those live in `results/RESULTS.md`.".format(
            res["generated"], res["n_train"], res["n_total"], res["n_holdout"]),
        "",
        "## 1. The confound",
        "",
        "| Question | n | mean mark | mean length (chars) |",
        "|---|---|---|---|",
    ]
    for q in qs:
        r = res["by_question"][q]
        lines.append("| Q{0} | {1} | {2:.2f} | {3:.0f} |".format(
            q, r["n"], r["mean_mark"], r["mean_chars"]))

    lines += [
        "",
        "Class composition within each question:",
        "",
        "| Question | " + " | ".join(res["class_order"]) + " |",
        "|---|" + "---|" * len(res["class_order"]),
    ]
    for q in qs:
        counts = res["by_question"][q]["class_counts"]
        lines.append("| Q{0} | ".format(q) + " | ".join(
            str(counts[c]) for c in res["class_order"]) + " |")

    lines += [
        "",
        "Any feature that identifies a question inherits that question's mark "
        "distribution.",
        "",
        "## 2. Answer length",
        "",
        "Pooled across questions, by mark:",
        "",
        "| Mark | n | mean length (chars) |",
        "|---|---|---|",
    ]
    for mark, r in res["length_by_mark"].items():
        lines.append("| {0} | {1} | {2:.0f} |".format(mark, r["n"], r["mean_chars"]))

    lines += [
        "",
        "Pooled correlation between length and mark is r = {0:.3f}. Computed "
        "inside each question, where question identity is held constant:".format(
            res["pooled_length_mark_pearson_r"]),
        "",
        "| Question | n | Pearson r | Spearman r |",
        "|---|---|---|---|",
    ]
    for q in qs:
        r = res["by_question"][q]
        lines.append("| Q{0} | {1} | {2:.2f} | {3:.2f} |".format(
            q, r["n"], r["pearson_r"], r["spearman_r"]))

    names = res["concept_names"]
    lines += [
        "",
        "The relationship survives in every question. The gap against the pooled "
        "figure is the size of the confound.",
        "",
        "## 3. Concepts",
        "",
        "Fraction of answers containing each term, by mark band:",
        "",
        "| Concept | " + " | ".join(res["class_order"]) + " |",
        "|---|" + "---|" * len(res["class_order"]),
    ]
    for name in names:
        lines.append("| {0} | ".format(name) + " | ".join(
            "{0:.2f}".format(res["concepts_by_class"][c][name])
            for c in res["class_order"]) + " |")

    lines += [
        "",
        "The same flags, by question:",
        "",
        "| Question | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for q in qs:
        c = res["by_question"][q]["concepts"]
        lines.append("| Q{0} | ".format(q) + " | ".join(
            "{0:.2f}".format(c[n]) for n in names) + " |")

    lines += [
        "",
        "Several flags sit near 1.00 for one question and near 0.00 for the rest. "
        "Those flags identify the question.",
        "",
        "## 4. Concepts within the mixed questions",
        "",
        "Questions carrying both a spread of marks and a mix of concepts. These "
        "are the only place a concept signal could show itself independently of "
        "question identity.",
    ]
    for q, r in res["concepts_within_question"].items():
        counts = ", ".join("{0} {1}".format(c, r["class_counts"][c])
                           for c in res["class_order"])
        lines += [
            "",
            "**Q{0}**, n = {1} ({2})".format(q, r["n"], counts),
            "",
            "| Band | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for cls in res["class_order"]:
            if cls in r["by_class"]:
                lines.append("| {0} | ".format(cls) + " | ".join(
                    "{0:.2f}".format(r["by_class"][cls][n]) for n in names) + " |")

    lines += [
        "",
        "The separation here is much weaker than the pooled table in section 3. "
        "FINDINGS.md carries the interpretation.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Exploratory analysis, train split.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    res = explore(load_dataset(), args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "exploration.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    markdown = render_markdown(res)
    (args.out / "EXPLORATION.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print("wrote {0}".format(args.out / "exploration.json"))
    print("wrote {0}".format(args.out / "EXPLORATION.md"))


if __name__ == "__main__":
    main()
