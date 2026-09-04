#!/usr/bin/env python3
"""Check that the numbers in the markdown match the generated result files.

    python scripts/evaluate.py
    python scripts/explore.py
    python scripts/check_docs.py

Exits non-zero and lists what drifted. The tables in README.md and FINDINGS.md
are copied by hand from generated output, so they can go stale the moment the
data, the seed, or a model definition changes. This script is the guard.

Sources: results/metrics.json for anything with an accuracy in it,
results/exploration.json for the descriptive tables in FINDINGS.md.

It checks the numbers that carry the argument. It does not check prose, except
where a number was once typed into prose and typed wrong.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "results" / "metrics.json"
EXPLORATION = REPO_ROOT / "results" / "exploration.json"


def pct(x):
    return "{0:.1f}%".format(100 * x)


def build_exploration_checks(exp):
    """Every descriptive table in FINDINGS.md, row by row."""
    checks = []
    names = exp["concept_names"]
    order = exp["class_order"]
    questions = sorted(exp["by_question"])

    def add(label, text):
        checks.append(("FINDINGS.md", label, text))

    for q in questions:
        r = exp["by_question"][q]
        add("confound Q{0}".format(q), "| Q{0} | {1} | {2:.2f} | {3:.0f} |".format(
            q, r["n"], r["mean_mark"], r["mean_chars"]))
        add("class counts Q{0}".format(q), "| Q{0} | ".format(q) + " | ".join(
            str(r["class_counts"][c]) for c in order) + " |")
        add("length corr Q{0}".format(q), "| Q{0} | {1} | {2:.2f} | {3:.2f} |".format(
            q, r["n"], r["pearson_r"], r["spearman_r"]))
        add("concepts by question Q{0}".format(q), "| Q{0} | ".format(q) + " | ".join(
            "{0:.2f}".format(r["concepts"][n]) for n in names) + " |")

    for name in names:
        add("concepts by band {0}".format(name), "| {0} | ".format(name) + " | ".join(
            "{0:.2f}".format(exp["concepts_by_class"][c][name]) for c in order) + " |")

    for q, r in exp["concepts_within_question"].items():
        counts = ", ".join("{0} {1}".format(c, r["class_counts"][c]) for c in order)
        add("within Q{0} header".format(q),
            "**Q{0}**, n = {1} ({2})".format(q, r["n"], counts))
        for cls in order:
            if cls in r["by_class"]:
                add("within Q{0} {1}".format(q, cls), "| {0} | ".format(cls) + " | ".join(
                    "{0:.2f}".format(r["by_class"][cls][n]) for n in names) + " |")

    add("pooled length correlation",
        "r = {0:.2f}".format(exp["pooled_length_mark_pearson_r"]))
    return checks


def build_checks(res):
    """Return a list of (document, label, literal string that must appear)."""
    models = res["models"]
    ref = res["reference"]
    checks = []

    def add(doc, label, text):
        checks.append((doc, label, text))

    # README headline claims.
    add("README.md", "full model CV", pct(models["full"]["cv_mean"]))
    add("README.md", "per-question majority", pct(ref["per_question_majority"]))
    add("README.md", "split sweep low", "{0:.0f}%".format(100 * ref["holdout_seed_sweep"]["min"]))
    add("README.md", "split sweep high", "{0:.0f}%".format(100 * ref["holdout_seed_sweep"]["max"]))

    # Feature counts quoted in prose. This is the claim that was typed wrong.
    add("README.md", "full model width",
        "model with {0} features".format(models["full"]["n_features"]))
    add("FINDINGS.md", "full model width",
        "{0}-feature `full` model".format(models["full"]["n_features"]))

    # The ladder table, in both README.md and results/RESULTS.md.
    for name, row in models.items():
        cell = "{0:.3f} ± {1:.3f}".format(row["cv_mean"], row["cv_std"])
        add("README.md", "ladder row {0}".format(name), cell)

    # Per-question breakdown, in README.md and FINDINGS.md.
    for doc in ("README.md", "FINDINGS.md"):
        for q, row in models["full"]["per_question"].items():
            cell = "| Q{0} | {1} | {2:.3f} | {3:.3f} |".format(
                q, row["n"], row["model_accuracy"], row["question_majority_accuracy"]
            )
            add(doc, "per-question Q{0}".format(q), cell)
        add(doc, "per-question total", "**{0:.3f}** | **{1:.3f}**".format(
            ref["full_model_oof"], ref["per_question_majority"]))

    # Confusion matrix rows in FINDINGS.md.
    for label, row in zip(res["class_order"], ref["full_model_confusion"]):
        add("FINDINGS.md", "confusion {0}".format(label),
            "| {0} | ".format(label) + " | ".join(str(v) for v in row) + " |")

    return checks


# Fields that legitimately differ between two runs of the same code: the date
# it ran and the library versions it ran under. Everything else is a number the
# repository reports, and a change in one is a change in the findings.
VOLATILE = ("generated", "config")


def numbers_only(res):
    return {k: v for k, v in res.items() if k not in VOLATILE}


def compare_to_fresh_run():
    """Recompute the metrics and diff the numbers against the committed file.

    Used in CI. The committed results/metrics.json is the version every table
    in the markdown was copied from, so if a fresh run disagrees with it, the
    repository is reporting numbers its own code no longer produces.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import evaluate  # noqa: E402
    from explore import explore  # noqa: E402

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from grading.data import load_dataset  # noqa: E402
    from grading.models import RANDOM_STATE  # noqa: E402

    df = load_dataset()
    failures = []
    for path, fresh in (
        (METRICS, evaluate(df, RANDOM_STATE)),
        (EXPLORATION, explore(df, RANDOM_STATE)),
    ):
        committed = json.loads(path.read_text(encoding="utf-8"))
        if numbers_only(committed) != numbers_only(fresh):
            failures.append(path.name)

    if failures:
        print("a fresh run disagrees with the committed results: {0}".format(
            ", ".join(failures)))
        print("regenerate with scripts/evaluate.py and scripts/explore.py, "
              "update the markdown, and commit.")
        return 1
    print("committed results match a fresh run")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-results",
        action="store_true",
        help="also recompute the metrics and diff them against the committed files",
    )
    args = parser.parse_args()

    missing = [p for p in (METRICS, EXPLORATION) if not p.exists()]
    if missing:
        print("missing {0}".format(", ".join(str(p) for p in missing)))
        print("run scripts/evaluate.py and scripts/explore.py first")
        return 1
    res = json.loads(METRICS.read_text(encoding="utf-8"))
    exp = json.loads(EXPLORATION.read_text(encoding="utf-8"))

    checks = build_checks(res) + build_exploration_checks(exp)

    cache = {}
    failures = []
    for doc, label, text in checks:
        if doc not in cache:
            cache[doc] = (REPO_ROOT / doc).read_text(encoding="utf-8")
        if text not in cache[doc]:
            failures.append((doc, label, text))

    if failures:
        print("{0} of {1} documented value(s) do not match the generated "
              "results:\n".format(len(failures), len(checks)))
        for doc, label, text in failures:
            print("  {0:<14} {1:<30} expected to find: {2}".format(doc, label, text))
        print("\nUpdate the markdown to the generated values.")
        return 1

    print("all {0} documented values match the generated results".format(len(checks)))
    if args.check_results:
        return compare_to_fresh_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
