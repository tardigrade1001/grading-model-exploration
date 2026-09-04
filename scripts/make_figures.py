#!/usr/bin/env python3
"""Regenerate the figures in figures/ from results/metrics.json.

    python scripts/evaluate.py      # must run first, writes results/metrics.json
    python scripts/make_figures.py

The figures are drawn from the metrics file so a panel can never disagree with
a table. Panels that show raw data read the training split, matching
scripts/explore.py.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grading.data import load_dataset  # noqa: E402
from grading.features import concept_table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = REPO_ROOT / "results" / "metrics.json"
FIGURES = REPO_ROOT / "figures"
STEM = "results_summary"

INK = "#1f2933"
MUTED = "#8c9aa8"
ACCENT = "#2d6a9f"
WARN = "#b5504a"


def load_metrics():
    if not METRICS.exists():
        raise SystemExit("run scripts/evaluate.py first, {0} is missing".format(METRICS))
    return json.loads(METRICS.read_text(encoding="utf-8"))


def panel_ladder(ax, res):
    names = list(res["models"])
    means = [res["models"][n]["cv_mean"] for n in names]
    sds = [res["models"][n]["cv_std"] for n in names]
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    sds = [sds[i] for i in order]

    colours = [ACCENT if n == "full" else MUTED for n in names]
    ax.barh(names, means, xerr=sds, color=colours, height=0.6,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    floor = res["reference"]["per_question_majority"]
    ax.axvline(floor, color=WARN, linestyle="--", linewidth=1.4)
    ax.text(floor + 0.006, -0.45, "question-majority rule\n{0:.1%}".format(floor),
            color=WARN, fontsize=8, va="bottom")
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("cross-validated accuracy")
    ax.set_title("A. No model clears the text-free floor", loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)


def panel_length(ax, df):
    for q, g in df.groupby("question_number"):
        ax.scatter(g["transcribed_text"].str.len(), g["normalized_mark"],
                   s=18, alpha=0.75, label="Q{0}".format(q))
    ax.set_xlabel("answer length (characters)")
    ax.set_ylabel("mark")
    ax.legend(fontsize=8, frameon=False, title="question", title_fontsize=8)
    ax.set_title("B. Length tracks marks, and it also tracks question",
                 loc="left", fontweight="bold")


def panel_concepts(ax, df):
    ax.grid(False)
    concepts = concept_table(df["transcribed_text"])
    by_q = concepts.groupby(df["question_number"]).mean()
    im = ax.imshow(by_q.to_numpy(), cmap="BuPu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(by_q.shape[1]))
    ax.set_xticklabels([c.replace("mentions_", "") for c in by_q.columns],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(by_q.shape[0]))
    ax.set_yticklabels(["Q{0}".format(q) for q in by_q.index], fontsize=8)
    for i in range(by_q.shape[0]):
        for j in range(by_q.shape[1]):
            v = by_q.iat[i, j]
            ax.text(j, i, "{0:.0f}".format(100 * v), ha="center", va="center",
                    fontsize=7, color="white" if v > 0.55 else INK)
    ax.set_title("C. Concept flags mark the question, in % of answers",
                 loc="left", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def panel_sweep(ax, res):
    sweep = res["reference"]["holdout_seed_sweep"]
    values = np.array(sweep["accuracies"])
    ax.hist(values, bins=np.arange(0.36, 0.84, 0.04), color=MUTED, edgecolor="white")
    ax.axvline(sweep["mean"], color=ACCENT, linewidth=1.6)
    ax.text(sweep["mean"] + 0.008, ax.get_ylim()[1] * 0.9,
            "mean {0:.0%}".format(sweep["mean"]), color=ACCENT, fontsize=8)
    ax.set_xlabel("held-out accuracy of the full model, {0} splits".format(len(values)))
    ax.set_ylabel("splits")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("D. A single 25-answer test set decides little",
                 loc="left", fontweight="bold")


def main():
    res = load_metrics()
    df = load_dataset()

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6eaee",
        "grid.linewidth": 0.7,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_ladder(axes[0, 0], res)
    panel_length(axes[0, 1], df)
    panel_concepts(axes[1, 0], df)
    panel_sweep(axes[1, 1], res)

    fig.suptitle(
        "Grading model exploration: what {0} answers can and cannot support".format(
            res["n_answers"]),
        fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        out = FIGURES / "{0}.{1}".format(STEM, ext)
        fig.savefig(out, bbox_inches="tight")
        print("wrote {0}".format(out))
    plt.close(fig)


if __name__ == "__main__":
    main()
