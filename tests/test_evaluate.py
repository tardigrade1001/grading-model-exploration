"""Tests for the evaluation helpers and the reported-results contract.

results/metrics.json is the single source of every accuracy in the repository.
These tests cover the statistics it computes and the invariants the write-up
depends on.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate import (  # noqa: E402
    per_question_majority_accuracy,
    wilson_interval,
)
from grading.data import load_dataset  # noqa: E402

METRICS = REPO_ROOT / "results" / "metrics.json"


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(16, 25)
    assert low < 16 / 25 < high


def test_wilson_interval_stays_inside_zero_and_one():
    for k, n in [(0, 25), (25, 25), (1, 3), (13, 122)]:
        low, high = wilson_interval(k, n)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_narrows_as_n_grows():
    narrow = wilson_interval(640, 1000)
    wide = wilson_interval(16, 25)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_is_wide_at_the_reported_holdout_size():
    """The write-up claims roughly 35 points of width at n=25."""
    low, high = wilson_interval(16, 25)
    assert 0.30 < (high - low) < 0.40


def test_wilson_interval_handles_an_empty_sample():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_per_question_majority_is_perfect_on_a_separable_frame():
    df = pd.DataFrame({"question_number": [1, 1, 1, 2, 2, 2]})
    y = np.array(["Low", "Low", "Low", "High", "High", "High"])
    assert per_question_majority_accuracy(df, y) == 1.0


def test_per_question_majority_scores_the_majority_share():
    df = pd.DataFrame({"question_number": [1, 1, 1, 1]})
    y = np.array(["Low", "Low", "Low", "High"])
    assert per_question_majority_accuracy(df, y) == pytest.approx(0.75)


def test_per_question_majority_beats_the_global_majority_on_this_data():
    """The confound the whole write-up rests on."""
    df = load_dataset()
    y = df["mark_class"].to_numpy()
    global_majority = df["mark_class"].value_counts(normalize=True).iloc[0]
    assert per_question_majority_accuracy(df, y) > global_majority


@pytest.mark.skipif(not METRICS.exists(), reason="run scripts/evaluate.py first")
class TestReportedMetrics:
    @pytest.fixture(scope="class")
    @classmethod
    def res(cls):
        return json.loads(METRICS.read_text(encoding="utf-8"))

    def test_every_model_records_the_fields_the_docs_cite(self, res):
        for name, row in res["models"].items():
            for field in ("cv_mean", "cv_std", "holdout_accuracy",
                          "holdout_ci95", "oof_accuracy", "per_question"):
                assert field in row, "{0} missing {1}".format(name, field)

    def test_accuracies_are_probabilities(self, res):
        for row in res["models"].values():
            assert 0.0 <= row["cv_mean"] <= 1.0
            assert 0.0 <= row["holdout_accuracy"] <= 1.0
            assert 0.0 <= row["oof_accuracy"] <= 1.0

    def test_cross_validation_ran_the_documented_number_of_fits(self, res):
        for row in res["models"].values():
            assert row["cv_n_fits"] == 50

    def test_majority_class_matches_the_largest_band(self, res):
        largest = max(res["class_counts"].values())
        assert res["models"]["majority_class"]["cv_mean"] == pytest.approx(
            largest / res["n_answers"], abs=0.02
        )

    def test_no_model_clears_the_text_free_floor(self, res):
        """The headline claim. If a model ever beats it, the README is wrong."""
        floor = res["reference"]["per_question_majority"]
        assert res["reference"]["full_model_oof"] <= floor

    def test_the_paired_comparison_is_not_significant(self, res):
        paired = res["reference"]["paired_comparison"]
        assert paired["n_folds"] == 50
        assert paired["wilcoxon_p"] > 0.05

    def test_confusion_matrix_totals_the_dataset(self, res):
        total = sum(sum(row) for row in res["reference"]["full_model_confusion"])
        assert total == res["n_answers"]

    def test_split_sweep_spread_is_documented_as_wide(self, res):
        sweep = res["reference"]["holdout_seed_sweep"]
        assert len(sweep["accuracies"]) == len(sweep["seeds"])
        assert sweep["max"] - sweep["min"] > 0.20
