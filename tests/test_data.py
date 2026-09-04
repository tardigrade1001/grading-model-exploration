"""Tests for the dataset contract and the label definition.

The band boundaries are the one definition every script shares. Moving one
silently would change every number in results/ without any script failing.
"""

import pandas as pd
import pytest

from grading.data import CLASS_ORDER, load_dataset, mark_to_class


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_shape_and_columns(dataset):
    assert len(dataset) == 122
    assert list(dataset.columns) == [
        "question_number", "transcribed_text", "normalized_mark", "mark_class",
    ]


def test_no_missing_values(dataset):
    assert not dataset.isna().any().any()


def test_no_duplicate_answers(dataset):
    """DATA_CARD.md states the cleaning removed exact duplicates."""
    assert not dataset["transcribed_text"].duplicated().any()


def test_marks_are_half_steps_in_range(dataset):
    marks = dataset["normalized_mark"]
    assert marks.between(1.0, 4.0).all()
    assert ((marks * 2) % 1 == 0).all()


def test_questions_are_one_to_four(dataset):
    assert sorted(dataset["question_number"].unique()) == [1, 2, 3, 4]


@pytest.mark.parametrize("mark,expected", [
    (1.0, "Low"),
    (1.5, "Low"),
    (2.0, "Mid"),
    (2.5, "Mid"),
    (3.0, "High"),
    (3.5, "High"),
    (4.0, "High"),
])
def test_band_boundaries(mark, expected):
    """The boundaries sit at 1.5 and 2.5 inclusive. METHODOLOGY.md says so."""
    assert mark_to_class(mark) == expected


def test_class_counts_match_the_documented_split(dataset):
    counts = dataset["mark_class"].value_counts().to_dict()
    assert counts == {"Mid": 46, "Low": 41, "High": 35}
    assert sum(counts.values()) == len(dataset)


def test_every_class_is_large_enough_to_stratify_five_folds(dataset):
    """The 3-class banding exists so a 5-fold stratified split is possible."""
    assert dataset["mark_class"].value_counts().min() >= 5


def test_class_order_covers_every_label(dataset):
    assert set(dataset["mark_class"]) == set(CLASS_ORDER)


def test_load_rejects_a_file_missing_a_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"question_number": [1], "normalized_mark": [2.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="transcribed_text"):
        load_dataset(path)
