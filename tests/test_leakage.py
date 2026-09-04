"""Tests for the leakage guarantee.

The central claim of this repository is that no fitted statistic crosses the
train/test boundary. Violating that claim is what produced the first version's
68%. These tests exercise the claim. The prose only asserts it.
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from grading.data import load_dataset
from grading.features import LengthFeatures
from grading.models import build_models


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def split(dataset):
    y = dataset["mark_class"].to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=0, stratify=y
    )
    return dataset.iloc[train_idx], dataset.iloc[test_idx], y[train_idx], y[test_idx]


def text_of(df):
    return df[["transcribed_text"]]


def test_length_thresholds_are_learned_not_constant(dataset):
    """Different training rows must produce different cut-offs.

    The first version hard-coded 600 and 400 characters after reading all 122
    answers. If the thresholds were still constants, this test would fail.
    """
    a = LengthFeatures().fit(text_of(dataset.iloc[:60]))
    b = LengthFeatures().fit(text_of(dataset.iloc[60:]))
    assert a.long_threshold_ != b.long_threshold_
    assert a.short_threshold_ != b.short_threshold_


def test_length_thresholds_ignore_rows_not_fitted_on(dataset):
    """Changing the held-out rows must not move the fitted thresholds."""
    train = dataset.iloc[:60]
    baseline = LengthFeatures().fit(text_of(train))

    tampered = dataset.copy()
    tampered.loc[tampered.index[60:], "transcribed_text"] = "x" * 5000
    after = LengthFeatures().fit(text_of(tampered.iloc[:60]))

    assert baseline.long_threshold_ == after.long_threshold_
    assert baseline.short_threshold_ == after.short_threshold_


def test_transform_does_not_refit_thresholds(dataset):
    """transform reuses the fitted cut-offs on whatever rows it is handed."""
    fitted = LengthFeatures().fit(text_of(dataset.iloc[:60]))
    before = (fitted.long_threshold_, fitted.short_threshold_)
    fitted.transform(text_of(dataset.iloc[60:]))
    assert (fitted.long_threshold_, fitted.short_threshold_) == before


def test_length_flags_use_the_fitted_thresholds(dataset):
    """is_long and is_short must be computed against the learned cut-offs."""
    fitted = LengthFeatures().fit(text_of(dataset.iloc[:60]))
    out = fitted.transform(text_of(dataset.iloc[:60]))
    chars, _, is_long, is_short = out.T
    assert np.array_equal(is_long, (chars >= fitted.long_threshold_).astype(float))
    assert np.array_equal(is_short, (chars <= fitted.short_threshold_).astype(float))


def test_tfidf_vocabulary_excludes_holdout_only_terms(split):
    """The vectoriser inside the pipeline must never see the held-out rows.

    A sentinel token is written into the held-out answers only. If it turns up
    in the fitted vocabulary, the pipeline leaked.
    """
    df_train, df_test, y_train, _ = split
    sentinel = "zzsentineltoken"
    contaminated = df_test.copy()
    contaminated["transcribed_text"] += " " + " ".join([sentinel] * 40)

    model = build_models()["full"]
    model.fit(df_train, y_train)
    vocab = dict(model[0].named_transformers_["tfidf"].vocabulary_)
    assert not any(sentinel in term for term in vocab)

    model.predict(contaminated)
    assert dict(model[0].named_transformers_["tfidf"].vocabulary_) == vocab


def test_per_question_majority_learns_only_from_training_labels(split):
    """The baseline's question-to-class map must come from the training rows."""
    df_train, _, y_train, _ = split
    model = build_models()["per_question_majority"].fit(df_train, y_train)

    import pandas as pd

    expected = (
        pd.Series(y_train)
        .groupby(df_train["question_number"].to_numpy())
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    assert model.map_ == expected


def test_per_question_majority_handles_an_unseen_question(split):
    """An unseen question number falls back to the global training majority."""
    df_train, df_test, y_train, _ = split
    model = build_models()["per_question_majority"].fit(df_train, y_train)

    unseen = df_test.copy()
    unseen["question_number"] = 99
    assert set(model.predict(unseen)) == {model.fallback_}


@pytest.mark.parametrize("name", sorted(build_models()))
def test_every_model_fits_on_train_and_predicts_on_unseen_rows(split, name):
    df_train, df_test, y_train, _ = split
    model = build_models()[name]
    model.fit(df_train, y_train)
    preds = model.predict(df_test)
    assert len(preds) == len(df_test)
    assert set(preds) <= set(np.unique(y_train))
