"""Feature extractors.

Every transformer here is stateless with respect to the labels, and any
threshold it needs is learned in ``fit`` from the training rows it is given.
That is what keeps the cross-validation folds and the held-out split honest.
The earlier version of this project hard-coded length cut-offs at 600 and 400
characters after reading the whole dataset, which leaked test-set information
into the feature definition.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Concept vocabulary. These were chosen by reading the exam questions, so the
# list is fixed before any model is fitted. FINDINGS.md explains why their
# apparent predictive power is mostly question identity in disguise.
CONCEPT_PATTERNS = {
    "mentions_antibody": r"\banti\w*bod\w*",
    "mentions_specificity": r"\bspecificit\w*",
    "mentions_sensitivity": r"\bsensitivit\w*",
    "mentions_nanomaterial": r"\bnano\w*",
    "mentions_tmb": r"\btmb\b",
    "mentions_binding": r"\bbind\w*",
    "mentions_target": r"\btarget\w*",
    "mentions_lod": r"\blod\b|\blimit.*detection\b",
    "mentions_calibration": r"\bcalibr\w*",
}


def _text_column(X) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0].astype(str)
    return pd.Series(np.asarray(X).ravel(), dtype=str)


class LengthFeatures(BaseEstimator, TransformerMixin):
    """Character count, word count, and two quantile-based length flags.

    The ``is_long`` and ``is_short`` cut-offs are the training-fold quantiles,
    learned in ``fit``. Nothing about the validation or test rows reaches them.
    """

    def __init__(self, long_quantile: float = 0.75, short_quantile: float = 0.25):
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile

    def fit(self, X, y=None):
        lengths = _text_column(X).str.len()
        self.long_threshold_ = float(lengths.quantile(self.long_quantile))
        self.short_threshold_ = float(lengths.quantile(self.short_quantile))
        return self

    def transform(self, X):
        text = _text_column(X)
        lengths = text.str.len()
        words = text.str.split().str.len()
        return np.column_stack([
            lengths.to_numpy(dtype=float),
            words.to_numpy(dtype=float),
            (lengths >= self.long_threshold_).to_numpy(dtype=float),
            (lengths <= self.short_threshold_).to_numpy(dtype=float),
        ])

    def get_feature_names_out(self, input_features=None):
        return np.array(["answer_chars", "answer_words", "is_long", "is_short"])


class ConceptFeatures(BaseEstimator, TransformerMixin):
    """Binary presence flags for a fixed regex vocabulary."""

    def __init__(self, patterns: dict[str, str] | None = None):
        self.patterns = patterns

    def _patterns(self) -> dict[str, str]:
        return CONCEPT_PATTERNS if self.patterns is None else self.patterns

    def fit(self, X, y=None):
        self.feature_names_ = list(self._patterns())
        return self

    def transform(self, X):
        text = _text_column(X)
        cols = [
            text.str.contains(pattern, case=False, na=False, regex=True).to_numpy(dtype=float)
            for pattern in self._patterns().values()
        ]
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self._patterns()))


def concept_table(text: pd.Series) -> pd.DataFrame:
    """Concept flags as a labelled frame. For exploratory reporting only."""
    return pd.DataFrame(
        {name: text.str.contains(p, case=False, na=False, regex=True).astype(int)
         for name, p in CONCEPT_PATTERNS.items()},
        index=text.index,
    )


assert all(re.compile(p) for p in CONCEPT_PATTERNS.values())
