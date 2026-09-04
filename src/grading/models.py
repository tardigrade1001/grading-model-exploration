"""The model zoo evaluated in scripts/evaluate.py.

Each entry is a complete sklearn Pipeline that consumes the raw dataframe.
Fitting a pipeline fits its vectoriser, its scaler, and its length thresholds
on the training rows alone, so the same object can be handed to
cross_val_score and to a held-out split without any manual bookkeeping.

The ladder is deliberate. A model is only interesting if it beats the
baselines below it.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ConceptFeatures, LengthFeatures

RANDOM_STATE = 0

TEXT = "transcribed_text"
QUESTION = ["question_number"]


class PerQuestionMajority(BaseEstimator, ClassifierMixin):
    """Predict each question's most common class in the training rows.

    Reads no answer text. This is the floor the whole project is measured
    against, so it is a fitted estimator like every other entry in the ladder
    and it goes through the same folds. Fitting it on the training rows only
    is what makes the comparison with the text models fair.
    """

    def fit(self, X, y):
        y = pd.Series(np.asarray(y))
        questions = np.asarray(X["question_number"])
        self.map_ = y.groupby(questions).agg(lambda s: s.mode().iloc[0]).to_dict()
        self.fallback_ = y.mode().iloc[0]
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array([self.map_.get(q, self.fallback_)
                         for q in X["question_number"]])


def _clf():
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=RANDOM_STATE)


def _tfidf():
    return TfidfVectorizer(max_features=30, min_df=2, max_df=0.8,
                           ngram_range=(1, 2))


def _length_block():
    return Pipeline([("length", LengthFeatures()), ("scale", StandardScaler())])


def build_models() -> dict[str, Pipeline]:
    """Return every model reported in results/, keyed by display name."""
    question_ohe = ("question", OneHotEncoder(handle_unknown="ignore"), QUESTION)
    length_block = ("length", _length_block(), [TEXT])
    concept_block = ("concept", ConceptFeatures(), [TEXT])
    tfidf_block = ("tfidf", _tfidf(), TEXT)

    return {
        # Floor 1: predict the single most common class every time.
        "majority_class": make_pipeline(
            ColumnTransformer([question_ohe]),
            DummyClassifier(strategy="most_frequent"),
        ),
        # Floor 2: predict each question's most common training class.
        # This is the baseline every text model has to clear.
        "per_question_majority": PerQuestionMajority(),
        # Floor 3: the question number as a fitted feature.
        "question_only": make_pipeline(
            ColumnTransformer([question_ohe]),
            _clf(),
        ),
        "length_only": make_pipeline(
            ColumnTransformer([length_block]),
            _clf(),
        ),
        "concepts_only": make_pipeline(
            ColumnTransformer([concept_block]),
            _clf(),
        ),
        "tfidf_only": make_pipeline(
            ColumnTransformer([tfidf_block]),
            _clf(),
        ),
        "question_plus_length": make_pipeline(
            ColumnTransformer([question_ohe, length_block]),
            _clf(),
        ),
        # Text-side model with question identity withheld, to show how much of
        # the full model's accuracy is question identity.
        "text_features_no_question": make_pipeline(
            ColumnTransformer([length_block, concept_block, tfidf_block]),
            _clf(),
        ),
        "full": make_pipeline(
            ColumnTransformer([question_ohe, length_block, concept_block, tfidf_block]),
            _clf(),
        ),
    }
