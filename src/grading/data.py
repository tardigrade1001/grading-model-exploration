"""Dataset loading and the label definition used everywhere in this repo."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "exam_results_cleaned_final.csv"

CLASS_ORDER = ["Low", "Mid", "High"]

# Marks run 1.0 to 4.0 in half steps. Three bands keep every class large
# enough to stratify a 5-fold split.
CLASS_EDGES = ((1.5, "Low"), (2.5, "Mid"), (4.0, "High"))


def mark_to_class(mark: float) -> str:
    """Map a numeric mark onto one of CLASS_ORDER."""
    for upper, name in CLASS_EDGES:
        if mark <= upper:
            return name
    return CLASS_ORDER[-1]


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the 122 transcribed answers with their marks and class labels.

    Returns a frame with columns question_number, transcribed_text,
    normalized_mark, mark_class.
    """
    df = pd.read_csv(Path(path) if path is not None else DATA_PATH)
    expected = {"question_number", "transcribed_text", "normalized_mark"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    df["mark_class"] = df["normalized_mark"].map(mark_to_class)
    return df
