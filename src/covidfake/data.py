"""Loading the COVID Fake News Dataset (columns: id, tweet, label = real / fake)."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocess import clean_text

LABELS: List[str] = ["real", "fake"]  # fake = 1 is the positive class


def normalise_label(label) -> int:
    key = str(label).strip().lower()
    if key in ("fake", "1", "false"):
        return 1
    if key in ("real", "0", "true"):
        return 0
    raise ValueError(f"Unknown label {label!r}; expected 'real' or 'fake'")


def load_data(path: str, text_col: str = "tweet", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_col not in df:
        for alt in ("text", "content", "news"):
            if alt in df:
                text_col = alt
                break
        else:
            raise KeyError(f"No text column found in {path}; columns: {list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col])
    return pd.DataFrame({
        "text": df[text_col].astype(str).values,
        "clean": [clean_text(t) for t in df[text_col]],
        "y": [normalise_label(l) for l in df[label_col]],
    })


def split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 70/30 train/test split, as in the paper."""
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["y"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)
