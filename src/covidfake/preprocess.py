"""Text preprocessing from the paper: remove URLs, punctuation and extra spaces,
tokenise, drop stop words and stem."""
from __future__ import annotations

import re
import string

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}“”‘’…]")
WS_RE = re.compile(r"\s+")
_stemmer = PorterStemmer()


def clean_text(text: str, remove_stopwords: bool = True, stem: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text.lower())
    text = MENTION_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    tokens = WS_RE.sub(" ", text).strip().split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
