"""
Feature extraction utilities for ML models.

Uses a TF-IDF vectorizer over:
- word unigrams and bigrams
- character n-grams
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    texts: Iterable[str],
) -> Tuple[TfidfVectorizer, "scipy.sparse.spmatrix"]:
    """
    Fit a TF-IDF vectorizer over provided texts.

    - word unigrams + bigrams
    - char 3- to 5-grams
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        strip_accents=None,
    )
    X_word = vectorizer.fit_transform(texts)

    # Character n-grams (3-5)
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=False,
    )
    X_char = char_vectorizer.fit_transform(texts)

    # Concatenate sparse matrices horizontally.
    from scipy.sparse import hstack

    X = hstack([X_word, X_char])

    # Store both inside one object so we can re-use easily.
    vectorizer.combined_char_vectorizer = char_vectorizer  # type: ignore[attr-defined]
    return vectorizer, X


def transform_with_tfidf(vectorizer: TfidfVectorizer, texts: Iterable[str]):
    """Transform texts using a fitted dual (word + char) TF-IDF vectorizer."""
    from scipy.sparse import hstack

    X_word = vectorizer.transform(texts)
    char_vectorizer = getattr(vectorizer, "combined_char_vectorizer")
    X_char = char_vectorizer.transform(texts)
    return hstack([X_word, X_char])


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    """Persist vectorizer (including char sub-vectorizer) to disk."""
    with path.open("wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: Path) -> TfidfVectorizer:
    with path.open("rb") as f:
        return pickle.load(f)

