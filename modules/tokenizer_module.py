"""
Tokenization module using NLTK.
"""

from __future__ import annotations

from typing import List

import nltk
from nltk.tokenize import word_tokenize


class NLTKTokenizer:
    """Wrapper around NLTK word tokenizer for cleaner project architecture."""

    def __init__(self) -> None:
        # punkt_tab is required by newer NLTK tokenizers in some versions.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a normalized sentence into word tokens."""
        return word_tokenize(text)
