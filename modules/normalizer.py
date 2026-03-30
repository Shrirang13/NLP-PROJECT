"""
Text normalization utilities for Hinglish input.

This module performs:
1) lowercase conversion
2) repeated character normalization
3) whitespace cleanup
4) abbreviation expansion
"""

from __future__ import annotations

import re
from typing import Dict


class TextNormalizer:
    """Normalize noisy Hinglish text before downstream NLP processing."""

    def __init__(self) -> None:
        # Common casual abbreviations or variants seen in Hinglish chats.
        self.abbreviation_map: Dict[str, str] = {
            "u": "you",
            "ur": "your",
            "plz": "please",
            "pls": "please",
            "btw": "by the way",
            "bcz": "because",
            "coz": "because",
            "dont": "don't",
            "cant": "can't",
            "wanna": "want to",
            "gonna": "going to",
            "kal": "kal",  # keep Hinglish temporal token unchanged
            "hn": "haan",
            "h": "hai",
        }

    def _remove_repeated_characters(self, text: str) -> str:
        """
        Reduce excessive character repetition.
        Example: 'cooool' -> 'cool', 'pleaaase' -> 'please'
        """
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def _normalize_abbreviations(self, text: str) -> str:
        """Expand abbreviations token-by-token while preserving unknown tokens."""
        tokens = text.split()
        normalized_tokens = [self.abbreviation_map.get(tok, tok) for tok in tokens]
        return " ".join(normalized_tokens)

    def normalize(self, text: str) -> str:
        """Run full text normalization pipeline."""
        text = text.lower()
        text = self._remove_repeated_characters(text)
        text = self._normalize_abbreviations(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
