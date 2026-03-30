"""
Dictionary-based token-level language detector.

Labels each token as:
- Hindi
- English
- Unknown
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set


class LanguageDetector:
    """Detect token language using simple dictionary matching."""

    def __init__(self, english_words: Iterable[str], hinglish_map: Dict[str, str]) -> None:
        # Hinglish map keys are noisy Hindi tokens in Roman script.
        self.hindi_roman_words: Set[str] = {w.lower() for w in hinglish_map.keys()}
        # Hinglish map values are normalized forms; still often Hindi (Romanized).
        self.hindi_roman_words.update({w.lower() for w in hinglish_map.values()})
        self.english_words: Set[str] = {w.lower() for w in english_words}

    def detect_token(self, token: str) -> str:
        """Return language label for a single token."""
        tok = token.lower()
        if tok in self.hindi_roman_words:
            return "Hindi"
        if tok in self.english_words:
            return "English"
        if tok.isalpha():
            return "Unknown"
        return "Unknown"

    def detect_tokens(self, tokens: List[str]) -> Dict[str, str]:
        """Return token -> language dictionary for all tokens."""
        return {token: self.detect_token(token) for token in tokens}
