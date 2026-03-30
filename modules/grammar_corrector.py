"""
Rule-based grammar correction for normalized Hinglish output.
"""

from __future__ import annotations

from typing import List

import nltk
from nltk import pos_tag


class GrammarCorrector:
    """
    Apply lightweight grammar rules:
    - remove consecutive repeated words
    - fill missing auxiliary after first-person pronoun 'main'
    - simple subject-auxiliary agreement fixes
    """

    def __init__(self) -> None:
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    @staticmethod
    def _remove_repeated_words(tokens: List[str]) -> List[str]:
        cleaned = []
        for tok in tokens:
            if not cleaned or cleaned[-1] != tok:
                cleaned.append(tok)
        return cleaned

    @staticmethod
    def _fix_common_auxiliaries(tokens: List[str]) -> List[str]:
        fixed = tokens[:]
        n = len(fixed)

        # Rule 1: "main ... hu" -> "main ... hoon"
        for i, tok in enumerate(fixed):
            if tok == "hu":
                fixed[i] = "hoon"

        # Rule 2: For "main" sentence missing auxiliary at end, add "hoon".
        if n > 0 and fixed[0] == "main":
            if fixed[-1] not in {"hoon", "hai", "the", "tha", "thi"}:
                fixed.append("hoon")

        # Rule 3: "woh ... hoon" is usually wrong; replace with "hai".
        if n > 0 and fixed[0] in {"woh", "vo", "yeh", "yah"}:
            fixed = ["hai" if tok == "hoon" else tok for tok in fixed]

        return fixed

    def correct(self, tokens: List[str]) -> List[str]:
        """Run grammar correction using simple token and POS-aware rules."""
        tokens = [tok.lower() for tok in tokens]
        tokens = self._remove_repeated_words(tokens)
        tokens = self._fix_common_auxiliaries(tokens)

        # POS tags are mainly used for visibility/extendability here.
        _ = pos_tag(tokens)
        return tokens
