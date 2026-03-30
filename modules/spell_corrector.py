"""
Dictionary-based spell correction using edit distance.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Set

from nltk.metrics.distance import edit_distance


class SpellCorrector:
    """Correct tokens by nearest dictionary word with optional frequency tie-break."""

    def __init__(
        self,
        vocabulary: Iterable[str],
        corpus_tokens: Optional[Iterable[str]] = None,
        protected_words: Optional[Iterable[str]] = None,
    ) -> None:
        self.vocabulary: Set[str] = {w.lower() for w in vocabulary}
        self.freq = Counter(tok.lower() for tok in (corpus_tokens or []))
        self.protected_words: Set[str] = {w.lower() for w in (protected_words or [])}

    def _best_candidate(self, token: str, max_distance: int = 2) -> str:
        """Pick best correction candidate within edit-distance threshold."""
        token_l = token.lower()
        if token_l in self.vocabulary or token_l in self.protected_words or not token_l.isalpha():
            return token

        candidates = []
        for word in self.vocabulary:
            # Avoid aggressive unrelated corrections like "vo" -> "go".
            if token_l and word and token_l[0] != word[0]:
                continue
            dist = edit_distance(token_l, word)
            if dist <= max_distance:
                # Lower distance is better, higher frequency is better.
                candidates.append((dist, -self.freq.get(word, 0), word))

        if not candidates:
            return token

        candidates.sort()
        return candidates[0][2]

    def correct_tokens(self, tokens: List[str]) -> List[str]:
        """Correct each token independently."""
        return [self._best_candidate(tok) for tok in tokens]
