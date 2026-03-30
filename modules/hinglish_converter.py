"""
Hinglish-to-standard token conversion using JSON mapping.
"""

from __future__ import annotations

from typing import Dict, List


class HinglishConverter:
    """Convert informal Hinglish variants into standard forms."""

    def __init__(self, hinglish_map: Dict[str, str]) -> None:
        self.hinglish_map = {k.lower(): v.lower() for k, v in hinglish_map.items()}

    def convert_tokens(self, tokens: List[str]) -> List[str]:
        """Apply mapping to each token if available."""
        return [self.hinglish_map.get(tok.lower(), tok.lower()) for tok in tokens]
