"""
Morphological processing using NLTK + spaCy.

Includes:
- stemming (NLTK PorterStemmer)
- lemmatization (spaCy)
- lightweight root-form report for each token
"""

from __future__ import annotations

from typing import Dict, List

import nltk
import spacy
from nltk.stem import PorterStemmer
from spacy.cli import download as spacy_download


class MorphologyAnalyzer:
    """Run stemming and lemmatization for mixed Hinglish tokens."""

    def __init__(self) -> None:
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        self.stemmer = PorterStemmer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Keep setup easy for beginners: auto-download model when missing.
            spacy_download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, tokens: List[str]) -> List[Dict[str, str]]:
        """
        Produce morphological details for each token.
        Returns list of dicts with token, stem, lemma, and POS.
        """
        text = " ".join(tokens)
        doc = self.nlp(text)
        analysis = []
        for token in doc:
            word = token.text
            analysis.append(
                {
                    "token": word,
                    "stem": self.stemmer.stem(word.lower()),
                    "lemma": token.lemma_.lower(),
                    "pos": token.pos_,
                }
            )
        return analysis
