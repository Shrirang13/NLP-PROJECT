"""
Main entry point for Morphology-Aware Hinglish Text Normalization project.

Pipeline stages:
1) normalization
2) tokenization
3) token-level language detection
4) morphology analysis
5) spell correction
6) Hinglish conversion
7) grammar correction
8) final clean output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords

from modules.grammar_corrector import GrammarCorrector
from modules.hinglish_converter import HinglishConverter
from modules.language_detector import LanguageDetector
from modules.morphology import MorphologyAnalyzer
from modules.normalizer import TextNormalizer
from modules.spell_corrector import SpellCorrector
from modules.tokenizer_module import NLTKTokenizer


def load_hinglish_map(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_english_words(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


class HinglishNLPPipeline:
    """End-to-end rule-based Hinglish correction pipeline."""

    def __init__(self, data_dir: Path) -> None:
        nltk.download("stopwords", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        self.hinglish_map = load_hinglish_map(data_dir / "hinglish_dictionary.json")
        self.english_words = load_english_words(data_dir / "english_dictionary.txt")

        vocabulary = set(self.english_words)
        protected_words = set(self.hinglish_map.keys()) | set(self.hinglish_map.values()) | {
            "vo",
            "ho",
            "tum",
            "main",
            "mai",
            "me",
            "hu",
            "rha",
            "rhi",
            "jaunga",
            "ghar",
            "ladka",
            "dekhne",
        }
        self.stop_words = set(stopwords.words("english"))
        self.normalizer = TextNormalizer()
        self.tokenizer = NLTKTokenizer()
        self.language_detector = LanguageDetector(self.english_words, self.hinglish_map)
        self.morphology = MorphologyAnalyzer()
        self.spell_corrector = SpellCorrector(
            vocabulary=vocabulary,
            corpus_tokens=self.english_words,
            protected_words=protected_words,
        )
        self.converter = HinglishConverter(self.hinglish_map)
        self.grammar_corrector = GrammarCorrector()

    def process(self, text: str) -> Dict[str, object]:
        """Run all modules and return intermediate + final outputs."""
        normalized = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(normalized)
        languages = self.language_detector.detect_tokens(tokens)
        morphology = self.morphology.analyze(tokens)
        corrected_spell = self.spell_corrector.correct_tokens(tokens)
        converted = self.converter.convert_tokens(corrected_spell)
        grammar_corrected = self.grammar_corrector.correct(converted)
        pos_tags = pos_tag(grammar_corrected)

        # Stopword filtering is shown as a side artifact (not used in correction).
        non_stopword_tokens = [t for t in grammar_corrected if t not in self.stop_words]
        final_text = " ".join(grammar_corrected)

        return {
            "input": text,
            "normalized": normalized,
            "tokens": tokens,
            "language_detection": languages,
            "morphology": morphology,
            "spell_corrected_tokens": corrected_spell,
            "converted_tokens": converted,
            "grammar_corrected_tokens": grammar_corrected,
            "pos_tags": pos_tags,
            "non_stopword_tokens": non_stopword_tokens,
            "final_output": final_text,
        }


def pretty_print_result(result: Dict[str, object]) -> None:
    print(f"Input: {result['input']}")
    print(f"Normalized: {result['normalized']}")
    print(f"Tokenized: {result['tokens']}")
    print(f"Detected Language: {result['language_detection']}")
    print(f"Morphology: {result['morphology']}")
    print(f"Spell Corrected Tokens: {result['spell_corrected_tokens']}")
    print(f"Hinglish Converted Tokens: {result['converted_tokens']}")
    print(f"POS Tags: {result['pos_tags']}")
    print(f"Non-stopword Tokens: {result['non_stopword_tokens']}")
    print(f"Corrected Output: {result['final_output']}")
    print("-" * 90)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    pipeline = HinglishNLPPipeline(data_dir=data_dir)

    test_sentences = [
        "mai kal colleg ja rha hu",
        "me ghar ja rha",
        "tum kal movie dekhne ja rhi ho",
        "mai mai market jaunga hu",
        "vo acha ladka hoon",
    ]

    for sentence in test_sentences:
        result = pipeline.process(sentence)
        pretty_print_result(result)
