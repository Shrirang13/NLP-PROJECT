"""
Simple evaluation script for correction quality.

Evaluates:
- exact sentence match accuracy
- token-level accuracy
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from main import HinglishNLPPipeline


def load_samples(path: Path) -> List[Tuple[str, str]]:
    """Load tab-separated (input, expected_output) pairs."""
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            inp, expected = line.split("\t", 1)
            pairs.append((inp.strip(), expected.strip()))
    return pairs


def token_accuracy(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not gold_tokens:
        return 1.0
    correct = sum(1 for p, g in zip(pred_tokens, gold_tokens) if p == g)
    return correct / len(gold_tokens)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    samples_file = data_dir / "hinglish_samples.txt"

    pipeline = HinglishNLPPipeline(data_dir=data_dir)
    samples = load_samples(samples_file)

    if not samples:
        raise ValueError("No evaluation samples found in data/hinglish_samples.txt")

    exact_matches = 0
    token_scores: List[float] = []

    print("Evaluation Results")
    print("=" * 80)
    for i, (inp, expected) in enumerate(samples, start=1):
        prediction = pipeline.process(inp)["final_output"]
        is_exact = prediction == expected
        if is_exact:
            exact_matches += 1
        t_score = token_accuracy(prediction, expected)
        token_scores.append(t_score)

        print(f"{i}. Input    : {inp}")
        print(f"   Expected : {expected}")
        print(f"   Predicted: {prediction}")
        print(f"   Exact    : {'Yes' if is_exact else 'No'}")
        print(f"   TokenAcc : {t_score:.2f}")
        print("-" * 80)

    sentence_accuracy = exact_matches / len(samples)
    avg_token_accuracy = sum(token_scores) / len(token_scores)

    print(f"Total samples           : {len(samples)}")
    print(f"Sentence-level accuracy : {sentence_accuracy:.2%}")
    print(f"Average token accuracy  : {avg_token_accuracy:.2%}")
