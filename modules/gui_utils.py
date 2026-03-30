"""
Helper utilities for the Gradio GUI.

This module:
- loads the trained TF-IDF vectorizer and best model
- reuses the existing Hinglish correction pipeline
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

from modules.feature_extraction import transform_with_tfidf, load_vectorizer
from main import HinglishNLPPipeline
from modules.model_training import preprocess_for_ml


def load_vectorizer_and_model(models_dir: Path):
    """Load persisted TF-IDF vectorizer and best classification model."""
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    model_path = models_dir / "best_model.pkl"

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found at {model_path}")

    vectorizer = load_vectorizer(vectorizer_path)
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def classify_and_correct(
    text: str,
    data_dir: Path,
    models_dir: Path,
) -> Tuple[str, str, float]:
    """
    Run ML classification, then feed sentence to existing correction pipeline.

    Returns:
    - predicted_label
    - corrected_sentence
    - confidence_score
    """
    vectorizer, model = load_vectorizer_and_model(models_dir)
    processed_text = preprocess_for_ml(text)
    X = transform_with_tfidf(vectorizer, [processed_text])

    probs = getattr(model, "predict_proba", None)
    if probs is not None:
        proba = probs(X)[0]
        pred_idx = proba.argmax()
        confidence = float(proba[pred_idx])
        predicted_label = model.classes_[pred_idx]
    else:
        # Models like LinearSVC do not expose predict_proba.
        pred = model.predict(X)[0]
        predicted_label = pred
        confidence = 0.0

    pipeline = HinglishNLPPipeline(data_dir=data_dir)
    result = pipeline.process(text)
    corrected = result["final_output"]
    return str(predicted_label), corrected, confidence

