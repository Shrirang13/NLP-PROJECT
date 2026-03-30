"""
Model training and evaluation utilities.

Trains multiple classifiers to detect Hinglish error type:
- spelling
- grammar
- repetition
- normalization
- clean
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from modules.feature_extraction import (
    build_tfidf_vectorizer,
    save_vectorizer,
    transform_with_tfidf,
)
from modules.normalizer import TextNormalizer
from modules.tokenizer_module import NLTKTokenizer


ERROR_LABELS = ["spelling", "grammar", "repetition", "normalization", "clean"]


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def preprocess_for_ml(text: str) -> str:
    """
    Preprocessing pipeline for ML models.

    Reuses existing modules:
    - normalizer
    - tokenizer
    and then rejoins cleaned tokens into a single string.
    """
    normalizer = TextNormalizer()
    tokenizer = NLTKTokenizer()

    normalized = normalizer.normalize(text)
    tokens = tokenizer.tokenize(normalized)
    # Simple form: rejoin tokens. If you want, you can add stemming/lemmatization here.
    return " ".join(tokens)


def load_and_explore_dataset(csv_path: Path) -> pd.DataFrame:
    """Load dataset and print basic exploration info for report screenshots."""
    df = pd.read_csv(csv_path)
    print("Dataset head:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    return df


def visualize_label_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plot label distribution and save to file for report."""
    import matplotlib.pyplot as plt

    counts = df["label"].value_counts().reindex(ERROR_LABELS)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color="skyblue")
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_models(
    df: pd.DataFrame,
    models_dir: Path,
    results_dir: Path,
) -> Tuple[Dict[str, object], Dict[str, ModelResult], str]:
    """
    Train multiple models and return:
    - trained_models: name -> model instance
    - metrics: name -> ModelResult
    - best_model_name
    """
    texts = [preprocess_for_ml(t) for t in df["sentence"].tolist()]
    labels = df["label"].astype(str).tolist()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build TF-IDF vectorizer on training texts only.
    vectorizer, X_train = build_tfidf_vectorizer(X_train_text)
    X_test = transform_with_tfidf(vectorizer, X_test_text)

    # Save vectorizer for future inference.
    models_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path as _P

    save_vectorizer(vectorizer, _P(models_dir, "tfidf_vectorizer.pkl"))

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MultinomialNB": MultinomialNB(),
        "LinearSVM": LinearSVC(),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    trained_models: Dict[str, object] = {}
    metrics: Dict[str, ModelResult] = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

        trained_models[name] = clf
        metrics[name] = ModelResult(name, acc, prec, rec, f1)

    # Save comparison table.
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(
        [
            {
                "Model": m.name,
                "Accuracy": m.accuracy,
                "Precision": m.precision,
                "Recall": m.recall,
                "F1": m.f1,
            }
            for m in metrics.values()
        ]
    ).sort_values(by="F1", ascending=False)
    comparison_path = results_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\nModel comparison saved to:", comparison_path)
    print(comparison_df)

    # Choose best model by F1.
    best_model_name = comparison_df.iloc[0]["Model"]
    return trained_models, metrics, best_model_name


def save_best_model(model: object, models_dir: Path) -> Path:
    """Persist the best model to disk."""
    import pickle

    path = models_dir / "best_model.pkl"
    with path.open("wb") as f:
        pickle.dump(model, f)
    print("Best model saved to:", path)
    return path


def generate_confusion_matrix_plot(
    model: object,
    vectorizer,
    df: pd.DataFrame,
    results_dir: Path,
) -> Path:
    """Train/test confusion matrix on a fresh split and save plot."""
    import matplotlib.pyplot as plt

    texts = [preprocess_for_ml(t) for t in df["sentence"].tolist()]
    labels = df["label"].astype(str).tolist()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=123, stratify=labels
    )
    X_train = transform_with_tfidf(vectorizer, X_train_text)
    X_test = transform_with_tfidf(vectorizer, X_test_text)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=ERROR_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ERROR_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()

    results_dir.mkdir(parents=True, exist_ok=True)
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close(fig)
    print("Confusion matrix saved to:", cm_path)
    return cm_path

