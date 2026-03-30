"""
Training script for error-type classification models.

Workflow (academic-report friendly):
1. Dataset loading and exploration
2. Text preprocessing (for ML)
3. Feature extraction (TF-IDF)
4. Train multiple models
5. Performance evaluation and comparison
6. Best model selection and persistence
7. Confusion matrix generation
"""

from __future__ import annotations

from pathlib import Path

from modules.feature_extraction import load_vectorizer
from modules.model_training import (
    generate_confusion_matrix_plot,
    load_and_explore_dataset,
    save_best_model,
    train_models,
    visualize_label_distribution,
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    results_dir = project_root / "results"

    csv_path = data_dir / "hinglish_error_dataset.csv"

    # 1. Dataset loading and exploration
    df = load_and_explore_dataset(csv_path)
    visualize_label_distribution(df, results_dir / "label_distribution.png")

    # 2–5. Training multiple models and collecting metrics
    trained_models, metrics, best_name = train_models(
        df=df, models_dir=models_dir, results_dir=results_dir
    )

    print(f"\nBest model selected: {best_name}")

    # 6. Persist best model
    best_model = trained_models[best_name]
    save_best_model(best_model, models_dir=models_dir)

    # 7. Confusion matrix for best model
    vectorizer = load_vectorizer(models_dir / "tfidf_vectorizer.pkl")
    generate_confusion_matrix_plot(
        model=best_model,
        vectorizer=vectorizer,
        df=df,
        results_dir=results_dir,
    )

