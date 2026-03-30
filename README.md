# Hinglish Error Classification and Morphology-Aware Text Correction

[Open in Colab](https://colab.research.google.com/drive/1IiSZyajIVLdU8dVR6x3nu1q7J7oUSwts?usp=sharing)

## Project Overview

This project is a hybrid NLP system for Hinglish (Hindi + English code-mixed Roman text).  
It combines:

- **Machine Learning Classification** to identify the error type in a sentence
- **Rule-based Morphology-Aware Correction Pipeline** to generate clean corrected output

The system is designed for academic demonstration, report screenshots, and practical text-cleaning experiments.

## Objective

Given a Hinglish sentence, the system performs:

1. error classification (`spelling`, `grammar`, `repetition`, `normalization`, `clean`)
2. full correction using the existing modular correction engine

Example:

- Input: `mai kal colleg ja rha hu`
- Predicted Class: `spelling`
- Corrected Output: `main kal college ja raha hoon`

## Key Features

### Classification Features

- End-to-end ML training workflow
- TF-IDF based feature extraction
- Multi-model training and comparison
- Automatic best model selection and saving
- Confusion matrix generation for analysis

### Correction Features

- Text normalization
- NLTK tokenization
- Morphology analysis (stemming + lemmatization)
- Dictionary + edit-distance spelling correction
- Hinglish-to-standard token conversion
- Rule-based grammar correction
- Final clean sentence generation

## Dataset Description

- **File**: `data/hinglish_error_dataset.csv`
- **Format**: `sentence,label`
- **Total Samples**: 200
- **Labels**:
  - `spelling`
  - `grammar`
  - `repetition`
  - `normalization`
  - `clean`

The dataset includes realistic Indian Roman Hindi style text with misspellings, repeated words, grammar inconsistencies, normalization issues, and clean examples.

## Technologies Used

- Python
- NLTK
- spaCy
- scikit-learn
- Gradio
- pandas
- matplotlib

## Models Used

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest

## Best Model Result

Based on weighted F1 from the generated comparison:

- **Best Model**: `LinearSVM`
- **Reference file**: `results/model_comparison.csv`
- **Saved model**: `models/best_model.pkl`
- **Vectorizer**: `models/tfidf_vectorizer.pkl`

## Project Structure

```text
project/
│
├── data/
│   ├── hinglish_samples.txt
│   ├── hinglish_dictionary.json
│   ├── english_dictionary.txt
│   └── hinglish_error_dataset.csv
│
├── modules/
│   ├── normalizer.py
│   ├── tokenizer_module.py
│   ├── language_detector.py
│   ├── morphology.py
│   ├── spell_corrector.py
│   ├── hinglish_converter.py
│   ├── grammar_corrector.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   └── gui_utils.py
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── label_distribution.png
│
├── app.py
├── train_models.py
├── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```

## How to Run Locally

From inside the `project` folder:

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Train and compare models:

```bash
python train_models.py
```

3. Run correction pipeline demo:

```bash
python main.py
```

4. Run evaluation script:

```bash
python evaluation.py
```

5. Launch Gradio GUI:

```bash
python app.py
```

## How to Run on Colab

Use this notebook link: [Open in Colab](https://colab.research.google.com/drive/1IiSZyajIVLdU8dVR6x3nu1q7J7oUSwts?usp=sharing)

Typical Colab steps:

```python
%cd /content/project
!python -m pip install -r requirements.txt
!python train_models.py
!python app.py
```

## Sample Input / Output

### Example 1
- Input: `mai kal colleg ja rha hu`
- Predicted Class: `spelling`
- Corrected Output: `main kal college ja raha hoon`

### Example 2
- Input: `mai mai market jaunga hu`
- Predicted Class: `repetition`
- Corrected Output: `main market jaunga hoon`

### Example 3
- Input: `vo acha ladka hoon`
- Predicted Class: `grammar`
- Corrected Output: `vo achha ladka hai`

## Future Scope

- Expand dataset size and diversity with real social media Hinglish text
- Add advanced linguistic features (POS patterns, morphology-derived features)
- Improve confidence calibration for all models
- Add REST API deployment for web/mobile integration
- Add unit tests and CI/CD for robust project maintenance
