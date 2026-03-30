# Morphology-Aware Hinglish Text Normalization, Spell Correction, Grammar Correction, and Error-Type Classification

This project is a modular NLP pipeline for Hinglish (Hindi + English code-mixed Roman text).  
It combines **rule-based correction** (normalization + spell + grammar) with a **machine-learning classifier** that predicts the type of issue in a sentence before correction.

## Project Objective

- **Input**: Hinglish sentence (e.g. `mai kal colleg ja rha hu`)
- **Step 1 – Classification**: Detect error type  
  (`spelling`, `grammar`, `repetition`, `normalization`, or `clean`)
- **Step 2 – Correction**: Run the existing morphology-aware pipeline to produce cleaned text  
  (e.g. `main kal college ja raha hoon`)

## Updated Project Structure

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

## Preprocessing and Correction Pipeline (Reused)

- **`normalizer.py`**: lowercasing, repeated character cleanup, spacing cleanup, abbreviation normalization.
- **`tokenizer_module.py`**: NLTK `word_tokenize` based tokenization.
- **`language_detector.py`**: token-level Hindi/English/Unknown tagging through dictionary matching.
- **`morphology.py`**: stemming via NLTK PorterStemmer + lemmatization via spaCy.
- **`spell_corrector.py`**: edit-distance spell correction with dictionary and frequency preference.
- **`hinglish_converter.py`**: maps casual Hinglish forms (e.g., `mai`, `rha`, `hu`) to standard forms.
- **`grammar_corrector.py`**: simple rule-based grammar fixes (auxiliary insertion/fixes, repeated words).
- **`main.py`**: runs the full correction pipeline and prints every intermediate stage.

These modules are also used indirectly by the ML part via `preprocess_for_ml` and by the GUI for final correction.

## Dataset Details

- **File**: `data/hinglish_error_dataset.csv`
- **Columns**: `sentence`, `label`
- **Labels**:
  - `spelling`
  - `grammar`
  - `repetition`
  - `normalization`
  - `clean`
- Contains **200+ realistic Hinglish samples** in Indian Roman Hindi style, with:
  - spelling mistakes
  - repeated words
  - missing/misaligned auxiliaries
  - noisy normalisation cases
  - clean sentences

For screenshots in the report, you can show:
- dataset head and shape
- label distribution print-out
- label distribution bar chart (`results/label_distribution.png`)

## ML Preprocessing for Classification

Implemented in `model_training.py` via `preprocess_for_ml(text)`:

- **lowercase + cleanup** using `TextNormalizer`
- **tokenization** using `NLTKTokenizer`
- rejoin tokens into a cleaned string for vectorization

This keeps the pipeline **beginner-friendly** while still realistic for ML experiments.

## Feature Extraction

Implemented in `feature_extraction.py`:

- Uses **TF-IDF** (`sklearn.feature_extraction.text.TfidfVectorizer`)
- **Word-level**:
  - unigrams and bigrams
- **Character-level**:
  - char n-grams (3–5)
- The combined vectorizer is saved as:
  - `models/tfidf_vectorizer.pkl`

## Classification Models

Implemented in `model_training.py` and orchestrated by `train_models.py`:

Trained models:
- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Linear SVM**
- **Random Forest**

Metrics computed:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

Comparison table is saved as:
- `results/model_comparison.csv` (easy to paste into report)

The **best model** (by F1-score) is automatically selected (currently `LinearSVM` in the sample run) and saved to:
- `models/best_model.pkl`

## Confusion Matrix and Results Visuals

- A confusion matrix for the best model is generated using matplotlib and saved as:
  - `results/confusion_matrix.png`
- Label distribution plot:
  - `results/label_distribution.png`

You can directly include these PNG files as **report figures**.

## Gradio GUI (Error Classification + Correction)

- Implementation in `app.py` using **Gradio Blocks**.
- Helper logic in `gui_utils.py`, which:
  - loads `tfidf_vectorizer.pkl` and `best_model.pkl`
  - uses `preprocess_for_ml` for ML input
  - calls `HinglishNLPPipeline` (from `main.py`) for final correction

GUI elements:
- **Text input box** for Hinglish sentence
- **Predicted error class** (`spelling`, `grammar`, `repetition`, `normalization`, `clean`)
- **Corrected sentence** (from rule-based pipeline)
- **Confidence score** (when model supports probabilities; `0.0` for non-probabilistic models)

This screen is ideal for **academic report screenshots**:
- before/after correction
- predicted label
- confidence

## How to Run (Local)

From inside the `project/` folder:

1. **Install dependencies**:

```bash
python -m pip install -r requirements.txt
```

2. **(Optional) Re-train models** and regenerate results:

```bash
python train_models.py
```

This will:
- load and explore `hinglish_error_dataset.csv`
- plot and save label distribution
- train all four models
- save `models/tfidf_vectorizer.pkl`
- save `models/best_model.pkl`
- create `results/model_comparison.csv`
- create `results/confusion_matrix.png`

3. **Run the original correction pipeline only** (for viva explanation of steps):

```bash
python main.py
```

4. **Run the evaluation script for rule-based correction**:

```bash
python evaluation.py
```

5. **Launch the Gradio GUI**:

```bash
python app.py
```

Then open the displayed local URL in a browser to interact with the system.

## How to Run on Google Colab

1. Upload the entire `project/` folder to your Colab environment (or clone from your repo).
2. In Colab, run:

```python
%cd /content/project
!python -m pip install -r requirements.txt
!python train_models.py
```

3. To launch the Gradio app in Colab:

```python
import os
os.chdir("/content/project")
import app  # or !python app.py
```

Gradio will show a shareable public URL that you can open from your browser.

## Good Places for Screenshots (Report)

- **Dataset exploration**: output of `train_models.py` (head, shape, label counts).
- **Label distribution plot**: `results/label_distribution.png`.
- **Model comparison table**: open `results/model_comparison.csv`.
- **Confusion matrix**: `results/confusion_matrix.png`.
- **GUI output**: browser window showing input sentence, predicted label, corrected sentence, confidence.
- **Correction pipeline stages**: terminal output of `python main.py`.

## Future Scope

- Use a larger real-world Hinglish corpus for training.
- Add more advanced features (POS tags, morphology features) into the classifier.
- Explore neural models (e.g. BiLSTM or transformer-based) for classification while keeping the same correction engine.
- Extend grammar rules for more complex sentence structures.
- Add unit tests and CI for automated quality checks.
