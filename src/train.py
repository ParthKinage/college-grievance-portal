"""
Train the complaint category and priority models.

Steps:
    1. Load the dataset and clean the text
    2. Split 80% train / 20% test
    3. Turn text into numbers with TF-IDF
    4. Train and compare Naive Bayes, Logistic Regression and Linear SVM
    5. Keep the best model (by macro F1) and train a priority model
    6. Save models to models/ (plus all candidates and model_info.json)
       and reports to reports/

Run from the project root:
    python -m src.train
"""

import json
import os
import re
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import clean_text

# All paths are built from the project root, so nothing is hard-coded
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "complaints.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CANDIDATES_DIR = os.path.join(MODELS_DIR, "candidates")  # all 3 compared models
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

RANDOM_STATE = 42  # same split and same results on every run


def load_data():
    """Load the complaint CSV and add a cleaned text column."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run 'python data/generate_dataset.py' first."
        )
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["complaint_text"].apply(clean_text)
    return df


def get_candidate_models():
    """Return the three classifiers we want to compare, keyed by name."""
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train one model and return its test scores and 5-fold CV accuracy."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # "macro" = average the score of each class equally, so no class dominates
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    # Cross-validation: train/test 5 times on different slices of the training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
    }


def save_confusion_matrix(model, X_test, y_test, model_name):
    """Save a confusion matrix image for the given model."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def save_comparison_chart(results):
    """Save a grouped bar chart of accuracy and macro F1 for every model."""
    names = list(results)
    accuracy = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]
    x = np.arange(len(names))  # one position per model
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2a78d6")
    bars2 = ax.bar(x + width / 2, f1, width, label="Macro F1", color="#eb6f34")
    ax.bar_label(bars1, fmt="%.3f", fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(min(accuracy + f1) - 0.1, 1.0)  # zoom in so differences are visible
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)


def file_name_for(model_name):
    """Turn a model name like "Linear SVM" into a file name like "linear_svm.joblib"."""
    return re.sub(r"[^a-z]+", "_", model_name.lower()).strip("_") + ".joblib"


def simple_params(model):
    """Return a model's settings that can be written to JSON (numbers, text, booleans)."""
    params = {}
    for key, value in model.get_params(deep=False).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            params[key] = value
        elif isinstance(value, tuple):
            params[key] = list(value)            # e.g. ngram_range (1, 2) -> [1, 2]
        elif isinstance(value, type):
            params[key] = value.__name__         # e.g. dtype float64
        else:
            params[key] = type(value).__name__  # e.g. the LinearSVC inside the calibrator
    return params


def save_model_info(df, train_df, test_df, vectorizer, results, winner, best_model,
                    final_accuracy, priority_model, priority_accuracy):
    """Write models/model_info.json: a readable description of every saved model."""
    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "scikit_learn_version": sklearn.__version__,
        "dataset": {
            "rows": len(df),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "categories": sorted(df["category"].unique()),
            "priorities": ["High", "Medium", "Low"],
        },
        "vectorizer": {
            "file": "vectorizer.joblib",
            "type": type(vectorizer).__name__,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "settings": simple_params(vectorizer),
        },
        "category_model": {
            "file": "category_model.joblib",
            "name": winner,
            "type": type(best_model).__name__,
            "test_accuracy": round(final_accuracy, 4),
            "settings": simple_params(best_model),
        },
        "priority_model": {
            "file": "priority_model.joblib",
            "name": "Logistic Regression",
            "type": type(priority_model).__name__,
            "test_accuracy": round(priority_accuracy, 4),
            "settings": simple_params(priority_model),
        },
        "candidates": [
            {"name": name, "file": f"candidates/{file_name_for(name)}", **scores}
            for name, scores in results.items()
        ],
    }
    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def print_summary(results, winner, final_accuracy, priority_accuracy):
    """Print a readable results table to the terminal."""
    table = pd.DataFrame(results).T
    table.columns = ["Accuracy", "Precision", "Recall", "F1 (macro)", "CV Acc", "CV Std"]
    print("\n" + "=" * 78)
    print("CATEGORY MODEL COMPARISON (test set = 20% of data)")
    print("=" * 78)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("-" * 78)
    print(f"Best model (by macro F1): {winner}")
    print(f"Saved category model accuracy: {final_accuracy:.4f}")
    print(f"Priority model (Logistic Regression) accuracy: {priority_accuracy:.4f}")
    print("=" * 78)


def main():
    """Run the full training pipeline and save models and reports."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load and clean
    df = load_data()
    print(f"Loaded {len(df)} complaints.")

    # 2. Split 80/20. stratify keeps the same category mix in train and test.
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["category"], random_state=RANDOM_STATE
    )

    # 3. TF-IDF: fit on training text only, so the test set stays truly unseen
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y_train, y_test = train_df["category"], test_df["category"]

    # 4-5. Train and score every candidate model
    models = get_candidate_models()
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 6. Pick the winner by macro F1
    winner = max(results, key=lambda n: results[n]["f1_macro"])
    best_model = models[winner]

    # LinearSVC has no predict_proba, so calibrate it to get probabilities
    if winner == "Linear SVM":
        best_model = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=5)
        best_model.fit(X_train, y_train)

    # Score the model we actually save (calibration can shift results slightly)
    final_accuracy = accuracy_score(y_test, best_model.predict(X_test))

    # 7. Separate priority model on the same TF-IDF features
    priority_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    priority_model.fit(X_train, train_df["priority"])
    priority_accuracy = accuracy_score(test_df["priority"], priority_model.predict(X_test))

    # 8. Save models
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "category_model.joblib"))
    joblib.dump(priority_model, os.path.join(MODELS_DIR, "priority_model.joblib"))
    # Also keep all three compared models, so they can be inspected and compared in the app
    for name, model in models.items():
        joblib.dump(model, os.path.join(CANDIDATES_DIR, file_name_for(name)))
    save_model_info(df, train_df, test_df, vectorizer, results, winner, best_model,
                    final_accuracy, priority_model, priority_accuracy)

    # 9. Save reports
    metrics = {
        "models": results,
        "winner": winner,
        "saved_category_model_accuracy": round(final_accuracy, 4),
        "priority_model_accuracy": round(priority_accuracy, 4),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, best_model.predict(X_test), digits=4)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Classification report for winning model: {winner}\n\n{report}")

    save_confusion_matrix(best_model, X_test, y_test, winner)
    save_comparison_chart(results)

    # 10. Summary
    print_summary(results, winner, final_accuracy, priority_accuracy)
    print(f"Models saved to {MODELS_DIR}")
    print(f"Reports saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
