"""
Analyse a new complaint: category, confidence, priority and routing.

The three saved model files are loaded once and reused for every call.
"""

import os

import joblib

from src.preprocess import clean_text
from src.priority import predict_priority
from src.router import MANUAL_REVIEW_DEPARTMENT, route

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Below this confidence the prediction goes to a human instead of a department
CONFIDENCE_THRESHOLD = 0.50

# Loaded models are cached here so the files are read from disk only once
_models = {}


def load_models():
    """Load the vectorizer, category model and priority model (only once)."""
    if not _models:
        names = ["vectorizer", "category_model", "priority_model"]
        for name in names:
            path = os.path.join(MODELS_DIR, f"{name}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file '{path}' not found. "
                    "Please run 'python -m src.train' first to train the models."
                )
            _models[name] = joblib.load(path)
    return _models


def analyze_complaint(text: str) -> dict:
    """Classify, prioritise and route a complaint; return all results as a dict."""
    models = load_models()

    # Same cleaning + TF-IDF as during training
    features = models["vectorizer"].transform([clean_text(text)])

    # Probability for every category, e.g. [0.05, 0.80, 0.03, ...]
    category_model = models["category_model"]
    probabilities = category_model.predict_proba(features)[0]
    classes = category_model.classes_

    # Sort categories from most to least likely and keep the top 3
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    top_3 = [{"category": c, "probability": round(float(p), 4)} for c, p in ranked[:3]]
    category, confidence = ranked[0][0], float(ranked[0][1])

    priority, priority_reason = predict_priority(text, features, models["priority_model"])
    routing = route(category, priority)

    # A weak prediction should be checked by a person, not auto-routed
    needs_review = confidence < CONFIDENCE_THRESHOLD
    department = MANUAL_REVIEW_DEPARTMENT if needs_review else routing["department"]

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "top_3": top_3,
        "priority": priority,
        "priority_reason": priority_reason,
        "department": department,
        "sla_hours": routing["sla_hours"],
        "needs_review": needs_review,
    }
