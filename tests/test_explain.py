"""Tests for the explainability helpers (needs trained models)."""

import joblib
import pytest

from src.explain import explain_prediction, get_word_weights, top_words
from src.predict import MODELS_DIR, load_models


@pytest.fixture(scope="module")
def models():
    """Load the saved vectorizer and models once for all tests in this file."""
    return load_models()


def test_top_words_has_every_category(models):
    """Each category gets the requested number of top words."""
    words = top_words(models["category_model"], models["vectorizer"], n=5)
    assert len(words) == 6
    assert all(len(pairs) == 5 for pairs in words.values())


def test_top_words_make_sense(models):
    """Obvious words are among the strongest for their category."""
    words = top_words(models["category_model"], models["vectorizer"], n=15)
    assert "mess" in [w for w, _ in words["Hostel & Mess"]]
    assert "scholarship" in [w for w, _ in words["Fees & Scholarships"]]


def test_explain_prediction_returns_words_from_the_text(models):
    """The explanation only uses words that appear in the grievance."""
    text = "My scholarship is pending and the fee receipt is wrong"
    reasons = explain_prediction(text, "Fees & Scholarships",
                                 models["category_model"], models["vectorizer"])
    assert reasons, "expected at least one contributing word"
    assert all(all(part in text.lower() for part in word.split()) for word, _ in reasons)
    assert all(score > 0 for _, score in reasons)


@pytest.mark.parametrize("file", ["naive_bayes", "logistic_regression", "linear_svm"])
def test_weights_work_for_every_candidate(file):
    """Word weights can be read from all three saved candidate models."""
    model = joblib.load(f"{MODELS_DIR}/candidates/{file}.joblib")
    classes, weights = get_word_weights(model)
    assert len(classes) == 6
    assert weights.shape[0] == 6
