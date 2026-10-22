"""
Explainability: look inside the trained models.

Linear models (Logistic Regression, Linear SVM) learn one weight per word per
category. A big positive weight means "this word pushes the grievance toward
this category". Naive Bayes learns word probabilities instead, which we turn
into a similar score. These helpers let the app show:
    - the top words each model learned for each category
    - which words in one grievance caused its prediction
"""

import numpy as np

from src.preprocess import clean_text


def get_word_weights(model):
    """Return (classes, weights) where weights[i][j] = how much word j pushes toward class i."""
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV trains one LinearSVC per CV fold: average their weights
        weights = np.mean([c.estimator.coef_ for c in model.calibrated_classifiers_], axis=0)
    elif hasattr(model, "coef_"):
        weights = model.coef_
    elif hasattr(model, "feature_log_prob_"):
        # Naive Bayes: how much more likely a word is in this class than on average
        log_prob = model.feature_log_prob_
        weights = log_prob - log_prob.mean(axis=0)
    else:
        raise TypeError(f"Cannot read word weights from {type(model).__name__}")

    weights = np.asarray(weights)
    # A two-class linear model stores a single row; expand it to one row per class
    if weights.shape[0] == 1 and len(model.classes_) == 2:
        weights = np.vstack([-weights[0], weights[0]])
    return list(model.classes_), weights


def top_words(model, vectorizer, n=10):
    """Return {class: [(word, weight), ...]} with the n strongest words per class."""
    classes, weights = get_word_weights(model)
    words = vectorizer.get_feature_names_out()
    result = {}
    for i, cls in enumerate(classes):
        best = np.argsort(weights[i])[::-1][:n]  # indexes of the largest weights
        result[cls] = [(words[j], round(float(weights[i][j]), 4)) for j in best]
    return result


def explain_prediction(text, category, model, vectorizer, n=6):
    """Return the words in this text that pushed it most toward `category`.

    contribution = (TF-IDF value of the word in this text) x (word weight for the category)
    Only words that actually appear in the text and push toward the category are returned.
    """
    classes, weights = get_word_weights(model)
    if category not in classes:
        return []
    row = vectorizer.transform([clean_text(text)])  # sparse: only words present are non-zero
    words = vectorizer.get_feature_names_out()
    class_weights = weights[classes.index(category)]

    contributions = []
    for j, tfidf_value in zip(row.indices, row.data):
        score = tfidf_value * class_weights[j]
        if score > 0:
            contributions.append((words[j], round(float(score), 4)))
    contributions.sort(key=lambda pair: pair[1], reverse=True)
    return contributions[:n]
