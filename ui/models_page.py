"""AI Models page: inspect the saved models, their performance, what they learned, and try them."""

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from src.clock import from_timestamp
from src.explain import explain_prediction, top_words
from src.predict import analyze_complaint, load_models
from src.preprocess import clean_text
from ui.common import (MODELS_DIR, REPORTS_DIR, badge, bar_chart, load_json, page_header,
                       priority_badge)

# What each saved file is for, shown in the file table
FILE_ROLES = {
    "vectorizer.joblib": "TF-IDF vectorizer: turns text into numbers (used by both models)",
    "category_model.joblib": "Category model IN USE: Linear SVM, calibrated for confidence",
    "priority_model.joblib": "Priority model IN USE: Logistic Regression",
    "model_info.json": "Description of all models: settings, accuracy, training date",
    "candidates/naive_bayes.joblib": "Candidate: Naive Bayes (compared, not used)",
    "candidates/logistic_regression.joblib": "Candidate: Logistic Regression (compared, not used)",
    "candidates/linear_svm.joblib": "Candidate: plain Linear SVM (winner, before calibration)",
}

# Files of the models that can be inspected on the "What it learned" and "Try" tabs
CANDIDATE_FILES = {
    "Naive Bayes": "candidates/naive_bayes.joblib",
    "Logistic Regression": "candidates/logistic_regression.joblib",
    "Linear SVM": "candidates/linear_svm.joblib",
}


@st.cache_resource
def load_model_file(relative_path):
    """Load a .joblib model once and keep it in memory (cached across page reloads)."""
    return joblib.load(os.path.join(MODELS_DIR, relative_path))


def saved_models_tab(info):
    """Overview of every saved model file and its settings."""
    trained = datetime.fromisoformat(info["trained_at"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trained on", f"{trained:%d %b %Y}", help=f"{trained:%H:%M}", border=True)
    k2.metric("Category accuracy", f"{info['category_model']['test_accuracy']:.2%}", border=True)
    k3.metric("Priority accuracy", f"{info['priority_model']['test_accuracy']:.2%}", border=True)
    k4.metric("Vocabulary", f"{info['vectorizer']['vocabulary_size']:,} words", border=True)

    st.markdown("##### Models in use")
    cards = [
        ("Text vectorizer", info["vectorizer"], None),
        ("Category model", info["category_model"], info["category_model"]["test_accuracy"]),
        ("Priority model", info["priority_model"], info["priority_model"]["test_accuracy"]),
    ]
    for col, (title, details, accuracy) in zip(st.columns(3), cards):
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            name = details.get("name", "TF-IDF")  # the vectorizer has no "name" in model_info.json
            st.markdown(badge(name, "violet") + " " + badge(details["type"], "gray"))
            st.caption(f"File: `models/{details['file']}`")
            if accuracy is not None:
                st.markdown(f"Test accuracy: **{accuracy:.2%}**")
            with st.expander("Settings (hyperparameters)"):
                st.json(details["settings"], expanded=True)

    st.markdown("##### All files in `models/`")
    files = []
    for root, _, names in os.walk(MODELS_DIR):
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, MODELS_DIR).replace("\\", "/")
            files.append({
                "file": rel,
                "role": FILE_ROLES.get(rel, ""),
                "size_kb": round(os.path.getsize(path) / 1024, 1),
                "modified": from_timestamp(os.path.getmtime(path)),
            })
    st.dataframe(pd.DataFrame(files).sort_values("file"), hide_index=True, width="stretch",
                 column_config={
                     "size_kb": st.column_config.NumberColumn("Size (KB)"),
                     "modified": st.column_config.DatetimeColumn("Saved", format="DD MMM YYYY, HH:mm"),
                 })
    st.caption(f"Trained with scikit-learn {info['scikit_learn_version']} on "
               f"{info['dataset']['rows']:,} grievances ({info['dataset']['train_size']:,} train / "
               f"{info['dataset']['test_size']} test). Models are saved with joblib.")


def performance_tab():
    """Comparison of the three candidate models, charts and the classification report."""
    metrics = load_json(os.path.join(REPORTS_DIR, "metrics.json"))
    table = pd.DataFrame(metrics["models"]).T
    table.columns = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
                     "CV accuracy (mean)", "CV accuracy (std)"]
    # Bold the best score in each column (works in both light and dark themes)
    styled = table.style.format("{:.4f}").highlight_max(
        axis=0, subset=table.columns[:5], props="font-weight: bold")
    st.dataframe(styled, width="stretch")
    st.markdown(f"**Winner:** {metrics['winner']}, chosen by macro F1. Saved model test accuracy: "
                f"**{metrics['saved_category_model_accuracy']:.2%}**.")

    col1, col2 = st.columns(2)
    col1.image(os.path.join(REPORTS_DIR, "model_comparison.png"), caption="Accuracy and F1 per model")
    col2.image(os.path.join(REPORTS_DIR, "confusion_matrix.png"),
               caption="Confusion matrix: rows are the true category, columns the prediction")

    with st.expander("Classification report (per category)"):
        with open(os.path.join(REPORTS_DIR, "classification_report.txt")) as f:
            st.code(f.read(), language=None)


def learned_tab():
    """Top words each model learned for every class."""
    options = ["Linear SVM (in use)", "Logistic Regression", "Naive Bayes", "Priority model"]
    choice = st.segmented_control("Model", options, default=options[0], key="learned_model")
    choice = choice or options[0]

    models = load_models()
    if choice == "Linear SVM (in use)":
        model = models["category_model"]
    elif choice == "Priority model":
        model = models["priority_model"]
    else:
        model = load_model_file(CANDIDATE_FILES[choice])

    st.caption("Each bar is a word (or two-word phrase) the model learned. A longer bar means the "
               "word pushes a grievance more strongly toward that class.")
    words = top_words(model, models["vectorizer"], n=8)
    columns = st.columns(3)
    for i, (cls, pairs) in enumerate(words.items()):
        with columns[i % 3], st.container(border=True):
            st.markdown(f"**{cls}**")
            data = pd.DataFrame(pairs, columns=["word", "weight"])
            st.altair_chart(bar_chart(data, "word", "weight", "Weight"), width="stretch")


def try_tab():
    """Type a grievance and compare what every model predicts."""
    text = st.text_area("Type a grievance", "My scholarship is pending and the hostel wifi is slow",
                        height=90)
    if not st.button("Compare models", type="primary", icon=":material/compare_arrows:"):
        return
    if len(text.strip()) < 3:
        st.warning("Please type a few words first.")
        return

    models = load_models()
    features = models["vectorizer"].transform([clean_text(text)])
    rows = []
    for name, path in CANDIDATE_FILES.items():
        model = load_model_file(path)
        prediction = model.predict(features)[0]
        # LinearSVC gives no probabilities, which is why the app uses a calibrated version
        confidence = model.predict_proba(features).max() if hasattr(model, "predict_proba") else None
        rows.append({"model": name, "prediction": prediction, "confidence": confidence})
    result = analyze_complaint(text)
    rows.append({"model": "Calibrated Linear SVM (in use)", "prediction": result["category"],
                 "confidence": result["confidence"]})

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "model": "Model", "prediction": "Predicted category",
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1,
                                                      format="percent"),
    })
    agree = len({row["prediction"] for row in rows}) == 1
    st.markdown(":green-badge[All models agree]" if agree else ":orange-badge[Models disagree]")

    reasons = explain_prediction(text, result["category"],
                                 models["category_model"], models["vectorizer"])
    if reasons:
        st.markdown("**Words that pushed the model in use toward "
                    f"{result['category']}:** " + " ".join(badge(w, "gray") for w, _ in reasons))
    st.markdown(f"Final routing: {priority_badge(result['priority'])} "
                f"→ **{result['department']}** within **{result['sla_hours']} hours** "
                f"({result['priority_reason']}).")


def how_it_works_tab():
    """Diagram and plain-English steps of the full pipeline."""
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#4F46E5",
                  fontname="Helvetica", fontsize=11];
            edge [color="#6366F1"];
            text [label="Grievance\\ntext"];
            clean [label="Preprocessing\\n(clean_text)"];
            tfidf [label="TF-IDF\\nvectorizer"];
            cat [label="Category model\\nLinear SVM"];
            pri [label="Priority model\\n+ safety keywords"];
            conf [label="Confidence\\n>= 50%?", shape=diamond, fillcolor="#FFF7ED", color="#F97316"];
            route [label="Routing table\\n(office + SLA)"];
            review [label="Manual\\nReview Desk", fillcolor="#FEF2F2", color="#EF4444"];
            db [label="SQLite\\nticket", shape=cylinder];
            text -> clean -> tfidf;
            tfidf -> cat -> conf;
            tfidf -> pri -> route;
            conf -> route [label=" yes"];
            conf -> review [label=" no"];
            route -> db; review -> db;
        }
    """, width="stretch")
    st.markdown("""
1. **Preprocessing**: the grievance is lowercased and URLs, emails, numbers and punctuation are removed.
2. **TF-IDF**: the cleaned text becomes a vector of 5,000 numbers. Words that are frequent in this
   grievance but rare across all grievances get a higher weight.
3. **Category model**: the Linear SVM gives a probability for each of the six categories. The highest
   one is the predicted category, and its probability is the *confidence*.
4. **Priority**: a second model predicts High / Medium / Low. Safety words such as *ragging*,
   *harassment* or *food poisoning* always force High.
5. **Routing**: lookup tables map the category to a college office and the priority to a deadline.
   Below 50% confidence the ticket goes to the Manual Review Desk instead.
6. **Ticket**: everything is saved in the SQLite database and shown on the Dashboard.
""")


def render():
    """Draw the Model Explorer page."""
    page_header("Model Explorer",
                "See the trained machine learning models, how well they perform and what they learned.")
    info = load_json(os.path.join(MODELS_DIR, "model_info.json"))
    if info is None:
        st.error("No trained models found. Run `python -m src.train` first.", icon=":material/error:")
        return

    tabs = st.tabs([":material/inventory_2: Saved models", ":material/leaderboard: Performance",
                    ":material/psychology: What it learned", ":material/science: Try the models",
                    ":material/account_tree: How it works"])
    with tabs[0]:
        saved_models_tab(info)
    with tabs[1]:
        performance_tab()
    with tabs[2]:
        learned_tab()
    with tabs[3]:
        try_tab()
    with tabs[4]:
        how_it_works_tab()
