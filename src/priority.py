"""
Hybrid priority logic: machine learning + simple rules.

The ML model predicts a priority from the complaint text. Then a rule checks
for urgent keywords; if one is present the priority is forced to High.
Rules act as a safety net for cases the model must never get wrong.
"""

import re

URGENT_KEYWORDS = [
    # General urgency, money and legal words
    "urgent", "immediately", "asap", "fraud", "hacked", "unauthorized",
    "stolen", "legal", "lawyer", "court", "police", "dangerous", "injury",
    "fire", "emergency",
    # Student safety words: these must always reach someone within hours
    "ragging", "ragged", "harassment", "harassed", "harassing", "threatened",
    "assaulted", "unsafe", "food poisoning", "suicide",
]


def find_urgent_keyword(text: str):
    """Return the first urgent keyword found in the text, or None."""
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    for keyword in URGENT_KEYWORDS:
        # \b = word boundary, so "fire" matches "fire!" but not "firefox"
        if re.search(rf"\b{keyword}\b", lowered):
            return keyword
    return None


def apply_priority_rules(text: str, ml_priority: str):
    """Combine the ML priority with keyword rules; return (priority, reason)."""
    keyword = find_urgent_keyword(text)
    if keyword:
        return "High", f"Escalated: contains keyword '{keyword}'"
    return ml_priority, "ML prediction"


def predict_priority(text: str, features, priority_model):
    """Predict priority with the ML model, then apply the keyword rules."""
    ml_priority = priority_model.predict(features)[0]
    return apply_priority_rules(text, ml_priority)
