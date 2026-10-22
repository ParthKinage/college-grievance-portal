"""Tests for the full prediction pipeline (needs trained models)."""

import pytest

from src.predict import analyze_complaint
from src.router import DEPARTMENT_BY_CATEGORY, MANUAL_REVIEW_DEPARTMENT

EXPECTED_KEYS = {
    "category", "confidence", "top_3", "priority", "priority_reason",
    "department", "sla_hours", "needs_review",
}


def test_returns_all_keys_and_valid_values():
    """The result has every key, a known category and a confidence in [0, 1]."""
    result = analyze_complaint("The wifi in the hostel has not worked for a week")
    assert set(result) == EXPECTED_KEYS
    assert result["category"] in DEPARTMENT_BY_CATEGORY
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["priority"] in {"High", "Medium", "Low"}
    assert len(result["top_3"]) == 3


def test_top_3_is_sorted_and_starts_with_prediction():
    """The top 3 list is ordered by probability and its first item is the prediction."""
    result = analyze_complaint("My hall ticket is not generated and the exam is tomorrow")
    probabilities = [item["probability"] for item in result["top_3"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert result["top_3"][0]["category"] == result["category"]


@pytest.mark.parametrize("text, category", [
    ("I paid my semester fees twice and the accounts office has not refunded it", "Fees & Scholarships"),
    ("The mess food was stale and the hostel bathrooms have no water", "Hostel & Mess"),
    ("My revaluation result is pending and marks are wrong on the marksheet", "Exams & Results"),
    ("The professor has not completed the syllabus and skips lectures", "Academics & Faculty"),
    ("The college wifi and the student portal login are not working", "IT & Wi-Fi"),
    ("Seniors force me to do their assignments and make fun of my accent", "Ragging & Harassment"),
])
def test_clear_grievances_are_classified_correctly(text, category):
    """A clearly worded grievance for each category gets the right category and office."""
    result = analyze_complaint(text)
    assert result["category"] == category
    assert result["department"] == DEPARTMENT_BY_CATEGORY[category]


def test_safety_keyword_escalates_in_full_pipeline():
    """A ragging grievance gets High priority and a 4 hour SLA."""
    result = analyze_complaint("Seniors are ragging me in the hostel every night")
    assert result["priority"] == "High"
    assert result["sla_hours"] == 4
    assert "ragging" in result["priority_reason"]


def test_low_confidence_goes_to_manual_review():
    """A vague grievance with low confidence is sent to the Manual Review Desk."""
    result = analyze_complaint("things are not good here")
    assert result["needs_review"] is True
    assert result["department"] == MANUAL_REVIEW_DEPARTMENT
