"""Tests for the hybrid (ML + rules) priority logic."""

import pytest

from src.priority import URGENT_KEYWORDS, apply_priority_rules, find_urgent_keyword


@pytest.mark.parametrize("keyword", URGENT_KEYWORDS)
def test_every_urgent_keyword_forces_high(keyword):
    """Any urgent keyword overrides a Low ML prediction to High."""
    priority, reason = apply_priority_rules(f"this is about {keyword} please help", "Low")
    assert priority == "High"
    assert reason == f"Escalated: contains keyword '{keyword}'"


def test_keyword_match_ignores_case():
    """Keywords are found regardless of upper or lower case."""
    priority, _ = apply_priority_rules("There is a FRAUD charge on my card", "Medium")
    assert priority == "High"


def test_no_keyword_keeps_ml_prediction():
    """Without urgent words the ML prediction is used unchanged."""
    assert apply_priority_rules("the colour is slightly off", "Low") == ("Low", "ML prediction")


def test_keyword_must_be_whole_word():
    """'fire' inside 'firefox' must not trigger escalation."""
    assert find_urgent_keyword("the site does not load in firefox") is None
