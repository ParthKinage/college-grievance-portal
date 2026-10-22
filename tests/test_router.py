"""Tests for rule-based routing."""

import pytest

from src.router import route

EXPECTED_DEPARTMENTS = {
    "Hostel & Mess": "Hostel Warden Office",
    "Exams & Results": "Examination Cell",
    "Fees & Scholarships": "Accounts Office",
    "Academics & Faculty": "Head of Department",
    "IT & Wi-Fi": "IT Services Cell",
    "Ragging & Harassment": "Anti-Ragging Committee",
}
EXPECTED_SLA = {"High": 4, "Medium": 24, "Low": 72}


@pytest.mark.parametrize("category", EXPECTED_DEPARTMENTS)
@pytest.mark.parametrize("priority", EXPECTED_SLA)
def test_route_every_category_and_priority(category, priority):
    """All 18 category/priority combinations give the right department and SLA."""
    result = route(category, priority)
    assert result == {
        "department": EXPECTED_DEPARTMENTS[category],
        "sla_hours": EXPECTED_SLA[priority],
    }


def test_unknown_category_raises():
    """An unknown category is an error, not a silent wrong route."""
    with pytest.raises(ValueError):
        route("Sports", "High")


def test_unknown_priority_raises():
    """An unknown priority is an error."""
    with pytest.raises(ValueError):
        route("Fees & Scholarships", "Critical")
