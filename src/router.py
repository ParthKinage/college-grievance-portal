"""
Rule-based routing.

Two plain dictionaries decide which college office handles a grievance and
how fast it must be answered. Edit these tables to change routing without
retraining anything.
"""

# Which college office handles each grievance category
DEPARTMENT_BY_CATEGORY = {
    "Hostel & Mess": "Hostel Warden Office",
    "Exams & Results": "Examination Cell",
    "Fees & Scholarships": "Accounts Office",
    "Academics & Faculty": "Head of Department",
    "IT & Wi-Fi": "IT Services Cell",
    "Ragging & Harassment": "Anti-Ragging Committee",
}

# Response deadline (SLA) in hours for each priority
SLA_HOURS_BY_PRIORITY = {
    "High": 4,
    "Medium": 24,
    "Low": 72,
}

# Used when the model is not confident enough to trust its category
MANUAL_REVIEW_DEPARTMENT = "Manual Review Desk"


def route(category: str, priority: str) -> dict:
    """Return the department and SLA hours for a category and priority."""
    if category not in DEPARTMENT_BY_CATEGORY:
        raise ValueError(f"Unknown category: {category!r}")
    if priority not in SLA_HOURS_BY_PRIORITY:
        raise ValueError(f"Unknown priority: {priority!r}")

    return {
        "department": DEPARTMENT_BY_CATEGORY[category],
        "sla_hours": SLA_HOURS_BY_PRIORITY[priority],
    }
