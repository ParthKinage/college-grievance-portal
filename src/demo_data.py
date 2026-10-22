"""
Demo tickets for presentations.

A fresh database is empty, which makes the dashboard look bare (and a cloud
deployment starts empty after every restart). This module fills it with
realistic grievances. Each one goes through the real AI pipeline, so the
categories, priorities and offices are genuine model predictions.
"""

from datetime import timedelta

from src import database
from src.clock import now
from src.predict import analyze_complaint

# (student name, grievance text, status, how many hours ago it was submitted)
DEMO_GRIEVANCES = [
    ("Aarav Mehta", "Seniors are ragging me in the hostel every night, please help.", "In Progress", 2),
    ("Anonymous", "A group of seniors threatened me near the canteen and took my money.", "Open", 7),
    ("Priya Nair", "My hall ticket is not generated and the exam is tomorrow.", "Open", 3),
    ("Rohan Gupta", "I paid my semester fees twice through the portal and the accounts office has not refunded it.", "In Progress", 20),
    ("Sneha Patil", "The mess food was stale yesterday and many students in our hostel fell sick.", "Open", 5),
    ("Kabir Singh", "The wifi in the hostel has not worked for a week and I cannot submit assignments.", "Open", 30),
    ("Ananya Rao", "My scholarship amount is pending for three months, please check.", "Open", 50),
    ("Vikram Joshi", "The professor has not completed the syllabus and the exams are near.", "Resolved", 70),
    ("Meera Iyer", "My revaluation result has been pending for two months.", "Open", 40),
    ("Arjun Das", "The college email account was hacked and is sending spam to my contacts.", "Resolved", 26),
    ("Ishita Verma", "There is no hot water in the hostel bathrooms since Monday.", "In Progress", 18),
    ("Anonymous", "A faculty member insulted me in front of the whole class.", "Open", 12),
    ("Neha Kulkarni", "Please add more variety to the mess breakfast menu.", "Resolved", 90),
    ("Siddharth Jain", "The lab computers are not working and our practical exam is next week.", "Open", 9),
    ("Pooja Reddy", "Please allow paying fees in installments.", "Open", 60),
    ("Rahul Sharma", "things are not good here", "Open", 1),
]


def create_demo_tickets():
    """Run every demo grievance through the AI pipeline and save it; return how many were added."""
    current_time = now()
    for name, text, status, hours_ago in DEMO_GRIEVANCES:
        result = analyze_complaint(text)
        created_at = (current_time - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
        ticket_id = database.add_ticket(
            name, text, result["category"], result["confidence"], result["priority"],
            result["department"], result["sla_hours"], created_at=created_at,
        )
        if status != "Open":
            database.update_status(ticket_id, status)
    return len(DEMO_GRIEVANCES)
