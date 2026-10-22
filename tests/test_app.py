"""Smoke tests for the Streamlit pages, using Streamlit's built-in AppTest runner."""

import pytest
from streamlit.testing.v1 import AppTest

from src import database
from src.demo_data import DEMO_GRIEVANCES, create_demo_tickets


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Use a temporary database so tests never touch the real complaints.db."""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))


def run_page(module_name):
    """Run one page's render() function as a mini Streamlit app and return the AppTest."""
    def page_script(module_name):
        """Mini app script: import the page module and draw it."""
        import importlib
        importlib.import_module(module_name).render()

    at = AppTest.from_function(page_script, kwargs={"module_name": module_name}, default_timeout=60)
    return at.run()


@pytest.mark.parametrize("module_name", [
    "ui.submit_page", "ui.dashboard_page", "ui.database_page", "ui.models_page",
])
@pytest.mark.parametrize("with_data", [False, True])
def test_every_page_renders_without_errors(module_name, with_data):
    """Each page loads with an empty database and with demo tickets."""
    if with_data:
        create_demo_tickets()
    at = run_page(module_name)
    assert not at.exception, at.exception


def test_full_app_loads():
    """app.py itself (logo, navigation, default page) runs without errors."""
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception, at.exception


def test_submitting_a_grievance_creates_a_ticket():
    """Typing a grievance and pressing Submit saves a routed ticket."""
    at = run_page("ui.submit_page")
    at.text_area(key="complaint_text").input("My hall ticket is not generated and the exam is tomorrow")
    at.button[0].click().run()
    assert not at.exception, at.exception
    tickets = database.get_all_tickets()
    assert len(tickets) == 1
    assert tickets[0]["category"] == "Exams & Results"
    assert tickets[0]["student_name"] == "Anonymous"


def test_demo_tickets_are_created():
    """Demo data adds every sample grievance with a valid category."""
    assert create_demo_tickets() == len(DEMO_GRIEVANCES)
    stats = database.get_stats()
    assert stats["total"] == len(DEMO_GRIEVANCES)
    assert "Manual Review Desk" in stats["by_department"]  # the vague sample goes to review
