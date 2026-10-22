"""Tests for SQLite ticket storage (uses a temporary database file)."""

import pytest

from src import database


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the database at a fresh temp file so tests never touch complaints.db."""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))


def add_sample(priority="Medium"):
    """Insert a sample ticket and return its ID."""
    return database.add_ticket("Asha", "paid fees twice", "Fees & Scholarships",
                               0.91, priority, "Accounts Office", 24)


def test_add_and_get_ticket():
    """A new ticket is stored with status Open and a timestamp."""
    ticket_id = add_sample()
    tickets = database.get_all_tickets()
    assert len(tickets) == 1
    assert tickets[0]["id"] == ticket_id
    assert tickets[0]["student_name"] == "Asha"
    assert tickets[0]["status"] == "Open"
    assert tickets[0]["created_at"]


def test_update_status():
    """Status can be changed; an unknown ID returns False; a bad status raises."""
    ticket_id = add_sample()
    assert database.update_status(ticket_id, "Resolved") is True
    assert database.get_all_tickets()[0]["status"] == "Resolved"
    assert database.update_status(9999, "Open") is False
    with pytest.raises(ValueError):
        database.update_status(ticket_id, "Deleted")


def test_get_stats():
    """Stats count totals, open tickets, high priority and per-group counts."""
    add_sample("High")
    second = add_sample("Low")
    database.update_status(second, "In Progress")
    stats = database.get_stats()
    assert stats["total"] == 2
    assert stats["open"] == 1
    assert stats["high_priority"] == 1
    assert stats["by_category"] == {"Fees & Scholarships": 2}
    assert stats["by_department"] == {"Accounts Office": 2}


def test_text_with_quotes_is_stored_safely():
    """Parameterized queries store quotes and SQL-like text as plain text."""
    nasty = "Robert'); DROP TABLE tickets;--"
    database.add_ticket(nasty, nasty, "Fees & Scholarships", 0.5, "Low", "Accounts Office", 72)
    assert database.get_all_tickets()[0]["student_name"] == nasty


def test_readonly_query_reads_data():
    """SELECT queries return column names and rows."""
    add_sample("High")
    columns, rows = database.run_readonly_query(
        "SELECT priority, COUNT(*) AS n FROM tickets GROUP BY priority")
    assert columns == ["priority", "n"]
    assert rows == [("High", 1)]


@pytest.mark.parametrize("sql", [
    "DELETE FROM tickets",
    "DROP TABLE tickets",
    "UPDATE tickets SET status = 'Resolved'",
    "INSERT INTO tickets (student_name) VALUES ('x')",
])
def test_readonly_query_cannot_change_data(sql):
    """The SQL console connection is read-only: any write raises an error."""
    add_sample()
    with pytest.raises(Exception):
        database.run_readonly_query(sql)
    assert len(database.get_all_tickets()) == 1  # data is untouched


def test_schema_and_info():
    """The schema lists all 10 columns and the info reports the row count."""
    add_sample()
    columns = [c["column"] for c in database.get_schema()]
    assert columns[0] == "id" and "student_name" in columns and len(columns) == 10
    info = database.get_db_info()
    assert info["ticket_rows"] == 1
    assert "tickets" in info["tables"]


def test_delete_all_tickets_restarts_ids():
    """Deleting all tickets empties the table and new tickets start again at ID 1."""
    add_sample()
    add_sample()
    assert database.delete_all_tickets() == 2
    assert database.get_all_tickets() == []
    assert add_sample() == 1
