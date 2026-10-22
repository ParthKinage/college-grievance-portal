"""
Grievance ticket storage with SQLite.

SQLite keeps the whole database in one file (complaints.db in the project
root) and is built into Python, so no database server is needed.
All queries use "?" placeholders (parameterized queries), which prevents
SQL injection because user text is never pasted into the SQL string.

The app's Database page can also run SELECT queries typed by an admin.
Those run on a separate READ-ONLY connection, so they can never change data.

To browse the file outside the app, open complaints.db in a free tool such as
"DB Browser for SQLite" (sqlitebrowser.org).
"""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from src.clock import now

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "complaints.db")

VALID_STATUSES = ["Open", "In Progress", "Resolved"]
MAX_QUERY_ROWS = 1000  # the SQL console shows at most this many rows

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name   TEXT,
    complaint_text TEXT,
    category       TEXT,
    confidence     REAL,
    priority       TEXT,
    department     TEXT,
    sla_hours      INTEGER,
    status         TEXT DEFAULT 'Open',
    created_at     TEXT
)
"""


def get_connection():
    """Open a connection whose rows can be read like dictionaries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us use row["category"] instead of row[3]
    return conn


def init_db():
    """Create the tickets table if it does not exist yet."""
    with closing(get_connection()) as conn, conn:  # second "conn" commits on success
        conn.execute(CREATE_TABLE_SQL)


def add_ticket(student_name, complaint_text, category, confidence, priority, department, sla_hours,
               created_at=None):
    """Insert a new ticket and return its ID (created_at defaults to now)."""
    init_db()
    if created_at is None:
        created_at = now().isoformat(timespec="seconds")
    with closing(get_connection()) as conn, conn:
        cursor = conn.execute(
            """INSERT INTO tickets
               (student_name, complaint_text, category, confidence,
                priority, department, sla_hours, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'Open', ?)""",
            (student_name, complaint_text, category, confidence,
             priority, department, sla_hours, created_at),
        )
        return cursor.lastrowid


def get_all_tickets():
    """Return every ticket as a list of dicts, newest first."""
    init_db()
    with closing(get_connection()) as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def update_status(ticket_id, status):
    """Change a ticket's status; return True if a ticket was updated."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Status must be one of {VALID_STATUSES}")
    init_db()
    with closing(get_connection()) as conn, conn:
        cursor = conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
        return cursor.rowcount > 0


def get_stats():
    """Return ticket counts: total, open, high priority, per category, per department."""
    init_db()
    with closing(get_connection()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        open_count = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE status = ?", ("Open",)
        ).fetchone()[0]
        high = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE priority = ?", ("High",)
        ).fetchone()[0]
        by_category = conn.execute(
            "SELECT category, COUNT(*) FROM tickets GROUP BY category"
        ).fetchall()
        by_department = conn.execute(
            "SELECT department, COUNT(*) FROM tickets GROUP BY department"
        ).fetchall()

    return {
        "total": total,
        "open": open_count,
        "high_priority": high,
        "by_category": {row[0]: row[1] for row in by_category},
        "by_department": {row[0]: row[1] for row in by_department},
    }


def delete_all_tickets():
    """Delete every ticket (used by the admin 'reset' button); return how many were deleted."""
    init_db()
    with closing(get_connection()) as conn, conn:
        deleted = conn.execute("DELETE FROM tickets").rowcount
        # Restart ticket numbers from 1 (SQLite remembers the last AUTOINCREMENT id here)
        conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", ("tickets",))
    return deleted


def get_db_info():
    """Return basic facts about the database file: path, size, tables and row count."""
    init_db()
    with closing(get_connection()) as conn:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )]
        rows = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    return {
        "file_name": os.path.basename(DB_PATH),
        "size_kb": round(os.path.getsize(DB_PATH) / 1024, 1),
        "tables": tables,
        "ticket_rows": rows,
        "sqlite_version": sqlite3.sqlite_version,
    }


def get_schema():
    """Return the columns of the tickets table: name, type, default value and key flag."""
    init_db()
    with closing(get_connection()) as conn:
        rows = conn.execute("PRAGMA table_info(tickets)").fetchall()
    return [
        {"column": row["name"], "type": row["type"], "default": row["dflt_value"],
         "primary_key": bool(row["pk"])}
        for row in rows
    ]


def run_readonly_query(sql):
    """Run one SQL query on a READ-ONLY connection; return (column_names, rows).

    Opening the file with mode=ro means INSERT, UPDATE, DELETE and DROP fail
    with an error, so the SQL console can look at data but never change it.
    """
    init_db()
    uri = Path(DB_PATH).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        cursor = conn.execute(sql)  # execute() also refuses more than one statement
        if cursor.description is None:
            return [], []
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchmany(MAX_QUERY_ROWS)
    return columns, rows
