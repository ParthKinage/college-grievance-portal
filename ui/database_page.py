"""Admin page: look inside the SQLite database - data, schema, a read-only SQL console and tools."""

import sqlite3

import pandas as pd
import streamlit as st

from src import database
from ui.common import page_header
from ui.dashboard_page import load_demo_button

# Plain-English meaning of every column, shown on the Schema tab
COLUMN_DESCRIPTIONS = {
    "id": "Ticket number, given automatically (1, 2, 3...)",
    "student_name": "Name typed by the student, or 'Anonymous'",
    "complaint_text": "The grievance exactly as the student wrote it",
    "category": "Category predicted by the AI model",
    "confidence": "How sure the model was (0 to 1)",
    "priority": "High / Medium / Low (ML model + safety keyword rules)",
    "department": "College office the ticket was sent to",
    "sla_hours": "Hours allowed for a response (4, 24 or 72)",
    "status": "Open, In Progress or Resolved",
    "created_at": "Date and time the grievance was submitted",
}

EXAMPLE_QUERIES = {
    "All tickets": "SELECT * FROM tickets ORDER BY id DESC",
    "Per category": "SELECT category, COUNT(*) AS tickets\nFROM tickets\nGROUP BY category\n"
                    "ORDER BY tickets DESC",
    "High priority open": "SELECT id, student_name, category, department, created_at\n"
                          "FROM tickets\nWHERE priority = 'High' AND status != 'Resolved'",
    "Average confidence": "SELECT category, ROUND(AVG(confidence) * 100, 1) AS avg_confidence_pct\n"
                          "FROM tickets\nGROUP BY category",
    "Manual review": "SELECT id, complaint_text, confidence\nFROM tickets\n"
                     "WHERE department = 'Manual Review Desk'",
}


def fill_query():
    """Callback for the example query chips: copy the chosen query into the editor."""
    choice = st.session_state.get("query_choice")
    if choice:
        st.session_state["sql_query"] = EXAMPLE_QUERIES[choice]


def browse_tab():
    """Raw table exactly as stored, plus CSV and .db downloads."""
    rows = database.get_all_tickets()
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("This is the raw `tickets` table exactly as it is stored in the file.")

    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download as CSV", pd.DataFrame(rows).to_csv(index=False),
                       "tickets.csv", "text/csv", icon=":material/table_view:", width="stretch")
    with open(database.DB_PATH, "rb") as f:
        c2.download_button("Download .db file", f.read(), "complaints.db",
                           "application/x-sqlite3", icon=":material/download:", width="stretch")
    st.info("Open the downloaded **complaints.db** in the free app "
            "[DB Browser for SQLite](https://sqlitebrowser.org) to explore it outside this portal.",
            icon=":material/lightbulb:")


def schema_tab():
    """Columns of the tickets table with types and meanings, plus the SQL that creates it."""
    schema = pd.DataFrame(database.get_schema())
    schema["meaning"] = schema["column"].map(COLUMN_DESCRIPTIONS)
    st.dataframe(schema, hide_index=True, width="stretch",
                 column_config={"primary_key": st.column_config.CheckboxColumn("Primary key")})
    st.markdown("**SQL used to create the table** (`src/database.py`)")
    st.code(database.CREATE_TABLE_SQL.strip(), language="sql")


def sql_tab():
    """Read-only SQL console: run SELECT queries safely."""
    st.caption(":material/shield: This console is **read-only**. SELECT queries work; "
               "INSERT, UPDATE, DELETE and DROP are blocked by the database itself.")
    st.pills("Example queries", list(EXAMPLE_QUERIES), key="query_choice", on_change=fill_query)
    if "sql_query" not in st.session_state:
        st.session_state["sql_query"] = EXAMPLE_QUERIES["Per category"]
    sql = st.text_area("SQL query", key="sql_query", height=130)

    if st.button("Run query", type="primary", icon=":material/play_arrow:"):
        try:
            columns, rows = database.run_readonly_query(sql)
        except (sqlite3.Error, sqlite3.Warning) as error:
            st.error(f"Query failed: {error}", icon=":material/error:")
            return
        if not columns:
            st.info("The query ran but returned no columns.")
            return
        st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True, width="stretch")
        st.caption(f"{len(rows)} row(s) returned"
                   + (f" (limited to {database.MAX_QUERY_ROWS})"
                      if len(rows) == database.MAX_QUERY_ROWS else ""))


def manage_tab():
    """Tools: add demo tickets, or delete everything (with confirmation)."""
    with st.container(border=True):
        st.markdown("**:material/dataset: Demo data**")
        st.write("Adds 16 realistic grievances. Each one goes through the real AI pipeline.")
        load_demo_button("demo_manage")

    with st.container(border=True):
        st.markdown("**:material/delete_forever: Delete all tickets**")
        st.write("Empties the tickets table and restarts ticket numbers from 1. This cannot be undone.")
        confirm = st.checkbox("I understand that all tickets will be permanently deleted")
        if st.button("Delete all tickets", disabled=not confirm, icon=":material/delete:"):
            deleted = database.delete_all_tickets()
            st.toast(f"Deleted {deleted} tickets", icon=":material/delete:")
            st.rerun()


def render():
    """Draw the Database page."""
    page_header("Database", "Look inside the SQLite database that stores every ticket.")
    info = database.get_db_info()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ticket rows", info["ticket_rows"], border=True)
    k2.metric("Columns", len(database.get_schema()), border=True)
    k3.metric("Tables", ", ".join(info["tables"]), border=True)
    k4.metric("File size", f"{info['size_kb']} KB", border=True)
    st.caption(f":material/database: File: `{info['file_name']}` in the project folder · "
               f"Engine: SQLite {info['sqlite_version']} (built into Python, no server needed)")

    tabs = st.tabs([":material/table: Browse data", ":material/schema: Schema",
                    ":material/terminal: SQL console", ":material/settings: Manage"])
    with tabs[0]:
        browse_tab()
    with tabs[1]:
        schema_tab()
    with tabs[2]:
        sql_tab()
    with tabs[3]:
        manage_tab()
