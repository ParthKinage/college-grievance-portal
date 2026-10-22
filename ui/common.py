"""Shared helpers used by every page: paths, page headers, badges, charts and ticket data."""

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from src import database
from src.clock import now

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

PRIMARY_COLOUR = "#4F46E5"  # same indigo as the theme in .streamlit/config.toml

# Streamlit's named colours for badges and coloured text
PRIORITY_COLOURS = {"High": "red", "Medium": "orange", "Low": "green"}
STATUS_COLOURS = {"Open": "blue", "In Progress": "orange", "Resolved": "green"}


def page_header(title, subtitle):
    """Show a page title with a short grey description under it."""
    st.title(title)
    st.caption(subtitle)


def badge(text, colour):
    """Return markdown for a small coloured label, e.g. badge("High", "red")."""
    return f":{colour}-badge[{text}]"


def priority_badge(priority):
    """Markdown badge for a priority (red High, orange Medium, green Low)."""
    return badge(f"{priority} priority", PRIORITY_COLOURS.get(priority, "gray"))


def bar_chart(data, label, value, value_title, percent=False, max_value=None, counts=False):
    """Horizontal bar chart from a DataFrame with a text column and a number column."""
    scale = alt.Scale(domain=[0, max_value]) if max_value else alt.Undefined
    if percent:
        axis = alt.Axis(format="%")
    elif counts:
        axis = alt.Axis(tickMinStep=1, format="d")  # ticket counts: whole numbers only
    else:
        axis = alt.Axis()
    return alt.Chart(data).mark_bar(color=PRIMARY_COLOUR, cornerRadiusEnd=3).encode(
        x=alt.X(f"{value}:Q", title=value_title, scale=scale, axis=axis),
        y=alt.Y(f"{label}:N", title=None, sort="-x",
                axis=alt.Axis(labelLimit=220, labelOverlap=False)),  # never hide a label
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=".1%" if percent else ("d" if counts else ".3f"))],
    ).properties(height=50 + 34 * len(data))  # about 34px per bar, so labels never overlap


def counts_chart(counts, value_title="Tickets"):
    """Bar chart from a dict like {"IT & Wi-Fi": 3, "Hostel & Mess": 5}."""
    data = pd.DataFrame({"name": list(counts), "count": list(counts.values())})
    return bar_chart(data, "name", "count", value_title, counts=True)


def load_json(path):
    """Read a JSON file, or return None if it does not exist yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tickets_dataframe():
    """All tickets as a DataFrame with extra columns: due_at and overdue."""
    columns = ["id", "student_name", "complaint_text", "category", "confidence", "priority",
               "department", "sla_hours", "status", "created_at"]
    df = pd.DataFrame(database.get_all_tickets(), columns=columns)
    df["created_at"] = pd.to_datetime(df["created_at"])
    # Deadline = time submitted + SLA hours. Overdue = past the deadline and not resolved.
    df["due_at"] = df["created_at"] + pd.to_timedelta(df["sla_hours"], unit="h")
    df["overdue"] = (df["status"] != "Resolved") & (df["due_at"] < pd.Timestamp(now()))
    return df
