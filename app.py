"""
College Grievance Portal (Smart Complaint Classification and Routing System).

This file only sets up the app and its navigation. Each page lives in ui/:
    Student         -> ui/submit_page.py    submit a grievance, see where the AI sends it
    Administration  -> ui/dashboard_page.py tickets, KPIs, filters, status updates
                    -> ui/database_page.py  look inside the SQLite database
    AI Models       -> ui/models_page.py    saved models, performance, what they learned

Run from the project root:
    streamlit run app.py
"""

import os

import streamlit as st

from src import database
from ui import dashboard_page, database_page, models_page, submit_page
from ui.common import ASSETS_DIR

st.set_page_config(page_title="College Grievance Portal", page_icon=":material/school:",
                   layout="wide")
database.init_db()  # create complaints.db on first run

st.logo(os.path.join(ASSETS_DIR, "logo.svg"), size="large",
        icon_image=os.path.join(ASSETS_DIR, "icon.svg"))

# Sidebar navigation, grouped by who uses each page
navigation = st.navigation({
    "Student": [
        st.Page(submit_page.render, title="Submit Grievance", icon=":material/edit_note:",
                url_path="submit", default=True),
    ],
    "Administration": [
        st.Page(dashboard_page.render, title="Dashboard", icon=":material/dashboard:",
                url_path="dashboard"),
        st.Page(database_page.render, title="Database", icon=":material/database:",
                url_path="database"),
    ],
    "AI Models": [
        st.Page(models_page.render, title="Model Explorer", icon=":material/model_training:",
                url_path="models"),
    ],
})

st.sidebar.caption("AI mini project · TF-IDF + Linear SVM · Streamlit · SQLite")
navigation.run()
