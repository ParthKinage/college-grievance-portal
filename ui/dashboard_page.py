"""Admin page: live overview of all tickets with KPIs, charts, filters and status updates."""

import streamlit as st

from src import database
from src.demo_data import create_demo_tickets
from ui.common import counts_chart, page_header, tickets_dataframe


def load_demo_button(key):
    """Button that fills the database with demo tickets, then reloads the page."""
    if st.button("Load demo tickets", icon=":material/dataset:", key=key):
        with st.spinner("Running demo grievances through the AI..."):
            count = create_demo_tickets()
        st.toast(f"Added {count} demo tickets", icon=":material/check_circle:")
        st.rerun()


def render():
    """Draw the Dashboard page."""
    page_header("Dashboard", "Live overview of every student grievance and its deadline.")
    df = tickets_dataframe()

    if df.empty:
        with st.container(border=True):
            st.markdown("#### :material/inbox: No tickets yet")
            st.write("Submit a grievance, or load sample tickets to see the dashboard in action.")
            load_demo_button("demo_empty")
        return

    # --- KPI tiles ---------------------------------------------------------
    active = df[df["status"] != "Resolved"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total tickets", len(df), border=True)
    k2.metric("Open", int((df["status"] == "Open").sum()),
              help="Not yet picked up by an office", border=True)
    k3.metric("High priority (active)", int((active["priority"] == "High").sum()), border=True)
    k4.metric("Overdue", int(df["overdue"].sum()),
              help="Past the response deadline and not resolved", border=True)

    # --- Charts --------------------------------------------------------------
    stats = database.get_stats()
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Tickets per category**")
        st.altair_chart(counts_chart(stats["by_category"]), width="stretch")
    with c2, st.container(border=True):
        st.markdown("**Tickets per office**")
        st.altair_chart(counts_chart(stats["by_department"]), width="stretch")

    # --- Ticket table with filters ------------------------------------------
    st.subheader("All tickets")
    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    search = f1.text_input("Search", placeholder="Name or words in the grievance",
                           icon=":material/search:")
    offices = f2.multiselect("Office", sorted(df["department"].unique()))
    priorities = f3.multiselect("Priority", ["High", "Medium", "Low"])
    statuses = f4.multiselect("Status", database.VALID_STATUSES)

    filtered = df
    if search:
        text_match = (filtered["complaint_text"].str.contains(search, case=False, regex=False)
                      | filtered["student_name"].str.contains(search, case=False, regex=False))
        filtered = filtered[text_match]
    if offices:  # an empty selection means "show everything"
        filtered = filtered[filtered["department"].isin(offices)]
    if priorities:
        filtered = filtered[filtered["priority"].isin(priorities)]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]

    st.caption(":material/edit: Click a ticket's **Status** cell to change it to Open, "
               "In Progress or Resolved. The change is saved straight away.")

    # The editor remembers edits by row position, so it gets a fresh key whenever the
    # filters change; otherwise an old edit could land on a different ticket.
    editor_key = f"tickets_{st.session_state.get('editor_round', 0)}_{search}_{offices}_{priorities}_{statuses}"
    edited = st.data_editor(
        filtered,
        key=editor_key,
        hide_index=True,
        width="stretch",
        column_order=["id", "student_name", "category", "priority", "department", "status",
                      "overdue", "due_at", "confidence", "created_at", "complaint_text"],
        disabled=[c for c in filtered.columns if c != "status"],   # only Status is editable
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "student_name": "Student",
            "category": "Category",
            "priority": "Priority",
            "department": "Office",
            "status": st.column_config.SelectboxColumn(
                "Status ✎", options=database.VALID_STATUSES, required=True),
            "overdue": st.column_config.CheckboxColumn("Overdue", width="small"),
            "due_at": st.column_config.DatetimeColumn("Due by", format="DD MMM, HH:mm"),
            "confidence": st.column_config.ProgressColumn(
                "AI confidence", min_value=0.0, max_value=1.0, format="percent"),
            "created_at": st.column_config.DatetimeColumn("Submitted", format="DD MMM, HH:mm"),
            "complaint_text": st.column_config.TextColumn("Grievance", width="large"),
        },
    )

    # Save every row whose status was changed in the table
    changed = edited[edited["status"] != filtered["status"]]
    if not changed.empty:
        for row in changed.itertuples():
            database.update_status(int(row.id), row.status)
        names = ", ".join(f"#{row.id} → {row.status}" for row in changed.itertuples())
        st.toast(f"Saved: {names}", icon=":material/check_circle:")
        # New editor key, so the saved edits are not applied again on the next run
        st.session_state["editor_round"] = st.session_state.get("editor_round", 0) + 1
        st.rerun()  # reload so the KPIs, charts and overdue flags use the new status

    left, right = st.columns([3, 1])
    left.caption(f"Showing {len(filtered)} of {len(df)} tickets.")
    right.download_button("Download CSV", filtered.to_csv(index=False), "tickets.csv",
                          "text/csv", icon=":material/download:", width="stretch")
