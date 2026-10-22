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

    st.dataframe(
        filtered,
        hide_index=True,
        width="stretch",
        column_order=["id", "student_name", "category", "priority", "department", "status",
                      "overdue", "due_at", "confidence", "created_at", "complaint_text"],
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "student_name": "Student",
            "category": "Category",
            "priority": "Priority",
            "department": "Office",
            "status": "Status",
            "overdue": st.column_config.CheckboxColumn("Overdue", width="small"),
            "due_at": st.column_config.DatetimeColumn("Due by", format="DD MMM, HH:mm"),
            "confidence": st.column_config.ProgressColumn(
                "AI confidence", min_value=0.0, max_value=1.0, format="percent"),
            "created_at": st.column_config.DatetimeColumn("Submitted", format="DD MMM, HH:mm"),
            "complaint_text": st.column_config.TextColumn("Grievance", width="large"),
        },
    )
    left, right = st.columns([3, 1])
    left.caption(f"Showing {len(filtered)} of {len(df)} tickets.")
    right.download_button("Download CSV", filtered.to_csv(index=False), "tickets.csv",
                          "text/csv", icon=":material/download:", width="stretch")

    # --- Status update -------------------------------------------------------
    with st.container(border=True):
        st.markdown("**:material/edit: Update a ticket**")
        c1, c2, c3 = st.columns([3, 3, 1], vertical_alignment="bottom")
        labels = {row.id: f"#{row.id} · {row.category} · {row.student_name}"
                  for row in df.itertuples()}
        ticket_id = c1.selectbox("Ticket", list(labels), format_func=labels.get)
        current = df.loc[df["id"] == ticket_id, "status"].iloc[0]
        new_status = c2.segmented_control("New status", database.VALID_STATUSES,
                                          default=current, key=f"status_{ticket_id}")
        if c3.button("Save", type="primary", width="stretch") and new_status:
            database.update_status(int(ticket_id), new_status)
            st.toast(f"Ticket #{ticket_id} set to {new_status}", icon=":material/check_circle:")
            st.rerun()  # reload so the table and KPIs show the new status
