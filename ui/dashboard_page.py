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
    resolved = df[df["status"] == "Resolved"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total tickets", len(df), border=True)
    k2.metric("Open", int((df["status"] == "Open").sum()),
              help="Not yet picked up by an office", border=True)
    k3.metric("High priority (active)", int((active["priority"] == "High").sum()), border=True)
    k4.metric("Overdue", int(df["overdue"].sum()),
              help="Past the response deadline and not resolved", border=True)
    k5.metric("Resolved", len(resolved), border=True)

    # --- Charts --------------------------------------------------------------
    stats = database.get_stats()
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Tickets per category**")
        st.altair_chart(counts_chart(stats["by_category"]), width="stretch")
    with c2, st.container(border=True):
        st.markdown("**Tickets per office**")
        st.altair_chart(counts_chart(stats["by_department"]), width="stretch")

    # --- Filters (apply to both sections) ------------------------------------
    st.subheader("Tickets")
    f1, f2, f3 = st.columns([2, 2, 1])
    search = f1.text_input("Search", placeholder="Name or words in the grievance",
                           icon=":material/search:")
    offices = f2.multiselect("Office", sorted(df["department"].unique()))
    priorities = f3.multiselect("Priority", ["High", "Medium", "Low"])
    filtered = apply_filters(df, search, offices, priorities)
    filter_sig = f"{search}_{offices}_{priorities}"

    # --- Two sections: work still to do, and work that is finished ------------
    tab_active, tab_resolved = st.tabs([":material/pending_actions: Active tickets",
                                        ":material/task_alt: Resolved tickets"])
    with tab_active:
        rows = filtered[filtered["status"] != "Resolved"]
        # Most urgent first: overdue tickets, then the nearest deadline
        rows = rows.sort_values(["overdue", "due_at"], ascending=[False, True])
        st.caption(f"{len(rows)} open or in-progress tickets, most urgent first. "
                   "Set a ticket's **Status** to Resolved to move it to the Resolved section.")
        ticket_table(rows, "active", filter_sig, show_overdue=True)
    with tab_resolved:
        rows = filtered[filtered["status"] == "Resolved"]
        st.caption(f"{len(rows)} resolved tickets, newest first. "
                   "Change a **Status** back to Open or In Progress to reopen a ticket.")
        ticket_table(rows, "resolved", filter_sig, show_overdue=False)


def apply_filters(df, search, offices, priorities):
    """Keep only the tickets that match the search box and the selected filters."""
    if search:
        text_match = (df["complaint_text"].str.contains(search, case=False, regex=False)
                      | df["student_name"].str.contains(search, case=False, regex=False))
        df = df[text_match]
    if offices:  # an empty selection means "show everything"
        df = df[df["department"].isin(offices)]
    if priorities:
        df = df[df["priority"].isin(priorities)]
    return df


def ticket_table(rows, section, filter_sig, show_overdue):
    """Editable ticket table: only the Status column can change, and changes save at once."""
    if rows.empty:
        st.info("No tickets here yet." if not filter_sig.strip("_[]")
                else "No tickets match these filters.", icon=":material/inbox:")
        return

    columns = ["id", "student_name", "category", "priority", "department", "status",
               "overdue", "due_at", "confidence", "created_at", "complaint_text"]
    if not show_overdue:
        columns.remove("overdue")   # a resolved ticket can never be overdue

    # The editor remembers edits by row position, so it gets a fresh key whenever the
    # rows change (filters or a save); otherwise an old edit could land on another ticket.
    key = f"{section}_{st.session_state.get('editor_round', 0)}_{filter_sig}"
    edited = st.data_editor(
        rows,
        key=key,
        hide_index=True,
        width="stretch",
        column_order=columns,
        disabled=[c for c in rows.columns if c != "status"],   # only Status is editable
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
    changed = edited[edited["status"] != rows["status"]]
    if not changed.empty:
        for row in changed.itertuples():
            database.update_status(int(row.id), row.status)
        names = ", ".join(f"#{row.id} → {row.status}" for row in changed.itertuples())
        st.toast(f"Saved: {names}", icon=":material/check_circle:")
        # New editor keys, so the saved edits are not applied again on the next run
        st.session_state["editor_round"] = st.session_state.get("editor_round", 0) + 1
        st.rerun()  # reload so tickets move section and the KPIs and charts update

    st.download_button(f"Download {section} tickets (CSV)", rows.to_csv(index=False),
                       f"{section}_tickets.csv", "text/csv", icon=":material/download:",
                       key=f"download_{section}")
