"""Student page: submit a grievance and see where the AI sends it, and why."""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src import database
from src.explain import explain_prediction
from src.predict import analyze_complaint, load_models
from ui.common import badge, bar_chart, page_header, priority_badge

MIN_LENGTH = 10  # grievances shorter than this are rejected

EXAMPLES = {
    "Mess food": "The mess food was stale yesterday and many students in our hostel fell sick.",
    "Hall ticket": "My hall ticket is not generated and the exam is tomorrow.",
    "Fees paid twice": "I paid my semester fees twice through the portal and the accounts office "
                       "has not refunded it.",
    "Hostel Wi-Fi": "The wifi in the hostel has not worked for a week and I cannot submit assignments.",
    "Ragging": "Seniors are ragging me in the hostel every night, please help.",
}


def fill_example():
    """Callback for the example chips: copy the chosen example into the text box."""
    choice = st.session_state.get("example_choice")
    if choice:
        st.session_state["complaint_text"] = EXAMPLES[choice]


def show_how_it_works():
    """Placeholder card shown before the first grievance is submitted."""
    with st.container(border=True):
        st.markdown("#### :material/route: What happens next")
        st.markdown(
            "1. **AI reads your grievance** and decides its category.\n"
            "2. **Priority is set**: safety words like *ragging* or *food poisoning* are always High.\n"
            "3. **It is sent to the right office** with a response deadline.\n"
            "4. **You get a ticket number** to follow up with."
        )
        st.caption(":material/lock: Your name is optional. Leave it blank to report anonymously.")


def show_result(ticket_id, text, result, created_at):
    """Result card: category, priority, office, deadline, confidence and explanation."""
    if result["needs_review"]:
        st.warning(
            f"The AI was only {result['confidence']:.0%} sure about this grievance, so it was sent "
            "to the **Manual Review Desk** for a staff member to check.",
            icon=":material/visibility:",
        )

    with st.container(border=True):
        st.markdown(f"#### :material/confirmation_number: Ticket #{ticket_id}")
        st.markdown(f"{badge(result['category'], 'violet')} {priority_badge(result['priority'])} "
                    f"{badge('Open', 'blue')}")

        deadline = created_at + timedelta(hours=result["sla_hours"])
        st.markdown(f":material/apartment: **Sent to:** {result['department']}  \n"
                    f":material/schedule: **Respond within:** {result['sla_hours']} hours "
                    f"(by {deadline:%d %b, %H:%M})")

        st.markdown(f"**AI confidence:** {result['confidence']:.1%}")
        st.progress(result["confidence"])
        st.caption(f"Priority: {result['priority_reason']}")

        # Explainability: which words in the text pushed the model to this category
        models = load_models()
        reasons = explain_prediction(text, result["category"],
                                     models["category_model"], models["vectorizer"])
        if reasons:
            st.markdown("**Why this category?** These words mattered most:")
            st.markdown(" ".join(badge(word, "gray") for word, _ in reasons))

        st.markdown("**Top 3 likely categories**")
        top3 = pd.DataFrame(result["top_3"])
        st.altair_chart(bar_chart(top3, "category", "probability", "Probability",
                                  percent=True, max_value=1), width="stretch")


def render():
    """Draw the Submit Grievance page."""
    page_header("Submit a Grievance",
                "Describe your problem in plain English. Our AI will classify it and send it "
                "to the right college office.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.pills("Try an example", list(EXAMPLES), key="example_choice", on_change=fill_example)

        with st.form("complaint_form", border=True):
            # Name is optional so students can report ragging or harassment anonymously
            name = st.text_input("Your name (optional)", placeholder="Leave blank to stay anonymous")
            text = st.text_area("Your grievance", key="complaint_text", height=170,
                                placeholder="e.g. The hostel wifi has not worked for a week...")
            submitted = st.form_submit_button("Submit grievance", type="primary",
                                              icon=":material/send:", width="stretch")

    if submitted:
        text = (text or "").strip()
        if len(text) < MIN_LENGTH:
            left.error(f"Please describe your grievance in a bit more detail "
                       f"(at least {MIN_LENGTH} characters).", icon=":material/error:")
        else:
            try:
                result = analyze_complaint(text)
            except FileNotFoundError as error:
                left.error(str(error))
                return
            ticket_id = database.add_ticket(
                student_name=name.strip() or "Anonymous",
                complaint_text=text,
                category=result["category"],
                confidence=result["confidence"],
                priority=result["priority"],
                department=result["department"],
                sla_hours=result["sla_hours"],
            )
            # Keep the result so it stays on screen after Streamlit reruns the script
            st.session_state["last_ticket"] = (ticket_id, text, result, datetime.now())
            st.toast(f"Ticket #{ticket_id} created", icon=":material/check_circle:")

    with right:
        if "last_ticket" in st.session_state:
            show_result(*st.session_state["last_ticket"])
        else:
            show_how_it_works()
