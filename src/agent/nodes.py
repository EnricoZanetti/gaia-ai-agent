"""
Node functions for the morning report LangGraph graph.

Each function maps directly to one node in the graph and follows the same
contract: it receives the current MorningReportState and returns a *partial*
dict that LangGraph merges back into the state via the field reducers.

Node responsibilities
---------------------
fetch_calendar_node  — pulls today's Google Calendar events
fetch_emails_node    — pulls recent unread Gmail messages
fetch_news_node      — pulls top headlines by topic/location via NewsAPI
generate_report_node — synthesises all data into an HTML report (LLM-powered)
send_email_node      — delivers the report to the configured recipient
"""

from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm import get_llm
from src.agent.state import CalendarEvent, EmailItem, MorningReportState, NewsItem
from src.tools.calendar import get_todays_events
from src.tools.email import get_recent_emails, send_email
from src.tools.news import get_news_for_topics


# ---------------------------------------------------------------------------
# Parallel fetch nodes
# ---------------------------------------------------------------------------


def fetch_calendar_node(state: MorningReportState) -> dict:
    """Fetch today's Google Calendar events and add them to state."""
    try:
        events: list[CalendarEvent] = [
            {
                "title": e["title"],
                "start": e["start"],
                "end": e["end"],
                "location": e.get("location"),
                "description": e.get("description"),
            }
            for e in get_todays_events()
        ]
        return {"calendar_events": events}
    except Exception as exc:
        return {"calendar_events": [], "errors": [f"Calendar fetch failed: {exc}"]}


def fetch_emails_node(state: MorningReportState) -> dict:
    """Fetch recent unread Gmail messages and add them to state."""
    try:
        items: list[EmailItem] = [
            {
                "sender": e["sender"],
                "subject": e["subject"],
                "snippet": e["snippet"],
                "received_at": e["received_at"],
            }
            for e in get_recent_emails(max_results=10)
        ]
        return {"email_items": items}
    except Exception as exc:
        return {"email_items": [], "errors": [f"Email fetch failed: {exc}"]}


def fetch_news_node(state: MorningReportState) -> dict:
    """Fetch top news articles for the user's topics and location."""
    try:
        items: list[NewsItem] = [
            {
                "title": a["title"],
                "source": a["source"],
                "url": a["url"],
                "description": a["description"],
                "published_at": a["published_at"],
            }
            for a in get_news_for_topics(
                topics=state["news_topics"],
                country=state["location"],
            )
        ]
        return {"news_items": items}
    except Exception as exc:
        return {"news_items": [], "errors": [f"News fetch failed: {exc}"]}


# ---------------------------------------------------------------------------
# Report generation node  (runs after all three fetch nodes complete)
# ---------------------------------------------------------------------------


def generate_report_node(state: MorningReportState) -> dict:
    """
    Synthesise calendar, email, and news data into an HTML morning report.

    Builds a structured context string from the fetched data and passes it
    to Claude, which formats it into a clean, scannable HTML email body.
    """
    today = datetime.now().strftime("%A, %B %d, %Y")

    # --- Build a plain-text context block for the LLM prompt ---
    sections: list[str] = [
        f"Today is {today}.",
        f"Preparing a morning briefing for {state['user_name']}.",
        "",
    ]

    # Calendar
    sections.append("## Today's Schedule")
    if state["calendar_events"]:
        for event in state["calendar_events"]:
            line = f"- {event['start']} — {event['title']}"
            if event.get("location"):
                line += f" @ {event['location']}"
            sections.append(line)
    else:
        sections.append("No events scheduled for today.")

    # Email
    sections.append("\n## Recent Unread Emails")
    if state["email_items"]:
        for item in state["email_items"]:
            sections.append(f"- From: {item['sender']} | Subject: {item['subject']}")
            if item["snippet"]:
                sections.append(f"  Preview: {item['snippet'][:200]}")
    else:
        sections.append("No unread emails in the last 24 hours.")

    # News
    topic_label = ", ".join(state["news_topics"])
    sections.append(f"\n## Top News — {topic_label} ({state['location'].upper()})")
    if state["news_items"]:
        for item in state["news_items"]:
            sections.append(f"- [{item['source']}] {item['title']}")
            if item.get("description"):
                sections.append(f"  {item['description'][:250]}")
    else:
        sections.append("No news articles fetched.")

    # Surface any errors so Claude can mention them in the report
    if state.get("errors"):
        sections.append("\n## Data Warnings")
        for err in state["errors"]:
            sections.append(f"- {err}")

    context = "\n".join(sections)

    # --- Call the configured LLM to write the polished email body ---
    llm = get_llm(temperature=0.3)

    messages = [
        SystemMessage(
            content=(
                "You are a personal assistant writing a morning briefing email. "
                "Using the structured data provided, produce a concise, friendly, and "
                "well-formatted HTML email body. Use <h2> headings for each section, "
                "<ul>/<li> for bullet points, and <strong> for key items. "
                "Keep the tone warm but professional. Aim for under 500 words. "
                "Do NOT wrap output in <html>, <head>, or <body> tags — "
                "return only the inner content that goes inside <body>."
            )
        ),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    return {"report_markdown": response.content}


# ---------------------------------------------------------------------------
# Email delivery node
# ---------------------------------------------------------------------------


def send_email_node(state: MorningReportState) -> dict:
    """Send the generated report to the configured recipient."""
    if not state.get("report_markdown"):
        return {"email_sent": False, "errors": ["Report generation produced no content."]}

    today = datetime.now().strftime("%B %d, %Y")
    subject = f"☀️ Morning Briefing — {today}"

    try:
        send_email(
            to=state["recipient_email"],
            subject=subject,
            body_html=state["report_markdown"],
        )
        return {"email_sent": True}
    except Exception as exc:
        return {"email_sent": False, "errors": [f"Email send failed: {exc}"]}
