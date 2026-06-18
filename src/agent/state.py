"""
LangGraph state definition for the morning report agent.

MorningReportState is the single shared object that flows through every node.
Fields that are written by the three parallel fetch nodes use operator.add as
their reducer so that each node's output is *appended* to the list rather than
overwriting it — this is how LangGraph merges results from parallel branches.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Typed sub-records  (one per data source)
# ---------------------------------------------------------------------------


class CalendarEvent(TypedDict):
    title: str
    start: str        # ISO-8601 datetime or date string
    end: str
    location: str | None
    description: str | None


class EmailItem(TypedDict):
    sender: str
    subject: str
    snippet: str      # short preview; full body is not downloaded
    received_at: str  # RFC 2822 date string


class NewsItem(TypedDict):
    title: str
    source: str
    url: str
    description: str
    published_at: str  # ISO-8601 datetime string


# ---------------------------------------------------------------------------
# Top-level graph state
# ---------------------------------------------------------------------------


class MorningReportState(TypedDict):
    """
    Shared state that flows through the morning report graph.

    Annotated list fields use ``operator.add`` as a reducer so that parallel
    fetch nodes can each append their results without racing or overwriting
    one another.  Scalar fields (report_markdown, email_sent) are written by
    exactly one node so no reducer is needed.
    """

    # Input — populated once at graph entry from UserConfig
    user_name: str
    recipient_email: str
    news_topics: list[str]
    location: str

    # Data collected by the parallel fetch nodes (reducer: append)
    calendar_events: Annotated[list[CalendarEvent], operator.add]
    email_items: Annotated[list[EmailItem], operator.add]
    news_items: Annotated[list[NewsItem], operator.add]

    # Errors from any node, accumulated across the whole run (reducer: append)
    errors: Annotated[list[str], operator.add]

    # Output produced by the generate_report and send_email nodes
    report_markdown: str
    email_sent: bool
