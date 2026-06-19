"""
LangGraph StateGraph for the morning report agent.

Graph topology - fan-out then fan-in:

                        START
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   fetch_calendar    fetch_emails     fetch_news     ← parallel
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                   generate_report              ← LLM synthesis
                          │
                      send_email               ← Gmail delivery
                          │
                         END

The three fetch nodes run in parallel because they all have an edge from START
and none depend on each other.  LangGraph waits for all three to complete
before advancing to generate_report (fan-in via the operator.add reducers on
the list fields in MorningReportState).
"""

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    fetch_calendar_node,
    fetch_emails_node,
    fetch_news_node,
    generate_report_node,
    send_email_node,
)
from src.agent.state import MorningReportState


def build_graph():
    """
    Construct and compile the morning report StateGraph.

    Returns a compiled LangGraph runnable that accepts an initial
    MorningReportState dict and returns the fully populated final state.
    """
    graph = StateGraph(MorningReportState)

    # Register every node
    graph.add_node("fetch_calendar", fetch_calendar_node)
    graph.add_node("fetch_emails", fetch_emails_node)
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("send_email", send_email_node)

    # Fan-out: all three fetch nodes start simultaneously from START
    graph.add_edge(START, "fetch_calendar")
    graph.add_edge(START, "fetch_emails")
    graph.add_edge(START, "fetch_news")

    # Fan-in: generate_report waits until all three fetch nodes have finished
    graph.add_edge("fetch_calendar", "generate_report")
    graph.add_edge("fetch_emails", "generate_report")
    graph.add_edge("fetch_news", "generate_report")

    # Sequential tail: send the report after it has been generated
    graph.add_edge("generate_report", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()
