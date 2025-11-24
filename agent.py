"""General multi tool agent • OpenAI + LangGraph + custom tools.

Usage:
    python agent.py "What city hosted Expo 2015?"
    python agent.py "Find recent papers about diffusion models."
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from constants import SYSTEM_PROMPT  # optional / your own prompt
except ImportError:
    SYSTEM_PROMPT = (
        "You are a helpful AI assistant with access to several tools:\n\n"
        "- similar_question: retrieve similar Q&A from a local knowledge base.\n"
        "- web_search: search the web for up-to-date information.\n"
        "- wiki_search: search Wikipedia for background knowledge.\n"
        "- arxiv_search: look up scientific papers.\n"
        "- calculator: perform arithmetic calculations.\n\n"
        "Decide when to call tools. Use tools for factual or numeric questions, "
        "and combine information from tools with your own reasoning. "
        "When you answer, be concise but clear, and cite tools you used in natural language."
    )

# local reusable tools -------------------------------------------------------
from tools import (
    arxiv_search,
    calculator,
    similar_question,
    web_search,
    wiki_search,
)

# ---------------------------------------------------------------------------
# 1. env & LLM setup
# ---------------------------------------------------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("[agent] 🔑  Missing OPENAI_API_KEY in environment or .env")

sys_msg = SystemMessage(content=SYSTEM_PROMPT)

TOOLS = [
    similar_question,
    web_search,
    wiki_search,
    calculator,
    arxiv_search,
]

# Bind tools to the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4o" if you prefer
    temperature=0.2,
).bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 2. LangGraph nodes
# ---------------------------------------------------------------------------


def node_assistant(state: MessagesState):
    """Single LLM step (can decide to call tools)."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph ---------------------------------------------------------------
builder = StateGraph(MessagesState)

builder.add_node("assistant", node_assistant)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

agent_graph = builder.compile()

# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------


def solve(question: str):
    """Run the agent and return both the final answer and a list of tool calls."""
    result = agent_graph.invoke(
        {"messages": [sys_msg, HumanMessage(content=question)]},
        config={"recursion_limit": 24},
    )

    messages = result["messages"]

    tool_calls = []
    final_answer = ""

    for msg in messages:
        # Only AI/assistant messages can be the final answer
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            # keep updating; the last such message is the final answer
            final_answer = msg.content

        # Extract tool calls (if any) from messages that have this attribute
        msg_tool_calls = getattr(msg, "tool_calls", None)
        if msg_tool_calls:
            for call in msg_tool_calls:
                tool_calls.append(call["name"])

    # Remove duplicates while preserving order
    seen = set()
    tool_calls_unique = [t for t in tool_calls if not (t in seen or seen.add(t))]

    return final_answer.strip(), tool_calls_unique


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General multi-tool AI agent (LangGraph + OpenAI)."
    )
    parser.add_argument("question", nargs="+", help="Question to ask the agent")
    args = parser.parse_args()

    question = " ".join(args.question)
    answer, tools_used = solve(question)

    print(answer)
    if tools_used:
        print("\n[tools used]", ", ".join(tools_used))
    else:
        print("\n[tools used] none")
