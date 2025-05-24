"""Minimal GAIA Level-1 agent

▶ Dependencies (install in *gaia_env*):
   pip install langchain openai langgraph sentence-transformers faiss-cpu requests

▶ Env variables required:
   OPENAI_API_KEY    – your OpenAI key
   GAIA_API_BASE     – base URL of the GAIA evaluation API (e.g. https://gaia.example.com)
   HF_USERNAME       – your Hugging Face username (for leaderboard)
   AGENT_CODE_URL    – public code URL of this Space (…/tree/main)

Run » python agent.py            → manual CLI test
Run » python agent.py --submit   → fetch 20 questions & submit to leaderboard
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from typing import List

import requests
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from constants import SYSTEM_PROMPT
from langchain.vectorstores import FAISS
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# 1. Config & safety checks
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("GAIA_API_BASE")
HF_USERNAME = os.environ.get("HF_USERNAME")
AGENT_CODE_URL = os.environ.get("AGENT_CODE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not all((API_BASE, HF_USERNAME, AGENT_CODE_URL, OPENAI_API_KEY)):
    sys.exit(
        "[agent] 🔑  Please export GAIA_API_BASE, HF_USERNAME, AGENT_CODE_URL, OPENAI_API_KEY"
    )

# ---------------------------------------------------------------------------
# 2. Load solved examples (metadata.jsonl)
# ---------------------------------------------------------------------------

DATA_PATH = pathlib.Path(__file__).with_name("metadata.jsonl")
if not DATA_PATH.exists():
    sys.exit(
        f"[agent] 📄  {DATA_PATH} not found – place metadata.jsonl beside agent.py"
    )

examples: List[dict] = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]

# ---------------------------------------------------------------------------
# 3. Build FAISS vector store & retrieval tool
# ---------------------------------------------------------------------------

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

docs = [
    Document(
        page_content=f"Question: {e['Question']}\nAnswer: {e['Final answer']}",
        metadata={"task_id": e["task_id"]},
    )
    for e in examples
]

vstore = FAISS.from_documents(docs, embedder)
retriever = vstore.as_retriever(search_kwargs={"k": 3})

similar_q_tool = create_retriever_tool(
    retriever,
    name="similar_questions",
    description="Return up to three previously‑solved GAIA Level‑1 questions similar to the given query.",
)

TOOLS = [similar_q_tool]

# ---------------------------------------------------------------------------
# 4. Build the system prompt (few‑shot examples)
# ---------------------------------------------------------------------------

few_shots = random.sample(examples, k=3)

for ex in few_shots:
    SYSTEM_PROMPT += f"\nQ: {ex['Question']}\nA: {ex['Final answer']}"

# ---------------------------------------------------------------------------
# 5. Initialise OpenAI chat model & bind tools
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False).bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# 6. LangGraph: assistant node + tool loop
# ---------------------------------------------------------------------------


def _assistant(state: MessagesState):  # noqa: D401
    """Run the LLM once."""
    return {
        "messages": [
            llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        ]
    }


builder = StateGraph(MessagesState)
builder.add_node("assistant", _assistant)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
agent_graph = builder.compile()

# ---------------------------------------------------------------------------
# 7. Public API: solve(question) & evaluate()
# ---------------------------------------------------------------------------


def solve(question: str) -> str:
    """Return the agent's final answer for a single GAIA question."""
    messages = [HumanMessage(content=question)]
    output = agent_graph.invoke({"messages": messages})
    return output["messages"][-1].content.strip()


def evaluate() -> None:
    """Retrieve 20 eval questions, answer, submit, and print score."""
    qs = requests.get(f"{API_BASE}/questions", timeout=30).json()
    answers = [
        {"task_id": q["id"], "submitted_answer": solve(q["question"])} for q in qs
    ]

    payload = {
        "username": HF_USERNAME,
        "agent_code": AGENT_CODE_URL,
        "answers": answers,
    }
    res = requests.post(f"{API_BASE}/submit", json=payload, timeout=60)
    print("[agent] 🏆  Leaderboard response:", res.status_code, res.text)


# ---------------------------------------------------------------------------
# 8. CLI entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAIA agent")
    parser.add_argument("question", nargs="*", help="Manual question to solve")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Run full evaluation & submit to leaderboard",
    )
    args = parser.parse_args()

    if args.submit:
        evaluate()
    elif args.question:
        q = " ".join(args.question)
        print("Answer:", solve(q))
    else:
        parser.print_help()
