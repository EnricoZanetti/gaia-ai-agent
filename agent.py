# --------------------- agent.py ---------------------
"""GAIA Level‑1 agent • OpenAI + FAISS + retrieval‑priming

`python agent.py "What city hosted Expo 2015?"`  → one‑off test
`python agent.py --submit`                         → answer 20 Qs & POST
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
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# local reusable tools -------------------------------------------------------
from tools import web_search, wiki_search, calculator, arxiv_search
from constants import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# 1. env & constants
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()

API_BASE = os.getenv("GAIA_API_BASE")
SPACE_HOST = os.getenv("SPACE_HOST")
AGENT_CODE_URL = os.getenv("AGENT_CODE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all((API_BASE, SPACE_HOST, AGENT_CODE_URL, OPENAI_API_KEY)):
    sys.exit(
        "[agent] 🔑  Missing GAIA_API_BASE / SPACE_HOST / AGENT_CODE_URL / OPENAI_API_KEY"
    )

# ---------------------------------------------------------------------------
# 2. load metadata examples
# ---------------------------------------------------------------------------
DATA_PATH = pathlib.Path(__file__).with_name("metadata.jsonl")
if not DATA_PATH.exists():
    sys.exit("[agent] 📄 metadata.jsonl missing next to agent.py")
examples: List[dict] = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]

# ---------------------------------------------------------------------------
# 3. FAISS retriever for similar‑question priming
# ---------------------------------------------------------------------------
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

_documents = [
    Document(
        page_content=f"Question: {e['Question']}\nAnswer: {e['Final answer']}",
        metadata={"task_id": e["task_id"]},
    )
    for e in examples
]

vstore = FAISS.from_documents(_documents, embedder)
retriever = vstore.as_retriever(search_kwargs={"k": 3})

similar_q_tool = create_retriever_tool(
    retriever,
    name="similar_questions",
    description="Return up to three previously‑solved GAIA Level‑1 Q&A pairs relevant to the query.",
)

TOOLS = [
    similar_q_tool,
    web_search,
    wiki_search,
    calculator,
    arxiv_search,
]

# ---------------------------------------------------------------------------
# 4. system prompt (few‑shot)
# ---------------------------------------------------------------------------
few_shots = random.sample(examples, k=3)
for ex in few_shots:
    SYSTEM_PROMPT += f"\nQ: {ex['Question']}\nA: {ex['Final answer']}"

sys_msg = SystemMessage(content=SYSTEM_PROMPT)

# ---------------------------------------------------------------------------
# 5. LLM + tool binding
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# 6. LangGraph nodes
# ---------------------------------------------------------------------------


def node_retriever(state: MessagesState):
    """Prepend best‑match solved QA as additional context."""
    query = state["messages"][0].content
    hit1, hit2, *_ = vstore.similarity_search(query, k=2)
    reference = HumanMessage(content=f"{hit1.page_content}\n\n---\n{hit2.page_content}")
    return {"messages": [sys_msg, reference] + state["messages"]}


def node_assistant(state: MessagesState):
    """Run the LLM once."""
    return {"messages": [llm.invoke(state["messages"])]}


# build graph ----------------------------------------------------------------
builder = StateGraph(MessagesState)

builder.add_node("retriever", node_retriever)
builder.add_node("assistant", node_assistant)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "assistant")

builder.add_conditional_edges("assistant", tools_condition)  # → tools or END
builder.add_edge("tools", "assistant")  # loop back after tool call

agent_graph = builder.compile()

# ---------------------------------------------------------------------------
# 7. helpers
# ---------------------------------------------------------------------------


def solve(question: str) -> str:
    """Return final answer for a single GAIA question."""
    out = agent_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, config={"recursion_limit": 24}
    )
    return out["messages"][-1].content.strip()


def evaluate() -> None:
    """Run evaluation batch and submit to leaderboard."""
    qs = requests.get(f"{API_BASE}/questions", timeout=30).json()
    answers = [
        {"task_id": q["id"], "submitted_answer": solve(q["question"])} for q in qs
    ]
    payload = {
        "username": SPACE_HOST,
        "agent_code": AGENT_CODE_URL,
        "answers": answers,
    }
    res = requests.post(f"{API_BASE}/submit", json=payload, timeout=60)
    print("[agent] 🏆", res.status_code, res.text)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*", help="manual question")
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()

    if args.submit:
        evaluate()
    elif args.question:
        print(solve(" ".join(args.question)))
    else:
        p.print_help()
