# --------------------- agent.py ---------------------
"""Minimal GAIA Level‑1 agent (OpenAI + FAISS)

▶ Dependencies
   pip install langchain langgraph sentence-transformers faiss-cpu \
               langchain-openai langchain-huggingface \ 
               python-dotenv requests openai

▶ Required env vars (can be placed in a local `.env` file)
   OPENAI_API_KEY   – your OpenAI key
   GAIA_API_BASE    – base URL of the GAIA evaluation API
   HF_USERNAME      – your Hugging Face username
   AGENT_CODE_URL   – public code URL of this Space (…/tree/main)

Usage
-----
$ python agent.py "What city hosted Expo 2015?"   # single‑question test
$ python agent.py --submit                         # answer & submit full eval set
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from typing import List
from tools import web_search, wiki_search, calculator, arvix_search


import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------------------
# 1. Load env vars (supports a local .env file) & validate
# ---------------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
API_BASE = os.getenv("GAIA_API_BASE")
HF_USERNAME = os.getenv("HF_USERNAME")
AGENT_CODE_URL = os.getenv("AGENT_CODE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all((API_BASE, HF_USERNAME, AGENT_CODE_URL, OPENAI_API_KEY)):
    sys.exit(
        "[agent] 🔑  Missing one or more env vars: GAIA_API_BASE, HF_USERNAME, AGENT_CODE_URL, OPENAI_API_KEY"
    )

# ---------------------------------------------------------------------------
# 2. Load solved examples from metadata.jsonl
# ---------------------------------------------------------------------------
DATA_PATH = pathlib.Path(__file__).with_name("metadata.jsonl")
if not DATA_PATH.exists():
    sys.exit("[agent] 📄 metadata.jsonl missing next to agent.py")
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
    description="Return up to three previously‑solved GAIA level‑1 Q&A pairs that resemble the current query.",
)


TOOLS = [
    similar_q_tool,
    web_search,
    wiki_search,
    calculator,
    arvix_search,
]

# ---------------------------------------------------------------------------
# 4. Build the system prompt with few‑shot examples
# ---------------------------------------------------------------------------
few_shots = random.sample(examples, k=3)
SYSTEM_PROMPT = "You are a GAIA level‑1 agent. Answer with ONLY the final answer – no additional text."
for ex in few_shots:
    SYSTEM_PROMPT += f"\nQ: {ex['Question']}\nA: {ex['Final answer']}"

# ---------------------------------------------------------------------------
# 5. Initialise OpenAI chat model & bind tools
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# 6. LangGraph orchestration
# ---------------------------------------------------------------------------


def _assistant(state: MessagesState):
    """Single LLM invocation step."""
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
# 7. Public helpers
# ---------------------------------------------------------------------------


def solve(question: str) -> str:
    """Return the agent's final answer for a single GAIA question."""
    result = agent_graph.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content.strip()


def evaluate() -> None:
    """Fetch 20 eval questions, solve, and POST to leaderboard."""
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
    print("[agent] 🏆 Leaderboard response:", res.status_code, res.text)


# ---------------------------------------------------------------------------
# 8. CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA agent runner")
    parser.add_argument("question", nargs="*", help="Manual question to solve")
    parser.add_argument(
        "--submit", action="store_true", help="Run full evaluation and submit"
    )
    args = parser.parse_args()

    if args.submit:
        evaluate()
    elif args.question:
        q = " ".join(args.question)
        print("Answer:", solve(q))
    else:
        parser.print_help()
