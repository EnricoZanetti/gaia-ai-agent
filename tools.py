"""Reusable tool definitions for the GAIA agent.

Each callable is decorated with `@tool`, carries a docstring, and can be
imported directly into `agent.py`.
"""

from __future__ import annotations

import json
import os

from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool

# 1. RAG over metadata.jsonl

_FAISS_INDEX = None


def _get_vectorstore() -> FAISS:
    global _FAISS_INDEX
    if _FAISS_INDEX is not None:
        return _FAISS_INDEX

    docs = []
    path = os.path.join(os.path.dirname(__file__), "data", "metadata.jsonl")
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            content = f"Q: {item['question']}\nA: {item['answer']}"
            docs.append(Document(page_content=content, metadata=item))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _FAISS_INDEX = FAISS.from_documents(docs, embeddings)
    return _FAISS_INDEX


@tool
def similar_question(query: str) -> str:
    """
    Retrieve the most similar question from the metadata.jsonl dataset.

    Args:
        query: The input question to find similar questions for.
    """
    vectorstore = _get_vectorstore()
    similar_docs = vectorstore.similarity_search(query, k=3)
    if similar_docs:
        doc = similar_docs[0]
        return f'Similar Question: "{doc.metadata["question"]}"\nAnswer: "{doc.metadata["answer"]}"'
    else:
        return "No similar questions found."


# 2. Wikipedia search


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for *query* and return up to two snippets."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# 3. Web search (Tavily)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY:
    # Tavily available → use the real tool
    from langchain_community.tools.tavily_search import TavilySearchResults

    _TAVILY_SEARCH = TavilySearchResults(
        max_results=5,
        include_answer=True,
        include_raw_content=True,
    )

    @tool
    def web_search(query: str) -> str:
        """Search the web using Tavily (if API key is configured)."""
        raw = _TAVILY_SEARCH.invoke({"query": query})

        if isinstance(raw, dict):
            data = raw
        else:
            try:
                data = json.loads(raw)
            except Exception:
                return str(raw)

        parts = []
        answer = data.get("answer")
        if answer:
            parts.append(f"Answer:\n{answer}")

        sources = data.get("results") or []
        if sources:
            formatted = []
            for i, r in enumerate(sources[:5], start=1):
                title = r.get("title", "Source")
                url = r.get("url") or ""
                snippet = (r.get("content") or r.get("snippet") or "").replace("\n", " ")
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                formatted.append(f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}")
            parts.append("\n\n".join(formatted))

        return "\n\n".join(parts) if parts else str(raw)

else:
    # Tavily NOT available → provide a dummy tool
    @tool
    def web_search(query: str) -> str:
        """Web search is disabled because TAVILY_API_KEY is not set."""
        return (
            "Web search is unavailable because no TAVILY_API_KEY is configured.\n"
            "You can still use all other tools normally."
        )

# 4. Arxiv search


@tool
def arxiv_search(query: str) -> str:
    """
    Search Arxiv for a query and return maximum 3 results.

    Args:
        query: The search query.
    """
    search_docs = ArxivLoader(query=query, load_max_doc=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arvix_results": formatted_search_docs}


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic Python arithmetic *expression* and return the result.

    Example inputs::
        "2 + 2"
        "(3 ** 2) / 5"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"
