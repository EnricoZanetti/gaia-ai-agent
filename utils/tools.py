"""Reusable tool definitions for the GAIA agent.

Each callable is decorated with `@tool`, carries a docstring, and can be
imported directly into `agent.py`.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader

__all__ = [
    "web_search",
    "arxiv_search",
    "wiki_search",
    "calculator",
]


@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.

    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 results.

    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_doc=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arvix_results": formatted_search_docs}


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for *query* and return up to two snippets."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n---\n\n".join(d.page_content for d in docs)


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
