"""
LLM provider factory.

Supports Anthropic (Claude) and OpenAI (GPT) interchangeably.
The provider is selected by the LLM_PROVIDER environment variable, or
auto-detected from whichever API key is present.  Both providers expose
the same LangChain ChatModel interface, so the rest of the codebase is
provider-agnostic.

Auto-detection priority: Anthropic → OpenAI (first key found wins).
"""

import os

from langchain_core.language_models import BaseChatModel

# Default models — override with LLM_MODEL in .env
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}


def get_llm(temperature: float = 0.3) -> BaseChatModel:
    """
    Return a configured LangChain chat model for the active provider.

    Provider resolution order:
    1. Explicit ``LLM_PROVIDER`` env variable ("anthropic" or "openai").
    2. Auto-detect: use Anthropic if ``ANTHROPIC_API_KEY`` is set,
       otherwise fall back to OpenAI if ``OPENAI_API_KEY`` is set.

    Raises
    ------
    ValueError
        If no supported API key is found in the environment.
    """
    provider = _resolve_provider()
    model = os.getenv("LLM_MODEL", _DEFAULT_MODELS[provider])

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    # provider == "openai"
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


def _resolve_provider() -> str:
    """Return the active provider name ("anthropic" or "openai")."""
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in ("anthropic", "openai"):
        return explicit

    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    raise ValueError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
    )
