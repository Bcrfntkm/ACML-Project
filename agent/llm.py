"""LLM factory for the chemical annotation agent.

Builds a :class:`langchain_openai.ChatOpenAI` instance pointed at the
ProxyAPI endpoint using the parameters from :class:`~config.Settings`.
"""

from __future__ import annotations

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-openai is required. Install with: pip install langchain-openai"
    ) from exc

from .config import Settings


def get_llm(settings: Settings) -> ChatOpenAI:
    """Return a :class:`ChatOpenAI` instance configured for ProxyAPI.

    Parameters
    ----------
    settings:
        A populated :class:`~config.Settings` dataclass.

    Returns
    -------
    ChatOpenAI
        Ready-to-use LangChain chat model.
    """
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )
