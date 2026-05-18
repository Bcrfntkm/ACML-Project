"""Compatibility shim — this file has been superseded by the LangGraph pipeline.

The original monolithic agent logic has been split across:

    Science/agent/
    ├── config.py    — Settings dataclass + loader
    ├── llm.py       — ChatOpenAI factory (ProxyAPI)
    ├── state.py     — AgentState TypedDict
    ├── prompts.py   — SYSTEM_PROMPT, few-shot examples, build_messages()
    ├── nodes.py     — LangGraph node functions
    ├── graph.py     — StateGraph definition + run_pipeline()
    └── rag.py       — RAGRetriever + rag_retrieve_node()

Entry point::

    from Science.agent import run_pipeline
    annotations = run_pipeline("/path/to/paper.pdf")
"""

# Re-export the public API so any code that previously did
#   ``from Science.agent.agent import ...``
# continues to work without modification.
from .graph import run_pipeline  # noqa: F401
from .config import Settings, load_settings, settings  # noqa: F401
from .llm import get_llm  # noqa: F401

__all__ = ["run_pipeline", "Settings", "load_settings", "settings", "get_llm"]
