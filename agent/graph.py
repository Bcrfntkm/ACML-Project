"""LangGraph StateGraph for the chemical annotation pipeline."""
from __future__ import annotations
import logging

from langgraph.graph import END, START, StateGraph
from .nodes import annotate_paragraphs_node, parse_pdf_node, rag_retrieve_node
from .state import AgentState

logger = logging.getLogger(__name__)

DEFAULT_QUERY = (
    "Annotate all chemical entities: compounds, IUPAC names, CAS numbers, "
    "reactions, physicochemical properties, elements, and units of measurement."
)


def build_graph():
    """
    Build and compile the LangGraph annotation pipeline.

    Topology:
      START → parse_pdf → rag_retrieve → annotate_paragraphs → END

    parse_pdf_node:
      - Calls generalParser.py via subprocess (avoids torch import)
      - Filters noise sections (References, Acknowledgments, etc.)
      - Returns filtered paragraphs

    rag_retrieve_node:
      - Optional; retrieves reference context chunks

    annotate_paragraphs_node:
      - Iterates over paragraphs one by one
      - Each call includes accumulated entity history (mini-RAG)
      - Returns merged, deduplicated annotations with absolute offsets
    """
    graph = StateGraph(AgentState)
    graph.add_node("parse_pdf", parse_pdf_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("annotate_paragraphs", annotate_paragraphs_node)
    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "rag_retrieve")
    graph.add_edge("rag_retrieve", "annotate_paragraphs")
    graph.add_edge("annotate_paragraphs", END)
    return graph.compile()


def run_pipeline(
    pdf_path: str,
    query: str | None = None,
    prompt_version: str = "v2",
) -> list[dict]:
    """Run full annotation pipeline; returns list of annotation dicts."""
    compiled = build_graph()
    initial_state: AgentState = {
        "pdf_path": pdf_path,
        "query": query or DEFAULT_QUERY,
        "raw_text": "",
        "paragraphs": [],
        "current_para_idx": 0,
        "accumulated_entities": [],
        "rag_context": [],
        "annotations": [],
        "error": None,
        "prompt_version": prompt_version,
    }
    final_state = compiled.invoke(initial_state)
    if final_state.get("error"):
        logger.error("Pipeline error: %s", final_state["error"])
    return final_state.get("annotations", [])
