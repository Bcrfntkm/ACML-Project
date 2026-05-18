"""LangGraph pipeline for property extraction (5-role schema).

Topology:
  START → parse_pdf → rag_retrieve → annotate_properties → END

Output: state["property_records"]
  [{"paragraph_id": "para_N", "text": "...", "records": [
      {"substance": "...", "property_name": "...", "property_value": "...",
       "conditions": "...", "measurement_method": "..."},
      ...
  ]}, ...]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .nodes import parse_pdf_node, rag_retrieve_node
from .nodes_property import annotate_properties_node
from .state import AgentState

logger = logging.getLogger(__name__)


def build_property_graph():
    """Build and compile the property extraction graph."""
    graph = StateGraph(AgentState)
    graph.add_node("parse_pdf",           parse_pdf_node)
    graph.add_node("rag_retrieve",        rag_retrieve_node)
    graph.add_node("annotate_properties", annotate_properties_node)
    graph.add_edge(START,                 "parse_pdf")
    graph.add_edge("parse_pdf",           "rag_retrieve")
    graph.add_edge("rag_retrieve",        "annotate_properties")
    graph.add_edge("annotate_properties", END)
    return graph.compile()


def run_property_pipeline(
    pdf_path: str,
    output_path: str | None = None,
) -> list[dict]:
    """
    Run the property extraction pipeline on a PDF.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF file.
    output_path:
        If given, save the result as JSON to this path.

    Returns
    -------
    List of paragraph result dicts, each with keys:
        paragraph_id, text, records
    where each record has: substance, property_name, property_value,
    and optionally conditions, measurement_method.
    """
    compiled = build_property_graph()
    initial_state: AgentState = {
        "pdf_path":            pdf_path,
        "query":               "",
        "raw_text":            "",
        "paragraphs":          [],
        "current_para_idx":    0,
        "accumulated_entities": [],
        "rag_context":         [],
        "annotations":         [],
        "property_records":    [],
        "error":               None,
        "prompt_version":      "v2",
    }
    final_state = compiled.invoke(initial_state)

    if final_state.get("error"):
        logger.error("Pipeline error: %s", final_state["error"])

    results = final_state.get("property_records", [])

    if output_path:
        Path(output_path).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved %d paragraph results → %s", len(results), output_path)

    return results
