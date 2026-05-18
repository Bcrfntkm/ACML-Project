"""LangGraph agent state definition for the chemical annotation pipeline."""
from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    pdf_path: str
    query: str

    # After parse_pdf_node
    raw_text: str            # full text (for reference)
    paragraphs: list[str]   # filtered content paragraphs

    # Paragraph iteration state
    current_para_idx: int           # which paragraph we're currently annotating
    accumulated_entities: list[str] # entity names found so far (for context)

    # RAG
    rag_context: list[str]

    # Output — NER (8-label schema)
    annotations: list[dict]  # all NER annotations accumulated across paragraphs

    # Output — property extraction (5-role schema)
    property_records: list[dict]  # list of {paragraph_id, text, records: [...]}

    error: Optional[str]

    # Experiment control
    prompt_version: str   # which prompt version to use, e.g. "v1" or "v2"
