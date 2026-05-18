"""LangGraph node for property extraction (5-role schema).

Extracted records per paragraph:
  {"substance": "...", "property_name": "...", "property_value": "...",
   "conditions": "...", "measurement_method": "..."}

Any absent role is omitted.  Output accumulates in state["property_records"] as:
  [{"paragraph_id": "para_0", "text": "...", "records": [...]}, ...]
"""
from __future__ import annotations

import json
import logging
import re

from .config import settings as _default_settings
from .llm import get_llm
from .prompts_property import build_property_messages
from .state import AgentState

logger = logging.getLogger(__name__)
__all__ = ["annotate_properties_node"]

PROPERTY_ROLES = frozenset({
    "substance", "property_name", "property_value", "conditions", "measurement_method",
})


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_property_response(raw: str) -> list[dict]:
    """Extract records list from model's raw JSON response."""
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        records = parsed.get("records", [])
    elif isinstance(parsed, list):
        records = parsed
    else:
        return []

    return [r for r in records if isinstance(r, dict)]


def _clean_records(records: list[dict]) -> list[dict]:
    """Keep only known role keys; drop records that have no key roles."""
    cleaned = []
    for rec in records:
        clean = {
            k: str(v).strip()
            for k, v in rec.items()
            if k in PROPERTY_ROLES and v and str(v).strip()
        }
        # A record must have at least property_name + property_value to be useful
        if clean.get("property_name") and clean.get("property_value"):
            cleaned.append(clean)
    return cleaned


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def annotate_properties_node(state: AgentState) -> dict:
    """
    Iterate over filtered paragraphs and extract property records from each.

    Each LLM call receives:
      - The paragraph text
      - Names of substances seen in previous paragraphs (for consistency)
      - Optional RAG context

    Returns updated state["property_records"]:
      [{"paragraph_id": "para_N", "text": "...", "records": [...]}, ...]
    """
    paragraphs   = state.get("paragraphs", [])
    rag_context  = state.get("rag_context", [])

    if not paragraphs or state.get("error"):
        return {"property_records": []}

    llm = get_llm(_default_settings)
    all_para_results: list[dict] = []
    accumulated_substances: list[str] = []

    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para or len(para) < 20:
            continue

        logger.info("Extracting properties from paragraph %d/%d (%d chars)",
                    idx + 1, len(paragraphs), len(para))

        try:
            messages = build_property_messages(
                paragraph=para,
                accumulated_substances=accumulated_substances,
                rag_context=rag_context,
            )
            response = llm.invoke(messages)
            raw_output = response.content if hasattr(response, "content") else str(response)

            records = _parse_property_response(raw_output)
            records = _clean_records(records)

            all_para_results.append({
                "paragraph_id": f"para_{idx}",
                "text": para,
                "records": records,
            })

            # Accumulate substance names for next paragraphs
            for rec in records:
                sub = rec.get("substance", "")
                if sub and sub not in accumulated_substances:
                    accumulated_substances.append(sub)
            accumulated_substances = list(dict.fromkeys(accumulated_substances))[-50:]

            logger.info("  → %d records extracted", len(records))

        except Exception as exc:
            logger.warning("Failed on paragraph %d: %s", idx, exc)
            all_para_results.append({
                "paragraph_id": f"para_{idx}",
                "text": para,
                "records": [],
                "error": str(exc),
            })

    return {"property_records": all_para_results}
