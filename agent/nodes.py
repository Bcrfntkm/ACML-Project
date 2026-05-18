"""LangGraph node functions for the chemical annotation pipeline."""
from __future__ import annotations
import json, logging, os, re, subprocess, sys
from typing import Any

from .config import settings as _default_settings
from .llm import get_llm
from .prompts import build_messages, build_paragraph_messages
from .rag import rag_retrieve_node   # re-exported
from .section_filter import filter_paragraphs
from .state import AgentState

logger = logging.getLogger(__name__)
__all__ = ["parse_pdf_node", "rag_retrieve_node", "annotate_paragraphs_node"]


# ── Node 1: PDF parsing via subprocess ──────────────────────────────────────

def parse_pdf_node(state: AgentState) -> dict:
    """
    Parse PDF using generalParser.py via subprocess (avoids torch import).
    Calls: python3 generalParser.py -i <pdf_path>
    Reads result from: pdf2text/results/<stem>.json
    Returns filtered paragraphs (noise sections removed).
    """
    pdf_path = state.get("pdf_path", "")
    parser_dir = _default_settings.parser_dir  # Science/parser/ReactionMiner (absolute)
    pdf2text_dir = os.path.join(parser_dir, "pdf2text")

    if not os.path.isfile(pdf_path):
        return {"error": f"PDF not found: {pdf_path}", "raw_text": "", "paragraphs": []}

    pdf_path_abs = os.path.abspath(pdf_path)
    stem = os.path.splitext(os.path.basename(pdf_path_abs))[0]
    result_json = os.path.join(pdf2text_dir, "results", f"{stem}.json")

    # If result already exists, skip parsing
    if not os.path.exists(result_json):
        # Find python executable — prefer the agent venv
        science_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        venv_python = os.path.join(science_dir, "agent-venv", "bin", "python3")
        python_exe = venv_python if os.path.isfile(venv_python) else sys.executable

        general_parser = os.path.join(pdf2text_dir, "generalParser.py")
        cmd = [python_exe, general_parser, "-i", pdf_path_abs]

        logger.info("Running parser: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=pdf2text_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                logger.warning("Parser stderr: %s", proc.stderr[:500])
        except subprocess.TimeoutExpired:
            return {"error": "Parser timed out after 120s", "raw_text": "", "paragraphs": []}
        except Exception as exc:
            return {"error": f"Parser subprocess failed: {exc}", "raw_text": "", "paragraphs": []}

    if not os.path.exists(result_json):
        return {"error": f"Parser produced no output at {result_json}", "raw_text": "", "paragraphs": []}

    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"error": f"Failed to read parser output: {exc}", "raw_text": "", "paragraphs": []}

    raw_text = data.get("fullText", "") or ""
    all_paragraphs = list(data.get("contents", []) or [])

    # Filter out noise sections
    filtered = filter_paragraphs(all_paragraphs)
    logger.info("Paragraphs: %d total → %d after filtering", len(all_paragraphs), len(filtered))

    return {
        "raw_text": raw_text,
        "paragraphs": filtered,
        "current_para_idx": 0,
        "accumulated_entities": [],
        "annotations": [],
    }


# ── Node 2: RAG retrieve (re-exported from rag.py) ──────────────────────────
# rag_retrieve_node is imported above


# ── Node 3: Paragraph-level iterative annotation ────────────────────────────

def _parse_llm_response(raw: str) -> list[dict]:
    """Extract annotations list from model's raw JSON response."""
    if not raw:
        return []
    s = raw.strip()
    # Strip markdown code fences
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
        anns = parsed.get("annotations", [])
    elif isinstance(parsed, list):
        anns = parsed
    else:
        anns = []
    return [a for a in anns if isinstance(a, dict)]


def _validate_spans(annotations: list[dict], text: str) -> list[dict]:
    """Validate char offsets against paragraph text; recover via substring search."""
    cleaned = []
    for ann in annotations:
        span_text = ann.get("text")
        label = ann.get("label", "OTHER")
        start = ann.get("start")
        end = ann.get("end")
        if not isinstance(span_text, str) or not span_text.strip():
            continue
        offsets_ok = (
            isinstance(start, int) and isinstance(end, int)
            and 0 <= start < end <= len(text)
            and text[start:end] == span_text
        )
        if not offsets_ok:
            idx = text.find(span_text)
            if idx == -1:
                continue
            start, end = idx, idx + len(span_text)
        cleaned.append({
            "text": span_text,
            "label": str(label).upper(),
            "start": int(start),
            "end": int(end),
        })
    return cleaned


def _remap_offsets(annotations: list[dict], para_text: str, full_text: str) -> list[dict]:
    """
    Remap paragraph-local offsets to full-text absolute offsets.

    Strategy: search for each annotation's text directly in full_text, starting
    near the paragraph's approximate position.  This is robust even when the
    paragraph text differs from full_text (e.g. the parser merges/cleans text).

    Falls back to a global search if the local search fails.
    Annotations whose text cannot be found anywhere in full_text are dropped.
    """
    # Estimate the paragraph's approximate start in full_text using a short prefix.
    # We only use this as a *search hint*, not as the definitive offset.
    approx_start = -1
    for prefix_len in [100, 50, 30, 20]:
        approx_start = full_text.find(para_text[:min(prefix_len, len(para_text))])
        if approx_start != -1:
            break

    # Search window: ±2000 chars around the approximate paragraph position.
    # If we couldn't locate the paragraph at all, search the whole document.
    if approx_start != -1:
        window_start = max(0, approx_start - 200)
        window_end = min(len(full_text), approx_start + len(para_text) + 2000)
    else:
        window_start = 0
        window_end = len(full_text)

    remapped = []
    for ann in annotations:
        span_text = ann["text"]
        if not span_text:
            continue

        # 1. Try to find the span within the local window first.
        local_idx = full_text.find(span_text, window_start, window_end)

        # 2. Fall back to a global search if not found in the window.
        if local_idx == -1:
            local_idx = full_text.find(span_text)

        if local_idx == -1:
            # Span text not found anywhere in full_text — skip this annotation.
            logger.debug("_remap_offsets: span %r not found in full_text; dropping.", span_text[:40])
            continue

        remapped.append({
            "text": span_text,
            "label": ann["label"],
            "start": local_idx,
            "end": local_idx + len(span_text),
        })
    return remapped


def annotate_paragraphs_node(state: AgentState) -> dict:
    """
    Annotate all filtered paragraphs one by one.
    Each LLM call receives:
      - The current paragraph text
      - A summary of entity names found so far (accumulated context / mini-RAG)
    Annotations are accumulated with absolute offsets into state["annotations"].
    """
    paragraphs = state.get("paragraphs", [])
    raw_text = state.get("raw_text", "")
    query = state.get("query", "Annotate all chemical entities.")
    rag_context = state.get("rag_context", [])

    if not paragraphs:
        return {"annotations": []}

    if state.get("error"):
        return {"annotations": []}

    llm = get_llm(_default_settings)
    all_annotations: list[dict] = []
    accumulated_entities: list[str] = []

    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para or len(para) < 20:  # skip very short paragraphs
            continue

        logger.info("Annotating paragraph %d/%d (%d chars)", idx + 1, len(paragraphs), len(para))

        try:
            messages = build_paragraph_messages(
                paragraph=para,
                query=query,
                accumulated_entities=accumulated_entities,
                rag_context=rag_context,
                version=state.get("prompt_version", "v2"),
            )
            response = llm.invoke(messages)
            raw_output = response.content if hasattr(response, "content") else str(response)

            annotations = _parse_llm_response(raw_output)
            annotations = _validate_spans(annotations, para)
            annotations = _remap_offsets(annotations, para, raw_text)

            all_annotations.extend(annotations)

            # Update accumulated entity names for next iteration
            new_entities = [a["text"] for a in annotations if a["text"] not in accumulated_entities]
            accumulated_entities.extend(new_entities)
            # Keep accumulated list manageable (last 50 unique entities)
            accumulated_entities = list(dict.fromkeys(accumulated_entities))[-50:]

        except Exception as exc:
            logger.warning("Failed to annotate paragraph %d: %s", idx, exc)
            continue

    # Deduplicate: remove exact duplicates (same text+label+start+end)
    seen = set()
    deduped = []
    for ann in all_annotations:
        key = (ann["text"], ann["label"], ann["start"], ann["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(ann)

    return {
        "annotations": deduped,
        "accumulated_entities": accumulated_entities,
    }
