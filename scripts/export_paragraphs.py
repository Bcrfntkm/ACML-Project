#!/usr/bin/env python3
"""
Export paragraph-level NER annotations with paragraph-local offsets.

Unlike annotate_paragraphs_node (which remaps offsets to document level),
this script keeps offsets local to each paragraph text — the format required
by bert/convert_to_bio.py.

Output format:
  [
    {
      "paragraph_id": 0,
      "text": "...",
      "annotations": [{"text": "...", "label": "...", "start": N, "end": M}, ...]
    },
    ...
  ]

Usage (from workspace root):
  Science/agent-venv/bin/python Science/scripts/export_paragraphs.py paper.pdf
  Science/agent-venv/bin/python Science/scripts/export_paragraphs.py paper.pdf \\
      -o data/annotations/copper_ner.json --skip-empty -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCIENCE_DIR = _SCRIPTS_DIR.parent
_WORKSPACE_ROOT = _SCIENCE_DIR.parent

if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import re

from dotenv import load_dotenv
load_dotenv(_SCIENCE_DIR / ".env")

# ---------------------------------------------------------------------------
# Paragraph quality filter
# ---------------------------------------------------------------------------

_REF_START    = re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.)\s+\w')
_NMR_HEAVY    = re.compile(r'δ\s*[\d.]+|J\s*=\s*[\d.]+\s*Hz', re.IGNORECASE)
_DOI_PATTERN  = re.compile(r'\bdoi\s*:\s*10\.\d{4}', re.IGNORECASE)
_NUMERIC_ONLY = re.compile(r'^[\d\s.,;:\-()\[\]±×%°/]+$')

_SKIP_HEADERS = re.compile(
    r'^\s*(references|bibliography|acknowledgements?|acknowledgments?|'
    r'supporting information|supplementary|author contributions?|'
    r'conflict of interest|funding|data availability)\s*$',
    re.IGNORECASE,
)

_CHEM_SIGNAL  = re.compile(
    r'\b(compound|synthesis|reaction|yield|solution|mixture|temperature|'
    r'solvent|catalyst|reagent|product|precursor|complex|ligand|crystal|'
    r'mol|mmol|equiv|°C|°F|pKa|pH|NMR|IR|HPLC|MS|m\.p\.|b\.p\.|'
    r'DMSO|THF|CDCl|acetone|ethanol|methanol|water|acid|base|amine|'
    r'ester|aldehyde|ketone|alkyl|aryl|phenyl|benzene|pyridine)\b',
    re.IGNORECASE,
)


def _is_useful_paragraph(text: str) -> bool:
    """Return True if the paragraph likely contains annotatable chemical content."""
    s = text.strip()

    # Too short or too long (>3000 chars → probably a merged section dump)
    if len(s) < 40 or len(s) > 3000:
        return False

    # Section headers
    if _SKIP_HEADERS.match(s):
        return False

    # Starts like a bibliography entry
    if _REF_START.match(s):
        return False

    # All numeric / table row
    if _NUMERIC_ONLY.match(s):
        return False

    # Dense NMR dump: many δ shifts with no prose (characterisation section can be OK)
    nmr_hits = len(_NMR_HEAVY.findall(s))
    if nmr_hits > 6 and len(s.split()) < 80:
        return False

    # Reference-list paragraph: multiple DOIs + short length
    if len(_DOI_PATTERN.findall(s)) >= 2 and len(s) < 600:
        return False

    # Needs at least one chemistry signal word to be worth annotating
    if not _CHEM_SIGNAL.search(s):
        return False

    return True


def _parse_args():
    p = argparse.ArgumentParser(
        description="Annotate PDF paragraphs and export with paragraph-local offsets."
    )
    p.add_argument("pdf_path", metavar="PDF", help="Path to input PDF")
    p.add_argument("-o", "--output", metavar="FILE", default=None,
                   help="Save JSON output to FILE (default: stdout)")
    p.add_argument("-q", "--query", default=None,
                   help="Custom annotation instruction")
    p.add_argument("--prompt-version", default="v2",
                   help="Prompt version: v1 or v2 (default: v2)")
    p.add_argument("--skip-empty", action="store_true",
                   help="Omit paragraphs where LLM found no annotations")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable paragraph quality pre-filtering")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Truncate extracted text to N chars")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _extract_text_pdfplumber(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] pdfplumber not installed. Run: pip install pdfplumber", file=sys.stderr)
        sys.exit(1)

    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            pages.append(t)
            logging.debug("Page %d: %d chars", i + 1, len(t))
        return "\n\n".join(pages)


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    pdf_path = str(Path(args.pdf_path).resolve())
    logging.info("Extracting text from: %s", pdf_path)

    text = _extract_text_pdfplumber(pdf_path)
    if not text.strip():
        logging.error("Extracted text is empty.")
        sys.exit(1)

    if args.max_chars and len(text) > args.max_chars:
        logging.info("Truncating to %d chars", args.max_chars)
        text = text[: args.max_chars]

    logging.info("Total extracted: %d chars", len(text))

    # Split into paragraphs
    raw_paragraphs = [p for p in text.split("\n\n") if len(p.strip()) >= 20]

    # Quality pre-filter
    if args.no_filter:
        paragraphs = raw_paragraphs
        logging.info("Paragraphs: %d (filter disabled)", len(paragraphs))
    else:
        paragraphs = [p for p in raw_paragraphs if _is_useful_paragraph(p)]
        logging.info(
            "Paragraphs: %d total → %d after quality filter",
            len(raw_paragraphs), len(paragraphs),
        )

    # ── Import agent helpers ──────────────────────────────────────────────────
    from Science.agent.config import settings as _settings
    from Science.agent.llm import get_llm
    from Science.agent.nodes import _parse_llm_response, _validate_spans
    from Science.agent.prompts import build_paragraph_messages

    logging.info("model=%s  timeout=%ss  max_tokens=%s",
                 _settings.model, _settings.timeout, _settings.max_tokens)

    query = args.query or (
        "Find all chemical entities: compounds, reactions, properties, "
        "CAS numbers, IUPAC names, elements, units"
    )

    llm = get_llm(_settings)
    accumulated_entities: list[str] = []
    results: list[dict] = []

    prev_para_suffix = ""

    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        logging.info("Paragraph %d/%d (%d chars)", idx + 1, len(paragraphs), len(para))

        # Build rag_context: pass tail of previous paragraph for cross-para context
        context_chunks = []
        if prev_para_suffix:
            context_chunks = [f"[Previous paragraph ending]\n{prev_para_suffix}"]

        try:
            messages = build_paragraph_messages(
                paragraph=para,
                query=query,
                accumulated_entities=accumulated_entities,
                rag_context=context_chunks,
                version=args.prompt_version,
            )
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)

            logging.debug("LLM raw tail: ...%s", raw[-120:])

            annotations = _parse_llm_response(raw)
            annotations = _validate_spans(annotations, para)

            # Update accumulated entities for context continuity
            new_entities = [a["text"] for a in annotations
                            if a["text"] not in accumulated_entities]
            accumulated_entities.extend(new_entities)
            accumulated_entities = list(dict.fromkeys(accumulated_entities))[-50:]

        except Exception as exc:
            logging.warning("Paragraph %d failed: %s", idx, exc)
            annotations = []

        # Keep tail of this paragraph as context for the next call
        prev_para_suffix = para[-250:] if len(para) > 250 else para

        if args.skip_empty and not annotations:
            continue

        results.append({
            "paragraph_id": idx,
            "text": para,
            "annotations": annotations,
        })

    logging.info("Done. Paragraphs saved: %d  Total annotations: %d",
                 len(results), sum(len(r["annotations"]) for r in results))

    output = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        logging.info("Saved to: %s", out_path)
    else:
        print(output)


if __name__ == "__main__":
    main()
