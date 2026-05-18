#!/usr/bin/env python3
"""
Bypass script: extracts text from PDF using pdfplumber (no SymbolScraper/torch needed),
then calls annotate_paragraphs_node directly, bypassing parse_pdf_node.

Usage (from workspace root, i.e. parent of Science/):
    Science/agent-venv/bin/python Science/scripts/test_direct.py path/to/paper.pdf
    Science/agent-venv/bin/python Science/scripts/test_direct.py path/to/paper.pdf -o data/annotations/out.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve workspace root (parent of Science/) regardless of cwd
_SCRIPTS_DIR = Path(__file__).resolve().parent          # Science/scripts/
_SCIENCE_DIR = _SCRIPTS_DIR.parent                      # Science/
_WORKSPACE_ROOT = _SCIENCE_DIR.parent                   # workspace root

if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# Load .env before importing agent modules
from dotenv import load_dotenv
load_dotenv(_SCIENCE_DIR / ".env")


def _parse_args():
    p = argparse.ArgumentParser(description="Test annotation pipeline without SymbolScraper.")
    p.add_argument("pdf_path", metavar="PDF", help="Path to input PDF")
    p.add_argument("-o", "--output", metavar="FILE", default=None,
                   help="Save JSON output to FILE (default: print to stdout)")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Truncate extracted text to N chars (default: no limit)")
    p.add_argument("-q", "--query", default=None,
                   help="Custom annotation instruction")
    return p.parse_args()


def main():
    args = _parse_args()
    pdf_path = str(Path(args.pdf_path).resolve())

    # ── Step 1: Extract text via pdfplumber ──────────────────────────────────
    print(f"[INFO] Extracting text from: {pdf_path}", file=sys.stderr)
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] pdfplumber not installed. Run: pip install pdfplumber", file=sys.stderr)
        sys.exit(1)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
                print(f"[INFO]   Page {i+1}: {len(page_text)} chars", file=sys.stderr)
            extracted_text = "\n\n".join(pages_text)
    except Exception as e:
        print(f"[ERROR] pdfplumber failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not extracted_text.strip():
        print("[ERROR] Extracted text is empty!", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Total extracted text: {len(extracted_text)} chars", file=sys.stderr)

    if args.max_chars and len(extracted_text) > args.max_chars:
        print(f"[INFO] Truncating to {args.max_chars} chars", file=sys.stderr)
        extracted_text = extracted_text[:args.max_chars]

    # ── Step 2: Call annotate_paragraphs_node directly ───────────────────────
    print("[INFO] Importing agent modules...", file=sys.stderr)
    from Science.agent.state import AgentState
    from Science.agent.nodes import annotate_paragraphs_node
    from Science.agent.config import settings as _settings

    print(f"[INFO] model={_settings.model}, timeout={_settings.timeout}s, "
          f"max_tokens={_settings.max_tokens}", file=sys.stderr)

    query = args.query or (
        "Find all chemical entities: compounds, reactions, properties, "
        "CAS numbers, IUPAC names, elements, units"
    )

    # Split into simple paragraphs for the node
    paragraphs = [p for p in extracted_text.split("\n\n") if len(p.strip()) >= 20]

    state: AgentState = {
        "pdf_path": pdf_path,
        "query": query,
        "raw_text": extracted_text,
        "paragraphs": paragraphs,
        "current_para_idx": 0,
        "accumulated_entities": [],
        "rag_context": [],
        "annotations": [],
        "error": None,
    }

    # Patch _parse_llm_response to log raw LLM output for debugging
    import Science.agent.nodes as _nodes_module
    _orig = _nodes_module._parse_llm_response

    def _debug_parse(raw: str) -> list:
        print(f"[DEBUG] LLM response: {len(raw)} chars, tail: ...{raw[-150:]}", file=sys.stderr)
        return _orig(raw)

    _nodes_module._parse_llm_response = _debug_parse

    print("[INFO] Running annotate_paragraphs_node...", file=sys.stderr)
    try:
        result = annotate_paragraphs_node(state)
    except Exception as e:
        print(f"[ERROR] annotate_paragraphs_node failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    annotations = result.get("annotations", [])
    if result.get("error"):
        print(f"[ERROR] Node returned error: {result['error']}", file=sys.stderr)

    print(f"[INFO] Annotations produced: {len(annotations)}", file=sys.stderr)

    # ── Step 3: Output ────────────────────────────────────────────────────────
    output = json.dumps(annotations, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"[INFO] Saved to: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
