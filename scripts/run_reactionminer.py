#!/usr/bin/env python3
"""
Run ReactionMiner LLaMA-7B extractor on PDF files and save structured output.

This script:
  1. Extracts text from PDFs via pdfplumber
  2. Splits into paragraphs and filters synthesis-relevant ones
  3. Runs ReactionMiner LLaMA (MingZhong/reaction-miner-7b-lora) on each paragraph
  4. Records per-paragraph inference time
  5. Saves: {text, reactions, time_sec} records → JSON

Usage (from workspace root):
  Science/agent-venv/bin/python Science/scripts/run_reactionminer.py \
      Science/data/pdfs/molecules-11-01000.pdf \
      -o Science/data/annotations/rm_molecules-11-01000.json -v

  # Batch all PDFs:
  for f in Science/data/pdfs/*.pdf; do
      stem=$(basename "$f" .pdf)
      Science/agent-venv/bin/python Science/scripts/run_reactionminer.py \
          "$f" -o "Science/data/annotations/rm_${stem}.json" --skip-empty &
  done; wait

Requirements:
  pip install peft  (already installed)
  LLaMA-2-7b-hf access on HuggingFace (licence accepted)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SCIENCE_DIR  = _SCRIPTS_DIR.parent
_WORKSPACE_ROOT = _SCIENCE_DIR.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# ── regex paragraph filter (same as export_paragraphs.py) ───────────────────
import re

_CHEM_SIGNAL = re.compile(
    r'\b(compound|synthesis|reaction|yield|solution|mixture|temperature|'
    r'solvent|catalyst|reagent|product|precursor|complex|ligand|crystal|'
    r'mol|mmol|equiv|°C|reflux|stirr|filtrat|precipitat|dissolv|heated|'
    r'added|prepared|obtained|treated|washed|dried)\b',
    re.IGNORECASE,
)
_REF_START   = re.compile(r'^\s*(\[\d+\]|\(\d+\)|\d+\.)\s+\w')
_NMR_HEAVY   = re.compile(r'δ\s*[\d.]+|J\s*=\s*[\d.]+\s*Hz', re.IGNORECASE)

def _is_synthesis_paragraph(text: str) -> bool:
    s = text.strip()
    if len(s) < 60 or len(s) > 8000:
        return False
    if _REF_START.match(s):
        return False
    if len(_NMR_HEAVY.findall(s)) > 5 and len(s.split()) < 80:
        return False  # pure NMR dump
    return bool(_CHEM_SIGNAL.search(s))


# ── PDF extraction ───────────────────────────────────────────────────────────
def extract_paragraphs(pdf_path: str, max_chars: int | None = None) -> list[str]:
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] pip install pdfplumber", file=sys.stderr)
        sys.exit(1)
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = "\n\n".join(pages)
    if max_chars:
        text = text[:max_chars]
    raw = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 60]
    return [p for p in raw if _is_synthesis_paragraph(p)]


# ── Load ReactionMiner model (once) ─────────────────────────────────────────
_extractor = None

def get_extractor(base_model: str = "meta-llama/Llama-2-7b-hf"):
    global _extractor
    if _extractor is None:
        logging.info("Loading ReactionMiner LLaMA-7B (first call — may take 1–2 min)...")
        sys.path.insert(0, str(_SCIENCE_DIR / "parser" / "ReactionMiner" / "extraction"))
        from extractor import ReactionExtractor
        _extractor = ReactionExtractor(model_size="7b", base_model=base_model)
        logging.info("Model loaded.")
    return _extractor


# ── Main ─────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pdf_path", metavar="PDF")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--skip-empty", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    pdf_path = str(Path(args.pdf_path).resolve())
    logging.info("Extracting text: %s", pdf_path)
    paragraphs = extract_paragraphs(pdf_path, args.max_chars)
    logging.info("Synthesis paragraphs to process: %d", len(paragraphs))

    if not paragraphs:
        logging.warning("No synthesis paragraphs found.")
        sys.exit(0)

    extractor = get_extractor(args.base_model)

    results = []
    total_time = 0.0

    for idx, para in enumerate(paragraphs):
        logging.info("Para %d/%d (%d chars)", idx + 1, len(paragraphs), len(para))
        t0 = time.perf_counter()
        try:
            out = extractor.extract([para])
        except Exception as exc:
            logging.warning("  failed: %s", exc)
            out = []
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        reactions = out[0]["reactions"] if out else []
        logging.info("  → %d reactions in %.2fs", len(reactions), elapsed)

        if args.skip_empty and not reactions:
            continue

        results.append({
            "paragraph_id": idx,
            "text": para,
            "reactions": reactions,
            "time_sec": round(elapsed, 3),
        })

    # Summary stats
    n_reactions = sum(len(r["reactions"]) for r in results)
    logging.info("Done: %d paragraphs, %d reactions, total %.1fs (%.2fs/para avg)",
                 len(results), n_reactions, total_time,
                 total_time / len(paragraphs) if paragraphs else 0)

    output = json.dumps({
        "source": pdf_path,
        "paragraphs": results,
        "stats": {
            "n_paragraphs": len(results),
            "n_reactions": n_reactions,
            "total_time_sec": round(total_time, 2),
            "avg_time_per_para_sec": round(total_time / max(len(paragraphs), 1), 3),
        },
    }, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        logging.info("Saved → %s", out_path)
    else:
        print(output)


if __name__ == "__main__":
    main()
