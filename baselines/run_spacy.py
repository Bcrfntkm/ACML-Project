#!/usr/bin/env python3
"""
Baseline: spaCy NER (en_core_web_sm) on a chemistry PDF.

Extracts text via pdfplumber, runs spaCy NER, maps entity types to our label schema.
Already available in agent-venv — no extra installation needed.

Usage:
    python Science/baselines/run_spacy.py path/to/paper.pdf
    python Science/baselines/run_spacy.py path/to/paper.pdf -o Science/data/annotations/spacy_out.json
"""

import argparse
import json
import sys
from pathlib import Path

# spaCy entity type → our label
_SPACY_TO_LABEL = {
    "CHEMICAL":   "COMPOUND",   # if using sci/chem models
    "CHEBI":      "COMPOUND",
    "ORG":        "COMPOUND",   # sometimes catches compound names
    "PRODUCT":    "COMPOUND",
    "GPE":        None,         # skip
    "PERSON":     None,
    "DATE":       None,
    "TIME":       None,
    "PERCENT":    "PROPERTY",
    "QUANTITY":   "PROPERTY",
    "CARDINAL":   None,
    "MONEY":      None,
    "LOC":        None,
    "FAC":        None,
    "WORK_OF_ART": None,
    "NORP":       None,
    "EVENT":      None,
    "LAW":        None,
    "LANGUAGE":   None,
    "ORDINAL":    None,
}

# Chemical keywords for heuristic post-labelling when spaCy misses them
_UNIT_SUFFIXES = ("°C", "°F", "K", "mol", "mmol", "mL", "L", "mg", "g", "kg",
                  "MHz", "ppm", "Hz", "nm", "μm", "bar", "atm", "Pa", "min", "h")


def extract_text(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] pip install pdfplumber", file=sys.stderr)
        sys.exit(1)
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def run_spacy(text: str) -> list[dict]:
    try:
        import spacy
    except ImportError:
        print("[ERROR] pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)
        sys.exit(1)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[ERROR] python -m spacy download en_core_web_sm", file=sys.stderr)
        sys.exit(1)

    # spaCy has a max doc length; chunk if needed
    chunk_size = 100_000
    annotations = []
    offset = 0

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        doc = nlp(chunk)
        for ent in doc.ents:
            label = _SPACY_TO_LABEL.get(ent.label_)
            if label is None:
                continue
            annotations.append({
                "text":  ent.text,
                "label": label,
                "start": offset + ent.start_char,
                "end":   offset + ent.end_char,
            })
        offset += len(chunk)

    return annotations


def parse_args():
    p = argparse.ArgumentParser(description="spaCy NER baseline for chemistry PDFs.")
    p.add_argument("pdf_path", metavar="PDF")
    p.add_argument("-o", "--output", metavar="FILE", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pdf_path = str(Path(args.pdf_path).resolve())

    print(f"[INFO] Extracting text: {pdf_path}", file=sys.stderr)
    text = extract_text(pdf_path)
    print(f"[INFO] Text length: {len(text)} chars", file=sys.stderr)

    print("[INFO] Running spaCy NER...", file=sys.stderr)
    annotations = run_spacy(text)
    print(f"[INFO] Annotations: {len(annotations)}", file=sys.stderr)

    output = json.dumps(annotations, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"[INFO] Saved to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
