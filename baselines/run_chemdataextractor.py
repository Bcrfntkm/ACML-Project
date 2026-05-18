#!/usr/bin/env python3
"""
Baseline: ChemDataExtractor 2 on a chemistry PDF.

ChemDataExtractor is a specialized toolkit for automatic extraction of chemical
information from scientific literature. It uses a combination of rule-based and
ML methods, and is specifically designed for chemistry text.

Reference: Mavracic et al., J. Chem. Inf. Model. 2021, 61(9), 4280–4289

Installation (separate from agent-venv due to dependency conflicts):
    pip install chemdataextractor2
    python -m chemdataextractor download   # downloads NER models and lexicon

Usage:
    python Science/baselines/run_chemdataextractor.py path/to/paper.pdf
    python Science/baselines/run_chemdataextractor.py path/to/paper.pdf -o Science/data/annotations/cde_out.json
"""

import argparse
import json
import sys
from pathlib import Path

# CDE entity type → our label schema
_CDE_TO_LABEL = {
    "Compound":      "COMPOUND",
    "Chemical":      "COMPOUND",
    "ChemicalName":  "COMPOUND",
    "IUPAC":         "IUPAC",
    "Formula":       "COMPOUND",
    "Smiles":        "COMPOUND",
    "CAS":           "CAS",
    "Temperature":   "PROPERTY",
    "Pressure":      "PROPERTY",
    "Yield":         "PROPERTY",
    "Time":          "PROPERTY",
    "Quantity":      "PROPERTY",
}


def extract_text_pdfplumber(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] pip install pdfplumber", file=sys.stderr)
        sys.exit(1)
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def run_cde_on_text(text: str) -> list[dict]:
    """Run ChemDataExtractor on plain text."""
    try:
        from chemdataextractor import Document as CdeDocument
        from chemdataextractor.doc import Paragraph
    except ImportError:
        print("[ERROR] pip install chemdataextractor2 && python -m chemdataextractor download",
              file=sys.stderr)
        sys.exit(1)

    # Split into paragraphs and process each so offsets are manageable
    paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 20]
    annotations = []
    cursor = 0

    for para in paragraphs:
        # Advance cursor to where this paragraph starts in original text
        para_start = text.find(para, cursor)
        if para_start == -1:
            para_start = cursor
        cursor = para_start + len(para)

        try:
            doc = CdeDocument([Paragraph(para)])
            for record in doc.records:
                # Each record is a chemical entity; get its name
                names = getattr(record, "names", None) or []
                for name in names:
                    if not name:
                        continue
                    idx = para.find(name)
                    if idx == -1:
                        continue
                    annotations.append({
                        "text":  name,
                        "label": "COMPOUND",
                        "start": para_start + idx,
                        "end":   para_start + idx + len(name),
                    })
        except Exception as e:
            print(f"[WARN] CDE failed on paragraph: {e}", file=sys.stderr)
            continue

    # Deduplicate
    seen = set()
    deduped = []
    for ann in annotations:
        key = (ann["text"], ann["label"], ann["start"], ann["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(ann)
    return deduped


def run_cde_on_pdf(pdf_path: str) -> list[dict]:
    """Try CDE's native PDF reader first, fall back to pdfplumber."""
    try:
        from chemdataextractor import Document as CdeDocument
        with open(pdf_path, "rb") as f:
            doc = CdeDocument.from_file(f, readers=[])   # PDF reader
        # Extract named chemical mentions with char offsets
        annotations = []
        for el in doc.elements:
            for ent in getattr(el, "cems", []):
                label = _CDE_TO_LABEL.get(type(ent).__name__, "COMPOUND")
                annotations.append({
                    "text":  ent.text,
                    "label": label,
                    "start": ent.start,
                    "end":   ent.end,
                })
        return annotations
    except Exception as e:
        print(f"[WARN] CDE native PDF reader failed ({e}), falling back to pdfplumber", file=sys.stderr)
        text = extract_text_pdfplumber(pdf_path)
        return run_cde_on_text(text)


def parse_args():
    p = argparse.ArgumentParser(description="ChemDataExtractor 2 NER baseline.")
    p.add_argument("pdf_path", metavar="PDF")
    p.add_argument("-o", "--output", metavar="FILE", default=None)
    p.add_argument("--text-only", action="store_true",
                   help="Skip CDE native PDF reader, use pdfplumber + text mode only")
    return p.parse_args()


def main():
    args = parse_args()
    pdf_path = str(Path(args.pdf_path).resolve())

    print(f"[INFO] Running ChemDataExtractor on: {pdf_path}", file=sys.stderr)

    if args.text_only:
        text = extract_text_pdfplumber(pdf_path)
        annotations = run_cde_on_text(text)
    else:
        annotations = run_cde_on_pdf(pdf_path)

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
