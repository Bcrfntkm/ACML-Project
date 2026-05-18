#!/usr/bin/env python3
"""
Baseline: GLiNER zero-shot NER on a chemistry PDF.

GLiNER is a generalist NER model that accepts arbitrary label sets without fine-tuning.
Reference: Zaratiana et al., arXiv:2311.08526 (2023)

Installation:
    pip install gliner

Models (choose one):
    urchade/gliner_mediumv2.1      — general English, good balance
    EmergentMethods/gliner_large_bio-v0.1  — biomedical domain, closer to chemistry

Usage:
    python Science/baselines/run_gliner.py path/to/paper.pdf
    python Science/baselines/run_gliner.py path/to/paper.pdf --model EmergentMethods/gliner_large_bio-v0.1
    python Science/baselines/run_gliner.py path/to/paper.pdf -o Science/data/annotations/gliner_out.json
"""

import argparse
import json
import sys
from pathlib import Path

# GLiNER labels → our schema
GLINER_LABELS = [
    "chemical compound",
    "IUPAC name",
    "CAS number",
    "chemical reaction",
    "physical property",
    "chemical element",
    "unit of measurement",
]

_GLINER_TO_LABEL = {
    "chemical compound":   "COMPOUND",
    "IUPAC name":          "IUPAC",
    "CAS number":          "CAS",
    "chemical reaction":   "REACTION",
    "physical property":   "PROPERTY",
    "chemical element":    "ELEMENT",
    "unit of measurement": "UNIT",
}

DEFAULT_MODEL = "urchade/gliner_mediumv2.1"
DEFAULT_THRESHOLD = 0.4
# GLiNER works on short spans; chunk text to avoid OOM
CHUNK_SIZE = 2000   # chars
CHUNK_OVERLAP = 200


def extract_text(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        print("[ERROR] pip install pdfplumber", file=sys.stderr)
        sys.exit(1)
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """Yield (chunk, start_offset) pairs."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append((text[start:end], start))
        if end == len(text):
            break
        start += size - overlap
    return chunks


def run_gliner(text: str, model_name: str, threshold: float) -> list[dict]:
    try:
        from gliner import GLiNER
    except ImportError:
        print("[ERROR] pip install gliner", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading GLiNER model: {model_name}", file=sys.stderr)
    model = GLiNER.from_pretrained(model_name)

    annotations = []
    seen = set()
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    for idx, (chunk, offset) in enumerate(chunks):
        print(f"[INFO] Chunk {idx+1}/{len(chunks)} ({len(chunk)} chars)", file=sys.stderr)
        entities = model.predict_entities(chunk, GLINER_LABELS, threshold=threshold)
        for ent in entities:
            label = _GLINER_TO_LABEL.get(ent["label"], "OTHER")
            abs_start = offset + ent["start"]
            abs_end   = offset + ent["end"]
            key = (ent["text"], label, abs_start, abs_end)
            if key in seen:
                continue
            seen.add(key)
            annotations.append({
                "text":  ent["text"],
                "label": label,
                "start": abs_start,
                "end":   abs_end,
                "score": round(ent["score"], 3),
            })

    return annotations


def parse_args():
    p = argparse.ArgumentParser(description="GLiNER zero-shot NER baseline.")
    p.add_argument("pdf_path", metavar="PDF")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"GLiNER model (default: {DEFAULT_MODEL})")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Confidence threshold (default: {DEFAULT_THRESHOLD})")
    p.add_argument("-o", "--output", metavar="FILE", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pdf_path = str(Path(args.pdf_path).resolve())

    print(f"[INFO] Extracting text: {pdf_path}", file=sys.stderr)
    text = extract_text(pdf_path)
    print(f"[INFO] Text length: {len(text)} chars", file=sys.stderr)

    annotations = run_gliner(text, args.model, args.threshold)
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
