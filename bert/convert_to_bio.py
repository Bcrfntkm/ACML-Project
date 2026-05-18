#!/usr/bin/env python3
"""
Convert LLM annotations to BIO token-level labels for BERT fine-tuning.

Supports two task schemas:

  --task ner
      8-label NER schema (COMPOUND, IUPAC, CAS, REACTION, PROPERTY, ELEMENT, UNIT, OTHER)
      Input: paragraph-level JSON  [{text, annotations: [{label, start, end}]}, ...]
              produced by  scripts/export_paragraphs.py  (or agent directly)

  --task property
      5-role property schema (SUBSTANCE, PROP_NAME, PROP_VALUE, CONDITIONS, METHOD)
      Input: property records JSON  [{paragraph_id, text, records: [{substance, ...}]}, ...]
              produced by  agent/graph_property.py

Output: JSONL where each line is one sentence window:
  {"tokens": ["The", "yield", "was", "64%"],
   "labels": ["O", "B-PROPERTY", "O", "B-PROPERTY"]}

Usage
-----
# NER task
python Science/bert/convert_to_bio.py \\
    --input  data/annotations/copper_acetate_v2.json \\
    --output data/training/copper_ner_bio.jsonl \\
    --task   ner

# Property task
python Science/bert/convert_to_bio.py \\
    --input  data/annotations/copper_properties.json \\
    --output data/training/copper_prop_bio.jsonl \\
    --task   property

# Split into train/val/test (80/10/10 by document)
python Science/bert/convert_to_bio.py \\
    --input  data/annotations/copper_ner_bio.jsonl \\
    --split  --output data/training/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

NER_LABELS = [
    "O",
    "B-COMPOUND",  "I-COMPOUND",
    "B-IUPAC",     "I-IUPAC",
    "B-CAS",       "I-CAS",
    "B-REACTION",  "I-REACTION",
    "B-PROPERTY",  "I-PROPERTY",
    "B-ELEMENT",   "I-ELEMENT",
    "B-UNIT",      "I-UNIT",
    "B-OTHER",     "I-OTHER",
]

PROPERTY_LABELS = [
    "O",
    "B-SUBSTANCE",  "I-SUBSTANCE",
    "B-PROP_NAME",  "I-PROP_NAME",
    "B-PROP_VALUE", "I-PROP_VALUE",
    "B-CONDITIONS", "I-CONDITIONS",
    "B-METHOD",     "I-METHOD",
]

LABEL_SETS = {"ner": NER_LABELS, "property": PROPERTY_LABELS}


# ---------------------------------------------------------------------------
# Tokenisation (whitespace + punctuation split, keeps offsets)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return list of (token, start, end) using simple whitespace+punct split."""
    tokens = []
    for m in re.finditer(r"\S+", text):
        word = m.group()
        start = m.start()
        # split off leading/trailing punctuation except hyphens, dots in formulas
        sub_start = start
        for i, ch in enumerate(word):
            if ch.isalnum() or ch in "-_.()[]{}+=#@%°'\"":
                break
            sub_start += 1
        sub_end = start + len(word)
        for i in range(len(word) - 1, -1, -1):
            if word[i].isalnum() or word[i] in "-_.()[]{}+=#@%°'\"":
                break
            sub_end -= 1
        if sub_start >= sub_end:
            tokens.append((word, start, start + len(word)))
        else:
            if sub_start > start:
                tokens.append((word[:sub_start - start], start, sub_start))
            tokens.append((word[sub_start - start:sub_end - start], sub_start, sub_end))
            if sub_end < start + len(word):
                tokens.append((word[sub_end - start:], sub_end, start + len(word)))
    return [(t, s, e) for t, s, e in tokens if t.strip()]


# ---------------------------------------------------------------------------
# Span → BIO labels alignment
# ---------------------------------------------------------------------------

def _spans_to_bio(
    tokens: list[tuple[str, int, int]],
    spans: list[tuple[int, int, str]],  # (start, end, label)
) -> list[str]:
    """Assign BIO labels to tokens given character-level spans."""
    labels = ["O"] * len(tokens)
    # Sort spans by length descending so longer spans take priority
    sorted_spans = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)

    for span_start, span_end, label in sorted_spans:
        label_upper = label.upper()
        first = True
        for i, (tok, tok_start, tok_end) in enumerate(tokens):
            # overlap: token intersects span
            if tok_start < span_end and tok_end > span_start:
                if labels[i] == "O":  # don't overwrite already-labeled tokens
                    prefix = "B" if first else "I"
                    labels[i] = f"{prefix}-{label_upper}"
                first = False

    return labels


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _convert_ner_paragraph(para: dict) -> dict | None:
    """Convert one paragraph dict {text, annotations} → {tokens, labels}."""
    text = para.get("text", "").strip()
    annotations = para.get("annotations", [])
    if not text:
        return None

    tokens_with_offsets = _tokenize(text)
    if not tokens_with_offsets:
        return None

    spans = [
        (int(a["start"]), int(a["end"]), a["label"])
        for a in annotations
        if "start" in a and "end" in a and "label" in a
    ]

    tokens = [t for t, _, _ in tokens_with_offsets]
    labels = _spans_to_bio(tokens_with_offsets, spans)

    return {"tokens": tokens, "labels": labels}


def _property_records_to_spans(
    text: str, records: list[dict]
) -> list[tuple[int, int, str]]:
    """Locate role values in text and return char spans with role labels."""
    ROLE_TO_LABEL = {
        "substance":          "SUBSTANCE",
        "property_name":      "PROP_NAME",
        "property_value":     "PROP_VALUE",
        "conditions":         "CONDITIONS",
        "measurement_method": "METHOD",
    }
    spans = []
    for rec in records:
        for role, label in ROLE_TO_LABEL.items():
            val = rec.get(role, "").strip()
            if not val:
                continue
            # find verbatim in text
            idx = text.find(val)
            if idx != -1:
                spans.append((idx, idx + len(val), label))
    return spans


def _convert_property_paragraph(para: dict) -> dict | None:
    text = para.get("text", "").strip()
    records = para.get("records", [])
    if not text:
        return None

    tokens_with_offsets = _tokenize(text)
    if not tokens_with_offsets:
        return None

    spans = _property_records_to_spans(text, records)
    tokens = [t for t, _, _ in tokens_with_offsets]
    labels = _spans_to_bio(tokens_with_offsets, spans)

    return {"tokens": tokens, "labels": labels}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_input(path: Path, task: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if task == "ner":
        # Accept flat list [{text, label, start, end}]  → wrap into paragraph dicts
        # OR already paragraph-level [{text, annotations}]
        if isinstance(data, list) and data and "text" in data[0] and "annotations" in data[0]:
            return data
        if isinstance(data, list) and data and "label" in data[0]:
            # flat annotation list — need raw_text to reconstruct paragraphs
            raise ValueError(
                "Flat annotation list detected. Re-run the pipeline with paragraph-level "
                "output, or use scripts/export_paragraphs.py to attach source text."
            )
        if isinstance(data, dict) and "annotations" in data:
            # single {raw_text, annotations} dict
            raw = data.get("raw_text", "")
            if raw:
                return [{"text": raw, "annotations": data["annotations"]}]
        raise ValueError("Unrecognised NER input format.")

    else:  # property
        if isinstance(data, list) and data and "records" in data[0]:
            return data
        raise ValueError("Unrecognised property input format.")


def convert(paragraphs: list[dict], task: str) -> list[dict]:
    results = []
    converter = _convert_ner_paragraph if task == "ner" else _convert_property_paragraph
    for para in paragraphs:
        result = converter(para)
        if result and any(l != "O" for l in result["labels"]):
            results.append(result)
        elif result:
            results.append(result)  # keep O-only paragraphs too (needed for class balance)
    return results


def split_dataset(
    samples: list[dict], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> tuple[list[dict], list[dict], list[dict]]:
    """Simple sequential split (preserves document order)."""
    n = len(samples)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return samples[:n_train], samples[n_train:n_train + n_val], samples[n_train + n_val:]


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(samples)} samples → {path}")


def print_stats(samples: list[dict], task: str) -> None:
    from collections import Counter
    all_labels = [l for s in samples for l in s["labels"]]
    b_counts = Counter(l for l in all_labels if l.startswith("B-"))
    total = len(all_labels)
    non_o = sum(1 for l in all_labels if l != "O")
    print(f"  Samples   : {len(samples)}")
    print(f"  Tokens    : {total}")
    print(f"  Non-O     : {non_o} ({100*non_o/total:.1f}%)")
    print(f"  Entities  :")
    for label, count in sorted(b_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<20} {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert LLM annotations to BIO token labels for BERT."
    )
    parser.add_argument("--input",  required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSONL file, or directory if --split is used")
    parser.add_argument("--task",   choices=["ner", "property"], default="ner")
    parser.add_argument("--split",  action="store_true",
                        help="Split into train/val/test and save to --output directory")
    args = parser.parse_args(argv)

    print(f"[INFO] Loading  {args.input}  (task={args.task})")
    paragraphs = load_input(args.input, args.task)
    print(f"[INFO] {len(paragraphs)} paragraphs loaded")

    samples = convert(paragraphs, args.task)
    print(f"[INFO] {len(samples)} samples after conversion")
    print_stats(samples, args.task)

    if args.split:
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        train, val, test = split_dataset(samples)
        save_jsonl(train, out_dir / "train.jsonl")
        save_jsonl(val,   out_dir / "val.jsonl")
        save_jsonl(test,  out_dir / "test.jsonl")
        label_path = out_dir / "labels.json"
        label_path.write_text(
            json.dumps(LABEL_SETS[args.task], indent=2), encoding="utf-8")
        print(f"[INFO] Labels   → {label_path}")
    else:
        save_jsonl(samples, args.output)


if __name__ == "__main__":
    main()
