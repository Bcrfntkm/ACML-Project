#!/usr/bin/env python3
"""
Convert ReactionMiner JSON output to BIO JSONL for BERT fine-tuning.

Input:  Science/data/annotations/rm_<paper>.json  (from run_reactionminer.py)
        or a merged JSON of multiple papers.

Output: train.jsonl / val.jsonl / test.jsonl  (BIO token-level labels)

Label schema (ReactionMiner roles):
  PRODUCT   — reaction product
  REACTANT  — reactant / starting material
  CATALYST  — catalyst
  SOLVENT   — reaction solvent
  TEMP      — temperature
  TIME      — reaction time
  YIELD     — yield value
  COND      — other condition

Usage:
  python Science/scripts/convert_reactions_to_bio.py \
      --input  Science/data/annotations/rm_merged.json \
      --output Science/data/training/rm_ner \
      --split
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Canonical label names for ReactionMiner roles
_ROLE_MAP = {
    "product":          "PRODUCT",
    "reactant":         "REACTANT",
    "catalyst":         "CATALYST",
    "solvent":          "SOLVENT",
    "temperature":      "TEMP",
    "time":             "TIME",
    "yield":            "YIELD",
    "othercondition":   "COND",
    "other condition":  "COND",
    "other":            "COND",
    # Additional roles output by LLaMA beyond the paper's 8-role schema
    "atmosphere":       "COND",
    "workup reagents":  "COND",
    "workup":           "COND",
    "reaction type":    "COND",
    "procedure":        "COND",
    "additive":         "COND",
    "base":             "COND",
    "acid":             "COND",
    "ligand":           "COND",
}


def _normalise_role(key: str) -> str:
    return _ROLE_MAP.get(key.strip().lower(), "COND")


def _find_spans(text: str, value: str) -> list[tuple[int, int]]:
    """Find all occurrences of value in text (exact), return list of (start, end)."""
    spans = []
    start = 0
    while True:
        idx = text.find(value, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(value)))
        start = idx + 1
    return spans


def _normalise_text(s: str) -> str:
    """Normalise whitespace around degree/special symbols for matching."""
    s = re.sub(r'\s*(°)\s*', r'°', s)          # "60 °C" → "60°C"
    s = re.sub(r'\s*(·)\s*', r'·', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _find_spans_fuzzy(text: str, value: str, min_ratio: float = 0.75) -> list[tuple[int, int]]:
    """Find the best-matching span for value in text using multiple strategies.

    1. Exact match
    2. Case-insensitive exact
    3. Whitespace-normalised (handles "60 °C" vs "60°C")
    4. Each comma-separated part independently
    5. Token-level Jaccard sliding window (catches normalised names)
    """
    val = value.strip()
    if not val:
        return []

    # 1. Exact
    spans = _find_spans(text, val)
    if spans:
        return spans

    # 2. Case-insensitive
    lo = text.lower().find(val.lower())
    if lo != -1:
        return [(lo, lo + len(val))]

    # 3. Normalised whitespace around symbols
    val_norm = _normalise_text(val)
    text_norm = _normalise_text(text)
    if val_norm != val:
        lo = text_norm.lower().find(val_norm.lower())
        if lo != -1:
            # Map back to original text offset (approximate — good enough for BIO)
            return [(lo, lo + len(val_norm))]

    # 4. Try each comma-separated part (e.g. "copper acetate, methanol" → two spans)
    parts = [p.strip() for p in val.split(',') if len(p.strip()) > 3]
    if len(parts) > 1:
        found = []
        for part in parts:
            lo = text.lower().find(part.lower())
            if lo != -1:
                found.append((lo, lo + len(part)))
        if found:
            return found

    # 5. Token-level Jaccard sliding window
    val_tokens = set(re.findall(r'\w+', val.lower()))
    # Skip very generic single-token values that will match anything
    if len(val_tokens) <= 1:
        return []

    words = list(re.finditer(r'\S+', text))
    if not words:
        return []

    win = max(1, min(len(val_tokens) + 2, len(words)))
    best_ratio, best_span = 0.0, None

    for i in range(len(words) - win + 1):
        chunk = text[words[i].start():words[i + win - 1].end()].lower()
        ct = set(re.findall(r'\w+', chunk))
        if not ct:
            continue
        ratio = len(val_tokens & ct) / len(val_tokens | ct)
        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (words[i].start(), words[i + win - 1].end())

    if best_ratio >= min_ratio and best_span:
        return [best_span]
    return []


# ── Tokenisation: whitespace-based, preserve char offsets ────────────────────

def _tokenize(text: str) -> tuple[list[str], list[int], list[int]]:
    """Return (tokens, starts, ends) using whitespace split with char offsets."""
    tokens, starts, ends = [], [], []
    for m in re.finditer(r'\S+', text):
        tokens.append(m.group())
        starts.append(m.start())
        ends.append(m.end())
    return tokens, starts, ends


def _bio_labels(text: str, annotations: list[dict]) -> list[str]:
    """Assign BIO labels to whitespace tokens given char-span annotations."""
    tokens, t_starts, t_ends = _tokenize(text)
    labels = ["O"] * len(tokens)

    # Sort annotations by start so B- is correctly placed
    for ann in sorted(annotations, key=lambda a: a["start"]):
        span_start, span_end, label = ann["start"], ann["end"], ann["label"]
        first = True
        for i, (ts, te) in enumerate(zip(t_starts, t_ends)):
            if te <= span_start or ts >= span_end:
                continue  # token outside span
            # token overlaps with annotation span
            if first:
                labels[i] = f"B-{label}"
                first = False
            else:
                labels[i] = f"I-{label}"

    return labels


def _paragraph_to_bio(para: dict, min_tokens: int = 8, fuzzy: bool = True) -> dict | None:
    """Convert one paragraph record (with reactions) to a BIO sample.

    min_tokens: skip very short segments (headers, captions, etc.)
    fuzzy:      use fuzzy span matching when exact search fails
    """
    text = para["text"]
    reactions = para.get("reactions", [])
    if not reactions:
        return None

    # Drop very short segments — they're usually section headers or figure captions
    tokens, _, _ = _tokenize(text)
    if len(tokens) < min_tokens:
        return None

    # Build span annotations from ReactionMiner structured records
    annotations: list[dict] = []
    seen_spans: set[tuple[int, int, str]] = set()
    # Key roles that must be grounded in the text; reactions where none are found
    # are likely hallucinated by LLaMA from context outside the segment.
    KEY_ROLES = {"product", "reactant", "catalyst", "solvent"}

    for reaction in reactions:
        # Hallucination filter: at least one key-role value must appear in the text
        grounded = False
        for rk, rv in reaction.items():
            if rk.strip().lower() not in KEY_ROLES:
                continue
            if isinstance(rv, str) and rv.strip():
                parts = [p.strip() for p in rv.split(',')]
                for part in parts:
                    if len(part) > 3 and text.lower().find(part.lower()) != -1:
                        grounded = True
                        break
            if grounded:
                break
        if not grounded:
            continue  # skip hallucinated reaction

        for role_key, value in reaction.items():
            if not isinstance(value, str) or not value.strip():
                continue
            label = _normalise_role(role_key)
            finder = _find_spans_fuzzy if fuzzy else _find_spans
            spans = finder(text, value.strip())
            for s, e in spans:
                key = (s, e, label)
                if key not in seen_spans:
                    seen_spans.add(key)
                    annotations.append({"start": s, "end": e, "label": label})

    if not annotations:
        return None

    tokens, _, _ = _tokenize(text)
    bio = _bio_labels(text, annotations)

    return {"tokens": tokens, "labels": bio}


# ── Merge multiple rm_*.json files ──────────────────────────────────────────

def load_rm_files(paths: list[Path]) -> list[dict]:
    """Load and merge paragraph records from one or more rm_*.json files."""
    records = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "paragraphs" in data:
            records.extend(data["paragraphs"])
        elif isinstance(data, list):
            records.extend(data)
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, nargs="+",
                    help="rm_*.json file(s) or a merged JSON")
    ap.add_argument("--output", required=True, help="Output dir for JSONL files")
    ap.add_argument("--split",  action="store_true", help="80/10/10 train/val/test split")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    input_paths = [Path(p) for p in args.input]
    raw_records = load_rm_files(input_paths)
    print(f"[INFO] Loaded {len(raw_records)} paragraph records")

    # Convert to BIO
    samples = []
    for rec in raw_records:
        bio = _paragraph_to_bio(rec)
        if bio:
            samples.append(bio)
    print(f"[INFO] {len(samples)} BIO samples after conversion")

    # Infer label list
    all_labels_flat = [l for s in samples for l in s["labels"]]
    counts = Counter(all_labels_flat)
    entity_labels = sorted({l[2:] for l in counts if l.startswith("B-")})
    label_list = ["O"] + [f"B-{l}" for l in entity_labels] + [f"I-{l}" for l in entity_labels]
    print(f"[INFO] Labels ({len(label_list)}): {label_list}")

    # Non-O stats
    total = sum(counts.values())
    non_o = total - counts.get("O", 0)
    print(f"  Total tokens: {total:,}  Non-O: {non_o:,} ({100*non_o/total:.1f}%)")
    for l in entity_labels:
        b = counts.get(f"B-{l}", 0)
        i = counts.get(f"I-{l}", 0)
        print(f"  {l:12s}: B={b} I={i}")

    def _save(name: str, data: list[dict]):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[INFO] {len(data):5d} → {path}")

    if args.split and len(samples) >= 10:
        n = len(samples)
        n_val  = max(1, n // 10)
        n_test = max(1, n // 10)
        n_train = n - n_val - n_test
        _save("train.jsonl", samples[:n_train])
        _save("val.jsonl",   samples[n_train:n_train + n_val])
        _save("test.jsonl",  samples[n_train + n_val:])
    else:
        _save("all.jsonl", samples)

    (out_dir / "labels.json").write_text(
        json.dumps(label_list, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Labels → {out_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
