#!/usr/bin/env python3
"""
Compare annotation outputs from different baselines or against a reference.

Metrics:
  - Exact match:   text, label, start, end all match
  - Partial match: char spans overlap AND label matches (more lenient)

Usage:
    # Compare two files
    python Science/baselines/compare.py \\
        --pred Science/data/annotations/gliner_out.json \\
        --ref  Science/data/annotations/copper_acetate_v2.json

    # Compare multiple files side-by-side (no ref, just overlap stats)
    python Science/baselines/compare.py \\
        --files Science/data/annotations/spacy_out.json \\
                Science/data/annotations/gliner_out.json \\
                Science/data/annotations/cde_out.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def spans_overlap(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def exact_key(ann: dict) -> tuple:
    return (ann["text"].strip().lower(), ann["label"], ann["start"], ann["end"])


def partial_key(ann: dict) -> tuple:
    """Used to bucket predictions by label for partial matching."""
    return ann["label"]


def compute_metrics(pred: list[dict], ref: list[dict]) -> dict:
    ref_exact = set(exact_key(a) for a in ref)
    pred_exact = set(exact_key(a) for a in pred)

    tp_exact = len(pred_exact & ref_exact)
    fp_exact = len(pred_exact - ref_exact)
    fn_exact = len(ref_exact - pred_exact)

    prec_e = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) else 0.0
    rec_e  = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) else 0.0
    f1_e   = 2 * prec_e * rec_e / (prec_e + rec_e) if (prec_e + rec_e) else 0.0

    # Partial match: for each pred span, check if any ref span overlaps with same label
    tp_partial = 0
    for p in pred:
        for r in ref:
            if p["label"] == r["label"] and spans_overlap(p, r):
                tp_partial += 1
                break

    prec_p = tp_partial / len(pred) if pred else 0.0
    rec_p  = tp_partial / len(ref)  if ref  else 0.0
    f1_p   = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) else 0.0

    return {
        "pred_count": len(pred),
        "ref_count":  len(ref),
        "exact":  {"tp": tp_exact,  "fp": fp_exact,  "fn": fn_exact,
                   "precision": round(prec_e, 3), "recall": round(rec_e, 3), "f1": round(f1_e, 3)},
        "partial": {"tp": tp_partial,
                    "precision": round(prec_p, 3), "recall": round(rec_p, 3), "f1": round(f1_p, 3)},
    }


def per_label_metrics(pred: list[dict], ref: list[dict]) -> dict[str, dict]:
    labels = sorted(set(a["label"] for a in pred) | set(a["label"] for a in ref))
    result = {}
    for label in labels:
        p_sub = [a for a in pred if a["label"] == label]
        r_sub = [a for a in ref  if a["label"] == label]
        result[label] = compute_metrics(p_sub, r_sub)
    return result


def print_comparison(pred_path: str, ref_path: str) -> None:
    pred = load(pred_path)
    ref  = load(ref_path)

    print(f"\n{'='*60}")
    print(f"PRED: {pred_path}  ({len(pred)} annotations)")
    print(f"REF:  {ref_path}  ({len(ref)} annotations)")
    print(f"{'='*60}")

    overall = compute_metrics(pred, ref)
    print(f"\n{'OVERALL':30s}  {'EXACT':>20s}  {'PARTIAL':>20s}")
    print(f"{'':-<30s}  {'':-<20s}  {'':-<20s}")
    e, p = overall["exact"], overall["partial"]
    print(f"{'':30s}  P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f}  "
          f"P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f}")

    print(f"\n{'BY LABEL':30s}  {'EXACT F1':>10s}  {'PARTIAL F1':>12s}  PRED  REF")
    print(f"{'':-<30s}  {'':-<10s}  {'':-<12s}  {'':-<4s}  {'':-<4s}")
    for label, m in sorted(per_label_metrics(pred, ref).items()):
        print(f"{label:30s}  {m['exact']['f1']:>10.3f}  {m['partial']['f1']:>12.3f}  "
              f"{m['pred_count']:>4d}  {m['ref_count']:>4d}")


def print_overlap_matrix(file_paths: list[str]) -> None:
    """Show pairwise exact-match overlap between multiple annotation files."""
    datasets = [(p, set(exact_key(a) for a in load(p))) for p in file_paths]
    names = [Path(p).stem for p in file_paths]
    col_w = max(len(n) for n in names) + 2

    print(f"\n{'Exact-match overlap (Jaccard)':}")
    header = f"{'':20s}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))

    for i, (_, si) in enumerate(datasets):
        row = f"{names[i]:20s}"
        for j, (_, sj) in enumerate(datasets):
            if i == j:
                row += f"{'1.000':>{col_w}}"
            else:
                inter = len(si & sj)
                union = len(si | sj)
                jaccard = inter / union if union else 0.0
                row += f"{jaccard:>{col_w}.3f}"
        print(row)

    print()
    for name, (_, s) in zip(names, datasets):
        print(f"  {name}: {len(s)} unique spans")


def parse_args():
    p = argparse.ArgumentParser(description="Compare NER annotation outputs.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred", metavar="FILE", help="Prediction file")
    g.add_argument("--files", nargs="+", metavar="FILE", help="Multiple files for pairwise overlap")
    p.add_argument("--ref", metavar="FILE", help="Reference file (used with --pred)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.pred:
        if not args.ref:
            print("ERROR: --ref is required when using --pred", flush=True)
            raise SystemExit(1)
        print_comparison(args.pred, args.ref)
    else:
        print_overlap_matrix(args.files)


if __name__ == "__main__":
    main()
