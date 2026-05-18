#!/usr/bin/env python3
"""
Evaluation script for structured record extraction.

Computes precision / recall / F1 at:
  - record level  (full record match: all required roles correct)
  - role  level   (per-role micro and macro F1)

Supports two task schemas:
  --schema property   Substance / Property_Name / Property_Value / Conditions / Measurement_Method
  --schema reaction   Product / Reactant / Catalyst / Solvent / Temperature / Time / Yield / ...

Match strategies (--match):
  exact     lowercase + strip, full string equality
  partial   substring containment (pred ⊆ gold or gold ⊆ pred)
  token     F1 over whitespace-split tokens (bag-of-words, used in SQuAD-style)

Usage
-----
# Evaluate a model's JSON output against a ground-truth file:
python Science/scripts/evaluate_records.py \\
    --pred  data/annotations/pred_properties.json \\
    --gold  data/annotations/gold_properties.json \\
    --schema property --match exact

# Compare two systems (no gold, just agreement):
python Science/scripts/evaluate_records.py \\
    --pred  data/annotations/system_a.json \\
    --gold  data/annotations/system_b.json \\
    --schema reaction --match partial

Input JSON format
-----------------
A list of paragraph objects:
[
  {
    "paragraph_id": "paper_X_para_3",
    "text": "...",
    "records": [
      {
        "substance": "aspirin",
        "property_name": "melting point",
        "property_value": "135 °C"
      },
      ...
    ]
  },
  ...
]

For reaction schema, use role names: product, reactant, catalyst, solvent,
atmosphere, inhibitor, reaction_type, temperature, time, pressure, ph, speed,
vacuum_condition, light_condition, cooling_heating_condition, spectroscopic_data,
yield, procedure, workup_reagent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Role sets
# ---------------------------------------------------------------------------

PROPERTY_ROLES = frozenset({
    "substance", "property_name", "property_value", "conditions", "measurement_method",
})

REACTION_ROLES = frozenset({
    "product", "reactant", "catalyst", "solvent", "atmosphere", "inhibitor",
    "reaction_type", "temperature", "time", "pressure", "ph", "speed",
    "vacuum_condition", "light_condition", "cooling_heating_condition",
    "spectroscopic_data", "yield", "procedure", "workup_reagent",
})

# Key roles that must be present for a record to be considered "core-valid"
PROPERTY_KEY_ROLES = frozenset({"substance", "property_name", "property_value"})
REACTION_KEY_ROLES = frozenset({"product"})

SCHEMA_ROLES = {
    "property": PROPERTY_ROLES,
    "reaction": REACTION_ROLES,
}
SCHEMA_KEY_ROLES = {
    "property": PROPERTY_KEY_ROLES,
    "reaction": REACTION_KEY_ROLES,
}


# ---------------------------------------------------------------------------
# Normalization & matching
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation at boundaries."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _token_f1(pred: str, gold: str) -> float:
    """Bag-of-words token F1 (as in SQuAD evaluation)."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens))
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec  = common / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def role_match(pred_val: str, gold_val: str, strategy: str) -> bool:
    """Return True if pred_val matches gold_val under the given strategy."""
    p = _normalize(pred_val)
    g = _normalize(gold_val)
    if strategy == "exact":
        return p == g
    if strategy == "partial":
        return p in g or g in p
    if strategy == "token":
        return _token_f1(pred_val, gold_val) >= 0.5
    raise ValueError(f"Unknown match strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Record alignment
# ---------------------------------------------------------------------------

def _record_role_overlap(pred_rec: dict, gold_rec: dict,
                          roles: frozenset, strategy: str) -> int:
    """Count how many roles in pred_rec match gold_rec."""
    count = 0
    for role in roles:
        pv = pred_rec.get(role, "")
        gv = gold_rec.get(role, "")
        if pv and gv and role_match(pv, gv, strategy):
            count += 1
    return count


def _match_records(pred_records: list[dict], gold_records: list[dict],
                   roles: frozenset, key_roles: frozenset,
                   strategy: str) -> tuple[list[tuple], list[dict], list[dict]]:
    """
    Greedy matching: for each gold record find the best-matching pred record.
    Returns (matches, unmatched_pred, unmatched_gold).
    A match requires all key_roles to match.
    """
    used_pred = set()
    matches: list[tuple] = []  # (pred_idx, gold_idx, overlap_count)

    for gi, grec in enumerate(gold_records):
        best_pi, best_score = -1, -1
        for pi, prec in enumerate(pred_records):
            if pi in used_pred:
                continue
            # Key roles must ALL match for any alignment
            key_ok = all(
                role_match(prec.get(r, ""), grec.get(r, ""), strategy)
                for r in key_roles
                if grec.get(r)
            )
            if not key_ok:
                continue
            score = _record_role_overlap(prec, grec, roles, strategy)
            if score > best_score:
                best_score, best_pi = score, pi
        if best_pi >= 0:
            matches.append((best_pi, gi, best_score))
            used_pred.add(best_pi)

    unmatched_pred = [pred_records[i] for i in range(len(pred_records)) if i not in used_pred]
    matched_gold   = {gi for _, gi, _ in matches}
    unmatched_gold = [gold_records[i] for i in range(len(gold_records)) if i not in matched_gold]
    return matches, unmatched_pred, unmatched_gold


# ---------------------------------------------------------------------------
# Per-paragraph evaluation
# ---------------------------------------------------------------------------

def eval_paragraph(pred_records: list[dict], gold_records: list[dict],
                   roles: frozenset, key_roles: frozenset,
                   strategy: str) -> dict:
    """Return per-paragraph stats dict."""
    matches, unmatched_pred, unmatched_gold = _match_records(
        pred_records, gold_records, roles, key_roles, strategy)

    n_pred = len(pred_records)
    n_gold = len(gold_records)
    n_matched = len(matches)

    # Record-level precision / recall
    rec_prec = n_matched / n_pred if n_pred else 0.0
    rec_rec  = n_matched / n_gold if n_gold else 0.0
    rec_f1   = (2 * rec_prec * rec_rec / (rec_prec + rec_rec)
                if rec_prec + rec_rec > 0 else 0.0)

    # Role-level: for each matched pair, which roles are correct?
    role_tp: dict[str, int] = defaultdict(int)
    role_fp: dict[str, int] = defaultdict(int)
    role_fn: dict[str, int] = defaultdict(int)

    for pi, gi, _ in matches:
        prec = pred_records[pi]
        grec = gold_records[gi]
        for role in roles:
            pv = prec.get(role, "")
            gv = grec.get(role, "")
            if gv:
                if pv and role_match(pv, gv, strategy):
                    role_tp[role] += 1
                else:
                    role_fn[role] += 1
                    if pv:
                        role_fp[role] += 1
            elif pv:
                role_fp[role] += 1

    # Unmatched pred records contribute FP for every role they have
    for prec in unmatched_pred:
        for role in roles:
            if prec.get(role):
                role_fp[role] += 1

    # Unmatched gold records contribute FN for every role they have
    for grec in unmatched_gold:
        for role in roles:
            if grec.get(role):
                role_fn[role] += 1

    return {
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_matched": n_matched,
        "record_precision": round(rec_prec, 4),
        "record_recall":    round(rec_rec,  4),
        "record_f1":        round(rec_f1,   4),
        "role_tp": dict(role_tp),
        "role_fp": dict(role_fp),
        "role_fn": dict(role_fn),
    }


# ---------------------------------------------------------------------------
# Aggregate over all paragraphs
# ---------------------------------------------------------------------------

def aggregate(para_stats: list[dict], roles: frozenset) -> dict:
    total_pred    = sum(s["n_pred"]    for s in para_stats)
    total_gold    = sum(s["n_gold"]    for s in para_stats)
    total_matched = sum(s["n_matched"] for s in para_stats)

    rec_prec = total_matched / total_pred if total_pred else 0.0
    rec_rec  = total_matched / total_gold if total_gold else 0.0
    rec_f1   = (2 * rec_prec * rec_rec / (rec_prec + rec_rec)
                if rec_prec + rec_rec > 0 else 0.0)

    # Aggregate role-level TP/FP/FN
    all_tp: dict[str, int] = defaultdict(int)
    all_fp: dict[str, int] = defaultdict(int)
    all_fn: dict[str, int] = defaultdict(int)
    for s in para_stats:
        for role in roles:
            all_tp[role] += s["role_tp"].get(role, 0)
            all_fp[role] += s["role_fp"].get(role, 0)
            all_fn[role] += s["role_fn"].get(role, 0)

    role_metrics: dict[str, dict] = {}
    macro_f1_sum = 0.0
    active_roles = 0
    for role in sorted(roles):
        tp = all_tp[role]; fp = all_fp[role]; fn = all_fn[role]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        role_metrics[role] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "support":   tp + fn,
        }
        if tp + fn > 0:
            macro_f1_sum += f
            active_roles += 1

    macro_f1 = macro_f1_sum / active_roles if active_roles else 0.0

    # Micro role-level (all roles pooled)
    micro_tp = sum(all_tp.values()); micro_fp = sum(all_fp.values())
    micro_fn = sum(all_fn.values())
    micro_p  = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) else 0.0)

    return {
        "record_level": {
            "precision": round(rec_prec, 4),
            "recall":    round(rec_rec,  4),
            "f1":        round(rec_f1,   4),
            "n_pred":    total_pred,
            "n_gold":    total_gold,
            "n_matched": total_matched,
        },
        "role_level": {
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall":    round(micro_r, 4),
            "per_role": role_metrics,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_paragraphs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "paragraphs" in data:
        return data["paragraphs"]
    raise ValueError(f"Unexpected JSON structure in {path}. Expected a list of paragraph objects.")


def print_report(results: dict, schema: str, match: str) -> None:
    rl = results["record_level"]
    ro = results["role_level"]
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Schema : {schema}   Match strategy : {match}")
    print(sep)
    print("  RECORD-LEVEL")
    print(f"    Precision : {rl['precision']:.4f}")
    print(f"    Recall    : {rl['recall']:.4f}")
    print(f"    F1        : {rl['f1']:.4f}")
    print(f"    Pred/Gold/Matched : {rl['n_pred']} / {rl['n_gold']} / {rl['n_matched']}")
    print(f"\n  ROLE-LEVEL  (micro F1={ro['micro_f1']:.4f}  macro F1={ro['macro_f1']:.4f})")
    print(f"  {'Role':<30} {'P':>7} {'R':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for role, m in ro["per_role"].items():
        if m["support"] == 0:
            continue
        print(f"  {role:<30} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['support']:>8}")
    print(sep)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate record extraction: precision/recall/F1 at record and role level."
    )
    parser.add_argument("--pred",   required=True, type=Path, help="Predicted records JSON")
    parser.add_argument("--gold",   required=True, type=Path, help="Ground truth JSON")
    parser.add_argument("--schema", choices=["property", "reaction"], default="property",
                        help="Role schema (default: property)")
    parser.add_argument("--match",  choices=["exact", "partial", "token"], default="exact",
                        help="Match strategy (default: exact)")
    parser.add_argument("--out",    type=Path, default=None,
                        help="Save full results JSON to this path")
    args = parser.parse_args(argv)

    pred_paras = load_paragraphs(args.pred)
    gold_paras = load_paragraphs(args.gold)

    # Index gold by paragraph_id
    gold_index: dict[str, list[dict]] = {}
    for p in gold_paras:
        pid = p.get("paragraph_id") or p.get("id") or str(gold_paras.index(p))
        gold_index[pid] = p.get("records", [])

    roles    = SCHEMA_ROLES[args.schema]
    key_roles = SCHEMA_KEY_ROLES[args.schema]

    para_stats = []
    for p in pred_paras:
        pid = p.get("paragraph_id") or p.get("id") or str(pred_paras.index(p))
        pred_records = p.get("records", [])
        gold_records = gold_index.get(pid, [])
        stats = eval_paragraph(pred_records, gold_records, roles, key_roles, args.match)
        stats["paragraph_id"] = pid
        para_stats.append(stats)

    results = aggregate(para_stats, roles)
    results["config"] = {"schema": args.schema, "match": args.match,
                         "pred_file": str(args.pred), "gold_file": str(args.gold)}

    print_report(results, args.schema, args.match)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] Full results saved → {args.out}")


if __name__ == "__main__":
    main()
