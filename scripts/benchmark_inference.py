#!/usr/bin/env python3
"""
Benchmark: ReactionMiner LLaMA-2-7b+LoRA vs BERT/SciBERT inference speed.

Measures wall-clock ms/paragraph on the same set of real chemistry paragraphs.
This is the key thesis result: distilled BERT is N× faster than the teacher LLaMA.

Usage (from workspace root):
  Science/agent-venv/bin/python Science/scripts/benchmark_inference.py \
      --rm-output  Science/data/annotations/rm_ao1c00906.json \
      --bert-models Science/models/bert_rm_001_v2/best_model \
                    Science/models/bert_rm_002_v2/best_model \
      --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# ── Offline mode — all models already cached ─────────────────────────────────
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

_SCRIPTS_DIR    = Path(__file__).resolve().parent
_SCIENCE_DIR    = _SCRIPTS_DIR.parent
_RM_DIR         = _SCIENCE_DIR / "parser" / "ReactionMiner"
_WORKSPACE_ROOT = _SCIENCE_DIR.parent

for p in [str(_WORKSPACE_ROOT), str(_RM_DIR), str(_RM_DIR / "extraction")]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_paragraphs(rm_json: Path, n: int, min_chars: int = 80) -> list[str]:
    """Load substantial paragraphs from a ReactionMiner output JSON."""
    data = json.loads(rm_json.read_text())
    paras = data["paragraphs"] if isinstance(data, dict) else data
    texts = [p["text"] for p in paras if len(p.get("text", "")) >= min_chars]
    if len(texts) < n:
        print(f"[WARN] Only {len(texts)} paragraphs ≥{min_chars} chars (wanted {n})")
    return texts[:n]


def bench_bert(paragraphs: list[str], model_dir: str,
               n_warmup: int = 3) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for BERT inference on given paragraphs."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"  [{Path(model_dir).parent.name}] device={device}, "
          f"params={sum(p.numel() for p in model.parameters())//1_000_000}M")

    def _infer(text: str) -> float:
        tokens = text.split()
        t0 = time.perf_counter()
        enc = tokenizer(
            [tokens], is_split_into_words=True,
            truncation=True, max_length=512,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)
        return (time.perf_counter() - t0) * 1000

    # Warmup
    for p in paragraphs[:n_warmup]:
        _infer(p)

    times = [_infer(p) for p in paragraphs]
    return mean(times), (stdev(times) if len(times) > 1 else 0.0)


def bench_llama(paragraphs: list[str],
                n_warmup: int = 1) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for LLaMA-2-7b+LoRA inference."""
    from extractor import ReactionExtractor  # from _RM_DIR/extraction/
    print("  Loading ReactionExtractor (LLaMA-2-7b+LoRA)…")
    ext = ReactionExtractor("7b")

    def _infer(text: str) -> float:
        t0 = time.perf_counter()
        ext.extract([text])
        return (time.perf_counter() - t0) * 1000

    # Warmup
    for p in paragraphs[:n_warmup]:
        _infer(p)

    times = [_infer(p) for p in paragraphs]
    return mean(times), (stdev(times) if len(times) > 1 else 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm-output",   required=True, type=Path,
                    help="ReactionMiner output JSON to take paragraphs from")
    ap.add_argument("--bert-models", nargs="+", required=True,
                    help="One or more BERT model dirs (best_model/)")
    ap.add_argument("--n-samples",   type=int, default=20)
    ap.add_argument("--skip-llama",  action="store_true",
                    help="Skip LLaMA benchmark (BERT only)")
    args = ap.parse_args()

    paras = load_paragraphs(args.rm_output, args.n_samples)
    n = len(paras)
    avg_chars = sum(len(p) for p in paras) // max(n, 1)
    print(f"\n[INFO] Paragraphs: {n}  avg length: {avg_chars} chars\n")

    results: list[dict] = []

    # ── BERT models ───────────────────────────────────────────────────────────
    for model_dir in args.bert_models:
        name = Path(model_dir).parent.name  # e.g. bert_rm_002_v2
        print(f"[BENCH] {name}")
        try:
            m, s = bench_bert(paras, model_dir)
            results.append({"name": name, "mean_ms": round(m, 1),
                             "std_ms": round(s, 1), "type": "bert"})
            print(f"  → {m:.1f} ± {s:.1f} ms/para\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

    # ── LLaMA ────────────────────────────────────────────────────────────────
    if not args.skip_llama:
        print("[BENCH] ReactionMiner LLaMA-2-7b+LoRA")
        try:
            m, s = bench_llama(paras)
            results.append({"name": "LLaMA-2-7b+LoRA", "mean_ms": round(m, 1),
                             "std_ms": round(s, 1), "type": "llama"})
            print(f"  → {m:.1f} ± {s:.1f} ms/para\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

    # ── Table ─────────────────────────────────────────────────────────────────
    if not results:
        print("No results to show.")
        return

    llama_mean = next((r["mean_ms"] for r in results if r["type"] == "llama"), None)

    W = 28
    print("=" * 65)
    print(f"{'Model':<{W}} {'ms/para (mean±std)':>20}  {'Speedup':>10}")
    print("-" * 65)
    for r in results:
        speedup_str = "1.0×" if r["type"] == "llama" else (
            f"{llama_mean / r['mean_ms']:.0f}×" if llama_mean else "—")
        print(f"{r['name']:<{W}} {r['mean_ms']:>8.1f} ± {r['std_ms']:<8.1f}  {speedup_str:>10}")
    print("=" * 65)

    if llama_mean and len(results) > 1:
        for r in results:
            if r["type"] == "bert":
                sx = llama_mean / r["mean_ms"]
                print(f"\n{r['name']}: {llama_mean/1000:.1f}s → {r['mean_ms']:.0f}ms  "
                      f"({sx:.0f}× faster, {r['mean_ms']/1000:.3f}s/para)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = _SCIENCE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "benchmark_llama_vs_bert.json"
    out.write_text(json.dumps(
        {"n_samples": n, "avg_chars": avg_chars, "results": results},
        indent=2))
    print(f"\n[INFO] Saved → {out}")


if __name__ == "__main__":
    main()
