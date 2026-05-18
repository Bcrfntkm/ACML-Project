#!/usr/bin/env python3
"""
Evaluate a fine-tuned BERT model on a test JSONL file.

Outputs:
  - seqeval precision / recall / F1 per label + micro/macro averages
  - Confusion-style breakdown (support per entity type)
  - Optional: saves predictions to JSON for inspection

Usage
-----
python Science/bert/evaluate.py \\
    --model  models/bert_002/best_model \\
    --test   data/training/test.jsonl \\
    --out    results/bert_002_eval.json

# Compare two models on the same test set:
python Science/bert/evaluate.py --model models/bert_001/best_model --test data/training/test.jsonl
python Science/bert/evaluate.py --model models/bert_002/best_model --test data/training/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCIENCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCIENCE_DIR.parent))


def _check_deps() -> None:
    missing = []
    for pkg in ["transformers", "torch", "seqeval"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing: {', '.join(missing)}  →  pip install {' '.join(missing)}")
        sys.exit(1)


_check_deps()

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_inference(
    model_dir: Path,
    samples: list[dict],
    batch_size: int = 16,
    max_length: int = 512,
) -> tuple[list[list[str]], list[list[str]]]:
    """Return (true_labels, pred_labels) for seqeval."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    id2label = model.config.id2label
    label2id = model.config.label2id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    true_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        token_lists = [s["tokens"] for s in batch]

        encoding = tokenizer(
            token_lists,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            output = model(**encoding)
        logits = output.logits.cpu()
        predictions = logits.argmax(-1).numpy()

        for j, sample in enumerate(batch):
            word_ids    = encoding.word_ids(j) if hasattr(encoding, "word_ids") else \
                          tokenizer(token_lists[j], is_split_into_words=True).word_ids()
            word_ids    = word_ids[:max_length]   # align with truncated predictions
            gold_labels = sample["labels"]

            true_seq, pred_seq = [], []
            seen_word_ids = set()
            for k, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen_word_ids:
                    continue
                seen_word_ids.add(word_id)
                if word_id < len(gold_labels):
                    true_seq.append(gold_labels[word_id])
                    pred_seq.append(id2label[int(predictions[j][k])])

            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)

    return true_seqs, pred_seqs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(true_seqs, pred_seqs, model_dir: Path) -> dict:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Model: {model_dir}")
    print(f"  Samples: {len(true_seqs)}")
    print(f"  Tokens:  {sum(len(s) for s in true_seqs)}")
    print(sep)
    print(classification_report(true_seqs, pred_seqs, digits=4))
    micro_f1  = f1_score(true_seqs,        pred_seqs)
    micro_p   = precision_score(true_seqs,  pred_seqs)
    micro_r   = recall_score(true_seqs,     pred_seqs)
    print(f"  Micro F1        : {micro_f1:.4f}")
    print(f"  Micro Precision : {micro_p:.4f}")
    print(f"  Micro Recall    : {micro_r:.4f}")
    print(sep)

    return {
        "model":   str(model_dir),
        "samples": len(true_seqs),
        "micro_f1":        round(micro_f1, 4),
        "micro_precision": round(micro_p,  4),
        "micro_recall":    round(micro_r,  4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BERT on test JSONL")
    parser.add_argument("--model",      required=True, type=Path, help="Path to saved model dir")
    parser.add_argument("--test",       required=True, type=Path, help="Test JSONL file")
    parser.add_argument("--out",        default=None,  type=Path, help="Save results JSON here")
    parser.add_argument("--batch-size", default=16,    type=int)
    parser.add_argument("--max-length", default=512,   type=int)
    args = parser.parse_args(argv)

    print(f"[INFO] Loading test data: {args.test}")
    samples = load_jsonl(args.test)
    print(f"[INFO] {len(samples)} samples")

    print("[INFO] Running inference...")
    true_seqs, pred_seqs = run_inference(
        args.model, samples, args.batch_size, args.max_length
    )

    results = print_report(true_seqs, pred_seqs, args.model)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] Results saved → {args.out}")


if __name__ == "__main__":
    main()
