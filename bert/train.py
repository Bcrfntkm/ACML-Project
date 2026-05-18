#!/usr/bin/env python3
"""
Fine-tune BERT (or SciBERT / BioBERT) for token classification (NER / property roles).

Reads JSONL produced by convert_to_bio.py.

Usage
-----
# With a config file:
python Science/bert/train.py Science/bert/configs/bert_002.yaml

# Override specific parameters:
python Science/bert/train.py Science/bert/configs/bert_001.yaml \\
    --train data/training/train.jsonl \\
    --val   data/training/val.jsonl  \\
    --output-dir models/bert_001_run2 \\
    --epochs 5

Requirements
------------
pip install transformers datasets seqeval torch pyyaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCIENCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCIENCE_DIR.parent))


def _check_deps() -> None:
    missing = []
    for pkg in ["transformers", "datasets", "seqeval", "torch", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != "yaml" else "pyyaml")
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)


_check_deps()

import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    SEQEVAL_OK = True
except ImportError:
    SEQEVAL_OK = False
    print("[WARN] seqeval not found — per-label F1 will not be computed", file=sys.stderr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_label_map(labels_path: Path | None, samples: list[dict]) -> tuple[list[str], dict, dict]:
    """Build label list from file or infer from data."""
    if labels_path and labels_path.exists():
        label_list = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        all_labels = {l for s in samples for l in s["labels"]}
        label_list = ["O"] + sorted(l for l in all_labels if l != "O")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    return label_list, label2id, id2label


# ---------------------------------------------------------------------------
# Tokenisation & label alignment
# ---------------------------------------------------------------------------

def tokenize_and_align(
    examples: dict,
    tokenizer,
    label2id: dict,
    max_length: int = 512,
) -> dict:
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_label_ids = []
    for i, word_labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # special tokens: [CLS], [SEP], padding
            elif word_id != prev_word_id:
                # first subtoken of a word → assign the label
                label_ids.append(label2id.get(word_labels[word_id], 0))
            else:
                # subsequent subtokens → ignore in loss
                label_ids.append(-100)
            prev_word_id = word_id
        all_label_ids.append(label_ids)

    tokenized["labels"] = all_label_ids
    return tokenized


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: dict):
    def compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(-1)

        true_labels  = [[id2label[l] for l in row if l != -100] for row in labels]
        pred_labels  = [
            [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        if SEQEVAL_OK:
            return {
                "f1":       f1_score(true_labels,       pred_labels),
                "precision": precision_score(true_labels, pred_labels),
                "recall":   recall_score(true_labels,    pred_labels),
            }
        else:
            # fallback: token-level accuracy
            correct = sum(p == t for pr, tr in zip(pred_labels, true_labels)
                          for p, t in zip(pr, tr))
            total   = sum(len(tr) for tr in true_labels)
            return {"accuracy": correct / total if total else 0.0}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, cli_overrides: dict) -> None:
    # Merge CLI overrides into config
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    set_seed(cfg.get("seed", 42))

    train_path  = Path(cfg["train"])
    val_path    = Path(cfg.get("val", ""))
    output_dir  = Path(cfg["output_dir"])
    model_name  = cfg["model_name"]
    max_length  = int(cfg.get("max_length",  512))
    epochs      = int(cfg.get("epochs",      3))
    batch_size  = int(cfg.get("batch_size",  16))
    lr          = float(cfg.get("learning_rate", 2e-5))
    warmup      = float(cfg.get("warmup_ratio",  0.1))
    labels_path = Path(cfg["labels"]) if cfg.get("labels") else None

    print(f"\n{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Train      : {train_path}")
    print(f"  Val        : {val_path}")
    print(f"  Output dir : {output_dir}")
    print(f"  Epochs     : {epochs}  BS={batch_size}  LR={lr}")
    print(f"{'='*60}\n")

    # Load data
    train_samples = load_jsonl(train_path)
    val_samples   = load_jsonl(val_path) if val_path.exists() else []

    label_list, label2id, id2label = build_label_map(labels_path, train_samples + val_samples)
    print(f"[INFO] Labels ({len(label_list)}): {label_list}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def preprocess(samples):
        return tokenize_and_align(samples, tokenizer, label2id, max_length)

    train_ds = Dataset.from_list(train_samples).map(
        preprocess, batched=True, remove_columns=["tokens", "labels"]
    )
    val_ds = (
        Dataset.from_list(val_samples).map(
            preprocess, batched=True, remove_columns=["tokens", "labels"]
        )
        if val_samples else None
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    # Class weights: inverse frequency to fight O-token dominance
    all_labels_flat = [l for s in train_samples for l in s["labels"]]
    from collections import Counter
    counts = Counter(all_labels_flat)
    total = sum(counts.values())
    weights = torch.ones(len(label_list))
    for lbl, idx in label2id.items():
        freq = counts.get(lbl, 0) / total if total else 1.0
        weights[idx] = 1.0 / (freq + 1e-6) if freq > 0 else 1.0
    weights = weights / weights.mean()  # normalise so loss scale stays ~same

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(
                weight=weights.to(logits.device),
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    total_steps = max(1, (len(train_samples) // batch_size) * epochs)
    warmup_steps = max(1, int(total_steps * warmup))

    training_args = TrainingArguments(
        output_dir            = str(output_dir),
        num_train_epochs      = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate         = lr,
        warmup_steps          = warmup_steps,
        weight_decay          = 0.01,
        eval_strategy         = "epoch" if val_ds else "no",
        save_strategy         = "no",          # save only at the end
        load_best_model_at_end = False,        # avoid loading epoch-1 on F1=0
        logging_steps         = 10,
        report_to             = "none",
        fp16                  = torch.cuda.is_available(),
        seed                  = cfg.get("seed", 42),
    )

    trainer = WeightedTrainer(
        model              = model,
        args               = training_args,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        processing_class   = tokenizer,
        data_collator      = collator,
        compute_metrics    = make_compute_metrics(id2label),
    )

    print("[INFO] Starting training...")
    trainer.train()

    # Save final model + label map
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    (output_dir / "best_model" / "labels.json").write_text(
        json.dumps(label_list, indent=2), encoding="utf-8"
    )
    print(f"\n[INFO] Model saved → {output_dir / 'best_model'}")

    # Final eval report
    if val_ds and SEQEVAL_OK:
        preds = trainer.predict(val_ds)
        y_pred = preds.predictions.argmax(-1)
        y_true = preds.label_ids
        true_seq = [[id2label[l] for l in row if l != -100] for row in y_true]
        pred_seq = [
            [id2label[p] for p, l in zip(pr, tr) if l != -100]
            for pr, tr in zip(y_pred, y_true)
        ]
        print("\n" + classification_report(true_seq, pred_seq, digits=4))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for chemical NER / property extraction")
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--train",       default=None, help="Override train JSONL path")
    parser.add_argument("--val",         default=None, help="Override val JSONL path")
    parser.add_argument("--output-dir",  default=None, dest="output_dir")
    parser.add_argument("--model",       default=None, dest="model_name")
    parser.add_argument("--epochs",      default=None, type=int)
    parser.add_argument("--batch-size",  default=None, type=int, dest="batch_size")
    args = parser.parse_args(argv)

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cli_overrides = {
        "train":       args.train,
        "val":         args.val,
        "output_dir":  args.output_dir,
        "model_name":  args.model_name,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
    }

    train(cfg, cli_overrides)


if __name__ == "__main__":
    main()
