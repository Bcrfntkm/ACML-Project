#!/usr/bin/env python3
"""
Full ReactionMiner pipeline — exactly as in the paper (EMNLP 2023):
  Step 1: parseFile       (SymbolScraper Java PDF parser)
  Step 2: TopicSegmentor  (allenai-specter + C99 segmentation)
  Step 3: ReactionExtractor (LLaMA-2-7b + LoRA MingZhong/reaction-miner-7b-lora)

Saves structured reactions to JSON, records per-paragraph inference time.

Usage (from ReactionMiner directory):
  cd "Science/parser/ReactionMiner"
  JAVA_HOME=$(brew --prefix openjdk@21) \
  HF_TOKEN=hf_... \
  ../../../agent-venv/bin/python ../../../scripts/run_reactionminer_full.py \
      ../../data/pdfs/ao1c00906.pdf \
      -o ../../../data/annotations/rm_ao1c00906.json -v

  # Batch:
  for pdf in ../../data/pdfs/*.pdf; do
      stem=$(basename "$pdf" .pdf)
      JAVA_HOME=$(brew --prefix openjdk@21) HF_TOKEN=hf_... \
      ../../../agent-venv/bin/python ../../../scripts/run_reactionminer_full.py \
          "$pdf" -o "../../../data/annotations/rm_${stem}.json" --skip-empty &
  done; wait
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent          # Science/scripts/
_SCIENCE_DIR  = _SCRIPT_DIR.parent                       # Science/
_RM_DIR       = _SCIENCE_DIR / "parser" / "ReactionMiner"  # ReactionMiner root

# Add ReactionMiner to sys.path so its imports resolve
sys.path.insert(0, str(_RM_DIR))
sys.path.insert(0, str(_SCIENCE_DIR.parent))             # workspace root

# Java 21 via Homebrew — SymbolScraper needs JAVA_HOME
_java_candidates = [
    os.environ.get("JAVA_HOME", ""),
    "/opt/homebrew/opt/openjdk@21",
    "/opt/homebrew/opt/openjdk",
    "/usr/local/opt/openjdk@21",
]
for _j in _java_candidates:
    if _j and Path(_j, "bin", "java").exists():
        os.environ["JAVA_HOME"] = _j
        os.environ["PATH"] = str(Path(_j, "bin")) + ":" + os.environ.get("PATH", "")
        break

# Force offline mode — all models already cached locally
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

if os.environ.get("HF_TOKEN"):
    try:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    except Exception:
        pass


def _parse_args():
    p = argparse.ArgumentParser(description="Full ReactionMiner pipeline on a PDF")
    p.add_argument("pdf_path", metavar="PDF")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--skip-empty", action="store_true",
                   help="Skip paragraphs with no extracted reactions")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s", stream=sys.stderr,
    )

    pdf_path = str(Path(args.pdf_path).resolve())
    logging.info("PDF: %s", pdf_path)

    # ── Step 1: PDF → text (SymbolScraper) ───────────────────────────────────
    logging.info("Step 1: PDF-to-Text (SymbolScraper)...")
    # Must run from RM dir so generalParser finds its relative paths
    orig_cwd = os.getcwd()
    os.chdir(str(_RM_DIR))
    try:
        sys.path.insert(0, str(_RM_DIR / "pdf2text"))
        from pdf2text.generalParser import parseFile
        result = parseFile(pdf_path)
    finally:
        os.chdir(orig_cwd)

    if not isinstance(result, dict):
        logging.error("SymbolScraper failed (returned %r). Check Java/Maven setup.", result)
        sys.exit(1)

    full_text  = result.get("fullText", "")
    paragraphs = result.get("contents", [])
    logging.info("  fullText: %d chars, %d paragraphs", len(full_text), len(paragraphs))

    if not paragraphs:
        logging.error("No paragraphs extracted. Check SymbolScraper setup.")
        sys.exit(1)

    # ── Step 2: Topic segmentation ────────────────────────────────────────────
    logging.info("Step 2: TopicSegmentor (allenai-specter)...")
    os.chdir(str(_RM_DIR))
    try:
        from segmentation.segmentor import TopicSegmentor
        # Fix: default device is cuda:0, use cpu on Mac (MPS causes issues with sentence-transformers)
        segmentor = TopicSegmentor(device="cpu")
        if "_SI" in pdf_path:
            seg_texts = segmentor.segment_si(paragraphs)
        else:
            seg_texts = segmentor.segment(paragraphs)
    finally:
        os.chdir(orig_cwd)

    logging.info("  Segments after filtering: %d", len(seg_texts))

    if not seg_texts:
        logging.warning("No reaction-related segments found.")
        if not args.skip_empty:
            json.dump({"source": pdf_path, "paragraphs": [], "stats": {}},
                      open(args.output, "w") if args.output else sys.stdout,
                      indent=2)
        return

    # ── Step 3: Reaction extraction (LLaMA-2-7b + LoRA) ─────────────────────
    logging.info("Step 3: ReactionExtractor (LLaMA-2-7b + LoRA)...")
    sys.path.insert(0, str(_RM_DIR / "extraction"))
    from extraction.extractor import ReactionExtractor
    extractor = ReactionExtractor("7b")

    results = []
    total_time = 0.0

    for idx, text in enumerate(seg_texts):
        logging.info("  Segment %d/%d (%d chars)", idx + 1, len(seg_texts), len(text))
        t0 = time.perf_counter()
        try:
            out = extractor.extract([text])
        except Exception as exc:
            logging.warning("  failed: %s", exc)
            out = []
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        reactions = out[0]["reactions"] if out else []
        logging.info("  → %d reactions in %.2fs", len(reactions), elapsed)

        if args.skip_empty and not reactions:
            continue

        results.append({
            "paragraph_id": idx,
            "text": text,
            "reactions": reactions,
            "time_sec": round(elapsed, 3),
        })

    n_reactions = sum(len(r["reactions"]) for r in results)
    avg_time = total_time / len(seg_texts) if seg_texts else 0
    logging.info("Done: %d segments → %d reactions, total %.1fs (%.2fs/seg avg)",
                 len(results), n_reactions, total_time, avg_time)

    output_data = {
        "source": pdf_path,
        "paragraphs": results,
        "stats": {
            "n_segments":    len(seg_texts),
            "n_with_results": len(results),
            "n_reactions":   n_reactions,
            "total_time_sec": round(total_time, 2),
            "avg_time_per_seg_sec": round(avg_time, 3),
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        logging.info("Saved → %s", out_path)
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
