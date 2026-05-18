#!/usr/bin/env bash
# Run full ReactionMiner pipeline sequentially on all pending PDFs.
# Usage: bash Science/scripts/batch_reactionminer.sh [PDF_DIR] [ANNO_DIR]
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
SCIENCE="$WORKSPACE/Science"
PDF_DIR="${1:-$SCIENCE/data/pdfs}"
ANNO_DIR="${2:-$SCIENCE/data/annotations}"
PYTHON="$SCIENCE/agent-venv/bin/python"
SCRIPT="$SCIENCE/scripts/run_reactionminer_full.py"

export HF_TOKEN="${HF_TOKEN:-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}"
export JAVA_HOME="${JAVA_HOME:-$(brew --prefix openjdk@21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21)}"
export PATH="$JAVA_HOME/bin:$PATH"

total=0; done_count=0; skip_count=0

for pdf in "$PDF_DIR"/*.pdf; do
    stem=$(basename "$pdf" .pdf)
    out="$ANNO_DIR/rm_${stem}.json"

    # Skip if already processed with full pipeline (n_reactions > 0 OR n_segments key exists)
    if [ -f "$out" ]; then
        has_segs=$(python3 -c "import json; d=json.load(open('$out')); print('yes' if 'n_segments' in d.get('stats',{}) else 'no')" 2>/dev/null || echo "no")
        if [ "$has_segs" = "yes" ]; then
            reactions=$(python3 -c "import json; d=json.load(open('$out')); print(d['stats'].get('n_reactions',0))" 2>/dev/null || echo "?")
            echo "[SKIP] $stem → already done ($reactions reactions)"
            ((skip_count++)) || true
            continue
        fi
    fi

    ((total++)) || true
    echo ""
    echo "========================================================"
    echo "[START] $stem  ($(date '+%H:%M:%S'))"
    echo "========================================================"

    log="/tmp/rm_${stem}.log"
    if "$PYTHON" "$SCRIPT" "$pdf" -o "$out" --skip-empty -v >"$log" 2>&1; then
        reactions=$(python3 -c "import json; d=json.load(open('$out')); print(d['stats'].get('n_reactions',0))" 2>/dev/null || echo "?")
        echo "[DONE] $stem → $reactions reactions  (log: $log)"
        ((done_count++)) || true
    else
        echo "[FAIL] $stem — see $log"
    fi
done

echo ""
echo "========================================================"
echo "Batch complete: $done_count processed, $skip_count skipped, $((total - done_count)) failed"
echo "========================================================"
