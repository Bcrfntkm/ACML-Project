#!/usr/bin/env bash
# auto_retrain.sh — wait for ReactionMiner batch to finish, then:
#   1. Regenerate BIO training data (with improved fuzzy converter)
#   2. Train bert-base and SciBERT in parallel
#   3. Report final metrics
#
# Usage:
#   nohup bash Science/scripts/auto_retrain.sh > /tmp/auto_retrain.log 2>&1 &

set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
SCIENCE="$WORKSPACE/Science"
PYTHON="$SCIENCE/agent-venv/bin/python"
BATCH_LOG="/tmp/rm_batch3.log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: wait for batch ────────────────────────────────────────────────────
log "Waiting for ReactionMiner batch to finish (monitoring $BATCH_LOG)..."
while true; do
    if grep -q "^Batch complete" "$BATCH_LOG" 2>/dev/null; then
        log "Batch finished!"
        grep "Batch complete" "$BATCH_LOG"
        break
    fi
    # Also check if no reactionminer process is running and batch log exists
    if [ -f "$BATCH_LOG" ] && ! pgrep -f "run_reactionminer_full" > /dev/null 2>&1; then
        log "No active reactionminer process — assuming batch done."
        break
    fi
    sleep 60
done

# ── Step 2: regenerate BIO data ───────────────────────────────────────────────
log "Regenerating BIO training data..."

ANNO="$SCIENCE/data/annotations"
TRAIN_DIR="$SCIENCE/data/training/rm_ner_v2"

# Collect all rm_*.json with reactions
rm_files=()
n_total_rx=0
for f in "$ANNO"/rm_*.json; do
    n_rx=$("$PYTHON" -c "import json; d=json.load(open('$f')); print(d['stats'].get('n_reactions',0))" 2>/dev/null || echo 0)
    if [ "$n_rx" -gt 0 ]; then
        rm_files+=("$f")
        n_total_rx=$((n_total_rx + n_rx))
        log "  OK: $(basename $f) — $n_rx reactions"
    fi
done

log "Papers with reactions: ${#rm_files[@]}, total reactions: $n_total_rx"

"$PYTHON" "$SCIENCE/scripts/convert_reactions_to_bio.py" \
    --input "${rm_files[@]}" \
    --output "$TRAIN_DIR" \
    --split

log "BIO data ready at $TRAIN_DIR"
ls -lh "$TRAIN_DIR/"

# ── Step 3: update configs to point at new data ───────────────────────────────
for cfg in bert_rm_001 bert_rm_002; do
    cfg_file="$SCIENCE/bert/configs/${cfg}.yaml"
    sed -i '' "s|data/training/rm_ner/|data/training/rm_ner_v2/|g" "$cfg_file" 2>/dev/null || \
    sed -i    "s|data/training/rm_ner/|data/training/rm_ner_v2/|g" "$cfg_file"
    # Update output dir to v2
    sed -i '' "s|models/${cfg}$|models/${cfg}_v2|g" "$cfg_file" 2>/dev/null || \
    sed -i    "s|models/${cfg}$|models/${cfg}_v2|g" "$cfg_file"
    log "Updated config: $cfg_file"
done

# ── Step 4: train both models ────────────────────────────────────────────────
log "Starting bert-base training (bert_rm_001_v2)..."
cd "$SCIENCE"
"$PYTHON" bert/train.py bert/configs/bert_rm_001.yaml \
    > /tmp/bert_rm_001_v2.log 2>&1 &
PID1=$!
log "  PID bert-base: $PID1"

log "Starting SciBERT training (bert_rm_002_v2)..."
"$PYTHON" bert/train.py bert/configs/bert_rm_002.yaml \
    > /tmp/bert_rm_002_v2.log 2>&1 &
PID2=$!
log "  PID SciBERT:   $PID2"

# ── Step 5: wait and report ───────────────────────────────────────────────────
log "Waiting for both training jobs..."
wait $PID1 && log "bert-base done" || log "bert-base FAILED"
wait $PID2 && log "SciBERT done"   || log "SciBERT FAILED"

log "=== FINAL METRICS ==="
echo ""
echo "--- bert-base ---"
grep -E "micro avg|precision|recall|f1-score|epoch.*10" /tmp/bert_rm_001_v2.log 2>/dev/null | tail -10
echo ""
echo "--- SciBERT ---"
grep -E "micro avg|precision|recall|f1-score|epoch.*10" /tmp/bert_rm_002_v2.log 2>/dev/null | tail -10

log "All done. Check /tmp/bert_rm_001_v2.log and /tmp/bert_rm_002_v2.log for full results."
