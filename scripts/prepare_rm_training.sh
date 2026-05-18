#!/usr/bin/env bash
# After batch_reactionminer.sh completes:
# 1. Merge all rm_*.json outputs
# 2. Convert to BIO JSONL (train/val/test split)
# 3. Print label distribution
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
SCIENCE="$WORKSPACE/Science"
ANNO_DIR="$SCIENCE/data/annotations"
TRAIN_DIR="$SCIENCE/data/training/rm_ner"
PYTHON="$SCIENCE/agent-venv/bin/python"

echo "=== Step 1: Listing available rm_*.json files ==="
rm_files=()
for f in "$ANNO_DIR"/rm_*.json; do
    n_seg=$(python3 -c "import json; d=json.load(open('$f')); print(d['stats'].get('n_segments',0))" 2>/dev/null || echo 0)
    n_rx=$(python3 -c "import json; d=json.load(open('$f')); print(d['stats'].get('n_reactions',0))" 2>/dev/null || echo 0)
    if [ "$n_rx" -gt 0 ]; then
        echo "  OK  $(basename $f)  — $n_rx reactions / $n_seg segments"
        rm_files+=("$f")
    else
        echo "  SKIP $(basename $f)  — 0 reactions"
    fi
done

echo ""
echo "Files with data: ${#rm_files[@]}"

if [ ${#rm_files[@]} -eq 0 ]; then
    echo "ERROR: No rm_*.json files with reactions found."
    exit 1
fi

echo ""
echo "=== Step 2: Convert to BIO JSONL ==="
"$PYTHON" "$SCIENCE/scripts/convert_reactions_to_bio.py" \
    --input "${rm_files[@]}" \
    --output "$TRAIN_DIR" \
    --split

echo ""
echo "=== Step 3: Training data ready at $TRAIN_DIR ==="
ls -lh "$TRAIN_DIR/"

echo ""
echo "=== Next: Train BERT ==="
echo "  cd $WORKSPACE && $PYTHON $SCIENCE/bert/train.py $SCIENCE/bert/configs/bert_rm_001.yaml"
echo "  cd $WORKSPACE && $PYTHON $SCIENCE/bert/train.py $SCIENCE/bert/configs/bert_rm_002.yaml"
