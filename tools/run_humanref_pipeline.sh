#!/usr/bin/env bash
# Run from repository root. Each run must use a new OUT directory.
set -euo pipefail
ANN=${ANN:-data/HumanRef/annotations.jsonl}
IMAGES=${IMAGES:-data/HumanRef/images}
UNI=${UNI:-checkpoints/wedetect_base_uni.pth}
REF=${REF:-checkpoints/WeDetect-Ref-4B}
OUT=${OUT:-results/humanref_uni_ref_k100}
GPUS=${GPUS:-8}
K=${K:-100}
LIMIT=${LIMIT:-0}
THRESHOLD=${THRESHOLD:-0.35}
ABC=${ABC:-0}
if [ -e "$OUT" ]; then
    echo "Output exists: $OUT. Set OUT to a new directory." >&2
    exit 1
fi
mkdir -p "$OUT"
python tools/test_humanref_pipeline.py
if [ "$ABC" = 1 ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node="$GPUS" \
        tools/humanref_pipeline.py ref --candidate-source dataset \
        --annotations "$ANN" --images "$IMAGES" --checkpoint "$REF" \
        --output "$OUT/ref_A" --num-proposals "$K" --limit "$LIMIT" \
        2>&1 | tee "$OUT/ref_A.log"
fi
torchrun --standalone --nnodes=1 --nproc-per-node="$GPUS" \
    tools/humanref_pipeline.py uni --annotations "$ANN" --images "$IMAGES" \
    --checkpoint "$UNI" --output "$OUT/uni" --num-proposals "$K" --limit "$LIMIT" \
    2>&1 | tee "$OUT/uni.log"
torchrun --standalone --nnodes=1 --nproc-per-node="$GPUS" \
    tools/humanref_pipeline.py ref --annotations "$ANN" --images "$IMAGES" \
    --checkpoint "$REF" --uni-dir "$OUT/uni" --output "$OUT/ref" \
    --num-proposals "$K" --limit "$LIMIT" \
    2>&1 | tee "$OUT/ref.log"
if [ "$ABC" = 1 ]; then
    python tools/humanref_pipeline.py compare --annotations "$ANN" \
        --a-ref-dir "$OUT/ref_A" --ref-dir "$OUT/ref" --output "$OUT/ABC" \
        --limit "$LIMIT" --score-threshold "$THRESHOLD" --nms-iou 0.7 \
        2>&1 | tee "$OUT/analysis.log"
else
  python tools/humanref_pipeline.py analyze --annotations "$ANN" \
    --ref-dir "$OUT/ref" --output "$OUT/analysis" --limit "$LIMIT" \
    --score-threshold "$THRESHOLD" 2>&1 | tee "$OUT/analysis.log"
fi
