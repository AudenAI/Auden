#!/bin/bash
# LALM evaluation script (single GPU).
#
# Usage (from examples/lalm/):
#   bash scripts/evaluate.sh
#
# Override checkpoint via env vars, e.g.:
#   ITER=5000 bash scripts/evaluate.sh
#   EPOCH=3   bash scripts/evaluate.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONPATH=/apdcephfs_cq10/share_1603164/user/yiwenyshao/lhotse:${PYTHONPATH:-}

# ── GPU ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ── Experiment ────────────────────────────────────────────────────────────────
exp_name="lalm_8gpus_aut_qwen2_ds4"
exp_dir="./exp/$exp_name"

# ── Checkpoint ────────────────────────────────────────────────────────────────
# Set ITER or EPOCH to select a checkpoint; leave both 0 to specify a filename.
iter=16000
epoch=${EPOCH:-0}
# filename=""   # uncomment to load a specific file, e.g. "checkpoint-5000.pt"

# ── Decoding ──────────────────────────────────────────────────────────────────
decoding_method="greedy_search"   # greedy_search | beam_search
max_new_tokens=200
dtype="fp16"

# ── Data ──────────────────────────────────────────────────────────────────────
test_data_config="configs/valid_data_config.yaml"
max_duration=1000

echo "========================================================"
echo "  Exp:      ${exp_dir}"
echo "  Iter:     ${iter}  |  Epoch: ${epoch}"
echo "  Decoding: ${decoding_method}"
echo "========================================================"

python evaluate.py \
    exp_dir="$exp_dir" \
    checkpoint.iter="$iter" \
    decoding_method="$decoding_method" \
    max_new_tokens="$max_new_tokens" \
    dtype="$dtype" \
    data.test_data_config="$test_data_config" \
    data.max_duration="$max_duration"
