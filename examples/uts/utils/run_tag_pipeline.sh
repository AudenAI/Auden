#!/bin/bash
# ============================================================
# UTS Tag Construction Pipeline  (Steps 2–4 of the full pipeline)
#
#   Step 2: Extract tags from captions  (extract_tags.py)
#   Step 3: Build TF-IDF label system   (build_labelsys.py)
#   Step 4: Filter samples per vocab    (filter_tags.py)
#
# Prerequisites:
#   - A captions jsonl file produced by the captioner in vllm_qwen/
#     Each line: {"id": ..., "caption": ...}
#
# Usage:
#   bash run_tag_pipeline.sh
# ============================================================

set -e

# Output of the captioner
CAPTIONS_JSONL="path/to/captions.jsonl"

# LLM for tag extraction (Qwen2.5-7B-Instruct or similar)
TAG_MODEL_PATH="path/to/Qwen2.5-7B-Instruct"

# GPUs to use for tag extraction
TAG_GPUS="0,1,2,3"
TENSOR_PARALLEL_SIZE=4
BATCH_SIZE=128

# Vocabulary sizes to build
VOCAB_SIZES="800 1000 1500 2000 3000"

# output paths
TAGS_JSONL="./output/tags.jsonl"       # Step 2 output: captions + raw parsed tags
LABEL_DIR="./label"                    # Step 3 output: one txt per vocab size
FILTERED_DIR="./filtered"             # Step 4 output: one jsonl per vocab size

# ==================== Script Setup ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  UTS Tag Construction Pipeline"
echo "============================================"
echo ""
echo "Config:"
echo "  Captions JSONL      : $CAPTIONS_JSONL"
echo "  Tag model           : $TAG_MODEL_PATH"
echo "  GPUs                : $TAG_GPUS  (TP=$TENSOR_PARALLEL_SIZE)"
echo "  Vocab sizes         : $VOCAB_SIZES"
echo "  Tags output         : $TAGS_JSONL"
echo "  Label dir           : $LABEL_DIR"
echo "  Filtered output dir : $FILTERED_DIR"
echo ""

if [ ! -f "$CAPTIONS_JSONL" ]; then
    echo "Error: Captions file not found: $CAPTIONS_JSONL"
    exit 1
fi

mkdir -p "$(dirname "$TAGS_JSONL")" "$LABEL_DIR" "$FILTERED_DIR"

# ==================== Step 2: Extract Tags ==================

echo "--------------------------------------------"
echo "Step 2: Extracting tags from captions"
echo "--------------------------------------------"

python3 "$SCRIPT_DIR/extract_tags.py" \
    --model_path "$TAG_MODEL_PATH" \
    --input      "$CAPTIONS_JSONL" \
    --output     "$TAGS_JSONL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --gpus       "$TAG_GPUS"

echo ""
echo "Step 2 complete → $TAGS_JSONL"
echo ""

# ==================== Step 3: Build Label System ============

echo "--------------------------------------------"
echo "Step 3: Building TF-IDF label system"
echo "--------------------------------------------"

python3 "$SCRIPT_DIR/build_label_system.py" \
    --input       "$TAGS_JSONL" \
    --output_dir  "$LABEL_DIR" \
    --vocab_sizes $VOCAB_SIZES \
    --show_top    30

echo ""
echo "Step 3 complete. Label files:"
ls -lh "$LABEL_DIR"
echo ""

# ==================== Step 4: Filter Tags ===================

echo "--------------------------------------------"
echo "Step 4: Filtering tags against label systems"
echo "--------------------------------------------"

python3 "$SCRIPT_DIR/filter_tags.py" \
    --input        "$TAGS_JSONL" \
    --label_dir    "$LABEL_DIR" \
    --label_pattern "label_*.txt" \
    --output_dir   "$FILTERED_DIR"

echo ""
echo "Step 4 complete. Filtered files:"
ls -lh "$FILTERED_DIR"
echo ""

echo "============================================"
echo "Pipeline finished."
echo "============================================"