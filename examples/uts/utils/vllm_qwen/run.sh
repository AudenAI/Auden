#!/bin/bash

# ==================== Configuration ====================
MODEL_PATH="/path/to/Qwen3-Omni-30B-A3B-Captioner"
SCP_FILE=
DATASET_SIZE=2048
OUT_DIR=

NUM_WORKERS=256      
QUEUE_MAX=2048     
MAX_SEQS=32  


# Auto-detect the directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# Python script path (same directory as run.sh)
SCRIPT_PATH="$SCRIPT_DIR/batch_inference_vllm.py"

PID_FILE="$SCRIPT_DIR/task.pid"


echo "=========================================="
echo "  vLLM Batch Inference Task Launcher"
echo "=========================================="
echo ""

# Check if a task is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Error: Task is already running (PID: $OLD_PID)"
        echo "   Please stop it first: bash stop.sh"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Check if the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if the SCP file exists
if [ ! -f "$SCP_FILE" ]; then
    echo "Error: SCP file not found at $SCP_FILE"
    exit 1
fi

# Check if the model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found at $MODEL_PATH"
    exit 1
fi

# Create output and log directories
mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"


# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. GPU not available?"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPU_COUNT GPU(s)"


echo ""
echo " Configuration:"
echo "  Model Path    : $MODEL_PATH"
echo "  SCP File      : $SCP_FILE"
echo "  Dataset Size  : $DATASET_SIZE"
echo "  Output Dir    : $OUT_DIR"
echo "  Log Dir       : $LOG_DIR"
echo ""
echo " Performance Parameters:"
echo "  NUM_WORKERS   : $NUM_WORKERS"
echo "  QUEUE_MAX     : $QUEUE_MAX"
echo "  MAX_SEQS      : $MAX_SEQS"
echo ""


TOTAL_LINES=$(wc -l < "$SCP_FILE")
echo "  Total Lines   : $TOTAL_LINES"
echo "  Estimated Batches: $((TOTAL_LINES / DATASET_SIZE + 1))"
echo ""


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/inference_${TIMESTAMP}.log"

# ==================== Launch Task ====================
echo "  Starting task in background..."
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""


nohup python3 "$SCRIPT_PATH" \
    --model_path "$MODEL_PATH" \
    --scp_file "$SCP_FILE" \
    --dataset_size "$DATASET_SIZE" \
    --out_dir "$OUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    --queue_max "$QUEUE_MAX" \
    --max_seqs "$MAX_SEQS" \
    > "$LOG_FILE" 2>&1 &


TASK_PID=$!
echo $TASK_PID > "$PID_FILE"


sleep 2
if ps -p $TASK_PID > /dev/null 2>&1; then
    echo "  Task started successfully!"
    echo "  Process ID: $TASK_PID"
    echo ""
    echo "  Monitoring commands:"
    echo "  - View log (live):  tail -f $LOG_FILE"
    echo "  - GPU status:       watch -n 1 nvidia-smi"
    echo "  - Stop task:        bash stop.sh"
    echo ""
    echo "  Quick monitor:"
    echo "  tail -f $LOG_FILE | grep -E '(batch|GPU|Error|finished)'"
else
    echo "  Error: Task failed to start. Check log file:"
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo " Task is running in background."
echo "=========================================="