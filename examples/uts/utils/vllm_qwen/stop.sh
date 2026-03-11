#!/bin/bash
# ============================================================
# Stop vLLM batch inference scrtpt launched by run.sh
# ============================================================


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/task.pid"

echo "=========================================="
echo "  vLLM Task Stopper"
echo "=========================================="
echo ""


if [ ! -f "$PID_FILE" ]; then
    echo "  Error: PID file not found at $PID_FILE"
    echo "   No task is running or task.pid was deleted."
    exit 1
fi


TASK_PID=$(cat "$PID_FILE")
echo "Found PID: $TASK_PID"


if ! ps -p $TASK_PID > /dev/null 2>&1; then
    echo "Process $TASK_PID is not running."
    echo "   Cleaning up PID file..."
    rm -f "$PID_FILE"
    echo "PID file removed."
    exit 0
fi


echo ""
echo "Process info:"
ps -p $TASK_PID -f
echo ""


echo "Sending SIGINT (Ctrl+C) to process $TASK_PID..."
kill -SIGINT $TASK_PID


echo "Waiting for process to exit gracefully (max 10 seconds)..."
for i in {1..10}; do
    sleep 1
    if ! ps -p $TASK_PID > /dev/null 2>&1; then
        echo "Process exited gracefully after $i seconds."
        break
    fi
    echo "   Waiting... ($i/10)"
done


if ps -p $TASK_PID > /dev/null 2>&1; then
    echo ""
    echo "Process still running, sending SIGTERM..."
    kill -SIGTERM $TASK_PID
    sleep 2
    
    if ps -p $TASK_PID > /dev/null 2>&1; then
        echo "Process still running, force killing with SIGKILL..."
        kill -9 $TASK_PID
        sleep 1
    fi
fi


if ps -p $TASK_PID > /dev/null 2>&1; then
    echo ""
    echo "  Failed to stop process $TASK_PID"
    echo "   You may need to manually kill it:"
    echo "   kill -9 $TASK_PID"
    exit 1
else
    echo ""
    echo "  Process $TASK_PID stopped successfully."
fi


echo ""
echo "Waiting for GPU memory to be released..."
sleep 2


echo ""
echo " Current GPU status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s | Util: %s%% | Memory: %s/%s MB\n", $1, $2, $3, $4, $5}'


echo ""
echo "🗑️  Cleaning up files..."


if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
    echo "Deleted PID file: task.pid"
else
    echo "PID file not found (already cleaned)"
fi


LOG_DIR="$SCRIPT_DIR/logs"
if [ -d "$LOG_DIR" ]; then
    LOG_COUNT=$(find "$LOG_DIR" -name "inference_*.log" 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        rm -f "$LOG_DIR"/inference_*.log
        echo "Deleted $LOG_COUNT log file(s) from logs/"
    else
        echo "No log files found in logs/"
    fi
else
    echo "Log directory not found: logs/"
fi

echo ""
echo "=========================================="
echo " Task stopped and all files cleaned."
echo "=========================================="
