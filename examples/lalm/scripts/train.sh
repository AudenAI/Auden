#!/bin/bash
# LALM training script.
#
# Single-node multi-GPU:
#   bash scripts/train.sh
#
# Multi-node (set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT externally):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<host0> bash scripts/train.sh
#
# Assumes the current working directory is examples/lalm:
#   bash scripts/train.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONPATH=/apdcephfs_cq10/share_1603164/user/yiwenyshao/lhotse:${PYTHONPATH:-}

# ── GPU / node ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# export CUDA_VISIBLE_DEVICES=0
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
num_nodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}

# ── Model ─────────────────────────────────────────────────────────────────────
# Point to the checkpoint created by scripts/build_model.sh.
model_dir="./models/aut_qwen2_7b_ds1"
mixed_precision=bf16

# Modules to freeze: "audio_tower" for stage-1 (projector-only training).
# Set to "" to unfreeze all (stage-2, full fine-tuning).
frozen_modules="audio_tower,language_model"

# ── Experiment ────────────────────────────────────────────────────────────────
exp_name=aut_qwen2_7b_ds1_asr
exp_dir="./exp/$exp_name"

echo "========================================================"
echo "  Nodes: ${num_nodes}  |  GPUs/node: ${num_gpus}"
echo "  Master: ${master_addr}:${master_port}"
echo "  Model:  ${model_dir}"
echo "  Exp:    ${exp_dir}"
echo "========================================================"

torchrun \
    --nnodes="$num_nodes" \
    --nproc_per_node="$num_gpus" \
    --node_rank="$node_rank" \
    --master_addr="$master_addr" \
    --master_port="$master_port" \
    train.py \
        exp_dir="$exp_dir" \
        model.pretrained_model="$model_dir" \
        trainer.mixed_precision="$mixed_precision" \
        trainer.frozen_modules="[$frozen_modules]" \
        trainer.optimizer.lr=1e-3 \
        trainer.optimizer.weight_decay=0 \
        trainer.scheduler.type=cosine \
        trainer.use_averaged_model=true \
        trainer.valid_interval=1000 \
        trainer.save_every_n=4 \
        trainer.keep_last_k=5 \
        trainer.log_interval=50 \
        data.feature=whisper_v3_fbank \
        data.audio_token_rate=12.5 \
        data.min_duration=0.5 \
        data.max_duration=30.0 \
        data.use_infinite_dataset=true \
        data.num_workers=8 \
        data.sampler.max_tokens=2000
