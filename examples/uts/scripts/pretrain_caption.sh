#!/bin/bash
# ----------------pretrain audio captioning------------------
exp_dir=exp/auden_captionstew400k_caption_pt
torchrun --nproc_per_node=4 \
        --master-port=29502 \
        train.py \
        --config-name train_caption \
        exp_dir=$exp_dir \
        trainer.optimizer.lr=0.045 \
        trainer.scheduler.lr_steps_per_epoch=10000 \
        data.train_data_config=configs/data_configs/train_data_config.yaml \
        data.valid_data_config=configs/data_configs/valid_data_config.yaml \
        data.sampler.max_duration=800 \
        data.use_infinite_dataset=true
