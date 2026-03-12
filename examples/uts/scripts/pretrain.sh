#!/bin/bash
# ----------------pretrain from scratch------------------
exp_dir=exp/auden_captionstew400k_tag_2k_pt
torchrun --nproc_per_node=4 \
        --master-port=29501 \
        train.py \
        exp_dir=$exp_dir \
        trainer.optimizer.lr=0.045 \
        trainer.scheduler.lr_steps_per_epoch=10000 \
        data.train_data_config=configs/data_configs/train_data_config.yaml \
        data.valid_data_config=configs/data_configs/valid_data_config.yaml \
        data.sampler.max_duration=800 \
        data.use_infinite_dataset=true \
        model.id2label_json=configs/label/label_2k.json \
        model.loss=bce
