#!/bin/bash
torchrun --nproc_per_node=1 train.py \
  exp_dir=experiments/diarization_AMI \
  data.train_data_config=configs/AMI/data_configs/train_data_config.yaml \
  data.valid_data_config=configs/AMI/data_configs/valid_data_config.yaml 

  # for better performance, we recommend set path to the pre-SOT-FT semantic encoder
  # model.audio_encoder.pretrained_model= 
