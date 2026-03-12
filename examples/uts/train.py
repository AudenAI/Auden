"""
Taked from examples/audio_tag/train.py

Audio Tagging training entry script using Hydra and torch.distributed.

This mirrors the ASR example structure while adapting the trainer/datamodule to
audio tagging. It wires model, datamodule and trainer, and supports DDP via torchrun.

Usage:
    torchrun --nproc_per_node=8 examples/audio_tag/train.py

Config highlights:
- cfg.model.encoder: encoder path or model_type for AutoConfig
- cfg.model.id2label: path to an id2label.json mapping
- cfg.data: Lhotse-based datamodule configs (see configs/)
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AudioTagDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import AudioTagTrainer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel
from auden.models.audio_tag.utils import load_id2label, save_id2label


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # DDP env
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # exp dir
    if cfg.get("exp_dir"):
        os.makedirs(cfg.exp_dir, exist_ok=True)

    # Build encoder config (path/repo or model_type)
    try:
        encoder_config = AutoConfig.from_pretrained(cfg.model.encoder)
    except Exception:
        encoder_config = AutoConfig.for_model(cfg.model.encoder)

    # AudioTag config and model (respect loss if provided)
    config_kwargs = {"encoder_config": encoder_config}
    if cfg.model.get("loss") is not None:
        config_kwargs["loss"] = cfg.model.loss
    config = AutoConfig.for_model(cfg.model.model_type, **config_kwargs)
    id2label = load_id2label(cfg.model.id2label_json)
    model = AutoModel.from_config(config, id2label=id2label)

    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        save_id2label(id2label, cfg.exp_dir)

    # Datamodule
    data_module = AudioTagDatamodule(cfg.data)

    # Trainer
    trainer = AudioTagTrainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
