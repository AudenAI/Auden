"""ASR training entry script using Hydra and torch.distributed.

This script wires together the model, tokenizer, data module and trainer, and
supports single- or multi-GPU training via torchrun. Key responsibilities:

- Parse configuration with Hydra (see ``configs/train.yaml``).
- Initialize torch.distributed process group when launched with torchrun.
- Build model/config/tokenizer via Auden ``Auto*`` APIs.
- Optionally load pretrained encoder weights.
- Save config and tokenizer on rank 0 for reproducibility.
- Initialize the ASR datamodule and task-specific trainer, then start training.

Environment variables (set by torchrun):
- ``RANK``: global rank of current process
- ``LOCAL_RANK``: local device index per node
- ``WORLD_SIZE``: total number of processes

Typical usage:
    # Single GPU
    python examples/asr/train.py

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 examples/asr/train.py

    # Override config values
    python examples/asr/train.py exp_dir=./exp/my_experiment

Config expectations (subset):
- ``cfg.model.model_type``: model class key for ``AutoConfig.for_model`` (e.g., "asr").
- ``cfg.model.encoder.model_type``: encoder type (e.g., "zipformer", "whisper-encoder").
- ``cfg.model.encoder.pretrained_encoder``: optional path/HF-repo for pretrained encoder weights.
  If provided, both config and weights are loaded; if null, only config is created.
- ``cfg.tokenizer``: tokenizer id or path for HF ``AutoTokenizer.from_pretrained``.
- ``cfg.data``: datamodule configuration (see ``AsrDatamodule`` and base datamodule).
- ``cfg.exp_dir``: experiment output directory (config/tokenizer will be saved here).
"""

import json
import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AsrDatamodule
from lhotse.utils import fix_random_seed
from omegaconf import DictConfig, OmegaConf
from trainer import AsrTrainer
from transformers import AutoTokenizer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel


def load_pretrained_encoder(cfg: DictConfig):
    """Load pretrained encoder module or build an empty encoder config.

    Args:
        cfg: Encoder configuration section (cfg.model.encoder) with fields:
            - model_type: Encoder model type (e.g., "zipformer", "whisper-encoder")
            - pretrained_encoder: Optional path/HF-repo to load pretrained weights

    Returns:
        tuple: (encoder_config, pretrained_encoder_or_None)
            - encoder_config: Configuration object for the encoder
            - pretrained_encoder: Loaded encoder model if pretrained_encoder was provided,
                                 None otherwise

    Example:
        >>> cfg = {"model_type": "zipformer", "pretrained_encoder": None}
        >>> encoder_config, pretrained_encoder = load_pretrained_encoder(cfg)
        >>> # encoder_config will be a ZipformerConfig, pretrained_encoder will be None

        >>> cfg = {"model_type": "zipformer", "pretrained_encoder": "path/to/model"}
        >>> encoder_config, pretrained_encoder = load_pretrained_encoder(cfg)
        >>> # encoder_config and pretrained_encoder both loaded from checkpoint
    """
    if cfg.get("pretrained_encoder") is not None:
        pretrained_encoder = AutoModel.from_pretrained(cfg.pretrained_encoder)
        encoder_config = pretrained_encoder.config
        return encoder_config, pretrained_encoder
    else:
        encoder_config = AutoConfig.for_model(cfg.model_type)
        return encoder_config, None


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """Hydra entrypoint for ASR training.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra/OMEGACONF configuration. Expected keys include:
        - ``seed``: random seed (optional).
        - ``exp_dir``: experiment directory to save artifacts.
        - ``model.model_type``: ASR model type (e.g., "asr").
        - ``model.encoder.model_type``: encoder type (e.g., "zipformer").
        - ``model.encoder.pretrained_encoder``: optional path to pretrained encoder.
        - ``tokenizer``: tokenizer id or path for HuggingFace AutoTokenizer.
        - ``data``: datamodule configuration.

    Workflow
    --------
    1. Fix random seed if provided
    2. Initialize DDP process group for multi-GPU training
    3. Load/create encoder config and optionally load pretrained encoder
    4. Create ASR model with encoder config and tokenizer
    5. Load pretrained encoder weights into model if provided
    6. Save config and tokenizer to exp_dir (rank 0 only)
    7. Initialize data module and trainer
    8. Run training loop
    9. Clean up DDP process group

    Side Effects
    ------------
    - Initializes/disposes torch.distributed process group when WORLD_SIZE > 1.
    - Saves model config and tokenizer on rank 0 under ``exp_dir``.
    - Starts the training loop via ``AsrTrainer.run()``.
    """
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # 1) Fix random seed
    if "seed" in cfg:
        fix_random_seed(cfg.seed)

    # 2) Gather torchrun environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 3) Initialize process group if multi-GPU
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # 4) Create experiment directory
    if "exp_dir" in cfg and cfg.exp_dir:
        os.makedirs(cfg.exp_dir, exist_ok=True)

    # 5) initialize model
    encoder_config, pretrained_encoder = load_pretrained_encoder(cfg.model.encoder)
    config = AutoConfig.for_model(cfg.model.model_type, encoder_config=encoder_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    model = AutoModel.from_config(config, tokenizer)

    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
        logging.info(
            f"Loaded pretrained encoder from {cfg.model.encoder.pretrained_encoder}"
        )

    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)

    # 6) initialize data module
    data_module = AsrDatamodule(cfg.data)

    # 7) Create the trainer, passing the model
    trainer = AsrTrainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    # 8) Destroy process group if used
    if world_size > 1:
        dist.destroy_process_group()

    logging.info("Training finished successfully.")


if __name__ == "__main__":
    main()
