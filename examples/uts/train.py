"""UTS training entry script supporting both audio tagging and audio captioning.

Usage:
    # audio tagging (default config)
    torchrun --nproc_per_node=4 train.py

    # audio captioning
    torchrun --nproc_per_node=4 train.py --config-name train_caption
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import UTSDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import AudioCaptionTrainer, AudioTagTrainer
from transformers import AutoTokenizer as HFTokenizer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel
from auden.models.audio_tag.utils import load_id2label, save_id2label


def load_pretrained_encoder(cfg: DictConfig):
    """Load a pretrained encoder, returning (config, module_or_None)."""
    if cfg.get("model_type") == "zipformer":
        from auden.models.zipformer.model import ZipformerEncoderModel
        from auden.models.zipformer.model_config import ZipformerConfig

        pretrained = cfg.get("pretrained_model")
        if pretrained:
            encoder_model = ZipformerEncoderModel.from_pretrained(pretrained)
            encoder_config = encoder_model.config
        else:
            encoder_config = ZipformerConfig()
            encoder_model = None
    else:
        raise ValueError(f"Unsupported encoder model type: {cfg.get('model_type')}")
    return encoder_config, encoder_model


def build_audio_tag(cfg, rank):
    try:
        encoder_config = AutoConfig.from_pretrained(cfg.model.encoder)
    except Exception:
        encoder_config = AutoConfig.for_model(cfg.model.encoder)

    config_kwargs = {"encoder_config": encoder_config}
    if cfg.model.get("loss") is not None:
        config_kwargs["loss"] = cfg.model.loss
    config = AutoConfig.for_model(cfg.model.model_type, **config_kwargs)
    id2label = load_id2label(cfg.model.id2label_json)
    model = AutoModel.from_config(config, id2label=id2label)

    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        save_id2label(id2label, cfg.exp_dir)

    return model


def build_audio_caption(cfg, rank):
    encoder_cfg = cfg.model.audio_encoder
    encoder_config, pretrained_encoder = load_pretrained_encoder(encoder_cfg)

    config = AutoConfig.for_model(
        cfg.model.model_type,
        audio_encoder_config=encoder_config,
    )

    tokenizer = HFTokenizer.from_pretrained(cfg.tokenizer)
    model = AutoModel.from_config(config, tokenizer=tokenizer)

    if pretrained_encoder is not None:
        model.audio_encoder.load_state_dict(
            pretrained_encoder.state_dict(), strict=True
        )
        if rank == 0:
            logging.info(
                f"Loaded audio encoder weights from {encoder_cfg.get('pretrained_model')}"
            )

    if encoder_cfg.get("frozen"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        if rank == 0:
            logging.info("Froze audio encoder weights")

    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)

    return model


TRAINER_MAP = {
    "audio-tag": AudioTagTrainer,
    "audio-caption": AudioCaptionTrainer,
}

MODEL_BUILDER_MAP = {
    "audio-tag": build_audio_tag,
    "audio-caption": build_audio_caption,
}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    if cfg.get("exp_dir"):
        os.makedirs(cfg.exp_dir, exist_ok=True)

    model_type = cfg.model.model_type
    model = MODEL_BUILDER_MAP[model_type](cfg, rank)

    data_module = UTSDatamodule(cfg.data)

    trainer_cls = TRAINER_MAP[model_type]
    trainer = trainer_cls(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
