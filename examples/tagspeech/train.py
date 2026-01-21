import json
import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AsrDatamodule
from lhotse.utils import fix_random_seed
from omegaconf import DictConfig, OmegaConf
from trainer import TagSpeechTrainer as Trainer
from transformers import AutoConfig as HFConfig
from transformers import AutoModelForCausalLM as HFCausalLM
from transformers import AutoTokenizer as HFTokenizer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel


def load_pretrained_audio_encoder(cfg: DictConfig):
    """Load pretrained encoder module or build an empty encoder config.

    Args:
        cfg: Encoder configuration section (cfg.model.encoder) with fields:
            - model_type: Encoder model type (e.g., "zipformer", "whisper-encoder")
            - pretrained_model: Optional path/HF-repo to load pretrained weights

    Returns:
        tuple: (encoder_config, pretrained_encoder_or_None)
            - encoder_config: Configuration object for the encoder
            - pretrained_model: Loaded encoder model if pretrained_model was provided,
                                 None otherwise

    Example:
        >>> cfg = {"model_type": "zipformer", "pretrained_encoder": None}
        >>> encoder_config, pretrained_encoder = load_pretrained_encoder(cfg)
        >>> # encoder_config will be a ZipformerConfig, pretrained_encoder will be None

        >>> cfg = {"model_type": "zipformer", "pretrained_encoder": "path/to/model"}
        >>> encoder_config, pretrained_encoder = load_pretrained_encoder(cfg)
        >>> # encoder_config and pretrained_encoder both loaded from checkpoint
    """
    if cfg.get("pretrained_model") is not None:
        pretrained_encoder = AutoModel.from_pretrained(cfg.pretrained_model)
        encoder_config = pretrained_encoder.config
        return encoder_config, pretrained_encoder
    else:
        encoder_config = AutoConfig.for_model(cfg.model_type)
        return encoder_config, None


def load_pretrained_llm(cfg):
    """Build LLM config and optionally load a pretrained HF module and tokenizer.

    Modes:
    - Pretrained: cfg.pretrained_model -> load weights/config/tokenizer
    - Empty-by-type: cfg.model_type -> build config via HF without weights; tokenizer from model_type
    """
    pretrained = cfg.get("pretrained_model")
    model_type = cfg.get("model_type", "qwen2")

    if pretrained:
        llm = HFCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16)
        llm_config = llm.config
        tokenizer = HFTokenizer.from_pretrained(pretrained)

        return llm_config, llm, tokenizer

    # Empty-by-type
    try:
        llm_config = HFConfig.from_pretrained(model_type)
    except Exception:
        llm_config = HFConfig.for_model(model_type)
    tokenizer = HFTokenizer.from_pretrained(model_type)
    return llm_config, None, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Register custom models (must be done before any model loading)
    from auden.auto.auto_config import register_config
    from auden.auto.auto_model import register_model

    # Register TagSpeech model
    register_model(
        model_type="tagspeech",
        module_path="model",
        class_name="TagSpeechModel",
        exist_ok=True,
    )
    register_config(
        config_type="tagspeech",
        module_path="model_config",
        class_name="TagSpeechConfig",
        exist_ok=True,
    )

    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # Seed
    fix_random_seed(114514)

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

    # 1) Build sub-configs and optionally load submodules
    audio_encoder_config, pretrained_audio_encoder = load_pretrained_audio_encoder(
        cfg.model.audio_encoder
    )

    # Load voice_encoder config (both final models use dual audio branches)
    pretrained_voice_encoder = None
    if cfg.model.get("voice_encoder"):
        voice_encoder_config, pretrained_voice_encoder = load_pretrained_audio_encoder(
            cfg.model.voice_encoder
        )
    else:
        # Default: use same config as audio_encoder
        voice_encoder_config = audio_encoder_config
        logging.info(
            "[tagspeech.train] No voice_encoder config provided, using audio_encoder config"
        )

    llm_config, pretrained_llm, tokenizer = load_pretrained_llm(cfg.model.llm)

    # 2) Tokenizer with audio token
    DEFAULT_AUDIO_TOKEN = cfg.model.get("audio_token", "<|AUDIO|>")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_AUDIO_TOKEN]},
        replace_additional_special_tokens=False,
    )
    tokenizer.padding_side = (
        "left" if cfg.model.get("use_flash_attn", False) else "right"
    )

    # 4) Assemble Audio-LLM config and model
    config_kwargs = dict(
        llm_config=llm_config,
        audio_encoder_config=audio_encoder_config,
        voice_encoder_config=voice_encoder_config,
        use_flash_attn=cfg.model.get("use_flash_attn", False),
        tag_audio_boundary=cfg.model.get("tag_audio_boundary", False),
        exclude_from_checkpoint=list(cfg.model.get("exclude_from_checkpoint", None)),
        audio_token=DEFAULT_AUDIO_TOKEN,
        max_length=cfg.model.get("max_length", 256),
        semantic_projector_ds_rate=cfg.model.get("semantic_projector_ds_rate", 4),
        voice_projector_ds_rate=cfg.model.get("voice_projector_ds_rate", 4),
    )

    # Add anchor-num model specific config
    if cfg.model.model_type == "tagspeech":
        config_kwargs["semantic_anchor_interval"] = cfg.model.get(
            "semantic_anchor_interval", 8
        )
        config_kwargs["voice_anchor_interval"] = cfg.model.get(
            "voice_anchor_interval", 8
        )
        config_kwargs["insert_anchors_at_ends"] = cfg.model.get(
            "insert_anchors_at_ends", True
        )

    config = AutoConfig.for_model(cfg.model.model_type, **config_kwargs)

    # Load digit embeddings for TagSpeech model
    digit_embeddings = None
    if cfg.model.model_type == "tagspeech":
        digit_embedding_path = cfg.model.get(
            "digit_embedding_path", "utils/digit_token_embeddings.pt"
        )
        # Resolve relative path
        if not os.path.isabs(digit_embedding_path):
            digit_embedding_path = os.path.join(
                os.path.dirname(__file__), digit_embedding_path
            )

        logging.info(
            f"[tagspeech.train] Loading digit embeddings from {digit_embedding_path}"
        )
        from model import TagSpeechModel

        digit_embeddings = TagSpeechModel.load_digit_embeddings(digit_embedding_path)

    model = AutoModel.from_config(
        config, tokenizer=tokenizer, digit_embeddings=digit_embeddings
    )

    # 5) Load pretrained weights (if provided)
    if pretrained_audio_encoder is not None:
        model.audio_encoder.load_state_dict(
            pretrained_audio_encoder.state_dict(), strict=True
        )
        src = cfg.model.audio_encoder.get("pretrained_model")
        num_params = sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6
        logging.info(
            f"[tagspeech.train] Loaded audio encoder from {src}; params={num_params:.2f} M"
        )

    # Load voice_encoder weights
    if pretrained_voice_encoder is not None:
        model.voice_encoder.load_state_dict(
            pretrained_voice_encoder.state_dict(), strict=True
        )
        src = cfg.model.voice_encoder.get("pretrained_model")
        num_params = sum(p.numel() for p in model.voice_encoder.parameters()) / 1e6
        logging.info(
            f"[tagspeech.train] Loaded voice encoder from {src}; params={num_params:.2f} M"
        )

    # Load LLM weights
    if pretrained_llm is not None:
        model.llm.load_state_dict(pretrained_llm.state_dict(), strict=False)
        src_txt = cfg.model.llm.get("pretrained_model")
        num_params_txt = sum(p.numel() for p in model.llm.parameters()) / 1e6
        logging.info(
            f"[tagspeech.train] Loaded LLM weights from {src_txt}; params={num_params_txt:.2f} M"
        )

    # 5.1) Load pretrained adapter (if provided)
    # Both final models use dual projectors (semantic + voice)
    if hasattr(model, "semantic_projector") and hasattr(model, "voice_projector"):
        # Support two ways: unified adapter or separate adapters
        pretrained_semantic_adapter = cfg.model.get("pretrained_semantic_adapter")
        pretrained_voice_adapter = cfg.model.get("pretrained_voice_adapter")
        pretrained_adapter = cfg.model.get("pretrained_adapter")

        # Load semantic_projector adapter
        if pretrained_semantic_adapter:
            # Load from separate semantic adapter file
            logging.info(
                f"[tagspeech.train] Loading semantic_projector adapter from {pretrained_semantic_adapter}"
            )
            sem_checkpoint = torch.load(pretrained_semantic_adapter, map_location="cpu")
            if isinstance(sem_checkpoint, dict):
                if "state_dict" in sem_checkpoint:
                    sem_checkpoint = sem_checkpoint["state_dict"]
                elif "model" in sem_checkpoint:
                    sem_checkpoint = sem_checkpoint["model"]
            else:
                raise ValueError(
                    f"Checkpoint from {pretrained_semantic_adapter} is not a dict. "
                    f"Expected dict with 'state_dict', 'model', or direct weight keys."
                )

            semantic_projector_state = {}
            for key, value in sem_checkpoint.items():
                if key.startswith("semantic_projector."):
                    new_key = key[len("semantic_projector.") :]
                    semantic_projector_state[new_key] = value
                elif key.startswith("model.semantic_projector."):
                    new_key = key[len("model.semantic_projector.") :]
                    semantic_projector_state[new_key] = value
                elif key.startswith("encoder_projector."):
                    # Support loading from old checkpoint (encoder_projector -> semantic_projector)
                    new_key = key[len("encoder_projector.") :]
                    semantic_projector_state[new_key] = value
                elif key.startswith("model.encoder_projector."):
                    # Support loading from old checkpoint with model prefix
                    new_key = key[len("model.encoder_projector.") :]
                    semantic_projector_state[new_key] = value
                elif (
                    not key.startswith("voice_projector.")
                    and not key.startswith("model.voice_projector.")
                    and not key.startswith("encoder_projector.")
                    and not key.startswith("model.encoder_projector.")
                ):
                    # If no prefix, assume it's semantic_projector weights (for backward compatibility)
                    semantic_projector_state[key] = value

            if semantic_projector_state:
                model.semantic_projector.load_state_dict(
                    semantic_projector_state, strict=True
                )
                num_params_sem = (
                    sum(p.numel() for p in model.semantic_projector.parameters()) / 1e6
                )
                logging.info(
                    f"[tagspeech.train] Loaded semantic_projector adapter; params={num_params_sem:.2f} M"
                )
            else:
                raise ValueError(
                    f"No semantic_projector weights found in {pretrained_semantic_adapter}"
                )

        # Load voice_projector adapter
        if pretrained_voice_adapter:
            # Load from separate voice adapter file
            logging.info(
                f"[tagspeech.train] Loading voice_projector adapter from {pretrained_voice_adapter}"
            )
            voice_checkpoint = torch.load(pretrained_voice_adapter, map_location="cpu")
            if isinstance(voice_checkpoint, dict):
                if "state_dict" in voice_checkpoint:
                    voice_checkpoint = voice_checkpoint["state_dict"]
                elif "model" in voice_checkpoint:
                    voice_checkpoint = voice_checkpoint["model"]
            else:
                raise ValueError(
                    f"Checkpoint from {pretrained_voice_adapter} is not a dict. "
                    f"Expected dict with 'state_dict', 'model', or direct weight keys."
                )

            voice_projector_state = {}
            for key, value in voice_checkpoint.items():
                if key.startswith("voice_projector."):
                    new_key = key[len("voice_projector.") :]
                    voice_projector_state[new_key] = value
                elif key.startswith("model.voice_projector."):
                    new_key = key[len("model.voice_projector.") :]
                    voice_projector_state[new_key] = value
                elif not key.startswith("semantic_projector.") and not key.startswith(
                    "model.semantic_projector."
                ):
                    # If no prefix, assume it's voice_projector weights
                    voice_projector_state[key] = value

            if voice_projector_state:
                model.voice_projector.load_state_dict(
                    voice_projector_state, strict=True
                )
                num_params_voice = (
                    sum(p.numel() for p in model.voice_projector.parameters()) / 1e6
                )
                logging.info(
                    f"[tagspeech.train] Loaded voice_projector adapter; params={num_params_voice:.2f} M"
                )
            else:
                raise ValueError(
                    f"No voice_projector weights found in {pretrained_voice_adapter}"
                )

        # Load from unified adapter file (if separate adapters not provided)
        if (
            pretrained_adapter
            and not pretrained_semantic_adapter
            and not pretrained_voice_adapter
        ):
            logging.info(
                f"[tagspeech.train] Loading unified adapter from {pretrained_adapter}"
            )
            adapter_checkpoint = torch.load(pretrained_adapter, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(adapter_checkpoint, dict):
                if "state_dict" in adapter_checkpoint:
                    adapter_checkpoint = adapter_checkpoint["state_dict"]
                elif "model" in adapter_checkpoint:
                    adapter_checkpoint = adapter_checkpoint["model"]

            semantic_projector_state = {}
            voice_projector_state = {}

            for key, value in adapter_checkpoint.items():
                # Handle semantic_projector
                if key.startswith("semantic_projector."):
                    new_key = key[len("semantic_projector.") :]
                    semantic_projector_state[new_key] = value
                elif key.startswith("model.semantic_projector."):
                    new_key = key[len("model.semantic_projector.") :]
                    semantic_projector_state[new_key] = value

                # Handle voice_projector
                elif key.startswith("voice_projector."):
                    new_key = key[len("voice_projector.") :]
                    voice_projector_state[new_key] = value
                elif key.startswith("model.voice_projector."):
                    new_key = key[len("model.voice_projector.") :]
                    voice_projector_state[new_key] = value

            # Load semantic_projector if found
            if semantic_projector_state:
                model.semantic_projector.load_state_dict(
                    semantic_projector_state, strict=True
                )
                num_params_sem = (
                    sum(p.numel() for p in model.semantic_projector.parameters()) / 1e6
                )
                logging.info(
                    f"[tagspeech.train] Loaded semantic_projector adapter from {pretrained_adapter}; params={num_params_sem:.2f} M"
                )
            else:
                logging.warning(
                    f"No semantic_projector weights found in {pretrained_adapter}"
                )

            # Load voice_projector if found
            if voice_projector_state:
                model.voice_projector.load_state_dict(
                    voice_projector_state, strict=True
                )
                num_params_voice = (
                    sum(p.numel() for p in model.voice_projector.parameters()) / 1e6
                )
                logging.info(
                    f"[tagspeech.train] Loaded voice_projector adapter from {pretrained_adapter}; params={num_params_voice:.2f} M"
                )
            else:
                logging.warning(
                    f"No voice_projector weights found in {pretrained_adapter}"
                )

            if not semantic_projector_state and not voice_projector_state:
                raise ValueError(
                    f"No semantic_projector or voice_projector weights found in {pretrained_adapter}. "
                    f"Expected keys starting with 'semantic_projector.' or 'voice_projector.'"
                )

    # 6) Freeze modules if requested
    if cfg.model.get("audio_encoder", {}).get("frozen"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        logging.info(f"[tagspeech.train] Froze audio encoder")

    if hasattr(model, "voice_encoder") and cfg.model.get("voice_encoder", {}).get(
        "frozen"
    ):
        for p in model.voice_encoder.parameters():
            p.requires_grad = False
        logging.info(f"[tagspeech.train] Froze voice encoder")

    if cfg.model.llm.get("frozen"):
        for p in model.llm.parameters():
            p.requires_grad = False
        logging.info(f"[tagspeech.train] Froze LLM")

    # 7) Save excluded modules (if any) and config/tokenizer
    if rank == 0 and cfg.get("exp_dir"):
        if getattr(config, "exclude_from_checkpoint", None):
            if "audio_encoder" in config.exclude_from_checkpoint:
                audio_encoder_path = os.path.join(cfg.exp_dir, "audio_encoder")
                model.audio_encoder.save_pretrained(audio_encoder_path)
                logging.info(
                    f"[tagspeech.train] Saved audio encoder to {audio_encoder_path}"
                )

            if "voice_encoder" in config.exclude_from_checkpoint and hasattr(
                model, "voice_encoder"
            ):
                voice_encoder_path = os.path.join(cfg.exp_dir, "voice_encoder")
                model.voice_encoder.save_pretrained(voice_encoder_path)
                logging.info(
                    f"[tagspeech.train] Saved voice encoder to {voice_encoder_path}"
                )

            if "llm" in config.exclude_from_checkpoint:
                llm_path = os.path.join(cfg.exp_dir, "llm")
                model.llm.save_pretrained(llm_path)
                logging.info(f"[tagspeech.train] Saved LLM to {llm_path}")

        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)
        logging.info(f"[tagspeech.train] Saved config/tokenizer to {cfg.exp_dir}")

    # 8) Data & Trainer
    data_module = AsrDatamodule(cfg.data)
    trainer = Trainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
