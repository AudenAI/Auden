"""Inference-only loading for the published Spatial-Omni checkpoints.

Adapted from Spatial-Omni's Apache-2.0 model builders and release loaders.
"""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from pathlib import Path

import torch

from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
from spatial_omni.utils.release import load_release_settings, load_release_state

MAX_AUDIO_SECONDS = 20
ENCODER_TOKEN_RATE = 10.0
TARGET_TOKEN_RATE = 2.5


def _model_args(config, settings, model_id, device):
    defaults = {
        "train_mode": "projector_only",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": [],
        "lora_target_prefixes": ["thinker.model"],
        "projector_type": "mlp",
        "projector_shuffle_factor": 1,
        "encoder_token_rate": ENCODER_TOKEN_RATE,
        "projector_fp32": False,
        "mixed_spatial_replay": False,
        "null_alignment_weight": 0.05,
        "train_moe_router": False,
        "moe_router_aux_loss_coef": 1e-3,
    }
    values = {key: settings.get(key, default) for key, default in defaults.items()}
    if values["train_mode"] not in {"projector_only", "encoder_lora", "beats_lora", "all"}:
        raise ValueError(f"Unsupported checkpoint train_mode: {values['train_mode']}")
    values.update(
        model_id=model_id, beats_checkpoint=settings["beats_checkpoint"],
        dtype=config["dtype"], attn_impl=config["attn_impl"],
        device=device or config["device"],
    )
    return Namespace(**values)


def _resolve_base_model(model_id, backbone, revision=None):
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("The checkpoint settings must specify model_id, or pass --base-model.")
    if backbone != "so30b":
        return model_id
    path = Path(model_id).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Base model config.json not found in {path}")
        return str(path.resolve())
    if path.is_absolute() or model_id.startswith("."):
        raise FileNotFoundError(f"Base model directory not found: {path}")
    from huggingface_hub import snapshot_download

    # SO-30B reads the raw Thinker config and chat template from the snapshot.
    return snapshot_download(repo_id=model_id, revision=revision)


def _configure_spatial_config(cfg, args):
    """Restore encoder dimensions and token rates from checkpoint settings."""
    cfg.spatial_encoder_type = "so_backbone"
    cfg.so_backbone_checkpoint_path = str(Path(args.beats_checkpoint).resolve())
    cfg.so_backbone_repo_path = ""
    cfg.so_encoder_dim = 768
    cfg.so_projector_hidden_dim = 768
    shuffle_factor = int(args.projector_shuffle_factor)
    if shuffle_factor < 1:
        raise ValueError("projector_shuffle_factor must be >= 1")
    if args.projector_type != "pixel_shuffle":
        shuffle_factor = 1
    encoder_rate = float(args.encoder_token_rate)
    effective_rate = encoder_rate / shuffle_factor
    cfg.so_encoder_token_rate = encoder_rate
    cfg.so_backbone_target_token_rate = effective_rate
    cfg.so_projector_type = args.projector_type
    cfg.so_projector_shuffle_factor = shuffle_factor
    cfg.so_backbone_freeze_backbone = args.train_mode in {"projector_only", "encoder_lora"}
    cfg.so_backbone_max_audio_seconds = float(MAX_AUDIO_SECONDS)
    cfg.enable_spatial_replay = bool(args.mixed_spatial_replay)
    if cfg.enable_spatial_replay:
        cfg.spatial_null_num_tokens = int(round(MAX_AUDIO_SECONDS * effective_rate))
        cfg.spatial_null_alignment_weight = float(args.null_alignment_weight)
    if abs(effective_rate - TARGET_TOKEN_RATE) > 1e-6:
        logging.warning("Checkpoint uses %.3f spatial tokens/s (usual rate: %.3f).",
                        effective_rate, TARGET_TOKEN_RATE)


def _base_loading_kwargs(args, config):
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    return {
        "config": config, "torch_dtype": dtype, "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_impl,
    }


def _build_so7b(args):
    from spatial_omni.model.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    from spatial_omni.model.processing_so import Qwen2_5OmniSpatialProcessor
    from spatial_omni.model.configuration import Qwen2_5OmniConfig
    from spatial_omni.model.modeling_so_thinker import Qwen2_5OmniSpatialForConditionalGeneration

    base = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    processor = Qwen2_5OmniSpatialProcessor(
        image_processor=base.image_processor, feature_extractor=base.feature_extractor,
        tokenizer=base.tokenizer, chat_template=base.chat_template,
    )
    processor.tokenizer.padding_side = "left"
    cfg = Qwen2_5OmniConfig.from_pretrained(args.model_id)
    cfg.loss_type = "ForCausalLMLoss"
    cfg.thinker_config.loss_type = "ForCausalLMLoss"
    _configure_spatial_config(cfg.thinker_config, args)
    model = Qwen2_5OmniSpatialForConditionalGeneration.from_pretrained(
        args.model_id, **_base_loading_kwargs(args, cfg),
    )
    processor.sync_spatial_tokenizer_with_model(model)
    model.disable_talker()
    encoder = getattr(model.thinker, "so_encoder", None)
    if encoder is not None:
        encoder._build_model()
    model.to(args.device)
    if args.projector_fp32:
        projector = getattr(model.thinker, "so_projector", None)
        if projector is not None:
            projector.to(dtype=torch.float32)
    if hasattr(model.thinker, "reinit_spatial_null_if_needed"):
        model.thinker.reinit_spatial_null_if_needed()
    return model, processor


def _validate_so30b_base_loading_info(loading_info):
    """Reject every unloaded base tensor except newly introduced spatial weights."""
    missing = [
        key for key in loading_info.get("missing_keys", [])
        if not key.startswith(("so_encoder.", "so_projector.", "spatial_null"))
    ]
    unexpected = [
        key for key in loading_info.get("unexpected_keys", [])
        if not key.startswith(("talker.", "code2wav."))
    ]
    mismatched = list(loading_info.get("mismatched_keys", []) or [])
    errors = list(loading_info.get("error_msgs", []) or [])
    if missing or unexpected or mismatched or errors:
        raise RuntimeError(
            "SO-30B base checkpoint did not load cleanly. "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"mismatched={mismatched[:8]} errors={errors[:3]}"
        )
    logging.info("SO-30B base-load check passed; no missing or unexpected base weights.")


def _safe_kaldi_epsilon(device, dtype):
    return torch.tensor(torch.finfo(torch.float).eps, device=device, dtype=dtype)


def _build_so30b(args):
    from transformers import AutoFeatureExtractor, AutoTokenizer
    from spatial_omni.model.configuration_qwen3_omni import Qwen3OmniMoeSpatialThinkerConfig
    from spatial_omni.model.modeling_so_thinker_qwen3 import Qwen3OmniMoeSpatialForConditionalGeneration, register_qwen3_spatial_auto_classes
    from spatial_omni.model.processing_so_qwen3 import Qwen3OmniMoeSpatialProcessor
    import torchaudio.compliance.kaldi as kaldi

    # Same fix as the released SO-30B builder: no cached cross-device epsilon.
    kaldi._get_epsilon = _safe_kaldi_epsilon
    register_qwen3_spatial_auto_classes()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    chat_template = getattr(tokenizer, "chat_template", None)
    template_path = Path(args.model_id) / "chat_template.json"
    if not chat_template and template_path.is_file():
        with template_path.open(encoding="utf-8") as handle:
            chat_template = json.load(handle).get("chat_template")
    processor = Qwen3OmniMoeSpatialProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.tokenizer.padding_side = "left"
    with (Path(args.model_id) / "config.json").open(encoding="utf-8") as handle:
        raw = json.load(handle)
    cfg = Qwen3OmniMoeSpatialThinkerConfig(**raw.get("thinker_config", raw))
    _configure_spatial_config(cfg, args)
    if hasattr(cfg.text_config, "router_aux_loss_coef"):
        train_router = bool(args.train_moe_router)
        cfg.text_config.router_aux_loss_coef = float(args.moe_router_aux_loss_coef) if train_router else 0.0
        cfg.text_config.output_router_logits = train_router
    cfg.loss_type = "ForCausalLMLoss"
    cfg.text_config.loss_type = "ForCausalLMLoss"
    model, loading_info = Qwen3OmniMoeSpatialForConditionalGeneration.from_pretrained(
        args.model_id, output_loading_info=True, **_base_loading_kwargs(args, cfg),
    )
    _validate_so30b_base_loading_info(loading_info)
    model.reinit_spatial_null_if_needed()
    processor.sync_spatial_tokenizer_with_model(model)
    model.disable_talker()
    encoder = getattr(model, "so_encoder", None)
    if encoder is not None:
        encoder._build_model()
    model.to(args.device)
    return model, processor


def _is_moe_router(name):
    return name.endswith(".mlp.gate.weight") or (
        ".mlp.gate.modules_to_save." in name and name.endswith(".weight")
    )


def _apply_lora(model, args, backbone):
    from peft import LoraConfig, TaskType, get_peft_model

    targets = sorted({
        name for name, _ in model.named_modules()
        if any(name.startswith(prefix) for prefix in args.lora_target_prefixes)
        and name.rsplit(".", 1)[-1] in args.lora_target_modules
    })
    if not targets:
        raise ValueError(f"No LoRA targets under {args.lora_target_prefixes} with {args.lora_target_modules}.")
    routers = []
    if backbone == "so30b" and args.train_moe_router:
        routers = sorted(name for name, _ in model.named_modules() if _is_moe_router(f"{name}.weight"))
        if not routers:
            raise ValueError("train_moe_router is set but no MoE router modules were found.")
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=targets,
        modules_to_save=routers or None,
    )
    return get_peft_model(model, lora_config)


def _mark_checkpoint_parameters(model, args, backbone):
    """Recreate the training-stage mask used by strict adaptation-weight checks."""
    mode = args.train_mode
    if mode == "all":
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "beats_lora":
        encoder = getattr(model.thinker, "so_encoder", None)
        if encoder is not None:
            encoder._freeze_backbone = False
            for parameter in encoder.model.parameters():
                parameter.requires_grad_(True)
    trained_routers = []
    for name, parameter in model.named_parameters():
        required = "so_projector" in name
        if mode != "projector_only":
            required |= "lora_" in name
            required |= mode == "beats_lora" and "so_encoder" in name
            if backbone == "so30b" and args.train_moe_router and _is_moe_router(name):
                required = True
                trained_routers.append(name)
        if required:
            parameter.requires_grad_(True)
    if backbone == "so30b" and mode != "projector_only" and args.train_moe_router and not trained_routers:
        raise ValueError("train_moe_router is set but no MoE router weights were found.")


def _align_peft_prefix(state_dict, model):
    target_keys = set(dict(model.named_parameters()))
    prefix = "base_model.model."
    aligned = {}
    for key, value in state_dict.items():
        target = key
        if key not in target_keys:
            if prefix + key in target_keys:
                target = prefix + key
            elif key.startswith(prefix) and key[len(prefix):] in target_keys:
                target = key[len(prefix):]
        if target in aligned:
            raise RuntimeError(f"Checkpoint keys collide after PEFT prefix alignment: {target}")
        aligned[target] = value
    return aligned


def load_model(config, checkpoint, encoder, base_model=None, device=None):
    """Load SO-7B or SO-30B entirely from this example and released checkpoints."""
    backbone = config["backbone"]
    if backbone not in {"so7b", "so30b"}:
        raise ValueError(f"Unsupported backbone: {backbone}")
    settings = load_release_settings(
        checkpoint, model_id=base_model, encoder_checkpoint=encoder,
    )
    if bool(settings.get("mixed_spatial_replay", False)) != (config["variant"] == "mix"):
        raise ValueError("The selected config and checkpoint disagree on MIX mode.")
    model_id = _resolve_base_model(
        settings.get("model_id"), backbone, revision=config.get("base_revision"),
    )
    args = _model_args(config, settings, model_id, device)
    from transformers import set_seed

    set_seed(config["seed"])
    model, processor = _build_so30b(args) if backbone == "so30b" else _build_so7b(args)
    if args.train_mode in {"encoder_lora", "beats_lora"}:
        model = _apply_lora(model, args, backbone)
    _mark_checkpoint_parameters(model, args, backbone)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError("The adaptation checkpoint must contain a state dictionary.")
    state = saved.get("trainable_state_dict", saved)
    state = remap_legacy_state_dict(state)
    if backbone == "so7b":
        state = _align_peft_prefix(state, model)
    result = load_release_state(model, state)
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trained = sorted(required.intersection(result.missing_keys))
    if missing_trained or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint did not restore all adaptation weights: "
            f"missing={missing_trained[:12]}, unexpected={list(result.unexpected_keys)[:12]}"
        )
    model.eval()
    logging.info("Loaded %s: %d adaptation keys; %d frozen base entries omitted.",
                 backbone, len(state), len(result.missing_keys))
    return model, processor
