"""
LALM (Language-Audio Language Model) configuration.

Combines a speech/audio encoder with a causal LLM and a linear projector.
Supports arbitrary encoder and LLM types via AutoConfig.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from transformers import AutoConfig, PretrainedConfig

DEFAULT_TEXT_MODEL_TYPE = "qwen2"
DEFAULT_AUDIO_MODEL_TYPE = "whisper"


class LALMConfig(PretrainedConfig):
    """
    Configuration for LALM (Language-Audio Language Model).

    LALM combines:
    - An audio encoder (e.g. Whisper, Zipformer)
    - A linear projector from audio dim to text dim
    - A causal LLM (e.g. Qwen2, LLaMA)

    Both ``text_config`` and ``audio_config`` accept a dict (with ``model_type``)
    or a ``PretrainedConfig`` subclass, so you can freely swap encoders and LLMs.

    Args:
        text_config: LLM config. Defaults to Qwen2.
        audio_config: Audio encoder config. Defaults to Whisper.
        audio_dim: Hidden size output by the audio encoder (projector input dim).
        text_dim: Hidden size of the LLM (projector output dim).
        projector_downsample_rate: Number of consecutive encoder frames merged into
            one token by the projector. Higher values = shorter audio sequences fed
            to the LLM. Defaults to 4.
        audio_token_id: Token id used as placeholder for audio embeddings.
    """

    model_type = "lalm"

    def __init__(
        self,
        text_config: Optional[Union[dict, PretrainedConfig]] = None,
        audio_config: Optional[Union[dict, PretrainedConfig]] = None,
        audio_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        projector_downsample_rate: int = 4,
        audio_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.text_config = _resolve_config(text_config, DEFAULT_TEXT_MODEL_TYPE)
        self.audio_config = _resolve_config(audio_config, DEFAULT_AUDIO_MODEL_TYPE)
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.projector_downsample_rate = projector_downsample_rate
        self.audio_token_id = audio_token_id
        super().__init__(**kwargs)

    def to_dict(self) -> dict:
        output = super().to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["audio_config"] = self.audio_config.to_dict()
        return output


def _resolve_config(
    config: Optional[Union[dict, PretrainedConfig]],
    default_model_type: str,
) -> PretrainedConfig:
    """Convert a dict or None into a concrete PretrainedConfig."""
    if isinstance(config, PretrainedConfig):
        return config
    if isinstance(config, dict):
        config = config.copy()
        config.setdefault("model_type", default_model_type)
        model_type = config["model_type"]
        try:
            return AutoConfig.for_model(**config)
        except ValueError:
            return _resolve_known_config(config, model_type)
    return AutoConfig.for_model(default_model_type)


def _resolve_known_config(config: dict, model_type: str) -> PretrainedConfig:
    """Handle HF config classes that exist but are not AutoConfig-registered."""
    if model_type == "whisper":
        from transformers.models.whisper.configuration_whisper import WhisperConfig

        return WhisperConfig.from_dict(config)

    if model_type == "qwen2_5_omni_audio_encoder":
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoderConfig,
        )

        return Qwen2_5OmniAudioEncoderConfig.from_dict(config)

    if model_type == "qwen3_omni_moe_audio_encoder":
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig,
        )

        return Qwen3OmniMoeAudioEncoderConfig.from_dict(config)

    raise ValueError(f"Unrecognized model identifier: {model_type}.")
