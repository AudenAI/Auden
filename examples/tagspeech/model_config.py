from __future__ import annotations

import os

from transformers import AutoConfig as HFConfig
from transformers import PretrainedConfig

from auden.auto.auto_config import AutoConfig
from auden.models.base.model_config import BaseConfig
from auden.models.lalm.model_config import LalmConfig


class TagSpeechBaseConfig(LalmConfig):
    """Configuration for AudioLLM with dual audio tokens and separate projectors.

    Architecture:
        Audio Input
            ↓
        ┌─────────────┬─────────────┐
        │             │             │
    Semantic Encoder  Voice Encoder
        │             │
        ↓             ↓
    Projector1    Projector2
        │             │
        ↓             ↓
    [B,L1,D_llm]  [B,L2,D_llm]
        │             │
        └─────┬───────┘
              │
        Text: "text <|AUDIO|> speaker <|AUDIO|>"
              │             │
              ↓             ↓
        Semantic Embedding Voice Embedding
              │             │
              └─────┬───────┘
                    │
                  LLM
    """

    model_type: str = "tagspeech-base"

    def __init__(
        self,
        *,
        voice_encoder_config: dict = None,
        semantic_projector_ds_rate: int = 4,
        voice_projector_ds_rate: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Voice encoder config
        if voice_encoder_config is None:
            voice_encoder_config = self.audio_encoder_config
        self.voice_encoder_config = voice_encoder_config

        # Dual projector downsampling rates
        self.semantic_projector_ds_rate = semantic_projector_ds_rate
        self.voice_projector_ds_rate = voice_projector_ds_rate


class TagSpeechConfig(TagSpeechBaseConfig):
    """Dual-audio-tokens model that inserts numeric anchors derived from digit embeddings.

    Similar to the anchor_text model, but anchors consist of character embeddings from natural number
    sequences (1, 2, 3, ...), ensuring that semantic/voice branches insert the same numbered anchors
    at the same real-time positions for precise time alignment.
    """

    model_type: str = "tagspeech"

    def __init__(
        self,
        *,
        semantic_anchor_interval: int = 8,
        voice_anchor_interval: int = 8,
        insert_anchors_at_ends: bool = True,
        semantic_projector_ds_rate: int = 4,
        voice_projector_ds_rate: int = 4,
        **kwargs,
    ):
        super().__init__(
            semantic_projector_ds_rate=semantic_projector_ds_rate,
            voice_projector_ds_rate=voice_projector_ds_rate,
            **kwargs,
        )

        self.semantic_anchor_interval = int(semantic_anchor_interval)
        self.voice_anchor_interval = int(voice_anchor_interval)
        self.insert_anchors_at_ends = bool(insert_anchors_at_ends)

        if self.semantic_anchor_interval <= 0 or self.voice_anchor_interval <= 0:
            raise ValueError(
                f"Anchor intervals must be positive integers, got "
                f"semantic={self.semantic_anchor_interval}, voice={self.voice_anchor_interval}"
            )

        # Require the two branches to share the same real-time spacing:
        # projector_ds_rate * anchor_interval (measured on encoder frames) should match.
        semantic_stride = (
            self.semantic_anchor_interval * self.semantic_projector_ds_rate
        )
        voice_stride = self.voice_anchor_interval * self.voice_projector_ds_rate
        if semantic_stride != voice_stride:
            raise ValueError(
                "semantic_anchor_interval * semantic_projector_ds_rate must equal "
                "voice_anchor_interval * voice_projector_ds_rate to keep anchors time-aligned.\n"
                f"Got semantic: {self.semantic_anchor_interval} * {self.semantic_projector_ds_rate} = {semantic_stride}, "
                f"voice: {self.voice_anchor_interval} * {self.voice_projector_ds_rate} = {voice_stride}"
            )
