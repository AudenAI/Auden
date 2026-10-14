# Adapted from Spatial-Omni for self-contained inference in Auden.
"""Configuration class for spatial-augmented Qwen3-Omni-MoE Thinker."""

from __future__ import annotations

import os
import sys


from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)


class Qwen3OmniMoeSpatialThinkerConfig(Qwen3OmniMoeThinkerConfig):
    """SO-30B configuration with checkpoint-compatible spatial fields."""

    model_type = "qwen3_omni_moe_thinker_spatial"
    # Qwen3 uses audio_token_id / image_token_id / video_token_id directly.
    # Only spatial keeps the *_index legacy alias for the spatial code path.
    attribute_map = {
        "spatial_token_id": "spatial_token_index",
    }
    sub_configs = {
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
        "vision_config": Qwen3OmniMoeVisionEncoderConfig,
        "text_config": Qwen3OmniMoeTextConfig,
    }

    def __init__(
        self,
        # Forwarded to parent Qwen3OmniMoeThinkerConfig
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        position_id_per_seconds=25,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        tie_word_embeddings=False,
        # ----- Spatial token placeholder -----
        spatial_token_index=None,
        spatial_start_token_id=None,
        spatial_end_token_id=None,
        require_spatial_boundaries=True,
        strict_spatial_placeholder_count=True,
        # ----- SELD233 spatial path -----
        use_seld_spatial_modality=False,
        seld_checkpoint_path=None,
        seld_baseline_repo_path=None,
        seld_task_id="233",
        seld_feature_stats_dir=None,
        seld_num_feature_channels=7,
        seld_num_mel_bins=64,
        seld_hop_length=320,
        seld_feature_to_seld_ratio=5,
        seld_encoder_dim=128,
        seld_token_dim=256,
        seld_token_rate_hz=2.5,
        seld_downsample_factor=4,
        seld_projector_hidden_dim=512,
        seld_freeze_backbone=True,
        seld_max_audio_seconds=20.0,
        # ----- Spatial encoder selection -----
        spatial_encoder_type="so_backbone",  # default to BEATs for Qwen3 path
        # ----- Spatial-BEATs encoder -----
        so_backbone_checkpoint_path=None,
        so_backbone_repo_path=None,
        so_encoder_dim=768,
        so_projector_hidden_dim=768,
        so_backbone_target_token_rate=2.5,
        so_encoder_token_rate=10.0,
        so_backbone_freeze_backbone=True,
        so_backbone_max_audio_seconds=20.0,
        so_projector_type="pixel_shuffle",
        so_projector_shuffle_factor=4,
        # ----- Mono replay (mixed spatial+mono training) -----
        enable_spatial_replay=False,
        spatial_null_num_tokens=None,
        spatial_null_alignment_weight=0.05,
        # ----- IV / Neural-IV baselines -----
        spatial_iv_feature_to_seld_ratio=5,
        spatial_iv_downsample_factor=4,
        spatial_iv_token_dim=256,
        spatial_iv_projector_hidden_dim=512,
        spatial_iv_num_mel_bins=64,
        spatial_iv_band_pool=0,
        spatial_iv_output_scale=0.02,
        spatial_iv_max_audio_seconds=20.0,
        spatial_neural_iv_hidden_channels=64,
        **kwargs,
    ):
        # Drop any inbound `model_type` from kwargs so the parent's
        # __init__ does not set self.model_type and thereby shadow our
        # subclass-level class attribute "qwen3_omni_moe_thinker_spatial".
        # (HF saved configs always serialize model_type, so this is hit on
        # every from_pretrained roundtrip.)
        kwargs.pop("model_type", None)

        # ---- Spatial fields (assigned BEFORE super().__init__ so they survive
        #      any save/load roundtrip via to_dict). ----
        self.spatial_token_index = spatial_token_index
        self.spatial_start_token_id = spatial_start_token_id
        self.spatial_end_token_id = spatial_end_token_id
        self.require_spatial_boundaries = bool(require_spatial_boundaries)
        self.strict_spatial_placeholder_count = bool(strict_spatial_placeholder_count)

        self.use_seld_spatial_modality = use_seld_spatial_modality
        self.seld_checkpoint_path = seld_checkpoint_path
        self.seld_baseline_repo_path = seld_baseline_repo_path
        self.seld_task_id = seld_task_id
        self.seld_feature_stats_dir = seld_feature_stats_dir
        self.seld_num_feature_channels = seld_num_feature_channels
        self.seld_num_mel_bins = seld_num_mel_bins
        self.seld_hop_length = seld_hop_length
        self.seld_feature_to_seld_ratio = seld_feature_to_seld_ratio
        self.seld_encoder_dim = seld_encoder_dim
        self.seld_token_dim = seld_token_dim
        self.seld_token_rate_hz = seld_token_rate_hz
        self.seld_downsample_factor = seld_downsample_factor
        self.seld_projector_hidden_dim = seld_projector_hidden_dim
        self.seld_freeze_backbone = seld_freeze_backbone
        self.seld_max_audio_seconds = seld_max_audio_seconds

        self.spatial_encoder_type = spatial_encoder_type

        self.so_backbone_checkpoint_path = so_backbone_checkpoint_path
        self.so_backbone_repo_path = so_backbone_repo_path
        self.so_encoder_dim = so_encoder_dim
        self.so_projector_hidden_dim = so_projector_hidden_dim
        self.so_backbone_target_token_rate = so_backbone_target_token_rate
        self.so_encoder_token_rate = so_encoder_token_rate
        self.so_backbone_freeze_backbone = so_backbone_freeze_backbone
        self.so_backbone_max_audio_seconds = so_backbone_max_audio_seconds
        self.so_projector_type = so_projector_type
        self.so_projector_shuffle_factor = so_projector_shuffle_factor

        self.enable_spatial_replay = bool(enable_spatial_replay)
        self.spatial_null_num_tokens = (
            None if spatial_null_num_tokens is None else int(spatial_null_num_tokens)
        )
        self.spatial_null_alignment_weight = float(spatial_null_alignment_weight)

        self.spatial_iv_feature_to_seld_ratio = spatial_iv_feature_to_seld_ratio
        self.spatial_iv_downsample_factor = spatial_iv_downsample_factor
        self.spatial_iv_token_dim = spatial_iv_token_dim
        self.spatial_iv_projector_hidden_dim = spatial_iv_projector_hidden_dim
        self.spatial_iv_num_mel_bins = spatial_iv_num_mel_bins
        self.spatial_iv_band_pool = spatial_iv_band_pool
        self.spatial_iv_output_scale = spatial_iv_output_scale
        self.spatial_iv_max_audio_seconds = spatial_iv_max_audio_seconds
        self.spatial_neural_iv_hidden_channels = spatial_neural_iv_hidden_channels

        # Delegate to upstream constructor for audio/vision/text + token ids.
        super().__init__(
            audio_config=audio_config,
            vision_config=vision_config,
            text_config=text_config,
            audio_token_id=audio_token_id,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            position_id_per_seconds=position_id_per_seconds,
            audio_start_token_id=audio_start_token_id,
            user_token_id=user_token_id,
            initializer_range=initializer_range,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience accessor matching Qwen2.5 spatial codebase expectations.
    # ------------------------------------------------------------------
    @property
    def llm_hidden_size(self) -> int:
        """Hidden size of the underlying text model (2048 for Qwen3-30B-A3B)."""
        text_cfg = getattr(self, "text_config", None)
        if text_cfg is None:
            raise ValueError("text_config must be set before accessing llm_hidden_size")
        return int(text_cfg.hidden_size)


def register_qwen3_spatial_config() -> None:
    """Register the spatial thinker config with Hugging Face auto classes."""

    from transformers import AutoConfig

    AutoConfig.register(
        Qwen3OmniMoeSpatialThinkerConfig.model_type,
        Qwen3OmniMoeSpatialThinkerConfig,
        exist_ok=True,
    )


register_qwen3_spatial_config()


__all__ = [
    "Qwen3OmniMoeSpatialThinkerConfig",
    "register_qwen3_spatial_config",
]
