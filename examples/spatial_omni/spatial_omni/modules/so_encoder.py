# Adapted from Spatial-Omni for self-contained inference in Auden.
"""SO-Encoder wrapper with deferred backbone construction after base-model loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SOEncoderOutput:
    spatial_tokens: torch.FloatTensor        # [B, T_s_max, D]
    spatial_token_lengths: torch.LongTensor  # [B]


class SOEncoder(nn.Module):

    def __init__(
        self,
        checkpoint_path: str,
        beats_repo_path: Optional[str] = None,
        freeze_backbone: bool = True,
        max_audio_seconds: float = 20.0,
        encoder_token_rate: float = 10.0,
    ) -> None:
        super().__init__()

        self._beats_repo_path: str = beats_repo_path
        self._checkpoint_path: str = checkpoint_path
        self._freeze_backbone: bool = freeze_backbone
        self.max_audio_seconds: float = float(max_audio_seconds)

        # Encoder native token rate (set by caller; SO-Encoder = 10 Hz). The cfg's
        # target_token_rate field is the framework-level "goal" rate and does
        # not necessarily match the native fused-token rate a given ckpt emits,
        # so we don't read it here.
        self.encoder_token_rate: float = float(encoder_token_rate)
        self.sample_rate: int = 16000
        self.encoder_dim: int = 768

        self._beats_cfg = None
        self._beats_state_dict = None
        self.model = None  # type: ignore[assignment]

        self._preload_checkpoint()

    @staticmethod
    def _config_to_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        if hasattr(cfg, "__dict__"):
            return {
                key: value
                for key, value in vars(cfg).items()
                if not key.startswith("_")
            }
        raise TypeError(f"Unsupported SOBackbone config type: {type(cfg)!r}")

    @classmethod
    def _materialize_config(cls, config_cls, raw_cfg: Any):
        # Start from the current upstream defaults, then overlay checkpoint
        # values so newly added config fields keep sane defaults.
        cfg = config_cls()
        cfg.update(cls._config_to_dict(raw_cfg))
        return cfg

    def _import_so_backbone(self):
        from ..encoders.beats.so_backbone import SOBackbone, SOBackboneConfig
        return SOBackbone, SOBackboneConfig


    def _preload_checkpoint(self) -> None:
        _, SOBackboneConfig = self._import_so_backbone()
        ckpt = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
        if "cfg" in ckpt:
            raw_cfg = ckpt["cfg"]
            state_dict = ckpt["model"]
        elif "train_cfg" in ckpt:
            raw_cfg = ckpt["train_cfg"]["model"]
            state_dict = ckpt["model_state_dict"]
        else:
            raise KeyError(
                f"Unrecognised checkpoint format in {self._checkpoint_path}. "
                f"Got keys: {list(ckpt.keys())}"
            )
        cfg = self._materialize_config(SOBackboneConfig, raw_cfg)
        self._beats_cfg = cfg
        self._beats_state_dict = state_dict
        self.sample_rate = int(cfg.sample_rate)
        self.encoder_dim = int(cfg.encoder_embed_dim)
        print(
            f"[BEATsWrapper] Loaded ckpt: readout_scheme={getattr(cfg, 'readout_scheme', None)!r} "
            f"patch_adapter_version={getattr(cfg, 'patch_adapter_version', None)!r} "
            f"use_trunk_spatial_adapters={getattr(cfg, 'use_trunk_spatial_adapters', None)} "
            f"cfg.target_token_rate={getattr(cfg, 'target_token_rate', None)} "
            f"wrapper.encoder_token_rate={self.encoder_token_rate}"
        )

    def _build_model(self) -> None:
        """Build SOBackbone on CPU. Call AFTER from_pretrained() returns."""
        if self.model is not None:
            return
        SOBackbone, _ = self._import_so_backbone()
        beats_model = SOBackbone(self._beats_cfg)
        missing, unexpected = beats_model.load_state_dict(self._beats_state_dict, strict=False)
        if missing:
            print(f"[BEATsWrapper] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[BEATsWrapper] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        if self._freeze_backbone:
            for p in beats_model.parameters():
                p.requires_grad = False
            beats_model.eval()
        self.model = beats_model
        self._beats_state_dict = None

    def train(self, mode: bool = True) -> "SOEncoder":
        super().train(mode)
        if self._freeze_backbone and self.model is not None:
            self.model.eval()
        return self

    def _compute_clip_durations(self, waveform, spatial_audio_lengths=None):
        B = waveform.shape[0]
        if spatial_audio_lengths is not None:
            d = spatial_audio_lengths.float() / float(self.sample_rate)
        else:
            T = waveform.shape[-1]
            d = torch.full((B,), float(T) / float(self.sample_rate),
                            device=waveform.device, dtype=torch.float32)
        return d.clamp(max=self.max_audio_seconds)

    def forward(self, spatial_audio, spatial_audio_attention_mask=None,
                spatial_audio_lengths=None):
        if self.model is None:
            raise RuntimeError(
                "model is None -- call _build_model() after from_pretrained()."
            )
        waveform = spatial_audio.transpose(1, 2).contiguous()  # [B,T,4]->[B,4,T]
        durations = self._compute_clip_durations(waveform, spatial_audio_lengths)
        embeddings, _ = self.model.extract_features(
            waveform=waveform, clip_duration_seconds=durations)
        steps = self.model.compute_target_num_steps(
            waveform=waveform, clip_duration_seconds=durations)
        return SOEncoderOutput(
            spatial_tokens=embeddings, spatial_token_lengths=steps)
