"""Checkpoint key remapping for legacy checkpoints.

This shim lets users load checkpoints saved before the rename
(``spatial_beats_*`` / ``seld233_*`` / ``seldnet233_*``) into the new
Spatial-Omni module hierarchy (``so_*`` / ``seld_*``).

Usage::

    from spatial_omni.utils.ckpt_compat import remap_legacy_state_dict
    state_dict = remap_legacy_state_dict(torch.load(path)["model_state_dict"])
    model.load_state_dict(state_dict, strict=False)
"""
from __future__ import annotations

from typing import Mapping, MutableMapping

# Ordered list — apply each substring rewrite in sequence.
# Order matters: more-specific prefixes first.
LEGACY_KEY_MAP = (
    ("spatial_beats_encoder.", "so_encoder."),
    ("spatial_beats_projector.", "so_projector."),
    ("spatial_beats_token_projector.", "so_token_projector."),
    ("audio_tower.spatial_beats_", "audio_tower.so_"),
    ("audio_tower.seldnet233_", "audio_tower.seld_"),
    ("audio_tower.seld233_", "audio_tower.seld_"),
    ("seldnet233_backbone.", "seld_backbone."),
    ("seldnet233_feature_bridge.", "seld_feature_bridge."),
    ("seldnet233_spatial_adapter.", "seld_spatial_adapter."),
    ("seldnet233_", "seld_"),
    ("seld233_", "seld_"),
)


def _rewrite_key(k: str) -> str:
    new_k = k
    for old, new in LEGACY_KEY_MAP:
        if old in new_k:
            new_k = new_k.replace(old, new)
    return new_k


def remap_legacy_state_dict(state_dict: Mapping[str, "torch.Tensor"]) -> "MutableMapping[str, torch.Tensor]":
    """Return a new dict with legacy keys rewritten to the new convention.

    Keys that do not match any legacy prefix are passed through unchanged.
    """
    return {_rewrite_key(k): v for k, v in state_dict.items()}


def remap_in_place(state_dict: MutableMapping[str, "torch.Tensor"]) -> None:
    """Mutate ``state_dict`` in place, replacing legacy keys."""
    rewritten = remap_legacy_state_dict(state_dict)
    state_dict.clear()
    state_dict.update(rewritten)


__all__ = ["LEGACY_KEY_MAP", "remap_legacy_state_dict", "remap_in_place"]
