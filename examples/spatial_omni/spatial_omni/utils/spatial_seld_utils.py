# Adapted from Spatial-Omni for self-contained inference in Auden.
"""Length conversions and attention masks for spatial audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Seld233LengthBundle:
    """Bundle together sample-, feature-, SELD-, and spatial-level lengths.

    Attributes:
        sample_lengths:
            Valid waveform lengths in samples, shape `[B]`.
        feature_lengths:
            Valid baseline feature frame lengths after STFT/mel extraction,
            shape `[B]`.
        seld_lengths:
            Valid time lengths seen by the SELD backbone after the baseline
            `feature -> 10 Hz` time reduction, shape `[B]`.
        spatial_token_lengths:
            Valid low-rate spatial token lengths after the spatial token head,
            shape `[B]`.
    """

    sample_lengths: torch.LongTensor
    feature_lengths: torch.LongTensor
    seld_lengths: torch.LongTensor
    spatial_token_lengths: torch.LongTensor


def _ceil_div(values: torch.LongTensor, divisor: int) -> torch.LongTensor:
    """Return `ceil(values / divisor)` for integer tensors."""

    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor}")
    return (values + divisor - 1) // divisor


def clamp_lengths(lengths: torch.LongTensor, max_length: int) -> torch.LongTensor:
    """Clamp 1D integer lengths to `[0, max_length]`.

    Args:
        lengths:
            Tensor of shape `[B]`.
        max_length:
            Scalar upper bound.

    Returns:
        Tensor of shape `[B]` with every element clamped to the valid range.
    """

    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1D, got shape {tuple(lengths.shape)}")
    return lengths.clamp(min=0, max=max_length)


def attention_mask_to_lengths(
    attention_mask: Optional[torch.Tensor],
    max_length: Optional[int] = None,
) -> Optional[torch.LongTensor]:
    """Convert a sample-level attention mask into valid lengths.

    Args:
        attention_mask:
            Tensor of shape `[B, T]`. Non-zero entries are treated as valid.
        max_length:
            Optional upper bound used to clamp the output lengths.

    Returns:
        `None` if the input is `None`, otherwise a tensor of shape `[B]`.
    """

    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [B, T], got {tuple(attention_mask.shape)}"
        )
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    if max_length is not None:
        lengths = clamp_lengths(lengths, max_length=max_length)
    return lengths


def build_1d_attention_mask(
    lengths: torch.LongTensor,
    max_length: Optional[int] = None,
) -> torch.BoolTensor:
    """Build a left-aligned 1D attention mask from valid lengths.

    Args:
        lengths:
            Tensor of shape `[B]`.
        max_length:
            Optional explicit output width. If omitted, `max(lengths)` is used.

    Returns:
        Boolean mask of shape `[B, T_max]`.
    """

    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1D, got shape {tuple(lengths.shape)}")
    if max_length is None:
        max_length = int(lengths.max().item()) if lengths.numel() else 0
    time_index = torch.arange(max_length, device=lengths.device)
    return time_index.unsqueeze(0) < lengths.unsqueeze(1)


def samples_to_feature_frames(
    num_samples: torch.LongTensor,
    hop_length: int = 320,
) -> torch.LongTensor:
    """Convert waveform lengths to baseline feature-frame lengths.

    The DCASE baseline uses `hop_len_s = 0.02` at `16 kHz`, which equals
    `hop_length = 320` samples. The baseline feature extractor effectively uses
    `floor(num_samples / hop_length)` frames.

    Args:
        num_samples:
            Tensor of shape `[B]`.
        hop_length:
            STFT hop length in samples.

    Returns:
        Tensor of shape `[B]` containing feature frame counts.
    """

    if num_samples.ndim != 1:
        raise ValueError(f"num_samples must be 1D, got shape {tuple(num_samples.shape)}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be > 0, got {hop_length}")
    return num_samples // hop_length


def feature_frames_to_seld_frames(
    num_feature_frames: torch.LongTensor,
    feature_to_seld_ratio: int = 5,
) -> torch.LongTensor:
    """Convert baseline feature frames to SELD backbone time steps.

    The DCASE baseline reduces time resolution from `50 Hz` features to `10 Hz`
    backbone frames, so the default ratio is `5`.

    Args:
        num_feature_frames:
            Tensor of shape `[B]`.
        feature_to_seld_ratio:
            Number of feature frames consumed by one SELD time step.

    Returns:
        Tensor of shape `[B]` containing SELD-frame counts.
    """

    if num_feature_frames.ndim != 1:
        raise ValueError(
            f"num_feature_frames must be 1D, got shape {tuple(num_feature_frames.shape)}"
        )
    if feature_to_seld_ratio <= 0:
        raise ValueError(
            f"feature_to_seld_ratio must be > 0, got {feature_to_seld_ratio}"
        )
    return num_feature_frames // feature_to_seld_ratio


def seld_frames_to_spatial_tokens(
    num_seld_frames: torch.LongTensor,
    downsample_factor: int = 4,
) -> torch.LongTensor:
    """Convert SELD backbone time steps to low-rate spatial token lengths.

    Args:
        num_seld_frames:
            Tensor of shape `[B]`.
        downsample_factor:
            Temporal downsampling factor applied by the spatial token head.

    Returns:
        Tensor of shape `[B]` with `max(1, ceil(T_seld / downsample_factor))`
        applied element-wise.
    """

    if num_seld_frames.ndim != 1:
        raise ValueError(
            f"num_seld_frames must be 1D, got shape {tuple(num_seld_frames.shape)}"
        )
    return torch.clamp(_ceil_div(num_seld_frames, downsample_factor), min=1)


def samples_to_spatial_length_bundle(
    num_samples: torch.LongTensor,
    hop_length: int = 320,
    feature_to_seld_ratio: int = 5,
    downsample_factor: int = 4,
) -> Seld233LengthBundle:
    """Convert waveform lengths into all downstream spatial-branch lengths.

    Args:
        num_samples:
            Valid waveform lengths in samples, shape `[B]`.
        hop_length:
            STFT hop length in samples. Default matches `16 kHz * 0.02 s`.
        feature_to_seld_ratio:
            Time reduction ratio from baseline features to SELD backbone frames.
        downsample_factor:
            Spatial token-head reduction factor from `T_seld` to `T_spat`.

    Returns:
        A [`Seld233LengthBundle`] where every field has shape `[B]`.
    """

    feature_lengths = samples_to_feature_frames(num_samples, hop_length=hop_length)
    seld_lengths = feature_frames_to_seld_frames(
        feature_lengths,
        feature_to_seld_ratio=feature_to_seld_ratio,
    )
    spatial_token_lengths = seld_frames_to_spatial_tokens(
        seld_lengths,
        downsample_factor=downsample_factor,
    )
    return Seld233LengthBundle(
        sample_lengths=num_samples,
        feature_lengths=feature_lengths,
        seld_lengths=seld_lengths,
        spatial_token_lengths=spatial_token_lengths,
    )
