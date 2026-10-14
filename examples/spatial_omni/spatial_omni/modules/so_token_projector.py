"""Projection layers used by the spatial modality.

Three variants are exposed for ablation:

* ``SOTokenProjector`` (``type="mlp"``, default):
    The original 2-layer MLP: ``Linear(D_in -> D_h) -> GELU -> Linear(D_h -> D_out)``.
    Matches the LLaVA 1.5 style projector; no normalization.

* ``LayerNormMLPProjector`` (``type="mlp_ln"``):
    Adds input and output ``LayerNorm`` to stabilize joint training with LoRA.
    Preserves token count (no temporal downsampling).

* ``PixelShuffleProjector`` (``type="pixel_shuffle"``):
    Groups every ``k`` consecutive tokens along the time axis into a single
    token whose feature dim is ``k * D_in``, then projects with an MLP.  This
    reduces spatial token count by a factor of ``k``, saving LLM compute.
    The consumer (processor + thinker) must know ``k`` so that placeholder
    count matches projector output.

Use :func:`build_so_token_projector` to construct the variant selected by
config, keeping call-sites identical.
"""

from __future__ import annotations

import torch
from torch import nn


class SOTokenProjector(nn.Module):
    """2-layer MLP projector (LLaVA 1.5 style, no norm). Default variant."""

    variant_name = "mlp"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.shuffle_factor = 1
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        if spatial_tokens.ndim != 3:
            raise ValueError(
                f"spatial_tokens must have shape [B, T_spat, D_in], got {tuple(spatial_tokens.shape)}"
            )
        if spatial_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {spatial_tokens.shape[-1]}"
            )
        target_dtype = self.proj[0].weight.dtype
        if spatial_tokens.dtype != target_dtype:
            spatial_tokens = spatial_tokens.to(dtype=target_dtype)
        return self.proj(spatial_tokens)


class LayerNormMLPProjector(nn.Module):
    """2-layer MLP with input + output LayerNorm.

    Intended to stabilize joint training (projector + LoRA): the input LN
    normalizes features coming from a frozen encoder whose output statistics
    are arbitrary, and the output LN keeps the projected embeddings on the
    same scale as the LLM's token embeddings so LoRA updates don't have to
    fight projector drift.
    """

    variant_name = "mlp_ln"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.shuffle_factor = 1
        self.pre_norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.post_norm = nn.LayerNorm(self.output_dim)

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        if spatial_tokens.ndim != 3:
            raise ValueError(
                f"spatial_tokens must have shape [B, T_spat, D_in], got {tuple(spatial_tokens.shape)}"
            )
        if spatial_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {spatial_tokens.shape[-1]}"
            )
        target_dtype = self.fc1.weight.dtype
        if spatial_tokens.dtype != target_dtype:
            spatial_tokens = spatial_tokens.to(dtype=target_dtype)
        x = self.pre_norm(spatial_tokens)
        x = self.fc2(self.act(self.fc1(x)))
        return self.post_norm(x)


class PixelShuffleProjector(nn.Module):
    """MLP projector with a leading temporal pixel-shuffle.

    Reshape ``[B, T, D]`` into ``[B, T // k, k * D]`` (dropping tail frames
    that don't divide evenly), then apply ``LN -> Linear -> GELU -> Linear ->
    LN``.  Reduces the number of spatial tokens seen by the LLM by ``k``.

    The processor's ``so_backbone_target_token_rate`` MUST be divided by
    ``k`` in the training script so placeholder count matches.
    """

    variant_name = "pixel_shuffle"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        shuffle_factor: int = 2,
    ) -> None:
        super().__init__()
        if shuffle_factor < 1:
            raise ValueError(f"shuffle_factor must be >=1, got {shuffle_factor}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.shuffle_factor = int(shuffle_factor)
        self.shuffled_input_dim = self.input_dim * self.shuffle_factor
        self.pre_norm = nn.LayerNorm(self.shuffled_input_dim)
        self.fc1 = nn.Linear(self.shuffled_input_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.post_norm = nn.LayerNorm(self.output_dim)

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        if spatial_tokens.ndim != 3:
            raise ValueError(
                f"spatial_tokens must have shape [B, T_spat, D_in], got {tuple(spatial_tokens.shape)}"
            )
        if spatial_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {spatial_tokens.shape[-1]}"
            )
        target_dtype = self.fc1.weight.dtype
        if spatial_tokens.dtype != target_dtype:
            spatial_tokens = spatial_tokens.to(dtype=target_dtype)
        B, T, D = spatial_tokens.shape
        k = self.shuffle_factor
        if k > 1:
            T_trunc = T - (T % k)
            if T_trunc == 0:
                # Too-short sequence: pad with zeros to exactly one group.
                pad = spatial_tokens.new_zeros((B, k - T, D))
                spatial_tokens = torch.cat([spatial_tokens, pad], dim=1)
                T_trunc = k
            else:
                spatial_tokens = spatial_tokens[:, :T_trunc]
            spatial_tokens = spatial_tokens.reshape(B, T_trunc // k, k * D)
        x = self.pre_norm(spatial_tokens)
        x = self.fc2(self.act(self.fc1(x)))
        return self.post_norm(x)


def build_so_token_projector(
    projector_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    shuffle_factor: int = 1,
) -> nn.Module:
    """Factory that builds one of the projector variants.

    Args:
        projector_type: ``"mlp"`` | ``"mlp_ln"`` | ``"pixel_shuffle"``.
        input_dim: per-token input feature dim from the encoder.
        hidden_dim: MLP hidden dim.
        output_dim: LLM hidden size.
        shuffle_factor: temporal grouping factor (``pixel_shuffle`` only).

    Returns:
        An ``nn.Module`` that maps ``[B, T, D_in]`` to
        ``[B, T_out, D_out]`` where ``T_out == T`` except for
        ``pixel_shuffle`` where ``T_out == T // shuffle_factor``.
    """
    pt = (projector_type or "mlp").lower()
    if pt == "mlp":
        return SOTokenProjector(input_dim, hidden_dim, output_dim)
    if pt == "mlp_ln":
        return LayerNormMLPProjector(input_dim, hidden_dim, output_dim)
    if pt == "pixel_shuffle":
        return PixelShuffleProjector(
            input_dim, hidden_dim, output_dim, shuffle_factor=shuffle_factor
        )
    raise ValueError(
        f"Unknown projector_type={projector_type!r}; "
        "expected 'mlp' | 'mlp_ln' | 'pixel_shuffle'."
    )
