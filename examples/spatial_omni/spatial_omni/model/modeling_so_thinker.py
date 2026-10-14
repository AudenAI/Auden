# Adapted from Spatial-Omni for self-contained inference in Auden.
"""SO-7B with native audio and a separate SO-Encoder input."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import torch
from torch import nn
import torch.nn.functional as F

from .modeling_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from ..modules.so_token_projector import (
    SOTokenProjector,
    build_so_token_projector,
)
from ..modules.so_encoder import SOEncoder


class Qwen2_5OmniSpatialThinkerForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    """SO-7B Thinker with spatial token injection and optional learned null tokens."""

    config_class = Qwen2_5OmniThinkerConfig

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        self._validate_spatial_config(config)

        encoder_type = getattr(config, "spatial_encoder_type", "so_backbone")

        # Initialize all branch-specific attributes to None so that attribute
        # checks / state_dict loading don't trip depending on which branch is
        # active for this run.
        self.so_encoder = None
        self.so_projector = None

        if encoder_type == "so_backbone":
            # --- Spatial-BEATs encoder path ---
            shuffle_factor = int(getattr(config, "so_projector_shuffle_factor", 1))
            encoder_rate = float(getattr(config, "so_encoder_token_rate", 10.0))
            llm_rate = float(getattr(config, "so_backbone_target_token_rate", 2.5))
            expected_llm_rate = encoder_rate / max(shuffle_factor, 1)
            if abs(expected_llm_rate - llm_rate) > 1e-6:
                raise ValueError(
                    f"so_backbone rate mismatch: encoder_token_rate={encoder_rate} / "
                    f"projector_shuffle_factor={shuffle_factor} = {expected_llm_rate}, "
                    f"but so_backbone_target_token_rate={llm_rate}. "
                    f"Set shuffle_factor={int(round(encoder_rate / llm_rate))} or "
                    f"target_token_rate={expected_llm_rate}."
                )
            self.so_encoder = SOEncoder(
                checkpoint_path=config.so_backbone_checkpoint_path,
                beats_repo_path=config.so_backbone_repo_path,
                freeze_backbone=config.so_backbone_freeze_backbone,
                max_audio_seconds=config.so_backbone_max_audio_seconds,
                encoder_token_rate=encoder_rate,
            )
            self.so_projector = build_so_token_projector(
                projector_type=getattr(config, "so_projector_type", "mlp"),
                input_dim=config.so_encoder_dim,
                hidden_dim=config.so_projector_hidden_dim,
                output_dim=config.text_config.hidden_size,
                shuffle_factor=shuffle_factor,
            )

        # ----- Optional mono-replay support (gated by enable_spatial_replay) -----
        # Mirrors the AF3 implementation: when enabled, allocate a learned
        # `spatial_null` token bank that fills the <|spatial|> placeholders
        # for mono-replay samples, and a small MSE alignment loss between the
        # W-only encoder output and `spatial_null.detach()` keeps the encoder
        # mono-equivariant. Default OFF so existing training is byte-identical.
        self.enable_spatial_replay = bool(getattr(config, "enable_spatial_replay", False))
        if self.enable_spatial_replay and encoder_type == "so_backbone":
            target_rate = float(getattr(config, "so_backbone_target_token_rate", 2.5))
            max_secs = float(getattr(config, "so_backbone_max_audio_seconds", 20.0))
            null_tokens = int(
                getattr(
                    config,
                    "spatial_null_num_tokens",
                    max(1, round(max_secs * target_rate)),
                )
            )
            self.spatial_null = nn.Parameter(
                torch.randn(null_tokens, config.text_config.hidden_size) * 0.02
            )
            self.spatial_null_alignment_weight = float(
                getattr(config, "spatial_null_alignment_weight", 0.05)
            )
        else:
            self.spatial_null = None
            self.spatial_null_alignment_weight = 0.0
        self._last_spatial_replay_stats: Dict[str, float] = {}
        self.post_init()

    def reinit_spatial_null_if_needed(self, std: float = 0.02) -> bool:
        """Re-initialize `spatial_null` if it is on `meta` device or contains non-finite
        values. This guards against HF `from_pretrained(torch_dtype=...)` materializing
        new (subclass-introduced) parameters from uninitialized memory: parameters not
        present in the pretrained checkpoint never go through `_init_weights`, so the
        `randn * 0.02` from `__init__` never actually lands and the tensor stays as
        NaN/Inf garbage. This is critical for the mono-replay path because
        `spatial_null` is injected into `inputs_embeds` for replay samples and any
        non-finite value there propagates into the LM and produces NaN CE loss.
        Returns True iff a re-init was performed.
        """
        if self.spatial_null is None:
            return False
        p = self.spatial_null
        needs = bool(p.is_meta) or bool(torch.isnan(p).any().item()) or bool(torch.isinf(p).any().item())
        if not needs:
            return False
        with torch.no_grad():
            new_data = (
                torch.randn(p.shape, device=("cpu" if p.is_meta else p.device))
                * float(std)
            ).to(dtype=p.dtype if not p.is_meta else torch.float32)
            if p.is_meta:
                # Replace the meta parameter with a real one.
                self.spatial_null = nn.Parameter(new_data)
            else:
                p.data.copy_(new_data.to(dtype=p.dtype, device=p.device))
        return True

    def get_output_embeddings(self):
        """Return the LM head so tokenizer resizing can also update output embeddings."""

        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Replace the LM head after tokenizer-resize operations."""

        self.lm_head = new_embeddings

    def sync_spatial_tokenizer(self, tokenizer, spatial_token: str = "<|spatial|>") -> int:
        """Synchronize tokenizer vocab, model embeddings, and spatial token id.

        Processing:
            1. Ensure `<|spatial|>` exists in the tokenizer.
            2. Resize thinker embeddings when tokenizer vocab size changes.
            3. Store the resolved token id in the thinker config.

        Returns:
            Integer token id assigned to `<|spatial|>`.
        """

        vocab = tokenizer.get_vocab()
        if spatial_token not in vocab:
            tokenizer.add_special_tokens({"additional_special_tokens": [spatial_token]})
        spatial_token_id = int(tokenizer.convert_tokens_to_ids(spatial_token))
        new_vocab_size = len(tokenizer)
        current_vocab_size = int(self.get_input_embeddings().num_embeddings)
        if current_vocab_size != new_vocab_size:
            self.resize_token_embeddings(new_vocab_size)
        self.config.spatial_token_index = spatial_token_id
        self.config.text_config.vocab_size = new_vocab_size
        self.vocab_size = new_vocab_size
        return spatial_token_id

    def _validate_spatial_config(self, config) -> None:
        if getattr(config, "spatial_encoder_type", "so_backbone") != "so_backbone":
            raise ValueError("This example supports the SO-Encoder backbone only.")
        if not config.so_backbone_checkpoint_path:
            raise ValueError("so_backbone_checkpoint_path is required.")

    def forward(
        self,
        *args,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        seld_features: Optional[torch.Tensor] = None,
        seld_feature_attention_mask: Optional[torch.Tensor] = None,
        seld_feature_lengths: Optional[torch.LongTensor] = None,
        seld_hidden_states: Optional[torch.Tensor] = None,
        seld_hidden_attention_mask: Optional[torch.Tensor] = None,
        seld_hidden_lengths: Optional[torch.LongTensor] = None,
        spatial_tokens: Optional[torch.Tensor] = None,
        projected_spatial_tokens: Optional[torch.Tensor] = None,
        spatial_token_lengths: Optional[torch.LongTensor] = None,
        has_spatial: Optional[torch.BoolTensor] = None,
        mono_audio: Optional[torch.Tensor] = None,
        mono_audio_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Run the thinker with optional SELD233 spatial-token injection.

        Args:
            input_ids:
                The text token ids that may contain repeated `<|spatial|>`
                placeholders, shape `[B, T_text]`. This value can arrive via
                `kwargs["input_ids"]` or positional `args[0]`.
            spatial_audio:
                Optional raw FOA batch `[B, T_audio_max, 4]`.
            spatial_audio_attention_mask:
                Optional waveform mask `[B, T_audio_max]`.
            spatial_audio_lengths:
                Optional waveform lengths `[B]`.
            seld_features:
                Optional offline baseline features `[B, 7, T_feat_max, 64]`.
            seld_feature_attention_mask:
                Optional feature mask `[B, T_feat_max]`.
            seld_feature_lengths:
                Optional feature lengths `[B]`.
            seld_hidden_states:
                Optional cached SELD hidden states `[B, T_seld_max, 128]`.
            seld_hidden_attention_mask:
                Optional hidden mask `[B, T_seld_max]`.
            seld_hidden_lengths:
                Optional hidden lengths `[B]`.
            spatial_tokens:
                Optional direct spatial-token input `[B, T_spat_max, D_spat]`.
            projected_spatial_tokens:
                Optional direct post-projector spatial embeddings
                `[B, T_spat_max, D_llm]`.
            spatial_token_lengths:
                Optional valid token counts `[B]`.

        Returns:
            The same causal-LM output type as the base thinker. When spatial
            inputs are provided, the function first injects the projected
            spatial tokens into `inputs_embeds`, then delegates the rest of the
            forward pass to the original thinker implementation.
        """

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        pixel_values = kwargs.get("pixel_values")
        pixel_values_videos = kwargs.get("pixel_values_videos")

        if not self._has_spatial_inputs(
            spatial_audio=spatial_audio,
            seld_features=seld_features,
            seld_hidden_states=seld_hidden_states,
            spatial_tokens=spatial_tokens,
            projected_spatial_tokens=projected_spatial_tokens,
            has_spatial=has_spatial,
        ):
            return super().forward(*args, **kwargs)

        if kwargs.get("use_audio_in_video"):
            raise NotImplementedError("Spatial video support requires use_audio_in_video=False.")

        if pixel_values is not None:
            raise NotImplementedError(
                "This example does not support image and spatial input together."
            )

        if input_ids is None:
            raise ValueError("input_ids are required when injecting spatial tokens.")

        loss_null = None
        replay_stats: Dict[str, float] = {}
        if has_spatial is not None:
            # ----- Mixed spatial+mono replay (mono replay path) -----------
            # Resolves projected spatial embeddings for a mixed batch where
            # `has_spatial[i]==False` indicates a mono-replay sample whose
            # <|spatial|> placeholders should be filled by `spatial_null`,
            # while `has_spatial[i]==True` runs the normal SOBackbone encoder
            # path. Returns the W-only encoder output's MSE alignment loss.
            current_encoder_type = getattr(self.config, "spatial_encoder_type", "so_backbone")
            if current_encoder_type != "so_backbone" or self.spatial_null is None:
                raise ValueError(
                    "has_spatial replay path requires spatial_encoder_type='so_backbone' "
                    "and config.enable_spatial_replay=True (so the spatial_null parameter exists)."
                )
            (
                projected_spatial,
                spatial_token_lengths,
                loss_null,
                replay_stats,
            ) = self._resolve_mixed_replay_spatial(
                input_ids=input_ids,
                spatial_audio=spatial_audio,
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
                has_spatial=has_spatial,
                mono_audio=mono_audio,
                mono_audio_lengths=mono_audio_lengths,
            )
        elif projected_spatial_tokens is not None:
            if projected_spatial_tokens.ndim != 3:
                raise ValueError(
                    "projected_spatial_tokens must have shape [B, T_spat, D_llm], "
                    f"got {tuple(projected_spatial_tokens.shape)}"
                )
            projected_spatial = projected_spatial_tokens
            if spatial_token_lengths is None:
                spatial_token_lengths = projected_spatial.new_full(
                    (projected_spatial.shape[0],),
                    fill_value=projected_spatial.shape[1],
                    dtype=torch.long,
                )
        else:
            spatial_tokens, spatial_token_lengths = self._resolve_spatial_tokens(
                spatial_audio=spatial_audio,
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
                seld_features=seld_features,
                seld_feature_attention_mask=seld_feature_attention_mask,
                seld_feature_lengths=seld_feature_lengths,
                seld_hidden_states=seld_hidden_states,
                seld_hidden_attention_mask=seld_hidden_attention_mask,
                seld_hidden_lengths=seld_hidden_lengths,
                spatial_tokens=spatial_tokens,
                spatial_token_lengths=spatial_token_lengths,
            )
            current_encoder_type = getattr(self.config, "spatial_encoder_type", "so_backbone")
            if current_encoder_type == "so_backbone":
                projected_spatial = self.so_projector(spatial_tokens)
                # Pixel-shuffle projectors reduce T by shuffle_factor; rescale
                # spatial_token_lengths so downstream alignment sees matching sizes.
                shuffle_factor = int(getattr(self.so_projector, "shuffle_factor", 1))
                if shuffle_factor > 1:
                    new_lengths = torch.clamp(
                        torch.div(spatial_token_lengths, shuffle_factor, rounding_mode="floor"),
                        min=0,
                        max=int(projected_spatial.shape[1]),
                    )
                    # Ensure at least 1 token when original had any content, to keep
                    # alignment consistent with padded batches.
                    nonzero = spatial_token_lengths > 0
                    new_lengths = torch.where(
                        nonzero & (new_lengths == 0),
                        torch.ones_like(new_lengths),
                        new_lengths,
                    )
                    spatial_token_lengths = new_lengths
        projected_spatial, spatial_token_lengths = self._align_projected_spatial_to_placeholders(
            projected_spatial=projected_spatial,
            spatial_token_lengths=spatial_token_lengths,
            input_ids=input_ids,
        )
        projected_spatial = self._flatten_projected_spatial(
            projected_spatial=projected_spatial,
            spatial_token_lengths=spatial_token_lengths,
        )
        inputs_embeds = self.get_input_embeddings()(input_ids)
        spatial_mask = self._build_spatial_mask(input_ids, inputs_embeds)
        self._validate_spatial_mask_count(
            spatial_mask=spatial_mask,
            projected_spatial=projected_spatial,
            spatial_token_lengths=spatial_token_lengths,
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            spatial_mask,
            projected_spatial.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        kwargs["inputs_embeds"] = inputs_embeds
        kwargs["spatial_features"] = None
        kwargs["spatial_audio"] = None
        out = super().forward(*args, **kwargs)
        # Mixed-replay bookkeeping: add the W-only null-alignment MSE term to
        # the LM loss and stash per-batch stats for the trainer to log. When
        # `has_spatial` is None (default training), this branch is skipped and
        # behavior is bit-identical to the pre-replay path.
        if has_spatial is not None and loss_null is not None:
            try:
                base_loss = out.loss if hasattr(out, "loss") else None
            except Exception:
                base_loss = None
            if base_loss is not None:
                out.loss = base_loss + float(self.spatial_null_alignment_weight) * loss_null
            replay_stats.setdefault("loss_null", float(loss_null.detach()))
            replay_stats.setdefault(
                "loss_ce",
                float(base_loss.detach()) if base_loss is not None else 0.0,
            )
            replay_stats.setdefault(
                "loss_total",
                float(out.loss.detach())
                if (hasattr(out, "loss") and out.loss is not None)
                else replay_stats["loss_ce"],
            )
            self._last_spatial_replay_stats = replay_stats
            try:
                out.loss_ce = base_loss
                out.loss_null = loss_null
            except Exception:
                pass
        return out

    def _has_spatial_inputs(
        self,
        spatial_audio: Optional[torch.Tensor],
        seld_features: Optional[torch.Tensor],
        seld_hidden_states: Optional[torch.Tensor],
        spatial_tokens: Optional[torch.Tensor],
        projected_spatial_tokens: Optional[torch.Tensor],
        has_spatial: Optional[torch.Tensor] = None,
    ) -> bool:
        """Return `True` when any spatial input path is active.

        Inputs:
            - `spatial_audio`: optional `[B, T_audio_max, 4]`
            - `seld_features`: optional `[B, 7, T_feat_max, 64]`
            - `seld_hidden_states`: optional `[B, T_seld_max, 128]`
            - `spatial_tokens`: optional `[B, T_spat_max, D_spat]`
            - `projected_spatial_tokens`: optional `[B, T_spat_max, D_llm]`

        Output:
            Boolean scalar. `True` means the spatial branch should be engaged.
        """

        return (
            spatial_audio is not None
            or seld_features is not None
            or seld_hidden_states is not None
            or spatial_tokens is not None
            or projected_spatial_tokens is not None
            or has_spatial is not None
        )

    def _resolve_spatial_tokens(
        self,
        spatial_audio: Optional[torch.Tensor],
        spatial_audio_attention_mask: Optional[torch.Tensor],
        spatial_audio_lengths: Optional[torch.LongTensor],
        seld_features: Optional[torch.Tensor],
        seld_feature_attention_mask: Optional[torch.Tensor],
        seld_feature_lengths: Optional[torch.LongTensor],
        seld_hidden_states: Optional[torch.Tensor],
        seld_hidden_attention_mask: Optional[torch.Tensor],
        seld_hidden_lengths: Optional[torch.LongTensor],
        spatial_tokens: Optional[torch.Tensor],
        spatial_token_lengths: Optional[torch.LongTensor],
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Resolve spatial tokens from direct input or SO-Encoder.

        Returns:
            Tuple `(tokens, lengths)` where:
            - `tokens` has shape `[B, T_spat_max, D_spat]`
            - `lengths` has shape `[B]`
        """

        if spatial_tokens is not None:
            if spatial_tokens.ndim != 3:
                raise ValueError(
                    f"spatial_tokens must have shape [B, T_spat, D_spat], got {tuple(spatial_tokens.shape)}"
                )
            if spatial_token_lengths is None:
                spatial_token_lengths = spatial_tokens.new_full(
                    (spatial_tokens.shape[0],),
                    fill_value=spatial_tokens.shape[1],
                    dtype=torch.long,
                )
            return spatial_tokens, spatial_token_lengths

        current_encoder_type = getattr(self.config, "spatial_encoder_type", "so_backbone")
        if current_encoder_type == "so_backbone":
            if spatial_audio is None:
                raise ValueError(
                    "spatial_audio is required for the so_backbone encoder path "
                    "when spatial_tokens is not provided directly."
                )
            beats_output = self.so_encoder(
                spatial_audio=spatial_audio,
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
            )
            return beats_output.spatial_tokens, beats_output.spatial_token_lengths

        raise ValueError("This example supports the SO-Encoder backbone only.")

    def _build_spatial_mask(self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor) -> torch.BoolTensor:
        """Build the expanded `masked_scatter` mask for `<|spatial|>` token positions.

        Input:
            - `input_ids`: `[B, T_text]`
            - `inputs_embeds`: `[B, T_text, D_llm]`

        Output:
            Boolean mask `[B, T_text, D_llm]` that is `True` exactly at
            `<|spatial|>` positions.
        """

        if self.config.spatial_token_index is None:
            raise ValueError("config.spatial_token_index must be set before using the spatial thinker.")
        return (
            (input_ids == self.config.spatial_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

    def _flatten_projected_spatial(
        self,
        projected_spatial: torch.Tensor,
        spatial_token_lengths: torch.LongTensor,
    ) -> torch.Tensor:
        """Pack only valid spatial tokens for `masked_scatter`.

        Input:
            - `projected_spatial`: `[B, T_spat_max, D_llm]`
            - `spatial_token_lengths`: `[B]`

        Processing:
            For each sample `i`, keep `projected_spatial[i, :T_i]` where
            `T_i = spatial_token_lengths[i]`, then concatenate valid rows across
            the batch.

        Output:
            Tensor `[sum_i T_i, D_llm]`, safe for `masked_scatter` when batch
            items have different spatial lengths.
        """

        if projected_spatial.ndim != 3:
            raise ValueError(
                "projected_spatial must have shape [B, T_spat_max, D_llm], "
                f"got {tuple(projected_spatial.shape)}"
            )
        if spatial_token_lengths.ndim != 1 or spatial_token_lengths.shape[0] != projected_spatial.shape[0]:
            raise ValueError(
                "spatial_token_lengths must have shape [B] matching projected_spatial batch size, "
                f"got {tuple(spatial_token_lengths.shape)}"
            )

        valid_rows = []
        max_tokens = projected_spatial.shape[1]
        for index, length in enumerate(spatial_token_lengths.tolist()):
            if length < 0 or length > max_tokens:
                raise ValueError(
                    f"spatial_token_lengths[{index}]={length} is outside [0, {max_tokens}]"
                )
            if length == 0:
                continue
            valid_rows.append(projected_spatial[index, :length])

        if not valid_rows:
            return projected_spatial.new_zeros((0, projected_spatial.shape[-1]))
        return torch.cat(valid_rows, dim=0)

    # ------------------------------------------------------------------ #
    # Mono-replay helpers (only active when config.enable_spatial_replay) #
    # ------------------------------------------------------------------ #
    def get_spatial_null(
        self,
        batch_size: int,
        token_lengths: Optional[torch.LongTensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the learned `spatial_null` token bank padded to max(T_i)."""

        if self.spatial_null is None:
            raise RuntimeError(
                "spatial_null is not allocated. Set config.enable_spatial_replay=True."
            )
        base = self.spatial_null
        if device is not None or dtype is not None:
            base = base.to(
                device=device if device is not None else base.device,
                dtype=dtype if dtype is not None else base.dtype,
            )
        if token_lengths is None:
            target_len = int(base.shape[0])
        else:
            target_len = int(token_lengths.max().item()) if token_lengths.numel() else 0
        if target_len <= 0:
            return base.new_zeros((batch_size, 0, base.shape[-1]))
        if target_len <= base.shape[0]:
            tokens = base[:target_len]
        else:
            pad = base[-1:].expand(target_len - base.shape[0], -1)
            tokens = torch.cat([base, pad], dim=0)
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)

    def _project_spatial_audio(
        self,
        spatial_audio: torch.Tensor,
        spatial_audio_attention_mask: Optional[torch.Tensor],
        spatial_audio_lengths: Optional[torch.LongTensor],
    ):
        """Run SOBackbone encoder + projector and return (projected, lengths)."""

        out = self.so_encoder(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
        )
        projected = self.so_projector(out.spatial_tokens)
        lengths = out.spatial_token_lengths
        shuffle_factor = int(getattr(self.so_projector, "shuffle_factor", 1))
        if shuffle_factor > 1:
            new_lengths = torch.clamp(
                torch.div(lengths, shuffle_factor, rounding_mode="floor"),
                min=0,
                max=int(projected.shape[1]),
            )
            nonzero = lengths > 0
            lengths = torch.where(
                nonzero & (new_lengths == 0),
                torch.ones_like(new_lengths),
                new_lengths,
            )
        return projected, lengths

    @staticmethod
    def _build_w_only_audio(
        mono_audio: torch.Tensor,
        mono_audio_lengths: Optional[torch.LongTensor],
    ):
        """Pack mono waveform into a `[B, T, 4]` FOA tensor with W=mono, X=Y=Z=0."""

        if mono_audio.ndim == 3:
            if mono_audio.shape[1] == 1:
                mono_audio = mono_audio[:, 0, :]
            elif mono_audio.shape[-1] == 1:
                mono_audio = mono_audio[..., 0]
            else:
                raise ValueError(
                    "mono_audio must have shape [B,T], [B,1,T], or [B,T,1], "
                    f"got {tuple(mono_audio.shape)}"
                )
        if mono_audio.ndim != 2:
            raise ValueError(
                f"mono_audio must be 2D after squeeze, got {tuple(mono_audio.shape)}"
            )
        # Spatial-BEATs expects channels-last [B, T, 4].
        w_only = mono_audio.new_zeros((mono_audio.shape[0], mono_audio.shape[1], 4))
        w_only[:, :, 0] = mono_audio
        if mono_audio_lengths is None:
            mono_audio_lengths = mono_audio.new_full(
                (mono_audio.shape[0],), mono_audio.shape[1], dtype=torch.long
            )
        return w_only, mono_audio_lengths

    def _resolve_mixed_replay_spatial(
        self,
        input_ids: torch.LongTensor,
        spatial_audio: Optional[torch.Tensor],
        spatial_audio_attention_mask: Optional[torch.Tensor],
        spatial_audio_lengths: Optional[torch.LongTensor],
        has_spatial: torch.BoolTensor,
        mono_audio: Optional[torch.Tensor],
        mono_audio_lengths: Optional[torch.LongTensor],
    ):
        """Build projected spatial embeddings for a mixed FOA + mono batch.

        Behavior:
            * `has_spatial[i]==True`  → run SOBackbone encoder + projector
              on `spatial_audio[i]` (shape `[B, T, 4]`), align to placeholder
              count.
            * `has_spatial[i]==False` → fill the placeholders with copies of
              the learned `spatial_null` token bank. Concurrently feed
              `[mono,0,0,0]` through SOBackbone and compute MSE between the
              W-only encoder output and `spatial_null.detach()` so the
              encoder learns to produce the null state for mono input.
        """

        spatial_id = int(getattr(self.config, "spatial_token_index", -1) or -1)
        if spatial_id < 0:
            raise ValueError("config.spatial_token_index is not set.")
        has_spatial = has_spatial.to(device=input_ids.device, dtype=torch.bool)
        placeholder_counts = (input_ids == spatial_id).sum(dim=1).to(dtype=torch.long)
        B = int(input_ids.shape[0])
        embed_dtype = self.get_input_embeddings().weight.dtype
        null_tokens = self.get_spatial_null(
            B,
            token_lengths=placeholder_counts,
            device=input_ids.device,
            dtype=embed_dtype,
        )
        projected_spatial = null_tokens.clone()
        spatial_token_lengths = placeholder_counts.to(device=projected_spatial.device)

        if bool(has_spatial.any().item()):
            if spatial_audio is None:
                raise ValueError("spatial_audio is required for has_spatial=True samples.")
            real_idx = has_spatial.nonzero(as_tuple=True)[0]
            real_audio = spatial_audio.index_select(0, real_idx)
            real_mask = (
                spatial_audio_attention_mask.index_select(0, real_idx)
                if spatial_audio_attention_mask is not None
                else None
            )
            real_lengths = (
                spatial_audio_lengths.index_select(0, real_idx)
                if spatial_audio_lengths is not None
                else None
            )
            real_projected, real_token_lengths = self._project_spatial_audio(
                real_audio, real_mask, real_lengths,
            )
            real_projected, real_token_lengths = self._align_projected_spatial_to_placeholders(
                projected_spatial=real_projected,
                spatial_token_lengths=real_token_lengths,
                input_ids=input_ids.index_select(0, real_idx),
            )
            max_real = int(real_projected.shape[1])
            if max_real > projected_spatial.shape[1]:
                pad = projected_spatial[:, -1:, :].expand(
                    B, max_real - projected_spatial.shape[1], projected_spatial.shape[-1]
                )
                projected_spatial = torch.cat([projected_spatial, pad], dim=1)
            projected_spatial[real_idx, :max_real, :] = real_projected.to(
                device=projected_spatial.device,
                dtype=projected_spatial.dtype,
            )
            spatial_token_lengths[real_idx] = real_token_lengths.to(spatial_token_lengths.device)

        replay_mask = ~has_spatial
        loss_null = projected_spatial.new_zeros(())
        stats: Dict[str, float] = {
            "spatial_samples": float(has_spatial.sum().detach().item()),
            "replay_samples": float(replay_mask.sum().detach().item()),
        }
        if bool(replay_mask.any().item()) and mono_audio is not None:
            replay_idx = replay_mask.nonzero(as_tuple=True)[0]
            replay_mono = mono_audio.index_select(0, replay_idx)
            replay_lengths = (
                mono_audio_lengths.index_select(0, replay_idx)
                if mono_audio_lengths is not None
                else None
            )
            w_only_audio, w_only_lengths = self._build_w_only_audio(replay_mono, replay_lengths)
            w_projected, w_lengths = self._project_spatial_audio(
                w_only_audio,
                spatial_audio_attention_mask=None,
                spatial_audio_lengths=w_only_lengths,
            )
            w_projected, w_lengths = self._align_projected_spatial_to_placeholders(
                projected_spatial=w_projected,
                spatial_token_lengths=w_lengths,
                input_ids=input_ids.index_select(0, replay_idx),
            )
            replay_targets = self.get_spatial_null(
                int(replay_idx.numel()),
                token_lengths=w_lengths,
                device=w_projected.device,
                dtype=w_projected.dtype,
            ).detach()
            w_flat = self._flatten_projected_spatial(w_projected, w_lengths)
            target_flat = self._flatten_projected_spatial(replay_targets, w_lengths)
            if w_flat.numel() > 0:
                loss_null = F.mse_loss(w_flat, target_flat)
                stats["w_only_tokens_norm"] = float(w_flat.norm(dim=-1).mean().detach())
                stats["spatial_null_norm"] = float(target_flat.norm(dim=-1).mean().detach())
                stats["w_only_null_cosine"] = float(
                    F.cosine_similarity(w_flat.float(), target_flat.float(), dim=-1)
                    .mean()
                    .detach()
                )
        else:
            null_ref = self.get_spatial_null(
                1,
                token_lengths=placeholder_counts[:1].clamp(min=1),
                device=projected_spatial.device,
                dtype=projected_spatial.dtype,
            )
            stats["spatial_null_norm"] = float(null_ref.norm(dim=-1).mean().detach())
            stats["w_only_tokens_norm"] = 0.0
            stats["w_only_null_cosine"] = 0.0
        return projected_spatial, spatial_token_lengths, loss_null, stats

    def _align_projected_spatial_to_placeholders(
        self,
        projected_spatial: torch.Tensor,
        spatial_token_lengths: torch.LongTensor,
        input_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Align projected spatial rows to the actual placeholder count in text.

        This makes spatial injection robust to occasional off-by-one differences
        between processor-side length estimation and encoder-side token counts.
        """

        if self.config.spatial_token_index is None:
            raise ValueError("config.spatial_token_index must be set before using the spatial thinker.")
        placeholder_counts = (input_ids == self.config.spatial_token_index).sum(dim=1).to(
            device=spatial_token_lengths.device,
            dtype=torch.long,
        )
        if torch.equal(placeholder_counts, spatial_token_lengths):
            return projected_spatial, spatial_token_lengths

        batch_size, _, hidden_dim = projected_spatial.shape
        target_max_tokens = int(placeholder_counts.max().item()) if batch_size > 0 else 0
        aligned = projected_spatial.new_zeros((batch_size, target_max_tokens, hidden_dim))
        source_max_tokens = projected_spatial.shape[1]

        for index, (source_len, target_len) in enumerate(
            zip(spatial_token_lengths.tolist(), placeholder_counts.tolist())
        ):
            if source_len < 0 or source_len > source_max_tokens:
                raise ValueError(
                    f"spatial_token_lengths[{index}]={source_len} is outside [0, {source_max_tokens}]"
                )
            if target_len < 0:
                raise ValueError(f"Placeholder count for sample {index} must be non-negative, got {target_len}")
            if target_len == 0:
                continue
            copy_len = min(source_len, target_len)
            if copy_len > 0:
                aligned[index, :copy_len] = projected_spatial[index, :copy_len]
            if target_len > source_len and source_len > 0:
                aligned[index, copy_len:target_len] = projected_spatial[index, source_len - 1].unsqueeze(0)

        return aligned, placeholder_counts

    def _validate_spatial_mask_count(
        self,
        spatial_mask: torch.BoolTensor,
        projected_spatial: torch.Tensor,
        spatial_token_lengths: torch.LongTensor,
    ) -> None:
        """Check that placeholder count matches projected token count.

        Inputs:
            - `spatial_mask`: `[B, T_text, D_llm]`
            - `projected_spatial`: `[B, T_spat_max, D_llm]`
            - `spatial_token_lengths`: `[B]`

        Behavior:
            Validates that the number of `<|spatial|>` placeholders in
            `input_ids` matches `sum(spatial_token_lengths)`. This guarantees
            that `masked_scatter` can replace placeholders with projected
            spatial-token embeddings without a shape mismatch.
        """

        expected_tokens = int(spatial_token_lengths.sum().item())
        actual_tokens = int(spatial_mask[..., 0].sum().item())
        if actual_tokens != expected_tokens:
            raise ValueError(
                "Spatial placeholder count does not match projected token count: "
                f"{actual_tokens} vs {expected_tokens}"
            )
        if projected_spatial.ndim != 2:
            raise ValueError(
                "Packed projected spatial tokens must have shape [sum(T_i), D_llm], "
                f"got {tuple(projected_spatial.shape)}"
            )

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ):
        """Return multimodal RoPE positions for `video + audio + spatial`.

        Supported modes:
        - text only
        - text + audio
        - text + audio + spatial
        - text + video + audio + spatial

        Constraints:
        - `use_audio_in_video=False`
        - image inputs remain unsupported in the spatial subclass
        - modal order inside each sample must be:
          `<|VIDEO|><|AUDIO|><|spatial|>` when video exists
          `<|AUDIO|><|spatial|>` when video is absent
        """

        if input_ids is None or self.config.spatial_token_index is None:
            return super().get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_seqlens,
                second_per_grids=second_per_grids,
            )

        has_spatial = bool((input_ids == self.config.spatial_token_index).any())
        has_video = video_grid_thw is not None

        if not has_spatial and not has_video:
            return super().get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_seqlens,
                second_per_grids=second_per_grids,
            )

        if image_grid_thw is not None:
            raise NotImplementedError("Spatial RoPE does not support image and spatial input together.")
        if use_audio_in_video:
            raise NotImplementedError("Spatial video support requires use_audio_in_video=False.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        spatial_merge_size = self.spatial_merge_size
        audio_token_id = self.config.audio_token_id
        video_token_id = self.config.video_token_id
        spatial_token_id = self.config.spatial_token_index
        position_id_per_seconds = self.config.position_id_per_seconds

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        attention_mask = attention_mask.to(input_ids.device)
        mrope_position_deltas = []
        video_idx = 0

        for batch_idx, sample_input_ids in enumerate(input_ids):
            valid_tokens = sample_input_ids[attention_mask[batch_idx] == 1]
            segment_positions = self._build_segment_positions(
                valid_tokens=valid_tokens,
                video_token_id=video_token_id,
                audio_token_id=audio_token_id,
                spatial_token_id=spatial_token_id,
                video_grid_thw=video_grid_thw,
                second_per_grids=second_per_grids,
                spatial_merge_size=spatial_merge_size,
                position_id_per_seconds=position_id_per_seconds,
                video_idx=video_idx,
            )
            llm_positions = segment_positions["position_ids"]
            video_idx = segment_positions["next_video_idx"]
            position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(valid_tokens))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def _build_segment_positions(
        self,
        valid_tokens: torch.LongTensor,
        video_token_id: int,
        audio_token_id: int,
        spatial_token_id: int,
        video_grid_thw: Optional[torch.LongTensor],
        second_per_grids: Optional[torch.Tensor],
        spatial_merge_size: int,
        position_id_per_seconds: int,
        video_idx: int,
    ) -> dict[str, torch.Tensor | int]:
        """Assign RoPE positions by scanning contiguous token runs.

        Input:
            - `valid_tokens`: `[T_valid]`

        Output:
            Dictionary with:
            - `position_ids`: `[3, T_valid]`
            - `next_video_idx`: global video-grid cursor after consuming this sample
        """

        modal_token_ids = {video_token_id, audio_token_id, spatial_token_id}
        modal_order = []
        llm_pos_ids_list = []
        cursor = 0
        max_token_count = len(valid_tokens)

        while cursor < max_token_count:
            token_id = int(valid_tokens[cursor].item())
            if token_id in modal_token_ids:
                end = cursor
                while end < max_token_count and int(valid_tokens[end].item()) == token_id:
                    end += 1
                run_length = end - cursor
                start_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                if token_id == video_token_id:
                    modal_order.append("video")
                    if video_grid_thw is None or second_per_grids is None:
                        raise ValueError("video tokens are present but video_grid_thw/second_per_grids are missing.")
                    expected_video_len = int(video_grid_thw[video_idx].prod().item() // (spatial_merge_size**2))
                    if run_length != expected_video_len:
                        raise ValueError(
                            "Video placeholder count does not match video token count: "
                            f"{run_length} vs {expected_video_len}"
                        )
                    grid_t = int(video_grid_thw[video_idx][0].item())
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        int(start_idx),
                        video_idx,
                        spatial_merge_size,
                        t_index,
                        grid_hs,
                        grid_ws,
                    )
                    video_idx += 1
                elif token_id == audio_token_id:
                    modal_order.append("audio")
                    llm_pos_ids = torch.arange(run_length, device=valid_tokens.device).view(1, -1).expand(3, -1)
                    llm_pos_ids = llm_pos_ids + int(start_idx)
                else:
                    modal_order.append("spatial")
                    llm_pos_ids = torch.arange(run_length, device=valid_tokens.device).view(1, -1).expand(3, -1)
                    llm_pos_ids = llm_pos_ids + int(start_idx)

                llm_pos_ids_list.append(llm_pos_ids)
                cursor = end
                continue

            end = cursor
            while end < max_token_count and int(valid_tokens[end].item()) not in modal_token_ids:
                end += 1
            text_length = end - cursor
            start_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                torch.arange(text_length, device=valid_tokens.device).view(1, -1).expand(3, -1) + int(start_idx)
            )
            cursor = end

        self._validate_modal_order(modal_order)
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != max_token_count:
            raise ValueError(
                f"RoPE position construction length mismatch: {llm_positions.shape[1]} vs {max_token_count}"
            )
        return {"position_ids": llm_positions, "next_video_idx": video_idx}

    def _validate_modal_order(self, modal_order: list[str]) -> None:
        """Validate the required modal order for the spatial subclass."""

        if not modal_order:
            return
        if modal_order not in (["audio", "spatial"], ["video", "audio", "spatial"]):
            raise ValueError(
                "Unsupported modal order for the spatial subclass. "
                "Expected ['audio', 'spatial'] or ['video', 'audio', 'spatial'], "
                f"got {modal_order}."
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        spatial_audio=None,
        spatial_audio_attention_mask=None,
        spatial_audio_lengths=None,
        seld_features=None,
        seld_feature_attention_mask=None,
        seld_feature_lengths=None,
        seld_hidden_states=None,
        seld_hidden_attention_mask=None,
        seld_hidden_lengths=None,
        spatial_tokens=None,
        projected_spatial_tokens=None,
        spatial_token_lengths=None,
        has_spatial=None,
        mono_audio=None,
        mono_audio_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        """Keep spatial inputs during prefill and drop them during decode.

        Prefill stage:
            Preserve the spatial raw-audio / feature / token tensors so the
            thinker can build or inject spatial embeddings once.

        Decode stage:
            After cache is active (`cache_position[0] != 0`), clear all spatial
            branch tensors to avoid recomputing the same spatial tokens for
            every autoregressive step.
        """

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
        model_inputs["spatial_audio"] = spatial_audio
        model_inputs["spatial_audio_attention_mask"] = spatial_audio_attention_mask
        model_inputs["spatial_audio_lengths"] = spatial_audio_lengths
        model_inputs["seld_features"] = seld_features
        model_inputs["seld_feature_attention_mask"] = seld_feature_attention_mask
        model_inputs["seld_feature_lengths"] = seld_feature_lengths
        model_inputs["seld_hidden_states"] = seld_hidden_states
        model_inputs["seld_hidden_attention_mask"] = seld_hidden_attention_mask
        model_inputs["seld_hidden_lengths"] = seld_hidden_lengths
        model_inputs["spatial_tokens"] = spatial_tokens
        model_inputs["projected_spatial_tokens"] = projected_spatial_tokens
        model_inputs["spatial_token_lengths"] = spatial_token_lengths
        model_inputs["has_spatial"] = has_spatial
        model_inputs["mono_audio"] = mono_audio
        model_inputs["mono_audio_lengths"] = mono_audio_lengths

        prepared_input_ids = model_inputs.get("input_ids")
        prepared_inputs_embeds = model_inputs.get("inputs_embeds")
        prepared_seq_len = None
        if prepared_input_ids is not None and hasattr(prepared_input_ids, "shape") and prepared_input_ids.ndim >= 2:
            prepared_seq_len = int(prepared_input_ids.shape[1])

        is_decode_step = bool(
            past_key_values is not None
            and prepared_seq_len is not None
            and prepared_seq_len <= 1
            and prepared_inputs_embeds is None
        )

        if is_decode_step:
            for key in (
                "spatial_audio",
                "spatial_audio_attention_mask",
                "spatial_audio_lengths",
                "seld_features",
                "seld_feature_attention_mask",
                "seld_feature_lengths",
                "seld_hidden_states",
                "seld_hidden_attention_mask",
                "seld_hidden_lengths",
                "spatial_tokens",
                "projected_spatial_tokens",
                "spatial_token_lengths",
                "has_spatial",
                "mono_audio",
                "mono_audio_lengths",
            ):
                model_inputs[key] = None
        return model_inputs


class Qwen2_5OmniSpatialForConditionalGeneration(Qwen2_5OmniForConditionalGeneration):
    """SO-7B model using the spatial-aware Thinker."""

    config_class = Qwen2_5OmniConfig

    def __init__(self, config):
        super().__init__(config)
        self.thinker = Qwen2_5OmniSpatialThinkerForConditionalGeneration(config.thinker_config)

    def sync_spatial_tokenizer(self, tokenizer, spatial_token: str = "<|spatial|>") -> int:
        """Synchronize tokenizer and thinker embeddings for the spatial token."""

        spatial_token_id = int(self.thinker.sync_spatial_tokenizer(tokenizer, spatial_token=spatial_token))
        self.config.thinker_config.spatial_token_index = spatial_token_id
        self.config.thinker_config.text_config.vocab_size = self.thinker.config.text_config.vocab_size
        return spatial_token_id

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        return_audio: Optional[bool] = None,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        seld_features: Optional[torch.Tensor] = None,
        seld_feature_attention_mask: Optional[torch.Tensor] = None,
        seld_feature_lengths: Optional[torch.LongTensor] = None,
        seld_hidden_states: Optional[torch.Tensor] = None,
        seld_hidden_attention_mask: Optional[torch.Tensor] = None,
        seld_hidden_lengths: Optional[torch.LongTensor] = None,
        spatial_tokens: Optional[torch.Tensor] = None,
        projected_spatial_tokens: Optional[torch.Tensor] = None,
        spatial_token_lengths: Optional[torch.LongTensor] = None,
        has_spatial: Optional[torch.BoolTensor] = None,
        mono_audio: Optional[torch.Tensor] = None,
        mono_audio_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Route spatial kwargs to the thinker while preserving the base API.

        Notes:
            The model supports spatial generation only when
            `return_audio=False`. The talker integration path will be added in
            the next implementation stage.
        """

        spatial_kwargs = {
            "spatial_audio": spatial_audio,
            "spatial_audio_attention_mask": spatial_audio_attention_mask,
            "spatial_audio_lengths": spatial_audio_lengths,
            "seld_features": seld_features,
            "seld_feature_attention_mask": seld_feature_attention_mask,
            "seld_feature_lengths": seld_feature_lengths,
            "seld_hidden_states": seld_hidden_states,
            "seld_hidden_attention_mask": seld_hidden_attention_mask,
            "seld_hidden_lengths": seld_hidden_lengths,
            "spatial_tokens": spatial_tokens,
            "projected_spatial_tokens": projected_spatial_tokens,
            "spatial_token_lengths": spatial_token_lengths,
            "has_spatial": has_spatial,
            "mono_audio": mono_audio,
            "mono_audio_lengths": mono_audio_lengths,
        }
        has_spatial_inputs = any(value is not None for value in spatial_kwargs.values())
        if not has_spatial_inputs:
            return super().generate(
                input_ids=input_ids,
                return_audio=return_audio,
                **kwargs,
            )
        if return_audio:
            raise NotImplementedError(
                "Spatial input supports text generation only. "
                "Talker/audio-output integration is deferred."
            )
        thinker_kwargs: Dict[str, Any] = dict(kwargs)
        thinker_kwargs.update({k: v for k, v in spatial_kwargs.items() if v is not None})
        return self.thinker.generate(input_ids=input_ids, **thinker_kwargs)


__all__ = [
    "Qwen2_5OmniSpatialThinkerForConditionalGeneration",
    "Qwen2_5OmniSpatialForConditionalGeneration",
]
