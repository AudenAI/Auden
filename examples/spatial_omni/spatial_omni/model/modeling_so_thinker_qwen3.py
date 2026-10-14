# Adapted from Spatial-Omni for self-contained inference in Auden.
"""SO-30B Thinker with parallel native-audio and SO-Encoder inputs.

Spatial tokens use dedicated placeholders and position IDs. Generation injects
the spatial payload during prefill; cached decoding reuses those embeddings.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerTextTopKRouter,
)
from transformers.utils.generic import OutputRecorder, _CAN_RECORD_REGISTRY  # noqa: E402

from .configuration_qwen3_omni import Qwen3OmniMoeSpatialThinkerConfig  # noqa: E402
from ..modules.so_encoder import SOEncoder  # noqa: E402
from ..modules.so_token_projector import build_so_token_projector  # noqa: E402


class Qwen3OmniMoeSpatialTopKRouter(Qwen3OmniMoeThinkerTextTopKRouter):
    """Qwen3 router that exposes pre-softmax logits for the auxiliary loss."""

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        router_probabilities = torch.nn.functional.softmax(
            router_logits,
            dtype=torch.float,
            dim=-1,
        )
        router_top_value, router_indices = torch.topk(
            router_probabilities,
            self.top_k,
            dim=-1,
        )
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1,
                keepdim=True,
            )
        return router_logits, router_top_value, router_indices


class Qwen3OmniMoeSpatialTextModel(Qwen3OmniMoeThinkerTextModel):
    """State-compatible text-model subtype with a correct router recorder."""

    _can_record_outputs = {
        **Qwen3OmniMoeThinkerTextModel._can_record_outputs,
        "router_logits": OutputRecorder(
            Qwen3OmniMoeSpatialTopKRouter,
            layer_name="mlp.gate",
            index=0,
        ),
    }


class Qwen3OmniMoeSpatialThinkerForConditionalGeneration(
    Qwen3OmniMoeThinkerForConditionalGeneration
):
    """Spatial-BEATs–augmented Qwen3-Omni-MoE Thinker."""

    config_class = Qwen3OmniMoeSpatialThinkerConfig
    input_modalities = ("image", "video", "audio", "spatial", "text")

    def __init__(self, config):
        super().__init__(config)
        # Transformers 5.0 records index 1 from SparseMoeBlock even though that
        # block returns only hidden states. Its native router also returns
        # already-softmaxed probabilities as "logits", which the auxiliary
        # loss softmaxes a second time. Preserve the routing behavior while
        # exposing raw logits as the recorder output.
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            native_gate = getattr(mlp, "gate", None)
            if not isinstance(native_gate, Qwen3OmniMoeThinkerTextTopKRouter):
                continue
            spatial_gate = Qwen3OmniMoeSpatialTopKRouter(config.text_config)
            spatial_gate.weight = native_gate.weight
            mlp.gate = spatial_gate
        self.model.__class__ = Qwen3OmniMoeSpatialTextModel
        self.model._can_record_outputs = Qwen3OmniMoeSpatialTextModel._can_record_outputs
        _CAN_RECORD_REGISTRY[str(self.model.__class__)] = self.model._can_record_outputs
        self._validate_spatial_config(config)

        # We only support the BEATs path on Qwen3 for now. Init non-BEATs
        # branches to None so state_dict / attribute checks behave.
        self.so_encoder = None
        self.so_projector = None

        encoder_type = getattr(config, "spatial_encoder_type", "so_backbone")
        if encoder_type != "so_backbone":
            raise NotImplementedError(
                f"Qwen3 spatial path only supports spatial_encoder_type='so_backbone', "
                f"got '{encoder_type}'."
            )

        shuffle_factor = int(getattr(config, "so_projector_shuffle_factor", 4))
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
            projector_type=getattr(config, "so_projector_type", "pixel_shuffle"),
            input_dim=config.so_encoder_dim,
            hidden_dim=config.so_projector_hidden_dim,
            output_dim=config.text_config.hidden_size,  # 2048 for Qwen3-30B-A3B
            shuffle_factor=shuffle_factor,
        )

        # ----- Optional mono-replay support (gated by enable_spatial_replay) -----
        # Mirrors the 7B implementation: when enabled, allocate a learned
        # `spatial_null` token bank that fills the <|spatial|> placeholders
        # for mono-replay samples, and a small MSE alignment loss between the
        # W-only encoder output and `spatial_null.detach()` keeps the encoder
        # mono-equivariant. Default OFF so existing training is bit-identical.
        self.enable_spatial_replay = bool(getattr(config, "enable_spatial_replay", False))
        if self.enable_spatial_replay:
            max_secs = float(getattr(config, "so_backbone_max_audio_seconds", 20.0))
            null_tokens = getattr(config, "spatial_null_num_tokens", None)
            if null_tokens is None:
                null_tokens = max(1, round(max_secs * llm_rate))
            self.spatial_null = nn.Parameter(
                torch.randn(int(null_tokens), config.text_config.hidden_size) * 0.02
            )
            self.spatial_null_alignment_weight = float(
                getattr(config, "spatial_null_alignment_weight", 0.05)
            )
        else:
            self.spatial_null = None
            self.spatial_null_alignment_weight = 0.0
        self._last_spatial_replay_stats: Dict[str, float] = {}

        # Re-run post_init so newly added modules get their initialization
        # (parent __init__ already called post_init once before our submodules
        # were added; calling again is safe and only initializes new params).
        self.post_init()

    # ------------------------------------------------------------------
    def _validate_spatial_config(self, config) -> None:
        encoder_type = getattr(config, "spatial_encoder_type", "so_backbone")
        if encoder_type == "so_backbone" and not config.so_backbone_checkpoint_path:
            raise ValueError(
                "so_backbone_checkpoint_path is required when spatial_encoder_type='so_backbone'"
            )

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

    # ------------------------------------------------------------------
    # Tokenizer / embedding sync (called by the spatial processor)
    # ------------------------------------------------------------------
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ------------------------------------------------------------------
    # Override get_audio_features to cast input_features to the audio_tower
    # dtype. The processor emits float32 mel; the audio_tower weights are
    # bf16 when from_pretrained loaded the model with torch_dtype=bf16.
    # Without this cast, F.conv2d crashes with
    # "Input type (float) and bias type (BFloat16) should be the same".
    # ------------------------------------------------------------------
    def get_audio_features(
        self,
        input_features,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        **kwargs,
    ):
        try:
            tower_dtype = next(self.audio_tower.parameters()).dtype
        except StopIteration:
            tower_dtype = input_features.dtype
        if input_features.dtype != tower_dtype:
            input_features = input_features.to(dtype=tower_dtype)
        return super().get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            **kwargs,
        )

    def sync_spatial_tokenizer(
        self,
        tokenizer,
        spatial_token: str = "<|spatial|>",
        spatial_start_token: str = "<|spatial_start|>",
        spatial_end_token: str = "<|spatial_end|>",
    ) -> int:
        vocab = tokenizer.get_vocab()
        spatial_special_tokens = (
            spatial_start_token,
            spatial_token,
            spatial_end_token,
        )
        new_tokens = [token for token in spatial_special_tokens if token not in vocab]
        if new_tokens:
            special_tokens = {"additional_special_tokens": new_tokens}
            try:
                tokenizer.add_special_tokens(
                    special_tokens,
                    replace_extra_special_tokens=False,
                )
            except TypeError:  # Transformers < 5
                tokenizer.add_special_tokens(
                    special_tokens,
                    replace_additional_special_tokens=False,
                )
        spatial_token_id = int(tokenizer.convert_tokens_to_ids(spatial_token))
        spatial_start_token_id = int(tokenizer.convert_tokens_to_ids(spatial_start_token))
        spatial_end_token_id = int(tokenizer.convert_tokens_to_ids(spatial_end_token))
        initialize_boundaries = {
            token
            for token, configured_id in (
                (spatial_start_token, getattr(self.config, "spatial_start_token_id", None)),
                (spatial_end_token, getattr(self.config, "spatial_end_token_id", None)),
            )
            if configured_id is None
        }
        current_vocab_size = int(self.get_input_embeddings().num_embeddings)
        required_vocab_size = max(
            len(tokenizer),
            spatial_token_id + 1,
            spatial_start_token_id + 1,
            spatial_end_token_id + 1,
        )
        # Qwen3 reserves more embedding rows than the tokenizer currently uses.
        # Never shrink those pretrained rows when adding spatial special tokens.
        if current_vocab_size < required_vocab_size:
            self.resize_token_embeddings(required_vocab_size)
            current_vocab_size = required_vocab_size
        self.config.spatial_token_index = spatial_token_id
        self.config.spatial_start_token_id = spatial_start_token_id
        self.config.spatial_end_token_id = spatial_end_token_id
        self.config.text_config.vocab_size = current_vocab_size
        self.vocab_size = current_vocab_size
        for token_kind in ("eos", "pad"):
            token_id = getattr(tokenizer, f"{token_kind}_token_id", None)
            if token_id is None:
                continue
            setattr(self.config, f"{token_kind}_token_id", int(token_id))
            setattr(self.config.text_config, f"{token_kind}_token_id", int(token_id))
            if getattr(self, "generation_config", None) is not None:
                setattr(self.generation_config, f"{token_kind}_token_id", int(token_id))

        # Boundary rows are new and otherwise random. Initialize them from the
        # pretrained audio boundaries, which have the same structural role.
        self._initialize_new_spatial_boundary_embeddings(
            new_tokens=set(new_tokens) | initialize_boundaries,
            tokenizer=tokenizer,
            spatial_start_token=spatial_start_token,
            spatial_end_token=spatial_end_token,
        )
        return spatial_token_id

    def generate(self, *args, **kwargs):
        """Accept Qwen2.5 trainer kwargs while using Thinker-only generation."""

        kwargs.pop("return_audio", None)
        kwargs.pop("speaker", None)
        # Router load balancing is a training objective. A checkpoint trained
        # with routers keeps output_router_logits=True in its config; allowing
        # that default during generation would collect all layer router
        # probabilities and recompute the auxiliary loss at every decode step.
        kwargs["output_router_logits"] = False
        return super().generate(*args, **kwargs)

    def _initialize_new_spatial_boundary_embeddings(
        self,
        new_tokens: set[str],
        tokenizer,
        spatial_start_token: str,
        spatial_end_token: str,
    ) -> None:
        token_pairs = (
            (spatial_start_token, getattr(tokenizer, "audio_bos_token", None)),
            (spatial_end_token, getattr(tokenizer, "audio_eos_token", None)),
        )
        input_embeddings = self.get_input_embeddings().weight
        output_layer = self.get_output_embeddings()
        output_embeddings = getattr(output_layer, "weight", None)
        with torch.no_grad():
            for spatial_boundary, audio_boundary in token_pairs:
                if spatial_boundary not in new_tokens or not audio_boundary:
                    continue
                source_id = tokenizer.convert_tokens_to_ids(audio_boundary)
                target_id = tokenizer.convert_tokens_to_ids(spatial_boundary)
                if source_id is None or target_id is None:
                    continue
                input_embeddings[int(target_id)].copy_(input_embeddings[int(source_id)])
                if output_embeddings is not None:
                    output_embeddings[int(target_id)].copy_(output_embeddings[int(source_id)])

    # ------------------------------------------------------------------
    # Unified multimodal placeholder masks
    # ------------------------------------------------------------------
    def get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
        spatial_features=None,
        return_spatial_mask: bool = False,
    ):
        """Return native Qwen3 masks plus the spatial placeholder mask.

        Upstream callers still receive the original three-tuple. Spatial-aware
        callers request the fourth mask explicitly, preserving compatibility
        with the native audio/image/video forward implementation.
        """

        native_masks = super().get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )
        spatial_token_id = getattr(self.config, "spatial_token_index", None)
        if spatial_token_id is None:
            spatial_token_mask = torch.zeros(
                inputs_embeds.shape[:2],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        elif input_ids is not None:
            spatial_token_mask = input_ids == int(spatial_token_id)
        else:
            spatial_embedding = self.get_input_embeddings()(
                torch.tensor(
                    int(spatial_token_id),
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            spatial_token_mask = (inputs_embeds == spatial_embedding).all(-1)

        spatial_mask = spatial_token_mask.unsqueeze(-1).expand_as(inputs_embeds)
        spatial_mask = spatial_mask.to(inputs_embeds.device)
        if spatial_features is not None:
            if spatial_features.ndim != 2 or spatial_features.shape[-1] != inputs_embeds.shape[-1]:
                raise ValueError(
                    "spatial_features must have shape [num_spatial_tokens, hidden_size], "
                    f"got {tuple(spatial_features.shape)}"
                )
            actual = int(spatial_mask[..., 0].sum().item())
            expected = int(spatial_features.shape[0])
            if actual != expected:
                raise ValueError(
                    "Spatial features and spatial placeholders do not match: "
                    f"tokens={actual}, features={expected}."
                )
        if return_spatial_mask:
            return (*native_masks, spatial_mask)
        return native_masks

    # ------------------------------------------------------------------
    # Qwen3 multimodal RoPE extension for audio + spatial prompts
    # ------------------------------------------------------------------
    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        use_audio_in_video: bool = False,
        audio_seqlens=None,
        second_per_grids=None,
    ):
        spatial_token_id = getattr(self.config, "spatial_token_index", None)
        has_spatial = bool(
            input_ids is not None
            and spatial_token_id is not None
            and (input_ids == int(spatial_token_id)).any().item()
        )
        if not has_spatial:
            return super().get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_seqlens,
                second_per_grids=second_per_grids,
            )
        if image_grid_thw is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "Spatial RoPE currently supports audio + spatial prompts only; "
                "image/video + spatial is not implemented."
            )
        if use_audio_in_video:
            raise NotImplementedError("Spatial RoPE requires use_audio_in_video=False.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        valid_attention = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        for batch_index, sample_ids in enumerate(input_ids):
            sample_mask = valid_attention[batch_index]
            valid_tokens = sample_ids[sample_mask]
            self._validate_audio_spatial_rope_layout(valid_tokens)

        # Spatial-BEATs emits a temporal sequence without height/width axes, so
        # it uses Qwen3's native audio/text 1D RoPE convention: the same
        # monotonically increasing coordinate on all three M-RoPE axes. Keep
        # this formula equivalent to the upstream no-vision branch while
        # validating the spatial segment above.
        position_ids = valid_attention.float().cumsum(-1) - 1
        position_ids.masked_fill_(~valid_attention, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        max_position_ids = position_ids.max(dim=0).values.max(dim=-1, keepdim=True).values
        rope_deltas = max_position_ids + 1 - valid_attention.sum(dim=-1, keepdim=True)
        return position_ids, rope_deltas

    def _validate_audio_spatial_rope_layout(self, valid_tokens: torch.LongTensor) -> None:
        spatial_token_id = int(self.config.spatial_token_index)
        spatial_positions = torch.where(valid_tokens == spatial_token_id)[0]
        if spatial_positions.numel() == 0:
            return
        if spatial_positions.numel() > 1 and not torch.all(
            spatial_positions[1:] == spatial_positions[:-1] + 1
        ):
            raise ValueError("Spatial placeholder tokens must form one contiguous run.")

        spatial_start_id = getattr(self.config, "spatial_start_token_id", None)
        spatial_end_id = getattr(self.config, "spatial_end_token_id", None)
        require_boundaries = bool(getattr(self.config, "require_spatial_boundaries", True))
        if require_boundaries and (spatial_start_id is None or spatial_end_id is None):
            raise ValueError(
                "Spatial tokenizer is not synchronized: spatial start/end token ids are missing."
            )
        if spatial_start_id is not None and spatial_end_id is not None:
            starts = torch.where(valid_tokens == int(spatial_start_id))[0]
            ends = torch.where(valid_tokens == int(spatial_end_id))[0]
            if starts.numel() != 1 or ends.numel() != 1:
                raise ValueError(
                    "Each spatial segment must contain exactly one spatial start and end token."
                )
            start = int(starts.item())
            end = int(ends.item())
            if start + 1 != int(spatial_positions[0]) or int(spatial_positions[-1]) + 1 != end:
                raise ValueError(
                    "Expected <|spatial_start|><|spatial|>...<|spatial_end|> with no gaps."
                )
        else:
            start = int(spatial_positions[0])

        audio_token_id = getattr(self.config, "audio_token_id", None)
        if audio_token_id is not None:
            audio_positions = torch.where(valid_tokens == int(audio_token_id))[0]
            if audio_positions.numel():
                if audio_positions.numel() > 1 and not torch.all(
                    audio_positions[1:] == audio_positions[:-1] + 1
                ):
                    raise ValueError("Audio placeholder tokens must form one contiguous run.")
                audio_start_id = getattr(self.config, "audio_start_token_id", None)
                audio_end_id = getattr(self.config, "audio_end_token_id", None)
                if audio_start_id is not None and audio_end_id is not None:
                    audio_starts = torch.where(valid_tokens == int(audio_start_id))[0]
                    audio_ends = torch.where(valid_tokens == int(audio_end_id))[0]
                    if audio_starts.numel() != 1 or audio_ends.numel() != 1:
                        raise ValueError(
                            "Each audio segment must contain exactly one audio start and end token."
                        )
                    if (
                        int(audio_starts.item()) + 1 != int(audio_positions[0])
                        or int(audio_positions[-1]) + 1 != int(audio_ends.item())
                    ):
                        raise ValueError(
                            "Expected <|audio_start|><|audio_pad|>...<|audio_end|> with no gaps."
                        )
                if int(audio_positions[-1]) >= start:
                    raise ValueError("Audio placeholders must precede the spatial segment.")

    # ------------------------------------------------------------------
    # forward — inject spatial tokens, then delegate to parent
    # ------------------------------------------------------------------
    def forward(
        self,
        *args,
        spatial_audio: Optional[torch.Tensor] = None,
        spatial_audio_attention_mask: Optional[torch.Tensor] = None,
        spatial_audio_lengths: Optional[torch.LongTensor] = None,
        spatial_tokens: Optional[torch.Tensor] = None,
        projected_spatial_tokens: Optional[torch.Tensor] = None,
        spatial_token_lengths: Optional[torch.LongTensor] = None,
        has_spatial: Optional[torch.BoolTensor] = None,
        mono_audio: Optional[torch.Tensor] = None,
        mono_audio_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._validate_spatial_payload_contract(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            spatial_tokens=spatial_tokens,
            projected_spatial_tokens=projected_spatial_tokens,
            spatial_token_lengths=spatial_token_lengths,
        )
        has_spatial_inputs = self._has_spatial_inputs(
            spatial_audio=spatial_audio,
            spatial_tokens=spatial_tokens,
            projected_spatial_tokens=projected_spatial_tokens,
        ) or has_spatial is not None
        has_spatial_placeholders = bool(
            input_ids is not None
            and getattr(self.config, "spatial_token_index", None) is not None
            and (input_ids == int(self.config.spatial_token_index)).any().item()
        )
        if not has_spatial_inputs:
            if has_spatial_placeholders:
                raise ValueError(
                    "Spatial placeholders are present but no spatial payload was provided. "
                    "Pass spatial_audio, spatial_tokens, or projected_spatial_tokens."
                )
            return super().forward(*args, **kwargs)

        if kwargs.get("use_audio_in_video"):
            raise NotImplementedError("Spatial-Omni 30b path requires use_audio_in_video=False.")
        if kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None:
            raise NotImplementedError(
                "Qwen3 spatial forward currently supports audio + spatial inputs only."
            )
        if input_ids is None:
            raise ValueError("input_ids are required when injecting spatial tokens.")

        loss_null = None
        replay_stats: Dict[str, float] = {}
        if has_spatial is not None:
            # ----- Mixed spatial+mono replay (mono replay path) -----------
            # Resolves projected spatial embeddings for a mixed batch where
            # `has_spatial[i]==False` indicates a mono-replay sample whose
            # <|spatial|> placeholders are filled by `spatial_null`, while
            # `has_spatial[i]==True` runs the normal SOBackbone encoder path.
            if self.spatial_null is None:
                raise ValueError(
                    "has_spatial replay path requires config.enable_spatial_replay=True "
                    "(so the spatial_null parameter exists)."
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
        else:
            projected_spatial, spatial_token_lengths = self.get_spatial_features(
                input_ids=input_ids,
                spatial_audio=spatial_audio,
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
                spatial_tokens=spatial_tokens,
                projected_spatial_tokens=projected_spatial_tokens,
                spatial_token_lengths=spatial_token_lengths,
            )
        flat_spatial = self._flatten_projected_spatial(projected_spatial, spatial_token_lengths)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        _, _, _, spatial_mask = self.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            spatial_features=flat_spatial,
            return_spatial_mask=True,
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            spatial_mask,
            flat_spatial.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        kwargs["inputs_embeds"] = inputs_embeds
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

    # ------------------------------------------------------------------
    # generate() bridge — forward spatial inputs through to forward()
    # ------------------------------------------------------------------
    # Preserve spatial inputs during prefill. Once generation uses the KV
    # cache, omit them so the spatial encoder runs once per prompt.
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
        spatial_tokens=None,
        spatial_token_lengths=None,
        projected_spatial_tokens=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
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
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Splice spatial inputs back into model_inputs for forward().
        model_inputs["spatial_audio"] = spatial_audio
        model_inputs["spatial_audio_attention_mask"] = spatial_audio_attention_mask
        model_inputs["spatial_audio_lengths"] = spatial_audio_lengths
        model_inputs["spatial_tokens"] = spatial_tokens
        model_inputs["spatial_token_lengths"] = spatial_token_lengths
        model_inputs["projected_spatial_tokens"] = projected_spatial_tokens

        # Qwen3 prefill starts at cache position zero. Keep spatial inputs only
        # for that iteration so the encoder is not recomputed during decode.
        prepared_input_ids = model_inputs.get("input_ids")
        prepared_inputs_embeds = model_inputs.get("inputs_embeds")
        prepared_seq_len = None
        if (
            prepared_input_ids is not None
            and hasattr(prepared_input_ids, "shape")
            and prepared_input_ids.ndim >= 2
        ):
            prepared_seq_len = int(prepared_input_ids.shape[1])

        cache_starts_after_zero = bool(
            cache_position is not None
            and cache_position.numel() > 0
            and int(cache_position.reshape(-1)[0].item()) > 0
        )
        # `is_first_iteration` defaults to False in upstream Qwen3, so its
        # false value alone cannot identify decode. Cache position is the
        # authoritative signal; the prepared one-token fallback covers older
        # generation loops that do not provide cache_position.
        is_decode_by_cache = bool(use_cache) and cache_starts_after_zero
        is_decode_by_seqlen = bool(
            past_key_values is not None
            and prepared_seq_len is not None
            and prepared_seq_len <= 1
            and prepared_inputs_embeds is None
        )

        if is_decode_by_cache or is_decode_by_seqlen:
            for key in (
                "spatial_audio",
                "spatial_audio_attention_mask",
                "spatial_audio_lengths",
                "spatial_tokens",
                "spatial_token_lengths",
                "projected_spatial_tokens",
                "has_spatial",
                "mono_audio",
                "mono_audio_lengths",
            ):
                model_inputs[key] = None
        return model_inputs

    # ------------------------------------------------------------------
    # helpers (mostly identical to Qwen2.5 spatial impl, BEATs-only)
    # ------------------------------------------------------------------
    @staticmethod
    def _has_spatial_inputs(
        spatial_audio,
        spatial_tokens,
        projected_spatial_tokens=None,
    ) -> bool:
        return any(
            value is not None
            for value in (spatial_audio, spatial_tokens, projected_spatial_tokens)
        )

    @staticmethod
    def _validate_spatial_payload_contract(
        spatial_audio,
        spatial_audio_attention_mask,
        spatial_audio_lengths,
        spatial_tokens,
        projected_spatial_tokens,
        spatial_token_lengths,
    ) -> None:
        payloads = {
            "spatial_audio": spatial_audio,
            "spatial_tokens": spatial_tokens,
            "projected_spatial_tokens": projected_spatial_tokens,
        }
        present = [name for name, value in payloads.items() if value is not None]
        if len(present) > 1:
            raise ValueError(
                "Spatial payloads are mutually exclusive; provide exactly one of "
                "spatial_audio, spatial_tokens, or projected_spatial_tokens. "
                f"Received: {present}."
            )

        audio_sidecars = {
            "spatial_audio_attention_mask": spatial_audio_attention_mask,
            "spatial_audio_lengths": spatial_audio_lengths,
        }
        invalid_audio_sidecars = [
            name
            for name, value in audio_sidecars.items()
            if value is not None and spatial_audio is None
        ]
        if invalid_audio_sidecars:
            raise ValueError(
                "Spatial audio masks and lengths require spatial_audio; received "
                f"{invalid_audio_sidecars} without spatial_audio."
            )
        if not present and spatial_token_lengths is not None:
            raise ValueError(
                "spatial_token_lengths requires spatial_audio, spatial_tokens, "
                "or projected_spatial_tokens."
            )

    def get_spatial_features(
        self,
        input_ids,
        spatial_audio=None,
        spatial_audio_attention_mask=None,
        spatial_audio_lengths=None,
        spatial_tokens=None,
        projected_spatial_tokens=None,
        spatial_token_lengths=None,
    ):
        """Encode/project spatial inputs into Qwen3 Thinker hidden states."""

        self._validate_spatial_payload_contract(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            spatial_tokens=spatial_tokens,
            projected_spatial_tokens=projected_spatial_tokens,
            spatial_token_lengths=spatial_token_lengths,
        )
        if projected_spatial_tokens is not None:
            if projected_spatial_tokens.ndim != 3:
                raise ValueError(
                    "projected_spatial_tokens must have shape [B, T_spat, hidden_size], "
                    f"got {tuple(projected_spatial_tokens.shape)}"
                )
            if projected_spatial_tokens.shape[-1] != self.config.text_config.hidden_size:
                raise ValueError(
                    "projected_spatial_tokens hidden size mismatch: "
                    f"{projected_spatial_tokens.shape[-1]} vs {self.config.text_config.hidden_size}"
                )
            projected_spatial = projected_spatial_tokens
            if spatial_token_lengths is None:
                spatial_token_lengths = projected_spatial.new_full(
                    (projected_spatial.shape[0],),
                    projected_spatial.shape[1],
                    dtype=torch.long,
                )
        else:
            spatial_tokens, spatial_token_lengths = self._resolve_spatial_tokens(
                spatial_audio=spatial_audio,
                spatial_audio_attention_mask=spatial_audio_attention_mask,
                spatial_audio_lengths=spatial_audio_lengths,
                spatial_tokens=spatial_tokens,
                spatial_token_lengths=spatial_token_lengths,
            )
            projected_spatial = self.so_projector(spatial_tokens)
            shuffle_factor = int(getattr(self.so_projector, "shuffle_factor", 1))
            if shuffle_factor > 1:
                new_lengths = torch.clamp(
                    torch.div(spatial_token_lengths, shuffle_factor, rounding_mode="floor"),
                    min=0,
                    max=int(projected_spatial.shape[1]),
                )
                new_lengths = torch.where(
                    (spatial_token_lengths > 0) & (new_lengths == 0),
                    torch.ones_like(new_lengths),
                    new_lengths,
                )
                spatial_token_lengths = new_lengths

        return self._align_projected_spatial_to_placeholders(
            projected_spatial=projected_spatial,
            spatial_token_lengths=spatial_token_lengths,
            input_ids=input_ids,
        )

    def _resolve_spatial_tokens(
        self,
        spatial_audio,
        spatial_audio_attention_mask,
        spatial_audio_lengths,
        spatial_tokens,
        spatial_token_lengths,
    ):
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

        if spatial_audio is None:
            raise ValueError(
                "spatial_audio is required for the so_backbone encoder path "
                "when spatial_tokens is not provided directly."
            )
        # When the model is loaded with device_map="auto" the parent thinker
        # has accelerate hooks that route activations across GPUs. The
        # so_encoder lives on a single device though, so we must
        # ensure inputs are on the encoder's device before calling forward.
        # Without this, training-time inputs may arrive on `meta` (a relic of
        # low_cpu_mem_usage init) and torchaudio's kaldi.fbank crashes when it
        # asks for an epsilon tensor on the meta device.
        try:
            enc_device = next(self.so_encoder.parameters()).device
        except StopIteration:
            enc_device = spatial_audio.device
        if spatial_audio.device != enc_device:
            spatial_audio = spatial_audio.to(enc_device)
            if spatial_audio_attention_mask is not None:
                spatial_audio_attention_mask = spatial_audio_attention_mask.to(enc_device)
            if spatial_audio_lengths is not None:
                spatial_audio_lengths = spatial_audio_lengths.to(enc_device)
        beats_output = self.so_encoder(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
        )
        return beats_output.spatial_tokens, beats_output.spatial_token_lengths

    def _build_spatial_mask(self, input_ids, inputs_embeds):
        if getattr(self.config, "spatial_token_index", None) is None:
            raise ValueError("config.spatial_token_index must be set before using the spatial thinker.")
        return (
            (input_ids == self.config.spatial_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )

    @staticmethod
    def _flatten_projected_spatial(projected_spatial, spatial_token_lengths):
        if projected_spatial.ndim != 3:
            raise ValueError(
                f"projected_spatial must have shape [B, T_spat, D_llm], got {tuple(projected_spatial.shape)}"
            )
        if spatial_token_lengths.ndim != 1 or spatial_token_lengths.shape[0] != projected_spatial.shape[0]:
            raise ValueError(
                f"spatial_token_lengths must have shape [B], got {tuple(spatial_token_lengths.shape)}"
            )
        valid_rows = []
        max_tokens = projected_spatial.shape[1]
        for index, length in enumerate(spatial_token_lengths.tolist()):
            if length < 0 or length > max_tokens:
                raise ValueError(
                    f"spatial_token_lengths[{index}]={length} outside [0, {max_tokens}]"
                )
            if length == 0:
                continue
            valid_rows.append(projected_spatial[index, :length])
        if not valid_rows:
            return projected_spatial.new_zeros((0, projected_spatial.shape[-1]))
        return torch.cat(valid_rows, dim=0)

    def _align_projected_spatial_to_placeholders(
        self, projected_spatial, spatial_token_lengths, input_ids
    ):
        if getattr(self.config, "spatial_token_index", None) is None:
            raise ValueError("config.spatial_token_index must be set before using the spatial thinker.")
        placeholder_counts = (input_ids == self.config.spatial_token_index).sum(dim=1).to(
            device=spatial_token_lengths.device, dtype=torch.long
        )
        if torch.equal(placeholder_counts, spatial_token_lengths):
            return projected_spatial, spatial_token_lengths
        if bool(getattr(self.config, "strict_spatial_placeholder_count", True)):
            raise ValueError(
                "Spatial placeholder counts must exactly match projected token lengths: "
                f"placeholders={placeholder_counts.tolist()}, "
                f"projected={spatial_token_lengths.tolist()}."
            )

        batch_size, _, hidden_dim = projected_spatial.shape
        target_max = int(placeholder_counts.max().item()) if batch_size > 0 else 0
        aligned = projected_spatial.new_zeros((batch_size, target_max, hidden_dim))
        source_max = projected_spatial.shape[1]
        for index, (src_len, tgt_len) in enumerate(
            zip(spatial_token_lengths.tolist(), placeholder_counts.tolist())
        ):
            if src_len < 0 or src_len > source_max:
                raise ValueError(f"spatial_token_lengths[{index}]={src_len} outside [0, {source_max}]")
            if tgt_len <= 0:
                continue
            copy_len = min(src_len, tgt_len)
            if copy_len > 0:
                aligned[index, :copy_len] = projected_spatial[index, :copy_len]
            if tgt_len > src_len and src_len > 0:
                aligned[index, copy_len:tgt_len] = projected_spatial[index, src_len - 1].unsqueeze(0)
        return aligned, placeholder_counts

    def _validate_spatial_mask_count(self, spatial_mask, projected_spatial, spatial_token_lengths):
        expected = int(spatial_token_lengths.sum().item())
        actual = int(spatial_mask[..., 0].sum().item())
        if actual != expected:
            raise ValueError(
                f"Spatial placeholder count does not match projected token count: {actual} vs {expected}"
            )
        if projected_spatial.ndim != 2:
            raise ValueError(
                f"Packed projected spatial tokens must be [sum(T_i), D_llm], got {tuple(projected_spatial.shape)}"
            )

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

        spatial_tokens, lengths = self._resolve_spatial_tokens(
            spatial_audio=spatial_audio,
            spatial_audio_attention_mask=spatial_audio_attention_mask,
            spatial_audio_lengths=spatial_audio_lengths,
            spatial_tokens=None,
            spatial_token_lengths=None,
        )
        projected = self.so_projector(spatial_tokens)
        shuffle_factor = int(getattr(self.so_projector, "shuffle_factor", 1))
        if shuffle_factor > 1:
            new_lengths = torch.clamp(
                torch.div(lengths, shuffle_factor, rounding_mode="floor"),
                min=0,
                max=int(projected.shape[1]),
            )
            lengths = torch.where(
                (lengths > 0) & (new_lengths == 0),
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


class Qwen3OmniMoeSpatialForConditionalGeneration(
    Qwen3OmniMoeSpatialThinkerForConditionalGeneration
):
    """Top-level wrapper that mimics the Qwen2.5 ``model.thinker`` shape.

    For Qwen3 we wrap the thinker only (no talker). The Qwen2.5 train script
    accesses spatial submodules via ``model.thinker.so_encoder`` and
    calls ``model.disable_talker()``. This wrapper makes both work without
    changing the underlying behavior:

      - ``self.thinker`` returns ``self`` (so ``model.thinker.X`` == ``model.X``)
      - ``disable_talker()`` is a no-op (the Qwen3 talker is never built here)

    Because Qwen3's top-level ``Qwen3OmniMoeForConditionalGeneration`` config
    has known bugs and we don't need the talker, we expose the thinker
    directly as the top-level model.
    """

    config_class = Qwen3OmniMoeSpatialThinkerConfig

    @property
    def thinker(self):
        return self

    def disable_talker(self):
        return None


def register_qwen3_spatial_auto_classes() -> None:
    """Register model auto classes and the Qwen3 fused-expert converter."""

    from transformers import AutoModel, AutoModelForCausalLM

    model_pairs = (
        (AutoModel, Qwen3OmniMoeSpatialForConditionalGeneration),
        (AutoModelForCausalLM, Qwen3OmniMoeSpatialForConditionalGeneration),
    )
    for auto_class, model_class in model_pairs:
        auto_class.register(
            Qwen3OmniMoeSpatialThinkerConfig,
            model_class,
            exist_ok=True,
        )
    try:
        from transformers import AutoModelForMultimodalLM

        AutoModelForMultimodalLM.register(
            Qwen3OmniMoeSpatialThinkerConfig,
            Qwen3OmniMoeSpatialForConditionalGeneration,
            exist_ok=True,
        )
    except ImportError:
        pass

    # Transformers 5 stores Qwen3 MoE experts as fused 3D parameters, while
    # the released checkpoint stores one gate/up/down tensor per expert. The
    # converter lookup is keyed by config.model_type, so the spatial subtype
    # must explicitly inherit the native thinker conversion mapping.
    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        return
    model_type = Qwen3OmniMoeSpatialThinkerConfig.model_type
    if get_checkpoint_conversion_mapping(model_type) is None:
        native_mapping = get_checkpoint_conversion_mapping("qwen3_omni_moe_thinker")
        if native_mapping is None:
            raise RuntimeError(
                "Transformers does not provide the Qwen3-Omni thinker checkpoint converter."
            )
        register_checkpoint_conversion_mapping(model_type, native_mapping)


register_qwen3_spatial_auto_classes()


__all__ = [
    "Qwen3OmniMoeSpatialThinkerForConditionalGeneration",
    "Qwen3OmniMoeSpatialForConditionalGeneration",
    "register_qwen3_spatial_auto_classes",
]
