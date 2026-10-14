# Adapted from Spatial-Omni for self-contained inference in Auden.
"""Spatial-aware processor for Qwen3-Omni-MoE Thinker (BEATs path only).

Differences vs the Qwen2.5 spatial processor:

- Subclasses upstream :class:`Qwen3OmniMoeProcessor` from Transformers.
- Drops SELD233 / IV explicit-payload branches — only Spatial-BEATs is wired.
- The Qwen3 processor signature has a `video_processor` (not collapsed into the
  image processor as Qwen2.5 did), and ``__call__`` does not accept Qwen2.5-only
  kwargs like ``return_spatial_audio``. Both differences are handled here.

The output schema is a strict subset of the Qwen2.5 one:

    spatial_audio                 [B, T_audio_max, 4]      float32 padded FOA
    spatial_audio_attention_mask  [B, T_audio_max]         bool
    spatial_audio_lengths         [B]                      int64 valid samples
    spatial_token_lengths         [B]                      int64 LLM-side count
"""

from __future__ import annotations

import os
import sys
import warnings
from fractions import Fraction
from typing import List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeProcessor,
)
from transformers.processing_utils import ProcessorMixin  # noqa: E402

from .configuration_qwen3_omni import Qwen3OmniMoeSpatialThinkerConfig  # noqa: E402
from ..utils.spatial_seld_utils import build_1d_attention_mask  # noqa: E402


_DEFAULT_TARGET_TOKEN_RATE = 2.5  # LLM-side spatial rate (Hz), matches SO-Encoder default
_DEFAULT_MAX_AUDIO_SECONDS = 20.0
_DEFAULT_SAMPLING_RATE = 16000


class Qwen3OmniMoeSpatialProcessor(Qwen3OmniMoeProcessor):
    """Qwen3-Omni-MoE processor with a `<|spatial|>` modality (BEATs only)."""

    # Inherit valid_kwargs from parent + spatial extras.
    spatial_extra_kwargs = (
        "spatial_token_lengths",
        "spatial_audio_max_seconds",
        "spatial_tokens",
        "projected_spatial_tokens",
        "allow_mono_spatial_tokens",
    )

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        spatial_encoder_type="so_backbone",
        so_backbone_target_token_rate=_DEFAULT_TARGET_TOKEN_RATE,
        so_encoder_token_rate=10.0,
        so_projector_type="pixel_shuffle",
        so_projector_shuffle_factor=4,
        spatial_audio_max_seconds=_DEFAULT_MAX_AUDIO_SECONDS,
    ):
        # The native processor requires image and video processors. This
        # audio+spatial variant deliberately exposes only the two components it
        # can use. Keeping this positional order aligned with get_attributes()
        # is required for ProcessorMixin/AutoProcessor.from_pretrained().
        ProcessorMixin.__init__(
            self,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        # Mirror the attributes the upstream __call__ reads off `self`.
        self.audio_token = getattr(self.tokenizer, "audio_token", "<|audio_pad|>")
        self.image_token = getattr(self.tokenizer, "image_token", "<|IMAGE|>")
        self.video_token = getattr(self.tokenizer, "video_token", "<|VIDEO|>")
        self.vision_bos_token = getattr(self.tokenizer, "vision_bos_token", "<|vision_bos|>")
        self.vision_eos_token = getattr(self.tokenizer, "vision_eos_token", "<|vision_eos|>")
        self.audio_bos_token = getattr(self.tokenizer, "audio_bos_token", "<|audio_bos|>")
        self.audio_eos_token = getattr(self.tokenizer, "audio_eos_token", "<|audio_eos|>")
        self.spatial_start_token = "<|spatial_start|>"
        self.spatial_token = "<|spatial|>"
        self.spatial_end_token = "<|spatial_end|>"
        self.audio_token_aliases = ("<|audio|>", "<|AUDIO|>")
        self.video_token_aliases = ("<|video|>", "<|VIDEO|>")
        # Persist the model/processor length contract so AutoProcessor
        # round-trips cannot silently return to the default 10 Hz / k=4 path.
        self.spatial_encoder_type = str(spatial_encoder_type)
        self.so_backbone_target_token_rate = float(so_backbone_target_token_rate)
        self.so_encoder_token_rate = float(so_encoder_token_rate)
        self.so_projector_type = str(so_projector_type)
        self.so_projector_shuffle_factor = int(so_projector_shuffle_factor)
        self.spatial_audio_max_seconds = float(spatial_audio_max_seconds)
        if self.so_encoder_token_rate <= 0:
            raise ValueError("so_encoder_token_rate must be > 0")
        if self.so_projector_shuffle_factor < 1:
            raise ValueError("so_projector_shuffle_factor must be >= 1")
        if self.spatial_audio_max_seconds <= 0:
            raise ValueError("spatial_audio_max_seconds must be > 0")
        (
            self.spatial_start_token_id,
            self.spatial_token_id,
            self.spatial_end_token_id,
        ) = self._register_spatial_tokens_on_tokenizer()

    # ------------------------------------------------------------------
    # tokenizer side
    # ------------------------------------------------------------------
    def _register_spatial_tokens_on_tokenizer(self) -> tuple[int, int, int]:
        vocab = self.tokenizer.get_vocab()
        tokens = (
            self.spatial_start_token,
            self.spatial_token,
            self.spatial_end_token,
        )
        missing = [token for token in tokens if token not in vocab]
        if missing:
            special_tokens = {"additional_special_tokens": missing}
            try:
                self.tokenizer.add_special_tokens(
                    special_tokens,
                    replace_extra_special_tokens=False,
                )
            except TypeError:  # Transformers < 5
                self.tokenizer.add_special_tokens(
                    special_tokens,
                    replace_additional_special_tokens=False,
                )
        return tuple(
            int(self.tokenizer.convert_tokens_to_ids(token)) for token in tokens
        )

    def sync_spatial_tokenizer_with_model(self, model) -> int:
        if not hasattr(model, "sync_spatial_tokenizer"):
            raise TypeError(
                "model must expose sync_spatial_tokenizer(tokenizer, spatial_token=...)"
            )
        token_id = int(
            model.sync_spatial_tokenizer(
                self.tokenizer,
                spatial_token=self.spatial_token,
                spatial_start_token=self.spatial_start_token,
                spatial_end_token=self.spatial_end_token,
            )
        )
        self.spatial_token_id = token_id
        self.spatial_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.spatial_start_token)
        )
        self.spatial_end_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self.spatial_end_token)
        )
        thinker_config = None
        if hasattr(model, "config") and hasattr(model.config, "thinker_config"):
            thinker_config = model.config.thinker_config
        elif hasattr(model, "thinker") and hasattr(model.thinker, "config"):
            thinker_config = model.thinker.config
        elif hasattr(model, "config"):
            thinker_config = model.config
        if thinker_config is not None:
            self.spatial_encoder_type = str(
                getattr(thinker_config, "spatial_encoder_type", self.spatial_encoder_type)
            )
            self.so_backbone_target_token_rate = float(
                getattr(
                    thinker_config,
                    "so_backbone_target_token_rate",
                    self.so_backbone_target_token_rate,
                )
            )
            self.so_encoder_token_rate = float(
                getattr(thinker_config, "so_encoder_token_rate", self.so_encoder_token_rate)
            )
            self.so_projector_type = str(
                getattr(thinker_config, "so_projector_type", self.so_projector_type)
            )
            self.so_projector_shuffle_factor = int(
                getattr(
                    thinker_config,
                    "so_projector_shuffle_factor",
                    self.so_projector_shuffle_factor,
                )
            )
            self.spatial_audio_max_seconds = float(
                getattr(
                    thinker_config,
                    "so_backbone_max_audio_seconds",
                    self.spatial_audio_max_seconds,
                )
            )
        return token_id

    # ------------------------------------------------------------------
    # length helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _round_divide_half_to_even(numerator: torch.LongTensor, denominator: int) -> torch.LongTensor:
        if denominator <= 0:
            raise ValueError(f"denominator must be > 0, got {denominator}")
        quotient = torch.div(numerator, denominator, rounding_mode="floor")
        remainder = torch.remainder(numerator, denominator)
        twice_remainder = remainder * 2
        round_up = (twice_remainder > denominator) | (
            (twice_remainder == denominator) & (torch.remainder(quotient, 2) == 1)
        )
        return quotient + round_up.to(dtype=quotient.dtype)

    def _samples_to_so_backbone_tokens(self, num_samples: torch.LongTensor) -> torch.LongTensor:
        # Mirror the model exactly: first round at the encoder's native rate,
        # then apply the pixel-shuffle floor reduction. Rounding directly at
        # the effective LLM rate differs for durations such as 1.5 seconds.
        rate = Fraction(str(self.so_encoder_token_rate)).limit_denominator(1000)
        sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", _DEFAULT_SAMPLING_RATE))
        numerator = num_samples.to(dtype=torch.long) * int(rate.numerator)
        denominator = sampling_rate * int(rate.denominator)
        native_steps = torch.clamp(
            self._round_divide_half_to_even(numerator, denominator),
            min=1,
        )
        if self.so_projector_type != "pixel_shuffle":
            return native_steps
        shuffle_factor = max(int(self.so_projector_shuffle_factor), 1)
        llm_steps = torch.div(native_steps, shuffle_factor, rounding_mode="floor")
        return torch.where(
            (native_steps > 0) & (llm_steps == 0),
            torch.ones_like(llm_steps),
            llm_steps,
        )

    def build_audio_spatial_prefix(self) -> str:
        """Return native Qwen3 audio boundaries followed by spatial boundaries."""

        return (
            f"{self.audio_bos_token}{self.audio_token}{self.audio_eos_token}"
            f"{self.spatial_start_token}{self.spatial_token}{self.spatial_end_token}"
        )

    def build_qa_text_parts(self, prompt: str, answer: str) -> tuple[str, str]:
        """Build an instruction-tuning sample using Qwen3 chat conventions."""

        prefix = (
            f"<|im_start|>user\n{self.build_audio_spatial_prefix()}\n"
            f"{str(prompt).rstrip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        answer_suffix = f"{str(answer).strip()}<|im_end|>"
        return prefix, answer_suffix

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------
    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        **kwargs,
    ) -> BatchFeature:
        if audio is None:
            raise ValueError(
                "Qwen3OmniMoeSpatialProcessor requires `audio`; mono or 4-ch FOA."
            )

        # Pull off spatial-only kwargs so they don't leak into upstream processor.
        spatial_token_lengths = kwargs.pop("spatial_token_lengths", None)
        spatial_tokens = kwargs.pop("spatial_tokens", None)
        projected_spatial_tokens = kwargs.pop("projected_spatial_tokens", None)
        allow_mono_spatial_tokens = bool(
            kwargs.pop("allow_mono_spatial_tokens", False)
        )
        if spatial_tokens is not None and projected_spatial_tokens is not None:
            raise ValueError(
                "spatial_tokens and projected_spatial_tokens are mutually exclusive."
            )
        spatial_audio_max_seconds = float(
            kwargs.pop("spatial_audio_max_seconds", self.spatial_audio_max_seconds)
        )
        if spatial_audio_max_seconds <= 0:
            raise ValueError(f"spatial_audio_max_seconds must be > 0, got {spatial_audio_max_seconds}")

        normalized_audio = self._normalize_audio_list(audio)
        normalized_audio = self._truncate_audio_list(
            normalized_audio,
            max_seconds=spatial_audio_max_seconds,
        )
        foa_mask = torch.tensor(
            [sample.shape[0] == 4 for sample in normalized_audio], dtype=torch.bool
        )
        if foa_mask.any() and (~foa_mask).any():
            raise ValueError(
                "Mixed mono/FOA batches are not supported. Each batch must be entirely mono or entirely FOA."
            )
        batch_size = len(normalized_audio)

        # Build padded FOA tensor + spatial token-length bookkeeping
        spatial_payload: dict[str, torch.Tensor] = {}
        has_explicit_spatial_payload = (
            spatial_tokens is not None or projected_spatial_tokens is not None
        )
        if (
            allow_mono_spatial_tokens
            and bool((~foa_mask).any())
            and not has_explicit_spatial_payload
            and spatial_token_lengths is None
        ):
            raise ValueError(
                "allow_mono_spatial_tokens=True requires spatial_tokens, "
                "projected_spatial_tokens, or explicit spatial_token_lengths "
                "(mixed-replay training fills placeholders with spatial_null)."
            )
        if foa_mask.any() and not has_explicit_spatial_payload:
            spatial_payload = self._build_spatial_audio_payload(
                normalized_audio,
                foa_mask=foa_mask,
                max_seconds=spatial_audio_max_seconds,
            )

        # Resolve effective spatial_token_lengths
        explicit_payload = (
            spatial_tokens
            if spatial_tokens is not None
            else projected_spatial_tokens
        )
        explicit_payload_name = (
            "spatial_tokens"
            if spatial_tokens is not None
            else "projected_spatial_tokens"
        )
        if explicit_payload is not None:
            tokens = torch.as_tensor(explicit_payload)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 3 or tokens.shape[0] != batch_size:
                raise ValueError(
                    f"{explicit_payload_name} must have shape [B, T_spat, D]; "
                    f"got {tuple(tokens.shape)}"
                )
            spatial_payload[explicit_payload_name] = tokens
            if spatial_token_lengths is None:
                effective_spatial_token_lengths = torch.full(
                    (batch_size,), tokens.shape[1], dtype=torch.long
                )
            else:
                effective_spatial_token_lengths = torch.as_tensor(
                    spatial_token_lengths, dtype=torch.long
                )
            if effective_spatial_token_lengths.ndim != 1:
                raise ValueError(
                    "spatial_token_lengths must have shape [B], got "
                    f"{tuple(effective_spatial_token_lengths.shape)}"
                )
            if bool(
                (
                    (effective_spatial_token_lengths < 0)
                    | (effective_spatial_token_lengths > tokens.shape[1])
                ).any()
            ):
                raise ValueError(
                    "spatial_token_lengths must stay within [0, T_spat] for "
                    f"{explicit_payload_name}."
                )
        elif spatial_token_lengths is not None:
            effective_spatial_token_lengths = torch.as_tensor(
                spatial_token_lengths, dtype=torch.long
            )
        else:
            effective_spatial_token_lengths = spatial_payload.get("spatial_token_lengths")

        # Pre-flight prompt validation + alias normalization + placeholder expansion
        processed_text = self._normalize_prompt_special_tokens(text)
        self._validate_text_modal_placeholders(
            processed_text,
            foa_mask=foa_mask,
            has_video=videos is not None,
            allow_mono_spatial_tokens=allow_mono_spatial_tokens,
        )
        if processed_text is not None:
            processed_text = self._expand_spatial_placeholders(
                text=processed_text,
                spatial_token_lengths=effective_spatial_token_lengths,
            )

        # Sanity-check effective_spatial_token_lengths
        if effective_spatial_token_lengths is not None:
            if int(effective_spatial_token_lengths.shape[0]) != batch_size:
                raise ValueError(
                    "spatial_token_lengths batch size must match audio batch size: "
                    f"{int(effective_spatial_token_lengths.shape[0])} vs {batch_size}"
                )
            invalid_mono = (
                (~foa_mask)
                & (effective_spatial_token_lengths > 0)
                & (not allow_mono_spatial_tokens)
            )
            if invalid_mono.any():
                raise ValueError("Mono samples must not carry spatial inputs.")
            invalid_foa = foa_mask & (effective_spatial_token_lengths <= 0)
            if invalid_foa.any():
                raise ValueError("FOA samples must provide positive spatial token lengths.")

        # Delegate audio/text processing to upstream processor.
        # We avoid super().__call__ because it dereferences self.image_processor /
        # self.video_processor for unrelated branches even when images/videos are
        # None (and our env lacks torchvision). Implement audio-only path inline.
        if images is not None or videos is not None:
            raise NotImplementedError(
                "Qwen3OmniMoeSpatialProcessor only supports the audio-only spatial path."
            )
        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
            _get_feat_extract_output_lengths,
        )
        upstream_audio = [arr[0] for arr in normalized_audio]
        audio_inputs = self.feature_extractor(
            upstream_audio,
            sampling_rate=getattr(self.feature_extractor, "sampling_rate", _DEFAULT_SAMPLING_RATE),
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors=kwargs.get("return_tensors", None),
        )
        audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        # Expand <|AUDIO|> placeholders into per-frame tokens.
        feat_attn = audio_inputs["feature_attention_mask"]
        if not isinstance(feat_attn, torch.Tensor):
            feat_attn = torch.as_tensor(feat_attn)
        audio_lengths = _get_feat_extract_output_lengths(feat_attn.sum(-1))
        audio_lengths_list = [int(x) for x in audio_lengths.tolist()]

        text_list = (
            [processed_text] if not isinstance(processed_text, list) else list(processed_text)
        )
        audio_iter = iter(audio_lengths_list)
        AUDIO_PH = "<|audio_placeholder|>"
        expanded_text: List[str] = []
        for sample in text_list:
            count = sample.count(self.audio_token)
            for _ in range(count):
                sample = sample.replace(self.audio_token, AUDIO_PH * next(audio_iter), 1)
            sample = sample.replace(AUDIO_PH, self.audio_token)
            expanded_text.append(sample)
        if not isinstance(processed_text, list):
            tokenizer_text = expanded_text[0]
        else:
            tokenizer_text = expanded_text

        text_kwargs = {
            "padding": kwargs.get("padding", False),
            "padding_side": kwargs.get("padding_side", "left"),
            "return_tensors": kwargs.get("return_tensors", None),
        }
        texts_inputs = self.tokenizer(tokenizer_text, **text_kwargs)

        batch = BatchFeature(
            data={**texts_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

        # Splice in spatial fields
        for key, value in spatial_payload.items():
            batch[key] = value
        if effective_spatial_token_lengths is not None:
            batch["spatial_token_lengths"] = effective_spatial_token_lengths
        return batch

    # ------------------------------------------------------------------
    # helpers (audio normalization + payload construction)
    # ------------------------------------------------------------------
    def _normalize_audio_list(self, audio) -> List[np.ndarray]:
        if not isinstance(audio, (list, tuple)):
            audio = [audio]
        normalized: List[np.ndarray] = []
        for index, item in enumerate(audio):
            array = np.asarray(item, dtype=np.float32)
            if array.ndim == 1:
                normalized.append(array[None, :].astype(np.float32, copy=False))
                continue
            if array.ndim != 2:
                raise ValueError(f"Audio item {index} must be 2D, got shape {array.shape}")
            if array.shape[0] in (1, 4):
                channels_first = array
            elif array.shape[1] in (1, 4):
                channels_first = array.T
            else:
                raise ValueError(
                    f"Audio item {index} must be mono or 4-channel FOA, got {array.shape}"
                )
            normalized.append(channels_first.astype(np.float32, copy=False))
        return normalized

    def _truncate_audio_list(
        self, audio_list: List[np.ndarray], max_seconds: float
    ) -> List[np.ndarray]:
        sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", _DEFAULT_SAMPLING_RATE))
        max_samples = int(round(max_seconds * sampling_rate))
        clipped: List[np.ndarray] = []
        for index, array in enumerate(audio_list):
            if array.shape[1] > max_samples:
                warnings.warn(
                    f"Audio sample {index} > {max_seconds:.1f}s, truncating "
                    f"from {array.shape[1] / sampling_rate:.2f}s",
                    stacklevel=2,
                )
                array = array[:, :max_samples]
            clipped.append(array)
        return clipped

    def _build_spatial_audio_payload(
        self,
        audio_list: List[np.ndarray],
        foa_mask: torch.BoolTensor,
        max_seconds: float,
    ) -> dict:
        sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", _DEFAULT_SAMPLING_RATE))
        max_samples = int(round(max_seconds * sampling_rate))
        batch_size = len(audio_list)
        spatial_audio = np.zeros((batch_size, max_samples, 4), dtype=np.float32)
        spatial_lengths = np.zeros((batch_size,), dtype=np.int64)
        for idx, array in enumerate(audio_list):
            if not bool(foa_mask[idx].item()):
                continue
            valid = min(array.shape[1], max_samples)
            spatial_audio[idx, :valid, :] = array[:, :valid].T
            spatial_lengths[idx] = valid

        spatial_audio_lengths = torch.from_numpy(spatial_lengths)
        spatial_audio_attention_mask = build_1d_attention_mask(
            spatial_audio_lengths, max_length=max_samples
        )
        spatial_token_lengths = torch.zeros((batch_size,), dtype=torch.long)
        if bool(foa_mask.any().item()):
            foa_lengths = spatial_audio_lengths[foa_mask]
            foa_token_lengths = self._samples_to_so_backbone_tokens(foa_lengths)
            spatial_token_lengths[foa_mask] = foa_token_lengths

        return {
            "spatial_audio": torch.from_numpy(spatial_audio),
            "spatial_audio_attention_mask": spatial_audio_attention_mask,
            "spatial_audio_lengths": spatial_audio_lengths,
            "spatial_token_lengths": spatial_token_lengths,
        }

    # ------------------------------------------------------------------
    # text normalization + validation
    # ------------------------------------------------------------------
    def _normalize_prompt_special_tokens(self, text):
        if text is None:
            return None
        single_input = not isinstance(text, list)
        text_list = [text] if single_input else text
        normalized = []
        for sample in text_list:
            if isinstance(sample, str):
                for alias in self.audio_token_aliases:
                    sample = sample.replace(alias, self.audio_token)
                for alias in self.video_token_aliases:
                    sample = sample.replace(alias, self.video_token)
                sample = self._ensure_modality_boundaries(
                    sample,
                    placeholder=self.audio_token,
                    start_token=self.audio_bos_token,
                    end_token=self.audio_eos_token,
                )
                sample = self._ensure_modality_boundaries(
                    sample,
                    placeholder=self.spatial_token,
                    start_token=self.spatial_start_token,
                    end_token=self.spatial_end_token,
                )
            normalized.append(sample)
        return normalized[0] if single_input else normalized

    @staticmethod
    def _ensure_modality_boundaries(
        sample: str,
        placeholder: str,
        start_token: str,
        end_token: str,
    ) -> str:
        if placeholder not in sample:
            return sample
        has_start = start_token in sample
        has_end = end_token in sample
        if not has_start and not has_end:
            return sample.replace(
                placeholder,
                f"{start_token}{placeholder}{end_token}",
                1,
            )
        return sample

    def _validate_text_modal_placeholders(
        self,
        text,
        foa_mask,
        has_video: bool,
        allow_mono_spatial_tokens: bool = False,
    ):
        if text is None:
            raise ValueError("text is required so the processor can validate placeholders.")
        text_list = [text] if not isinstance(text, list) else text
        if len(text_list) != int(foa_mask.shape[0]):
            raise ValueError(
                f"#text vs #audio mismatch: {len(text_list)} vs {int(foa_mask.shape[0])}"
            )
        for index, (sample, is_foa) in enumerate(zip(text_list, foa_mask.tolist())):
            if not isinstance(sample, str):
                raise ValueError("Spatial processor currently supports string prompts only.")
            audio_count = sample.count(self.audio_token)
            if audio_count != 1:
                raise ValueError(
                    f"Sample {index} must contain exactly one {self.audio_token}, found {audio_count}."
                )
            spatial_count = sample.count(self.spatial_token)
            if is_foa and spatial_count != 1:
                raise ValueError(
                    f"FOA sample {index} must contain exactly one {self.spatial_token}, found {spatial_count}."
                )
            if not is_foa and not allow_mono_spatial_tokens and spatial_count != 0:
                raise ValueError(
                    f"Mono sample {index} must not contain {self.spatial_token}."
                )
            if not is_foa and allow_mono_spatial_tokens and spatial_count != 1:
                raise ValueError(
                    f"Mono sample {index} must contain exactly one {self.spatial_token} "
                    "when allow_mono_spatial_tokens=True."
                )
            if is_foa or allow_mono_spatial_tokens:
                boundary_counts = {
                    self.audio_bos_token: sample.count(self.audio_bos_token),
                    self.audio_eos_token: sample.count(self.audio_eos_token),
                    self.spatial_start_token: sample.count(self.spatial_start_token),
                    self.spatial_end_token: sample.count(self.spatial_end_token),
                }
                invalid_boundaries = {
                    token: count for token, count in boundary_counts.items() if count != 1
                }
                if invalid_boundaries:
                    raise ValueError(
                        f"Sample {index} has invalid modality boundaries: {invalid_boundaries}."
                    )
                ordered_tokens = (
                    self.audio_bos_token,
                    self.audio_token,
                    self.audio_eos_token,
                    self.spatial_start_token,
                    self.spatial_token,
                    self.spatial_end_token,
                )
                positions = [sample.find(token) for token in ordered_tokens]
                if positions != sorted(positions) or len(set(positions)) != len(positions):
                    raise ValueError(
                        f"Sample {index} must order modalities as "
                        "audio_start/audio/audio_end/spatial_start/spatial/spatial_end."
                    )

    def _expand_spatial_placeholders(self, text, spatial_token_lengths):
        if spatial_token_lengths is None:
            return text
        single_input = not isinstance(text, list)
        text_list = [text] if single_input else text
        if len(text_list) != int(spatial_token_lengths.shape[0]):
            raise ValueError(
                f"#text vs spatial_token_lengths mismatch: {len(text_list)} vs {int(spatial_token_lengths.shape[0])}"
            )
        expanded: List[str] = []
        for sample, token_length in zip(text_list, spatial_token_lengths.tolist()):
            count = sample.count(self.spatial_token)
            if count > 1:
                raise ValueError("Each sample may contain at most one unexpanded <|spatial|> placeholder.")
            if count == 1:
                if int(token_length) <= 0:
                    raise ValueError("Sample with <|spatial|> must have positive spatial token length.")
                sample = sample.replace(
                    self.spatial_token, self.spatial_token * int(token_length), 1
                )
            expanded.append(sample)
        return expanded[0] if single_input else expanded

    # ProcessorMixin.get_attributes() introspects __init__ params and assumes
    # every modality attribute is a non-None object exposing `_auto_class`.
    # Our audio-only subclass keeps image_processor/video_processor=None, so
    # we override get_attributes to only enumerate what's actually present.
    @classmethod
    def get_attributes(cls):
        return ["feature_extractor", "tokenizer"]

    @property
    def model_input_names(self):
        # The native property dereferences image_processor/video_processor,
        # which are intentionally absent in this audio-only processor.
        base_names = (
            list(self.tokenizer.model_input_names)
            + list(self.feature_extractor.model_input_names)
            + ["feature_attention_mask"]
        )
        return list(
            dict.fromkeys(
                base_names
                + [
                    "spatial_audio",
                    "spatial_audio_attention_mask",
                    "spatial_audio_lengths",
                    "spatial_tokens",
                    "projected_spatial_tokens",
                    "spatial_token_lengths",
                ]
            )
        )


def register_qwen3_spatial_processor() -> None:
    from transformers import AutoProcessor

    AutoProcessor.register(
        Qwen3OmniMoeSpatialThinkerConfig,
        Qwen3OmniMoeSpatialProcessor,
        exist_ok=True,
    )


register_qwen3_spatial_processor()


__all__ = [
    "Qwen3OmniMoeSpatialProcessor",
    "register_qwen3_spatial_processor",
]
