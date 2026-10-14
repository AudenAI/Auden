# Adapted from Spatial-Omni for self-contained inference in Auden.
"""Prepare native-audio features and FOA spatial inputs for SO-7B."""

from __future__ import annotations

from fractions import Fraction
import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from .processing_qwen2_5_omni import (
    AudioInput,
    BatchFeature,
    ImageInput,
    PreTokenizedInput,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniProcessorKwargs,
    TextInput,
    Unpack,
    VideoInput,
)
from ..utils.spatial_seld_utils import (
    attention_mask_to_lengths,
    build_1d_attention_mask,
    clamp_lengths,
)


class Qwen2_5OmniSpatialProcessor(Qwen2_5OmniProcessor):
    """Prepare native audio and FOA spatial tensors, placeholders, and masks for SO-7B."""

    valid_kwargs = Qwen2_5OmniProcessor.valid_kwargs + [
        "spatial_token_lengths",
        "spatial_audio_max_seconds",
        "spatial_tokens",
        "projected_spatial_tokens",
        "seld_features",
        "seld_feature_attention_mask",
        "seld_feature_lengths",
        "seld_hidden_states",
        "seld_hidden_attention_mask",
        "seld_hidden_lengths",
        "allow_mono_spatial_tokens",
    ]

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(
            image_processor=image_processor,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self.spatial_token = "<|spatial|>"
        self.audio_token_aliases = ("<|audio|>",)
        self.video_token_aliases = ("<|video|>",)
        self.spatial_encoder_type = "so_backbone"
        self.so_backbone_target_token_rate = 2.5
        self.seld_num_feature_channels = 7
        self.seld_num_mel_bins = 64
        self.seld_hop_length = 320
        self.seld_feature_to_seld_ratio = 5
        self.seld_downsample_factor = 4
        self.spatial_token_id = self._register_spatial_token_on_tokenizer()
        valid = getattr(self, "_valid_kwargs", None)
        if not isinstance(valid, set):
            valid = set(valid or [])
        valid.update(
            {
                "spatial_token_lengths",
                "spatial_audio_max_seconds",
                "spatial_tokens",
                "projected_spatial_tokens",
                "seld_features",
                "seld_feature_attention_mask",
                "seld_feature_lengths",
                "seld_hidden_states",
                "seld_hidden_attention_mask",
                "seld_hidden_lengths",
                "allow_mono_spatial_tokens",
            }
        )
        self._valid_kwargs = valid

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
        rate = Fraction(str(self.so_backbone_target_token_rate)).limit_denominator(1000)
        numerator = num_samples.to(dtype=torch.long) * int(rate.numerator)
        denominator = int(self.feature_extractor.sampling_rate) * int(rate.denominator)
        steps = self._round_divide_half_to_even(numerator, denominator)
        return torch.clamp(steps, min=1)

    def _register_spatial_token_on_tokenizer(self) -> int:
        """Register `<|spatial|>` as a tokenizer special token and return its id.

        Processing:
            1. Check whether `<|spatial|>` already exists in the tokenizer
               vocabulary.
            2. If not, add it through `additional_special_tokens`.
            3. Resolve and return the token id used later for masking.

        Returns:
            Integer tokenizer id for `<|spatial|>`.
        """

        vocab = self.tokenizer.get_vocab()
        if self.spatial_token not in vocab:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.spatial_token]}
            )
        return int(self.tokenizer.convert_tokens_to_ids(self.spatial_token))

    def sync_spatial_tokenizer_with_model(self, model) -> int:
        """Synchronize tokenizer and model embeddings for `<|spatial|>`.

        Args:
            model:
                A spatial-aware model instance that exposes
                `sync_spatial_tokenizer(tokenizer, spatial_token=...)`.

        Returns:
            The integer token id assigned to `<|spatial|>`.
        """

        if not hasattr(model, "sync_spatial_tokenizer"):
            raise TypeError(
                "model must expose a sync_spatial_tokenizer(tokenizer, spatial_token=...) method."
            )
        token_id = int(model.sync_spatial_tokenizer(self.tokenizer, spatial_token=self.spatial_token))
        self.spatial_token_id = token_id
        thinker_config = None
        if hasattr(model, "config") and hasattr(model.config, "thinker_config"):
            thinker_config = model.config.thinker_config
        elif hasattr(model, "thinker") and hasattr(model.thinker, "config"):
            thinker_config = model.thinker.config
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
            self.seld_hop_length = int(
                getattr(thinker_config, "seld_hop_length", self.seld_hop_length)
            )
            self.seld_num_feature_channels = int(
                getattr(
                    thinker_config,
                    "seld_num_feature_channels",
                    self.seld_num_feature_channels,
                )
            )
            self.seld_num_mel_bins = int(
                getattr(
                    thinker_config,
                    "seld_num_mel_bins",
                    self.seld_num_mel_bins,
                )
            )
            self.seld_feature_to_seld_ratio = int(
                getattr(
                    thinker_config,
                    "seld_feature_to_seld_ratio",
                    self.seld_feature_to_seld_ratio,
                )
            )
            self.seld_downsample_factor = int(
                getattr(
                    thinker_config,
                    "seld_downsample_factor",
                    self.seld_downsample_factor,
                )
            )
        return token_id

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
        """Package audio inputs for both the original audio path and `<|spatial|>`.

        Args:
            text:
                Raw text or batch of raw texts.
            audio:
                Mono or FOA waveform batch.
            spatial_token_lengths:
                Optional precomputed lengths `[B]`. If omitted and `audio` is
                present, lengths are derived from waveform masks.
            spatial_audio_max_seconds:
                Optional override for the spatial branch window size.

        Returns:
            [`BatchFeature`] with the additional spatial fields described in the
            class docstring.
        """

        if audio is None:
            raise ValueError("Qwen2_5OmniSpatialProcessor requires `audio`; audio may be mono or 4-channel FOA.")

        spatial_token_lengths = kwargs.pop("spatial_token_lengths", None)
        spatial_tokens = kwargs.pop("spatial_tokens", None)
        projected_spatial_tokens = kwargs.pop("projected_spatial_tokens", None)
        seld_features = kwargs.pop("seld_features", None)
        seld_feature_attention_mask = kwargs.pop("seld_feature_attention_mask", None)
        seld_feature_lengths = kwargs.pop("seld_feature_lengths", None)
        seld_hidden_states = kwargs.pop("seld_hidden_states", None)
        seld_hidden_attention_mask = kwargs.pop("seld_hidden_attention_mask", None)
        seld_hidden_lengths = kwargs.pop("seld_hidden_lengths", None)
        spatial_audio_max_seconds = float(kwargs.pop("spatial_audio_max_seconds", 20.0))
        allow_mono_spatial_tokens = bool(kwargs.pop("allow_mono_spatial_tokens", False))
        if spatial_audio_max_seconds <= 0:
            raise ValueError(f"spatial_audio_max_seconds must be > 0, got {spatial_audio_max_seconds}")

        normalized_audio = self._normalize_audio_list(audio)
        normalized_audio = self._truncate_audio_list(
            normalized_audio,
            max_seconds=spatial_audio_max_seconds,
        )
        foa_mask = torch.tensor([sample.shape[0] == 4 for sample in normalized_audio], dtype=torch.bool)
        if foa_mask.any() and (~foa_mask).any():
            raise ValueError(
                "Mixed mono/FOA batches are not supported. "
                "Each batch must be entirely mono or entirely 4-channel FOA."
            )
        batch_size = len(normalized_audio)
        has_video = self._has_video_inputs(videos)
        if has_video and not bool(foa_mask.all().item()):
            raise ValueError(
                "Video input is currently supported only for FOA batches with the spatial branch enabled."
            )

        explicit_spatial_payload = self._build_explicit_spatial_payload(
            batch_size=batch_size,
            spatial_tokens=spatial_tokens,
            projected_spatial_tokens=projected_spatial_tokens,
            spatial_token_lengths=spatial_token_lengths,
            seld_features=seld_features,
            seld_feature_attention_mask=seld_feature_attention_mask,
            seld_feature_lengths=seld_feature_lengths,
            seld_hidden_states=seld_hidden_states,
            seld_hidden_attention_mask=seld_hidden_attention_mask,
            seld_hidden_lengths=seld_hidden_lengths,
            allow_mono_spatial_tokens=allow_mono_spatial_tokens,
        )
        effective_spatial_token_lengths = explicit_spatial_payload.get("spatial_token_lengths")
        if (
            not allow_mono_spatial_tokens
            and not bool(foa_mask.all().item())
            and explicit_spatial_payload
        ):
            raise ValueError(
                "Spatial inputs are only supported for FOA batches. "
                "Mono batches must not provide spatial_tokens, seld_features, or seld_hidden_states."
            )

        spatial_payload = None
        processed_text = self._normalize_prompt_special_tokens(text)
        if (
            foa_mask.any()
            and "spatial_tokens" not in explicit_spatial_payload
            and "projected_spatial_tokens" not in explicit_spatial_payload
            and "seld_features" not in explicit_spatial_payload
            and "seld_hidden_states" not in explicit_spatial_payload
        ):
            spatial_payload = self._build_spatial_audio_payload(
                normalized_audio,
                foa_mask=foa_mask,
                max_seconds=spatial_audio_max_seconds,
            )
            if effective_spatial_token_lengths is None:
                effective_spatial_token_lengths = spatial_payload["spatial_token_lengths"]

        if effective_spatial_token_lengths is not None:
            if effective_spatial_token_lengths.shape[0] != batch_size:
                raise ValueError(
                    "spatial_token_lengths batch size must match audio batch size: "
                    f"{int(effective_spatial_token_lengths.shape[0])} vs {batch_size}"
                )
            invalid_mono = (~foa_mask) & (effective_spatial_token_lengths > 0)
            if invalid_mono.any() and not allow_mono_spatial_tokens:
                invalid_indices = torch.nonzero(invalid_mono, as_tuple=False).flatten().tolist()
                raise ValueError(
                    "Mono samples must not carry spatial inputs. Invalid indices: "
                    f"{invalid_indices}"
                )
            invalid_foa = foa_mask & (effective_spatial_token_lengths <= 0)
            if invalid_foa.any():
                invalid_indices = torch.nonzero(invalid_foa, as_tuple=False).flatten().tolist()
                raise ValueError(
                    "FOA samples must provide positive spatial token lengths. Invalid indices: "
                    f"{invalid_indices}"
                )

        self._validate_text_modal_placeholders(
            text=processed_text,
            foa_mask=foa_mask,
            has_video=has_video,
            allow_mono_spatial_tokens=allow_mono_spatial_tokens,
        )

        if processed_text is not None:
            processed_text = self._expand_spatial_placeholders(
                text=processed_text,
                spatial_token_lengths=effective_spatial_token_lengths,
            )

        batch = super().__call__(
            text=processed_text,
            images=images,
            videos=videos,
            audio=normalized_audio,
            return_spatial_audio=False,
            spatial_features_from_processor=False,
            **kwargs,
        )

        if spatial_payload is not None:
            for key, value in spatial_payload.items():
                batch[key] = value
        for key, value in explicit_spatial_payload.items():
            batch[key] = value
        if effective_spatial_token_lengths is not None:
            batch["spatial_token_lengths"] = effective_spatial_token_lengths
        return batch

    def _normalize_audio_list(self, audio: AudioInput) -> List[np.ndarray]:
        """Normalize audio inputs into `[C, T_audio]` float32 arrays.

        Supported channel layouts per sample:
            - mono: `[T]`, `[1, T]`, or `[T, 1]`
            - FOA: `[4, T]` or `[T, 4]`

        Returns:
            List of arrays with shape `[C, T_audio]`, where `C` is `1` or `4`.
        """

        if not isinstance(audio, (list, tuple)):
            audio = [audio]

        normalized: List[np.ndarray] = []
        for index, item in enumerate(audio):
            array = np.asarray(item, dtype=np.float32)
            if array.ndim == 1:
                normalized.append(array[None, :].astype(np.float32, copy=False))
                continue
            if array.ndim != 2:
                raise ValueError(
                    f"Audio item {index} must be 2D, got shape {array.shape}"
                )
            if array.shape[0] in (1, 4):
                channels_first = array
            elif array.shape[1] in (1, 4):
                channels_first = array.T
            else:
                raise ValueError(
                    f"Audio item {index} must be mono or 4-channel FOA, got shape {array.shape}"
                )
            normalized.append(channels_first.astype(np.float32, copy=False))
        return normalized

    def _truncate_audio_list(self, audio_list: List[np.ndarray], max_seconds: float) -> List[np.ndarray]:
        """Truncate every audio sample to `0-max_seconds` and warn when clipping.

        Input:
            `audio_list`: list of `[C, T_audio]` arrays where `C` is `1` or `4`.

        Output:
            List of `[C, min(T_audio, T_audio_max)]` arrays using the same
            channel layout as the input.
        """

        max_samples = int(round(max_seconds * self.feature_extractor.sampling_rate))
        clipped: List[np.ndarray] = []
        for index, array in enumerate(audio_list):
            if array.shape[1] > max_samples:
                warnings.warn(
                    "Audio sample "
                    f"{index} is longer than {max_seconds:.1f}s and will be truncated "
                    f"from {array.shape[1] / self.feature_extractor.sampling_rate:.2f}s "
                    f"to {max_seconds:.2f}s.",
                    stacklevel=2,
                )
                array = array[:, :max_samples]
            clipped.append(array)
        return clipped

    def _normalize_prompt_special_tokens(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None],
    ) -> Union[List[str], TextInput, PreTokenizedInput, None]:
        """Normalize prompt token aliases to the tokenizer's canonical tokens.

        Current normalization:
            - replace `<|audio|>` with the canonical audio token used by the
              underlying QwenOmni tokenizer, typically `<|AUDIO|>`.
            - replace `<|video|>` with the canonical video token used by the
              underlying QwenOmni tokenizer, typically `<|VIDEO|>`.

        Input and output preserve the original batch container shape.
        """

        if text is None:
            return None

        if not isinstance(text, list):
            text_list = [text]
            single_input = True
        else:
            text_list = text
            single_input = False

        normalized: List[Union[str, PreTokenizedInput]] = []
        for sample in text_list:
            if isinstance(sample, str):
                for alias in self.audio_token_aliases:
                    sample = sample.replace(alias, self.audio_token)
                for alias in self.video_token_aliases:
                    sample = sample.replace(alias, self.video_token)
            normalized.append(sample)
        return normalized[0] if single_input else normalized

    def _has_video_inputs(self, videos: VideoInput) -> bool:
        """Return `True` when the caller provided video content for the batch."""

        if videos is None:
            return False
        if isinstance(videos, (list, tuple)):
            return len(videos) > 0
        return True

    def _build_explicit_spatial_payload(
        self,
        batch_size: int,
        spatial_tokens: Optional[torch.Tensor],
        projected_spatial_tokens: Optional[torch.Tensor],
        spatial_token_lengths: Optional[torch.Tensor],
        seld_features: Optional[torch.Tensor],
        seld_feature_attention_mask: Optional[torch.Tensor],
        seld_feature_lengths: Optional[torch.Tensor],
        seld_hidden_states: Optional[torch.Tensor],
        seld_hidden_attention_mask: Optional[torch.Tensor],
        seld_hidden_lengths: Optional[torch.Tensor],
        allow_mono_spatial_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Normalize optional precomputed spatial inputs for processor output.

        Supported explicit inputs:
            - `spatial_tokens`: `[B, T_spat_max, D_spat]`
            - `projected_spatial_tokens`: `[B, T_spat_max, D_llm]`
            - `seld_features`: `[B, 7, T_feat_max, 64]`
            - `seld_feature_attention_mask`: `[B, T_feat_max]`
            - `seld_feature_lengths`: `[B]`
            - `seld_hidden_states`: `[B, T_seld_max, 128]`
            - `seld_hidden_attention_mask`: `[B, T_seld_max]`
            - `seld_hidden_lengths`: `[B]`

        Returns:
            A dictionary ready to be merged into the processor batch. When
            `spatial_token_lengths` is not supplied explicitly, it is inferred
            from the provided tokens or feature lengths.
        """

        if seld_features is not None or seld_hidden_states is not None:
            raise ValueError("SELD payloads are not supported by this example.")
        payload: dict[str, torch.Tensor] = {}
        resolved_spatial_lengths = None if spatial_token_lengths is None else torch.as_tensor(spatial_token_lengths, dtype=torch.long)
        has_explicit_source = (
            spatial_tokens is not None
            or projected_spatial_tokens is not None
            or seld_features is not None
            or seld_hidden_states is not None
        )
        if resolved_spatial_lengths is not None and not has_explicit_source:
            if not allow_mono_spatial_tokens:
                raise ValueError(
                    "spatial_token_lengths cannot be provided without either spatial_tokens or seld_features."
                )
            # mono-replay path: caller (mixed-replay collator) explicitly
            # opts into expanding `<|spatial|>` placeholders without supplying
            # the actual spatial source. The model fills these placeholders
            # at forward time from learned `spatial_null` + the spatial encoder.
            payload["spatial_token_lengths"] = resolved_spatial_lengths
            return payload

        if spatial_tokens is not None:
            tokens = torch.as_tensor(spatial_tokens)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 3:
                raise ValueError(
                    "spatial_tokens must have shape [B, T_spat_max, D_spat], "
                    f"got {tuple(tokens.shape)}"
                )
            if tokens.shape[0] != batch_size:
                raise ValueError(
                    "spatial_tokens batch size must match audio batch size: "
                    f"{tokens.shape[0]} vs {batch_size}"
                )
            payload["spatial_tokens"] = tokens
            if resolved_spatial_lengths is None:
                resolved_spatial_lengths = torch.full(
                    (batch_size,),
                    fill_value=tokens.shape[1],
                    dtype=torch.long,
                )
            elif (resolved_spatial_lengths > tokens.shape[1]).any() or (resolved_spatial_lengths < 0).any():
                raise ValueError(
                    "spatial_token_lengths must stay within [0, T_spat_max] when spatial_tokens are provided."
                )

        if projected_spatial_tokens is not None:
            tokens = torch.as_tensor(projected_spatial_tokens)
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 3:
                raise ValueError(
                    "projected_spatial_tokens must have shape [B, T_spat_max, D_llm], "
                    f"got {tuple(tokens.shape)}"
                )
            if tokens.shape[0] != batch_size:
                raise ValueError(
                    "projected_spatial_tokens batch size must match audio batch size: "
                    f"{tokens.shape[0]} vs {batch_size}"
                )
            payload["projected_spatial_tokens"] = tokens
            if resolved_spatial_lengths is None:
                resolved_spatial_lengths = torch.full(
                    (batch_size,),
                    fill_value=tokens.shape[1],
                    dtype=torch.long,
                )
            elif (resolved_spatial_lengths > tokens.shape[1]).any() or (resolved_spatial_lengths < 0).any():
                raise ValueError(
                    "spatial_token_lengths must stay within [0, T_spat_max] "
                    "when projected_spatial_tokens are provided."
                )



        if resolved_spatial_lengths is not None:
            if resolved_spatial_lengths.ndim != 1 or resolved_spatial_lengths.shape[0] != batch_size:
                raise ValueError(
                    "spatial_token_lengths must have shape [B], "
                    f"got {tuple(resolved_spatial_lengths.shape)}"
                )
            payload["spatial_token_lengths"] = resolved_spatial_lengths.to(dtype=torch.long)

        return payload

    def _build_spatial_audio_payload(
        self,
        audio_list: List[np.ndarray],
        foa_mask: torch.BoolTensor,
        max_seconds: float,
    ) -> dict[str, torch.Tensor]:
        """Pad or truncate FOA audio into the fixed spatial-branch window.

        Args:
            audio_list:
                List of channels-first arrays shaped `[C, T_audio]`, where `C`
                is `1` or `4`.
            foa_mask:
                Boolean tensor `[B]`. `True` marks the samples that should enter
                the spatial branch.
            max_seconds:
                Window size in seconds. Default design target is `20 s`.

        Returns:
            Dictionary with:
            - `spatial_audio`: `[B, T_audio_max, 4]`
            - `spatial_audio_attention_mask`: `[B, T_audio_max]`
            - `spatial_audio_lengths`: `[B]`
            - `spatial_token_lengths`: `[B]`
        """

        max_samples = int(round(max_seconds * self.feature_extractor.sampling_rate))
        batch_size = len(audio_list)
        spatial_audio = np.zeros((batch_size, max_samples, 4), dtype=np.float32)
        spatial_lengths = np.zeros((batch_size,), dtype=np.int64)
        for idx, array in enumerate(audio_list):
            if not bool(foa_mask[idx].item()):
                continue
            valid_samples = min(array.shape[1], max_samples)
            spatial_audio[idx, :valid_samples, :] = array[:, :valid_samples].T
            spatial_lengths[idx] = valid_samples

        spatial_audio_lengths = torch.from_numpy(spatial_lengths)
        spatial_audio_attention_mask = build_1d_attention_mask(
            spatial_audio_lengths,
            max_length=max_samples,
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

    def _validate_text_modal_placeholders(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], None],
        foa_mask: torch.BoolTensor,
        has_video: bool,
        allow_mono_spatial_tokens: bool = False,
    ) -> None:
        """Validate audio/spatial placeholder counts against audio channel mode.

        Rules:
            - every sample must contain exactly one `<|AUDIO|>`
            - FOA samples must contain exactly one `<|spatial|>`
            - mono samples must contain zero `<|spatial|>`
            - if videos are provided, every sample must contain exactly one
              `<|VIDEO|>`
            - if videos are not provided, `<|VIDEO|>` must not appear
            - modal order must be fixed as:
              `<|VIDEO|><|AUDIO|><|spatial|>` when video exists
              `<|AUDIO|><|spatial|>` when video is absent
        """

        if text is None:
            raise ValueError("text is required so the processor can validate <|AUDIO|> and <|spatial|> placeholders.")

        text_list = [text] if not isinstance(text, list) else text
        if len(text_list) != int(foa_mask.shape[0]):
            raise ValueError(
                "Number of text samples and audio samples must match: "
                f"{len(text_list)} vs {int(foa_mask.shape[0])}"
            )

        for index, (sample, is_foa) in enumerate(zip(text_list, foa_mask.tolist())):
            if not isinstance(sample, str):
                raise ValueError("Spatial processor currently supports string prompts only.")
            video_count = sample.count(self.video_token)
            expected_video_count = 1 if has_video else 0
            if video_count != expected_video_count:
                raise ValueError(
                    f"Sample {index} must contain exactly {expected_video_count} {self.video_token}, found {video_count}."
                )
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
                    f"Mono sample {index} must not contain {self.spatial_token}, found {spatial_count}."
                )
            if not is_foa and allow_mono_spatial_tokens and spatial_count != 1:
                raise ValueError(
                    f"Mono sample {index} must contain exactly one {self.spatial_token} "
                    f"when allow_mono_spatial_tokens=True, found {spatial_count}."
                )
            video_pos = sample.find(self.video_token) if video_count else -1
            audio_pos = sample.find(self.audio_token)
            spatial_pos = sample.find(self.spatial_token) if spatial_count else -1
            if has_video and not (video_pos < audio_pos):
                raise ValueError(
                    f"Sample {index} must place {self.video_token} before {self.audio_token}."
                )
            if is_foa or allow_mono_spatial_tokens:
                if not (audio_pos < spatial_pos):
                    raise ValueError(
                        f"Sample {index} must place {self.audio_token} before {self.spatial_token}."
                    )

    def _expand_spatial_placeholders(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        spatial_token_lengths: Optional[torch.Tensor],
    ) -> Union[List[str], TextInput, PreTokenizedInput]:
        """Expand one `<|spatial|>` placeholder into `T_spat` repeated tokens.

        Args:
            text:
                Raw text or batch of texts.
            spatial_token_lengths:
                Tensor of shape `[B]` or `None`.

        Returns:
            Text with each `<|spatial|>` expanded to repeated placeholder tokens.
        """

        if spatial_token_lengths is None:
            return text

        if not isinstance(text, list):
            text_list = [text]
            single_input = True
        else:
            text_list = text
            single_input = False

        if len(text_list) != int(spatial_token_lengths.shape[0]):
            raise ValueError(
                "Number of text samples and spatial_token_lengths entries must match: "
                f"{len(text_list)} vs {int(spatial_token_lengths.shape[0])}"
            )

        expanded: List[str] = []
        for sample, token_length in zip(text_list, spatial_token_lengths.tolist()):
            if not isinstance(sample, str):
                raise ValueError("Spatial placeholder expansion currently supports string inputs only.")
            count = sample.count(self.spatial_token)
            if count > 1:
                raise ValueError("Each sample may contain at most one <|spatial|> placeholder.")
            if count == 1:
                if int(token_length) <= 0:
                    raise ValueError("A sample containing <|spatial|> must have a positive spatial token length.")
                sample = sample.replace(self.spatial_token, self.spatial_token * int(token_length), 1)
            expanded.append(sample)

        return expanded[0] if single_input else expanded

    @property
    def model_input_names(self):
        """Extend the base processor input names with spatial-branch tensors."""

        base_names = super().model_input_names
        return list(
            dict.fromkeys(
                base_names
                + [
                    "spatial_audio",
                    "spatial_audio_attention_mask",
                    "spatial_audio_lengths",
                    "spatial_tokens",
                    "projected_spatial_tokens",
                    "seld_features",
                    "seld_feature_attention_mask",
                    "seld_feature_lengths",
                    "seld_hidden_states",
                    "seld_hidden_attention_mask",
                    "seld_hidden_lengths",
                    "spatial_token_lengths",
                ]
            )
        )


__all__ = ["Qwen2_5OmniSpatialProcessor"]
