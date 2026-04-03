import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel

from .configuration_lalm import LALMConfig


class LALMProjector(nn.Module):
    """
    Temporal downsampling projector that maps audio encoder outputs into the LLM's
    embedding space.

    ``downsample_rate`` consecutive frames are concatenated along the feature axis
    before the linear projection, reducing the sequence length fed to the LLM by
    that factor. E.g. with Whisper (50 frames/sec) and ``downsample_rate=5``, the
    LLM receives 10 audio tokens per second.
    """

    def __init__(self, audio_dim: int, text_dim: int, downsample_rate: int = 5):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(audio_dim * downsample_rate, text_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(text_dim, text_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        remainder = T % self.downsample_rate
        if remainder:
            x = x[:, :-remainder, :]
        x = x.reshape(B, T // self.downsample_rate, D * self.downsample_rate)
        return self.linear2(self.act(self.linear1(x)))


class LALMForConditionalGeneration(PreTrainedModel):
    """
    LALM: audio encoder + projector + causal LLM.

    During forward, audio features are encoded, projected, then inserted at
    positions marked by ``modality_token_id`` in the input token sequence.

    Note on audio encoder:
        ``audio_tower`` should be an *encoder-only* module whose output has a
        ``last_hidden_state`` attribute (e.g. ``WhisperEncoder``, or
        ``WhisperModel.encoder``). Full seq2seq models such as ``WhisperModel``
        return the *decoder*'s ``last_hidden_state`` and must be unwrapped first
        (``assemble_model.py`` handles this automatically).
    """

    config_class = LALMConfig

    def __init__(
        self,
        config: LALMConfig,
        language_model: nn.Module | None = None,
        audio_tower: nn.Module | None = None,
    ):
        super().__init__(config)
        self.language_model = (
            language_model
            if language_model is not None
            else AutoModelForCausalLM.from_config(config.text_config)
        )
        self.audio_tower = (
            audio_tower
            if audio_tower is not None
            else self._build_audio_tower(config.audio_config)
        )
        self.projector = LALMProjector(
            config.audio_dim, config.text_dim, config.projector_downsample_rate
        )

    @staticmethod
    def _build_audio_tower(audio_config):
        """Build encoder explicitly to avoid relying on AutoModel registration."""
        model_type = getattr(audio_config, "model_type", "")

        if model_type == "whisper":
            from transformers.models.whisper.modeling_whisper import WhisperEncoder

            return WhisperEncoder(audio_config)

        if model_type == "qwen2_5_omni_audio_encoder":
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniAudioEncoder,
            )

            return Qwen2_5OmniAudioEncoder(audio_config)

        if model_type == "qwen3_omni_moe_audio_encoder":
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeAudioEncoder,
            )

            return Qwen3OmniMoeAudioEncoder(audio_config)

        return AutoModel.from_config(audio_config)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_lens: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text auto-regressively, optionally conditioned on audio.

        Parameter names mirror the processor's output keys so that the standard
        ``model.generate(**inputs, feature_lens=..., **gen_cfg)`` pattern works.

        Args:
            input_ids: Token ids with audio-token placeholders, (N, L).
            attention_mask: Padding mask, (N, L).
            input_features: Packed audio features (C, T_total).
            feature_lens: Actual frame counts per sample (N,).
            **generate_kwargs: Forwarded to ``language_model.generate``.

        Returns:
            Generated token ids, (N, T_gen).
        """
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if input_features is not None:
            audio_param = next(self.audio_tower.parameters(), None)
            if audio_param is not None and input_features.dtype != audio_param.dtype:
                input_features = input_features.to(dtype=audio_param.dtype)
            inputs_embeds = self._merge_input_ids_with_audio_features(
                input_ids, inputs_embeds, input_features, feature_lens=feature_lens
            )
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if audio_features is not None:
            inputs_embeds = self._merge_input_ids_with_audio_features(
                input_ids,
                inputs_embeds,
                audio_features,
                feature_lens=feature_lens,
            )

        if labels is not None and attention_mask is not None:
            outputs, packed_labels = self._forward_packed(
                inputs_embeds, attention_mask, labels, **kwargs
            )
            outputs.packed_labels = packed_labels
            return outputs

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )

    def encode_audio(
        self,
        audio_features: torch.Tensor,
        feature_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run the packed audio tower and return padded hidden states."""
        model_type = getattr(self.config.audio_config, "model_type", "")
        if feature_lens is None:
            raise ValueError("Packed-only LALM requires feature_lens.")

        feature_lens = feature_lens.to(device=audio_features.device, dtype=torch.long)
        if model_type == "qwen2_5_omni_audio_encoder":
            aftercnn_lens = (feature_lens - 1) // 2 + 1
            encoded = self.audio_tower(
                input_features=audio_features,
                feature_lens=feature_lens,
                aftercnn_lens=aftercnn_lens,
            ).last_hidden_state
            output_lens = (aftercnn_lens - 2) // 2 + 1
            return self._pad_packed_audio_outputs(encoded, output_lens)

        if model_type == "qwen3_omni_moe_audio_encoder":
            encoded = self.audio_tower(
                input_features=audio_features,
                feature_lens=feature_lens,
            ).last_hidden_state
            output_lens = self._get_audio_output_lengths(feature_lens)
            return self._pad_packed_audio_outputs(encoded, output_lens)

        raise NotImplementedError(
            "Packed-only LALM currently supports "
            "'qwen2_5_omni_audio_encoder' and 'qwen3_omni_moe_audio_encoder'."
        )

    def _get_audio_output_lengths(self, feature_lens: torch.Tensor) -> torch.Tensor:
        model_type = getattr(self.config.audio_config, "model_type", "")

        if model_type == "whisper":
            return (feature_lens - 1) // 2 + 1

        if model_type == "qwen2_5_omni_audio_encoder":
            after_first = (feature_lens - 1) // 2 + 1
            return (after_first - 2) // 2 + 1

        if model_type == "qwen3_omni_moe_audio_encoder":
            remainder = feature_lens % 100
            feat = (remainder - 1) // 2 + 1
            return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (feature_lens // 100) * 13

        raise ValueError(f"Unsupported audio encoder model_type: {model_type!r}")

    def _pad_packed_audio_outputs(
        self, encoded: torch.Tensor, output_lens: torch.Tensor
    ) -> torch.Tensor:
        if encoded.ndim != 2:
            return encoded

        lengths = [int(x) for x in output_lens.tolist()]
        max_len = max(lengths)
        padded = encoded.new_zeros((len(lengths), max_len, encoded.size(-1)))
        chunks = encoded.split(lengths, dim=0)
        for i, chunk in enumerate(chunks):
            padded[i, : chunk.size(0)] = chunk
        return padded

    def _forward_packed(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        device = inputs_embeds.device

        seq_lens = attention_mask.sum(dim=1).long()
        segment_lens = seq_lens.tolist()

        valid = attention_mask.bool()
        packed_embeds = inputs_embeds[valid].unsqueeze(0)
        packed_labels = labels[valid].unsqueeze(0)

        T_total = packed_embeds.size(1)
        ones = torch.ones(T_total, device=device, dtype=torch.long)
        if seq_lens.size(0) > 1:
            starts = seq_lens.cumsum(0)[:-1]
            ones[starts] -= seq_lens[:-1]
        position_ids = ones.cumsum(0).sub_(1).unsqueeze(0)

        attn_mask = _block_diagonal_causal_mask(
            segment_lens, device=device, dtype=inputs_embeds.dtype
        )
        outputs = self.language_model(
            inputs_embeds=packed_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            labels=packed_labels,
            **kwargs,
        )
        return outputs, packed_labels

    def _merge_input_ids_with_audio_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        audio_features: torch.Tensor,
        feature_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Replace audio token positions in inputs_embeds with projected audio embeddings."""
        audio_embeds = self.projector(
            self.encode_audio(
                audio_features,
                feature_lens=feature_lens,
            )
        )
        mask = input_ids == self.config.audio_token_id
        if not mask.any():
            return inputs_embeds

        B, audio_len, D = audio_embeds.shape
        if B != input_ids.shape[0]:
            raise ValueError(
                f"Batch size mismatch: input_ids batch={input_ids.shape[0]}, "
                f"audio_embeds batch={B}."
            )
        if audio_len == 0:
            return inputs_embeds

        rank = mask.long().cumsum(dim=1) - 1
        valid = mask & (rank < audio_len)
        gather_index = rank.clamp(min=0, max=audio_len - 1)
        gathered_audio = audio_embeds.gather(
            1, gather_index.unsqueeze(-1).expand(-1, -1, D)
        ).to(inputs_embeds.dtype)

        out = inputs_embeds.reshape(-1, D)
        src = gathered_audio.reshape(-1, D)
        flat_valid = valid.reshape(-1)
        out[flat_valid] = src[flat_valid]
        return out.view_as(inputs_embeds)


def _block_diagonal_causal_mask(
    segment_lens: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    T = sum(segment_lens)
    seg_ids = torch.repeat_interleave(
        torch.arange(len(segment_lens), device=device),
        torch.tensor(segment_lens, device=device),
    )

    q_pos = torch.arange(T, device=device).unsqueeze(1)
    k_pos = torch.arange(T, device=device).unsqueeze(0)
    q_seg = seg_ids.unsqueeze(1)
    k_seg = seg_ids.unsqueeze(0)

    allow = (k_pos <= q_pos) & (k_seg == q_seg)

    mask = torch.zeros(1, 1, T, T, device=device, dtype=dtype)
    mask.masked_fill_(~allow, float("-inf"))
    return mask
