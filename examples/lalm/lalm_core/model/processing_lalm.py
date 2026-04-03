import torch
from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import TextInput

_ENCODER_LENGTH_FNS: dict[str, callable] = {}
_AUDIO_TOKEN_LENGTH_FNS: dict[str, callable] = {}


def _whisper_output_length(mel_frames: int) -> int:
    return (mel_frames - 1) // 2 + 1


def _qwen25_ae_output_length(mel_frames: int) -> int:
    after_first = (mel_frames - 1) // 2 + 1
    return (after_first - 2) // 2 + 1


def _qwen3_aut_output_length(mel_frames: int) -> int:
    remainder = mel_frames % 100
    feat = (remainder - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13


_ENCODER_LENGTH_FNS = {
    "whisper": _whisper_output_length,
    "qwen2_5_omni_audio_encoder": _qwen25_ae_output_length,
    "qwen3_omni_moe_audio_encoder": _qwen3_aut_output_length,
}


def make_audio_token_length_fn(
    encoder_name: str, projector_downsample_rate: int = 4
) -> callable:
    """
    Return a ``(mel_frames: int) -> int`` function combining encoder downsampling
    and projector frame-concat rate.
    """
    if encoder_name not in _ENCODER_LENGTH_FNS:
        raise ValueError(
            f"Unknown encoder {encoder_name!r}. Known: {list(_ENCODER_LENGTH_FNS)}."
        )
    if projector_downsample_rate <= 0:
        raise ValueError("projector_downsample_rate must be a positive integer.")
    name = f"{encoder_name}_ds{projector_downsample_rate}"
    if name not in _AUDIO_TOKEN_LENGTH_FNS:
        encoder_fn = _ENCODER_LENGTH_FNS[encoder_name]
        ds = projector_downsample_rate

        def fn(mel_frames: int) -> int:
            return encoder_fn(mel_frames) // ds

        fn.__name__ = name
        _AUDIO_TOKEN_LENGTH_FNS[name] = fn
    return _AUDIO_TOKEN_LENGTH_FNS[name]


class LALMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
        },
    }


class LALMProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        encoder_name: str | None = None,
        projector_downsample_rate: int = 4,
        audio_token: str = "<|audio|>",
    ):
        self.audio_token = audio_token
        self.encoder_name = encoder_name or "whisper"
        self.projector_downsample_rate = projector_downsample_rate
        if self.encoder_name not in _ENCODER_LENGTH_FNS:
            raise ValueError(
                f"Unknown encoder {self.encoder_name!r}. Known: {list(_ENCODER_LENGTH_FNS)}."
            )
        if self.projector_downsample_rate <= 0:
            raise ValueError("projector_downsample_rate must be a positive integer.")

        if self.audio_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.audio_token]}
            )
        super().__init__(feature_extractor, tokenizer)

    @property
    def audio_token_length_fn(self) -> callable:
        return make_audio_token_length_fn(
            self.encoder_name,
            self.projector_downsample_rate,
        )

    def apply_chat_template(
        self, conversations, chat_template=None, tokenize=False, **kwargs
    ) -> str | list[str]:
        is_batched = isinstance(conversations[0], list)
        conversations = conversations if is_batched else [conversations]
        results = [
            self.tokenizer.apply_chat_template(
                self._normalize_conversation(conv),
                tokenize=tokenize,
                chat_template=chat_template,
                **kwargs,
            )
            for conv in conversations
        ]
        return results if is_batched else results[0]

    def _normalize_conversation(self, conversation: list) -> list:
        out = []
        for turn in conversation:
            content = turn["content"]
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item["type"] == "audio":
                        parts.append(self.audio_token)
                    elif item["type"] == "text":
                        parts.append(item["text"])
                    else:
                        raise ValueError(
                            f"Unknown content type {item.get('type')!r} in multimodal message."
                        )
                content = " ".join(parts)
            out.append({**turn, "content": content})
        return out

    def __call__(
        self,
        text: TextInput | None = None,
        audio: AudioInput | None = None,
        audio_feature: tuple | None = None,
        prepare_labels: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")
        if audio is not None and audio_feature is not None:
            raise ValueError("Provide either `audio` or `audio_feature`, not both.")

        output_kwargs = self._merge_kwargs(
            LALMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            out = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            # feature_lens = number of non-padded mel frames per sample
            feature_lens = out.attention_mask.sum(dim=-1).long()
            feature_lens_list = feature_lens.tolist()
            # Pack padded (N, C, T_max) → flat (C, T_total) as the audio encoders expect
            packed_features = self._pack_from_padded(out.input_features, feature_lens)
            audio_lengths = iter(
                self.audio_token_length_fn(int(length)) for length in feature_lens_list
            )
            audio_inputs = {
                "input_features": packed_features,
                "feature_lens": feature_lens,
            }
        elif audio_feature is not None:
            features, feature_lens = audio_feature
            if torch.is_tensor(feature_lens):
                feature_lens = feature_lens.to(dtype=torch.long)
                feature_lens_list = feature_lens.tolist()
            else:
                feature_lens_list = [int(x) for x in feature_lens]
                feature_lens = torch.tensor(feature_lens_list, dtype=torch.long)

            packed_features = self._pack_from_padded(features, feature_lens)
            audio_lengths = iter(
                self.audio_token_length_fn(int(length)) for length in feature_lens_list
            )
            audio_inputs = {
                "input_features": packed_features,
                "feature_lens": feature_lens,
            }
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(text, audio_lengths)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if prepare_labels:
            text_inputs["labels"] = self._prepare_labels(
                text_inputs["input_ids"], text_inputs["attention_mask"]
            )

        return BatchFeature(
            data={**text_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def _prepare_labels(self, input_ids, attention_mask):
        labels = torch.full_like(input_ids, -100)

        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_header = self.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        header_len = len(assistant_header)
        header_tensor = torch.tensor(
            assistant_header, dtype=input_ids.dtype, device=input_ids.device
        )

        B, L = input_ids.shape

        for b in range(B):
            ids = input_ids[b]  # [L]

            # 1. Find positions where <|im_start|> is followed by "assistant\n".
            #    unfold(0, header_len, 1) on ids[1:] gives windows where
            #    windows[i] == ids[i+1 : i+1+header_len].
            im_start_mask = ids == im_start_id  # [L]
            if header_len > 0 and L > header_len:
                windows = ids[1:].unfold(0, header_len, 1)  # [L-header_len, header_len]
                header_match = (windows == header_tensor).all(dim=-1)  # [L-header_len]
                header_match_full = torch.cat(
                    [header_match, header_match.new_zeros(header_len)]
                )  # [L]
            else:
                header_match_full = ids.new_zeros(L, dtype=torch.bool)

            assistant_starts = (im_start_mask & header_match_full).nonzero(
                as_tuple=True
            )[0]  # positions of matching <|im_start|>
            if len(assistant_starts) == 0:
                continue

            # response content starts right after "<|im_start|>assistant\n"
            response_starts = (assistant_starts + 1 + header_len).clamp(max=L)

            # 2. For each response_start, find the first <|im_end|> at or after it.
            #    searchsorted on the sorted im_end_pos tensor replaces list.index().
            im_end_pos = (ids == im_end_id).nonzero(as_tuple=True)[0]  # [M], sorted
            if len(im_end_pos) > 0:
                idx = torch.searchsorted(im_end_pos, response_starts)  # [N]
                has_end = idx < len(im_end_pos)
                safe_idx = idx.clamp(max=len(im_end_pos) - 1)
                # +1 to include the <|im_end|> token itself (matches original logic)
                response_ends = torch.where(
                    has_end,
                    im_end_pos[safe_idx] + 1,
                    ids.new_full((), L),
                ).clamp(max=L)
            else:
                response_ends = ids.new_full((len(response_starts),), L)

            # 3. Fill [response_start, response_end) ranges using the cumsum trick:
            #    +1 at each start, -1 at each end → cumsum > 0 marks label positions.
            signal = ids.new_zeros(L + 1)
            signal.scatter_add_(
                0, response_starts, torch.ones_like(response_starts)
            )
            signal.scatter_add_(
                0, response_ends, -torch.ones_like(response_ends)
            )
            label_mask = signal[:L].cumsum(0).bool()  # [L]

            labels[b] = ids.masked_fill(~label_mask, -100)

        return labels

    @staticmethod
    def _pack_from_padded(
        features: torch.Tensor, feature_lens: torch.Tensor
    ) -> torch.Tensor:
        """Pack padded (N, C, T_max) into flat (C, T_total) expected by audio encoders."""
        # features: (N, C, T_max) — HF feature extractor output layout
        chunks = [
            features[i, :, : int(feature_lens[i].item())]
            for i in range(features.size(0))
        ]
        return torch.cat(chunks, dim=1).contiguous()  # (C, T_total)

    def replace_multimodal_special_tokens(
        self, text: list[str], audio_lengths
    ) -> list[str]:
        processed = []
        for sample in text:
            if self.audio_token not in sample:
                processed.append(sample)
                continue

            parts = sample.split(self.audio_token)
            rebuilt = [parts[0]]
            for i in range(1, len(parts)):
                n = max(int(next(audio_lengths, 1)), 1)
                rebuilt.append(self.audio_token * n)
                rebuilt.append(parts[i])
            processed.append("".join(rebuilt))
        return processed

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.feature_extractor.model_input_names
                + ["feature_lens"]
            )
        )
