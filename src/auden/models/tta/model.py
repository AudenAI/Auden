from __future__ import annotations

import math
import os
from pathlib import Path
from re import L
from typing import List, Tuple

try:
    import k2
except Exception as e:
    raise ImportError(
        "k2 is required for RNNT/Transducer components in TTA. "
        "Install via conda: `conda install -c k2-fsa k2` (ensure PyTorch/CUDA match), "
        "or see https://k2-fsa.github.io/k2/."
    ) from e
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel as HFModel
from transformers import AutoTokenizer as HFTokenizer

from ...auto.auto_config import AutoConfig
from ...auto.auto_model import AutoModel
from ...auto.auto_tokenizer import AutoTokenizer
from ..asr.utils import add_sos
from ..zipformer.utils.padding import make_pad_mask
from ..zipformer.utils.scaling import ScaledLinear
from .asr_decoder.attention_decoder import AttentionDecoderModel
from .asr_decoder.decoder import Decoder
from .asr_decoder.joiner import Joiner


class TtaModel(nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> "TtaModel":
        # Resolve model_dir and checkpoint
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)
        if os.path.isdir(model_path):
            model_dir = model_path
            weight_path = None
            for ext in (".safetensors", ".pt"):
                for name in ("pretrained", "model"):
                    p = os.path.join(model_dir, f"{name}{ext}")
                    if os.path.exists(p):
                        weight_path = p
                        break
                if weight_path is not None:
                    break
            if weight_path is None:
                raise FileNotFoundError(
                    f"Expected one of ['pretrained.safetensors','model.safetensors','pretrained.pt','model.pt'] under {model_dir}"
                )
        else:
            weight_path = model_path
            model_dir, _ = os.path.split(model_path)

        # Load config and tokenizers
        config = AutoConfig.from_pretrained(model_dir)
        asr_tok_dir = Path(model_dir) / "asr_tokenizer"
        asr_tokenizer = AutoTokenizer.from_pretrained(asr_tok_dir)

        text_tok_dir = Path(model_dir) / "text_tokenizer"
        text_tokenizer = (
            HFTokenizer.from_pretrained(text_tok_dir) if text_tok_dir.exists() else None
        )

        model = cls(config, asr_tokenizer, text_tokenizer=text_tokenizer)

        # Load weights (pt or safetensors)
        ext = os.path.splitext(weight_path)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file as safe_load_file

            device_arg = (
                str(map_location)
                if isinstance(map_location, torch.device)
                else map_location
            )
            state_obj = safe_load_file(weight_path, device=device_arg)
        else:
            state_obj = torch.load(weight_path, map_location=map_location)
        state_dict = (
            state_obj["state_dict"]
            if isinstance(state_obj, dict) and "state_dict" in state_obj
            else state_obj
        )

        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model

    def __init__(self, config, asr_tokenizer, text_tokenizer=None):
        super().__init__()
        self.config = config
        self.asr_tokenizer = asr_tokenizer
        self.blank_id = self.asr_tokenizer.blank_id
        self.vocab_size = self.asr_tokenizer.vocab_size
        self.speech_encoder_config = config.speech_encoder_config
        self.speech_encoder_out_dim = max(config.speech_encoder_config.encoder_dim)
        self.speech_encoder = AutoModel.from_config(self.config.speech_encoder_config)

        self.special_tokens = config.special_tokens
        if self.special_tokens is not None:
            self.special_to_id = {
                spec: self.vocab_size + i for i, spec in enumerate(self.special_tokens)
            }
            self.id_to_special = {
                self.vocab_size + i: spec for i, spec in enumerate(self.special_tokens)
            }
            self.attn_vocab_size = self.vocab_size + len(self.special_tokens)
        else:
            self.attn_vocab_size = self.vocab_size

        # RNNT branch
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            decoder_dim=self.config.decoder_dim,
            blank_id=self.blank_id,
            context_size=self.config.context_size,
        )

        self.joiner = Joiner(
            encoder_dim=self.speech_encoder_out_dim,
            decoder_dim=self.config.decoder_dim,
            joiner_dim=self.config.joiner_dim,
            vocab_size=self.vocab_size,
        )

        self.simple_am_proj = ScaledLinear(
            self.speech_encoder_out_dim, self.vocab_size, initial_scale=0.25
        )
        self.simple_lm_proj = ScaledLinear(
            self.config.decoder_dim, self.vocab_size, initial_scale=0.25
        )

        # Attention decoder branch
        self.attention_decoder = AttentionDecoderModel(
            vocab_size=self.attn_vocab_size,
            decoder_dim=self.config.decoder_dim,
            num_decoder_layers=self.config.num_decoder_layers,
            attention_dim=self.config.decoder_dim,
            num_heads=self.config.num_heads,
            feedforward_dim=4 * self.config.decoder_dim,
            memory_dim=self.speech_encoder_out_dim,
            sos_id=self.asr_tokenizer.bos_token_id,
            eos_id=self.asr_tokenizer.eos_token_id,
            ignore_id=self.config.ignore_id,
            label_smoothing=self.config.label_smoothing,
        )

        # s2t alignment branch
        self.text_tokenizer = text_tokenizer
        self.text_encoder = HFModel.from_config(
            self.config.text_encoder_config, torch_dtype=torch.float16
        )
        self.text_encoder_dim = self.config.text_encoder_config.hidden_size
        self.align_dim = self.text_encoder_dim
        self.encoder_align2text_proj = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.speech_encoder_out_dim, self.align_dim)
        )
        self.s2t_align_temp = nn.Parameter(torch.ones([]) * math.log(10))
        self.s2t_align_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.amp.autocast('cuda', enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.amp.autocast('cuda', enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
            )
        return simple_loss, pruned_loss

    def calc_s2t_align_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        texts: List[str],
    ) -> torch.Tensor:
        # Use unified embedders
        speech_embed = self._compute_audio_embedding(encoder_out, encoder_out_lens)
        text_embed = self._compute_text_embedding(texts)

        speech_text_logits = (
            speech_embed @ text_embed.T
        ) * self.s2t_align_temp + self.s2t_align_bias
        # SigLIP loss
        bsz = speech_text_logits.size(0)
        labels = 2 * torch.eye(bsz, device=speech_text_logits.device) - 1
        return -F.logsigmoid(labels * speech_text_logits).mean()

    def forward_attention_decoder_language_token(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        topk: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k token ids and logits for first decoder step.

        Returns (indices, logits) both shaped (B, K).
        """
        bs = encoder_out.size(0)
        prefix_tokens = (
            torch.tensor(
                [self.attention_decoder.sos_id],
                dtype=torch.long,
                device=encoder_out.device,
            )
            .repeat(bs)
            .unsqueeze(1)
        )
        dec_out = self.attention_decoder.forward_one_step(
            encoder_out, encoder_out_lens, prefix_tokens
        )
        logits = dec_out[:, -1, :]  # (B, V)
        topk_logp, topk_idx = torch.topk(logits, k=topk, dim=-1)
        return topk_idx, topk_logp

    def decorate_decoder_input(
        self,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        source_language: List[str],
        target_language: List[str],
    ):
        batch_size = y.dim0
        assert len(source_language) == batch_size
        assert len(target_language) == batch_size

        src_tags = [self.special_to_id[f"<{l}>"] for l in source_language]
        tgt_tags = [self.special_to_id[f"<{l}>"] for l in target_language]
        insert_tokens = torch.tensor(
            list(zip(src_tags, tgt_tags)), dtype=y.dtype, device=y.device
        )
        y_aed = k2.ragged.cat([k2.RaggedTensor(insert_tokens), y], axis=1)
        y_lens = y_lens + 2

        ignore_indices = [
            [1] for _ in range(batch_size)
        ]  # pos 1, <tgt_lang> will be ignored
        return y_aed, y_lens, ignore_indices

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        source_texts: List[str],
        target_texts: List[str],
        source_language: List[str] | None = None,
        target_language: List[str] | None = None,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        forward_attention_decoder: bool = True,
        forward_s2t_alignment: bool = True,
    ) -> Tuple[torch.Tensor | None, ...]:
        """Training forward supporting RNNT, attention decoder, and s2t alignment.

        - RNNT (ASR): provide `source_texts`.
        - AED/AST (translation): provide `target_texts` and both `source_language`/`target_language`.
        - s2t alignment: enabled when `use_s2t_alignment` and `target_texts` are given.
        """
        device = x.device

        assert x.ndim == 3
        assert x_lens.ndim == 1

        # Encoder forward
        encoder_output = self.speech_encoder(x, x_lens)
        encoder_out = encoder_output.encoder_out
        encoder_out_lens = encoder_output.encoder_out_lens

        # RNNT branch (ASR)
        y_list = self.asr_tokenizer.encode(source_texts)
        y = k2.RaggedTensor(y_list).to(device)
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        simple_loss, pruned_loss = self.forward_transducer(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            y=y.to(device),
            y_lens=y_lens,
            prune_range=prune_range,
            am_scale=am_scale,
            lm_scale=lm_scale,
        )

        # Attention decoder branch (ASR/AST)
        if forward_attention_decoder:
            y_target_list = self.asr_tokenizer.encode(target_texts)
            y_target = k2.RaggedTensor(y_target_list).to(device)
            row_splits = y_target.shape.row_splits(1)
            y_target_lens = row_splits[1:] - row_splits[:-1]

            y_aed, y_aed_lens, prefix_ignore_indices = self.decorate_decoder_input(
                y=y_target,
                y_lens=y_target_lens,
                source_language=source_language,
                target_language=target_language,
            )
            attention_decoder_loss, _ = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y_aed.to(device),
                ys_lens=y_aed_lens.to(device),
                prefix_ignore_indices=prefix_ignore_indices,
            )
        else:
            attention_decoder_loss = None

        # s2t alignment branch
        if forward_s2t_alignment:
            s2t_align_loss = self.calc_s2t_align_loss(
                encoder_out, encoder_out_lens, target_texts
            )
        else:
            s2t_align_loss = None

        return (
            simple_loss,
            pruned_loss,
            attention_decoder_loss,
            s2t_align_loss,
        )

    def _compute_audio_embedding(
        self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor
    ) -> torch.Tensor:
        """Compute a single pooled audio embedding per sample, L2-normalized.

        Uses the alignment projection if available; otherwise pools encoder_out.
        """
        if hasattr(self, "encoder_align2text_proj"):
            speech_embed = self.encoder_align2text_proj(encoder_out)
        else:
            speech_embed = encoder_out
        speech_mask = ~make_pad_mask(encoder_out_lens, encoder_out.size(1)).to(
            encoder_out.device
        )
        speech_embed = (speech_embed * speech_mask.unsqueeze(-1)).sum(dim=1)
        speech_embed = speech_embed / encoder_out_lens.reshape(-1, 1)
        speech_embed = F.normalize(speech_embed, p=2, dim=-1)
        return speech_embed

    def _compute_text_embedding(self, texts: List[str]) -> torch.Tensor:
        """Compute one embedding per text via the HF text encoder, L2-normalized."""
        device = next(self.parameters()).device
        text_inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True).to(
            device
        )
        text_inputs.pop("encoder_attention_mask", None)
        text_embed = self.text_encoder(**text_inputs)["last_hidden_state"]
        text_embed = torch.sum(
            text_embed * (text_inputs["attention_mask"].unsqueeze(-1)), dim=1
        )
        text_embed = text_embed / text_inputs["attention_mask"].sum(-1).unsqueeze(-1)
        text_embed = text_embed.float()
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        return text_embed

    def generate(
        self,
        input,
        *,
        task: str,
        beam_size: int = 1,
        blank_penalty: float = 0,
        source_language: List[str] | None = None,
        target_language: List[str] | None = None,
        return_timestamps: bool = False,
        texts: List[str] | None = None,
    ) -> dict:
        """Unified generate interface for three use cases.

        Args:
            task: One of {"transcribe", "translate", "align"}.
            beam_size: Beam size for attention decoder.
            blank_penalty: Blank penalty for RNNT greedy decoding.
            source_language/target_language: List of language tags per utterance for translation
                (length must equal batch size). If None, source/target languages will be
                predicted/defaulted inside the decoder.
            return_timestamps: Return timestamps for RNNT greedy decoding.
            texts: Text list for align; required when task=="align".

        Returns:
            dict containing task-specific outputs:
            - transcribe: {"hypotheses": List[str], "timestamps": Optional[List[List[int]]]}
            - translate: {"hypotheses": List[str]}
            - align: {"similarities": Tensor[N, M], "audio_emb": Tensor[N, D], "text_emb": Tensor[M, D]}
        """
        # Handle flexible input
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        else:
            x, x_lens = self.speech_encoder.extract_feature(input)
        output = self.speech_encoder(x, x_lens)

        if task == "transcribe":
            # RNNT greedy search
            from .decode import greedy_search_batch

            decoding_results = greedy_search_batch(
                model=self,
                encoder_out=output.encoder_out,
                encoder_out_lens=output.encoder_out_lens,
                blank_penalty=blank_penalty,
                return_timestamps=return_timestamps,
            )
            hyp_tokens = decoding_results.hyps
            hyps = self.asr_tokenizer.decode(hyp_tokens)
            resp: dict = {"hypotheses": hyps}
            if return_timestamps:
                resp["timestamps"] = decoding_results.timestamps
            return resp

        if task == "translate":
            # Attention decoder beam search
            batch_size = output.encoder_out.size(0)

            def _lang_tags_to_ids(tags: List[str] | None) -> List[int] | None:
                if tags is None:
                    return None
                if len(tags) != batch_size:
                    raise ValueError(
                        f"language list length {len(tags)} != batch size {batch_size}"
                    )
                ids: List[int] = []
                for t in tags:
                    key = f"<{t}>"
                    if key not in self.special_to_id.keys():
                        raise ValueError(
                            f"Unknown language tag: {t}. Supported tags: {self.special_tokens}"
                        )
                    ids.append(self.special_to_id[key])
                return ids

            src_lang_ids = _lang_tags_to_ids(source_language)
            tgt_lang_ids = _lang_tags_to_ids(target_language)
            from .decode import attention_beam_search

            decoding_results = attention_beam_search(
                model=self,
                encoder_out=output.encoder_out,
                encoder_out_lens=output.encoder_out_lens,
                source_language=src_lang_ids,
                target_language=tgt_lang_ids,
                beam_size=beam_size,
            )
            hyp_tokens = decoding_results.hyps
            hyps = self.asr_tokenizer.decode(hyp_tokens, skip_bos_eos=True)
            source_language = [
                self.id_to_special[id] for id in decoding_results.source_language
            ]
            target_language = [
                self.id_to_special[id] for id in decoding_results.target_language
            ]
            return {
                "hypotheses": hyps,
                "source_language": source_language,
                "target_language": target_language,
            }

        if task == "align":
            assert (
                texts is not None and len(texts) > 0
            ), "texts must be provided for align."
            # Compute pooled audio emb and text emb
            audio_emb = self._compute_audio_embedding(
                output.encoder_out, output.encoder_out_lens
            )
            text_emb = self._compute_text_embedding(texts)
            # cosine similarity as dot product because both are normalized
            sims = audio_emb @ text_emb.T  # (N, M)
            return {"similarities": sims, "audio_emb": audio_emb, "text_emb": text_emb}

        raise ValueError(f"Unsupported task: {task}")
