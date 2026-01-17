import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import k2
except Exception as e:
    raise ImportError(
        "k2 is required for RNNT/Transducer training and decoding. "
        "Install via conda: `conda install -c k2-fsa k2` (ensure PyTorch/CUDA match), "
        "or see https://k2-fsa.github.io/k2/ for install options."
    ) from e
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from ...auto.auto_config import AutoConfig
from ...auto.auto_model import AutoModel
from ..zipformer.utils.scaling import ScaledLinear
from .asr_decoder.decoder import Decoder
from .asr_decoder.joiner import Joiner
from .decode import greedy_search_batch
from .utils import add_sos, remove_whitespace_marker


class AsrModel(nn.Module):
    """ASR model with a pluggable encoder and optional Transducer/CTC heads.

    Typical usage:
        - Build config/tokenizer, then initialize: ``model = AsrModel(config, tokenizer)``
        - Optionally load a pretrained encoder externally in your script (recommended),
          or via the minimal convenience ``load_pretrained_modules({"encoder": src})`` if present.
        - Use ``forward`` for training (returns losses); use ``generate`` for decoding.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        # Resolve model_dir and checkpoint (supports HF Hub repo IDs)
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

        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(Path(model_dir))
        model = cls(config, tokenizer)

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

    def __init__(self, config, tokenizer, pretrained_modules=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.blank_id = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
        logging.info(f"Vocab size: {self.vocab_size}")
        self.encoder = AutoModel.from_config(self.config.encoder_config)
        self.encoder_out_dim = self.encoder.encoder_out_dim
        # Initialize decoder
        if config.use_transducer:
            self.decoder = Decoder(
                vocab_size=self.vocab_size,
                decoder_dim=config.decoder_dim,
                blank_id=self.blank_id,
                context_size=config.context_size,
            )

            self.joiner = Joiner(
                encoder_dim=self.encoder_out_dim,
                decoder_dim=config.decoder_dim,
                joiner_dim=config.joiner_dim,
                vocab_size=self.vocab_size,
            )

            self.simple_am_proj = ScaledLinear(
                self.encoder_out_dim, self.vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                config.decoder_dim, self.vocab_size, initial_scale=0.25
            )

        else:
            self.decoder = None
            self.joiner = None

        if config.use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(self.encoder_out_dim, self.vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            blank=self.blank_id,
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

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

        with torch.amp.autocast("cuda", enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
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

        with torch.amp.autocast("cuda", enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )
        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        texts: List[str],
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          texts:
            A list of text of shape (N,)
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        y_list = self.tokenizer(texts)["input_ids"]
        y_list = remove_whitespace_marker(y_list, self.tokenizer)
        device = x.device
        y = k2.RaggedTensor(y_list).to(device)
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_output = self.encoder(x, x_lens)
        encoder_out = encoder_output["encoder_out"]
        encoder_out_lens = encoder_output["encoder_out_lens"]

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.config.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = None
            pruned_loss = None

        if self.config.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = None

        if return_dict:
            return {
                "simple_loss": simple_loss,
                "pruned_loss": pruned_loss,
                "ctc_loss": ctc_loss,
            }
        else:
            return (simple_loss, pruned_loss, ctc_loss)

    def generate(self, input, decoding_method="greedy_search", blank_penalty=0):
        # Handle flexible input
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        else:
            x, x_lens = self.encoder.extract_feature(input)

        encoder_output = self.encoder(x, x_lens)
        encoder_out = encoder_output["encoder_out"]
        encoder_out_lens = encoder_output["encoder_out_lens"]
        if decoding_method == "greedy_search":
            hyp_tokens = greedy_search_batch(
                model=self,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                blank_penalty=blank_penalty,
            )
        else:
            raise NotImplementedError(
                f"Decoding method '{decoding_method}' is not supported"
            )
        hyps = self.tokenizer.batch_decode(hyp_tokens, skip_special_tokens=True)
        return hyps
