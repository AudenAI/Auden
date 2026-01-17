import logging
import os

import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperModel

from auden.auto.auto_config import AutoConfig

from .features import _construct_feature_extractor, _extract_features
from .utils import replace_whisper_encoder_forward


class WhisperEncoderModel(nn.Module):
    """
    Thin wrapper around Hugging Face WhisperEncoder

    - forward(x, x_lens, return_dict=True) returns either:
        * dict with keys {'encoder_out', 'encoder_out_lens'}
        * or a tuple (encoder_out, encoder_out_lens) when return_dict=False
    - encoder_out_dim: hidden size of the encoder.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        map_location: str | torch.device = "cpu",
        dtype=None,
        device=None,
    ) -> "WhisperEncoderModel":
        """Load WhisperEncoder weights and wrap into WhisperEncoderModel.

        Supports loading from:
        - HuggingFace Whisper models (e.g., "openai/whisper-base")
        - Our saved WhisperEncoderModel models (with encoder weights)

        Args:
            model_path: HF repo id or local directory containing Whisper weights/config
            map_location: Device to load weights to
            dtype: Optional dtype for the model
            device: Optional device to move model to

        Returns:
            WhisperEncoderModel instance
        """
        import json

        # Detect if it's our format or HF format
        is_our_format = False
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    is_our_format = config_dict.get("model_type") == "whisper-encoder"

        if is_our_format:
            # Load from our saved format
            from .model_config import WhisperEncoderConfig

            config = WhisperEncoderConfig.from_pretrained(model_path)
            model = cls(config)

            # Load weights
            weight_path = None
            for ext in (".safetensors", ".pt"):
                for name in ("encoder", "model", "pretrained"):
                    p = os.path.join(model_path, f"{name}{ext}")
                    if os.path.exists(p):
                        weight_path = p
                        break
                if weight_path:
                    break

            if weight_path:
                ext = os.path.splitext(weight_path)[1].lower()
                if ext == ".safetensors":
                    from safetensors.torch import load_file as safe_load_file

                    device_arg = (
                        str(map_location)
                        if isinstance(map_location, torch.device)
                        else map_location
                    )
                    state_dict = safe_load_file(weight_path, device=device_arg)
                else:
                    state_dict = torch.load(weight_path, map_location=map_location)

                # Handle state_dict wrapper
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                model.load_state_dict(state_dict, strict=True)
        else:
            # Load from HuggingFace Whisper format
            from .model_config import WhisperEncoderConfig

            # Load config (will handle HF format)
            config = WhisperEncoderConfig.from_pretrained(model_path)

            # Load HF WhisperEncoder weights
            hf_encoder = WhisperModel.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=False
            ).encoder
            if map_location is not None:
                hf_encoder.to(map_location)

            model = cls(config)
            model.encoder.load_state_dict(hf_encoder.state_dict(), strict=True)

        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = None
        # Convert our encoder config to full HF config for WhisperEncoder
        hf_config = config.to_hf_config()
        self.encoder = WhisperEncoder(hf_config)
        self.num_mel_bins = getattr(config, "num_mel_bins", 80)
        replace_whisper_encoder_forward()
        self.encoder_out_dim = config.d_model

    def forward(
        self, x: Tensor, x_lens: Tensor, return_dict: bool = True
    ) -> dict | tuple[Tensor, Tensor]:
        """
        Args:
            x: (N, T, C) log-Mel features for Whisper (already precomputed).
            x_lens: (N,) valid frame counts per sample.
        """
        # WhisperEncoder expects (batch, feature_dim, seq_len)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        encoder_out = self.encoder(x)[0]
        encoder_out_lens = (x_lens - 1) // 2 + 1

        if return_dict:
            return {
                "encoder_out": encoder_out,
                "encoder_out_lens": encoder_out_lens,
            }
        else:
            return encoder_out, encoder_out_lens

    def save_pretrained(
        self,
        save_directory: str,
        *,
        filename: str | None = None,
        use_safetensors: bool = True,
    ) -> str:
        """Save model weights and config to a directory (HuggingFace style).

        Writes config via config.save_pretrained(save_directory) and weights to
        encoder.safetensors by default (falls back to encoder.pt if safetensors
        is unavailable or use_safetensors=False).

        Args:
            save_directory: Target directory.
            filename: Optional explicit filename for weights. If None, chooses
                encoder.safetensors or encoder.pt.
            use_safetensors: Prefer safetensors format when possible.

        Returns:
            Path of the saved weight file.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration next to weights
        self.config.save_pretrained(save_directory)

        # Decide filename and format
        weight_path: str
        chosen_filename = filename
        if chosen_filename is None:
            if use_safetensors:
                try:
                    from safetensors.torch import (  # noqa: F401
                        save_file as safe_save_file,
                    )

                    chosen_filename = "encoder.safetensors"
                except Exception:
                    chosen_filename = "encoder.pt"
            else:
                chosen_filename = "encoder.pt"

        weight_path = os.path.join(save_directory, chosen_filename)

        # Always save CPU state_dict for portability
        state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        ext = os.path.splitext(weight_path)[1].lower()
        if ext == ".safetensors":
            try:
                from safetensors.torch import save_file as safe_save_file

                safe_save_file(state_dict, weight_path)
            except Exception as e:
                # Fallback to .pt if safetensors is unavailable
                logging.warning(
                    f"Failed to save safetensors ({e}); falling back to PyTorch .pt"
                )
                weight_path = os.path.splitext(weight_path)[0] + ".pt"
                torch.save(state_dict, weight_path)
        else:
            torch.save(state_dict, weight_path)

        logging.info(f"Saved model to {weight_path}")
        return weight_path

    def extract_feature(self, input):
        """Thin wrapper to extract features via the helper utility.

        Accepts the same input forms as before and returns (features, feature_lens).
        """
        if not self.feature_extractor:
            self.feature_extractor = self.construct_feature_extractor(
                num_mel_bins=self.num_mel_bins
            )
        device = next(self.encoder.parameters()).device
        features, feature_lens = _extract_features(
            input, self.feature_extractor, target_sample_rate=16000, device=device
        )
        return features, feature_lens

    def construct_feature_extractor(
        self,
        *,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
    ):
        """Construct a default FBANK feature extractor"""
        return _construct_feature_extractor(
            sample_rate=sample_rate, num_mel_bins=num_mel_bins
        )
