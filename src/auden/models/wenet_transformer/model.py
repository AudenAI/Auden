import logging
import os

import torch
import torch.nn as nn
from torch import Tensor

from ...auto.auto_config import AutoConfig
from ...auto.auto_model import AutoModel

try:
    from wenet.models.transformer.encoder import ConformerEncoder, TransformerEncoder
except ImportError as e:
    raise ImportError(
        "WenetTransformerEncoderModel requires the 'wenet' package. "
        "Please install it or add the wenet path to sys.path. "
        f"Original error: {e}"
    )


class WenetTransformerEncoderModel(nn.Module):
    """
    Wrapper for WeNet Transformer/Conformer encoder models.

    Provides a unified interface for WeNet-style encoders (Transformer or Conformer).

    - forward(x, x_lens, return_dict=True) returns:
        * dict with keys {'encoder_out', 'encoder_out_lens'} when return_dict=True
        * tuple (encoder_out, encoder_out_lens) when return_dict=False
    - encoder_out_dim: hidden dimension of the encoder
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        module_name: str | None = None,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
        dtype=None,
        device=None,
    ) -> "WenetTransformerEncoderModel":
        """Load WeNet encoder from a pretrained checkpoint.

        Args:
            model_path: Directory containing weights and config, or direct path to .pt/.safetensors file.
            module_name: If loading from a composite model, specify the submodule name containing
                encoder weights (e.g., "audio_encoder" or "encoder"). If None, will be auto-detected
                based on config.
            strict: Passed to load_state_dict.
            map_location: Passed to torch.load.
            dtype: Optional dtype to move the model to after loading.
            device: Optional device to move the model to after loading.
        """
        # Support HuggingFace Hub repo IDs
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)

        # Resolve model_dir and weight file
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
                    f"Expected one of ['pretrained.safetensors','model.safetensors',"
                    f"'pretrained.pt','model.pt'] under {model_dir}"
                )
        else:
            weight_path = model_path
            model_dir, _ = os.path.split(model_path)

        config = AutoConfig.from_pretrained(model_dir)

        # Load weights
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

        if isinstance(state_obj, dict) and "state_dict" in state_obj:
            state_dict = state_obj["state_dict"]
        else:
            state_dict = state_obj

        logging.info(f"Loaded weights for type {config.model_type} from {weight_path}")

        # Determine sub-config: allow both composite configs and direct encoder configs
        detected_module = None
        if module_name is not None:
            detected_module = module_name
        elif hasattr(config, "audio_encoder_config"):
            detected_module = "audio_encoder"
        elif hasattr(config, "speech_encoder_config"):
            detected_module = "speech_encoder"
        elif hasattr(config, "encoder_config"):
            detected_module = "encoder"

        if detected_module is not None:
            sub_config = getattr(config, f"{detected_module}_config", None)
            if sub_config is None:
                raise ValueError(
                    f"Config does not contain '{detected_module}_config' needed for encoder."
                )
        else:
            # Direct encoder config saved at model_dir
            sub_config = config

        model = cls(sub_config)

        # Strip possible prefixes: DDP 'module.' and parent module prefix
        def _strip_prefix(d: dict, prefix: str):
            if any(k.startswith(prefix) for k in d.keys()):
                return {
                    k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)
                }
            return None

        candidate = state_dict
        # Strip DDP prefix at most once
        stripped = _strip_prefix(candidate, "module.")
        if stripped:
            candidate = stripped

        # For composite parent models, strip one parent-level prefix; skip for direct encoders
        if detected_module is not None:
            for pfx in [
                f"{detected_module}.",
                "audio_encoder.",
                "speech_encoder.",
                "encoder.",
            ]:
                stripped = _strip_prefix(candidate, pfx)
                if stripped:
                    candidate = stripped
                    break
        state_dict = candidate

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if not strict and (missing or unexpected):
            logging.warning(
                f"from_pretrained loaded with missing keys: {missing}, "
                f"unexpected keys: {unexpected}"
            )

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

        # Select encoder type based on encoder_type in config
        encoder_type = config.encoder_type

        # Prepare common parameters
        common_params = {
            "input_size": config.input_size,
            "output_size": config.output_size,
            "attention_heads": config.attention_heads,
            "linear_units": config.linear_units,
            "num_blocks": config.num_blocks,
            "dropout_rate": config.dropout_rate,
            "positional_dropout_rate": config.positional_dropout_rate,
            "attention_dropout_rate": config.attention_dropout_rate,
            "input_layer": config.input_layer,
            "pos_enc_layer_type": config.pos_enc_layer_type,
            "normalize_before": config.normalize_before,
            "static_chunk_size": config.static_chunk_size,
            "use_dynamic_chunk": config.use_dynamic_chunk,
            "global_cmvn": config.global_cmvn,
            "use_dynamic_left_chunk": config.use_dynamic_left_chunk,
            "query_bias": config.query_bias,
            "key_bias": config.key_bias,
            "value_bias": config.value_bias,
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_sdpa": config.use_sdpa,
            "layer_norm_type": config.layer_norm_type,
            "norm_eps": config.norm_eps,
            "n_kv_head": config.n_kv_head,
            "head_dim": config.head_dim,
            "mlp_type": config.mlp_type,
            "mlp_bias": config.mlp_bias,
            "n_expert": config.n_expert,
            "n_expert_activated": config.n_expert_activated,
        }

        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                **common_params,
                positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
                macaron_style=config.macaron_style,
                selfattention_layer_type=config.selfattention_layer_type,
                activation_type=config.activation_type,
                use_cnn_module=config.use_cnn_module,
                cnn_module_kernel=config.cnn_module_kernel,
                causal=config.causal,
                cnn_module_norm=config.cnn_module_norm,
                conv_bias=config.conv_bias,
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                **common_params,
                activation_type=config.activation_type,
                selfattention_layer_type=config.selfattention_layer_type,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Expected 'conformer' or 'transformer'."
            )

        self.encoder_out_dim = self.encoder.output_size()

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        return_dict: bool = True,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> dict | tuple[Tensor, Tensor]:
        """
        Args:
            x: (N, T, C) feature tensor (e.g., log-Mel features or FBANK features).
            x_lens: (N,) number of valid frames per sample.
            return_dict: If True, return dict; otherwise return tuple.
            decoding_chunk_size: Decoding chunk size for dynamic chunking.
                0: Default for training, use random dynamic chunk.
                <0: For decoding, use full chunk.
                >0: For decoding, use fixed chunk size.
            num_decoding_left_chunks: Number of left chunks for decoding.
                >=0: Use num_decoding_left_chunks
                <0: Use all left chunks

        Returns:
            Based on return_dict value:
            - If True: {'encoder_out': Tensor, 'encoder_out_lens': Tensor}
            - If False: (encoder_out, encoder_out_lens)
        """
        # WeNet encoder forward method
        encoder_out, chunk_masks = self.encoder(
            xs=x,
            xs_lens=x_lens,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
        )

        # Compute output lengths from chunk_masks
        # chunk_masks shape is (B, T', T') or (B, 1, T')
        if chunk_masks.dim() == 3 and chunk_masks.size(1) == 1:
            # (B, 1, T') -> compute valid length for each sample
            encoder_out_lens = chunk_masks.squeeze(1).sum(dim=1).long()
        elif chunk_masks.dim() == 2:
            # (B, T')
            encoder_out_lens = chunk_masks.sum(dim=1).long()
        else:
            # If full attention mask (B, T', T'), compute from diagonal or first row
            encoder_out_lens = chunk_masks[:, 0, :].sum(dim=1).long()

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
        model.safetensors by default (falls back to model.pt if safetensors
        is unavailable or use_safetensors=False).

        Args:
            save_directory: Target directory.
            filename: Optional explicit filename for weights. If None, chooses
                model.safetensors or model.pt.
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

                    chosen_filename = "model.safetensors"
                except Exception:
                    chosen_filename = "model.pt"
            else:
                chosen_filename = "model.pt"

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
        """Thin wrapper to extract features.

        Note: WeNet models typically expect pre-computed FBANK features.
        This method needs to be implemented in subclasses or through external
        feature extractors.

        Args:
            input: Input audio data

        Returns:
            (features, feature_lens) tuple
        """
        if not self.feature_extractor:
            self.feature_extractor = self.construct_feature_extractor()

        # This needs to be completed based on actual feature extraction implementation
        raise NotImplementedError(
            "extract_feature needs to be implemented with actual feature extraction logic"
        )

    def construct_feature_extractor(
        self, *, sample_rate: int = 16000, num_mel_bins: int = 80
    ):
        """Construct a default feature extractor.

        Note: Needs to be implemented based on your actual feature extraction method.
        """
        raise NotImplementedError(
            "construct_feature_extractor needs to be implemented based on your feature extraction method"
        )
