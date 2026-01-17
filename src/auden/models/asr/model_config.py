"""
ASR model configuration.

This config describes a simple transducer/CTC ASR stack with a pluggable
encoder configuration (e.g., Zipformer, WenetTransformer, etc.).
It follows the HF-style explicit __init__ defaults and stores arbitrary
extra kwargs via BaseConfig.
"""

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig
from ..zipformer.model_config import ZipformerConfig


class AsrConfig(BaseConfig):
    """Configuration for ASR models with a pluggable encoder.

    Args:
        encoder_config: Encoder configuration instance, a dict, or None.
            - If None, defaults to ZipformerConfig().
            - If a dict, must contain a "model_type" key (e.g., "zipformer",
              "wenet-transformer"). The config will be automatically created
              via AutoConfig.
            - If a config instance, used as-is.
        decoder_dim: Decoder hidden dimension.
        context_size: Decoder context size (e.g., for transducer joiner context).
        joiner_dim: Joiner hidden dimension.
        use_transducer: Whether to enable transducer criterion.
        use_ctc: Whether to enable CTC criterion.
        **kwargs: Extra fields stored for forward compatibility.

    Examples:
        >>> # Use default Zipformer encoder
        >>> config = AsrConfig()

        >>> # Use WenetTransformer encoder
        >>> config = AsrConfig(encoder_config={"model_type": "wenet-transformer", "output_size": 512})

        >>> # Use explicit config instance
        >>> from auden.models.wenet_transformer import WenetTransformerConfig
        >>> enc_cfg = WenetTransformerConfig(output_size=512)
        >>> config = AsrConfig(encoder_config=enc_cfg)
    """

    model_type = "asr"

    def __init__(
        self,
        encoder_config=None,
        decoder_dim: int = 512,
        context_size: int = 2,
        joiner_dim: int = 512,
        use_transducer: bool = True,
        use_ctc: bool = False,
        **kwargs,
    ):
        # Handle encoder_config with flexibility
        if encoder_config is None:
            # Default to Zipformer for backward compatibility
            enc_cfg = ZipformerConfig()
        elif isinstance(encoder_config, dict):
            # Use AutoConfig to support any encoder type
            enc_type = encoder_config.get("model_type", "zipformer")
            try:
                enc_cfg = AutoConfig.for_model(**encoder_config)
            except Exception as e:
                raise ValueError(
                    f"Failed to create encoder config for model_type='{enc_type}'. "
                    f"Supported types: zipformer, wenet-transformer, etc. "
                    f"Error: {e}"
                )
        else:
            # Assume it's already a config instance
            enc_cfg = encoder_config

        self.encoder_config = enc_cfg
        self.decoder_dim = decoder_dim
        self.context_size = context_size
        self.joiner_dim = joiner_dim
        self.use_transducer = use_transducer
        self.use_ctc = use_ctc

        super().__init__(**kwargs)
