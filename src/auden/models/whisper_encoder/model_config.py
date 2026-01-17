"""
Whisper encoder model configuration.

Attribution:
    This configuration focuses on encoder-relevant attributes from HuggingFace's WhisperConfig.
    Reference: https://github.com/openai/whisper
"""

from transformers import WhisperConfig as HFWhisperConfig

from ..base.model_config import BaseConfig


class WhisperEncoderConfig(BaseConfig):
    """Configuration for Whisper encoder models.

    Focuses on encoder-relevant attributes. Can be converted to a full
    HFWhisperConfig for initializing WhisperEncoder.

    Args:
        num_mel_bins (int, default=80): Number of mel filterbank features
        d_model (int, default=768): Encoder hidden size
        encoder_layers (int, default=12): Number of encoder layers
        encoder_attention_heads (int, default=12): Number of attention heads
        encoder_ffn_dim (int, default=3072): FFN dimension
        dropout (float, default=0.0): Dropout rate
        attention_dropout (float, default=0.0): Attention dropout
        activation_dropout (float, default=0.0): Activation dropout
        activation_function (str, default="gelu"): Activation function
        init_std (float, default=0.02): Initialization std
        encoder_layerdrop (float, default=0.0): Encoder layer drop rate
        scale_embedding (bool, default=False): Scale embeddings
        max_source_positions (int, default=1500): Max input positions
        apply_spec_augment (bool, default=False): Apply spec augment
        mask_time_prob (float, default=0.05): Time mask probability
        mask_time_length (int, default=10): Time mask length
        mask_time_min_masks (int, default=2): Min time masks
        mask_feature_prob (float, default=0.0): Feature mask probability
        mask_feature_length (int, default=10): Feature mask length
        mask_feature_min_masks (int, default=0): Min feature masks
        median_filter_width (int, default=7): Median filter width

    Example:
        >>> # Use default config
        >>> config = WhisperEncoderConfig()

        >>> # Custom config
        >>> config = WhisperEncoderConfig(
        ...     d_model=1280,
        ...     encoder_layers=32,
        ...     encoder_attention_heads=20,
        ...     encoder_ffn_dim=5120,
        ... )

        >>> # From HuggingFace pretrained
        >>> config = WhisperEncoderConfig.from_pretrained("openai/whisper-large-v3")
    """

    model_type: str = "whisper-encoder"

    def __init__(
        self,
        # Encoder-specific attributes
        num_mel_bins: int = 80,
        d_model: int = 768,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_function: str = "gelu",
        init_std: float = 0.02,
        encoder_layerdrop: float = 0.0,
        scale_embedding: bool = False,
        max_source_positions: int = 1500,
        apply_spec_augment: bool = False,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_time_min_masks: int = 2,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        mask_feature_min_masks: int = 0,
        median_filter_width: int = 7,
        **kwargs,
    ):
        """Initialize WhisperEncoderConfig with encoder-focused attributes."""
        super().__init__(**kwargs)

        # Store encoder-relevant attributes
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks
        self.median_filter_width = median_filter_width

    def to_hf_config(self) -> HFWhisperConfig:
        """Convert to HuggingFace WhisperConfig.

        Creates a full WhisperConfig with decoder defaults, suitable for
        initializing WhisperEncoder.

        Returns:
            HFWhisperConfig instance
        """
        return HFWhisperConfig(
            # Encoder attributes from our config
            num_mel_bins=self.num_mel_bins,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_ffn_dim=self.encoder_ffn_dim,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            activation_function=self.activation_function,
            init_std=self.init_std,
            encoder_layerdrop=self.encoder_layerdrop,
            scale_embedding=self.scale_embedding,
            max_source_positions=self.max_source_positions,
            apply_spec_augment=self.apply_spec_augment,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length,
            mask_time_min_masks=self.mask_time_min_masks,
            mask_feature_prob=self.mask_feature_prob,
            mask_feature_length=self.mask_feature_length,
            mask_feature_min_masks=self.mask_feature_min_masks,
            median_filter_width=self.median_filter_width,
            # Decoder attributes - use HF defaults (not used by encoder)
            vocab_size=51865,
            decoder_layers=12,
            decoder_attention_heads=12,
            decoder_ffn_dim=3072,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load config from a pretrained model.

        Supports loading from:
        - HuggingFace Whisper models (e.g., "openai/whisper-base")
        - Our saved WhisperEncoderConfig models (with model_type="whisper-encoder")

        Args:
            pretrained_model_name_or_path: Model name or path
            **kwargs: Additional arguments

        Returns:
            WhisperEncoderConfig instance
        """
        import json
        import os

        # Check if it's our format (model_type="whisper-encoder")
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    if config_dict.get("model_type") == "whisper-encoder":
                        # Load as our encoder config (use BaseConfig's from_pretrained)
                        return super(WhisperEncoderConfig, cls).from_pretrained(
                            pretrained_model_name_or_path, **kwargs
                        )

        # Load from HuggingFace Whisper format
        hf_config = HFWhisperConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Extract encoder-relevant attributes
        return cls(
            num_mel_bins=hf_config.num_mel_bins,
            d_model=hf_config.d_model,
            encoder_layers=hf_config.encoder_layers,
            encoder_attention_heads=hf_config.encoder_attention_heads,
            encoder_ffn_dim=hf_config.encoder_ffn_dim,
            dropout=hf_config.dropout,
            attention_dropout=hf_config.attention_dropout,
            activation_dropout=hf_config.activation_dropout,
            activation_function=hf_config.activation_function,
            init_std=hf_config.init_std,
            encoder_layerdrop=hf_config.encoder_layerdrop,
            scale_embedding=hf_config.scale_embedding,
            max_source_positions=hf_config.max_source_positions,
            apply_spec_augment=hf_config.apply_spec_augment,
            mask_time_prob=hf_config.mask_time_prob,
            mask_time_length=hf_config.mask_time_length,
            mask_time_min_masks=hf_config.mask_time_min_masks,
            mask_feature_prob=hf_config.mask_feature_prob,
            mask_feature_length=hf_config.mask_feature_length,
            mask_feature_min_masks=hf_config.mask_feature_min_masks,
            median_filter_width=hf_config.median_filter_width,
        )


# Preset configurations
whisper_base_config = WhisperEncoderConfig(
    d_model=512,
    encoder_layers=6,
    encoder_attention_heads=8,
    encoder_ffn_dim=2048,
)

whisper_small_config = WhisperEncoderConfig(
    d_model=768,
    encoder_layers=12,
    encoder_attention_heads=12,
    encoder_ffn_dim=3072,
)

whisper_medium_config = WhisperEncoderConfig(
    d_model=1024,
    encoder_layers=24,
    encoder_attention_heads=16,
    encoder_ffn_dim=4096,
)

whisper_large_config = WhisperEncoderConfig(
    d_model=1280,
    encoder_layers=32,
    encoder_attention_heads=20,
    encoder_ffn_dim=5120,
)
