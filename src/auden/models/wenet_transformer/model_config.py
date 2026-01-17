"""
WeNet Transformer/Conformer encoder model configuration.

Attribution:
    This configuration corresponds to the Transformer and Conformer architectures
    introduced by the WeNet project, adapted for the auden framework.
    Reference implementation (WeNet):
    https://github.com/wenet-e2e/wenet
"""

from typing import Optional

from ..base.model_config import BaseConfig


class WenetTransformerConfig(BaseConfig):
    """Configuration for WeNet Transformer/Conformer encoder models.

    Supports two encoder types:
    - "transformer": Standard Transformer encoder
    - "conformer": Conformer encoder (combines Transformer and convolution modules)

    Args:
        encoder_type (str, default="conformer"):
            Encoder type, choose "transformer" or "conformer"
        input_size (int, default=80):
            Input feature dimension (e.g., FBANK feature dimension)
        output_size (int, default=256):
            Encoder output dimension (attention dimension)
        attention_heads (int, default=4):
            Number of multi-head attention heads
        linear_units (int, default=2048):
            Number of hidden units in feedforward network
        num_blocks (int, default=6):
            Number of encoder blocks
        dropout_rate (float, default=0.1):
            Dropout rate
        positional_dropout_rate (float, default=0.1):
            Dropout rate after positional encoding
        attention_dropout_rate (float, default=0.0):
            Dropout rate in attention layer
        input_layer (str, default="conv2d"):
            Input layer type, options: ["linear", "conv2d", "conv2d6", "conv2d8"]
        pos_enc_layer_type (str, default="rel_pos" for conformer, "abs_pos" for transformer):
            Positional encoding layer type, options: ["abs_pos", "scaled_abs_pos", "rel_pos", "no_pos", "rope_pos"]
        normalize_before (bool, default=True):
            True: use layer_norm before each sub-block
            False: use layer_norm after each sub-block
        static_chunk_size (int, default=0):
            Static chunk size for training and decoding
        use_dynamic_chunk (bool, default=False):
            Whether to use dynamic chunk size for training
        global_cmvn (None):
            Optional GlobalCMVN module (typically None)
        use_dynamic_left_chunk (bool, default=False):
            Whether to use dynamic left chunk in dynamic chunk training
        gradient_checkpointing (bool, default=False):
            Whether to use gradient checkpointing to save memory
        use_sdpa (bool, default=False):
            Whether to use SDPA (Scaled Dot-Product Attention)
        layer_norm_type (str, default="layer_norm"):
            Normalization layer type, options: ["layer_norm", "rms_norm"]
        norm_eps (float, default=1e-5):
            Epsilon value for normalization layer

        # Conformer-specific parameters (only used when encoder_type="conformer"):
        positionwise_conv_kernel_size (int, default=1):
            Kernel size for position-wise convolution layer
        macaron_style (bool, default=True):
            Whether to use macaron style for position-wise layer
        selfattention_layer_type (str, default="rel_selfattn" for conformer, "selfattn" for transformer):
            Encoder attention layer type
        activation_type (str, default="swish" for conformer, "relu" for transformer):
            Activation function type
        use_cnn_module (bool, default=True):
            Whether to use convolution module (Conformer only)
        cnn_module_kernel (int, default=15):
            Kernel size for convolution module
        causal (bool, default=False):
            Whether to use causal convolution
        cnn_module_norm (str, default="batch_norm"):
            Normalization type for convolution module

        # Attention bias parameters:
        query_bias (bool, default=True):
            Whether to use bias in attention layer linear_q
        key_bias (bool, default=True):
            Whether to use bias in attention layer linear_k (False for Whisper models)
        value_bias (bool, default=True):
            Whether to use bias in attention layer linear_v
        conv_bias (bool, default=True):
            Whether to use bias in convolution layers

        # Advanced parameters:
        n_kv_head (Optional[int], default=None):
            Number of key-value heads (for grouped query attention)
        head_dim (Optional[int], default=None):
            Dimension per head
        mlp_type (str, default="position_wise_feed_forward"):
            MLP type
        mlp_bias (bool, default=True):
            Whether MLP uses bias
        n_expert (int, default=8):
            Number of experts (for MoE)
        n_expert_activated (int, default=2):
            Number of activated experts (for MoE)

        **kwargs:
            Extra parameters for forward compatibility
    """

    model_type: str = "wenet-transformer"

    def __init__(
        self,
        encoder_type: str = "conformer",
        input_size: int = 80,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: Optional[str] = None,
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn=None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        # Conformer
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        # Attention
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        # Advanced parameters
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = "position_wise_feed_forward",
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
        **kwargs,
    ):
        """HF-style initializer with explicit defaults and forward-compatible kwargs."""
        # Preserve unknown keys for forward compatibility
        super().__init__(**kwargs)

        # Validate encoder_type
        if encoder_type not in ["conformer", "transformer"]:
            raise ValueError(
                f"encoder_type must be 'conformer' or 'transformer', got '{encoder_type}'"
            )

        # Basic parameters
        self.encoder_type = encoder_type
        self.input_size = input_size
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.input_layer = input_layer

        # Set defaults based on encoder_type
        if pos_enc_layer_type is None:
            self.pos_enc_layer_type = (
                "rel_pos" if encoder_type == "conformer" else "abs_pos"
            )
        else:
            self.pos_enc_layer_type = pos_enc_layer_type

        if selfattention_layer_type is None:
            self.selfattention_layer_type = (
                "rel_selfattn" if encoder_type == "conformer" else "selfattn"
            )
        else:
            self.selfattention_layer_type = selfattention_layer_type

        if activation_type is None:
            self.activation_type = "swish" if encoder_type == "conformer" else "relu"
        else:
            self.activation_type = activation_type

        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.global_cmvn = global_cmvn
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa
        self.layer_norm_type = layer_norm_type
        self.norm_eps = norm_eps

        # Conformer-specific parameters
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.macaron_style = macaron_style
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.causal = causal
        self.cnn_module_norm = cnn_module_norm

        # Attention bias
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.conv_bias = conv_bias

        # Advanced parameters
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.mlp_type = mlp_type
        self.mlp_bias = mlp_bias
        self.n_expert = n_expert
        self.n_expert_activated = n_expert_activated


# Preset configurations
wenet_conformer_base_config = WenetTransformerConfig(
    encoder_type="conformer",
    output_size=256,
    attention_heads=4,
    linear_units=2048,
    num_blocks=12,
)

wenet_transformer_base_config = WenetTransformerConfig(
    encoder_type="transformer",
    output_size=256,
    attention_heads=4,
    linear_units=2048,
    num_blocks=6,
)
