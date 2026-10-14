# Adapted from Spatial-Omni for self-contained inference in Auden.
"""SO-Encoder backbone for first-order ambisonic audio."""

import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm
from tqdm.auto import tqdm

from .backbone import TransformerEncoder
from .so_modules import (
    ACCDOAHeads,
    FixedSlotReadout,
    FrameACCDOAPredictionOutput,
    FrameSlotHead,
    FrameSlotPredictionOutput,
    FrameTrackPredictionHeads,
    FrameTrackPredictionOutput,
    FrameWisePredictionHeads,
    FrameWisePredictionOutput,
    FrequencyPool,
    LocalSpatialCrossFuser,
    LocalSpatialEncoder,
    LocalSpatialPredictionHeads,
    MonoTaskPredictionHeads,
    MonoTaskPredictionOutput,
    MonoTaskTokenReadout,
    PreTrunkASTPredictionHeads,
    PreTrunkASTPredictionOutput,
    ShallowTemporalReadout,
    SourceQueryDecoder,
    SpatialAdapterLayer,
    SOBackbonePreprocessor,
    SpatialDeltaPatchAdapter,
    SpatialDeltaPatchAdapterV2,
    SpatialDeltaPatchAdapterV3,
    SpatialPatchEmbedding,
    SpatialPredictionHeads,
    SpatialPredictionOutput,
    SOTokenProjector,
    TemporalResampler,
    TrackRefinementDecoder,
)

LOCAL_SPATIAL_FRAME_SCHEMES: Tuple[str, ...] = (
    "local_spatial_slot",
    "local_spatial_track",
    "local_spatial_accdoa",
    "local_spatial_framewise",
)

class BEATsConfig:
    """Fallback copy of BEATsConfig used when BEATs import is unavailable."""

    def __init__(self, cfg=None):
        self.input_patch_size = -1
        self.embed_dim = 512
        self.conv_bias = False
        self.encoder_layers = 12
        self.encoder_embed_dim = 768
        self.encoder_ffn_embed_dim = 3072
        self.encoder_attention_heads = 12
        self.activation_fn = "gelu"
        self.layer_wise_gradient_decay_ratio = 1.0
        self.layer_norm_first = False
        self.deep_norm = False
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.encoder_layerdrop = 0.0
        self.dropout_input = 0.0
        self.conv_pos = 128
        self.conv_pos_groups = 16
        self.relative_position_embedding = True
        self.num_buckets = 320
        self.max_distance = 1280
        self.gru_rel_pos = True
        self.finetuned_model = False
        self.predictor_dropout = 0.1
        self.predictor_class = 527
        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: Dict) -> None:
        self.__dict__.update(cfg)



class SOBackboneConfig(BEATsConfig):
    """Configuration for the simplified Spatial-BEATs encoder.

    This extends the original BEATs config with the components required by the
    FOA spatial encoder and the supervision heads used during encoder-only
    training.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg=None)

        # 与 BEATs_iter3_plus_AS2M 保持完全一致的 encoder 结构：
        # relative position embedding + GRU gating (grep) + deep norm
        # BEATs.py 的默认值是 False，这里强制覆盖为与预训练模型一致的值
        self.relative_position_embedding: bool = True
        self.gru_rel_pos: bool = True
        self.deep_norm: bool = True
        self.max_distance: int = 800
        self.encoder_layerdrop: float = 0.05

        # FOA front-end configuration.
        self.sample_rate: int = 16000
        self.num_mel_bins: int = 128
        self.n_fft: int = 400
        self.hop_length: int = 160
        self.win_length: int = 400
        self.frame_length_ms: float = 25.0
        self.frame_shift_ms: float = 10.0
        self.dither: float = 0.0
        self.waveform_scale: float = float(2**15)
        self.fbank_mean: float = 15.41663
        self.fbank_std: float = 6.55582
        self.normalize_logmel: bool = True
        self.padding_side: str = "right"
        self.padding_value: float = 0.0
        self.return_attention_mask: bool = True
        self.qwen_like_chunk_length_seconds: float = 300.0
        self.foa_feature_channels: int = 7
        self.patch_adapter_hidden_dim: int = 32
        self.patch_adapter_residual_alpha_init: float = 0.1
        self.patch_adapter_out_proj_scale_init: float = 0.1
        self.input_patch_size: Tuple[int, int] = (16, 16)
        self.use_kaldi_w_channel: bool = False

        # SpecAugment on W channel logmel. Defaults off to preserve behavior.
        self.spec_augment_freq_masks: int = 0
        self.spec_augment_freq_width: int = 0
        self.spec_augment_time_masks: int = 0
        self.spec_augment_time_width: int = 0

        # Prediction head dropout. Default 0.0 preserves existing behavior.
        self.head_dropout: float = 0.0

        # Semantic anchor: auxiliary class head on pre-fusion BEATs tokens.
        # When True, an extra Linear(D, num_classes) is added to
        # LocalSpatialPredictionHeads and its loss keeps the BEATs trunk
        # semantically grounded under spatial gradient pressure.
        # Default False preserves existing behavior.
        self.use_semantic_anchor: bool = False
        # When True, pred_class_logits comes from mean-pool(semantic_tokens)
        # instead of attention-pool(fused_tokens). Classification is fully
        # decoupled from spatial fusion — same path as pure BEATs cls.
        # Default False preserves existing behavior.
        self.use_direct_cls: bool = False

        # Temporal tokenization.
        self.target_token_rate: float = 2.5
        self.readout_layers: int = 1
        self.readout_scheme: str = "fixed_slot"
        self.mono_task_readout_layers: int = 1
        self.local_spatial_dim: int = 256
        self.local_spatial_layers: int = 2
        self.local_spatial_heads: int = 4
        self.local_spatial_dropout: float = 0.1
        self.local_spatial_proj_scale_init: float = 0.05
        # How semantic and spatial streams are fused after the BEATs trunk:
        #   add               -> LN(semantic + spatial)
        #   cross_attn_gated  -> semantic<-spatial cross-attn blocks + gated
        #                        spatial residual before the same final LN
        self.local_spatial_fusion_mode: str = "add"
        self.local_spatial_fusion_layers: int = 2
        self.local_spatial_fusion_heads: int = 8
        self.local_spatial_fusion_dropout: float = 0.1
        self.local_spatial_fusion_gate_bias: float = -2.0
        self.local_spatial_fusion_direct_gate_bias: float = -1.5

        # Bypass local spatial fusion during classwarmup so the class head
        # reads from pure BEATs semantic tokens (fused = LayerNorm(semantic)).
        # During bypass, local_spatial_encoder is kept in the model but its
        # output is skipped — set ddp_find_unused_parameters=True when using
        # this. Stage 2 should disable bypass to restore normal fusion.
        # Default False preserves existing behavior.
        self.bypass_local_fusion: bool = False

        # When True, the SpatialDeltaPatchAdapter output is NOT added to the
        # BEATs patch tokens before the trunk. The W-channel base tokens enter
        # the trunk alone (identical to how pure BEATs classification works),
        # and spatial information reaches the model only via local_spatial_encoder
        # AFTER the trunk. This preserves the trunk's pretrained semantic
        # features and prevents spatial gradients from polluting the trunk via
        # the patch-level delta path.
        # Default False preserves existing behavior.
        self.bypass_spatial_delta: bool = False

        # Multi-source frame-level supervision heads built on top of the
        # ``local_spatial`` fusion. These fields only take effect when
        # ``readout_scheme`` is one of LOCAL_SPATIAL_FRAME_SCHEMES.
        # ``enable_frame_track`` additionally allows the frame-level track head
        # to coexist with the clip-level ``local_spatial`` readout, running
        # both mono_ast clip loss AND frame-level track loss in parallel.
        self.enable_frame_track: bool = False
        self.use_original_beats_semantic_frontend_for_local_spatial_frame: bool = True
        self.enable_clip_aux_head: bool = True
        self.frame_slot_num_slots: int = 4
        self.frame_slot_hidden_dim: int = 192
        self.frame_slot_dropout: float = 0.1
        self.frame_track_num_queries: int = 4
        self.frame_track_num_heads: int = 8
        self.frame_track_num_track_layers: int = 2
        self.frame_track_num_time_layers: int = 1
        self.frame_track_max_time_steps: int = 64
        self.frame_track_dropout: float = 0.1
        self.frame_accdoa_hidden_dim: int = 256
        self.frame_accdoa_dropout: float = 0.1

        # optional zero-initialised MLP residual branch inside
        # FrameTrackPredictionHeads.  When True, the class head output is
        # `class_head(x) + class_head_mlp(x) * gate`, where class_head_mlp
        # is a 2-layer MLP and `gate` starts at 0.  This preserves the
        # legacy Linear(class_head) output on load (hot-start safe) while
        # giving the head strictly more capacity for multi-source demixing.
        self.use_class_head_mlp_residual: bool = False
        self.class_head_mlp_hidden_multiplier: int = 2
        self.class_head_mlp_dropout: float = 0.1
        # optional spectral demixing cross-attention branch.  When True,
        # each track latent attends to the pre-frequency-pool patch tokens
        # (i.e. BEATs trunk output BEFORE frequency_pool) with a
        # time-locality mask, and the attention result is added to the
        # class_head input via a zero-initialised gate.  This gives the
        # class head a frequency-axis demixing path for multi-source frames.
        self.use_class_head_demixer: bool = False
        self.class_head_demixer_layers: int = 1
        self.class_head_demixer_heads: int = 8
        self.class_head_demixer_dropout: float = 0.1

        # Symmetric spectral demixer for the direction / distance heads.
        # Structurally identical to the class-head demixer (same zero-init
        # + tiny-gate trick), but independent parameters, and the residual is
        # added to the DOA/distance head inputs instead of the class-head
        # input.  Targets the real_ov2 "class right, angle wrong" failure
        # mode where the post-frequency-pool single vector can't represent
        # multiple source directions in multi-source frames.
        self.use_spatial_head_demixer: bool = False
        self.spatial_head_demixer_layers: int = 1
        self.spatial_head_demixer_heads: int = 8
        self.spatial_head_demixer_dropout: float = 0.1
        # switch the spatial demixer KV to the local_spatial (IV)
        # pre-pool grid rather than the BEATs mono trunk pre-pool grid.
        # Ignored when use_spatial_head_demixer is False.
        self.spatial_demixer_use_local_spatial_kv: bool = False

        # optional per-frame num-active-source head on top of
        # FrameTrackPredictionHeads.  Predicts how many of the K tracks should
        # be considered active at each frame so downstream CSV and validation
        # metrics can pick a top-K̂ subset instead of relying on a hard 0.5
        # activity threshold.  Zero-initialised to "predict 0 active" at load,
        # so hot-starting a a legacy checkpoint with strict=False produces
        # identical forward output (validation falls back to 0.5 gating).
        self.use_num_active_head: bool = False
        self.num_active_max: int = 4

        # enhanced spatial delta adapter (V2) — deeper Conv-ResBlock-SE
        # front-end that replaces the thin V1 bottleneck.  Default "v1"
        # preserves existing behavior exactly.
        self.patch_adapter_version: str = "v1"
        self.patch_adapter_v2_hidden: int = 128
        self.patch_adapter_v2_blocks: int = 2
        self.patch_adapter_v2_se_reduction: int = 4

        # per-layer trunk spatial adapters — zero-init bottleneck adapters
        # injected after each BEATs trunk layer to maintain spatial conditioning
        # throughout the 12-layer self-attention.  Default False preserves
        # existing behavior.
        self.use_trunk_spatial_adapters: bool = False
        self.trunk_adapter_rank: int = 64
        self.trunk_adapter_layers: str = "all"   # "all" / "top4" / "top8"
        self.trunk_adapter_gate_init: float = 1e-2

        # === per-class learnable activity bias (FrameTrackHeads) =
        self.use_class_activity_bias: bool = False
        # === class-conditional activity gate =====================
        self.use_class_conditional_gate: bool = False
        self.gate_class_emb_dim: int = 32
        self.gate_hidden_dim: int = 128
        self.gate_scale: float = 0.5

        # === track-wise refinement decoder =======================
        self.use_track_refinement: bool = False
        self.track_refinement_layers: int = 2
        self.track_refinement_heads: int = 8
        self.track_refinement_ffn: int = 2048
        self.track_refinement_dropout: float = 0.0

        # === log-distance + Laplace NLL head =====================
        self.use_log_distance_head: bool = False
        self.log_distance_init_mean: float = 0.4        # log(1.5) ≈ 0.405
        self.log_distance_init_log_var: float = -3.2   # log(0.04) ≈ -3.22

        # Source label vocabulary.
        self.source_vocab_path: str = ""
        self.source_label_id_field: str = "label_id"
        self.source_label_name_field: str = "final_label"
        self.source_num_classes: int = 63

        # Fixed-slot supervision readout.
        self.max_sources: int = 4
        self.slot_hidden_dim: int = 768
        self.num_azi_bins: int = 360
        self.num_ele_bins: int = 180
        self.distance_head_type: str = "regression"
        self.num_distance_bins: int = 21
        self.distance_bin_size_m: float = 0.5

        # LLM projection.
        self.llm_hidden_dim: int = 4096
        self.projector_hidden_dim: int = 768

        if cfg is not None:
            self.update(cfg)


@dataclass
class SOBackboneOutput:
    """Structured outputs from the simplified Spatial-BEATs forward pass.

    Attributes:
        foa_feat:
            [B, 7, T_f, F] multi-channel FOA spatial feature map.
        fused_feat:
            [B, 1, T_f, F] single-channel W log-mel map fed into the original
            pretrained BEATs patch embedding.
        delta_patch_tokens:
            Optional [B, N_p, D_in] additive patch-token deltas produced by the
            full 7-channel spatial adapter before the BEATs trunk.
        patch_tokens:
            [B, N_p, D_in] flattened patch tokens before the BEATs trunk.
        grid_size:
            (T_p, F_p) patch grid size used to reshape N_p back to 2D.
        encoder_memory:
            [B, N_p, D] BEATs trunk output over patch tokens.
        temporal_patch_tokens:
            [B, T_p, D] sequence after frequency pooling.
        temporal_tokens:
            [B, T_s_max, D] sequence after resampling and padding to the
            longest valid sequence in the batch.
        spatial_embeddings:
            [B, T_s_max, D] main spatial embedding sequence used for both
            supervision and projection to the LLM space.
        slot_latents:
            Optional [B, T_s_max, K, H] fixed-slot supervision features.
        prediction_output:
            Optional structured slot-level supervision outputs.
        mono_task_tokens:
            Optional [B, 2, D] class/spatial task tokens used by the
            single-source Spatial-AST-style readout.
        mono_prediction_output:
            Optional structured clip/source-level outputs for the single-source
            Spatial-AST-style supervision path.
        local_spatial_tokens:
            Optional [B, T_s_max, D_s] local CNN/attention spatial sequence
            resampled to the same target rate as the BEATs temporal tokens.
        fused_spatial_embeddings:
            Optional [B, T_s_max, D] fusion of BEATs semantic temporal tokens
            and projected local spatial tokens.
        pretrunk_task_tokens:
            Optional [B, 3, D] distance/DoA/class task tokens that participated
            in the BEATs trunk self-attention.
        pretrunk_prediction_output:
            Optional structured classification outputs for the pre-trunk
            Spatial-AST-style supervision path.
        llm_spatial_tokens:
            [B, T_s_max, d_llm] final spatial tokens that will be sent to the LLM.
        temporal_padding_mask:
            Optional [B, T_s_max] mask where True marks padded time steps after
            resampling.
        target_num_steps:
            [B] valid temporal lengths before padding.
    """

    foa_feat: Tensor
    fused_feat: Tensor
    delta_patch_tokens: Optional[Tensor]
    patch_tokens: Tensor
    grid_size: Tuple[int, int]
    encoder_memory: Tensor
    temporal_patch_tokens: Tensor
    temporal_tokens: Tensor
    spatial_embeddings: Tensor
    slot_latents: Optional[Tensor]
    prediction_output: Optional[SpatialPredictionOutput]
    mono_task_tokens: Optional[Tensor]
    mono_prediction_output: Optional[MonoTaskPredictionOutput]
    local_spatial_tokens: Optional[Tensor]
    fused_spatial_embeddings: Optional[Tensor]
    pretrunk_task_tokens: Optional[Tensor]
    pretrunk_prediction_output: Optional[PreTrunkASTPredictionOutput]
    llm_spatial_tokens: Tensor
    temporal_padding_mask: Optional[Tensor]
    target_num_steps: Optional[Tensor]
    frame_slot_prediction_output: Optional[FrameSlotPredictionOutput] = None
    frame_track_prediction_output: Optional[FrameTrackPredictionOutput] = None
    frame_accdoa_prediction_output: Optional[FrameACCDOAPredictionOutput] = None
    frame_wise_prediction_output: Optional[FrameWisePredictionOutput] = None
    clip_aux_prediction_output: Optional[MonoTaskPredictionOutput] = None


class SOBackbone(nn.Module):
    """Encode FOA waveforms with a BEATs trunk and spatial supervision heads.

    Temporal outputs carry per-sample valid lengths and padding masks.
    The configured token rate determines the temporal resolution; task
    heads provide the class, activity, direction and distance losses.
    """

    def __init__(self, cfg: SOBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim

        self.preprocessor = SOBackbonePreprocessor(
            sample_rate=cfg.sample_rate,
            num_mel_bins=cfg.num_mel_bins,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            frame_length_ms=cfg.frame_length_ms,
            frame_shift_ms=cfg.frame_shift_ms,
            dither=cfg.dither,
            waveform_scale=cfg.waveform_scale,
            fbank_mean=cfg.fbank_mean,
            fbank_std=cfg.fbank_std,
            normalize_logmel=cfg.normalize_logmel,
            use_kaldi_w_channel=cfg.use_kaldi_w_channel,
        )
        self.preprocessor.spec_augment_freq_masks = cfg.spec_augment_freq_masks
        self.preprocessor.spec_augment_freq_width = cfg.spec_augment_freq_width
        self.preprocessor.spec_augment_time_masks = cfg.spec_augment_time_masks
        self.preprocessor.spec_augment_time_width = cfg.spec_augment_time_width
        if cfg.patch_adapter_version == "v2":
            self.spatial_patch_adapter = SpatialDeltaPatchAdapterV2(
                in_channels=cfg.foa_feature_channels,
                hidden_channels=cfg.patch_adapter_v2_hidden,
                embed_dim=cfg.embed_dim,
                patch_size=cfg.input_patch_size,
                num_blocks=cfg.patch_adapter_v2_blocks,
                se_reduction=cfg.patch_adapter_v2_se_reduction,
                residual_scale_init=cfg.patch_adapter_residual_alpha_init,
                out_proj_scale_init=cfg.patch_adapter_out_proj_scale_init,
            )
        elif cfg.patch_adapter_version == "v3":
            # multi-scale adapter (3x3 + 5x5 + dilated)
            self.spatial_patch_adapter = SpatialDeltaPatchAdapterV3(
                in_channels=cfg.foa_feature_channels,
                hidden_channels=cfg.patch_adapter_v2_hidden,
                embed_dim=cfg.embed_dim,
                patch_size=cfg.input_patch_size,
                num_blocks=cfg.patch_adapter_v2_blocks,
                se_reduction=cfg.patch_adapter_v2_se_reduction,
                residual_scale_init=cfg.patch_adapter_residual_alpha_init,
                out_proj_scale_init=cfg.patch_adapter_out_proj_scale_init,
            )
        else:
            self.spatial_patch_adapter = SpatialDeltaPatchAdapter(
                in_channels=cfg.foa_feature_channels,
                hidden_channels=cfg.patch_adapter_hidden_dim,
                embed_dim=cfg.embed_dim,
                patch_size=cfg.input_patch_size,
                residual_scale_init=cfg.patch_adapter_residual_alpha_init,
                out_proj_scale_init=cfg.patch_adapter_out_proj_scale_init,
            )
        self.patch_embedding = SpatialPatchEmbedding(
            in_channels=1,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.input_patch_size,
            bias=cfg.conv_bias,
        )

        self.layer_norm = LayerNorm(cfg.embed_dim)
        self.post_extract_proj = (
            nn.Linear(cfg.embed_dim, cfg.encoder_embed_dim)
            if cfg.embed_dim != cfg.encoder_embed_dim
            else None
        )
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.encoder = TransformerEncoder(cfg)

        # trunk spatial adapters (zero-init bottleneck after each layer)
        self.trunk_spatial_adapters: Optional[nn.ModuleList] = None
        if cfg.use_trunk_spatial_adapters:
            num_layers = len(self.encoder.layers)
            if cfg.trunk_adapter_layers == "all":
                adapter_indices = list(range(num_layers))
            elif cfg.trunk_adapter_layers == "top4":
                adapter_indices = list(range(max(0, num_layers - 4), num_layers))
            elif cfg.trunk_adapter_layers == "top8":
                adapter_indices = list(range(max(0, num_layers - 8), num_layers))
            else:
                raise ValueError(f"Unknown trunk_adapter_layers: {cfg.trunk_adapter_layers}")
            adapters = [None] * num_layers
            for idx in adapter_indices:
                adapters[idx] = SpatialAdapterLayer(
                    embed_dim=cfg.encoder_embed_dim,
                    rank=cfg.trunk_adapter_rank,
                    gate_init=cfg.trunk_adapter_gate_init,
                )
            self.trunk_spatial_adapters = nn.ModuleList(adapters)

        self.frequency_pool = FrequencyPool(mode="mean")
        self.temporal_resampler = TemporalResampler(
            target_token_rate=cfg.target_token_rate,
            mode="linear",
        )
        self.temporal_readout = ShallowTemporalReadout(
            embed_dim=cfg.encoder_embed_dim,
            num_layers=cfg.readout_layers,
            num_heads=cfg.encoder_attention_heads,
            dropout=cfg.dropout,
        )
        if cfg.readout_scheme == "fixed_slot":
            self.slot_readout = FixedSlotReadout(
                input_dim=cfg.encoder_embed_dim,
                slot_hidden_dim=cfg.slot_hidden_dim,
                num_slots=cfg.max_sources,
            )
            self.prediction_heads = SpatialPredictionHeads(
                slot_hidden_dim=cfg.slot_hidden_dim,
                num_classes=cfg.source_num_classes,
                num_azi_bins=cfg.num_azi_bins,
                num_ele_bins=cfg.num_ele_bins,
            )
            self.mono_task_readout = None
            self.mono_prediction_heads = None
            self.local_spatial_encoder = None
            self.local_spatial_resampler = None
            self.local_spatial_proj = None
            self.local_spatial_pre_pool_proj = None
            self.local_spatial_fusion_norm = None
            self.local_spatial_prediction_heads = None
            self.pretrunk_task_tokens = None
            self.pretrunk_prediction_heads = None
        elif cfg.readout_scheme == "mono_ast":
            self.slot_readout = None
            self.prediction_heads = None
            self.mono_task_readout = MonoTaskTokenReadout(
                embed_dim=cfg.encoder_embed_dim,
                num_layers=cfg.mono_task_readout_layers,
                num_heads=cfg.encoder_attention_heads,
                dropout=cfg.dropout,
            )
            self.mono_prediction_heads = MonoTaskPredictionHeads(
                embed_dim=cfg.encoder_embed_dim,
                num_classes=cfg.source_num_classes,
            )
            self.local_spatial_encoder = None
            self.local_spatial_resampler = None
            self.local_spatial_proj = None
            self.local_spatial_pre_pool_proj = None
            self.local_spatial_fusion_norm = None
            self.local_spatial_prediction_heads = None
            self.pretrunk_task_tokens = None
            self.pretrunk_prediction_heads = None
        elif cfg.readout_scheme == "local_spatial":
            self.slot_readout = None
            self.prediction_heads = None
            self.mono_task_readout = None
            self.mono_prediction_heads = None
            self.local_spatial_encoder = LocalSpatialEncoder(
                in_channels=cfg.foa_feature_channels,
                hidden_dim=cfg.local_spatial_dim,
                num_layers=cfg.local_spatial_layers,
                num_heads=cfg.local_spatial_heads,
                dropout=cfg.local_spatial_dropout,
            )
            self.local_spatial_resampler = TemporalResampler(
                target_token_rate=cfg.target_token_rate,
                mode="linear",
            )
            self.local_spatial_proj = nn.Linear(
                cfg.local_spatial_dim,
                cfg.encoder_embed_dim,
            )
            nn.init.xavier_uniform_(self.local_spatial_proj.weight)
            self.local_spatial_proj.weight.data.mul_(cfg.local_spatial_proj_scale_init)
            nn.init.zeros_(self.local_spatial_proj.bias)
            # dedicated projection for the pre-F-pool local_spatial
            # tokens fed as KV to the DOA spectral demixer.  Separate from
            # ``local_spatial_proj`` so training the DOA demixer path does
            # not perturb the post-pool fusion branch (legacy ckpt semantics).
            # Only built when explicitly requested; strict=False load.
            if getattr(cfg, "use_spatial_head_demixer", False) and getattr(
                cfg, "spatial_demixer_use_local_spatial_kv", False
            ):
                self.local_spatial_pre_pool_proj = nn.Linear(
                    cfg.local_spatial_dim,
                    cfg.encoder_embed_dim,
                )
                nn.init.xavier_uniform_(self.local_spatial_pre_pool_proj.weight)
                self.local_spatial_pre_pool_proj.weight.data.mul_(
                    cfg.local_spatial_proj_scale_init
                )
                nn.init.zeros_(self.local_spatial_pre_pool_proj.bias)
            else:
                self.local_spatial_pre_pool_proj = None
            self.local_spatial_fusion_norm = nn.LayerNorm(cfg.encoder_embed_dim)
            self.local_spatial_fuser = (
                LocalSpatialCrossFuser(
                    embed_dim=cfg.encoder_embed_dim,
                    num_layers=cfg.local_spatial_fusion_layers,
                    num_heads=cfg.local_spatial_fusion_heads,
                    dropout=cfg.local_spatial_fusion_dropout,
                    gate_bias=cfg.local_spatial_fusion_gate_bias,
                    direct_gate_bias=cfg.local_spatial_fusion_direct_gate_bias,
                )
                if cfg.local_spatial_fusion_mode == "cross_attn_gated"
                else None
            )
            self.local_spatial_prediction_heads = LocalSpatialPredictionHeads(
                embed_dim=cfg.encoder_embed_dim,
                num_classes=cfg.source_num_classes,
                head_dropout=cfg.head_dropout,
                use_semantic_anchor=cfg.use_semantic_anchor,
                use_direct_cls=cfg.use_direct_cls,
            )
            # Optional frame-level track head (coexists with clip-level head).
            # When enable_frame_track=True, SourceQueryDecoder + FrameTrack heads
            # run in parallel with the clip-level mono_ast supervision, producing
            # per-frame per-track predictions for DCASE-style evaluation.
            if cfg.enable_frame_track:
                self.source_query_decoder = SourceQueryDecoder(
                    embed_dim=cfg.encoder_embed_dim,
                    num_queries=cfg.frame_track_num_queries,
                    num_heads=cfg.frame_track_num_heads,
                    num_track_layers=cfg.frame_track_num_track_layers,
                    num_time_layers=cfg.frame_track_num_time_layers,
                    max_time_steps=cfg.frame_track_max_time_steps,
                    dropout=cfg.frame_track_dropout,
                )
                self.frame_track_prediction_heads = FrameTrackPredictionHeads(
                    embed_dim=cfg.encoder_embed_dim,
                    num_classes=cfg.source_num_classes,
                    dropout=cfg.frame_track_dropout,
                    use_class_head_mlp_residual=cfg.use_class_head_mlp_residual,
                    class_head_mlp_hidden_multiplier=cfg.class_head_mlp_hidden_multiplier,
                    class_head_mlp_dropout=cfg.class_head_mlp_dropout,
                    use_class_head_demixer=cfg.use_class_head_demixer,
                    class_head_demixer_layers=cfg.class_head_demixer_layers,
                    class_head_demixer_heads=cfg.class_head_demixer_heads,
                    class_head_demixer_dropout=cfg.class_head_demixer_dropout,
                    use_spatial_head_demixer=cfg.use_spatial_head_demixer,
                    spatial_head_demixer_layers=cfg.spatial_head_demixer_layers,
                    spatial_head_demixer_heads=cfg.spatial_head_demixer_heads,
                    spatial_head_demixer_dropout=cfg.spatial_head_demixer_dropout,
                    use_num_active_head=cfg.use_num_active_head,
                    num_active_max=cfg.num_active_max,
                    use_class_activity_bias=cfg.use_class_activity_bias,
                    use_class_conditional_gate=cfg.use_class_conditional_gate,
                    gate_class_emb_dim=cfg.gate_class_emb_dim,
                    gate_hidden_dim=cfg.gate_hidden_dim,
                    gate_scale=cfg.gate_scale,
                    use_log_distance_head=cfg.use_log_distance_head,
                    log_distance_init_mean=cfg.log_distance_init_mean,
                    log_distance_init_log_var=cfg.log_distance_init_log_var,
                )
                # optional track-wise refinement decoder
                if cfg.use_track_refinement:
                    self.track_refinement_decoder = TrackRefinementDecoder(
                        num_tracks=cfg.frame_track_num_queries,
                        embed_dim=cfg.encoder_embed_dim,
                        num_layers=cfg.track_refinement_layers,
                        num_heads=cfg.track_refinement_heads,
                        dim_feedforward=cfg.track_refinement_ffn,
                        dropout=cfg.track_refinement_dropout,
                    )
                else:
                    self.track_refinement_decoder = None
            self.pretrunk_task_tokens = None
            self.pretrunk_prediction_heads = None
        elif cfg.readout_scheme == "pretrunk_ast":
            self.slot_readout = None
            self.prediction_heads = None
            self.mono_task_readout = None
            self.mono_prediction_heads = None
            self.local_spatial_encoder = None
            self.local_spatial_resampler = None
            self.local_spatial_proj = None
            self.local_spatial_pre_pool_proj = None
            self.local_spatial_fusion_norm = None
            self.local_spatial_prediction_heads = None
            self.pretrunk_task_tokens = nn.Parameter(torch.zeros(1, 3, cfg.encoder_embed_dim))
            nn.init.trunc_normal_(self.pretrunk_task_tokens, std=0.02)
            self.pretrunk_prediction_heads = PreTrunkASTPredictionHeads(
                embed_dim=cfg.encoder_embed_dim,
                num_classes=cfg.source_num_classes,
                num_distance_bins=cfg.num_distance_bins,
                num_azi_bins=cfg.num_azi_bins,
                num_ele_bins=cfg.num_ele_bins,
            )
        elif cfg.readout_scheme in LOCAL_SPATIAL_FRAME_SCHEMES:
            # Shared local_spatial fusion stack — identical to the
            # readout_scheme='local_spatial' branch above. Keeping the same
            # parameter names lets the ov1 local_spatial checkpoint load
            # cleanly when --init-from-spatial-ckpt is used.
            self.slot_readout = None
            self.prediction_heads = None
            self.mono_task_readout = None
            self.mono_prediction_heads = None
            self.local_spatial_encoder = LocalSpatialEncoder(
                in_channels=cfg.foa_feature_channels,
                hidden_dim=cfg.local_spatial_dim,
                num_layers=cfg.local_spatial_layers,
                num_heads=cfg.local_spatial_heads,
                dropout=cfg.local_spatial_dropout,
            )
            self.local_spatial_resampler = TemporalResampler(
                target_token_rate=cfg.target_token_rate,
                mode="linear",
            )
            self.local_spatial_proj = nn.Linear(
                cfg.local_spatial_dim,
                cfg.encoder_embed_dim,
            )
            nn.init.xavier_uniform_(self.local_spatial_proj.weight)
            self.local_spatial_proj.weight.data.mul_(cfg.local_spatial_proj_scale_init)
            nn.init.zeros_(self.local_spatial_proj.bias)
            # dedicated projection for the pre-F-pool local_spatial
            # tokens fed as KV to the DOA spectral demixer.  Separate from
            # ``local_spatial_proj`` so training the DOA demixer path does
            # not perturb the post-pool fusion branch (legacy ckpt semantics).
            # Only built when explicitly requested; strict=False load.
            if getattr(cfg, "use_spatial_head_demixer", False) and getattr(
                cfg, "spatial_demixer_use_local_spatial_kv", False
            ):
                self.local_spatial_pre_pool_proj = nn.Linear(
                    cfg.local_spatial_dim,
                    cfg.encoder_embed_dim,
                )
                nn.init.xavier_uniform_(self.local_spatial_pre_pool_proj.weight)
                self.local_spatial_pre_pool_proj.weight.data.mul_(
                    cfg.local_spatial_proj_scale_init
                )
                nn.init.zeros_(self.local_spatial_pre_pool_proj.bias)
            else:
                self.local_spatial_pre_pool_proj = None
            self.local_spatial_fusion_norm = nn.LayerNorm(cfg.encoder_embed_dim)
            self.local_spatial_fuser = (
                LocalSpatialCrossFuser(
                    embed_dim=cfg.encoder_embed_dim,
                    num_layers=cfg.local_spatial_fusion_layers,
                    num_heads=cfg.local_spatial_fusion_heads,
                    dropout=cfg.local_spatial_fusion_dropout,
                    gate_bias=cfg.local_spatial_fusion_gate_bias,
                    direct_gate_bias=cfg.local_spatial_fusion_direct_gate_bias,
                )
                if cfg.local_spatial_fusion_mode == "cross_attn_gated"
                else None
            )
            # Clip-level aux head (same as ov1 local_spatial), optionally
            # disabled from the trainer by setting enable_clip_aux_head=False.
            # When disabled (e.g. local_spatial_track pure per-frame path),
            # the module is not built at all — build_local_spatial_fusion
            # returns None for mono_task_tokens / mono_prediction_output.
            if cfg.enable_clip_aux_head:
                self.local_spatial_prediction_heads = LocalSpatialPredictionHeads(
                    embed_dim=cfg.encoder_embed_dim,
                    num_classes=cfg.source_num_classes,
                    head_dropout=cfg.head_dropout,
                    use_semantic_anchor=cfg.use_semantic_anchor,
                    use_direct_cls=cfg.use_direct_cls,
                )
            else:
                self.local_spatial_prediction_heads = None
            self.pretrunk_task_tokens = None
            self.pretrunk_prediction_heads = None
            # Scheme-specific frame-level multi-source supervision head.
            self.frame_slot_head: Optional[FrameSlotHead] = None
            self.source_query_decoder: Optional[SourceQueryDecoder] = None
            self.frame_track_prediction_heads: Optional[FrameTrackPredictionHeads] = None
            self.accdoa_heads: Optional[ACCDOAHeads] = None
            self.frame_wise_heads: Optional[FrameWisePredictionHeads] = None
            if cfg.readout_scheme == "local_spatial_slot":
                self.frame_slot_head = FrameSlotHead(
                    embed_dim=cfg.encoder_embed_dim,
                    num_slots=cfg.frame_slot_num_slots,
                    slot_hidden_dim=cfg.frame_slot_hidden_dim,
                    num_classes=cfg.source_num_classes,
                    dropout=cfg.frame_slot_dropout,
                )
            elif cfg.readout_scheme == "local_spatial_track":
                self.source_query_decoder = SourceQueryDecoder(
                    embed_dim=cfg.encoder_embed_dim,
                    num_queries=cfg.frame_track_num_queries,
                    num_heads=cfg.frame_track_num_heads,
                    num_track_layers=cfg.frame_track_num_track_layers,
                    num_time_layers=cfg.frame_track_num_time_layers,
                    max_time_steps=cfg.frame_track_max_time_steps,
                    dropout=cfg.frame_track_dropout,
                )
                self.frame_track_prediction_heads = FrameTrackPredictionHeads(
                    embed_dim=cfg.encoder_embed_dim,
                    num_classes=cfg.source_num_classes,
                    dropout=cfg.frame_track_dropout,
                    use_class_head_mlp_residual=cfg.use_class_head_mlp_residual,
                    class_head_mlp_hidden_multiplier=cfg.class_head_mlp_hidden_multiplier,
                    class_head_mlp_dropout=cfg.class_head_mlp_dropout,
                    use_class_head_demixer=cfg.use_class_head_demixer,
                    class_head_demixer_layers=cfg.class_head_demixer_layers,
                    class_head_demixer_heads=cfg.class_head_demixer_heads,
                    class_head_demixer_dropout=cfg.class_head_demixer_dropout,
                    use_spatial_head_demixer=cfg.use_spatial_head_demixer,
                    spatial_head_demixer_layers=cfg.spatial_head_demixer_layers,
                    spatial_head_demixer_heads=cfg.spatial_head_demixer_heads,
                    spatial_head_demixer_dropout=cfg.spatial_head_demixer_dropout,
                    use_num_active_head=cfg.use_num_active_head,
                    num_active_max=cfg.num_active_max,
                    use_class_activity_bias=cfg.use_class_activity_bias,
                    use_class_conditional_gate=cfg.use_class_conditional_gate,
                    gate_class_emb_dim=cfg.gate_class_emb_dim,
                    gate_hidden_dim=cfg.gate_hidden_dim,
                    gate_scale=cfg.gate_scale,
                    use_log_distance_head=cfg.use_log_distance_head,
                    log_distance_init_mean=cfg.log_distance_init_mean,
                    log_distance_init_log_var=cfg.log_distance_init_log_var,
                )
                # optional track-wise refinement decoder
                if cfg.use_track_refinement:
                    self.track_refinement_decoder = TrackRefinementDecoder(
                        num_tracks=cfg.frame_track_num_queries,
                        embed_dim=cfg.encoder_embed_dim,
                        num_layers=cfg.track_refinement_layers,
                        num_heads=cfg.track_refinement_heads,
                        dim_feedforward=cfg.track_refinement_ffn,
                        dropout=cfg.track_refinement_dropout,
                    )
                else:
                    self.track_refinement_decoder = None
            elif cfg.readout_scheme == "local_spatial_accdoa":
                self.accdoa_heads = ACCDOAHeads(
                    embed_dim=cfg.encoder_embed_dim,
                    num_classes=cfg.source_num_classes,
                    hidden_dim=cfg.frame_accdoa_hidden_dim,
                    dropout=cfg.frame_accdoa_dropout,
                )
            elif cfg.readout_scheme == "local_spatial_framewise":
                self.frame_wise_heads = FrameWisePredictionHeads(
                    embed_dim=cfg.encoder_embed_dim,
                    num_classes=cfg.source_num_classes,
                    hidden_dim=cfg.local_spatial_dim,
                    dropout=cfg.head_dropout,
                    use_semantic_anchor=cfg.use_semantic_anchor,
                    num_anchor_classes=cfg.source_num_classes,
                )
        else:
            raise ValueError(f"Unsupported readout_scheme: {cfg.readout_scheme}")
        # Ensure every branch has the frame-level head attributes declared so
        # forward() can check them without AttributeError.
        if (
            cfg.readout_scheme not in LOCAL_SPATIAL_FRAME_SCHEMES
            and cfg.readout_scheme != "local_spatial"
        ):
            self.frame_slot_head = None
            self.source_query_decoder = None
            self.frame_track_prediction_heads = None
            self.accdoa_heads = None
            self.frame_wise_heads = None
            self.local_spatial_fuser = None
        # declare attribute on all branches so forward() check works
        if not hasattr(self, "track_refinement_decoder"):
            self.track_refinement_decoder = None
        self.projector = SOTokenProjector(
            input_dim=cfg.encoder_embed_dim,
            llm_hidden_dim=cfg.llm_hidden_dim,
            hidden_dim=cfg.projector_hidden_dim,
        )

    def _compute_time_freq_lengths(
        self,
        clip_duration_seconds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_samples = torch.clamp(
            torch.round(clip_duration_seconds * self.cfg.sample_rate).long(),
            min=1,
        )
        # Matches torch.stft(..., center=True) frame count.
        t_f = torch.div(num_samples, self.cfg.hop_length, rounding_mode="floor") + 1
        patch_t = torch.div(
            torch.clamp(t_f - self.cfg.input_patch_size[0], min=0),
            self.cfg.input_patch_size[0],
            rounding_mode="floor",
        ) + 1
        return t_f, patch_t

    def _infer_clip_duration_seconds(
        self,
        waveform: Tensor,
        padding_mask: Optional[Tensor] = None,
        clip_duration_seconds: Optional[Tensor] = None,
    ) -> Tensor:
        if clip_duration_seconds is not None:
            return clip_duration_seconds.to(device=waveform.device, dtype=waveform.dtype)
        if padding_mask is not None:
            valid_samples = (~padding_mask.to(torch.bool)).sum(dim=1)
            return valid_samples.to(device=waveform.device, dtype=waveform.dtype) / float(self.cfg.sample_rate)
        return torch.full(
            (waveform.size(0),),
            float(waveform.size(-1)) / float(self.cfg.sample_rate),
            device=waveform.device,
            dtype=waveform.dtype,
        )

    @staticmethod
    def _round_divide_half_to_even(numerator: Tensor, denominator: int) -> Tensor:
        if denominator <= 0:
            raise ValueError(f"denominator must be > 0, got {denominator}")
        quotient = torch.div(numerator, denominator, rounding_mode="floor")
        remainder = torch.remainder(numerator, denominator)
        twice_remainder = remainder * 2
        round_up = (twice_remainder > denominator) | (
            (twice_remainder == denominator) & (torch.remainder(quotient, 2) == 1)
        )
        return quotient + round_up.to(dtype=quotient.dtype)

    def _build_patch_padding_mask(
        self,
        grid_size: Tuple[int, int],
        clip_duration_seconds: Tensor,
        device: torch.device,
    ) -> Tensor:
        t_p, f_p = grid_size
        _, valid_t_p = self._compute_time_freq_lengths(clip_duration_seconds)
        valid_t_p = torch.clamp(valid_t_p, min=1, max=t_p)
        time_mask = torch.arange(t_p, device=device).unsqueeze(0) >= valid_t_p.unsqueeze(1)
        patch_padding_mask = time_mask.unsqueeze(-1).expand(-1, -1, f_p).reshape(time_mask.size(0), -1)
        return patch_padding_mask

    def _derive_pre_pool_time_mask(
        self,
        patch_padding_mask: Optional[Tensor],
        grid_size: Tuple[int, int],
    ) -> Optional[Tensor]:
        """Return a [B, T_p] boolean mask where True marks *valid* trunk time
        steps.  Used by the legacy class-head spectral demixer to ignore padded
        tail frames.  Returns None when no padding mask is available."""
        if patch_padding_mask is None:
            return None
        t_p, f_p = grid_size
        B = patch_padding_mask.size(0)
        # patch_padding_mask: [B, T_p * F_p], True = padded (to ignore)
        pad_grid = patch_padding_mask.view(B, t_p, f_p)
        # A time step is valid iff at least one freq position is not padded.
        time_valid = ~pad_grid.all(dim=-1)
        return time_valid

    def compute_target_num_steps(
        self,
        waveform: Tensor,
        clip_duration_seconds: Optional[Tensor] = None,
    ) -> Tensor:
        """Infer the valid output length T_s_i for each sample in the batch.

        Args:
            waveform:
                [B, 4, T] waveform batch. Used as fallback when explicit clip
                durations are not provided.
            clip_duration_seconds:
                Optional [B] clip durations in seconds.

        Returns:
            Tensor:
                [B] valid number of output temporal steps for each sample after
                resampling, e.g. 25 for a 10-second sample at 2.5Hz.
        """
        clip_duration_seconds = self._infer_clip_duration_seconds(
            waveform=waveform,
            clip_duration_seconds=clip_duration_seconds,
        )
        rate = Fraction(str(self.cfg.target_token_rate)).limit_denominator(1000)
        num_samples = torch.round(clip_duration_seconds * self.cfg.sample_rate).long()
        numerator = num_samples * int(rate.numerator)
        denominator = int(self.cfg.sample_rate) * int(rate.denominator)
        target_steps = self._round_divide_half_to_even(numerator, denominator)
        return torch.clamp(target_steps, min=1)

    def extract_patch_tokens(
        self,
        waveform: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tuple[int, int]]:
        """Run the FOA front-end and patchify its output.

        Args:
            waveform:
                [B, 4, T] FOA waveform.

        Returns:
            Tuple[Tensor, Tensor, Tuple[int, int]]:
                foa_feat:
                    [B, 7, T_f, F] preprocessed FOA feature map.
                fused_feat:
                    [B, 1, T_f, F] W-only base feature map for the pretrained
                    patch path.
                delta_patch_tokens:
                    [B, N_p, D_in] additive patch-token deltas from the 7ch
                    spatial adapter.
                patch_tokens:
                    [B, N_p, D_in] summed patch tokens before BEATs trunk.
                grid_size:
                    (T_p, F_p) patch grid shape before flattening.
        """
        foa_feat = self.preprocessor(waveform)
        fused_feat = foa_feat[:, 0:1]
        base_patch_tokens, grid_size = self.patch_embedding(fused_feat)
        delta_patch_tokens, delta_grid_size = self.spatial_patch_adapter(foa_feat)
        if delta_grid_size != grid_size:
            raise ValueError(
                f"Delta patch grid {delta_grid_size} must match base grid {grid_size}"
            )
        if self.cfg.bypass_spatial_delta:
            # Pure W-channel path: trunk sees only pretrained-compatible tokens.
            # Spatial info is handled entirely by local_spatial_encoder post-trunk.
            patch_tokens = base_patch_tokens
        else:
            patch_tokens = base_patch_tokens + delta_patch_tokens
        return foa_feat, fused_feat, delta_patch_tokens, patch_tokens, grid_size

    def encode_patches(
        self,
        patch_tokens: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode patch tokens with the BEATs trunk.

        Args:
            patch_tokens:
                [B, N_p, D_in] flattened patch tokens.
            padding_mask:
                Optional mask aligned with the patch token sequence.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                encoder_memory:
                    [B, N_p, D] BEATs trunk output.
                padding_mask:
                    Optional mask after alignment to the encoded patch sequence.
        """
        features = self.layer_norm(patch_tokens)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        if self.trunk_spatial_adapters is not None:
            encoder_memory, padding_mask = self._encode_with_trunk_adapters(
                features, padding_mask
            )
        else:
            encoder_memory, _ = self.encoder(features, padding_mask=padding_mask)
        return encoder_memory, padding_mask

    def _encode_with_trunk_adapters(
        self,
        features: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Encode with per-layer spatial adapter injection.

        Mirrors ``TransformerEncoder.extract_features`` but inserts
        :class:`SpatialAdapterLayer` calls after each trunk layer.  This avoids
        modifying ``backbone.py`` while providing persistent spatial conditioning
        throughout the 12-layer self-attention stack.

        Only called when ``self.trunk_spatial_adapters is not None``.
        """
        import numpy as np  # used for layerdrop probability, matching backbone

        enc = self.encoder
        x = features  # [B, N_p, D]
        if padding_mask is not None:
            x = x.clone()
            x[padding_mask] = 0

        # positional convolution
        x_conv = enc.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        if not enc.layer_norm_first:
            x = enc.layer_norm(x)

        x = F.dropout(x, p=enc.dropout, training=self.training)
        x = x.transpose(0, 1)  # B,T,C → T,B,C

        pos_bias = None
        for i, layer in enumerate(enc.layers):
            if enc.layer_wise_gradient_decay_ratio != 1.0:
                from .modules import GradMultiply
                x = GradMultiply.apply(x, enc.layer_wise_gradient_decay_ratio)
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > enc.layerdrop):
                x, _, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    pos_bias=pos_bias,
                )
            # inject spatial adapter (if one exists for this layer)
            if (
                self.trunk_spatial_adapters is not None
                and i < len(self.trunk_spatial_adapters)
                and self.trunk_spatial_adapters[i] is not None
            ):
                x = self.trunk_spatial_adapters[i](x)

        if enc.layer_norm_first:
            x = enc.layer_norm(x)

        x = x.transpose(0, 1)  # T,B,C → B,T,C
        return x, padding_mask

    def _run_encoder_layers_after_pos_conv(
        self,
        features: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run BEATs encoder blocks after positional convolution is already applied.

        This mirrors ``TransformerEncoder.extract_features`` after its
        ``pos_conv`` step. It is used by the pre-trunk AST path so prefix task
        tokens can bypass BEATs' convolutional positional embedding while still
        participating in every self-attention block.
        """
        if padding_mask is not None:
            features = features.clone()
            features[padding_mask] = 0

        x = features
        if not self.encoder.layer_norm_first:
            x = self.encoder.layer_norm(x)

        x = F.dropout(x, p=self.encoder.dropout, training=self.training)
        x = x.transpose(0, 1)

        pos_bias = None
        for layer in self.encoder.layers:
            if self.encoder.training and self.encoder.layerdrop > 0.0:
                if torch.rand((), device=x.device).item() <= self.encoder.layerdrop:
                    continue
            x, _, pos_bias = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_weights=False,
                pos_bias=pos_bias,
            )

        x = x.transpose(0, 1)
        if self.encoder.layer_norm_first:
            x = self.encoder.layer_norm(x)
        return x

    def encode_patches_with_pretrunk_task_tokens(
        self,
        patch_tokens: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Encode patch tokens with Spatial-AST-style task tokens in the trunk.

        Token order:
            0: distance token
            1: DoA token
            2: class token

        Args:
            patch_tokens:
                [B, N_p, D_in] flattened patch tokens.
            padding_mask:
                Optional [B, N_p] patch padding mask.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]:
                encoder_memory:
                    [B, N_p, D] encoded patch tokens after removing the task
                    tokens from the trunk output.
                task_tokens:
                    [B, 3, D] encoded distance/DoA/class task tokens.
                padding_mask:
                    Original patch padding mask aligned with encoder_memory.
        """
        if self.pretrunk_task_tokens is None:
            raise RuntimeError("pretrunk_task_tokens are only available for readout_scheme='pretrunk_ast'.")
        features = self.layer_norm(patch_tokens)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)

        patch_features = features
        if padding_mask is not None:
            patch_features = patch_features.clone()
            patch_features[padding_mask] = 0
        patch_pos = self.encoder.pos_conv(patch_features.transpose(1, 2)).transpose(1, 2)
        patch_features = patch_features + patch_pos

        task_tokens = self.pretrunk_task_tokens.expand(features.size(0), -1, -1)
        features_with_tasks = torch.cat([task_tokens, patch_features], dim=1)
        task_padding_mask = torch.zeros(
            features.size(0),
            task_tokens.size(1),
            dtype=torch.bool,
            device=features.device,
        )
        if padding_mask is not None:
            padding_mask_with_tasks = torch.cat([task_padding_mask, padding_mask], dim=1)
        else:
            padding_mask_with_tasks = None
        encoded = self._run_encoder_layers_after_pos_conv(
            features=features_with_tasks,
            padding_mask=padding_mask_with_tasks,
        )
        encoded_task_tokens = encoded[:, : task_tokens.size(1)]
        encoder_memory = encoded[:, task_tokens.size(1) :]
        return encoder_memory, encoded_task_tokens, padding_mask

    def build_spatial_embeddings(
        self,
        encoder_memory: Tensor,
        grid_size: Tuple[int, int],
        waveform: Tensor,
        padding_mask: Optional[Tensor] = None,
        clip_duration_seconds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Convert BEATs patch outputs into the fixed-rate spatial sequence.

        Args:
            encoder_memory:
                [B, N_p, D] BEATs trunk output.
            grid_size:
                (T_p, F_p) patch grid used for reshape.
            waveform:
                [B, 4, T] waveform batch, used only to infer target duration if
                clip durations are not passed in explicitly.
            padding_mask:
                Optional mask aligned with encoder_memory.
            clip_duration_seconds:
                Optional [B] per-clip durations in seconds.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
                temporal_patch_tokens:
                    [B, T_p, D] after frequency pooling.
                temporal_tokens:
                    [B, T_s_max, D] after per-sample resampling and padding.
                spatial_embeddings:
                    [B, T_s_max, D] after shallow temporal readout.
                temporal_padding_mask:
                    Optional [B, T_s_max] mask where True marks padded time steps.
                target_num_steps:
                    [B] valid number of temporal steps for each sample.
        """
        temporal_patch_tokens = self.frequency_pool(encoder_memory, grid_size)
        target_num_steps = self.compute_target_num_steps(
            waveform=waveform,
            clip_duration_seconds=clip_duration_seconds,
        )
        temporal_tokens, temporal_padding_mask = self.temporal_resampler(
            temporal_patch_tokens,
            target_num_steps=target_num_steps,
        )
        spatial_embeddings = self.temporal_readout(
            temporal_tokens,
            padding_mask=temporal_padding_mask,
        )
        return (
            temporal_patch_tokens,
            temporal_tokens,
            spatial_embeddings,
            temporal_padding_mask,
            target_num_steps,
        )

    def decode_spatial_supervision(
        self,
        spatial_embeddings: Tensor,
    ) -> Tuple[Tensor, SpatialPredictionOutput]:
        """Build fixed-slot supervision outputs from the main spatial sequence.

        Args:
            spatial_embeddings:
                [B, T_s, D] main spatial embedding sequence.

        Returns:
            Tuple[Tensor, SpatialPredictionOutput]:
                slot_latents:
                    [B, T_s_max, K, H] fixed-slot supervision features.
                prediction_output:
                    Structured slot-level supervision outputs.
        """
        slot_latents = self.slot_readout(spatial_embeddings)
        prediction_output = self.prediction_heads(slot_latents)
        return slot_latents, prediction_output

    def decode_mono_ast_supervision(
        self,
        spatial_embeddings: Tensor,
        temporal_padding_mask: Optional[Tensor] = None,
        mono_window_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, MonoTaskPredictionOutput]:
        """Build single-source Spatial-AST-style task-token outputs.

        Args:
            spatial_embeddings:
                [B, T_s_max, D] main temporal spatial embedding sequence.
            temporal_padding_mask:
                Optional [B, T_s_max] padded-step mask.
            mono_window_mask:
                Optional [B, T_s_max] weak valid-time mask for the single source.

        Returns:
            Tuple[Tensor, MonoTaskPredictionOutput]:
                mono_task_tokens:
                    [B, 2, D] class/spatial task tokens.
                mono_prediction_output:
                    Structured single-source outputs.
        """
        mono_task_tokens = self.mono_task_readout(
            spatial_embeddings,
            padding_mask=temporal_padding_mask,
            active_window_mask=mono_window_mask,
        )
        mono_prediction_output = self.mono_prediction_heads(mono_task_tokens)
        return mono_task_tokens, mono_prediction_output

    def build_local_spatial_fusion(
        self,
        foa_feat: Tensor,
        semantic_embeddings: Tensor,
        target_num_steps: Tensor,
        temporal_padding_mask: Optional[Tensor] = None,
        mono_window_mask: Optional[Tensor] = None,
        pre_readout_tokens: Optional[Tensor] = None,
        return_local_pre_pool: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, MonoTaskPredictionOutput]:
        """Fuse BEATs semantic tokens with a local CNN/attention spatial branch.

        Args:
            foa_feat:
                [B, 7, T_f, F] full FOA + IV feature map.
            semantic_embeddings:
                [B, T_s_max, D] BEATs W-channel semantic temporal sequence.
            target_num_steps:
                [B] valid target lengths used to align local spatial tokens.
            temporal_padding_mask:
                Optional [B, T_s_max] padding mask after resampling.
            mono_window_mask:
                Optional [B, T_s_max] weak active-time mask for ov1 supervision.

        Returns:
            Tuple:
                local_spatial_tokens:
                    [B, T_s_max, D_s] resampled local spatial sequence.
                fused_embeddings:
                    [B, T_s_max, D] semantic + local-spatial fused sequence.
                mono_task_tokens:
                    [B, 2, D] attention-pooled class/spatial tokens.
                mono_prediction_output:
                    Single-source class/direction/distance predictions.
        """
        if (
            self.local_spatial_encoder is None
            or self.local_spatial_resampler is None
            or self.local_spatial_proj is None
            or self.local_spatial_fusion_norm is None
        ):
            raise RuntimeError("local_spatial modules are only available for readout_scheme='local_spatial'.")
        local_pre_pool_features: Optional[Tensor] = None
        local_pre_pool_grid: Optional[Tuple[int, int]] = None
        if self.cfg.bypass_local_fusion:
            # Classwarmup bypass: skip CNN branch entirely so fused = pure
            # BEATs semantic tokens.  local_spatial_encoder still exists in
            # the model but receives no gradient here.
            fused_embeddings = self.local_spatial_fusion_norm(semantic_embeddings)
            # Provide a dummy local_spatial_tokens for return value shape.
            local_spatial_tokens = torch.zeros(
                semantic_embeddings.size(0),
                semantic_embeddings.size(1),
                self.cfg.local_spatial_dim,
                device=semantic_embeddings.device,
                dtype=semantic_embeddings.dtype,
            )
            effective_padding_mask = temporal_padding_mask
        else:
            if return_local_pre_pool:
                local_patch_rate_tokens, local_cnn_features = self.local_spatial_encoder(
                    foa_feat, return_pre_pool=True
                )
                # [B, D_s, T_f, F_cnn] -> [B, T_f, F_cnn, D_s]
                b_, d_s_, t_f_, f_cnn_ = local_cnn_features.shape
                local_pre_pool_features = local_cnn_features.permute(0, 2, 3, 1).contiguous()
                local_pre_pool_features = local_pre_pool_features.view(
                    b_, t_f_ * f_cnn_, d_s_
                )
                local_pre_pool_grid = (int(t_f_), int(f_cnn_))
            else:
                local_patch_rate_tokens = self.local_spatial_encoder(foa_feat)
            local_spatial_tokens, local_padding_mask = self.local_spatial_resampler(
                local_patch_rate_tokens,
                target_num_steps=target_num_steps,
            )
            if local_spatial_tokens.shape[:2] != semantic_embeddings.shape[:2]:
                raise ValueError(
                    "Local spatial tokens must align with semantic embeddings, got "
                    f"{tuple(local_spatial_tokens.shape[:2])} vs {tuple(semantic_embeddings.shape[:2])}"
                )
            local_update = self.local_spatial_proj(local_spatial_tokens)
            fusion_padding_mask = (
                temporal_padding_mask
                if temporal_padding_mask is not None
                else local_padding_mask
            )
            if self.local_spatial_fuser is None:
                fused_pre_norm = semantic_embeddings + local_update
            else:
                fused_pre_norm = self.local_spatial_fuser(
                    semantic_embeddings=semantic_embeddings,
                    spatial_embeddings=local_update,
                    padding_mask=fusion_padding_mask,
                )
            fused_embeddings = self.local_spatial_fusion_norm(fused_pre_norm)
            effective_padding_mask = temporal_padding_mask
            if effective_padding_mask is None:
                effective_padding_mask = local_padding_mask
        if self.local_spatial_prediction_heads is None:
            # Clip-level aux head disabled (e.g. pure per-frame track path).
            if return_local_pre_pool:
                return (
                    local_spatial_tokens,
                    fused_embeddings,
                    None,
                    None,
                    local_pre_pool_features,
                    local_pre_pool_grid,
                )
            return local_spatial_tokens, fused_embeddings, None, None
        mono_task_tokens, mono_prediction_output = self.local_spatial_prediction_heads(
            fused_tokens=fused_embeddings,
            padding_mask=effective_padding_mask,
            active_window_mask=mono_window_mask,
            semantic_tokens=semantic_embeddings,
            pre_readout_tokens=pre_readout_tokens,
        )
        if return_local_pre_pool:
            return (
                local_spatial_tokens,
                fused_embeddings,
                mono_task_tokens,
                mono_prediction_output,
                local_pre_pool_features,
                local_pre_pool_grid,
            )
        return local_spatial_tokens, fused_embeddings, mono_task_tokens, mono_prediction_output

    def project_to_llm_tokens(self, spatial_embeddings: Tensor) -> Tensor:
        """Project main spatial embeddings into the LLM token space.

        Args:
            spatial_embeddings:
                [B, T_s_max, D] main spatial embedding sequence.

        Returns:
            Tensor:
                [B, T_s_max, d_llm] final spatial tokens for the LLM.
        """
        return self.projector(spatial_embeddings)

    def extract_features(
        self,
        waveform: Tensor,
        padding_mask: Optional[Tensor] = None,
        clip_duration_seconds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compatibility-style feature extraction entry point.

        Args:
            waveform:
                [B, 4, T] FOA waveform.
            padding_mask:
                Optional raw waveform-level or sequence-level mask.
            clip_duration_seconds:
                Optional [B] clip durations used to determine T_s.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                spatial_embeddings:
                    [B, T_s_max, D] encoder spatial embeddings before the projector.
                padding_mask:
                    Optional [B, T_s_max] mask aligned with the returned spatial
                    embedding sequence, where True marks padded steps.
        """
        duration_tensor = self._infer_clip_duration_seconds(
            waveform=waveform,
            padding_mask=padding_mask,
            clip_duration_seconds=clip_duration_seconds,
        )
        foa_feat, fused_feat, delta_patch_tokens, patch_tokens, grid_size = self.extract_patch_tokens(waveform)
        patch_padding_mask = self._build_patch_padding_mask(
            grid_size=grid_size,
            clip_duration_seconds=duration_tensor,
            device=waveform.device,
        )

        if self.cfg.readout_scheme == "pretrunk_ast":
            encoder_memory, _, _ = self.encode_patches_with_pretrunk_task_tokens(
                patch_tokens=patch_tokens,
                padding_mask=patch_padding_mask,
            )
        else:
            encoder_memory, _ = self.encode_patches(
                patch_tokens=patch_tokens,
                padding_mask=patch_padding_mask,
            )
        _, _, spatial_embeddings, temporal_padding_mask, target_num_steps = self.build_spatial_embeddings(
            encoder_memory=encoder_memory,
            grid_size=grid_size,
            waveform=waveform,
            padding_mask=patch_padding_mask,
            clip_duration_seconds=duration_tensor,
        )
        if (
            self.cfg.readout_scheme == "local_spatial"
            or self.cfg.readout_scheme in LOCAL_SPATIAL_FRAME_SCHEMES
        ):
            _, spatial_embeddings, _, _ = self.build_local_spatial_fusion(
                foa_feat=foa_feat,
                semantic_embeddings=spatial_embeddings,
                target_num_steps=target_num_steps,
                temporal_padding_mask=temporal_padding_mask,
            )
        return spatial_embeddings, temporal_padding_mask

    def forward(
        self,
        waveform: Tensor,
        padding_mask: Optional[Tensor] = None,
        clip_duration_seconds: Optional[Tensor] = None,
        mono_window_mask: Optional[Tensor] = None,
    ) -> SOBackboneOutput:
        """Forward pass for the simplified Spatial-BEATs encoder.

        Args:
            waveform:
                [B, 4, T] FOA waveform batch.
            padding_mask:
                Optional mask carried through the sequence stages.
            clip_duration_seconds:
                Optional [B] durations in seconds used to determine the final
                per-sample temporal lengths T_s_i.

        Returns:
            SOBackboneOutput:
                Structured object containing all major intermediate tensors and
                final outputs needed for supervision and LLM projection.
        """
        duration_tensor = self._infer_clip_duration_seconds(
            waveform=waveform,
            padding_mask=padding_mask,
            clip_duration_seconds=clip_duration_seconds,
        )
        foa_feat, fused_feat, delta_patch_tokens, patch_tokens, grid_size = self.extract_patch_tokens(waveform)
        patch_padding_mask = self._build_patch_padding_mask(
            grid_size=grid_size,
            clip_duration_seconds=duration_tensor,
            device=waveform.device,
        )

        pretrunk_task_tokens: Optional[Tensor] = None
        if self.cfg.readout_scheme == "pretrunk_ast":
            encoder_memory, pretrunk_task_tokens, _ = self.encode_patches_with_pretrunk_task_tokens(
                patch_tokens=patch_tokens,
                padding_mask=patch_padding_mask,
            )
        else:
            encoder_memory, _ = self.encode_patches(
                patch_tokens=patch_tokens,
                padding_mask=patch_padding_mask,
            )
        (
            temporal_patch_tokens,
            temporal_tokens,
            spatial_embeddings,
            temporal_padding_mask,
            target_num_steps,
        ) = self.build_spatial_embeddings(
            encoder_memory=encoder_memory,
            grid_size=grid_size,
            waveform=waveform,
            padding_mask=patch_padding_mask,
            clip_duration_seconds=duration_tensor,
        )
        slot_latents: Optional[Tensor] = None
        prediction_output: Optional[SpatialPredictionOutput] = None
        mono_task_tokens: Optional[Tensor] = None
        mono_prediction_output: Optional[MonoTaskPredictionOutput] = None
        local_spatial_tokens: Optional[Tensor] = None
        fused_spatial_embeddings: Optional[Tensor] = None
        pretrunk_prediction_output: Optional[PreTrunkASTPredictionOutput] = None
        frame_slot_prediction_output: Optional[FrameSlotPredictionOutput] = None
        frame_track_prediction_output: Optional[FrameTrackPredictionOutput] = None
        frame_accdoa_prediction_output: Optional[FrameACCDOAPredictionOutput] = None
        frame_wise_prediction_output: Optional[FrameWisePredictionOutput] = None
        frame_clip_aux_output: Optional[MonoTaskPredictionOutput] = None
        if self.cfg.readout_scheme == "fixed_slot":
            slot_latents, prediction_output = self.decode_spatial_supervision(spatial_embeddings)
        elif self.cfg.readout_scheme == "mono_ast":
            mono_task_tokens, mono_prediction_output = self.decode_mono_ast_supervision(
                spatial_embeddings=spatial_embeddings,
                temporal_padding_mask=temporal_padding_mask,
                mono_window_mask=mono_window_mask,
            )
        elif self.cfg.readout_scheme == "local_spatial":
            _need_local_pre_pool = bool(
                getattr(self.cfg, "use_spatial_head_demixer", False)
                and getattr(self.cfg, "spatial_demixer_use_local_spatial_kv", False)
                and self.local_spatial_pre_pool_proj is not None
            )
            _local_fusion_out = self.build_local_spatial_fusion(
                foa_feat=foa_feat,
                semantic_embeddings=spatial_embeddings,
                target_num_steps=target_num_steps,
                temporal_padding_mask=temporal_padding_mask,
                mono_window_mask=mono_window_mask,
                pre_readout_tokens=encoder_memory,
                return_local_pre_pool=_need_local_pre_pool,
            )
            if _need_local_pre_pool:
                (
                    local_spatial_tokens,
                    fused_spatial_embeddings,
                    mono_task_tokens,
                    mono_prediction_output,
                    _local_pre_pool_features,
                    _local_pre_pool_grid,
                ) = _local_fusion_out
            else:
                (
                    local_spatial_tokens,
                    fused_spatial_embeddings,
                    mono_task_tokens,
                    mono_prediction_output,
                ) = _local_fusion_out
                _local_pre_pool_features = None
                _local_pre_pool_grid = None
            spatial_embeddings = fused_spatial_embeddings
            # Optional frame-level track head (parallel with clip-level head).
            if (
                self.source_query_decoder is not None
                and self.frame_track_prediction_heads is not None
            ):
                track_time_features, track_latents = self.source_query_decoder(
                    fused=fused_spatial_embeddings,
                    padding_mask=temporal_padding_mask,
                )
                # optional track-wise refinement (zero-init residual)
                if self.track_refinement_decoder is not None:
                    track_time_features = self.track_refinement_decoder(
                        track_tokens=track_time_features,
                        memory=fused_spatial_embeddings,
                    )
                _pre_pool_time_mask = self._derive_pre_pool_time_mask(
                    patch_padding_mask=patch_padding_mask,
                    grid_size=grid_size,
                )
                _spatial_kv = None
                _spatial_kv_grid = None
                if (
                    _need_local_pre_pool
                    and _local_pre_pool_features is not None
                    and _local_pre_pool_grid is not None
                ):
                    _spatial_kv = self.local_spatial_pre_pool_proj(_local_pre_pool_features)
                    _spatial_kv_grid = _local_pre_pool_grid
                frame_track_prediction_output = self.frame_track_prediction_heads(
                    track_time_features=track_time_features,
                    track_latents=track_latents,
                    pre_pool_features=encoder_memory,
                    pre_pool_grid_size=grid_size,
                    pre_pool_time_mask=_pre_pool_time_mask,
                    spatial_pre_pool_features=_spatial_kv,
                    spatial_pre_pool_grid_size=_spatial_kv_grid,
                    spatial_pre_pool_time_mask=None,
                )
        elif self.cfg.readout_scheme == "pretrunk_ast":
            if pretrunk_task_tokens is None or self.pretrunk_prediction_heads is None:
                raise RuntimeError("pretrunk_ast requires pretrunk task tokens and prediction heads.")
            pretrunk_prediction_output = self.pretrunk_prediction_heads(pretrunk_task_tokens)
        elif self.cfg.readout_scheme in LOCAL_SPATIAL_FRAME_SCHEMES:
            # Shared local_spatial fusion: build the fused sequence and reuse
            # the existing clip-level single-source supervision as an auxiliary
            # head (so ov1 warmup weights stay directly useful).
            _need_local_pre_pool = bool(
                self.cfg.readout_scheme == "local_spatial_track"
                and getattr(self.cfg, "use_spatial_head_demixer", False)
                and getattr(self.cfg, "spatial_demixer_use_local_spatial_kv", False)
                and self.local_spatial_pre_pool_proj is not None
            )
            _local_fusion_out = self.build_local_spatial_fusion(
                foa_feat=foa_feat,
                semantic_embeddings=spatial_embeddings,
                target_num_steps=target_num_steps,
                temporal_padding_mask=temporal_padding_mask,
                mono_window_mask=mono_window_mask,
                pre_readout_tokens=encoder_memory,
                return_local_pre_pool=_need_local_pre_pool,
            )
            if _need_local_pre_pool:
                (
                    local_spatial_tokens,
                    fused_spatial_embeddings,
                    mono_task_tokens,
                    mono_prediction_output,
                    _local_pre_pool_features,
                    _local_pre_pool_grid,
                ) = _local_fusion_out
            else:
                (
                    local_spatial_tokens,
                    fused_spatial_embeddings,
                    mono_task_tokens,
                    mono_prediction_output,
                ) = _local_fusion_out
                _local_pre_pool_features = None
                _local_pre_pool_grid = None
            spatial_embeddings = fused_spatial_embeddings
            # Move the clip-level head output into clip_aux_prediction_output so
            # the existing ``mono_prediction_output`` field remains reserved for
            # the ov1 single-source supervision path.
            frame_clip_aux_output: Optional[MonoTaskPredictionOutput] = mono_prediction_output
            mono_prediction_output = None
            if self.cfg.readout_scheme == "local_spatial_slot":
                if self.frame_slot_head is None:
                    raise RuntimeError("local_spatial_slot requires frame_slot_head.")
                frame_slot_prediction_output = self.frame_slot_head(
                    fused=fused_spatial_embeddings,
                    padding_mask=temporal_padding_mask,
                )
                frame_track_prediction_output = None
                frame_accdoa_prediction_output = None
            elif self.cfg.readout_scheme == "local_spatial_track":
                if self.source_query_decoder is None or self.frame_track_prediction_heads is None:
                    raise RuntimeError("local_spatial_track requires source_query_decoder and frame_track_prediction_heads.")
                track_time_features, track_latents = self.source_query_decoder(
                    fused=fused_spatial_embeddings,
                    padding_mask=temporal_padding_mask,
                )
                # optional track-wise refinement (zero-init residual)
                if self.track_refinement_decoder is not None:
                    track_time_features = self.track_refinement_decoder(
                        track_tokens=track_time_features,
                        memory=fused_spatial_embeddings,
                    )
                _pre_pool_time_mask = self._derive_pre_pool_time_mask(
                    patch_padding_mask=patch_padding_mask,
                    grid_size=grid_size,
                )
                _spatial_kv = None
                _spatial_kv_grid = None
                if (
                    _need_local_pre_pool
                    and _local_pre_pool_features is not None
                    and _local_pre_pool_grid is not None
                ):
                    _spatial_kv = self.local_spatial_pre_pool_proj(_local_pre_pool_features)
                    _spatial_kv_grid = _local_pre_pool_grid
                frame_track_prediction_output = self.frame_track_prediction_heads(
                    track_time_features=track_time_features,
                    track_latents=track_latents,
                    pre_pool_features=encoder_memory,
                    pre_pool_grid_size=grid_size,
                    pre_pool_time_mask=_pre_pool_time_mask,
                    spatial_pre_pool_features=_spatial_kv,
                    spatial_pre_pool_grid_size=_spatial_kv_grid,
                    spatial_pre_pool_time_mask=None,
                )
                frame_slot_prediction_output = None
                frame_accdoa_prediction_output = None
            elif self.cfg.readout_scheme == "local_spatial_accdoa":
                if self.accdoa_heads is None:
                    raise RuntimeError("local_spatial_accdoa requires accdoa_heads.")
                frame_accdoa_prediction_output = self.accdoa_heads(fused=fused_spatial_embeddings)
                frame_slot_prediction_output = None
                frame_track_prediction_output = None
            elif self.cfg.readout_scheme == "local_spatial_framewise":
                if self.frame_wise_heads is None:
                    raise RuntimeError("local_spatial_framewise requires frame_wise_heads.")
                frame_wise_prediction_output = self.frame_wise_heads(
                    fused=fused_spatial_embeddings,
                    semantic_tokens=spatial_embeddings,  # BEATs trunk 输出 [B,T,768]
                )
                frame_slot_prediction_output = None
                frame_track_prediction_output = None
                frame_accdoa_prediction_output = None
        else:
            raise ValueError(f"Unsupported readout_scheme: {self.cfg.readout_scheme}")
        llm_spatial_tokens = self.project_to_llm_tokens(spatial_embeddings)
        return SOBackboneOutput(
            foa_feat=foa_feat,
            fused_feat=fused_feat,
            delta_patch_tokens=delta_patch_tokens,
            patch_tokens=patch_tokens,
            grid_size=grid_size,
            encoder_memory=encoder_memory,
            temporal_patch_tokens=temporal_patch_tokens,
            temporal_tokens=temporal_tokens,
            spatial_embeddings=spatial_embeddings,
            slot_latents=slot_latents,
            prediction_output=prediction_output,
            mono_task_tokens=mono_task_tokens,
            mono_prediction_output=mono_prediction_output,
            local_spatial_tokens=local_spatial_tokens,
            fused_spatial_embeddings=fused_spatial_embeddings,
            pretrunk_task_tokens=pretrunk_task_tokens,
            pretrunk_prediction_output=pretrunk_prediction_output,
            llm_spatial_tokens=llm_spatial_tokens,
            temporal_padding_mask=temporal_padding_mask,
            target_num_steps=target_num_steps,
            frame_slot_prediction_output=frame_slot_prediction_output,
            frame_track_prediction_output=frame_track_prediction_output,
            frame_accdoa_prediction_output=frame_accdoa_prediction_output,
            frame_wise_prediction_output=frame_wise_prediction_output,
            clip_aux_prediction_output=frame_clip_aux_output,
        )
