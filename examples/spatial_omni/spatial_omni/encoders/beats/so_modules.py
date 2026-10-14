# Adapted from Spatial-Omni for self-contained inference in Auden.
"""Spatial feature extraction and projection modules for SO-Encoder."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math

try:
    import torchaudio.compliance.kaldi as ta_kaldi
except ImportError:
    ta_kaldi = None  # type: ignore[assignment]


@dataclass
class SpatialPredictionOutput:
    """Structured outputs from the fixed-slot supervision heads.

    Attributes:
        pred_activity:
            [B, T_s_max, K] weak activity / objectness logits for each time step
            and fixed source slot.
        pred_azi_logits:
            [B, T_s_max, K, num_azi_bins] azimuth classification logits.
        pred_ele_logits:
            [B, T_s_max, K, num_ele_bins] elevation classification logits.
        pred_dist:
            [B, T_s_max, K, 1] continuous distance regression output.
        pred_class_logits:
            [B, T_s_max, K, num_classes] auxiliary source class logits.
    """

    pred_activity: Tensor
    pred_azi_logits: Tensor
    pred_ele_logits: Tensor
    pred_dist: Tensor
    pred_class_logits: Tensor


@dataclass
class MonoTaskPredictionOutput:
    """Structured outputs from the single-source Spatial-AST-style heads.

    Attributes:
        pred_class_logits:
            [B, num_classes] clip/source-level class logits read from the
            dedicated class task token.
        pred_direction:
            [B, 3] normalized 3D direction vectors predicted from the spatial
            task token in Cartesian coordinates [x, y, z].
        pred_distance:
            [B, 1] continuous source distance prediction in meters.
        sem_class_logits:
            Optional [B, num_classes] class logits from the semantic anchor
            head (mean-pooled pre-fusion BEATs tokens). Training-only
            auxiliary signal to prevent the trunk from forgetting semantics
            under spatial gradient pressure. Not used for evaluation or LLM.
    """

    pred_class_logits: Tensor
    pred_direction: Tensor
    pred_distance: Tensor
    sem_class_logits: Optional[Tensor] = None


@dataclass
class PreTrunkASTPredictionOutput:
    """Structured outputs from Spatial-AST-style pre-trunk task tokens.

    Attributes:
        pred_class_logits:
            [B, num_classes] class logits from the class task token.
        pred_distance_logits:
            [B, num_distance_bins] distance classification logits.
        pred_azi_logits:
            [B, num_azi_bins] azimuth classification logits.
        pred_ele_logits:
            [B, num_ele_bins] elevation classification logits.
    """

    pred_class_logits: Tensor
    pred_distance_logits: Tensor
    pred_azi_logits: Tensor
    pred_ele_logits: Tensor


class SOBackbonePreprocessor(nn.Module):
    """Convert FOA waveform into multi-channel spatial feature maps.

    The default mel front-end settings are intentionally aligned with the
    Qwen-2.5-Omni audio tower / Whisper-style feature extractor at the low-level
    acoustic parameter level:
        - sample_rate = 16000
        - num_mel_bins = 128
        - n_fft = 400
        - win_length = 400
        - hop_length = 160
        - dither = 0.0

    Only the low-level mel parameters are aligned. The overall audio tower
    architecture is not copied, because Spatial-BEATs needs a FOA-specific
    front-end and a BEATs trunk.

    Expected feature channels:
        0: W_logmel
        1: X_logmel
        2: Y_logmel
        3: Z_logmel
        4: IVx
        5: IVy
        6: IVz

    Shape contract:
        Input:
            waveform: [B, 4, T]
                B: batch size
                4: FOA channels stored in DCASE-style waveform order
                   [W, Y, Z, X]. The preprocessor reorders them internally to
                   [W, X, Y, Z] before computing log-mel and intensity cues.
                T: number of waveform samples, e.g. 160000 for 10s @ 16kHz
        Output:
            foa_feat: [B, 7, T_f, num_mel_bins]
                7: feature channels listed above
                T_f: acoustic frame count before patchification
                num_mel_bins: default 128
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_mel_bins: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        dither: float = 0.0,
        waveform_scale: float = float(2**15),
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        normalize_logmel: bool = True,
        use_kaldi_w_channel: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.dither = dither
        self.waveform_scale = float(waveform_scale)
        self.fbank_mean = float(fbank_mean)
        self.fbank_std = float(fbank_std)
        self.normalize_logmel = bool(normalize_logmel)
        self.use_kaldi_w_channel = bool(use_kaldi_w_channel)

        # SpecAugment: time and frequency masking applied to the W channel
        # logmel only (channels 1-6 carry spatial physics that should not be
        # masked).  Defaults to off (0, 0) to preserve existing behavior.
        self.spec_augment_freq_masks: int = 0
        self.spec_augment_freq_width: int = 0
        self.spec_augment_time_masks: int = 0
        self.spec_augment_time_width: int = 0
        self.register_buffer(
            "mel_filterbank",
            self._build_mel_filterbank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                num_mel_bins=num_mel_bins,
            ),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )

    @staticmethod
    def _reorder_dcase_wyzx_to_wxyz(waveform: Tensor) -> Tensor:
        """Convert stored DCASE FOA order [W, Y, Z, X] to internal [W, X, Y, Z].

        The simulation pipeline writes FOA as listener-local DCASE order
        [W, Y, Z, X]. The Spatial-BEATs front-end computes spatial cues in a
        canonical internal order [W, X, Y, Z], so waveform channels must be
        permuted before STFT and IV extraction.
        """
        return waveform[:, [0, 3, 1, 2], :]

    @staticmethod
    def _hz_to_mel(freq_hz: Tensor) -> Tensor:
        return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)

    @staticmethod
    def _mel_to_hz(freq_mel: Tensor) -> Tensor:
        return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)

    def _build_mel_filterbank(
        self,
        sample_rate: int,
        n_fft: int,
        num_mel_bins: int,
    ) -> Tensor:
        num_freqs = n_fft // 2 + 1
        min_mel = self._hz_to_mel(torch.tensor(0.0))
        max_mel = self._hz_to_mel(torch.tensor(float(sample_rate) / 2.0))
        mel_points = torch.linspace(min_mel, max_mel, num_mel_bins + 2)
        hz_points = self._mel_to_hz(mel_points)
        fft_freqs = torch.linspace(0.0, float(sample_rate) / 2.0, num_freqs)

        fbanks = torch.zeros(num_freqs, num_mel_bins)
        for i in range(num_mel_bins):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            up_slope = (fft_freqs - left) / (center - left + 1e-8)
            down_slope = (right - fft_freqs) / (right - center + 1e-8)
            fbanks[:, i] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

        return fbanks

    def _compute_channel_logmel(self, waveform: Tensor) -> Tensor:
        batch, channels, num_samples = waveform.shape
        x = waveform.reshape(batch * channels, num_samples) * self.waveform_scale
        window = self.window.to(dtype=waveform.dtype, device=waveform.device)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        mel_filterbank = self.mel_filterbank.to(dtype=power.dtype, device=power.device)
        mel = torch.matmul(power.transpose(1, 2), mel_filterbank).transpose(1, 2)
        logmel = torch.log(torch.clamp(mel, min=1e-10))
        if self.normalize_logmel:
            logmel = (logmel - self.fbank_mean) / (2.0 * self.fbank_std)
        logmel = logmel.reshape(batch, channels, self.num_mel_bins, -1).transpose(2, 3)
        return logmel, stft

    def _apply_spec_augment_w(self, w_logmel: Tensor) -> Tensor:
        """Apply SpecAugment-style masking to the W channel logmel only.

        Only active during ``self.training``.  Other channels (XYZ logmel, IV)
        are never masked because they encode spatial physics.

        Args:
            w_logmel: [B, 1, T_f, F] W-channel logmel spectrogram.

        Returns:
            [B, 1, T_f, F] with random time/frequency bands zeroed out.
        """
        if not self.training:
            return w_logmel

        x = w_logmel.clone()
        _, _, T, F = x.shape

        for _ in range(self.spec_augment_freq_masks):
            f = torch.randint(0, max(1, self.spec_augment_freq_width + 1), ()).item()
            f0 = torch.randint(0, max(1, F - f), ()).item()
            x[:, :, :, f0 : f0 + f] = 0.0

        for _ in range(self.spec_augment_time_masks):
            t = torch.randint(0, max(1, self.spec_augment_time_width + 1), ()).item()
            t0 = torch.randint(0, max(1, T - t), ()).item()
            x[:, :, t0 : t0 + t, :] = 0.0

        return x

    def _compute_kaldi_fbank_w(self, w_waveform: Tensor) -> Tensor:
        """Compute Kaldi-style fbank for the W channel only.

        This replicates the original ``BEATs.preprocess()`` path so the
        pretrained BEATs trunk sees the same spectral distribution it was
        trained on.  Only used when ``use_kaldi_w_channel=True``.

        Args:
            w_waveform: [B, 1, T] W-channel waveform (already reordered).

        Returns:
            [B, 1, T_f, num_mel_bins] Kaldi fbank features, normalized with
            the BEATs mean/std statistics.
        """
        if ta_kaldi is None:
            raise ImportError(
                "torchaudio is required for Kaldi fbank computation. "
                "Install it with: pip install torchaudio"
            )
        batch = w_waveform.size(0)
        fbanks = []
        for i in range(batch):
            waveform_i = w_waveform[i, 0] * self.waveform_scale  # [T]
            fbank_i = ta_kaldi.fbank(
                waveform_i.unsqueeze(0),
                num_mel_bins=self.num_mel_bins,
                sample_frequency=float(self.sample_rate),
                frame_length=self.frame_length_ms,
                frame_shift=self.frame_shift_ms,
            )  # [T_f, num_mel_bins]
            fbanks.append(fbank_i)
        fbank = torch.stack(fbanks, dim=0)  # [B, T_f, num_mel_bins]
        fbank = (fbank - self.fbank_mean) / (2.0 * self.fbank_std)
        return fbank.unsqueeze(1)  # [B, 1, T_f, num_mel_bins]

    def _compute_intensity_features(self, stft: Tensor, batch_size: int) -> Tensor:
        """Mel-projected normalized intensity vector (W-power normalization).

        Standard DCASE / Spatial-AST normalization:
            IV(t, f_mel) = Re(W · conj(XYZ))(t, f_mel) / |W(t, f_mel)|^2

        The old per-axis max-over-mel normalization was a bug — it divided
        IVx, IVy, IVz independently by their own per-axis maxima, which
        destroyed the relative magnitudes between axes and therefore the
        DOA direction the model was supposed to read off.  The W-power
        normalization here divides all three axes by the same |W|^2
        scalar at each (t, f_mel) bin, preserving the IV vector's
        direction.  For a single source this gives a unit-vector cue
        scaled by directional cosine; for a diffuse field it goes to 0.
        """
        stft = stft.reshape(batch_size, 4, stft.size(-2), stft.size(-1))
        w = stft[:, 0]
        xyz = stft[:, 1:]
        active_intensity = torch.real(w.unsqueeze(1) * torch.conj(xyz))  # [B, 3, F, T_f]
        w_power = (w.conj() * w).real  # [B, F, T_f] = |W(t,f)|^2

        mel_filterbank = self.mel_filterbank.to(
            dtype=active_intensity.dtype, device=active_intensity.device
        )
        # Project intensity onto mel: [B, 3, F, T_f] -> [B, 3, T_f, F_mel]
        iv = torch.matmul(active_intensity.transpose(-2, -1), mel_filterbank).transpose(-2, -1)
        iv = iv.transpose(2, 3)
        # Project W power onto mel: [B, F, T_f] -> [B, T_f, F_mel]
        w_power_mel = torch.matmul(w_power.transpose(-2, -1), mel_filterbank)
        # Per-(t, f_mel) divide all three axes by the same W-power scalar
        # so the IV vector direction is preserved.
        iv = iv / w_power_mel.unsqueeze(1).clamp_min(1e-6)  # [B, 3, T_f, F_mel]
        # Clamp to avoid extreme values in low-energy (silent) TF bins where
        # W-power → 0.  p50 ≈ 0.4, p99 ≈ 3.4 on real data; values beyond
        # ±10 are pure numerical noise from near-zero denominators.
        iv = iv.clamp(-10.0, 10.0)
        return iv

    def forward(self, waveform: Tensor) -> Tensor:
        """Build FOA spatial features from waveform.

        Args:
            waveform:
                [B, 4, T] FOA waveform.

        Returns:
            Tensor:
                [B, 7, T_f, num_mel_bins] multi-channel spatial feature map.
        """
        if waveform.ndim != 3 or waveform.size(1) != 4:
            raise ValueError(
                f"Expected waveform with shape [B, 4, T], got {tuple(waveform.shape)}"
            )

        waveform = self._reorder_dcase_wyzx_to_wxyz(waveform)
        logmel, stft = self._compute_channel_logmel(waveform)
        iv = self._compute_intensity_features(stft, batch_size=waveform.size(0))

        if self.use_kaldi_w_channel:
            kaldi_w = self._compute_kaldi_fbank_w(waveform[:, 0:1])
            # Align frame count: Kaldi and custom STFT both use hop=160 but
            # may differ by ±1 frame due to padding/centering differences.
            # logmel: [B, C, T_f, F], kaldi_w: [B, 1, T_f', F]
            kaldi_T = kaldi_w.size(2)
            custom_T = logmel.size(2)
            if kaldi_T != custom_T:
                min_T = min(kaldi_T, custom_T)
                kaldi_w = kaldi_w[:, :, :min_T, :]
                logmel = logmel[:, :, :min_T, :]
                iv = iv[:, :, :min_T, :]
            w_channel = kaldi_w
        else:
            w_channel = logmel[:, 0:1]

        # SpecAugment on W channel only (training mode, when configured).
        if self.spec_augment_freq_masks > 0 or self.spec_augment_time_masks > 0:
            w_channel = self._apply_spec_augment_w(w_channel)

        foa_feat = torch.cat(
            [
                w_channel,
                logmel[:, 1:2],
                logmel[:, 2:3],
                logmel[:, 3:4],
                iv[:, 0:1],
                iv[:, 1:2],
                iv[:, 2:3],
            ],
            dim=1,
        )
        return foa_feat.contiguous()


class SpatialDeltaPatchAdapter(nn.Module):
    """Build additive patch-token deltas from the full 7-channel FOA feature map.

    This module replaces the previous early 7ch -> 1ch fusion bottleneck.
    The pretrained BEATs patch path now stays:

        W_logmel -> original pretrained single-channel patch embedding

    while the spatial branch produces a residual token update:

        7ch FOA feature map -> spatial adapter -> delta patch tokens

    The two are summed before entering the BEATs trunk:

        patch_tokens = patch_tokens_from_W + delta_patch_tokens

    Shape contract:
        Input:
            foa_feat: [B, 7, T_f, F]
        Output:
            delta_patch_tokens: [B, N_p, embed_dim]
            grid_size: (T_p, F_p)
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 32,
        embed_dim: int = 512,
        patch_size: Tuple[int, int] = (16, 16),
        residual_scale_init: float = 0.1,
        out_proj_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.residual_scale_init = residual_scale_init
        self.out_proj_scale_init = out_proj_scale_init
        self.pre_mix = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.depthwise_tf = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )
        self.patch_proj = nn.Conv2d(
            hidden_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.activation = nn.GELU()
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the spatial delta path near zero but fully trainable."""
        nn.init.kaiming_uniform_(self.pre_mix.weight, a=5 ** 0.5)
        if self.pre_mix.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pre_mix.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0.0
            nn.init.uniform_(self.pre_mix.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.depthwise_tf.weight, a=5 ** 0.5)
        if self.depthwise_tf.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise_tf.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0.0
            nn.init.uniform_(self.depthwise_tf.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.patch_proj.weight, a=5 ** 0.5)
        self.patch_proj.weight.data.mul_(self.out_proj_scale_init)

    def forward(self, foa_feat: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Produce additive patch-token deltas from the 7-channel FOA feature map."""
        if foa_feat.ndim != 4 or foa_feat.size(1) != self.in_channels:
            raise ValueError(
                f"Expected foa_feat with shape [B, {self.in_channels}, T_f, F], got {tuple(foa_feat.shape)}"
            )
        mixed = self.pre_mix(foa_feat)
        mixed = self.activation(mixed)
        mixed = self.depthwise_tf(mixed)
        mixed = self.activation(mixed)
        delta_grid = self.patch_proj(mixed)
        t_p, f_p = delta_grid.shape[-2], delta_grid.shape[-1]
        delta_patch_tokens = delta_grid.flatten(2).transpose(1, 2).contiguous()
        delta_patch_tokens = self.residual_alpha * delta_patch_tokens
        return delta_patch_tokens, (t_p, f_p)


class SpatialPatchEmbedding(nn.Module):
    """Patchify the fused single-channel spatial feature map.

    The FOA channels are fused before this point by SpatialChannelMixer so that
    the original pretrained single-channel BEATs patch embedding can be reused
    directly.

    Shape contract:
        Input:
            fused_feat: [B, 1, T_f, F]
        Intermediate:
            patch_grid: [B, embed_dim, T_p, F_p]
        Output:
            patch_tokens: [B, N_p, embed_dim], where N_p = T_p * F_p
            grid_size: (T_p, F_p)
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        patch_size: Tuple[int, int] = (16, 16),
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.bias = bias
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, fused_feat: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Patchify spatial features.

        Args:
            fused_feat:
                [B, 1, T_f, F] fused spatial feature map.

        Returns:
            Tuple[Tensor, Tuple[int, int]]:
                patch_tokens:
                    [B, N_p, embed_dim] flattened patch tokens.
                grid_size:
                    (T_p, F_p) patch grid before flattening.
        """
        if fused_feat.ndim != 4 or fused_feat.size(1) != self.in_channels:
            raise ValueError(
                f"Expected fused_feat with shape [B, {self.in_channels}, T_f, F], got {tuple(fused_feat.shape)}"
            )
        patch_grid = self.proj(fused_feat)
        t_p, f_p = patch_grid.shape[-2], patch_grid.shape[-1]
        patch_tokens = patch_grid.flatten(2).transpose(1, 2).contiguous()
        return patch_tokens, (t_p, f_p)


class FrequencyPool(nn.Module):
    """Collapse the frequency axis of the BEATs patch grid.

    Shape contract:
        Input:
            encoder_memory: [B, N_p, D]
            grid_size: (T_p, F_p)
        Intermediate:
            grid_memory: [B, T_p, F_p, D]
        Output:
            temporal_patch_tokens: [B, T_p, D]
    """

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, encoder_memory: Tensor, grid_size: Tuple[int, int]) -> Tensor:
        """Pool BEATs patch features along the frequency dimension.

        Args:
            encoder_memory:
                [B, N_p, D] BEATs encoder output over flattened patches.
            grid_size:
                (T_p, F_p) patch grid shape used to reshape N_p back to 2D.

        Returns:
            Tensor:
                [B, T_p, D] temporal patch sequence before resampling.
        """
        t_p, f_p = grid_size
        batch_size, num_patches, embed_dim = encoder_memory.shape
        if num_patches != t_p * f_p:
            raise ValueError(
                f"grid_size {grid_size} incompatible with num_patches={num_patches}"
            )
        grid_memory = encoder_memory.reshape(batch_size, t_p, f_p, embed_dim)
        if self.mode == "mean":
            return grid_memory.mean(dim=2)
        if self.mode == "max":
            return grid_memory.max(dim=2).values
        raise ValueError(f"Unsupported frequency pooling mode: {self.mode}")


class TemporalResampler(nn.Module):
    """Resample patch-rate temporal tokens to the target token rate.

    Shape contract:
        Input:
            temporal_patch_tokens: [B, T_p, D]
        Output:
            temporal_tokens: [B, T_s_max, D]
            temporal_padding_mask: [B, T_s_max]

    Notes:
        Each sample i has its own valid temporal length:
            T_s_i = round(duration_i * target_token_rate)
        Within one batch, the resampled sequences are padded to:
            T_s_max = max_i T_s_i
        For a 10-second sample at 2.5Hz, T_s_i = 25.
    """

    def __init__(self, target_token_rate: float = 2.5, mode: str = "linear") -> None:
        super().__init__()
        self.target_token_rate = target_token_rate
        self.mode = mode

    def forward(
        self,
        temporal_patch_tokens: Tensor,
        target_num_steps: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Resample temporal tokens to a fixed number of steps.

        Args:
            temporal_patch_tokens:
                [B, T_p, D] patch-rate temporal sequence.
            target_num_steps:
                [B] target time lengths for each sample after resampling,
                e.g. 25 for a 10-second sample at 2.5Hz.

        Returns:
            Tuple[Tensor, Tensor]:
                temporal_tokens:
                    [B, T_s_max, D] resampled temporal tokens padded to the
                    max valid length in the batch.
                temporal_padding_mask:
                    [B, T_s_max] boolean mask where True marks padded steps.
        """
        if target_num_steps.ndim != 1:
            raise ValueError("target_num_steps must have shape [B]")
        batch_size, _, embed_dim = temporal_patch_tokens.shape
        if target_num_steps.numel() != batch_size:
            raise ValueError("target_num_steps length must match batch size")

        max_steps = int(target_num_steps.max().item())
        max_steps = max(max_steps, 1)
        device = temporal_patch_tokens.device
        dtype = temporal_patch_tokens.dtype

        outputs = torch.zeros(batch_size, max_steps, embed_dim, device=device, dtype=dtype)
        padding_mask = torch.ones(batch_size, max_steps, device=device, dtype=torch.bool)

        for idx in range(batch_size):
            steps = int(target_num_steps[idx].item())
            steps = max(steps, 1)
            sample = temporal_patch_tokens[idx : idx + 1].transpose(1, 2)
            resized = F.interpolate(
                sample,
                size=steps,
                mode="linear",
                align_corners=False if self.mode == "linear" else None,
            ).transpose(1, 2).squeeze(0)
            outputs[idx, :steps] = resized
            padding_mask[idx, :steps] = False

        return outputs, padding_mask


class ShallowTemporalReadout(nn.Module):
    """Lightweight temporal neck placed after BEATs trunk features.

    The output of this module is the main spatial embedding sequence that will
    later be both supervised by prediction heads and projected to the LLM space.

    Shape contract:
        Input:
            temporal_tokens: [B, T_s_max, D]
        Output:
            spatial_embeddings: [B, T_s_max, D]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 1,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.encoder = None
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        temporal_tokens: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Refine the fixed-rate temporal sequence for spatial supervision.

        Args:
            temporal_tokens:
                [B, T_s_max, D] temporal sequence after resampling and padding.
            padding_mask:
                Optional [B, T_s_max] mask for padded time steps.

        Returns:
            Tensor:
                [B, T_s_max, D] spatial embeddings for supervision and projection.
        """
        if self.encoder is None:
            return self.final_norm(temporal_tokens)
        output = self.encoder(
            temporal_tokens,
            src_key_padding_mask=padding_mask,
        )
        return self.final_norm(output)


class LocalSpatialEncoder(nn.Module):
    """CNN + temporal-attention encoder for local FOA spatial cues.

    This branch is intentionally independent from the BEATs semantic trunk. It
    keeps BEATs focused on W-channel event semantics while learning spatial
    cues from the full FOA + IV feature map:

        [W, X, Y, Z, IVx, IVy, IVz] -> CNN/ResNet-like front-end -> time tokens
        -> Transformer attention.

    Shape contract:
        Input:
            foa_feat: [B, 7, T_f, F]
                T_f: STFT/mel frame count
                F: mel bins, default 128
        Output:
            spatial_tokens: [B, T_f, D_s]
                D_s: local spatial hidden dimension.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_attention = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        else:
            self.temporal_attention = None
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        foa_feat: Tensor,
        return_pre_pool: bool = False,
    ) -> Tensor:
        """Encode local spatial features.

        Args:
            foa_feat:
                [B, 7, T_f, F] FOA feature map in canonical channel order
                [W, X, Y, Z, IVx, IVy, IVz].
            return_pre_pool:
                When True, also returns the 4D CNN output before the
                frequency collapse (``features.mean(dim=-1)``).  Used by
                legacy so the DOA spectral demixer can attend to
                ``[B, D_s, T_f, F_cnn]`` directly — i.e. the full
                time-frequency grid that still carries the IV directional
                cues.

        Returns:
            Tensor, or tuple when return_pre_pool=True:
                [B, T_f, D_s] local spatial token sequence (always).
                [B, D_s, T_f, F_cnn] pre-F-pool CNN features (optional).
        """
        if foa_feat.ndim != 4 or foa_feat.size(1) != self.in_channels:
            raise ValueError(
                f"Expected foa_feat with shape [B, {self.in_channels}, T_f, F], got {tuple(foa_feat.shape)}"
            )
        features = self.cnn(foa_feat)
        # Collapse mel/frequency bins after local TF convolution. Time length is
        # preserved because all CNN strides are (1, *).
        spatial_tokens = features.mean(dim=-1).transpose(1, 2).contiguous()
        if self.temporal_attention is not None:
            spatial_tokens = self.temporal_attention(spatial_tokens)
        spatial_tokens = self.final_norm(spatial_tokens)
        if return_pre_pool:
            return spatial_tokens, features
        return spatial_tokens


class LocalSpatialPredictionHeads(nn.Module):
    """Attention-pooling heads for the local-spatial fusion branch.

    Shape contract:
        Input:
            fused_tokens: [B, T_s_max, D]
                Temporally aligned fused sequence used for both class and
                spatial prediction.
            padding_mask: optional [B, T_s_max], True means padded.
            active_window_mask: optional [B, T_s_max], True means the weak
                single-source active window.
        Output:
            task_tokens: [B, 2, D]
                [:, 0] class pooled token
                [:, 1] spatial pooled token
            prediction_output:
                MonoTaskPredictionOutput with class logits, Cartesian
                direction vector, and distance.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 63,
        head_dropout: float = 0.0,
        use_semantic_anchor: bool = False,
        use_direct_cls: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_semantic_anchor = use_semantic_anchor
        self.use_direct_cls = use_direct_cls
        self.class_score = nn.Linear(embed_dim, 1)
        self.spatial_score = nn.Linear(embed_dim, 1)
        self.head_drop = nn.Dropout(head_dropout)
        self.class_head = nn.Linear(embed_dim, num_classes)
        # Semantic anchor: a separate lightweight classifier that reads
        # directly from pre-fusion BEATs tokens via mean-pool.  Its loss
        # acts as a gradient anchor keeping the trunk semantically grounded
        # while the spatial loss pushes hard on fused_tokens.
        if use_semantic_anchor:
            self.semantic_anchor_head = nn.Linear(embed_dim, num_classes)
        # Direct cls head: reads from mean-pooled semantic_tokens (trunk output)
        # instead of attention-pooled fused_tokens. Decouples classification
        # from spatial fusion — class prediction sees the same feature as
        # pure BEATs classification, closing the 20pt gap vs pure cls.
        if use_direct_cls:
            self.direct_cls_head = nn.Linear(embed_dim, num_classes)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Start from uniform temporal pooling so a finetuned BEATs classifier
        # sees the same kind of mean-pooled semantic feature as the baseline.
        nn.init.zeros_(self.class_score.weight)
        nn.init.zeros_(self.class_score.bias)
        nn.init.zeros_(self.spatial_score.weight)
        nn.init.zeros_(self.spatial_score.bias)

    @staticmethod
    def _build_keep_mask(
        tokens: Tensor,
        padding_mask: Optional[Tensor],
        active_window_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, num_steps, _ = tokens.shape
        keep_mask = torch.ones(
            batch_size,
            num_steps,
            dtype=torch.bool,
            device=tokens.device,
        )
        if padding_mask is not None:
            keep_mask &= ~padding_mask.to(torch.bool)
        if active_window_mask is not None:
            active_keep_mask = keep_mask & active_window_mask.to(torch.bool)
            has_active = active_keep_mask.any(dim=1, keepdim=True)
            keep_mask = torch.where(has_active, active_keep_mask, keep_mask)
        return keep_mask

    @staticmethod
    def _attention_pool(tokens: Tensor, scores: Tensor, keep_mask: Tensor) -> Tensor:
        scores = scores.squeeze(-1).masked_fill(~keep_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weights = weights.masked_fill(~keep_mask.unsqueeze(-1), 0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return (tokens * weights).sum(dim=1)

    def forward(
        self,
        fused_tokens: Tensor,
        padding_mask: Optional[Tensor] = None,
        active_window_mask: Optional[Tensor] = None,
        semantic_tokens: Optional[Tensor] = None,
        pre_readout_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, MonoTaskPredictionOutput]:
        """Pool fused temporal tokens and predict single-source labels.

        Args:
            fused_tokens:
                [B, T_s, D] fused semantic + spatial tokens. Used for all
                prediction heads including class.
            padding_mask:
                Optional [B, T_s] mask where True = padded.
            active_window_mask:
                Optional [B, T_s] weak active-time mask.
            semantic_tokens:
                Optional [B, T_s, D] pre-fusion BEATs semantic tokens.
                Not used for prediction — passed through to output so the
                loss layer can compute a semantic anchor loss that keeps the
                BEATs trunk from forgetting semantics under spatial gradient
                pressure. LLM tokens still come from fused_tokens.
            pre_readout_tokens:
                Optional [B, T_s, D] tokens after TemporalResampler but
                before ShallowTemporalReadout. When use_direct_cls=True,
                these are preferred over semantic_tokens for mean-pooling
                because they match the foa_cls feature space exactly:
                trunk → FreqPool → Resample → mean_pool.
        """
        if fused_tokens.ndim != 3 or fused_tokens.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected fused_tokens with shape [B, T_s, {self.embed_dim}], got {tuple(fused_tokens.shape)}"
            )
        keep_mask = self._build_keep_mask(fused_tokens, padding_mask, active_window_mask)

        # Spatial token always from fused (spatial CNN contributes here)
        spatial_token = self._attention_pool(
            tokens=fused_tokens,
            scores=self.spatial_score(fused_tokens),
            keep_mask=keep_mask,
        )
        normalized_spatial_token = self.spatial_norm(spatial_token)

        # Classification: either direct mean-pool from trunk (use_direct_cls)
        # or attention-pool from fused tokens (default)
        sem_class_logits: Optional[Tensor] = None
        if self.use_direct_cls and hasattr(self, "direct_cls_head"):
            # Direct path: mean-pool encoder_memory (BEATs trunk output, [B, N_p, D])
            # This is IDENTICAL to foa_cls feature space:
            #   foa_cls: trunk_output [B,144,D] → mean_pool → classifier
            #   trunk_output [B,144,D] → mean_pool → direct_cls_head
            # encoder_memory has no padding (all N_p patches valid), so plain mean.
            if pre_readout_tokens is not None:
                # pre_readout_tokens = encoder_memory [B, N_p, D], no padding mask needed
                sem_pooled = pre_readout_tokens.mean(dim=1)
            else:
                # fallback: use fused_tokens with keep_mask (degraded path)
                cls_src = semantic_tokens if semantic_tokens is not None else fused_tokens
                sem_pooled = cls_src.masked_fill(
                    keep_mask.unsqueeze(-1) == 0, 0.0
                ).sum(dim=1) / keep_mask.float().sum(dim=1, keepdim=True).clamp_min(1)
            pred_class_logits = self.direct_cls_head(self.head_drop(sem_pooled))
            # class_score/class_head must still participate in loss for DDP
            # (find_unused_parameters=True). Add ×0 term to keep gradient alive.
            class_token = self._attention_pool(
                tokens=fused_tokens,
                scores=self.class_score(fused_tokens),
                keep_mask=keep_mask,
            )
            pred_class_logits = pred_class_logits + self.class_head(class_token) * 0.0
        else:
            class_token = self._attention_pool(
                tokens=fused_tokens,
                scores=self.class_score(fused_tokens),
                keep_mask=keep_mask,
            )
            pred_class_logits = self.class_head(self.head_drop(class_token))

        task_tokens = torch.stack([class_token, spatial_token], dim=1)

        # Semantic anchor loss path (separate from direct_cls)
        if semantic_tokens is not None and hasattr(self, "semantic_anchor_head"):
            sem_pooled = semantic_tokens.masked_fill(
                keep_mask.unsqueeze(-1) == 0, 0.0
            ).sum(dim=1) / keep_mask.float().sum(dim=1, keepdim=True).clamp_min(1)
            sem_class_logits = self.semantic_anchor_head(self.head_drop(sem_pooled))

        prediction_output = MonoTaskPredictionOutput(
            pred_class_logits=pred_class_logits,
            pred_direction=F.normalize(self.direction_head(self.head_drop(normalized_spatial_token)), dim=-1),
            pred_distance=F.softplus(self.distance_head(self.head_drop(normalized_spatial_token))),
            sem_class_logits=sem_class_logits,
        )
        return task_tokens, prediction_output


class MonoTaskTokenReadout(nn.Module):
    """Single-source readout with dedicated class/spatial task tokens.

    This module mirrors the high-level idea used by Spatial-AST:
        - keep the pretrained encoder as the main representation learner
        - append a small number of learnable task tokens
        - let a shallow transformer read task-specific information from the
          temporal embedding sequence

    Shape contract:
        Input:
            spatial_embeddings: [B, T_s_max, D]
            padding_mask: [B, T_s_max] optional padded-step mask
            active_window_mask: [B, T_s_max] optional weak valid-time mask for
                the single source. When provided, tokens outside the active
                window are hidden from the task-token readout.
        Output:
            task_tokens: [B, 2, D]
                [:, 0] class token
                [:, 1] spatial token
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 1,
        num_heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_token, std=0.02)

    def forward(
        self,
        spatial_embeddings: Tensor,
        padding_mask: Optional[Tensor] = None,
        active_window_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Read single-source class/spatial tokens from the temporal sequence."""
        batch_size, num_steps, embed_dim = spatial_embeddings.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dim {self.embed_dim}, got {embed_dim}"
            )

        temporal_keep_mask = torch.ones(
            batch_size,
            num_steps,
            dtype=torch.bool,
            device=spatial_embeddings.device,
        )
        if padding_mask is not None:
            temporal_keep_mask &= ~padding_mask.to(torch.bool)
        if active_window_mask is not None:
            candidate_keep_mask = temporal_keep_mask & active_window_mask.to(torch.bool)
            has_active = candidate_keep_mask.any(dim=1, keepdim=True)
            temporal_keep_mask = torch.where(
                has_active,
                candidate_keep_mask,
                temporal_keep_mask,
            )

        task_tokens = torch.cat(
            [
                self.class_token.expand(batch_size, -1, -1),
                self.spatial_token.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        readout_tokens = torch.cat([task_tokens, spatial_embeddings], dim=1)
        readout_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 2, dtype=torch.bool, device=spatial_embeddings.device),
                ~temporal_keep_mask,
            ],
            dim=1,
        )
        output = self.encoder(
            readout_tokens,
            src_key_padding_mask=readout_padding_mask,
        )
        return self.final_norm(output[:, :2])


class MonoTaskPredictionHeads(nn.Module):
    """Single-source heads fed by the class/spatial task tokens.

    Shape contract:
        Input:
            task_tokens: [B, 2, D]
                [:, 0] class token
                [:, 1] spatial token
        Output:
            pred_class_logits: [B, num_classes]
            pred_direction: [B, 3]
            pred_distance: [B, 1]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 63,
    ) -> None:
        super().__init__()
        self.class_norm = nn.LayerNorm(embed_dim)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, task_tokens: Tensor) -> MonoTaskPredictionOutput:
        """Predict single-source labels and spatial targets from task tokens."""
        if task_tokens.ndim != 3 or task_tokens.size(1) != 2:
            raise ValueError(
                f"Expected task_tokens with shape [B, 2, D], got {tuple(task_tokens.shape)}"
            )
        class_token = self.class_norm(task_tokens[:, 0])
        spatial_token = self.spatial_norm(task_tokens[:, 1])
        pred_class_logits = self.class_head(class_token)
        pred_direction = F.normalize(self.direction_head(spatial_token), dim=-1)
        pred_distance = F.softplus(self.distance_head(spatial_token))
        return MonoTaskPredictionOutput(
            pred_class_logits=pred_class_logits,
            pred_direction=pred_direction,
            pred_distance=pred_distance,
        )


class PreTrunkASTPredictionHeads(nn.Module):
    """Spatial-AST-style heads fed by task tokens that passed through the trunk.

    Token order follows the local Spatial-AST implementation:
        0: distance token
        1: DoA token
        2: class token

    Shape contract:
        Input:
            task_tokens: [B, 3, D]
        Output:
            pred_class_logits: [B, num_classes]
            pred_distance_logits: [B, num_distance_bins]
            pred_azi_logits: [B, num_azi_bins]
            pred_ele_logits: [B, num_ele_bins]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 63,
        num_distance_bins: int = 21,
        num_azi_bins: int = 360,
        num_ele_bins: int = 180,
    ) -> None:
        super().__init__()
        self.dist_norm = nn.LayerNorm(embed_dim)
        self.doa_norm = nn.LayerNorm(embed_dim)
        self.class_norm = nn.LayerNorm(embed_dim)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.distance_head = nn.Linear(embed_dim, num_distance_bins)
        self.azimuth_head = nn.Linear(embed_dim, num_azi_bins)
        self.elevation_head = nn.Linear(embed_dim, num_ele_bins)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.class_head.weight, std=2e-5)
        nn.init.zeros_(self.class_head.bias)
        nn.init.trunc_normal_(self.distance_head.weight, std=2e-5)
        nn.init.zeros_(self.distance_head.bias)
        nn.init.trunc_normal_(self.azimuth_head.weight, std=2e-5)
        nn.init.zeros_(self.azimuth_head.bias)
        nn.init.trunc_normal_(self.elevation_head.weight, std=2e-5)
        nn.init.zeros_(self.elevation_head.bias)

    def forward(self, task_tokens: Tensor) -> PreTrunkASTPredictionOutput:
        """Predict source class, distance bin, azimuth bin, and elevation bin."""
        if task_tokens.ndim != 3 or task_tokens.size(1) != 3:
            raise ValueError(
                f"Expected task_tokens with shape [B, 3, D], got {tuple(task_tokens.shape)}"
            )
        dist_token = self.dist_norm(task_tokens[:, 0])
        doa_token = self.doa_norm(task_tokens[:, 1])
        class_token = self.class_norm(task_tokens[:, 2])
        return PreTrunkASTPredictionOutput(
            pred_class_logits=self.class_head(class_token),
            pred_distance_logits=self.distance_head(dist_token),
            pred_azi_logits=self.azimuth_head(doa_token),
            pred_ele_logits=self.elevation_head(doa_token),
        )


class FixedSlotReadout(nn.Module):
    """Expand one time-step embedding into K fixed supervision slots.

    This is a simple encoder-only alternative to a query decoder.
    It exists only to expose multi-source supervision during training.

    Shape contract:
        Input:
            spatial_embeddings: [B, T_s_max, D]
        Output:
            slot_latents: [B, T_s_max, K, H]
                K: max number of simultaneous sources
                H: slot hidden size
    """

    def __init__(
        self,
        input_dim: int = 768,
        slot_hidden_dim: int = 768,
        num_slots: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.slot_hidden_dim = slot_hidden_dim
        self.num_slots = num_slots
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_slots * slot_hidden_dim),
        )

    def forward(self, spatial_embeddings: Tensor) -> Tensor:
        """Expand per-step embeddings into fixed source slots.

        Args:
            spatial_embeddings:
                [B, T_s_max, D] main spatial embedding sequence.

        Returns:
            Tensor:
                [B, T_s_max, K, H] fixed slot latents used only by supervision heads.
        """
        batch_size, num_steps, _ = spatial_embeddings.shape
        slot_latents = self.proj(spatial_embeddings)
        slot_latents = slot_latents.reshape(
            batch_size,
            num_steps,
            self.num_slots,
            self.slot_hidden_dim,
        )
        return slot_latents


class SpatialPredictionHeads(nn.Module):
    """Predict explicit spatial supervision targets from fixed slot latents.

    Shape contract:
        Input:
            slot_latents: [B, T_s_max, K, H]
        Output:
            pred_activity: [B, T_s_max, K]
            pred_azi_logits: [B, T_s_max, K, num_azi_bins]
            pred_ele_logits: [B, T_s_max, K, num_ele_bins]
            pred_dist: [B, T_s_max, K, 1]
            pred_class_logits: [B, T_s_max, K, num_classes]
    """

    def __init__(
        self,
        slot_hidden_dim: int = 768,
        num_classes: int = 527,
        num_azi_bins: int = 360,
        num_ele_bins: int = 180,
    ) -> None:
        super().__init__()
        self.slot_hidden_dim = slot_hidden_dim
        self.num_classes = num_classes
        self.num_azi_bins = num_azi_bins
        self.num_ele_bins = num_ele_bins
        self.input_norm = nn.LayerNorm(slot_hidden_dim)
        self.activity_head = nn.Linear(slot_hidden_dim, 1)
        self.azi_head = nn.Linear(slot_hidden_dim, num_azi_bins)
        self.ele_head = nn.Linear(slot_hidden_dim, num_ele_bins)
        self.dist_head = nn.Linear(slot_hidden_dim, 1)
        self.class_head = nn.Linear(slot_hidden_dim, num_classes)

    def forward(self, slot_latents: Tensor) -> SpatialPredictionOutput:
        """Apply supervision heads to each fixed slot.

        Args:
            slot_latents:
                [B, T_s_max, K, H] slot features after fixed-slot expansion.

        Returns:
            SpatialPredictionOutput:
                Structured prediction tensors described in the dataclass above.
        """
        normalized = self.input_norm(slot_latents)
        pred_activity = self.activity_head(normalized).squeeze(-1)
        pred_azi_logits = self.azi_head(normalized)
        pred_ele_logits = self.ele_head(normalized)
        pred_dist = self.dist_head(normalized)
        pred_class_logits = self.class_head(normalized)
        return SpatialPredictionOutput(
            pred_activity=pred_activity,
            pred_azi_logits=pred_azi_logits,
            pred_ele_logits=pred_ele_logits,
            pred_dist=pred_dist,
            pred_class_logits=pred_class_logits,
        )


class SOTokenProjector(nn.Module):
    """Project spatial embeddings into the hidden size expected by the LLM.

    Shape contract:
        Input:
            spatial_embeddings: [B, T_s_max, D]
        Output:
            llm_spatial_tokens: [B, T_s_max, d_llm]

    Notes:
        The projector consumes the main spatial embedding sequence directly.
        It does not consume fixed slot predictions.
    """

    def __init__(
        self,
        input_dim: int = 768,
        llm_hidden_dim: int = 4096,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.hidden_dim = hidden_dim or input_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, llm_hidden_dim),
        )

    def forward(self, spatial_embeddings: Tensor) -> Tensor:
        """Project encoder spatial embeddings to the LLM token space.

        Args:
            spatial_embeddings:
                [B, T_s_max, D] main spatial embedding sequence.

        Returns:
            Tensor:
                [B, T_s_max, d_llm] final spatial tokens sent to the LLM.
        """
        return self.proj(spatial_embeddings)


# ---------------------------------------------------------------------------
# Multi-source frame-level heads for ov1+ov2+ov3 training.
#
# All three heads consume the fused spatial embeddings [B, T_s, D] produced by
# the existing ``local_spatial`` fusion path and preserve the LLM token
# contract ``llm_spatial_tokens = projector(fused_embeddings)``. They differ in
# how they decode per-frame per-source predictions:
#
#   * ``FrameSlotHead``            — per-frame K slot heads (route A).
#   * ``SourceQueryDecoder`` +
#     ``FrameTrackPredictionHeads`` — K track queries with per-(k, t)
#                                      cross-attn readout (route B).
#   * ``ACCDOAHeads``              — per-class ACCDOA vector field (route C).
#
# The heads are additive — existing readout schemes do not reference them.
# ---------------------------------------------------------------------------


@dataclass
class FrameSlotPredictionOutput:
    """Per-frame K-slot predictions consumed by ``readout_scheme='local_spatial_slot'``.

    Attributes:
        pred_activity:
            [B, T_s, K] per-frame per-slot activity logits.
        pred_class_logits:
            [B, T_s, K, num_classes] per-frame per-slot class logits.
        pred_direction:
            [B, T_s, K, 3] per-frame per-slot unit-norm direction vectors.
        pred_distance:
            [B, T_s, K] per-frame per-slot distance in meters (softplus).
    """

    pred_activity: Tensor
    pred_class_logits: Tensor
    pred_direction: Tensor
    pred_distance: Tensor


@dataclass
class FrameTrackPredictionOutput:
    """Per-frame per-track predictions for ``readout_scheme='local_spatial_track'``.

    Attributes:
        pred_activity:
            [B, K, T_s] per-track per-frame activity logits.
        pred_class_logits:
            [B, K, T_s, num_classes] per-track per-frame class logits.
        pred_direction:
            [B, K, T_s, 3] per-track per-frame unit-norm direction vectors.
        pred_distance:
            [B, K, T_s] per-track per-frame distance in meters (softplus).
        track_latents:
            [B, K, D] per-track pooled latent for optional analysis.
        pred_num_active_logits:
            [B, T_s, K+1] optional per-frame active-source-count logits
            (0..K inclusive).  None when the model does not carry the legacy
            num_active head; when present, downstream CSV / validation can use
            ``argmax`` as the adaptive K̂ for activity gating.
        pred_distance_log_var:
            [B, K, T_s] optional Laplace scale (log-variance) from the
            log-distance head.  Only present when ``use_log_distance_head`` is
            enabled; otherwise None.  The NLL loss uses this to weight per-frame
            distance errors; inference ignores it.
    """

    pred_activity: Tensor
    pred_class_logits: Tensor
    pred_direction: Tensor
    pred_distance: Tensor
    track_latents: Tensor
    pred_num_active_logits: Optional[Tensor] = None
    pred_distance_log_var: Optional[Tensor] = None


@dataclass
class FrameACCDOAPredictionOutput:
    """Per-frame per-class ACCDOA predictions for ``readout_scheme='local_spatial_accdoa'``.

    Attributes:
        pred_accdoa:
            [B, T_s, num_classes, 3] per-frame per-class activity-coupled
            Cartesian DoA vector. The L2 norm encodes activity, the direction
            encodes DoA.
        pred_distance:
            [B, T_s, num_classes] per-frame per-class distance in meters
            (softplus).
    """

    pred_accdoa: Tensor
    pred_distance: Tensor


class FrameSlotHead(nn.Module):
    """Route A — per-frame K-slot head for multi-source supervision.

    Shape contract:
        Input:
            fused:        [B, T_s, D]
            padding_mask: optional [B, T_s], True marks padded steps.
        Output:
            FrameSlotPredictionOutput with tensors described above.

    Design notes:
        - ``slot_proj`` expands each time step into ``K`` slot latents of size
          ``slot_hidden_dim`` (smaller than ``D`` by default to keep head cheap).
        - Direction uses an L2-normalized 3D vector (1-cos loss downstream) and
          distance uses softplus for a non-negative regression. This keeps the
          head style consistent with the existing ``LocalSpatialPredictionHeads``.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_slots: int = 4,
        slot_hidden_dim: int = 192,
        num_classes: int = 63,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.slot_hidden_dim = slot_hidden_dim
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(embed_dim)
        self.slot_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_slots * slot_hidden_dim),
        )
        self.slot_norm = nn.LayerNorm(slot_hidden_dim)
        self.activity_head = nn.Linear(slot_hidden_dim, 1)
        self.class_head = nn.Linear(slot_hidden_dim, num_classes)
        self.direction_head = nn.Sequential(
            nn.Linear(slot_hidden_dim, slot_hidden_dim),
            nn.GELU(),
            nn.Linear(slot_hidden_dim, 3),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(slot_hidden_dim, slot_hidden_dim),
            nn.GELU(),
            nn.Linear(slot_hidden_dim, 1),
        )

    def forward(
        self,
        fused: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> FrameSlotPredictionOutput:
        if fused.ndim != 3 or fused.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected fused with shape [B, T_s, {self.embed_dim}], got {tuple(fused.shape)}"
            )
        batch_size, num_steps, _ = fused.shape
        x = self.input_norm(fused)
        slot_latents = self.slot_proj(x).reshape(
            batch_size, num_steps, self.num_slots, self.slot_hidden_dim
        )
        slot_latents = self.slot_norm(slot_latents)
        activity = self.activity_head(slot_latents).squeeze(-1)
        class_logits = self.class_head(slot_latents)
        direction = F.normalize(self.direction_head(slot_latents), dim=-1)
        distance = F.softplus(self.distance_head(slot_latents)).squeeze(-1)
        if padding_mask is not None:
            # zero-out activity on padded steps so downstream aggregation is
            # well-defined; loss path also masks these but being defensive here.
            pad_expand = padding_mask.to(torch.bool).unsqueeze(-1)
            activity = activity.masked_fill(pad_expand, 0.0)
        return FrameSlotPredictionOutput(
            pred_activity=activity,
            pred_class_logits=class_logits,
            pred_direction=direction,
            pred_distance=distance,
        )


class SourceQueryDecoder(nn.Module):
    """Route B — K learnable source queries + per-(k, t) cross-attn decoder.

    Two-stage decoding:
        Stage 1 (track-level):
            K queries run a TransformerDecoder over fused ``[B, T_s, D]`` memory
            to produce track latents ``[B, K, D]`` (each track = a putative
            source, identity-persistent over the entire clip).
        Stage 2 (per-frame):
            Expand queries into ``K * T_s`` per-(track, time) queries by adding
            a learned temporal positional embedding. Run a second
            TransformerDecoder over the same memory to produce
            ``[B, K, T_s, D]`` per-track per-frame features.

    Shape contract:
        Input:
            fused:        [B, T_s, D]
            padding_mask: optional [B, T_s] (True marks padded)
        Output:
            track_time_features: [B, K, T_s, D]
            track_latents:       [B, K, D]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_queries: int = 4,
        num_heads: int = 8,
        num_track_layers: int = 2,
        num_time_layers: int = 1,
        max_time_steps: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.max_time_steps = max_time_steps

        self.query = nn.Parameter(torch.zeros(num_queries, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.time_pos = nn.Parameter(torch.zeros(max_time_steps, embed_dim))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        track_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.track_decoder = nn.TransformerDecoder(
            track_layer,
            num_layers=num_track_layers,
        )

        time_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_decoder = nn.TransformerDecoder(
            time_layer,
            num_layers=num_time_layers,
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def _resolve_time_pos(self, num_steps: int) -> Tensor:
        if num_steps <= self.max_time_steps:
            return self.time_pos[:num_steps]
        pos = self.time_pos.transpose(0, 1).unsqueeze(0)
        pos = F.interpolate(pos, size=num_steps, mode="linear", align_corners=False)
        return pos.squeeze(0).transpose(0, 1)

    def forward(
        self,
        fused: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if fused.ndim != 3 or fused.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected fused with shape [B, T_s, {self.embed_dim}], got {tuple(fused.shape)}"
            )
        batch_size, num_steps, _ = fused.shape
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        mem_mask = padding_mask.to(torch.bool) if padding_mask is not None else None
        track_latents = self.track_decoder(
            tgt=queries,
            memory=fused,
            memory_key_padding_mask=mem_mask,
        )
        time_pos = self._resolve_time_pos(num_steps)
        track_time_queries = (
            track_latents.unsqueeze(2)
            + time_pos.unsqueeze(0).unsqueeze(0)
        )
        track_time_queries = track_time_queries.reshape(
            batch_size, self.num_queries * num_steps, self.embed_dim
        )
        refined = self.time_decoder(
            tgt=track_time_queries,
            memory=fused,
            memory_key_padding_mask=mem_mask,
        )
        refined = self.out_norm(refined)
        refined = refined.reshape(
            batch_size, self.num_queries, num_steps, self.embed_dim
        )
        return refined, track_latents


class FrameTrackPredictionHeads(nn.Module):
    """Route B — heads on top of per-track per-frame features from ``SourceQueryDecoder``.

    Shape contract:
        Input:
            track_time_features: [B, K, T_s, D]
        Output:
            FrameTrackPredictionOutput.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 63,
        dropout: float = 0.1,
        use_class_head_mlp_residual: bool = False,
        class_head_mlp_hidden_multiplier: int = 2,
        class_head_mlp_dropout: float = 0.1,
        use_class_head_demixer: bool = False,
        class_head_demixer_layers: int = 1,
        class_head_demixer_heads: int = 8,
        class_head_demixer_dropout: float = 0.1,
        use_spatial_head_demixer: bool = False,
        spatial_head_demixer_layers: int = 1,
        spatial_head_demixer_heads: int = 8,
        spatial_head_demixer_dropout: float = 0.1,
        use_num_active_head: bool = False,
        num_active_max: int = 4,
        # --- per-class learnable activity bias ------------------------
        use_class_activity_bias: bool = False,
        # --- class-conditional activity gate --------------------------
        use_class_conditional_gate: bool = False,
        gate_class_emb_dim: int = 32,
        gate_hidden_dim: int = 128,
        gate_scale: float = 0.5,
        # --- log-distance + uncertainty head --------------------------
        use_log_distance_head: bool = False,
        log_distance_init_mean: float = 0.4,   # log(1.5) ≈ 0.405
        log_distance_init_log_var: float = -3.2,  # log(0.04) ≈ -3.22
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.input_norm = nn.LayerNorm(embed_dim)
        self.activity_head = nn.Linear(embed_dim, 1)
        self.class_head = nn.Linear(embed_dim, num_classes)
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        # optional zero-init residual MLP on top of the linear
        # class_head.  When enabled, the final class_logits become
        #   class_head(x) + gate * class_head_mlp(x)
        # The MLP *output layer* is zero-initialised so the residual is 0 at
        # load (identical to legacy), but the gate starts at a small positive
        # value (0.01) so dL/dmlp_weight flows through gate from step 0.
        # Without this, gate*output-layer both zero would freeze the
        # residual permanently at 0.
        self.use_class_head_mlp_residual = bool(use_class_head_mlp_residual)
        if self.use_class_head_mlp_residual:
            hidden = embed_dim * max(1, int(class_head_mlp_hidden_multiplier))
            self.class_head_mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(class_head_mlp_dropout),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, num_classes),
            )
            # Zero-init the output layer so the residual starts at 0.
            nn.init.zeros_(self.class_head_mlp[-1].weight)
            nn.init.zeros_(self.class_head_mlp[-1].bias)
            # Small-positive gate so gradient flows into the MLP weights on
            # step 0; the forward output is still gate*0 = 0 at load.
            self.class_head_mlp_gate = nn.Parameter(torch.full((1,), 1e-2))
        else:
            self.class_head_mlp = None
            self.class_head_mlp_gate = None

        # optional spectral demixing cross-attention.  Each track
        # latent attends to the pre-frequency-pool BEATs trunk tokens at
        # its time step (a per-track, per-frame frequency-axis demixer).
        # The demixer output is projected and added to the class_head input
        # through a zero-init gate, so checkpoints without demixer
        # parameters still produce identical outputs.
        self.use_class_head_demixer = bool(use_class_head_demixer)
        if self.use_class_head_demixer:
            self.class_head_demixer = ClassHeadSpectralDemixer(
                embed_dim=embed_dim,
                num_layers=class_head_demixer_layers,
                num_heads=class_head_demixer_heads,
                dropout=class_head_demixer_dropout,
            )
        else:
            self.class_head_demixer = None

        # Symmetric spectral demixer for direction/distance heads.
        # Reuses ClassHeadSpectralDemixer (identical structure) with its
        # own parameters.  The residual is added to the direction_head /
        # distance_head inputs rather than the class_head input; the class
        # demixer is untouched.  Zero-init out_proj + gate=1e-2 keeps the
        # first forward identical to legacy until the gate warms up.
        self.use_spatial_head_demixer = bool(use_spatial_head_demixer)
        if self.use_spatial_head_demixer:
            self.spatial_head_demixer = ClassHeadSpectralDemixer(
                embed_dim=embed_dim,
                num_layers=spatial_head_demixer_layers,
                num_heads=spatial_head_demixer_heads,
                dropout=spatial_head_demixer_dropout,
            )
        else:
            self.spatial_head_demixer = None

        # optional per-frame num-active-source head.  Predicts the
        # number of *truly* active sources (0..num_active_max) on each frame
        # so downstream evaluation can pick top-K̂ tracks instead of a hard
        # 0.5 activity threshold.  Load-safe: when strict=False loading a
        # a legacy checkpoint, this head falls back to "predict 0 for every
        # frame" (bias[0]=+large, everything else zero) so the
        # fallback-to-hard-0.5 logic in FrameMetric / CSV dumpers keeps
        # working until training warms it up.
        self.use_num_active_head = bool(use_num_active_head)
        self.num_active_max = int(num_active_max)
        if self.use_num_active_head:
            out_dim = self.num_active_max + 1
            self.num_active_head = nn.Linear(embed_dim, out_dim)
            nn.init.zeros_(self.num_active_head.weight)
            nn.init.zeros_(self.num_active_head.bias)
            # Bias[0] = +4 so argmax defaults to "0 active" at ep0. A trainable
            # bias will move away from this as soon as supervision kicks in.
            with torch.no_grad():
                self.num_active_head.bias[0] = 4.0
        else:
            self.num_active_head = None

        # --- per-class learnable activity bias ------------------
        # Adds a per-class logit bias to activity_logit, using the predicted
        # class-softmax as a soft assignment so gradients flow. Zero-init →
        # identical to legacy behavior at ep0.
        self.use_class_activity_bias = bool(use_class_activity_bias)
        if self.use_class_activity_bias:
            self.class_activity_bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.class_activity_bias = None

        # --- class-conditional activity gate --------------------
        # Small MLP that fuses token + soft-class-embedding + direction vector
        # into an additive activity logit. Last Linear is zero-init → ep0 gate
        # contribution = 0, identical to legacy.
        self.use_class_conditional_gate = bool(use_class_conditional_gate)
        self.gate_scale = float(gate_scale)
        if self.use_class_conditional_gate:
            self.gate_class_embedding = nn.Embedding(num_classes, int(gate_class_emb_dim))
            # Smaller init so softmax-weighted embeddings stay bounded.
            nn.init.normal_(self.gate_class_embedding.weight, mean=0.0, std=0.02)
            gate_in_dim = embed_dim + int(gate_class_emb_dim) + 3  # +3 for direction
            self.class_conditional_gate = nn.Sequential(
                nn.Linear(gate_in_dim, int(gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(gate_hidden_dim), 1),
            )
            # Zero-init final layer → gate_logit = 0 at ep0
            nn.init.zeros_(self.class_conditional_gate[-1].weight)
            nn.init.zeros_(self.class_conditional_gate[-1].bias)
        else:
            self.gate_class_embedding = None
            self.class_conditional_gate = None

        # --- log-distance + uncertainty (Laplace) head ----------
        # Upgrades distance_head output from 1 scalar (distance) to 2 scalars
        # [log_distance, log_var]. Recognised by spatial_loss when
        # distance_loss_type == "laplace_nll".  At init:
        #   bias[0] = log_distance_init_mean  (≈ log(1.5) ≈ 0.4)
        #   bias[1] = log_distance_init_log_var (≈ log(0.04) ≈ -3.2)
        # This replaces the existing distance_head. Checkpoint compatibility:
        # - legacy ckpts carry distance_head.3.weight[1, 768] / bias[1]. When
        #   loading strict=False into the 2-output head, the extra row is
        #   missing and will be initialized from log_distance_init_log_var.
        # - legacy ckpts can load back into the legacy 1-output head only if
        #   strict=False (the extra row will be silently dropped via
        #   unexpected-keys; to switch back to legacy, retrain).
        self.use_log_distance_head = bool(use_log_distance_head)
        if self.use_log_distance_head:
            # Replace the last Linear in distance_head with a 2-output one
            self.distance_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
            )
            with torch.no_grad():
                # Zero weights so bias dominates at init
                self.distance_head[-1].weight.zero_()
                self.distance_head[-1].bias[0] = float(log_distance_init_mean)
                self.distance_head[-1].bias[1] = float(log_distance_init_log_var)

    def forward(
        self,
        track_time_features: Tensor,
        track_latents: Tensor,
        pre_pool_features: Optional[Tensor] = None,
        pre_pool_grid_size: Optional[Tuple[int, int]] = None,
        pre_pool_time_mask: Optional[Tensor] = None,
        spatial_pre_pool_features: Optional[Tensor] = None,
        spatial_pre_pool_grid_size: Optional[Tuple[int, int]] = None,
        spatial_pre_pool_time_mask: Optional[Tensor] = None,
    ) -> FrameTrackPredictionOutput:
        if track_time_features.ndim != 4 or track_time_features.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected track_time_features [B, K, T_s, {self.embed_dim}], got "
                f"{tuple(track_time_features.shape)}"
            )
        x = self.input_norm(track_time_features)
        activity = self.activity_head(x).squeeze(-1)
        class_input = x
        if (
            self.class_head_demixer is not None
            and pre_pool_features is not None
            and pre_pool_grid_size is not None
        ):
            demix_residual = self.class_head_demixer(
                track_time_features=x,
                pre_pool_features=pre_pool_features,
                pre_pool_grid_size=pre_pool_grid_size,
                pre_pool_time_mask=pre_pool_time_mask,
            )
            class_input = class_input + demix_residual
        class_logits = self.class_head(class_input)
        if self.class_head_mlp is not None and self.class_head_mlp_gate is not None:
            class_logits = class_logits + self.class_head_mlp_gate * self.class_head_mlp(class_input)
        # Symmetric spectral demixer for direction/distance heads.
        # When legacy is enabled (spatial_pre_pool_features supplied), the KV
        # comes from the LocalSpatialEncoder pre-pool grid (carries IV
        # directional cues); otherwise it falls back to the same BEATs
        # trunk pre-pool grid as the class demixer.
        spatial_input = x
        if self.spatial_head_demixer is not None:
            if (
                spatial_pre_pool_features is not None
                and spatial_pre_pool_grid_size is not None
            ):
                spatial_kv = spatial_pre_pool_features
                spatial_grid = spatial_pre_pool_grid_size
                spatial_tmask = spatial_pre_pool_time_mask
            else:
                spatial_kv = pre_pool_features
                spatial_grid = pre_pool_grid_size
                spatial_tmask = pre_pool_time_mask
            if spatial_kv is not None and spatial_grid is not None:
                spatial_residual = self.spatial_head_demixer(
                    track_time_features=x,
                    pre_pool_features=spatial_kv,
                    pre_pool_grid_size=spatial_grid,
                    pre_pool_time_mask=spatial_tmask,
                )
                spatial_input = spatial_input + spatial_residual
        direction = F.normalize(self.direction_head(spatial_input), dim=-1)

        # --- log-distance + uncertainty head --------------------
        distance_log_var: Optional[Tensor] = None
        if self.use_log_distance_head:
            dist_out = self.distance_head(spatial_input)  # [B, K, T_s, 2]
            pred_log_dist = dist_out[..., 0]               # [B, K, T_s]
            pred_log_var = dist_out[..., 1]                # [B, K, T_s]
            distance = torch.exp(pred_log_dist)            # positive by construction
            distance_log_var = pred_log_var
        else:
            distance = F.softplus(self.distance_head(spatial_input)).squeeze(-1)

        # --- per-class learnable activity logit bias -----------
        # Add per-class bias weighted by predicted class softmax; since this
        # enters the same logit used by BCE/ASL loss, the bias is learned via
        # backprop without any explicit threshold tuning.
        if self.class_activity_bias is not None:
            class_probs = F.softmax(class_logits, dim=-1)  # [B, K, T_s, C]
            expected_bias = torch.einsum(
                'bktc,c->bkt', class_probs, self.class_activity_bias
            )  # [B, K, T_s]
            activity = activity + expected_bias

        # --- class-conditional activity gate -------------------
        if self.class_conditional_gate is not None and self.gate_class_embedding is not None:
            # soft class embedding: softmax(class_logits) @ embedding_weight
            class_probs_for_gate = F.softmax(class_logits, dim=-1)  # [B, K, T_s, C]
            soft_class_emb = class_probs_for_gate @ self.gate_class_embedding.weight
            # direction is already unit-norm [B, K, T_s, 3]
            gate_input = torch.cat([x, soft_class_emb, direction], dim=-1)
            gate_logit = self.class_conditional_gate(gate_input).squeeze(-1)  # [B, K, T_s]
            activity = activity + self.gate_scale * gate_logit

        # per-frame "how many sources are active" logits.  Aggregate
        # across the K-axis by mean so a single (b, t) gets one K+1-way
        # prediction that downstream eval can argmax into a top-K̂ gate.
        num_active_logits: Optional[Tensor] = None
        if self.num_active_head is not None:
            frame_feat = x.mean(dim=1)  # [B, T_s, D]
            num_active_logits = self.num_active_head(frame_feat)  # [B, T_s, K+1]
        return FrameTrackPredictionOutput(
            pred_activity=activity,
            pred_class_logits=class_logits,
            pred_direction=direction,
            pred_distance=distance,
            track_latents=track_latents,
            pred_num_active_logits=num_active_logits,
            pred_distance_log_var=distance_log_var,
        )


class ClassHeadSpectralDemixer(nn.Module):
    """Per-track, per-frame frequency-axis cross-attention (legacy demixer).

    Given:
        track_time_features: [B, K, T_s, D]
            Post-input_norm track latents at the 2.5 Hz spatial rate.
        pre_pool_features:   [B, N_p, D]
            BEATs trunk output BEFORE frequency_pool, where N_p = T_p * F_p
            is the patch grid (time-first, i.e. row-major with time varying
            slowest is also handled via explicit grid reshape).
        pre_pool_grid_size:  (T_p, F_p)
            Patch grid geometry.

    Each (b, k, t) query attends to the F_p frequency tokens at the trunk
    time step aligned with t.  An optional [B, T_p] time mask marks valid
    patch time steps (e.g. padded tails).  The output is projected back to
    the original D and gated by a learnable scalar initialised to 0 so
    that demixer-less checkpoints produce unchanged outputs.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = max(1, int(num_layers))
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Zero-init output projection; tiny positive gate so dL/dattn flows
        # through gate even at step 0 (gate*zero_output = 0 forward, but
        # dL/dattn_weight = gate != 0).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate = nn.Parameter(torch.full((1,), 1e-2))

    def forward(
        self,
        track_time_features: Tensor,
        pre_pool_features: Tensor,
        pre_pool_grid_size: Tuple[int, int],
        pre_pool_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, K, T_s, D = track_time_features.shape
        T_p, F_p = int(pre_pool_grid_size[0]), int(pre_pool_grid_size[1])
        if pre_pool_features.size(-1) != D:
            raise ValueError(
                f"Demixer expected pre_pool_features dim {D}, got {pre_pool_features.size(-1)}"
            )
        expected = T_p * F_p
        if pre_pool_features.size(1) != expected:
            # Fall back gracefully — demixer is additive & zero-gated.
            return track_time_features.new_zeros(track_time_features.shape)
        # Reshape trunk output to [B, T_p, F_p, D]
        kv_grid = pre_pool_features.view(B, T_p, F_p, D)
        # For each frame t in [0, T_s), map it to a trunk time step
        # t_p = round(t * T_p / T_s) clipped to valid range.
        if T_s > 0 and T_p > 0:
            time_idx = torch.arange(T_s, device=kv_grid.device).float() * (T_p / max(1, T_s))
            time_idx = time_idx.round().clamp_(0, T_p - 1).long()
        else:
            time_idx = torch.zeros((T_s,), dtype=torch.long, device=kv_grid.device)
        # Gather frequency tokens at mapped trunk steps: [B, T_s, F_p, D]
        kv_per_frame = kv_grid[:, time_idx, :, :]
        # Expand per-track so each (b, k, t) has its own KV bank.
        # [B, K, T_s, F_p, D]
        kv_per_frame = kv_per_frame.unsqueeze(1).expand(B, K, T_s, F_p, D).contiguous()
        kv_flat = kv_per_frame.view(B * K * T_s, F_p, D)
        kv_flat = self.kv_norm(kv_flat)
        q_flat = track_time_features.reshape(B * K * T_s, 1, D)
        q_flat = self.q_norm(q_flat)
        key_padding_mask: Optional[Tensor] = None
        if pre_pool_time_mask is not None:
            # pre_pool_time_mask: [B, T_p] with True marking *valid* time
            # steps.  A fully-padded time step would set all F_p keys
            # invalid — in that case mask it out.
            per_frame_valid = pre_pool_time_mask[:, time_idx]  # [B, T_s]
            # -> [B, K, T_s, F_p] -> True = should attend
            per_frame_valid = per_frame_valid.unsqueeze(1).expand(B, K, T_s).reshape(-1)
            if per_frame_valid.all():
                key_padding_mask = None
            else:
                # nn.MultiheadAttention uses `key_padding_mask` where True
                # marks positions to *ignore*.  We want to ignore frames
                # where per_frame_valid == False on every key.
                ignore = ~per_frame_valid  # [B*K*T_s]
                key_padding_mask = ignore.unsqueeze(1).expand(-1, F_p).contiguous()
        attn_out = q_flat
        for layer in self.layers:
            attn_out, _ = layer(
                attn_out,
                kv_flat,
                kv_flat,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        residual = self.out_proj(attn_out).view(B, K, T_s, D)
        return residual * self.gate


class LocalSpatialCrossFusionBlock(nn.Module):
    """Semantic<-spatial cross-attention block used by legacy fusion.

    The block is intentionally identity-biased so legacy checkpoints can hot-start
    safely:
      - semantic tokens stay as the residual backbone
      - spatial tokens enter through gated cross-attention
      - gates are initialized near 0 (sigmoid(bias) ~ 0.12)

    Shape contract:
        semantic_embeddings: [B, T_s, D]
        spatial_embeddings:  [B, T_s, D]
        padding_mask:        optional [B, T_s]
        output:              [B, T_s, D]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_multiplier: int = 4,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.semantic_norm = nn.LayerNorm(embed_dim)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_drop = nn.Dropout(dropout)
        self.cross_gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.cross_gate.weight)
        nn.init.constant_(self.cross_gate.bias, gate_bias)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_multiplier, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        semantic_embeddings: Tensor,
        spatial_embeddings: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        sem = self.semantic_norm(semantic_embeddings)
        spa = self.spatial_norm(spatial_embeddings)
        attn_out, _ = self.cross_attn(
            query=sem,
            key=spa,
            value=spa,
            key_padding_mask=padding_mask.to(torch.bool) if padding_mask is not None else None,
            need_weights=False,
        )
        cross_gate = torch.sigmoid(self.cross_gate(torch.cat([sem, spa], dim=-1)))
        fused = semantic_embeddings + self.cross_drop(cross_gate * attn_out)
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused


class LocalSpatialCrossFuser(nn.Module):
    """Two-branch fusion module for the legacy fusion.

    Semantic tokens remain the main backbone. Spatial tokens are injected via
    stacked semantic<-spatial cross-attention blocks plus one final gated
    spatial residual before the outer ``local_spatial_fusion_norm``.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_bias: float = -2.0,
        direct_gate_bias: float = -1.5,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                LocalSpatialCrossFusionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    gate_bias=gate_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_semantic_norm = nn.LayerNorm(embed_dim)
        self.out_spatial_norm = nn.LayerNorm(embed_dim)
        self.direct_gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.zeros_(self.direct_gate.weight)
        nn.init.constant_(self.direct_gate.bias, direct_gate_bias)

    def forward(
        self,
        semantic_embeddings: Tensor,
        spatial_embeddings: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        fused = semantic_embeddings
        for block in self.blocks:
            fused = block(
                semantic_embeddings=fused,
                spatial_embeddings=spatial_embeddings,
                padding_mask=padding_mask,
            )
        sem = self.out_semantic_norm(fused)
        spa = self.out_spatial_norm(spatial_embeddings)
        direct_gate = torch.sigmoid(self.direct_gate(torch.cat([sem, spa], dim=-1)))
        return fused + direct_gate * spatial_embeddings


class ACCDOAHeads(nn.Module):
    """Route C — per-class per-frame ACCDOA head.

    For each (batch, time, class) emits a 3D vector ``v`` whose magnitude
    encodes class activity at that frame and direction encodes DoA. Distance is
    predicted separately per (batch, time, class) with softplus.

    Shape contract:
        Input:
            fused: [B, T_s, D]
        Output:
            FrameACCDOAPredictionOutput with
                pred_accdoa:   [B, T_s, num_classes, 3]
                pred_distance: [B, T_s, num_classes]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 63,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_input_norm: bool = False,
        output_scale: float = 0.01,
        distance_prior_mean: float = 1.75,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        # NOTE(bugfix): fused_embeddings already passes local_spatial_fusion_norm
        # (the fuser's outer LayerNorm).  Re-normalizing here *discards the
        # magnitude* of the fused token, which ACCDOA critically needs — pred
        # magnitude encodes activity.  Default use_input_norm=False so the head
        # sees the raw fused token.  Keep the flag for backward-compat with the
        # legacy preset, but new presets should leave it False.
        self.input_norm = nn.LayerNorm(embed_dim) if use_input_norm else nn.Identity()
        self.doa_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes * 3),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # distance_prior_mean = desired softplus(bias) at init  →  bias = log(exp(m) - 1)
        self._init_heads(output_scale=output_scale, distance_prior_mean=distance_prior_mean)

    def _init_heads(
        self, output_scale: float = 0.01, distance_prior_mean: float = 1.75
    ) -> None:
        # NOTE(bugfix): zero-init breaks ACCDOA direction learning.  With
        # pred = 0, only the "inactive (b,t,c) targets" (98%+ of the target
        # tensor, all zero) contribute non-trivial MSE — the 1.6% of active
        # targets contribute ~0 gradient because pred-target = -unit_vec has
        # tiny norm when averaged across the full num_classes grid.  The
        # direction signal is thus diluted by ~60x vs what a non-zero init
        # would give.  Keep outputs *small* (output_scale=0.01) so initial
        # predictions do not dominate the loss, but *non-zero* so every
        # (b,t,c) cell has a meaningful gradient from step 1.
        last_doa = self.doa_head[-1]
        last_dist = self.distance_head[-1]

        nn.init.xavier_uniform_(last_doa.weight, gain=output_scale)
        nn.init.zeros_(last_doa.bias)

        nn.init.xavier_uniform_(last_dist.weight, gain=output_scale)
        # softplus(b) = m  →  b = log(exp(m) - 1); clamp for numerical safety.
        m = max(float(distance_prior_mean), 1e-3)
        bias_val = math.log(math.exp(m) - 1.0) if m > 20 else math.log1p(math.expm1(m))
        nn.init.constant_(last_dist.bias, bias_val)

    def forward(self, fused: Tensor) -> FrameACCDOAPredictionOutput:
        if fused.ndim != 3 or fused.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected fused with shape [B, T_s, {self.embed_dim}], got {tuple(fused.shape)}"
            )
        batch_size, num_steps, _ = fused.shape
        x = self.input_norm(fused)
        accdoa = self.doa_head(x).reshape(batch_size, num_steps, self.num_classes, 3)
        distance = F.softplus(self.distance_head(x))
        return FrameACCDOAPredictionOutput(
            pred_accdoa=accdoa,
            pred_distance=distance,
        )


# ---------------------------------------------------------------------------
# Frame-wise single-source prediction (ov1 style, but per-frame rather than
# clip-level).  Each frame independently predicts: class, direction, distance.
# This is a direct supervised path — no Hungarian matching, no slot queries.
# Designed for ov1 data where only one source is active at a time.
# ---------------------------------------------------------------------------

@dataclass
class FrameWisePredictionOutput:
    """Per-frame predictions from FrameWisePredictionHeads.

    Attributes:
        pred_activity:      [B, T_s, 1]            — activity logit (before sigmoid)
        pred_class_logits:  [B, T_s, num_classes]  — unnormalized class scores
        pred_direction:     [B, T_s, 3]            — unit Cartesian direction
        pred_distance:      [B, T_s, 1]            — distance in metres (softplus)
        sem_class_logits:   [B, T_s, num_classes]  — semantic anchor logits (optional)
    """

    pred_activity: Tensor
    pred_class_logits: Tensor
    pred_direction: Tensor
    pred_distance: Tensor
    sem_class_logits: Optional[Tensor] = None


class FrameWisePredictionHeads(nn.Module):
    """Per-frame cls + spatial prediction heads for ``local_spatial_framewise``.

    Applies three lightweight MLP heads to the fused token sequence
    ``[B, T_s, D]``, producing one set of predictions per time step.
    The class head uses the full embedding, while direction and distance
    share a spatial branch with a shared LayerNorm.

    Shape contract:
        Input:
            fused: [B, T_s, D]
        Output:
            FrameWisePredictionOutput
                pred_activity:     [B, T_s, 1]       — activity logit (sigmoid → prob)
                pred_class_logits: [B, T_s, num_classes]
                pred_direction:    [B, T_s, 3]
                pred_distance:     [B, T_s, 1]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 63,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_semantic_anchor: bool = False,
        num_anchor_classes: int = 63,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_semantic_anchor = use_semantic_anchor

        self.act_norm = nn.LayerNorm(embed_dim)
        self.act_drop = nn.Dropout(dropout)
        self.activity_head = nn.Linear(embed_dim, 1)

        self.cls_norm = nn.LayerNorm(embed_dim)
        self.cls_drop = nn.Dropout(dropout)
        self.class_head = nn.Linear(embed_dim, num_classes)

        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.spatial_drop = nn.Dropout(dropout)
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        if use_semantic_anchor:
            self.semantic_anchor_head = nn.Linear(embed_dim, num_anchor_classes)

        self._init_parameters()

    def _init_parameters(self) -> None:
        # activity head: init near zero so initial P(active) ≈ 0.5
        nn.init.zeros_(self.activity_head.weight)
        nn.init.zeros_(self.activity_head.bias)
        # direction head: 小随机初始化，确保初始输出不是零向量
        # 全零初始化会导致 F.normalize(zeros) 在零点不可导，梯度无法传导
        nn.init.trunc_normal_(self.direction_head[-1].weight, std=0.02)
        nn.init.zeros_(self.direction_head[-1].bias)
        # distance head: 全零初始化可以，softplus(0)=log(2)≈0.69m 是合理起点
        nn.init.zeros_(self.distance_head[-1].weight)
        nn.init.zeros_(self.distance_head[-1].bias)

    def forward(
        self,
        fused: Tensor,
        semantic_tokens: Optional[Tensor] = None,
    ) -> FrameWisePredictionOutput:
        """Forward pass.

        Args:
            fused:           [B, T_s, D] fused spatial-semantic tokens
            semantic_tokens: [B, T_s, D] pre-fusion BEATs tokens for the
                             semantic anchor (only used when
                             ``use_semantic_anchor=True``)
        """
        if fused.ndim != 3 or fused.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected fused [B, T_s, {self.embed_dim}], got {tuple(fused.shape)}"
            )

        # Activity head — shared norm with class head
        act_feat = self.act_drop(self.act_norm(fused))
        pred_activity = self.activity_head(act_feat)          # [B, T_s, 1] logit

        cls_feat = self.cls_drop(self.cls_norm(fused))        # [B, T_s, D]
        pred_class_logits = self.class_head(cls_feat)          # [B, T_s, C]

        spa_feat = self.spatial_drop(self.spatial_norm(fused))
        pred_direction = F.normalize(
            self.direction_head(spa_feat), dim=-1
        )  # [B, T_s, 3]
        pred_distance = F.softplus(self.distance_head(spa_feat))  # [B, T_s, 1]

        sem_class_logits: Optional[Tensor] = None
        if self.use_semantic_anchor and semantic_tokens is not None and hasattr(self, "semantic_anchor_head"):
            sem_feat = self.cls_drop(self.cls_norm(semantic_tokens))
            sem_class_logits = self.semantic_anchor_head(sem_feat)  # [B, T_s, C]

        return FrameWisePredictionOutput(
            pred_activity=pred_activity,
            pred_class_logits=pred_class_logits,
            pred_direction=pred_direction,
            pred_distance=pred_distance,
            sem_class_logits=sem_class_logits,
        )


# ========================================================================
# Enhanced spatial modules — V2 adapter + trunk spatial adapters
# ========================================================================


class SqueezeExcitation(nn.Module):
    """Channel attention via Squeeze-and-Excitation block.

    Applies global average pooling followed by a two-layer FC bottleneck to
    produce per-channel multiplicative gates.

    Shape:
        Input:  [B, C, H, W]
        Output: [B, C, H, W]  (same shape, channels re-weighted)
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)        # [B, C]
        w = self.fc(w).view(b, c, 1, 1)    # [B, C, 1, 1]
        return x * w


class SpatialDeltaPatchAdapterV2(nn.Module):
    """Enhanced spatial delta patch adapter (V2).

    Replaces the thin V1 bottleneck (7→32→512, ~200K params) with a deeper
    Conv-ResBlock-SE architecture (7→hidden→hidden via ResBlocks→512, ~1.5M
    params) while keeping the same forward signature:

        foa_feat [B, 7, T_f, F]  →  delta_patch_tokens [B, N_p, 512], grid_size

    Architecture::

        Conv1×1(7→H) + GroupNorm + GELU
        → ResBlock × N:
            Conv3×3 + GN + GELU → Conv3×3 + GN → SE → + residual → GELU
        → Conv2d(H→embed_dim, k=patch_size, s=patch_size)  (patchify)
        → flatten → × residual_alpha

    The ``residual_alpha`` is initialised small (default 0.1) so that the
    delta contribution is near-zero at hot-start, preserving pretrained BEATs
    patch-token quality.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 128,
        embed_dim: int = 512,
        patch_size: Tuple[int, int] = (16, 16),
        num_blocks: int = 2,
        se_reduction: int = 4,
        residual_scale_init: float = 0.1,
        out_proj_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # --- channel projection ---
        self.stem_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.stem_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.stem_act = nn.GELU()

        # --- residual blocks ---
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_AdapterResBlock(hidden_channels, se_reduction))
        self.blocks = nn.ModuleList(blocks)

        # --- patchify projection ---
        self.patch_proj = nn.Conv2d(
            hidden_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )

        # --- residual gating ---
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.out_proj_scale_init = out_proj_scale_init
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # stem: kaiming
        nn.init.kaiming_uniform_(self.stem_conv.weight, a=5 ** 0.5)
        # patch_proj: kaiming scaled down
        nn.init.kaiming_uniform_(self.patch_proj.weight, a=5 ** 0.5)
        self.patch_proj.weight.data.mul_(self.out_proj_scale_init)

    def forward(self, foa_feat: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Produce additive patch-token deltas from the 7-channel FOA feature map.

        Same signature as :class:`SpatialDeltaPatchAdapter` (V1).
        """
        if foa_feat.ndim != 4 or foa_feat.size(1) != self.in_channels:
            raise ValueError(
                f"Expected foa_feat [B, {self.in_channels}, T_f, F], got {tuple(foa_feat.shape)}"
            )
        x = self.stem_act(self.stem_norm(self.stem_conv(foa_feat)))
        for block in self.blocks:
            x = block(x)
        delta_grid = self.patch_proj(x)                              # [B, D, T_p, F_p]
        t_p, f_p = delta_grid.shape[-2], delta_grid.shape[-1]
        delta_patch_tokens = delta_grid.flatten(2).transpose(1, 2).contiguous()  # [B, N_p, D]
        delta_patch_tokens = self.residual_alpha * delta_patch_tokens
        return delta_patch_tokens, (t_p, f_p)


class _AdapterResBlock(nn.Module):
    """Conv3×3 → GN → GELU → Conv3×3 → GN → SE → + residual → GELU."""

    def __init__(self, channels: int, se_reduction: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.se = SqueezeExcitation(channels, reduction=se_reduction)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)


class SpatialAdapterLayer(nn.Module):
    """Zero-init bottleneck adapter for injection after each BEATs trunk layer.

    Inserts a lightweight down→up projection after a transformer layer.
    The ``up`` projection is zero-initialised so that at init the adapter is
    an identity transform (safe for hot-start from pretrained checkpoints).

    Shape:
        Input:  [T, B, D]  (backbone internal time-first layout)
        Output: [T, B, D]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        rank: int = 64,
        gate_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, rank)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, embed_dim)
        self.gate = nn.Parameter(torch.full((1,), gate_init))

        # Zero-init up projection → adapter output ≈ 0 at init
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: [T, B, D] → [T, B, D]."""
        return x + self.gate * self.up(self.act(self.down(self.norm(x))))


# =============================================================================
# Track refinement and multi-scale adapters
# =============================================================================


class TrackRefinementDecoder(nn.Module):
    """Track-wise temporal refinement transformer decoder.

    Takes per-frame K-track tokens ``[B, K, T_s, D]`` (output of
    SourceQueryDecoder) and the underlying fused memory ``[B, T_s, D]``,
    then produces **refined** per-track per-frame tokens of the same shape.

    Each layer performs:
        - self-attention among the K track queries at the same time step
          (so slots can "repel" each other when modelling overlapping
          sources)
        - cross-attention from each slot to the time-aligned memory token
          (refreshes with encoder context)

    Zero-init design (hot-start safety):
        Each layer's contribution is gated by ``layer_scale`` initialised
        to 0, so at ep0 the output equals the input track tokens exactly.
        Once training starts, layer_scale is free to grow and the
        refinement activates.

    Shape:
        Input:
            track_tokens: [B, K, T_s, D]
            memory:       [B, T_s, D]
        Output:
            refined_track_tokens: [B, K, T_s, D]
    """

    def __init__(
        self,
        num_tracks: int = 4,
        embed_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tracks = int(num_tracks)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=int(num_heads),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation='gelu',
                norm_first=True,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])
        # Zero-init layer-scale: ep0 contribution = 0, identical to input.
        self.layer_scale = nn.Parameter(torch.zeros(self.num_layers))

    def forward(self, track_tokens: Tensor, memory: Tensor) -> Tensor:
        """Refine per-track per-frame tokens via K-slot self-attn + memory cross-attn.

        Args:
            track_tokens: [B, K, T_s, D]
            memory:       [B, T_s, D]

        Returns:
            refined_track_tokens: [B, K, T_s, D]
        """
        if track_tokens.ndim != 4 or track_tokens.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected track_tokens [B, K, T_s, {self.embed_dim}], got "
                f"{tuple(track_tokens.shape)}"
            )
        if memory.ndim != 3 or memory.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected memory [B, T_s, {self.embed_dim}], got {tuple(memory.shape)}"
            )
        B, K, T_s, D = track_tokens.shape
        # Flatten (B, T_s) into batch dim so each time-step independently runs
        # the K-track self-attn + 1-token cross-attn.
        q = track_tokens.permute(0, 2, 1, 3).reshape(B * T_s, K, D)
        mem = memory.reshape(B * T_s, 1, D)
        for i, layer in enumerate(self.layers):
            out = layer(q, mem)
            q = q + self.layer_scale[i] * (out - q)
        refined = q.reshape(B, T_s, K, D).permute(0, 2, 1, 3).contiguous()
        return refined


class SpatialDeltaPatchAdapterV3(nn.Module):
    """Multi-scale spatial delta patch adapter (V3).

    Extends V2 by adding parallel Conv branches at multiple kernel sizes and
    dilations, fused via 1x1 conv. Intended to capture longer-time-scale
    reverberation cues (50-150ms early reflections) that V2's 3x3 kernel
    misses.

    Architecture::

        Conv1×1(7→H)
        → MultiScaleConv fuse:
            Conv3×3(H→H)       (short-time, 30ms)
            Conv5×5(H→H)       (mid-time, 50ms)
            Conv3×3 dilated=2  (long-time, 60ms w/ same receptive as 5x5)
          → cat → Conv1×1(3H→H) → GN → GELU
        → ResBlock × N  (same as V2)
        → Conv2d(H→embed_dim, patchify)
        → × residual_alpha

    Forward signature identical to V1/V2 ⇒ drop-in replacement.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 128,
        embed_dim: int = 512,
        patch_size: Tuple[int, int] = (16, 16),
        num_blocks: int = 2,
        se_reduction: int = 4,
        residual_scale_init: float = 0.1,
        out_proj_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # --- channel projection (stem) ---
        self.stem_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.stem_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.stem_act = nn.GELU()

        # --- multi-scale fuse ---
        self.ms_branch_3x3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False,
        )
        self.ms_branch_5x5 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False,
        )
        self.ms_branch_dilated = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2, bias=False,
        )
        self.ms_fuse = nn.Conv2d(
            3 * hidden_channels, hidden_channels, kernel_size=1, bias=False,
        )
        self.ms_norm = nn.GroupNorm(min(8, hidden_channels), hidden_channels)
        self.ms_act = nn.GELU()

        # --- residual blocks (reuse V2's _AdapterResBlock) ---
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_AdapterResBlock(hidden_channels, se_reduction))
        self.blocks = nn.ModuleList(blocks)

        # --- patchify projection ---
        self.patch_proj = nn.Conv2d(
            hidden_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )

        # --- residual gating ---
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.out_proj_scale_init = out_proj_scale_init
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.stem_conv.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.ms_branch_3x3.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.ms_branch_5x5.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.ms_branch_dilated.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.ms_fuse.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.patch_proj.weight, a=5 ** 0.5)
        self.patch_proj.weight.data.mul_(self.out_proj_scale_init)

    def forward(self, foa_feat: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        if foa_feat.ndim != 4 or foa_feat.size(1) != self.in_channels:
            raise ValueError(
                f"Expected foa_feat [B, {self.in_channels}, T_f, F], got {tuple(foa_feat.shape)}"
            )
        x = self.stem_act(self.stem_norm(self.stem_conv(foa_feat)))
        # multi-scale fuse
        b3 = self.ms_branch_3x3(x)
        b5 = self.ms_branch_5x5(x)
        bd = self.ms_branch_dilated(x)
        ms = torch.cat([b3, b5, bd], dim=1)
        ms = self.ms_act(self.ms_norm(self.ms_fuse(ms)))
        x = x + ms  # residual so zero-init of branches still lets x flow
        for block in self.blocks:
            x = block(x)
        delta_grid = self.patch_proj(x)
        t_p, f_p = delta_grid.shape[-2], delta_grid.shape[-1]
        delta_patch_tokens = delta_grid.flatten(2).transpose(1, 2).contiguous()
        delta_patch_tokens = self.residual_alpha * delta_patch_tokens
        return delta_patch_tokens, (t_p, f_p)
