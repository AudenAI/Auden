"""Prepare one FOA or mono recording for Spatial-Omni generation."""

import soundfile as sf
import torch

SAMPLE_RATE = 16000
MAX_SAMPLES = 20 * SAMPLE_RATE


def prepare_inputs(processor, audio_path, question, backbone, mono=False, spatial_dim=None):
    waveform, _ = sf.read(audio_path, frames=MAX_SAMPLES, dtype="float32", always_2d=True)
    waveform = waveform.T
    lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
    if hasattr(processor, "build_qa_text_parts"):
        prompt, _ = processor.build_qa_text_parts(prompt=question, answer="")
    else:
        prompt = processor.audio_token + processor.spatial_token + f"\n{question.rstrip()}\n"

    kwargs = {}
    audio = waveform
    if mono:
        token_lengths = processor._samples_to_so_backbone_tokens(lengths)
        kwargs["spatial_token_lengths"] = token_lengths
        kwargs["allow_mono_spatial_tokens"] = True
        if backbone == "so30b":
            kwargs["projected_spatial_tokens"] = torch.zeros(
                1, int(token_lengths[0]), spatial_dim, dtype=torch.float32,
            )
        else:
            audio = waveform[0]

    batch = processor(
        text=[prompt], audio=[audio], padding=True, padding_side="right",
        return_tensors="pt", **kwargs,
    )
    if not mono:
        batch["spatial_audio"] = torch.from_numpy(waveform.T).unsqueeze(0).contiguous()
        batch["spatial_audio_lengths"] = lengths
        batch["spatial_audio_attention_mask"] = torch.ones(1, waveform.shape[1], dtype=torch.bool)
    elif backbone == "so7b":
        batch["spatial_audio"] = torch.zeros(1, MAX_SAMPLES, 4, dtype=torch.float32)
        batch["spatial_audio_lengths"] = torch.zeros(1, dtype=torch.long)
        batch["spatial_audio_attention_mask"] = torch.zeros(1, MAX_SAMPLES, dtype=torch.bool)
        batch["has_spatial"] = torch.tensor([False])
        batch["mono_audio"] = torch.zeros(1, MAX_SAMPLES, dtype=torch.float32)
        batch["mono_audio"][0, :waveform.shape[1]] = torch.from_numpy(waveform[0])
        batch["mono_audio_lengths"] = lengths

    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
