"""CPU checks for generation inputs using a small fake processor."""

from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

from processing import MAX_SAMPLES, SAMPLE_RATE, prepare_inputs


class FakeProcessor:
    audio_token = "<audio>"
    spatial_token = "<spatial>"

    def __init__(self):
        self.calls = []

    def _samples_to_so_backbone_tokens(self, lengths):
        return torch.round(lengths * (2.5 / SAMPLE_RATE)).long()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch = {
            "input_ids": torch.tensor([[11, 12, 13]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "metadata": "not a model input",
        }
        for name in ("spatial_token_lengths", "projected_spatial_tokens"):
            if name in kwargs:
                batch[name] = kwargs[name]
        return batch


class ChatProcessor(FakeProcessor):
    def __init__(self):
        super().__init__()
        self.text_calls = []

    def build_qa_text_parts(self, *, prompt, answer):
        self.text_calls.append((prompt, answer))
        return f"USER: {prompt}\nASSISTANT:", answer


def audio_file(tmp_path, channels, frames=2 * SAMPLE_RATE):
    waveform = np.tile(np.arange(1, channels + 1, dtype=np.float32) / 10, (frames, 1))
    path = tmp_path / "input.wav"
    sf.write(path, waveform, SAMPLE_RATE, subtype="FLOAT")
    return path, waveform


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
def test_foa_preserves_channels_and_only_supplies_question(tmp_path, backbone):
    path, waveform = audio_file(tmp_path, channels=4)
    processor = ChatProcessor() if backbone == "so30b" else FakeProcessor()
    question = "Which source is on the left?"
    batch = prepare_inputs(processor, path, question, backbone)
    call = processor.calls[0]
    assert np.array_equal(call["audio"][0], waveform.T)
    assert question in call["text"][0]
    assert "answer" not in call and "labels" not in batch
    if backbone == "so30b":
        assert processor.text_calls == [(question, "")]
    assert torch.equal(batch["spatial_audio"][0], torch.from_numpy(waveform))
    assert batch["spatial_audio_lengths"].tolist() == [len(waveform)]
    assert bool(batch["spatial_audio_attention_mask"].all())
    assert "has_spatial" not in batch and "projected_spatial_tokens" not in batch
    assert all(isinstance(value, torch.Tensor) for value in batch.values())


def test_so7b_mono_uses_learned_null_inputs(tmp_path):
    path, waveform = audio_file(tmp_path, channels=1)
    processor = FakeProcessor()
    batch = prepare_inputs(processor, path, "What is audible?", "so7b", mono=True, spatial_dim=3584)
    call = processor.calls[0]
    assert call["audio"][0].ndim == 1
    assert call["allow_mono_spatial_tokens"] is True
    assert batch["spatial_token_lengths"].tolist() == [5]
    assert batch["has_spatial"].tolist() == [False]
    assert batch["spatial_audio"].shape == (1, MAX_SAMPLES, 4)
    assert torch.count_nonzero(batch["spatial_audio"]) == 0
    assert not bool(batch["spatial_audio_attention_mask"].any())
    assert batch["spatial_audio_lengths"].tolist() == [0]
    assert batch["mono_audio_lengths"].tolist() == [len(waveform)]
    assert torch.equal(batch["mono_audio"][0, :len(waveform)], torch.from_numpy(waveform[:, 0]))
    assert torch.count_nonzero(batch["mono_audio"][0, len(waveform):]) == 0
    assert "projected_spatial_tokens" not in batch


def test_so30b_mono_uses_zero_projected_tokens(tmp_path):
    path, waveform = audio_file(tmp_path, channels=1)
    processor = ChatProcessor()
    batch = prepare_inputs(processor, path, "What is audible?", "so30b", mono=True, spatial_dim=2048)
    call = processor.calls[0]
    assert np.array_equal(call["audio"][0], waveform.T)
    assert call["allow_mono_spatial_tokens"] is True
    assert batch["projected_spatial_tokens"].shape == (1, 5, 2048)
    assert torch.count_nonzero(batch["projected_spatial_tokens"]) == 0
    assert batch["spatial_token_lengths"].tolist() == [5]
    assert "has_spatial" not in batch and "mono_audio" not in batch


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
def test_input_is_limited_to_twenty_seconds_without_overwriting(tmp_path, backbone):
    path, waveform = audio_file(tmp_path, channels=4, frames=MAX_SAMPLES + SAMPLE_RATE)
    processor = FakeProcessor()
    batch = prepare_inputs(processor, path, "Where is the source?", backbone)
    assert processor.calls[0]["audio"][0].shape == (4, MAX_SAMPLES)
    assert batch["spatial_audio"].shape == (1, MAX_SAMPLES, 4)
    assert batch["spatial_audio_lengths"].tolist() == [MAX_SAMPLES]
    assert sf.info(path).frames == len(waveform)
