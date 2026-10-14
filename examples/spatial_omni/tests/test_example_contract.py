"""Small CPU contract tests; no model weights or network services are required."""

import argparse
import json
from pathlib import Path
import sys
import types

import pytest
import soundfile as sf

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

from data import load_manifest, resolve_audio_path, validate_audio
import evaluate


def write_audio(path, channels=4, samplerate=16000, frames=32):
    sf.write(str(path), [[0.0] * channels] * frames, samplerate)
    return path


def write_manifest(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return path


def arguments(tmp_path, manifest, **kwargs):
    values = dict(manifest=str(manifest), audio_root=None, output=str(tmp_path / "predictions.jsonl"), limit=16, num_workers=2, dry_run=False)
    values.update(kwargs)
    return argparse.Namespace(**values)


def fake_model(monkeypatch, answers=None):
    calls = []
    loads = []

    class FakeModel:
        def answer(self, path, question, mono=False):
            calls.append((path, question, mono))
            return answers[len(calls) - 1] if answers else "90 degrees"

    def load_model(args):
        loads.append(args)
        return FakeModel()

    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(load_model=load_model))
    return loads, calls


def test_relative_paths_metadata_prompt_and_single_load(tmp_path, monkeypatch):
    audio = write_audio(tmp_path / "scene.wav")
    rows = [
        {"qa_id": "gold-a", "audio_path": "scene.wav", "question": "Where?", "prompt": "Where? A. left B. right", "answer": "90 degrees", "answer_meta": {"azimuth_deg": 90}, "task_name": "estimate_azimuth"},
        {"id": "b", "pair_id": "gold-b", "audio_path": "scene.wav", "question": "Direction?", "canonical_answer": "90"},
    ]
    manifest = write_manifest(tmp_path / "manifest.jsonl", rows)
    loads, calls = fake_model(monkeypatch)
    args = arguments(tmp_path, manifest)
    assert evaluate.run(args) == 2
    output = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
    assert output == [dict(row, prediction="90 degrees") for row in rows]
    assert len(loads) == 1
    assert calls == [(str(audio.resolve()), rows[0]["prompt"], False), (str(audio.resolve()), rows[1]["question"], False)]
    assert resolve_audio_path("scene.wav", manifest) == audio.resolve()


@pytest.mark.parametrize("bad", [
    {"id": "a", "audio_path": "scene.wav", "question": ""},
    {"id": 1, "audio_path": "scene.wav", "question": "Q"},
    {"id": "a", "audio_path": "scene.wav", "question": "Q", "mono": "false"},
    {"id": "a", "audio_path": "scene.wav", "question": "Q", "prompt": ""},
    {"id": "a", "audio_path": "scene.wav", "question": "Q", "prediction_cleaned": "stale"},
    ["not", "an", "object"],
])
def test_invalid_manifest_rows_rejected(tmp_path, bad):
    manifest = write_manifest(tmp_path / "manifest.jsonl", [bad])
    with pytest.raises(ValueError, match="manifest.jsonl:1"):
        load_manifest(manifest)


@pytest.mark.parametrize("text", [
    '{"id":"a","id":"b","audio_path":"scene.wav","question":"Q"}',
    '{"id":"a","audio_path":"scene.wav","question":"Q","answer_meta":{"x":NaN}}',
    '{"id":',
])
def test_ambiguous_or_invalid_json_rejected(tmp_path, text):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(manifest)


def test_duplicates_after_limit_are_rejected(tmp_path):
    row = {"id": "a", "audio_path": "scene.wav", "question": "Q"}
    manifest = write_manifest(tmp_path / "manifest.jsonl", [row, row])
    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(manifest, limit=1)


@pytest.mark.parametrize("channels,samplerate,mono", [(1, 16000, False), (4, 16000, True), (2, 16000, False), (4, 48000, False)])
def test_audio_layout_is_explicit(tmp_path, channels, samplerate, mono):
    audio = write_audio(tmp_path / "audio.wav", channels=channels, samplerate=samplerate)
    with pytest.raises(ValueError, match="expected 16000 Hz"):
        validate_audio(audio, mono=mono)


def test_explicit_mono_and_bad_audio(tmp_path):
    audio = write_audio(tmp_path / "mono.wav", channels=1)
    assert validate_audio(audio, mono=True).channels == 1
    with pytest.raises(ValueError, match="does not exist"):
        validate_audio(tmp_path / "missing.wav")
    audio.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Cannot read audio header"):
        validate_audio(audio)


def test_dry_run_does_not_load_or_write(tmp_path, monkeypatch):
    write_audio(tmp_path / "scene.wav")
    manifest = write_manifest(tmp_path / "manifest.jsonl", [{"id": "a", "audio_path": "scene.wav", "question": "Q"}])
    loads, calls = fake_model(monkeypatch)
    args = arguments(tmp_path, manifest, dry_run=True, output=str(tmp_path / "new-dir" / "predictions.jsonl"))
    assert evaluate.run(args) == 1
    assert not loads and not calls
    assert not Path(args.output).parent.exists()


def test_existing_output_and_invalid_audio_fail_before_model_load(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path / "manifest.jsonl", [{"id": "a", "audio_path": "missing.wav", "question": "Q"}])
    loads, calls = fake_model(monkeypatch)
    args = arguments(tmp_path, manifest)
    Path(args.output).write_text("keep me\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate.run(args)
    assert Path(args.output).read_text() == "keep me\n"
    args.output = str(tmp_path / "new.jsonl")
    with pytest.raises(ValueError, match="does not exist"):
        evaluate.run(args)
    assert not Path(args.output).exists()
    assert not loads and not calls


def test_limit_and_partial_failure_are_visible(tmp_path, monkeypatch):
    write_audio(tmp_path / "scene.wav")
    rows = [{"id": str(i), "audio_path": "scene.wav", "question": "Q"} for i in range(3)]
    manifest = write_manifest(tmp_path / "manifest.jsonl", rows)
    loads, calls = fake_model(monkeypatch, answers=["valid", ""])
    args = arguments(tmp_path, manifest, limit=2)
    with pytest.raises(ValueError, match="empty/non-string"):
        evaluate.run(args)
    assert len(calls) == 2 and len(loads) == 1
    assert [json.loads(line)["id"] for line in Path(args.output).read_text().splitlines()] == ["0"]



def test_explicit_audio_root_and_broken_output_symlink(tmp_path, monkeypatch):
    audio_root = tmp_path / "dataset"
    audio_root.mkdir()
    write_audio(audio_root / "scene.wav")
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    rows = [{"id": "a", "audio_path": "scene.wav", "question": "Q"}]
    manifest = write_manifest(qa_dir / "test.jsonl", rows)
    loads, calls = fake_model(monkeypatch)
    args = arguments(tmp_path, manifest, audio_root=str(audio_root))
    assert evaluate.run(args) == 1
    assert calls[0][0] == str((audio_root / "scene.wav").resolve())
    assert json.loads(Path(args.output).read_text())["audio_path"] == "scene.wav"
    target = tmp_path / "missing-target.jsonl"
    output = tmp_path / "symlink.jsonl"
    output.symlink_to(target)
    args.output = str(output)
    with pytest.raises(FileExistsError):
        evaluate.run(args)
    assert output.is_symlink() and not target.exists()
    assert len(loads) == 1

def test_qa_id_precedence_and_required_ids(tmp_path):
    rows = [
        {"id": "local-a", "qa_id": "same-qa", "audio_path": "scene.wav", "question": "Q"},
        {"id": "local-b", "qa_id": "same-qa", "audio_path": "scene.wav", "question": "Q"},
    ]
    manifest = write_manifest(tmp_path / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="duplicate id: same-qa"):
        load_manifest(manifest)
    write_manifest(manifest, [{"audio_path": "scene.wav", "question": "Q"}])
    with pytest.raises(ValueError, match="qa_id or id"):
        load_manifest(manifest)

def test_prompt_only_generation_and_dry_run_without_output(tmp_path, monkeypatch):
    write_audio(tmp_path / "scene.wav")
    row = {"qa_id": "a", "audio_path": "scene.wav", "prompt": "Choose A. left B. right"}
    manifest = write_manifest(tmp_path / "manifest.jsonl", [row])
    loads, calls = fake_model(monkeypatch)
    args = arguments(tmp_path, manifest, output=None, dry_run=True)
    assert evaluate.run(args) == 1
    assert not loads and not calls
    args.dry_run = False
    with pytest.raises(ValueError, match="--output is required"):
        evaluate.run(args)
    assert not loads and not calls
    args.output = str(tmp_path / "predictions.jsonl")
    assert evaluate.run(args) == 1
    assert calls[0][1] == row["prompt"]
    assert json.loads(Path(args.output).read_text()) == dict(row, prediction="90 degrees")


def test_manifest_requires_question_or_prompt(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.jsonl", [{"id": "a", "audio_path": "scene.wav"}])
    with pytest.raises(ValueError, match="question or prompt"):
        load_manifest(manifest)
