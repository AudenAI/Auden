"""CPU checks for local imports, release downloads and model setup."""

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))

import model


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
@pytest.mark.parametrize("variant,external_encoder", [("so", False), ("mix", False), ("so", True)])
def test_selective_download_preserves_adjacent_config(tmp_path, monkeypatch, backbone, variant, external_encoder):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HOME = str(tmp_path / "hf-cache")

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        for name in kwargs["allow_patterns"]:
            file = destination / name
            file.parent.mkdir(parents=True, exist_ok=True)
            # Placeholder files exercise layout without deserializing weights.
            file.write_text("{}", encoding="utf-8")
        return str(destination)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    filename = backbone + ("_mix.yaml" if variant == "mix" else ".yaml")
    config = model.read_config(EXAMPLE / "configs" / filename)
    override = tmp_path / "shared-encoder.pt" if external_encoder else None
    if override:
        override.write_bytes(b"placeholder")
    checkpoint, encoder = model.resolve_checkpoint(config, encoder_checkpoint=override)
    assert len(calls) == 1
    request = calls[0]
    assert request["repo_id"] == config["repo_id"]
    assert request["revision"] == config["revision"]
    assert set(request["allow_patterns"]) == {
        model.CHECKPOINTS[backbone][variant],
        str(Path(model.CHECKPOINTS[backbone][variant]).parent / "train_args.json"),
        *([] if external_encoder else ["SO-Encoder/SO-Encoder.pt"]),
    }
    assert Path(request["local_dir"]).is_relative_to(tmp_path / "hf-cache")
    assert not checkpoint.is_symlink()
    assert (checkpoint.resolve().parent / "train_args.json").is_file()
    assert encoder.is_file()
    if override:
        assert encoder == override.resolve()
    assert model.resolve_checkpoint(config, request["local_dir"], override) == (checkpoint, encoder)
    assert len(calls) == 1
    other_checkpoint, _ = model.resolve_checkpoint(
        dict(config, revision="another-revision"), encoder_checkpoint=override,
    )
    assert other_checkpoint != checkpoint


def release_files(tmp_path, mixed):
    checkpoint = tmp_path / "checkpoint.pt"
    encoder = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"placeholder")
    encoder.write_bytes(b"placeholder")
    (tmp_path / "train_args.json").write_text(
        json.dumps({"mixed_spatial_replay": mixed}), encoding="utf-8",
    )
    return checkpoint, encoder


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
@pytest.mark.parametrize("variant,saved_mix", [("so", True), ("mix", False)])
def test_mix_mismatch_fails_before_heavy_imports(tmp_path, monkeypatch, backbone, variant, saved_mix):
    paths = release_files(tmp_path, saved_mix)
    monkeypatch.setattr(model, "resolve_checkpoint", lambda *args: paths)
    for name in ("transformers", "torch", "loading"):
        monkeypatch.setitem(sys.modules, name, None)
    filename = backbone + ("_mix.yaml" if variant == "mix" else ".yaml")
    with pytest.raises(ValueError, match="disagree on MIX mode"):
        model.SpatialOmniModel.from_config(EXAMPLE / "configs" / filename)


@pytest.mark.parametrize("backbone", ["so7b", "so30b"])
def test_local_loader_receives_runtime_overrides(tmp_path, monkeypatch, backbone):
    paths = release_files(tmp_path, mixed=True)
    calls = []
    seed_calls = []
    pretrained_model, processor = object(), object()
    loading = types.ModuleType("loading")

    def load_model(config, checkpoint, encoder, base_model, device):
        calls.append((config, checkpoint, encoder, base_model, device))
        return pretrained_model, processor

    loading.load_model = load_model
    transformers = types.ModuleType("transformers")
    transformers.set_seed = seed_calls.append
    monkeypatch.setitem(sys.modules, "loading", loading)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(model, "resolve_checkpoint", lambda *args: paths)
    config_path = EXAMPLE / "configs" / f"{backbone}_mix.yaml"
    instance = model.SpatialOmniModel.from_config(
        config_path, base_model="base-override", device="cuda:1",
        encoder_checkpoint=paths[1],
    )
    assert instance.model is pretrained_model and instance.processor is processor
    assert instance.is_mix is True
    assert calls == [(model.read_config(config_path), *paths, "base-override", "cuda:1")]
    assert seed_calls == [instance.config["seed"]]


@pytest.mark.parametrize("script", ["infer.py", "evaluate.py"])
def test_help_avoids_model_and_download_imports(script):
    code = """
import importlib.abc
from pathlib import Path
import runpy
import sys

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "huggingface_hub"}:
            raise AssertionError("Unexpected heavyweight import: " + fullname)
        return None

target = Path(sys.argv[1])
sys.meta_path.insert(0, RejectHeavyImports())
sys.path.insert(0, str(target.parent))
sys.argv = [str(target), "--help"]
runpy.run_path(str(target), run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(EXAMPLE / script)],
        text=True, capture_output=True, check=True,
    )
    assert "--config" in result.stdout and "--checkpoint-dir" in result.stdout
    assert "--source-root" not in result.stdout


def test_inference_imports_are_self_contained():
    code = """
import importlib
import importlib.metadata
from pathlib import Path
import sys

example = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(example))
modules = ["spatial_omni.encoders.beats.so_backbone",
           "spatial_omni.modules.so_encoder",
           "spatial_omni.modules.so_token_projector",
           "spatial_omni.utils.release", "loading"]
major = int(importlib.metadata.version("transformers").split(".")[0])
if major < 5:
    modules += ["spatial_omni.model.configuration",
                "spatial_omni.model.modeling_so_thinker",
                "spatial_omni.model.processing_so"]
else:
    for pattern in ("configuration_*3_omni.py", "modeling_so_thinker_*3.py",
                    "processing_so_*3.py"):
        matches = list((example / "spatial_omni/model").glob(pattern))
        assert len(matches) == 1
        modules.append("spatial_omni.model." + matches[0].stem)
for name in modules:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(example)
for name, module in tuple(sys.modules.items()):
    assert not any(part in name for part in
                   ("train_so_qa", "train_so_pretrain", "so_loss",
                    "seld_backbone", "iv_spatial_adapters"))
    if name.startswith("spatial_omni") and getattr(module, "__file__", None):
        assert Path(module.__file__).resolve().is_relative_to(example)
print("Local inference imports passed")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("SPATIAL_OMNI_ROOT", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(EXAMPLE)],
        text=True, capture_output=True, check=True, env=environment,
    )
    assert "Local inference imports passed" in result.stdout
