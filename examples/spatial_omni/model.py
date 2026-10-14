"""Load Spatial-Omni checkpoints and answer audio questions."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import logging
from pathlib import Path
import sys

from data import validate_audio

EXAMPLE_ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "so7b": {"so": "SO-7B/so/SO-7B.pt", "mix": "SO-7B/mix/SO-7B-MIX.pt"},
    "so30b": {"so": "SO-30B/so/SO-30B.pt", "mix": "SO-30B/mix/SO-30B-MIX.pt"},
}


def read_config(config_path):
    import yaml

    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"backbone", "variant", "repo_id", "revision", "device", "dtype",
                "attn_impl", "max_new_tokens", "seed"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError(f"Config requires: {', '.join(sorted(required))}")
    if config.keys() - required - {"base_revision"}:
        raise ValueError("Unknown model configuration fields.")
    if config["backbone"] not in CHECKPOINTS:
        raise ValueError("backbone must be so7b or so30b.")
    if config["variant"] not in ("so", "mix"):
        raise ValueError("variant must be so or mix.")
    if config["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("dtype must be bfloat16, float16 or float32.")
    if config["attn_impl"] not in ("sdpa", "eager", "flash_attention_2"):
        raise ValueError("Unsupported attn_impl.")
    if type(config["max_new_tokens"]) is not int or config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    return config


def resolve_checkpoint(config, checkpoint_dir=None, encoder_checkpoint=None):
    filename = CHECKPOINTS[config["backbone"]][config["variant"]]
    if checkpoint_dir is None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.constants import HF_HOME

        folder = str(Path(filename).parent)
        cache_key = hashlib.sha256(
            f'{config["repo_id"]}@{config["revision"]}'.encode()
        ).hexdigest()[:16]
        checkpoint_dir = snapshot_download(
            repo_id=config["repo_id"], revision=config["revision"],
            local_dir=Path(HF_HOME) / "auden" / "spatial_omni" / cache_key,
            allow_patterns=[
                filename, f"{folder}/train_args.json",
                *([] if encoder_checkpoint else ["SO-Encoder/SO-Encoder.pt"]),
            ],
        )
    root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint = root / filename
    encoder = (Path(encoder_checkpoint).expanduser().resolve() if encoder_checkpoint
               else root / "SO-Encoder/SO-Encoder.pt")
    for path in (checkpoint, checkpoint.parent / "train_args.json", encoder):
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {path}")
    return checkpoint, encoder


class SpatialOmniModel:
    @classmethod
    def from_config(cls, config_path, checkpoint_dir=None, base_model=None,
                    device=None, encoder_checkpoint=None):
        config = read_config(config_path)
        checkpoint, encoder = resolve_checkpoint(config, checkpoint_dir, encoder_checkpoint)
        with (checkpoint.parent / "train_args.json").open(encoding="utf-8") as stream:
            settings = json.load(stream)
        is_mix = bool(settings.get("mixed_spatial_replay", False))
        if is_mix != (config["variant"] == "mix"):
            raise ValueError("The selected config and checkpoint disagree on MIX mode.")

        from transformers import set_seed
        from loading import load_model as load_pretrained_model

        set_seed(config["seed"])
        with redirect_stdout(sys.stderr):
            model, processor = load_pretrained_model(
                config, checkpoint, encoder, base_model, device or config["device"],
            )
        instance = cls()
        instance.model = model
        instance.processor = processor
        instance.config = config
        instance.is_mix = is_mix
        logging.info("Loaded %s", checkpoint)
        return instance

    def answer(self, audio_path, question, mono=False):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string.")
        if mono and not self.is_mix:
            raise ValueError("Mono input requires a MIX config and checkpoint.")
        audio = Path(audio_path).expanduser().resolve()
        info = validate_audio(audio, mono=mono)
        logging.info("Audio: %s; %.2f s, %d Hz, %d channels",
                     audio, info.duration, info.samplerate, info.channels)
        if info.duration > 20:
            logging.warning("Using the first 20 seconds of the recording.")

        import torch
        from processing import prepare_inputs

        thinker = getattr(self.model, "thinker", self.model)
        text_config = getattr(thinker.config, "text_config", None) or thinker.config
        inputs = prepare_inputs(
            self.processor, str(audio), question, self.config["backbone"],
            mono=mono, spatial_dim=text_config.hidden_size,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, return_audio=False, do_sample=False, num_beams=1,
                max_new_tokens=self.config["max_new_tokens"],
            )
        return self.processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:].cpu(), skip_special_tokens=True,
        ).strip()


def add_model_arguments(parser):
    parser.add_argument("--config", default=str(EXAMPLE_ROOT / "configs/so7b.yaml"))
    parser.add_argument("--checkpoint-dir", help="Local HF release directory; otherwise download the selected checkpoint.")
    parser.add_argument("--encoder-checkpoint", help="Use an SO-Encoder.pt outside the release directory.")
    parser.add_argument("--base-model", help="Local base model directory or Hugging Face model ID.")
    parser.add_argument("--device", help="Override the config's device, e.g. cuda:0.")


def load_model(args):
    return SpatialOmniModel.from_config(
        args.config, checkpoint_dir=args.checkpoint_dir, base_model=args.base_model,
        device=args.device, encoder_checkpoint=getattr(args, "encoder_checkpoint", None),
    )
