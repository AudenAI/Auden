"""Read a small JSONL evaluation manifest without changing its QA metadata."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
from tqdm.auto import tqdm


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def record_id(record: dict) -> str:
    """Use the canonical SO-Bench qa_id when present, otherwise the example id."""
    value = record.get("qa_id")
    if value is None:
        value = record.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("qa_id or id must be a non-empty string")
    if value != value.strip():
        raise ValueError("qa_id or id must not have surrounding whitespace")
    return value


def load_manifest(path: str | Path, limit: int = 16) -> list[dict]:
    """Validate every JSONL row and return at most limit records (0 means all)."""
    path = Path(path)
    if limit < 0:
        raise ValueError("limit must be non-negative (0 means all)")
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(tqdm(handle, desc="Manifest", unit="rows"), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
                )
                if not isinstance(record, dict):
                    raise ValueError("each row must be a JSON object")
                for key in ("audio_path",):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f"{key} must be a non-empty string")
                identifier = record_id(record)
                if identifier in seen:
                    raise ValueError(f"duplicate id: {identifier}")
                seen.add(identifier)
                if "mono" in record and not isinstance(record["mono"], bool):
                    raise ValueError("mono must be a JSON boolean")
                if not any(key in record for key in ("question", "prompt")):
                    raise ValueError("question or prompt is required")
                for key in ("question", "prompt"):
                    if key in record and (
                        not isinstance(record[key], str) or not record[key].strip()
                    ):
                        raise ValueError(f"{key} must be a non-empty string when provided")
                if record.get("answer") is not None and not isinstance(record["answer"], str):
                    raise ValueError("answer must be a string or null")
                if any(key in record for key in ("prediction", "prediction_raw", "prediction_cleaned")):
                    raise ValueError("manifest must not contain prediction fields")
                if "\x00" in record["audio_path"]:
                    raise ValueError("audio_path must not contain a null byte")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if limit == 0 or len(records) < limit:
                records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def resolve_audio_path(
    audio_path: str, manifest_path: str | Path, audio_root: str | Path | None = None
) -> Path:
    """Resolve relative to audio_root or the manifest, leaving the source row intact."""
    path = Path(audio_path).expanduser()
    if not path.is_absolute():
        root = Path(audio_root).expanduser().resolve() if audio_root else Path(manifest_path).resolve().parent
        path = root / path
    return path.resolve()


def validate_audio(path: str | Path, mono: bool = False):
    """Require nonempty 16 kHz audio: four FOA channels, or explicit mono."""
    path = Path(path)
    if not isinstance(mono, bool):
        raise ValueError("mono must be a boolean")
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as error:
        raise ValueError(f"Cannot read audio header {path}: {error}") from error
    expected_channels = 1 if mono else 4
    if info.frames <= 0:
        raise ValueError(f"Audio file is empty: {path}")
    if info.samplerate != 16000 or info.channels != expected_channels:
        raise ValueError(
            f"Invalid audio {path}: {info.samplerate} Hz, {info.channels} channels; "
            f"expected 16000 Hz, {expected_channels} channels (mono={mono}). "
            "No resampling or channel conversion is performed."
        )
    return info
