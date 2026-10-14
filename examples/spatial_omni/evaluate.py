#!/usr/bin/env python3
"""Generate predictions for a JSONL manifest; score separately with score_sobench.py."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

from data import load_manifest, record_id, resolve_audio_path, validate_audio


def parse_args(argv=None):
    from model import add_model_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--manifest", required=True, help="JSONL: qa_id or id, audio_path, question or prompt; optional answer, mono and QA metadata.")
    parser.add_argument("--audio-root", help="Resolve relative audio paths under this directory instead of the manifest directory.")
    parser.add_argument("--output", "--output-jsonl", dest="output", help="New predictions JSONL, required unless --dry-run; existing files are never overwritten.")
    parser.add_argument("--limit", type=int, default=16, help="Maximum inference rows (default: 16); 0 selects all. The entire manifest is checked for invalid or duplicate records.")
    parser.add_argument("--num-workers", type=int, default=max(1, ((os.cpu_count() or 1) + 1) // 2), help="Parallel audio-header prechecks only; inference is serial.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and audio without loading a model or writing output.")
    return parser.parse_args(argv)


def run(args) -> int:
    """Precheck selected audio, then load one model and write each prediction once."""
    if args.num_workers < 1:
        raise ValueError("num-workers must be positive")
    if not args.output and not args.dry_run:
        raise ValueError("--output is required unless --dry-run is set")
    output = Path(args.output).expanduser().absolute() if args.output else None
    if output is not None and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"Output already exists: {output}. Choose a new path.")
    records = load_manifest(args.manifest, limit=args.limit)

    def check(record):
        path = resolve_audio_path(record["audio_path"], args.manifest, getattr(args, "audio_root", None))
        info = validate_audio(path, mono=record.get("mono", False))
        logging.info(
            "Audio id=%s path=%s duration=%.3fs sample_rate=%s channels=%s",
            record_id(record), path, info.duration, info.samplerate, info.channels,
        )
        return path

    with ThreadPoolExecutor(max_workers=min(args.num_workers, len(records))) as pool:
        paths = list(tqdm(pool.map(check, records), total=len(records), desc="Audio precheck", unit="clips"))
    logging.info("Validated %d records; predictions output=%s", len(records), output)
    if args.dry_run:
        return len(records)

    from model import load_model

    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    # Exclusive creation also protects against another process creating this path.
    with output.open("x", encoding="utf-8") as handle:
        try:
            model = load_model(args)
            for record, audio_path in tqdm(zip(records, paths), total=len(records), desc="Inference", unit="clips"):
                prediction = model.answer(
                    str(audio_path), record.get("prompt") or record.get("question"),
                    mono=record.get("mono", False),
                )
                if not isinstance(prediction, str) or not prediction.strip():
                    raise ValueError(f"Model returned an empty/non-string prediction for id={record_id(record)}")
                result = dict(record)
                result["prediction"] = prediction
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                completed += 1
        except Exception:
            logging.exception(
                "Evaluation stopped after %d/%d records; partial output retained at %s. "
                "Choose a new output path before retrying.",
                completed, len(records), output,
            )
            raise
    logging.info("Wrote %d predictions to %s", completed, output)
    return completed


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
