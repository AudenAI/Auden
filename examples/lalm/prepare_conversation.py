#!/usr/bin/env python3
"""Offline prepare cut.conversation for LALM training.

This script reads a Lhotse CutSet manifest, builds a conversation object for each
cut, writes it to ``cut.custom["conversation"]`` (accessible as ``cut.conversation``),
and saves a new manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lhotse import CutSet
from transformers import AutoTokenizer

from auden.utils.text_normalization import text_normalization


def _build_conversation(cut, instruction=None, system=None):
    response = cut.supervisions[0].text  # set it to your own response field
    response = text_normalization(
        response,
        case="lower",
        space_between_cjk=False,
        remove_diacritics=True,
        remove_symbols=True,
        remove_in_parenthesis=True,
        remove_in_brackets=True,
    )

    audio_source = cut.recording.sources[0].source

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    user_content = [{"type": "audio", "audio": audio_source}]
    if instruction:
        user_content.append({"type": "text", "text": str(instruction)})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})
    return messages


def _render_conversation(
    conversation: list[dict], audio_token: str = "<|audio|>"
) -> str:
    """Render conversation to Qwen-style chat text with im tags.

    Example:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "audio"}, {"type": "text", "text": "Please transcribe."}],
            },
            {"role": "assistant", "content": "Trading has almost stalled."},
        ]

        rendered_text =
            <|im_start|>user
            <|audio|>Please transcribe.<|im_end|>
            <|im_start|>assistant
            Trading has almost stalled.<|im_end|>
    """
    chunks = []
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type == "audio":
                    parts.append(audio_token)
                elif item_type == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    raise ValueError(f"Unknown content type: {item_type!r}")
            rendered_content = "".join(parts)
        else:
            rendered_content = str(content)
        chunks.append(f"<|im_start|>{role}\n{rendered_content}<|im_end|>\n")
    return "".join(chunks)


def _estimate_text_tokens(text: str, tokenizer) -> int:
    """Estimate text tokens with a real tokenizer."""
    text = text.strip()
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def prepare_cut(cut, tokenizer, instruction=None, system=None):
    if cut.custom is None:
        cut.custom = {}
    conversation = _build_conversation(cut, instruction, system)
    rendered_conversation = _render_conversation(conversation)
    cut.custom["conversation"] = conversation
    cut.custom["rendered_conversation"] = rendered_conversation
    cut.custom["num_text_tokens"] = _estimate_text_tokens(
        rendered_conversation,
        tokenizer,
    )
    return cut


def main():
    parser = argparse.ArgumentParser(
        description="Prepare cut.conversation offline for a CutSet."
    )
    parser.add_argument(
        "--input_manifest",
        required=True,
        help="Input CutSet manifest path (e.g. *.jsonl.gz).",
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        help="Output CutSet manifest path.",
    )
    parser.add_argument(
        "--instruction",
        help="Additional user prompt",
    )
    parser.add_argument(
        "--system",
        help="System for the conversation.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path used to estimate num_text_tokens.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    cuts = CutSet.from_file(args.input_manifest)
    prepared = cuts.map(
        lambda c: prepare_cut(
            c,
            tokenizer=tokenizer,
            instruction=args.instruction,
            system=args.system,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_file(str(out_path))
    print(f"Saved prepared CutSet to: {out_path}")


if __name__ == "__main__":
    main()
