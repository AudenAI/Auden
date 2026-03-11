#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Filter each sample's tags to only keep those present in a given label system.
Samples with no remaining tags after filtering are discarded.

Input:
    - A jsonl file with {"id": ..., "caption": ..., "tags": [...]}
    - A label system txt file (one tag per line) from Step 3

Output:
    - Filtered jsonl file
"""

import json
import argparse
import fnmatch
from pathlib import Path
from tqdm import tqdm


def load_label_system(label_file: str) -> set:
    with open(label_file, "r", encoding="utf-8") as f:
        labels = {line.strip() for line in f if line.strip()}
    print(f"  Loaded {len(labels)} labels from {label_file}")
    return labels


def filter_jsonl(input_file: str, label_set: set, output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = kept = removed = 0

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"  Filtering → {output_path.name}"):
            line = line.strip()
            if not line:
                continue
            processed += 1
            data = json.loads(line)

            filtered_tags = [t for t in data.get("tags", []) if t in label_set]

            if filtered_tags:
                data["tags"] = filtered_tags
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1

    print(f"  Processed: {processed} | Kept: {kept} | Removed (no valid tags): {removed}")
    return kept, removed


def main():
    parser = argparse.ArgumentParser(
        description="Filter tags in a jsonl file against a UTS label system."
    )
    parser.add_argument("--input", required=True,
                        help="Input jsonl file with 'tags' field")

    # Single-file mode
    single = parser.add_argument_group("Single mode")
    single.add_argument("--label", default=None,
                        help="Path to a single label system txt file")
    single.add_argument("--output", default=None,
                        help="Output jsonl path (single mode)")

    # Batch mode
    batch = parser.add_argument_group("Batch mode (multiple label files)")
    batch.add_argument("--label_dir", default=None,
                       help="Directory containing label system txt files")
    batch.add_argument("--label_pattern", default="label_*.txt",
                       help="Glob pattern for label files (default: label_*.txt)")
    batch.add_argument("--output_dir", default=None,
                       help="Output directory for filtered jsonl files (batch mode)")

    args = parser.parse_args()


    if args.label is None and args.label_dir is None:
        parser.error("Provide either --label (single mode) or --label_dir (batch mode).")

    if args.label:
        if args.output is None:
            stem = Path(args.input).stem
            label_stem = Path(args.label).stem
            args.output = f"{stem}_{label_stem}.jsonl"
        label_set = load_label_system(args.label)
        filter_jsonl(args.input, label_set, args.output)

    else:
        label_dir = Path(args.label_dir)
        label_files = sorted(label_dir.glob(args.label_pattern))
        if not label_files:
            print(f"No files matching '{args.label_pattern}' in {label_dir}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent / "filtered"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(label_files)} label file(s). Processing...\n")
        for label_file in label_files:
            print(f"[Label] {label_file.name}")
            label_set = load_label_system(str(label_file))
            out_name = Path(args.input).stem + f"_{label_file.stem}.jsonl"
            filter_jsonl(args.input, label_set, str(output_dir / out_name))
            print()

    print("All done.")


if __name__ == "__main__":
    main()