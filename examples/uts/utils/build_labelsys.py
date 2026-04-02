#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3: Build Unified Tag System (UTS) from parsed tags using TF-IDF scoring.

Input:  A jsonl file where each line has {"id": ..., "caption": ..., "tags": [...]}
Output: Label system files at specified vocabulary sizes (e.g., label_800.txt, label_1k.txt)

"""

import json
import math
import argparse
from pathlib import Path
from collections import Counter


def build_label_system(jsonl_filepath: str):
    """
    Score(t) = df(t) * log((N+1) / (df(t)+1))
    where df(t) = number of documents containing tag t, N = total documents.
    """
    doc_frequency = Counter()
    total_documents = 0

    print(f"[Pass 1] Reading {jsonl_filepath} ...")
    with open(jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_documents += 1
            data = json.loads(line)
            unique_tags = set(data.get("tags", []))
            doc_frequency.update(unique_tags)

    print(f"[Pass 1] Done. {total_documents} documents, {len(doc_frequency)} unique tags.")

    N = total_documents
    tag_scores = {}
    for tag, df in doc_frequency.items():
        idf = math.log((N + 1) / (df + 1))   # matches paper formula: df * log((N+1)/(df+1))
        tag_scores[tag] = df * idf

    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_tags, doc_frequency, total_documents


def save_label_file(tags: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for tag in tags:
            f.write(f"{tag}\n")
    print(f"  Saved {len(tags)} tags → {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Build UTS label system files from parsed tag jsonl."
    )
    parser.add_argument("--input", required=True,
                        help="Input jsonl file (each line: {id, caption, tags})")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save label system txt files")
    parser.add_argument("--vocab_sizes", nargs="+", type=int,
                        default=[800, 1000, 1500, 2000, 3000],
                        help="Vocabulary sizes to export")
    parser.add_argument("--show_top", type=int, default=20,
                        help="Print top-N tags after scoring")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sorted_tags, doc_freq, total_docs = build_label_system(args.input)

    print(f"\n--- Top {args.show_top} Tags (by TF-IDF score) ---")
    for tag, score in sorted_tags[:args.show_top]:
        print(f"  {tag:<25} score={score:>9.2f}  df={doc_freq[tag]}")

    print("\n--- Document Frequency at key ranks ---")
    for k in args.vocab_sizes:
        if k <= len(sorted_tags):
            tag, score = sorted_tags[k - 1]
            print(f"  Rank {k:<5} → '{tag}'  df={doc_freq[tag]}")

    for k in args.vocab_sizes:
        top_k_tags = [tag for tag, _ in sorted_tags[:k]]
        name = f"label_{k}k.txt" if k >= 1000 and k % 1000 == 0 else f"label_{k}.txt"
        save_label_file(top_k_tags, output_dir / name)

    print("\nDone.")


if __name__ == "__main__":
    main()
