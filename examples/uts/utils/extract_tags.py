#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Extract structured tags from audio captions using an LLM (Qwen2.5-7B-Instruct).

Input:  A jsonl file where each line has {"id": ..., "caption": ...}
        (the caption field is the high-fidelity caption from Step 1)
Output: A jsonl file where each line has {"id": ..., "caption": ..., "tags": [...]}

"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams


TEMPLATE = """You are a labeling assistant for audio descriptions.
Extract essential, compact English labels for the audio caption.

Output rules:
- Output JSON only with a single key "labels": a list of 10 labels.
- Each label must be ONE WORD.
- Return ONLY a raw JSON object with key "labels". Do not use code fences or any extra text.
- Use lowercase.
- No full sentences, no punctuation beyond hyphens.
- Prefer canonical, domain-relevant terms with high reusability across captions.
- Avoid synonyms: collapse to a single consistent term set (e.g., always use "car" not "automobile").
- Avoid rare or obscure words; choose common, widely recognized terms.
- Cover five aspects when possible: scene/environment, sound sources, sound types/events, audio/production traits, semantic function/intent.
- Avoid duplicates.

Expected JSON:
{{"labels": ["xxx", "xxx", "xxx", "xxx", "xxx"]}}

Now process the following caption and return JSON only:
CAPTION:
{caption}
"""


def build_prompt(caption):
    return TEMPLATE.format(caption=caption.strip())


def parse_labels(text: str) -> list:
    try:
        json_str = text[text.find("{") : text.rfind("}") + 1]
        if json_str:
            return json.loads(json_str).get("labels", [])
    except Exception:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract tags from audio captions using a vLLM-served LLM."
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to the LLM (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--input", required=True,
                        help="Input jsonl file with 'id' and 'caption' fields")
    parser.add_argument("--output", required=True,
                        help="Output jsonl file path (adds 'tags' field)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Inference batch size")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens for LLM output")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max_input_tokens", type=int, default=32768,
                        help="Skip samples whose prompt exceeds this token length")
    parser.add_argument("--gpus", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES override")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---------- Load model ----------
    print(f"Loading model from {args.model_path} ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # ---------- load & filter data ----------
    print(f"Loading data from {args.input} ...")
    all_prompts, all_data, skipped = [], [], 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            caption = data.get("caption", "").strip()
            file_id = data.get("id", "")

            prompt = build_prompt(caption)
            if len(tokenizer.encode(prompt)) > args.max_input_tokens:
                print(f"  Skip id={file_id}: prompt too long")
                skipped += 1
                continue

            all_prompts.append(prompt)
            all_data.append({"id": file_id, "caption": caption})

    print(f"Loaded {len(all_prompts)} samples ({skipped} skipped as too long).")


    # ---------- batch inference ----------
    print("Running inference ...")
    error_cnt = 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(all_prompts), args.batch_size), desc="Batches"):
            batch_prompts = all_prompts[i : i + args.batch_size]
            batch_data    = all_data[i : i + args.batch_size]

            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            for data, out in zip(batch_data, outputs):
                text   = out.outputs[0].text.strip()
                labels = parse_labels(text)
                if not labels:
                    print(f"  Warning: failed to parse tags for id='{data['id']}'")
                    error_cnt += 1

                f_out.write(json.dumps(
                    {"id": data["id"], "caption": data["caption"], "tags": labels},
                    ensure_ascii=False,
                ) + "\n")

    print(f"\nDone. Parsing errors: {error_cnt}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()