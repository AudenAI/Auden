"""Evaluation / decoding script for lalm.

Assumes test manifests have been prepared by prepare_conversation.py.

Usage::

    python evaluate.py \\
        exp_dir=/path/to/exp \\
        checkpoint.iter=5000 \\
        data.test_data_config=configs/valid_data_config.yaml
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import yaml
from lalm_core.model import LALMConfig, LALMForConditionalGeneration, LALMProcessor
from lhotse import CutSet, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler
from omegaconf import DictConfig, OmegaConf
from results_utils import save_results
from transformers.modeling_utils import no_init_weights

from auden.utils.text_normalization import text_normalization

# ---------------------------------------------------------------------------
# Model loading  (standard MLLM from_pretrained pattern)
# ---------------------------------------------------------------------------


def _has_hf_weights(model_dir: str) -> bool:
    return any(
        os.path.exists(os.path.join(model_dir, n))
        for n in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _resolve_checkpoint_path(cfg: DictConfig) -> tuple[str, str]:
    ckpt_cfg = cfg.checkpoint
    filename = ckpt_cfg.get("filename", None)
    if filename:
        path = (
            filename if os.path.isabs(filename) else os.path.join(cfg.exp_dir, filename)
        )
        return path, Path(filename).stem

    iters = int(ckpt_cfg.get("iter", 0))
    epoch = int(ckpt_cfg.get("epoch", 0))
    if iters > 0:
        return os.path.join(cfg.exp_dir, f"checkpoint-{iters}.pt"), f"iter-{iters}"
    if epoch > 0:
        return os.path.join(cfg.exp_dir, f"epoch-{epoch}.pt"), f"epoch-{epoch}"
    raise ValueError(
        "[evaluate] Specify checkpoint.filename, checkpoint.iter, or checkpoint.epoch."
    )


def prepare_model_dir(cfg: DictConfig) -> tuple[str, str]:
    """Return (model_dir, results_suffix).

    If the target dir already has HF weights, use it directly (from_pretrained).
    Otherwise export from the trainer checkpoint and save via save_pretrained,
    caching to {exp_dir}/export/{suffix}/ for reuse.
    """
    explicit_dir = cfg.checkpoint.get("model_dir", None)
    checkpoint_path, results_suffix = _resolve_checkpoint_path(cfg)

    model_dir = explicit_dir or os.path.join(cfg.exp_dir, "export", results_suffix)
    if explicit_dir:
        results_suffix = Path(explicit_dir).name

    if _has_hf_weights(model_dir):
        logging.info(f"[evaluate] Using existing model dir: {model_dir}")
        return model_dir, results_suffix

    logging.info(f"[evaluate] Exporting {checkpoint_path} -> {model_dir}")
    hf_dir = os.path.join(cfg.exp_dir, "hf")
    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(
            f"[evaluate] hf_dir not found: {hf_dir}. Run train.py first."
        )

    with no_init_weights():
        model = LALMForConditionalGeneration(LALMConfig.from_pretrained(hf_dir))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    LALMProcessor.from_pretrained(hf_dir).save_pretrained(model_dir)
    logging.info(f"[evaluate] Saved to {model_dir}")
    del model, checkpoint, state_dict
    return model_dir, results_suffix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ASST_PREFIX = "<|im_start|>assistant\n"


def _strip_last_assistant(rendered: str) -> str:
    """Remove the last assistant turn and return the prompt up to (and including)
    the generation prefix '<|im_start|>assistant\\n'."""
    idx = rendered.rfind(_ASST_PREFIX)
    if idx == -1:
        return rendered + _ASST_PREFIX
    return rendered[: idx + len(_ASST_PREFIX)]


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
@torch.no_grad()
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    set_audio_duration_mismatch_tolerance(0.1)

    model_dir, results_file_suffix = prepare_model_dir(cfg)

    dtype_name = cfg.get("dtype", "fp16")
    dtype = (
        torch.float16
        if dtype_name == "fp16"
        else torch.bfloat16 if dtype_name == "bf16" else torch.float32
    )

    model = LALMForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
    processor = LALMProcessor.from_pretrained(model_dir)
    # Decoder-only generation requires left-padding so every sample's last real
    # token sits at the same position (L_max - 1) and the next generated token
    # is correctly placed at L_max for all samples in the batch.
    processor.tokenizer.padding_side = "left"
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device).eval()
    logging.info(
        f"[evaluate] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    max_new_tokens = int(cfg.get("max_new_tokens", 200))
    generate_config = (
        dict(max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        if cfg.decoding_method == "greedy_search"
        else dict(
            max_new_tokens=max_new_tokens,
            num_beams=int(cfg.get("num_beams", 4)),
            do_sample=False,
        )
    )

    sampling_rate = int(cfg.data.get("sampling_rate", 16000))

    with open(cfg.data.test_data_config) as f:
        test_data_config = yaml.load(f, Loader=yaml.FullLoader)

    res_dir = Path(cfg.exp_dir) / cfg.decoding_method
    os.makedirs(res_dir, exist_ok=True)

    for test_set in test_data_config:
        logging.info(f"[evaluate] Test set: {test_set['name']}")
        cutset = CutSet.from_file(test_set["manifest"]).resample(sampling_rate)
        sampler = DynamicBucketingSampler(
            cutset, max_duration=cfg.data.max_duration, shuffle=False
        )
        results = defaultdict(list)
        num_cuts = 0

        for batch_idx, cuts in enumerate(sampler):
            cuts = cuts.sort_by_duration(ascending=False)

            # Load raw audio (processor handles feature extraction + packing internally)
            audios = [cut.load_audio()[0] for cut in cuts]

            # Build prompts from pre-rendered text (matches training exactly).
            # Strip the last assistant turn and append the generation prefix.
            texts = [_strip_last_assistant(cut.rendered_conversation) for cut in cuts]

            inputs = processor(
                text=texts,
                audio=audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                **generate_config,
            )

            hyps = processor.batch_decode(output_ids, skip_special_tokens=True)
            refs = [cut.supervisions[0].text for cut in cuts]
            cut_ids = [cut.id for cut in cuts]

            def norm(s):
                return text_normalization(
                    s,
                    case="lower",
                    remove_diacritics=True,
                    simplified_chinese=True,
                    space_between_cjk=True,
                ).split()

            for cut_id, ref, hyp in zip(cut_ids, refs, hyps):
                results[cfg.decoding_method].append((cut_id, norm(ref), norm(hyp)))

            num_cuts += len(cut_ids)
            if batch_idx % 50 == 0:
                logging.info(f"  batch {batch_idx}, cuts: {num_cuts}")

        save_results(res_dir, test_set["name"], results, suffix=results_file_suffix)


if __name__ == "__main__":
    main()
