#!/usr/bin/env python3
"""
TagSpeech decode script.
Generate XML outputs for multi-speaker conversations without evaluation.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import yaml
from lhotse import CutSet, Fbank, FbankConfig, set_audio_duration_mismatch_tolerance
from lhotse.dataset import DynamicBucketingSampler, OnTheFlyFeatures, SimpleCutSampler
from multi_speaker_dataset import MultiSpeakerAsrDataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils.xml_utils import construct_multi_speaker_xml

from auden.auto.auto_model import AutoModel
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints


def get_test_dataloaders(cfg):
    """Prepare test dataloaders."""
    test_dls = []
    test_names = []

    input_strategy = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))

    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(
            getattr(cfg.data, "sampling_rate", 16000)
        )
        # Optional decode-time filtering to avoid OOM
        min_dur = cfg.decode.get("min_duration", 0.0) if hasattr(cfg, "decode") else 0.0
        max_dur = (
            cfg.decode.get("max_duration", float("inf"))
            if hasattr(cfg, "decode")
            else float("inf")
        )
        max_sups = (
            cfg.decode.get("max_supervisions_per_cut", 10)
            if hasattr(cfg, "decode")
            else 10
        )

        def _decode_filter(c):
            if c.duration < min_dur or c.duration > max_dur:
                return False
            if hasattr(c, "supervisions") and len(c.supervisions) > max_sups:
                return False
            return True

        try:
            cutset = cutset.filter(_decode_filter)
            logging.info(
                f"Applied decode-time filter: min_dur={min_dur}, max_dur={max_dur}, max_sups={max_sups}"
            )
        except Exception:
            pass

        test_name = test_set["name"]
        testset = MultiSpeakerAsrDataset(
            input_strategy=input_strategy,
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cutset, max_duration=cfg.data.max_duration, shuffle=False
        )

        test_dl = DataLoader(
            testset,
            batch_size=None,
            sampler=sampler,
            num_workers=cfg.data.num_workers,
        )
        test_dls.append(test_dl)
        test_names.append(test_name)
    return test_names, test_dls


def save_xml_outputs(res_dir, test_set_name, results, suffix=""):
    """Save XML outputs to file for comparison."""
    output_file = res_dir / f"xml-outputs-{test_set_name}-{suffix}.txt"

    logging.info(f"Saving XML outputs to {output_file}")

    # Calculate token-level accuracy if available
    total_acc = 0.0
    num_with_acc = 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TagSpeech Decode Results: {test_set_name}\n")
        f.write("=" * 80 + "\n\n")

        for item in results:
            if len(item) == 3:
                cut_id, gt_xml, hyp_xml = item
                acc = None
            else:
                cut_id, gt_xml, hyp_xml, acc = item
                if acc is not None:
                    total_acc += acc
                    num_with_acc += 1

            f.write(f"Cut ID: {cut_id}\n")
            if acc is not None:
                f.write(f"Token Accuracy: {acc:.4f}\n")
            f.write("-" * 80 + "\n")
            f.write("Ground Truth XML:\n")
            f.write(gt_xml + "\n")
            f.write("-" * 40 + "\n")
            f.write("Hypothesis XML:\n")
            f.write(hyp_xml + "\n")
            f.write("=" * 80 + "\n\n")

        # Write summary
        if num_with_acc > 0:
            avg_acc = total_acc / num_with_acc
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total samples: {len(results)}\n")
            f.write(f"Average token accuracy: {avg_acc:.4f}\n")
            f.write("=" * 80 + "\n")

    logging.info(f"Saved {len(results)} outputs to {output_file}")
    if num_with_acc > 0:
        avg_acc = total_acc / num_with_acc
        logging.info(f"Average token accuracy: {avg_acc:.4f}")


def register_custom_models():
    """Register custom model variants for decoding."""
    from auden.auto.auto_config import register_config
    from auden.auto.auto_model import register_model

    # Register TagSpeech model
    register_model(
        model_type="tagspeech",
        module_path="model",
        class_name="TagSpeechModel",
        exist_ok=True,
    )
    register_config(
        config_type="tagspeech",
        module_path="model_config",
        class_name="TagSpeechConfig",
        exist_ok=True,
    )


@hydra.main(version_base=None, config_path="configs", config_name="decode")
@torch.no_grad()
def main(cfg: DictConfig):
    # Register custom models before loading
    register_custom_models()

    logging.info("\n" + OmegaConf.to_yaml(cfg))
    set_audio_duration_mismatch_tolerance(0.1)

    # Initialize dataloader
    test_sets, test_dls = get_test_dataloaders(cfg)

    # Initialize model
    checkpoint_path = None
    ckpt_cfg = cfg.checkpoint
    filename = ckpt_cfg.get("filename", None)
    if filename:  # it should be the model checkpoint
        checkpoint_path = (
            filename if os.path.isabs(filename) else os.path.join(cfg.exp_dir, filename)
        )
    else:  # generate the model checkpoint from trainer checkpoints
        avg = ckpt_cfg.get("avg", 0)
        iters = ckpt_cfg.get("iter", 0)
        epoch = ckpt_cfg.get("epoch", 0)
        if iters > 0:
            model_name = f"averaged-iter-{iters}-avg-{avg}.pt"
        elif epoch > 0:
            model_name = f"averaged-epoch-{epoch}-avg-{avg}.pt"
        else:
            raise ValueError(
                "When averaging, set either checkpoint.iter or checkpoint.epoch"
            )
        checkpoint_path = os.path.join(cfg.exp_dir, model_name)
        if not os.path.exists(checkpoint_path):
            generate_model_checkpoint_from_trainer_checkpoints(
                model_dir=cfg.exp_dir,
                epochs=epoch or None,
                iters=iters or None,
                avg=avg,
                model_name=model_name,
            )

    model = AutoModel.from_pretrained(checkpoint_path)
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)
    model.eval()
    num_param = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of model parameters: {num_param}")

    # Result dir
    if cfg.decoding_method == "greedy_search":
        generate_config = {
            "max_new_tokens": 1000,  # for XML output. Increase if decode very long samples.
            "num_beams": 1,
            "do_sample": False,
            "min_length": 1,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "top_p": None,
            "top_k": None,
            "temperature": None,
        }
    elif cfg.decoding_method == "beam_search":
        generate_config = {
            "max_new_tokens": 1000,  # for XML output. Increase if decode very long samples.
            "num_beams": 4,
            "do_sample": False,
            "min_length": 1,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "top_p": None,
            "top_k": None,
            "temperature": None,
        }
    res_dir = Path(cfg.exp_dir) / cfg.decoding_method
    os.makedirs(res_dir, exist_ok=True)

    # Determine results file suffix
    ckpt_cfg = cfg.checkpoint
    if ckpt_cfg.get("filename"):
        results_file_suffix = Path(checkpoint_path).stem
    elif ckpt_cfg.get("iter", 0) > 0:
        results_file_suffix = (
            f"iter-{ckpt_cfg.get('iter')}-avg-{ckpt_cfg.get('avg', 0)}"
        )
    elif ckpt_cfg.get("epoch", 0) > 0:
        results_file_suffix = (
            f"epoch-{ckpt_cfg.get('epoch')}-avg-{ckpt_cfg.get('avg', 0)}"
        )
    else:
        results_file_suffix = "pretrained"

    # Load prompt
    with open(cfg.prompt_file, "r", encoding="utf-8") as f:
        prompt_list = [line.strip() for line in f if line.strip()]
    prompt = prompt_list[0] if prompt_list else ""

    # Dual audio tokens model expects two audio tokens
    audio_token = model.config.audio_token
    user_content_template = (
        f"<text>{audio_token}</text>\n<speaker>{audio_token}</speaker>"
    )
    if prompt:
        user_content_template += f" {prompt}"

    for test_set_name, test_dl in zip(test_sets, test_dls):
        num_cuts = 0
        try:
            num_batches = len(test_dl)
        except TypeError:
            num_batches = "?"

        # Store results: (cut_id, gt_xml, hyp_xml, acc)
        results = []

        # Go through the dataset
        for batch_idx, batch in enumerate(test_dl):
            feature = batch["inputs"]
            feature = feature.to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            cuts = batch["supervisions"]["cut"]

            # Get ground truth XML
            gt_xmls = [construct_multi_speaker_xml(cut) for cut in cuts]

            acc_value = None
            if hasattr(cfg, "decode") and cfg.decode.get("compute_accuracy", False):
                # Construct messages with GT for accuracy calculation
                messages = [
                    [
                        {
                            "role": "user",
                            "content": user_content_template,
                        },
                        {
                            "role": "assistant",
                            "content": gt_xml,
                        },
                    ]
                    for gt_xml in gt_xmls
                ]

                # Get model outputs with logits for accuracy calculation
                model_outputs, acc = model(
                    x=feature,
                    x_lens=feature_lens,
                    messages=messages,
                )
                # Convert batch accuracy to per-sample (approximate)
                acc_value = acc.item() if isinstance(acc, torch.Tensor) else acc

            # Generate hypotheses (without GT in message)
            messages_gen = [
                [
                    {
                        "role": "user",
                        "content": user_content_template,
                    },
                ]
                for _ in range(len(feature))
            ]
            hyps = model.generate(
                (feature, feature_lens), messages_gen, **generate_config
            )

            cut_ids = [cut.id for cut in cuts]

            # Store results with accuracy
            for cut_id, gt_xml, hyp_xml in zip(cut_ids, gt_xmls, hyps):
                results.append((cut_id, gt_xml, hyp_xml, acc_value))

            num_cuts += len(cuts)
            if batch_idx % 10 == 0:
                batch_str = f"{batch_idx}/{num_batches}"
                logging.info(
                    f"batch {batch_str}, cuts processed until now is {num_cuts}"
                )

        # Save results
        save_xml_outputs(res_dir, test_set_name, results, suffix=results_file_suffix)
        logging.info(f"Finished decoding {test_set_name}: {num_cuts} cuts")


if __name__ == "__main__":
    main()
