"""ASR decoding script for evaluation/inference.

This script builds test DataLoaders with on-the-fly FBANK features, prepares a
model checkpoint (optionally averaged from trainer checkpoints), runs greedy
search decoding, and saves results for scoring.

Two ways to provide a model checkpoint:
- Provide ``cfg.checkpoint.filename`` pointing to a trainer checkmodel state dict (``.pt``).
- Or provide averaging options (``iter``/``epoch`` + ``avg``), and an averaged
  model will be generated via Auden's checkpoint utilities.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import yaml
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    WhisperFbank,
    WhisperFbankConfig,
    set_audio_duration_mismatch_tolerance,
)
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures,
)
from omegaconf import DictConfig, OmegaConf
from results_utils import save_results
from torch.utils.data import DataLoader

from auden.auto.auto_model import AutoModel
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints
from auden.utils.text_normalization import text_normalization


def get_test_dataloaders(cfg):
    """Construct test DataLoaders from YAML manifest list.

    Expects ``cfg.data.test_data_config`` (YAML) with a list of entries:
    - ``manifest``: path to Lhotse CutSet jsonl.gz
    - ``name``: dataset name used in result filenames

    Returns
    -------
    (test_names, test_dls): Tuple[List[str], List[DataLoader]]
        Names and corresponding DataLoaders for each test set.
    """
    set_audio_duration_mismatch_tolerance(0.1)
    test_dls = []
    test_names = []
    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(16000)
        test_name = test_set["name"]
        if cfg.data.feature == "fbank":
            input_strategy = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        elif cfg.data.feature == "whisper_fbank":
            input_strategy = OnTheFlyFeatures(
                WhisperFbank(WhisperFbankConfig(num_filters=80))
            )
        elif cfg.data.feature == "whisper_v3_fbank":
            input_strategy = OnTheFlyFeatures(
                WhisperFbank(WhisperFbankConfig(num_filters=128))
            )
        else:
            raise ValueError(f"Unsupported feature type: {cfg.data.feature}")
        testset = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            return_cuts=True,
        )
        if test_set.get("encoder_input_strategy", None):
            input_strategy = test_set["encoder_input_strategy"]
            testset = K2SpeechRecognitionDataset(
                input_strategy=input_strategy,
                return_cuts=True,
            )
        sampler = DynamicBucketingSampler(
            cutset,
            max_duration=cfg.data.max_duration,
            shuffle=False,
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


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
@torch.no_grad()
def main(cfg: DictConfig):
    """Hydra entrypoint for ASR decoding/evaluation.

    Parameters
    ----------
    cfg : DictConfig
        Expected keys include:
        - ``exp_dir``: experiment directory containing checkpoints and results.
        - ``checkpoint``: either ``filename`` for a model state dict, or
          averaging parameters ``iter``/``epoch`` with ``avg``.
        - ``data.test_data_config``: YAML file describing test manifests.
        - ``data.max_duration``: duration constraint for dynamic bucketing.
        - ``data.num_workers``: DataLoader worker count.
    """
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # initialize dataloader
    test_sets, test_dls = get_test_dataloaders(cfg)
    # Initialize model
    checkpoint_path = None
    ckpt_cfg = cfg.checkpoint
    filename = ckpt_cfg.get("filename", None)
    if filename:
        checkpoint_path = (
            filename if os.path.isabs(filename) else os.path.join(cfg.exp_dir, filename)
        )
    else:
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

    # result dir
    res_dir = Path(cfg.exp_dir) / "greedy_search"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    results_file_suffix = Path(checkpoint_path).stem

    for test_set_name, test_dl in zip(test_sets, test_dls):
        num_cuts = 0
        try:
            num_batches = len(test_dl)
        except TypeError:
            num_batches = "?"
        # decoding result
        results = defaultdict(list)

        # go through the dataset
        for batch_idx, batch in enumerate(test_dl):
            feature = batch["inputs"]
            feature = feature.to(device)
            # at entry, feature is (N, T, C)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            hyps = model.generate(
                input=(feature, feature_lens), decoding_method="greedy_search"
            )

            hyps = [
                text_normalization(
                    hyp,
                    case="lower",
                    remove_symbols=True,
                    remove_diacritics=True,
                    space_between_cjk=True,
                    simplified_chinese=True,
                    merge_single_char=True,
                    remove_erhua=True,
                    remove_fillers=True,
                ).split()
                for hyp in hyps
            ]

            texts = batch["supervisions"]["text"]
            texts = [
                text_normalization(
                    text,
                    case="lower",
                    remove_symbols=True,
                    remove_diacritics=True,
                    space_between_cjk=True,
                    simplified_chinese=True,
                    remove_erhua=True,
                    remove_fillers=True,
                ).split()
                for text in texts
            ]

            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                this_batch.append((cut_id, ref_text, hyp_words))
            results["greedy_search"].extend(this_batch)

            num_cuts += len(texts)
            if batch_idx % 50 == 0:
                batch_str = f"{batch_idx}/{num_batches}"
                logging.info(
                    f"batch {batch_str}, cuts processed until now is {num_cuts}"
                )

        save_results(res_dir, test_set_name, results, suffix=results_file_suffix)


if __name__ == "__main__":
    main()
