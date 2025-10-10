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
    test_dls = []
    test_names = []
    if cfg.data.whisper_fbank:
        input_strategy = OnTheFlyFeatures(
            WhisperFbank(WhisperFbankConfig(num_filters=80))
        )
    else:
        input_strategy = OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))

    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(
            getattr(cfg.data, "sampling_rate", 16000)
        )
        if cfg.data.get("pad_to_30s", False):
            cutset = cutset.pad(duration=30)
            logging.info(f"Padded all audio to 30s")
        test_name = test_set["name"]
        testset = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cutset, max_duration=cfg.data.max_duration, shuffle=False  # , max_cuts=1
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
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    set_audio_duration_mismatch_tolerance(0.1)

    # initialize dataloader
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

    # result dir
    if cfg.decoding_method == "greedy_search":
        generate_config = {
            "max_new_tokens": 200,
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
            "max_new_tokens": 200,
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
    if iters > 0:
        results_file_suffix = f"iter-{iters}-avg-{avg}"
    else:
        results_file_suffix = f"epoch-{epoch}-avg-{avg}"

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
            messages = [
                [
                    {
                        "role": "user",
                        "content": f"{model.config.audio_token} 请重复以上内容",
                    },  # {model.config.audio_token}
                ]
                for _ in range(len(feature))
            ]

            hyps = model.generate((feature, feature_lens), messages, **generate_config)
            texts = batch["supervisions"]["text"]

            def unified_text_normalization(s):
                return text_normalization(
                    s,
                    case="lower",
                    remove_diacritics=True,
                    simplified_chinese=True,
                    space_between_cjk=True,
                ).split()

            # only basic text normalization
            hyps = [unified_text_normalization(hyp) for hyp in hyps]
            texts = [unified_text_normalization(text) for text in texts]
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                this_batch.append((cut_id, ref_text, hyp_words))
            results[cfg.decoding_method].extend(this_batch)

            num_cuts += len(texts)
            if batch_idx % 50 == 0:
                batch_str = f"{batch_idx}/{num_batches}"
                logging.info(
                    f"batch {batch_str}, cuts processed until now is {num_cuts}"
                )

        save_results(res_dir, test_set_name, results, suffix=results_file_suffix)


if __name__ == "__main__":
    main()
