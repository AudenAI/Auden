"""Batch retrieval evaluation for CLAP models.

Loads a pretrained/exported CLAP checkpoint, builds test DataLoaders from Lhotse
CutSets per configs, and reports a2t/t2a retrieval metrics. When
`multi_caption_eval=true`, captions are flattened per cut and audio features are
repeated to align; we then compute multi-caption aware metrics with global
negatives.
"""

import logging
import os

import hydra
import torch
import yaml
from data_module import AudioCaptionDataset
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import DynamicBucketingSampler, OnTheFlyFeatures
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from auden.auto.auto_model import AutoModel
from auden.models.clap.utils import a2t_metric, multi_a2t, multi_t2a, t2a_metric
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints


def get_test_dataloaders(cfg):
    test_dls = []
    test_names = []
    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    def remove_short_utterance(cut):
        if not cfg.get("multi_caption_eval", False):
            cut.supervisions = [cut.supervisions[0]]
        return cut.duration >= 1.0

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(
            getattr(cfg.data, "sampling_rate", 16000)
        )
        cutset = cutset.filter(remove_short_utterance)
        test_name = test_set["name"]
        testset = AudioCaptionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            return_cuts=True,
            label_field=getattr(cfg.data, "label_field", "caption"),
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
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    test_sets, test_dls = get_test_dataloaders(cfg)

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

    for test_set_name, test_dl in zip(test_sets, test_dls):
        num_cuts = 0
        num_repeats = 1
        audio_embeds_all, text_embeds_all = [], []
        for batch_idx, batch in enumerate(test_dl):
            cut_ids = [cut.id for cut in batch["supervisions"].get("cut", [])]
            num_cuts += len(cut_ids)

            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            caption = batch["supervisions"]["caption"]

            # For multi-caption eval, repeat features to align with flattened captions
            if cfg.get("multi_caption_eval", False):
                num_repeats = len(caption) // feature.size(0)
                feature = feature.repeat_interleave(num_repeats, dim=0)

            audio_embeds = model.encode_audio(x=feature, x_lens=feature_lens)
            text_embeds = model.encode_text(text=caption)
            audio_embeds_all.append(audio_embeds.cpu())
            text_embeds_all.append(text_embeds.cpu())

        audio_embeds_all = torch.cat(audio_embeds_all, dim=0)
        text_embeds_all = torch.cat(text_embeds_all, dim=0)

        if cfg.get("multi_caption_eval", False):
            # Use last batch's repeats (assumes consistent repeats across eval set)
            r1_a, r5_a, r10_a, medr_a, meanr_a, mAP10_a = multi_a2t(
                audio_embeds_all, text_embeds_all, num_repeats
            )
            r1_t, r5_t, r10_t, medr_t, meanr_t, mAP10_t = multi_t2a(
                text_embeds_all, audio_embeds_all, num_repeats
            )
        else:
            r1_a, r5_a, r10_a, medr_a, meanr_a, mAP10_a = a2t_metric(
                audio_embeds_all, text_embeds_all
            )
            r1_t, r5_t, r10_t, medr_t, meanr_t, mAP10_t = t2a_metric(
                text_embeds_all, audio_embeds_all
            )
        logging.info(
            f"a2t with {num_repeats} captions per audio {test_set_name}: r1 {r1_a:.1f}, r5 {r5_a:.1f}, r10 {r10_a:.1f}, medr {medr_a:.1f}, meanr {meanr_a:.1f}, mAP10 {mAP10_a:.1f}"
        )
        logging.info(
            f"t2a with {num_repeats} captions per audio {test_set_name}: r1 {r1_t:.1f}, r5 {r5_t:.1f}, r10 {r10_t:.1f}, medr {medr_t:.1f}, meanr {meanr_t:.1f}, mAP10 {mAP10_t:.1f}"
        )


if __name__ == "__main__":
    main()
