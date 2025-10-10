import logging
import os

import hydra
import torch
import yaml
from data_module import AudioTagDataset
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import DynamicBucketingSampler, OnTheFlyFeatures
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from auden.auto.auto_model import AutoModel
from auden.models.audio_tag.model import compute_acc
from auden.utils.checkpoint import generate_model_checkpoint_from_trainer_checkpoints


def get_test_dataloaders(cfg):
    test_dls = []
    test_names = []
    with open(cfg.data.test_data_config, "r") as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    def remove_short_utterance(cut):
        return cut.duration >= 1.0

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['manifest']} cuts")
        cutset = CutSet.from_file(test_set["manifest"]).resample(
            getattr(cfg.data, "sampling_rate", 16000)
        )
        cutset = cutset.filter(remove_short_utterance)
        test_name = test_set["name"]
        testset = AudioTagDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            return_cuts=True,
            label_field=getattr(cfg.data, "label_field", "audio_tag"),
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

    # Get dataloaders
    test_sets, test_dls = get_test_dataloaders(cfg)

    # Initialize model (same logic as ASR decode.py)
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

    # test metric mapping
    test_metric = {
        "map": ["audioset", "fsd50k"],
        "acc": ["esc50", "urbansound", "vggsound", "vocalsound"],
    }

    # do evaluation
    for test_set_name, test_dl in zip(test_sets, test_dls):
        test_base_name = test_set_name.split("-")[0]
        num_cuts = 0
        try:
            num_batches = len(test_dl)
        except TypeError:
            num_batches = "?"

        all_logits = []
        all_labels = []
        result_str = ""

        for batch_idx, batch in enumerate(test_dl):
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
            num_cuts += len(cut_ids)

            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            tags = batch["supervisions"]["tags"]

            audio_label = model.tag2multihot(tags)
            audio_logits = model.generate(
                (feature, feature_lens), return_full_logits=True
            )

            all_logits.append(audio_logits)
            all_labels.append(audio_label)

            if batch_idx % 20 == 1:
                logging.info(f"Processed {num_cuts} cuts already.")
        logging.info("Finish collecting audio logits")

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if test_base_name in test_metric["map"]:
            try:
                from sklearn.metrics import average_precision_score

                mAP = average_precision_score(
                    y_true=all_labels.numpy(),
                    y_score=all_logits.cpu().numpy(),
                )
                result_str += f"{test_set_name}: mAP: {mAP:.4f}\n"
            except Exception as e:
                logging.warning(f"sklearn not available or mAP failed: {e}")

        top1_acc, top5_acc = compute_acc(all_logits.cpu(), all_labels)
        result_str += (
            f"{test_set_name}: Top1 Acc: {top1_acc:.2f}, Top5 Acc: {top5_acc:.2f}\n"
        )
        logging.info(result_str)
        result_file = os.path.join(cfg.exp_dir, f"{test_set_name}_result.txt")
        logging.info(f"save result to {result_file}")
        with open(result_file, "w") as f:
            f.write(result_str)
        logging.info("Done")


if __name__ == "__main__":
    main()
