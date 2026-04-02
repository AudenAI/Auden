import logging

import torch
from sklearn.metrics import average_precision_score

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class AudioTagTrainer(BaseTrainer):
    def unwrap_model(self):
        return (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )

    def _forward_one_batch(self, batch: dict, is_training: bool, return_logits=False):
        device = self.device
        feature = batch["inputs"].to(device)
        feature_lens = batch["supervisions"]["num_frames"].to(device)
        tags = batch["supervisions"]["tags"]

        batch_size = len(tags)

        with torch.set_grad_enabled(is_training):
            outputs = self.model(feature, feature_lens, tags, return_dict=True)
            loss = outputs["loss"]
            logits = outputs["logits"]
            top1 = outputs["top1_acc"]
            top5 = outputs["top5_acc"]

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", batch_size, normalization="sum")
        info.set_value(
            "loss", loss.detach().cpu().item() / batch_size, normalization="sample_avg"
        )
        info.set_value("top1", float(top1), normalization="sample_avg")
        info.set_value("top5", float(top5), normalization="sample_avg")

        if not return_logits:
            return loss, info
        else:
            return loss, info, logits

    def validate(self, epoch: int):
        self.model.eval()
        with torch.no_grad():
            for valid_name, valid_dl in zip(
                self.data_module.valid_names, self.data_module.valid_dls
            ):
                logits_all = []
                labels_all = []
                tot_info = MetricsTracker()
                for batch_idx, batch in enumerate(valid_dl):
                    loss, info, logits = self._forward_one_batch(
                        batch=batch, is_training=False, return_logits=True
                    )

                    assert loss.requires_grad is False
                    tot_info.update(info)

                    tags = batch["supervisions"]["tags"]
                    labels = self.unwrap_model().tag2multihot(tags)

                    logits_all.append(logits)
                    labels_all.append(labels)

                logits_all = torch.cat(logits_all, dim=0)
                labels_all = torch.cat(labels_all, dim=0)
                is_multilabel = (labels_all.sum(dim=1) > 1).any()

                if is_multilabel:
                    mAP = average_precision_score(
                        y_true=labels_all.numpy(),
                        y_score=logits_all.sigmoid().cpu().detach().numpy(),
                    )

                    tot_info.set_value("mAP", mAP, normalization="sample_avg")

                if self.world_size > 1:
                    tot_info.reduce(loss.device)

                if self.rank == 0:
                    logging.info(
                        f"Epoch {epoch}, global batch {self.global_step}, validation: {tot_info}"
                    )
                    if self.tb_writer is not None:
                        tot_info.write_summary(
                            self.tb_writer,
                            f"train/valid_{valid_name}_",
                            self.global_step,
                        )

        self.model.train()


class AudioCaptionTrainer(BaseTrainer):
    def _forward_one_batch(self, batch: dict, is_training: bool):
        device = self.device
        feature = batch["inputs"]
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        text = supervisions["caption"]
        batch_size = len(text)

        with torch.set_grad_enabled(is_training):
            outputs = self.model(
                x=feature,
                x_lens=feature_lens,
                text=text,
                parallel_decoding_prob=(
                    self.cfg.trainer.parallel_decoding_prob if is_training else 0.0
                ),
                max_length=self.cfg.trainer.max_length,
                return_dict=True,
            )
            loss = outputs["loss"]

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", batch_size, normalization="sum")
        info.set_value(
            "caption_loss",
            loss.detach().cpu().item(),
            normalization="sample_avg",
        )
        return loss, info
