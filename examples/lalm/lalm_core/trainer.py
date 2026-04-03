from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class LALMTrainer(BaseTrainer):

    def __init__(self, cfg, model, data_module, rank=0, local_rank=0, world_size=1):
        t = cfg.trainer
        self._grad_accum_steps: int = int(getattr(t, "grad_accum_steps", 1))
        self._max_grad_norm: float = float(getattr(t, "max_grad_norm", 1.0))
        self._ema_decay: float = float(getattr(t, "ema_decay", 0.9999))
        self._accum_step: int = 0
        super().__init__(cfg, model, data_module, rank, local_rank, world_size)

    def setup_model(self, model: nn.Module):
        """float32 EMA on rank-0 (not float64), find_unused_parameters=False."""
        if self.rank == 0:
            model_avg = copy.deepcopy(model).to(torch.float32).to("cpu")
        else:
            model_avg = None

        model = model.to(self.device)

        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.trainer.get(
                    "find_unused_parameters", False
                ),
            )

        num_param = sum(p.numel() for p in model.parameters()) / 1e6
        num_trainable = (
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        )
        logging.info(
            f"Parameters: {num_param:.2f}M total, {num_trainable:.2f}M trainable"
        )
        return model, model_avg

    def _maybe_update_model_average(self):
        """EMA over trainable params only — skips frozen weights to avoid copying
        the entire model (e.g. 7B LLM backbone) from GPU to CPU every N steps."""
        if (
            self.rank != 0
            or not self.cfg.trainer.get("use_averaged_model", False)
            or self.global_step == 0
            or self.global_step % self.cfg.trainer.average_period != 0
        ):
            return

        model_cur = self.model.module if isinstance(self.model, DDP) else self.model
        decay = self._ema_decay
        with torch.no_grad():
            for (_, avg_p), (_, cur_p) in zip(
                self.model_avg.named_parameters(),
                model_cur.named_parameters(),
            ):
                if not cur_p.requires_grad:
                    continue
                avg_p.data.mul_(decay).add_(cur_p.data.float().cpu(), alpha=1.0 - decay)

    def _forward_backward_optimize(self, batch: dict):
        """zero_grad at start of accumulation window, grad accumulation, grad clipping."""
        amp_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else None
        )

        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            loss, batch_metrics = self._forward_one_batch(batch, is_training=True)

        scaled_loss = loss / self._grad_accum_steps
        self.scaler.scale(scaled_loss).backward()

        self._accum_step += 1

        if self._accum_step >= self._grad_accum_steps:
            self._accum_step = 0

            self.scaler.unscale_(self.optimizer)

            if self._max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.grad is not None),
                    self._max_grad_norm,
                )

            self.scheduler.step_batch(self.global_step)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss, batch_metrics

    def _forward_one_batch(self, batch, is_training=True):
        device = self.device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        audio_features = batch["features"].to(device, non_blocking=True)
        feature_lens = batch["feature_lens"].to(device=device, dtype=torch.long, non_blocking=True)

        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        audio_param = next(model_ref.audio_tower.parameters(), None)
        if audio_param is not None and audio_features.dtype != audio_param.dtype:
            audio_features = audio_features.to(dtype=audio_param.dtype)

        with torch.set_grad_enabled(is_training):
            outputs = self.model(
                input_ids=input_ids,
                audio_features=audio_features,
                feature_lens=feature_lens,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits
            packed_labels = getattr(outputs, "packed_labels", labels)

        info = MetricsTracker()
        B = int(input_ids.size(0))
        info.set_value("samples", B, normalization="sum")
        info.set_value(
            "tokens", int((attention_mask > 0).sum().item()), normalization="sum"
        )
        info.set_value(
            "loss", float(loss.detach().cpu().item()), normalization="sample_avg"
        )
        info.set_value(
            "acc",
            float(self._token_accuracy(logits, packed_labels).detach().cpu().item()),
            normalization="sample_avg",
        )
        return loss, info

    @staticmethod
    def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        valid = shift_labels != -100
        n = valid.sum()
        if n == 0:
            return logits.new_zeros(())
        preds = shift_logits.argmax(-1)[valid]
        correct = (preds == shift_labels[valid]).sum()
        return correct.float() / n
