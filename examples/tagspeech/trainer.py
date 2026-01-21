"""TagSpeech training loop built on BaseTrainer.

This trainer wires the TagSpeech model with the Lhotse-based datamodule and logs
token-level loss and accuracy on validation. See examples/tagspeech/configs for details.
"""

import logging
import random

import torch
from utils.xml_utils import construct_multi_speaker_xml

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class TagSpeechTrainer(BaseTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        prompt_file = cfg.prompt_file
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_list = [line.strip() for line in f if line.strip()]

        # Cache for XML generation to avoid recomputation
        self._xml_cache = {}

        # Log max_length configuration
        max_length = getattr(self.model.config, "max_length", 800)
        logging.info(f"[TagSpeechTrainer] Model max_length: {max_length}")

    def _forward_one_batch(self, batch: dict, is_training: bool, return_emb=False):
        device = self.device
        feature = batch["inputs"]

        # Check feature dimensions: fbank (N, T, C)
        assert feature.ndim == 3, f"Expected fbank (B, T, C), got shape {feature.shape}"

        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        # Get cuts for multi-speaker processing
        cuts = batch["supervisions"]["cut"]
        audio_token = self.cfg.model.audio_token
        batch_size = len(cuts)
        messages = []

        for i, cut in enumerate(cuts):
            # Dual audio tokens model: use two audio tokens
            user_content = (
                f"<text>{audio_token}</text>\n<speaker>{audio_token}</speaker>"
            )

            # Construct multi-speaker XML target from cut supervisions (with caching)
            target_xml = self._construct_multi_speaker_xml(cut)

            message = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_xml},
            ]
            messages.append(message)
        with torch.set_grad_enabled(is_training):
            # Get max_length from config if available
            max_length = getattr(self.model.config, "max_length", 800)
            model_outputs, acc = self.model(
                x=feature,
                x_lens=feature_lens,
                messages=messages,
                max_length=max_length,
            )
            loss = model_outputs.loss

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = sum(len(text) for text in messages)
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", batch_size, normalization="sum")
        info.set_value("loss", loss.detach().cpu().item(), normalization="frame_avg")
        info.set_value("acc", acc, normalization="sample_avg")

        # Explicit cleanup to prevent memory leaks
        del feature, feature_lens, messages, model_outputs
        if not is_training and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss, info

    def _construct_multi_speaker_xml(self, cut):
        """Construct XML format target from cut with multiple supervisions.

        Uses the configured XML format version.

        Args:
            cut: Lhotse Cut object with multiple supervisions

        Returns:
            str: XML formatted multi-speaker transcript
        """
        # Check cache first
        cut_id = cut.id
        if cut_id in self._xml_cache:
            return self._xml_cache[cut_id]

        # Use XML constructor
        result = construct_multi_speaker_xml(cut)

        # Cache the result
        self._xml_cache[cut_id] = result

        return result

    def validate(self, epoch: int):
        """
        Validation is provided by BaseTrainer.

        Override in a subclass if you need TagSpeech specific validation logic
        (e.g., generation-based metrics like WER/CER).
        """
        return super().validate(epoch)
