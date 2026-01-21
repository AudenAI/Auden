"""Custom dataset for multi-speaker TagSpeech training.

This dataset handles multi-speaker cuts correctly by ensuring each cut appears only once
in the batch, while preserving all supervisions for XML generation.
"""

import random
from typing import Any, Dict, List

import torch
from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.dataset.speech_recognition import validate_for_asr
from lhotse.utils import compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data.dataloader import default_collate


class MultiSpeakerAsrDataset(K2SpeechRecognitionDataset):
    """Custom dataset for multi-speaker TagSpeech that handles cuts with multiple supervisions correctly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, cuts):
        """Get a batch item, ensuring each cut appears only once."""
        validate_for_asr(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Calculate num_frames for each cut based on the extracted features
        # For multi-speaker cuts, we need the TOTAL frames of the cut, not per supervision
        # CRITICAL: Use the actual feature length from inputs, not computed from duration
        # This avoids off-by-one errors due to rounding in feature extraction

        # The actual feature length in the batch (all cuts padded to this length)
        actual_feature_len = inputs.size(1)

        # For each cut, compute its unpadded length
        num_frames = []
        for cut in cuts:
            # The input_strategy has already extracted features for this cut
            # We need to compute the number of frames for the entire cut
            if hasattr(cut, "num_frames") and cut.num_frames is not None:
                # If the cut already has num_frames attribute, use it
                cut_frames = cut.num_frames
            else:
                # Compute from cut duration (not supervision duration)
                # Use the same method as the input_strategy
                if hasattr(self.input_strategy, "extractor"):
                    frame_shift = self.input_strategy.extractor.frame_shift
                else:
                    frame_shift = 0.01  # default 10ms

                # Use lhotse's compute_num_frames for exact calculation
                cut_frames = compute_num_frames(
                    duration=cut.duration,  # Use cut.duration, not supervision.duration
                    frame_shift=frame_shift,
                    sampling_rate=cut.sampling_rate,
                )

            # Clamp to actual feature length to avoid assertion errors
            # (in case of minor rounding differences)
            cut_frames = min(cut_frames, actual_feature_len)
            num_frames.append(cut_frames)

        # Apply input transforms (without supervision segments)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs)

        # Create batch with one entry per cut (not per supervision)
        batch = {
            "inputs": inputs,
            "supervisions": {
                "text": [
                    cut.supervisions[0].text for cut in cuts
                ],  # Use first supervision text
                "sequence_idx": torch.arange(len(cuts)),
                "start_frame": torch.zeros(
                    len(cuts), dtype=torch.long
                ),  # Start from 0 for each cut
                "num_frames": torch.tensor(num_frames, dtype=torch.long),
            },
        }

        # Add cuts if requested
        if self.return_cuts:
            batch["supervisions"][
                "cut"
            ] = cuts  # One cut per entry, not per supervision

        return batch
