import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, WhisperFbank, WhisperFbankConfig
from lhotse.dataset import (
    CutMix,
    DynamicBucketingSampler,
    PerturbSpeed,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class BaseLhotseDatamodule(ABC):
    """
    Abstract base datamodule for Lhotse-based audio data processing.

    This class provides a foundation for audio datamodules that use Lhotse
    for audio data loading, preprocessing, and augmentation. It handles:
    - Lhotse CutSet loading and processing
    - Audio feature extraction (Fbank, WhisperFbank, raw audio)
    - Data augmentation (MUSAN, SpecAugment, speed perturbation)
    - Sampler configuration (DynamicBucketingSampler, SimpleCutSampler)

    Task-specific datamodules should inherit from this class and implement
    the abstract methods for their specific data requirements.

    Dependencies:
        - lhotse: For audio data processing
        - torch: For PyTorch integration

    Example:
        ```python
        class AsrDatamodule(BaseLhotseDatamodule):
            def setup_train(self):
                # ASR-specific training setup
                pass

            def setup_valid(self):
                # ASR-specific validation setup
                pass
        ```
    """

    def __init__(self, cfg):
        """
        Initialize the Lhotse datamodule with configuration.

        Args:
            cfg: Configuration object containing all data processing parameters

        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        self._validate_config(cfg)
        self.cfg = cfg
        self.sampling_rate = cfg.get("sampling_rate", 16000)

        # Setup common components
        self._setup_common_components()

        # Setup task-specific data
        self.setup_train()
        self.setup_valid()

    def _validate_config(self, cfg):
        """Validate configuration parameters."""
        required_sections = ["data_augmentation", "sampler"]
        for section in required_sections:
            if section not in cfg:
                raise ValueError(f"Missing required config section: {section}")

        # Validate sampling rate
        if cfg.get("sampling_rate", 16000) <= 0:
            raise ValueError("Sampling rate must be positive")

    def _setup_common_components(self):
        """Setup components that are common across tasks."""
        # Data augmentation
        self.transforms, self.input_transforms = self._build_data_augmentation()

        # Speed perturbation (if enabled)
        if (
            self.cfg.data_augmentation.get("enable_speed_perturb", False)
            and self.cfg.on_the_fly_feats
        ):
            self.transforms = [
                PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)
            ] + self.transforms

        # Feature extraction
        self._setup_feature_extraction()

    def _setup_feature_extraction(self):
        """Setup feature extraction strategy."""
        if self.cfg.on_the_fly_feats:
            feature_type = self.cfg.get("feature", "fbank")
            if feature_type == "fbank":
                self.input_strategy = OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                )
                logging.info("Using default kaldi-fbank")
            elif feature_type == "whisper_fbank":
                self.input_strategy = OnTheFlyFeatures(
                    WhisperFbank(WhisperFbankConfig(num_filters=80))
                )
                logging.info("Using Whisper fbank (80 dims)")
            elif feature_type == "whisper_v3_fbank":
                self.input_strategy = OnTheFlyFeatures(
                    WhisperFbank(WhisperFbankConfig(num_filters=128))
                )
                logging.info("Using Whisper v3 fbank (128 dims)")
            elif feature_type == "wav":
                self.input_strategy = AudioSamples()
                logging.info("Using raw waveform")
        else:
            self.input_strategy = PrecomputedFeatures()

    def _build_data_augmentation(self):
        """
        Build data augmentation transforms.

        This method can be overridden by task-specific datamodules to implement
        custom data augmentation strategies.

        Returns:
            tuple: (transforms, input_transforms) for data augmentation
        """
        transforms = []
        input_transforms = []

        # MUSAN augmentation
        if self.cfg.data_augmentation.get("enable_musan", False):
            logging.info("Enable MUSAN")
            musan_path = self.cfg.data_augmentation.get("musan")
            if not musan_path:
                raise ValueError("MUSAN path not specified in config")

            if not os.path.exists(musan_path):
                raise FileNotFoundError(f"MUSAN file not found: {musan_path}")

            cuts_musan = CutSet.from_file(musan_path)
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        # SpecAugment
        if self.cfg.data_augmentation.get("enable_spec_aug", False):
            logging.info("Enable SpecAugment")
            spec_aug_config = self._get_spec_aug_config()
            input_transforms.append(SpecAugment(**spec_aug_config))
        else:
            logging.info("Disable SpecAugment")

        return transforms, input_transforms

    def _get_spec_aug_config(self):
        """Get SpecAugment configuration with version compatibility."""
        # Version compatibility logic
        num_frame_masks = 10
        num_frame_masks_parameter = inspect.signature(SpecAugment.__init__).parameters[
            "num_frame_masks"
        ]
        if num_frame_masks_parameter.default == 1:
            num_frame_masks = 2

        logging.info(f"Num frame mask: {num_frame_masks}")

        return {
            "time_warp_factor": 80,
            "num_frame_masks": num_frame_masks,
            "features_mask_size": 27,
            "num_feature_masks": 2,
            "frames_mask_size": 100,
        }

    def _build_train_mux_cutset(self, train_data_config):
        """
        Build training cutset with proper error handling.

        Args:
            train_data_config: List of training data configurations

        Returns:
            CutSet: Combined training cutset

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If manifest files don't exist
        """
        if not train_data_config:
            raise ValueError("No training data configuration provided")

        cutset_list = []
        cutset_hours = []

        for train_set in train_data_config:
            manifest_path = train_set.get("manifest")
            if not manifest_path:
                raise ValueError("Manifest path not specified in training config")

            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

            logging.info(f"Loading {manifest_path}")
            try:
                cutset = CutSet.from_file(manifest_path).resample(self.sampling_rate)
            except Exception as e:
                raise RuntimeError(f"Failed to load cutset from {manifest_path}: {e}")

            hours = train_set.get("hours", 0)
            weight = train_set.get("weights", 1.0)

            if self.cfg.use_infinite_dataset:
                cutset = cutset.repeat()
            else:
                cutset = cutset.repeat(weight)

            cutset[0].load_audio()  # Validate audio access
            cutset_hours.append(weight * hours)
            cutset_list.append(cutset)

        if not cutset_list:
            raise ValueError("No valid training cutsets found")

        logging.info(
            f"Total {sum(cutset_hours):.1f} hours of training data from {len(cutset_hours)} manifests"
        )

        if len(cutset_list) > 1:
            logging.info("Muxing cuts")
            cutset_train = CutSet.mux(
                *cutset_list,
                weights=cutset_hours,
                stop_early=True,
            )
        else:
            cutset_train = cutset_list[0]

        return cutset_train

    def _filter_cutset(self, cutset, split="train"):
        """
        Filter cutset - override in subclasses if needed.

        Args:
            cutset: Input cutset
            split: "train" or "valid"

        Returns:
            Filtered cutset
        """
        return cutset

    def _build_train_sampler(self, train_cutset):
        """Build training sampler - override if needed."""
        if self.cfg.sampler.type == "bucketing_sampler":
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                train_cutset,
                max_duration=self.cfg.sampler.max_duration,
                max_cuts=getattr(self.cfg.sampler, 'max_cuts', None),
                shuffle=self.cfg.sampler.shuffle,
                num_buckets=self.cfg.sampler.num_buckets,
                buffer_size=self.cfg.sampler.num_buckets * 2000,
                shuffle_buffer_size=self.cfg.sampler.num_buckets * 5000,
                drop_last=self.cfg.sampler.drop_last,
            )
        elif self.cfg.sampler.type == "simple_sampler":
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                train_cutset,
                max_duration=self.cfg.sampler.max_duration,
                max_cuts=getattr(self.cfg.sampler, "max_cuts", None),
                shuffle=self.cfg.sampler.shuffle,
            )
        else:
            raise ValueError(f"Unsupported Sampler: {self.cfg.sampler.type}")
        return train_sampler

    # Abstract methods - must be implemented by subclasses
    @abstractmethod
    def setup_train(self):
        """
        Setup training data loaders and datasets.

        This method should be implemented by task-specific datamodules to:
        - Load and process training data
        - Create training data loaders
        - Set up training-specific data transformations

        Note:
            Should set self.train_dl attribute with the training DataLoader.
        """
        pass

    @abstractmethod
    def setup_valid(self):
        """
        Setup validation data loaders and datasets.

        This method should be implemented by task-specific datamodules to:
        - Load and process validation data
        - Create validation data loaders
        - Set up validation-specific data transformations

        Note:
            Should set self.valid_dls and self.valid_names attributes.
        """
        pass
