"""
Taked from examples/audio_tag/data_module.py

Audio Tagging example datamodule built on BaseLhotseDatamodule.

Provides train/valid DataLoaders for audio tagging tasks using Lhotse CutSets.
Mirrors the ASR datamodule but emits tag strings as supervision labels.

Manifest expectations:
- Each cut should contain exactly one supervision used for labeling. If multiple
  supervisions are present, this datamodule will keep only the first one.
- Tags are read from a configurable supervision field `label_field` (default: "audio_tag").
  The field may appear either under `supervision.custom[label_field]` or as
  a direct attribute `supervision.<label_field>`.
- Multi-label is supported: when the value is a list, it is joined with ';' to a single
  semicolon-separated string. A single string value is used as-is. If a semicolon-joined
  string is already stored (e.g., "Blues;Music"), it will be consumed as-is.
- The resulting tag string(s) must match entries in `id2label.json` used by the model.

Example YAML for data configs (each entry is one dataset):
    - name: "audioset_valid"
      manifest: /abs/path/to/cuts.jsonl
      hours: 100          # for logging/mixing
      weights: 1          # optional, train-side repetition/mix weight

Example MonoCut with embedded recording and supervision (multi-label under custom.audio_tag):
    {
        "id": "cut-0001",
        "start": 0.0,
        "duration": 3.2,
        "recording_id": "rec-0001",
        "recording": {
            "id": "rec-0001",
            "sources": [
                {"type": "file", "channels": [0], "source": "/abs/path/to/audio.wav"}
            ],
            "sampling_rate": 16000,
            "num_samples": 51200,
            "duration": 3.2,
            "channel_ids": [0]
        },
        "supervisions": [
            {
                "id": "utt-0001",
                "recording_id": "rec-0001",
                "start": 0.0,
                "duration": 3.2,
                "channel": 0,
                "custom": {"audio_tag": ["Blues", "Music"]}
            }
        ]
    }
"""

import logging

import torch
import yaml
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers


class AudioTagDataset(torch.utils.data.Dataset):
    """Wraps a CutSet and emits (features, num_frames, tag_string).

    Reads label text from a configurable supervision field `label_field` when available,
    falling back to the standard `text` field. If the field contains a list, it is
    joined with ';' to form a semicolon-separated tag string.
    """

    def __init__(
        self,
        input_strategy,
        cut_transforms=None,
        input_transforms=None,
        return_cuts=False,
        label_field: str = "audio_tag",
    ):
        self.input_strategy = input_strategy
        self.cut_transforms = cut_transforms
        self.input_transforms = input_transforms
        self.return_cuts = return_cuts
        self.label_field = label_field
        # Workaround for HDF5 memory growth when using precomputed features
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts):
        # Periodically close HDF5 handles to avoid memory growth
        self.hdf5_fix.update()
        # Sort by duration so the first determines batch dims
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms (e.g., speed perturb, padding)
        if self.cut_transforms is not None:
            for tnfm in self.cut_transforms:
                cuts = tnfm(cuts)

        # Sort again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Convert cuts to inputs via the input strategy
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            features, _, cuts = input_tpl
        else:
            features, _ = input_tpl

        # Supervision intervals (sequence_idx/start/num frames or samples)
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply input transforms if any
        if self.input_transforms is not None:
            for tnfm in self.input_transforms:
                features = tnfm(features)

        # Expect a single supervision with a tag string strictly from label_field
        def _get_label_str(cut):
            sup = cut.supervisions[0]
            label_val = None
            if hasattr(sup, "custom") and isinstance(sup.custom, dict):
                label_val = sup.custom.get(self.label_field, None)
            if label_val is None and hasattr(sup, self.label_field):
                label_val = getattr(sup, self.label_field)
            if label_val is None:
                raise ValueError(
                    f"Missing label_field '{self.label_field}' in supervision (custom or attribute)."
                )
            if isinstance(label_val, (list, tuple)):
                return ";".join(map(str, label_val))
            return str(label_val)

        tags = [_get_label_str(c) for c in cuts]
        batch = {
            "inputs": features,
            "supervisions": {"tags": tags},
        }
        # Merge supervision intervals into supervisions (includes num_frames/samples)
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for _ in cut.supervisions
            ]
        return batch

    def __len__(self):
        return 0  # unused with bucketing samplers


class AudioTagDatamodule(BaseLhotseDatamodule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _filter_cutset(self, cutset, split="train"):
        # Example placeholder: keep 1s..60s and ensure single supervision
        def keep(c):
            if c.duration < 1.0 or c.duration > 60.0:
                return False
            if len(c.supervisions) > 1:
                c.supervisions = [c.supervisions[0]]
            return True

        return cutset.filter(keep)

    def _build_mux_cutset(self, cfg_list):
        cutset_list = []
        cutset_hours = []
        for spec in cfg_list:
            logging.info(f"Getting {spec['manifest']} cuts")
            cutset = CutSet.from_file(spec["manifest"]).resample(self.sampling_rate)
            hours = spec.get("hours", 0)
            weight = spec.get("weights", 1)
            if self.cfg.use_infinite_dataset:
                cutset = cutset.repeat()
            else:
                cutset = cutset.repeat(weight)
            cutset[0].load_audio()
            cutset_hours.append(weight * hours)
            cutset_list.append(cutset)

        logging.info(
            f"Total {sum(cutset_hours)} hours from {len(cutset_hours)} manifests"
        )

        if len(cutset_list) > 1:
            return CutSet.mux(*cutset_list, weights=cutset_hours, stop_early=True)
        return cutset_list[0]

    def setup_train(self):
        with open(self.cfg.train_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        cutset = self._build_mux_cutset(cfg_list)
        cutset = self._filter_cutset(cutset, split="train")
        sampler = self._build_train_sampler(cutset)
        dataset = AudioTagDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
            label_field=self.cfg.get("label_field", "audio_tag"),
        )
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        self.train_dl = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.cfg.get("num_workers", 8),
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def setup_valid(self):
        with open(self.cfg.valid_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        self.valid_dls = []
        self.valid_names = []
        for spec in cfg_list:
            cutset = CutSet.from_file(spec["manifest"]).resample(self.sampling_rate)
            cutset = self._filter_cutset(cutset, split="valid")
            dataset = AudioTagDataset(
                input_strategy=self.input_strategy,
                return_cuts=True,
                label_field=self.cfg.get("label_field", "audio_tag"),
            )
            sampler = DynamicBucketingSampler(
                cutset, max_duration=self.cfg.sampler.max_duration, shuffle=False
            )
            dl = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=None,
                num_workers=self.cfg.get("num_workers", 8),
                persistent_workers=False,
            )
            self.valid_names.append(spec["name"])
            self.valid_dls.append(dl)
