"""UTS datamodule supporting both audio_tag and caption fields.

Provides train/valid DataLoaders for UTS pre-training using Lhotse CutSets.
Each sample can carry both tags (for classification) and captions (for
captioning/contrastive learning). Either field is optional per sample — a
missing field is emitted as None so downstream trainers can decide which
label(s) to use.

Manifest expectations:
- Each cut should contain exactly one supervision used for labeling. If multiple
  supervisions are present, this datamodule will keep only the first one.
- Tags are read from supervision field `tag_field` (default: "audio_tag").
  Multi-label is supported: list values are joined with ';'.
- Captions are read from supervision field `caption_field` (default: "caption").
  If the field is a list, one caption is randomly chosen per sample.
- Both fields are looked up under `supervision.custom[field]` first, then as
  a direct attribute `supervision.<field>`.

Example MonoCut with both fields:
    {
        "id": "cut-0001",
        ...
        "supervisions": [
            {
                "id": "utt-0001",
                ...
                "custom": {
                    "audio_tag": ["Blues", "Music"],
                    "caption": "A blues guitar riff with bass accompaniment."
                }
            }
        ]
    }
"""

import logging
import random
from re import sub

import torch
import yaml
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data import DataLoader

from auden.data.lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers


def _get_supervision_field(supervision, field_name):
    """Read a field from a Lhotse supervision, checking custom dict first."""
    val = None
    if hasattr(supervision, "custom") and isinstance(supervision.custom, dict):
        val = supervision.custom.get(field_name, None)
    if val is None and hasattr(supervision, field_name):
        val = getattr(supervision, field_name)
    return val


class UTSDataset(torch.utils.data.Dataset):
    """Wraps a CutSet and emits features with both tags and captions.

    For each cut the batch contains:
      - supervisions["tags"]: semicolon-joined tag string, or None if absent
      - supervisions["caption"]: preprocessed caption string, or None if absent
    """

    def __init__(
        self,
        input_strategy,
        cut_transforms=None,
        input_transforms=None,
        return_cuts=False,
        tag_field: str = "audio_tag",
        caption_field: str = "caption",
    ):
        self.input_strategy = input_strategy
        self.cut_transforms = cut_transforms
        self.input_transforms = input_transforms
        self.return_cuts = return_cuts
        self.tag_field = tag_field
        self.caption_field = caption_field
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    @staticmethod
    def _text_preprocess(sentence):
        """Lowercase and strip punctuation, matching CLAP/audio_caption convention."""
        sentence = sentence.lower()
        sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
        sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
        return sentence

    def _get_tag_str(self, cut):
        sup = cut.supervisions[0]
        label_val = _get_supervision_field(sup, self.tag_field)
        if label_val is None:
            return None
        if isinstance(label_val, (list, tuple)):
            return ";".join(map(str, label_val))
        return str(label_val)

    def _get_caption_str(self, cut):
        sup = cut.supervisions[0]
        cap_val = _get_supervision_field(sup, self.caption_field)
        if cap_val is None:
            return None
        if isinstance(cap_val, (list, tuple)):
            chosen = random.choice(cap_val) if len(cap_val) > 0 else ""
            return self._text_preprocess(str(chosen))
        return self._text_preprocess(str(cap_val))

    def __getitem__(self, cuts):
        self.hdf5_fix.update()
        cuts = cuts.sort_by_duration(ascending=False)

        if self.cut_transforms is not None:
            for tnfm in self.cut_transforms:
                cuts = tnfm(cuts)

        cuts = cuts.sort_by_duration(ascending=False)

        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            features, _, cuts = input_tpl
        else:
            features, _ = input_tpl

        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        if self.input_transforms is not None:
            for tnfm in self.input_transforms:
                features = tnfm(features)

        tags = [self._get_tag_str(c) for c in cuts]
        captions = [self._get_caption_str(c) for c in cuts]

        batch = {
            "inputs": features,
            "supervisions": {
                "tags": tags,
                "caption": captions,
            },
        }
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for _ in cut.supervisions
            ]
        return batch

    def __len__(self):
        return 0  # unused with bucketing samplers


class UTSDatamodule(BaseLhotseDatamodule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _filter_cutset(self, cutset, split="train"):
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

    def _make_dataset(self):
        return UTSDataset(
            input_strategy=self.input_strategy,
            cut_transforms=self.transforms,
            input_transforms=self.input_transforms,
            return_cuts=True,
            tag_field=self.cfg.get("tag_field", "audio_tag"),
            caption_field=self.cfg.get("caption_field", "caption"),
        )

    def setup_train(self):
        with open(self.cfg.train_data_config, "r") as f:
            cfg_list = yaml.load(f, Loader=yaml.FullLoader)
        cutset = self._build_mux_cutset(cfg_list)
        cutset = self._filter_cutset(cutset, split="train")
        sampler = self._build_train_sampler(cutset)
        dataset = self._make_dataset()
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
            dataset = UTSDataset(
                input_strategy=self.input_strategy,
                return_cuts=True,
                tag_field=self.cfg.get("tag_field", "audio_tag"),
                caption_field=self.cfg.get("caption_field", "caption"),
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
