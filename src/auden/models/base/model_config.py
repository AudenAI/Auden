"""
Configuration utilities for auden models.

This module provides a lightweight, inheritance-friendly ``BaseConfig`` class
that model-specific configs can extend. It is intentionally similar in spirit
to Hugging Face's ``PretrainedConfig`` to provide a predictable developer
experience while keeping implementation minimal.

Key features:
- Convert to a Python dict (including nested configs implementing ``to_dict()``)
- Save to and load from JSON files (``config.json``)
- Optional backup of existing config files when overwriting
- Convenience helpers: ``to_json_string()``, ``to_json_file()``, ``from_dict()``,
  ``update()``, and simple dict-like accessors (``keys()``, ``get()``,
  ``__contains__``)

Notes:
- The field ``model_type`` is REQUIRED for subclasses. It must be a non-empty
  string so that ``AutoConfig`` / ``AutoModel`` can resolve the correct classes.
- By default, arbitrary keyword fields are accepted and stored; concrete
  config classes can override this behavior if strictness is desired.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path


class BaseConfig:
    model_type: str = None

    def __init__(self, **kwargs):
        """Initialize a config instance.

        Behavior mirrors common ML config bases:
        - Class attributes become default values.
        - Any provided keyword arguments override defaults and are stored as
          attributes, even if they were not declared ahead of time. This keeps
          configs forward-compatible with newer versions.
        """
        for key, value in self.__class__.__dict__.items():
            # Skip private attributes, callables, and method descriptors
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            # Skip classmethod, staticmethod, and other descriptors
            if isinstance(value, (classmethod, staticmethod, property)):
                continue
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.__class__ is not BaseConfig:
            model_type_value = getattr(self, "model_type", None)
            if not isinstance(model_type_value, str) or model_type_value.strip() == "":
                raise ValueError(
                    "'model_type' must be set to a non-empty string in subclasses of BaseConfig."
                )

    def to_dict(self):
        """Return a JSON-serializable dict representation of this config.

        Nested config objects are supported if they implement a ``to_dict()``
        method. Private attributes (starting with ``_``) and callables are
        skipped.
        """

        # Prefer HF-style minimal dict for Hugging Face configs, otherwise keep full
        try:
            from transformers import (
                PretrainedConfig as _HFPretrainedConfig,  # type: ignore
            )
        except Exception:  # transformers is optional
            _HFPretrainedConfig = None  # type: ignore

        output = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if _HFPretrainedConfig is not None and isinstance(
                value, _HFPretrainedConfig
            ):
                # Delegate to HF's compact serialization for nested HF configs
                nested = value.to_diff_dict()
            elif hasattr(value, "to_dict"):
                nested = value.to_dict()
            else:
                nested = value
            output[key] = nested
        return output

    def to_diff_dict(self):
        """Return only fields that differ from class defaults (HF-like)."""

        # Prefer HF-style minimal dict for Hugging Face configs when diffing
        try:
            from transformers import (
                PretrainedConfig as _HFPretrainedConfig,  # type: ignore
            )
        except Exception:
            _HFPretrainedConfig = None  # type: ignore

        def _public_dict(obj):
            if _HFPretrainedConfig is not None and isinstance(obj, _HFPretrainedConfig):
                return obj.to_diff_dict()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if isinstance(obj, dict):
                return obj
            return obj

        def _diff(current: dict, default: dict) -> dict:
            diff = {}
            for key, cur_val in current.items():
                if key.startswith("_"):
                    continue
                def_val = default.get(key, None)
                if hasattr(cur_val, "to_dict") or isinstance(cur_val, dict):
                    cur_map = _public_dict(cur_val)
                    def_map = (
                        def_val if isinstance(def_val, dict) else _public_dict(def_val)
                    )
                    if isinstance(cur_map, dict) and isinstance(def_map, dict):
                        sub = _diff(cur_map, def_map)
                        if sub:
                            diff[key] = sub
                    else:
                        if cur_map != def_val:
                            diff[key] = cur_val
                else:
                    if cur_val != def_val:
                        diff[key] = cur_val
            return diff

        current = self.to_dict()
        try:
            defaults = self.__class__().to_dict()
        except Exception:
            return current
        return _diff(current, defaults)

    def save_pretrained(
        self,
        output_dir: str,
        *,
        subfolder: str | None = None,
        filename: str | None = None,
        use_diff: bool = False,
    ):
        """Save this config to config.json (HF-like)."""
        target_dir = Path(output_dir)
        if subfolder:
            target_dir = target_dir / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)

        data = self.to_diff_dict() if use_diff else self.to_dict()
        config_filename = filename or "config.json"
        config_path = target_dir / config_filename

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    existing_config = json.load(f)
                if existing_config == data:
                    logging.info(
                        f"[save_config] Skipped saving. Config identical to existing one."
                    )
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = config_path.with_name(f"config.{timestamp}.bak.json")
                    shutil.move(config_path, backup_path)
                    logging.info(
                        f"[save_config] Existing config backed up to: {backup_path}"
                    )
            except Exception as e:
                logging.warning(
                    f"[save_config] Could not compare with existing config: {e}. Proceeding to save."
                )

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"[save_config] Saved config to: {config_path}")

    @classmethod
    def from_pretrained(cls, config_path: str):
        """Load a config from a directory or a JSON file.

        Args:
            config_path: A directory containing ``config.json`` or a path to a JSON file.

        Returns:
            An instance of ``cls`` initialized with values loaded from JSON.
        """
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, "config.json")
        else:
            config_file = config_path
        with open(config_file, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_string(self, indent: int = 2) -> str:
        """Return a JSON string representation of this config.

        Args:
            indent: Indentation level for pretty printing.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_json_file(self, json_file: str, indent: int = 2) -> None:
        """Write this config to a JSON file.

        Args:
            json_file: File path to write to.
            indent: Indentation level for pretty printing.
        """
        with open(json_file, "w") as f:
            f.write(self.to_json_string(indent=indent))

    @classmethod
    def from_dict(cls, data: dict):
        """Instantiate config from a Python dict."""
        return cls(**data)

    def update(self, values: dict):
        """Update config values in-place from a dict and return self."""
        for key, value in values.items():
            setattr(self, key, value)
        return self

    # Dict-like conveniences
    def keys(self):
        """Return public attribute names treated as config keys."""
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def get(self, key, default=None):
        """Get a config value with a default, similar to dict.get."""
        return getattr(self, key, default)

    def __contains__(self, item: object) -> bool:
        return (
            hasattr(self, item) and not item.startswith("_")
            if isinstance(item, str)
            else False
        )

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}({self.to_json_string(indent=2)})"
