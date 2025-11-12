import importlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

try:
    from huggingface_hub import hf_hub_download, snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from ..models.base.model_config import BaseConfig

CONFIG_MAPPING_NAMES: "OrderedDict[str, tuple[str, str]]" = OrderedDict(
    [
        ("zipformer", ("auden.models.zipformer.model_config", "ZipformerConfig")),
        ("asr", ("auden.models.asr.model_config", "AsrConfig")),
        ("audio-tag", ("auden.models.audio_tag.model_config", "AudioTagConfig")),
        ("clap", ("auden.models.clap.model_config", "ClapConfig")),
        (
            "audio-caption",
            ("auden.models.audio_caption.model_config", "AudioCaptionConfig"),
        ),
        ("audio-llm", ("auden.models.audio_llm.model_config", "AudioLLMConfig")),
        ("tta", ("auden.models.tta.model_config", "TtaConfig")),
    ]
)

# Global registry for dynamic configs (examples)
_DYNAMIC_CONFIGS = {}


class _LazyConfigMapping(OrderedDict[str, type[BaseConfig]]):
    """
    A dictionary that lazily loads configuration classes when accessed.

    This avoids importing all config modules at startup, only loading
    them when a specific config type is requested.
    """

    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str) -> type[BaseConfig]:
        if key in self._extra_content:
            return self._extra_content[key]

        # Check dynamic registry first
        if key in _DYNAMIC_CONFIGS:
            module_path, class_name = _DYNAMIC_CONFIGS[key]
            if module_path not in self._modules:
                self._modules[module_path] = importlib.import_module(module_path)
            if hasattr(self._modules[module_path], class_name):
                return getattr(self._modules[module_path], class_name)
            raise ImportError(
                f"Could not find class '{class_name}' in module '{module_path}'."
            )

        # Then check core configs
        if key not in self._mapping:
            raise KeyError(
                f"Config type '{key}' not found. Available configs: {list(self.keys())}"
            )

        module_path, class_name = self._mapping[key]
        if module_path not in self._modules:
            self._modules[module_path] = importlib.import_module(module_path)
        if hasattr(self._modules[module_path], class_name):
            return getattr(self._modules[module_path], class_name)

        raise ImportError(
            f"Could not find class '{class_name}' in module '{module_path}'."
        )

    def keys(self) -> list[str]:
        return (
            list(self._mapping.keys())
            + list(_DYNAMIC_CONFIGS.keys())
            + list(self._extra_content.keys())
        )

    def values(self) -> list[type[BaseConfig]]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, type[BaseConfig]]]:
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, item: object) -> bool:
        return (
            item in self._mapping
            or item in _DYNAMIC_CONFIGS
            or item in self._extra_content
        )

    def register(
        self, key: str, value: type[BaseConfig], exist_ok: bool = False
    ) -> None:
        """
        Register a configuration class for lazy loading.

        Args:
            key: Config type identifier
            value: Config class to register
            exist_ok: Whether to allow overwriting existing registrations

        Raises:
            ValueError: If key already exists and exist_ok is False
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already registered as a core config.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


def register_config(
    config_type: str, module_path: str, class_name: str, exist_ok: bool = False
):
    """
    Register a new configuration type for AutoConfig.

    Args:
        config_type: Unique identifier for the config
        module_path: Python import path to the module
        class_name: Name of the config class
        exist_ok: Whether to allow overwriting existing registrations

    Raises:
        ValueError: If config_type is a core config and exist_ok is False
    """
    if config_type in CONFIG_MAPPING_NAMES and not exist_ok:
        raise ValueError(
            f"Core config '{config_type}' cannot be overridden. Use exist_ok=True to force."
        )

    _DYNAMIC_CONFIGS[config_type] = (module_path, class_name)


def list_available_configs() -> list[str]:
    """Return list of all available configuration types."""
    return list(CONFIG_MAPPING.keys())


class AutoConfig:
    """
    Factory class for automatically loading audio model configurations.

    Supports loading configurations from JSON files, HuggingFace Hub, or
    creating configs directly for specific model types.
    """

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, **kwargs) -> BaseConfig:
        """
        Load a configuration from various sources.

        Args:
            pretrained_name_or_path: Can be:
                - HuggingFace model repo ID (e.g., 'your-org/zipformer-base')
                - Path to directory containing config.json
                - Path to config.json file
                - Path to .pt checkpoint file (looks for config.json in same dir)
            **kwargs: Additional config parameters to override

        Returns:
            Configuration object ready for model instantiation

        Raises:
            FileNotFoundError: If config.json is not found
            ValueError: If model_type is missing or unsupported
        """
        # Try to resolve as local path first
        try:
            config_file = cls._resolve_config_path(pretrained_name_or_path)
        except (ValueError, FileNotFoundError):
            # If local path fails, try HuggingFace Hub
            config_file = cls._download_from_hub(pretrained_name_or_path, "config.json")

        # Load and parse config
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)

        model_type = config_dict.get("model_type", None)
        if model_type is None:
            raise ValueError("Missing 'model_type' in config.json")
        model_type = model_type.lower()

        if model_type not in CONFIG_MAPPING:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. Supported types: {list(CONFIG_MAPPING.keys())}"
            )

        config_cls = CONFIG_MAPPING[model_type]
        return config_cls(**config_dict)

    @classmethod
    def for_model(cls, model_type: str, **kwargs) -> BaseConfig:
        """
        Instantiate configuration class directly for the given model type.

        Args:
            model_type: Type of model to create config for
            **kwargs: Configuration parameters

        Returns:
            Configuration object

        Raises:
            ValueError: If model_type is unknown
        """
        model_type = model_type.lower()
        if model_type not in CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: {list(CONFIG_MAPPING.keys())}"
            )
        return CONFIG_MAPPING[model_type](**kwargs)

    @staticmethod
    def _resolve_config_path(pretrained_name_or_path: str) -> str:
        """
        Resolve the path to config.json from various input formats.

        Args:
            pretrained_name_or_path: Input path or name

        Returns:
            Path to config.json file

        Raises:
            ValueError: If path is invalid
            FileNotFoundError: If config.json is not found
        """
        if os.path.isfile(pretrained_name_or_path):
            if pretrained_name_or_path.endswith(".json"):
                return pretrained_name_or_path
            elif pretrained_name_or_path.endswith(".pt"):
                # Case: path is like exp_dir/model_xxx.pt -> look for exp_dir/config.json
                config_file = os.path.join(
                    os.path.dirname(pretrained_name_or_path), "config.json"
                )
                if not os.path.exists(config_file):
                    raise FileNotFoundError(
                        f"No config.json found in {os.path.dirname(pretrained_name_or_path)}"
                    )
                return config_file
            else:
                raise ValueError(f"Unsupported file type: {pretrained_name_or_path}")

        elif os.path.isdir(pretrained_name_or_path):
            config_file = os.path.join(pretrained_name_or_path, "config.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"No config.json found in {pretrained_name_or_path}"
                )
            return config_file
        else:
            raise ValueError(
                f"{pretrained_name_or_path} is not a valid file or directory path."
            )

    @staticmethod
    def _download_from_hub(repo_id: str, filename: str) -> str:
        """
        Download file from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID
            filename: File to download

        Returns:
            Path to downloaded file

        Raises:
            ImportError: If huggingface_hub is not installed
            Exception: If download fails
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required to download from HuggingFace Hub. "
                "Install it with: pip install huggingface_hub"
            )

        try:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download {filename} from {repo_id}: {e}"
            )
