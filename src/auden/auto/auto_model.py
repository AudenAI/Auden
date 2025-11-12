# Standard library imports
import importlib
import os
from collections import OrderedDict
from typing import Iterator, Type

# Third-party imports
import torch.nn as nn

try:
    from huggingface_hub import snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Local imports
from .auto_config import AutoConfig

MODEL_MAPPING_NAMES: "OrderedDict[str, tuple[str, str]]" = OrderedDict(
    [
        ("zipformer", ("auden.models.zipformer.model", "ZipformerEncoderModel")),
        ("asr", ("auden.models.asr.model", "AsrModel")),
        ("audio-tag", ("auden.models.audio_tag.model", "AudioTagModel")),
        ("clap", ("auden.models.clap.model", "ClapModel")),
        ("audio-caption", ("auden.models.audio_caption.model", "AudioCaptionModel")),
        ("audio-llm", ("auden.models.audio_llm.model", "AudioLLMModel")),
        ("tta", ("auden.models.tta.model", "TtaModel")),
    ]
)

# Global registry for dynamic models (i.e. from examples)
_DYNAMIC_MODELS = {}


class _LazyModelMapping(OrderedDict[str, Type[nn.Module]]):
    """
    A dictionary that lazily loads model classes when accessed.

    This avoids importing all model modules at startup, only loading
    them when a specific model type is requested.
    """

    def __init__(self, mapping) -> None:
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str) -> Type[nn.Module]:
        if key in self._extra_content:
            return self._extra_content[key]

        # Check dynamic registry first
        if key in _DYNAMIC_MODELS:
            module_path, class_name = _DYNAMIC_MODELS[key]
            if module_path not in self._modules:
                self._modules[module_path] = importlib.import_module(module_path)
            if hasattr(self._modules[module_path], class_name):
                return getattr(self._modules[module_path], class_name)
            raise ImportError(
                f"Could not find class '{class_name}' in module '{module_path}'."
            )

        # Then check core models
        if key not in self._mapping:
            raise KeyError(
                f"Model type '{key}' not found. Available models: {list(self.keys())}"
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
            + list(_DYNAMIC_MODELS.keys())
            + list(self._extra_content.keys())
        )

    def values(self) -> list[Type[nn.Module]]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, Type[nn.Module]]]:
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, item: object) -> bool:
        return (
            item in self._mapping
            or item in _DYNAMIC_MODELS
            or item in self._extra_content
        )

    def register(
        self, key: str, value: Type[nn.Module], exist_ok: bool = False
    ) -> None:
        """
        Register a model class for lazy loading.

        Args:
            key: Model type identifier
            value: Model class to register
            exist_ok: Whether to allow overwriting existing registrations

        Raises:
            ValueError: If key already exists and exist_ok is False
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already registered as a core model.")
        self._extra_content[key] = value


MODEL_MAPPING = _LazyModelMapping(MODEL_MAPPING_NAMES)


def register_model(
    model_type: str, module_path: str, class_name: str, exist_ok: bool = False
):
    """
    Register a new model type for AutoModel.

    Args:
        model_type: Unique identifier for the model
        module_path: Python import path to the module
        class_name: Name of the model class
        exist_ok: Whether to allow overwriting existing registrations
    """
    if model_type in MODEL_MAPPING_NAMES and not exist_ok:
        raise ValueError(
            f"Core model '{model_type}' cannot be overridden. Use exist_ok=True to force."
        )

    _DYNAMIC_MODELS[model_type] = (module_path, class_name)


def list_available_models() -> list[str]:
    """Return list of all available model types."""
    return list(MODEL_MAPPING.keys())


class AutoModel:
    """
    Factory class for automatically loading audio and multimodal models based on configuration.

    Supports various audio & multimodal understanding tasks including ASR, audio captioning,
    speaker verification, speech-LLM, and more.
    """

    @classmethod
    def from_pretrained(cls, model_path: str, *args, **kwargs) -> nn.Module:
        """
        Load model from local path or HuggingFace Hub.

        Args:
            model_path: Can be:
                - HuggingFace model name (e.g., 'your-org/auden-zipformer-base')
                - Path to model directory or checkpoint file
            *args: Additional arguments passed to model constructor
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated model ready for inference

        Raises:
            FileNotFoundError: If config.json is not found
            KeyError: If model_type is not supported
        """
        # Try local path first
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                model_dir = model_path
            else:
                model_dir, _ = os.path.split(model_path)
        else:
            # Try HuggingFace Hub
            model_dir = cls._download_from_hub(model_path)
            model_path = model_dir  # Use downloaded directory

        config_file = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No config.json found at {model_dir}")

        config = AutoConfig.from_pretrained(model_dir)
        model_type = config.model_type
        model_class = MODEL_MAPPING[model_type]

        return model_class.from_pretrained(model_path, *args, **kwargs)

    @staticmethod
    def _download_from_hub(repo_id: str) -> str:
        """
        Download model repository from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            Path to downloaded model directory

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
            return snapshot_download(repo_id=repo_id)
        except Exception as e:
            raise FileNotFoundError(f"Could not download model from {repo_id}: {e}")

    @classmethod
    def from_config(cls, config, *args, **kwargs) -> nn.Module:
        """
        Create model from configuration object.

        Args:
            config: Configuration object with model_type attribute
            *args: Additional arguments passed to model constructor
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated model

        Raises:
            ValueError: If config lacks model_type or model_type is unknown
        """
        model_type = getattr(config, "model_type", None)
        if model_type is None:
            raise ValueError("Config object must have 'model_type' field.")
        if model_type not in MODEL_MAPPING:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_class = MODEL_MAPPING[model_type]
        return model_class(config, *args, **kwargs)
