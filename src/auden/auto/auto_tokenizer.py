import importlib
import json
import os
from collections import OrderedDict
from typing import Iterator, Type

try:
    from huggingface_hub import hf_hub_download, snapshot_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("asr-spm", "AsrSpmTokenizer"),
        ("asr-spm-bbpe", "AsrSpmBbpeTokenizer"),
        ("asr-tiktoken", "AsrTiktokenTokenizer"),
    ]
)

# Global registry for dynamic tokenizers
_DYNAMIC_TOKENIZERS = {}


class _LazyTokenizerMapping(OrderedDict):
    def __init__(self, mapping):
        self._mapping = mapping
        self._modules = {}
        self._extra_content = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]

        # Check dynamic registry first
        if key in _DYNAMIC_TOKENIZERS:
            module_path, class_name = _DYNAMIC_TOKENIZERS[key]
            if module_path not in self._modules:
                self._modules[module_path] = importlib.import_module(module_path)
            if hasattr(self._modules[module_path], class_name):
                return getattr(self._modules[module_path], class_name)
            raise ImportError(
                f"Could not find class '{class_name}' in module '{module_path}'."
            )

        # Then check core tokenizers
        if key not in self._mapping:
            raise KeyError(
                f"Tokenizer type '{key}' not found. Available tokenizers: {list(self.keys())}"
            )

        class_name = self._mapping[key]
        module_name = key.replace("-", "_")
        # Our package path is 'auden.tokenizer' (singular)
        module_path = f"auden.tokenizer.{module_name}"
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(module_path)
        return getattr(self._modules[module_name], class_name)

    def keys(self) -> list[str]:
        return (
            list(self._mapping.keys())
            + list(_DYNAMIC_TOKENIZERS.keys())
            + list(self._extra_content.keys())
        )

    def values(self) -> list[Type]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, Type]]:
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, item: object) -> bool:
        return (
            item in self._mapping
            or item in _DYNAMIC_TOKENIZERS
            or item in self._extra_content
        )

    def register(self, key: str, cls: Type, exist_ok: bool = False):
        """
        Register a tokenizer class for lazy loading.

        Args:
            key: Tokenizer type identifier
            cls: Tokenizer class to register
            exist_ok: Whether to allow overwriting existing registrations
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(f"'{key}' is already registered as a core tokenizer.")
        self._extra_content[key] = cls


TOKENIZER_MAPPING = _LazyTokenizerMapping(TOKENIZER_MAPPING_NAMES)


def register_tokenizer(
    tokenizer_type: str, module_path: str, class_name: str, exist_ok: bool = False
):
    """
    Register a new tokenizer type for AutoTokenizer.

    Args:
        tokenizer_type: Unique identifier for the tokenizer
        module_path: Python import path to the module
        class_name: Name of the tokenizer class
        exist_ok: Whether to allow overwriting existing registrations
    """
    if tokenizer_type in TOKENIZER_MAPPING_NAMES and not exist_ok:
        raise ValueError(
            f"Core tokenizer '{tokenizer_type}' cannot be overridden. Use exist_ok=True to force."
        )

    _DYNAMIC_TOKENIZERS[tokenizer_type] = (module_path, class_name)


def list_available_tokenizers() -> list[str]:
    """Return list of all available tokenizer types."""
    return list(TOKENIZER_MAPPING.keys())


class AutoTokenizer:
    """
    Factory class for automatically loading audio tokenizers based on configuration.

    Supports various tokenization methods for audio & multimodal understanding tasks.
    """

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, *args, **kwargs):
        """
        Load tokenizer from local path or HuggingFace Hub.

        Args:
            pretrained_name_or_path: Can be:
                - HuggingFace model name (e.g., 'your-org/auden-tokenizer')
                - Path to directory containing config.json
            *args: Additional arguments passed to tokenizer constructor
            **kwargs: Additional keyword arguments passed to tokenizer constructor

        Returns:
            Instantiated tokenizer ready for use

        Raises:
            FileNotFoundError: If config.json is not found
            ValueError: If tokenizer_type is missing or unsupported
        """
        # Try local path first
        if os.path.exists(pretrained_name_or_path):
            if os.path.isdir(pretrained_name_or_path):
                tokenizer_dir = pretrained_name_or_path
            else:
                tokenizer_dir, _ = os.path.split(pretrained_name_or_path)
        else:
            # Try HuggingFace Hub
            tokenizer_dir = cls._download_from_hub(pretrained_name_or_path)

        config_path = os.path.join(tokenizer_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        tokenizer_type = config.get("tokenizer_type", None)
        if tokenizer_type is None:
            raise ValueError("'tokenizer_type' is required in config.json")

        tokenizer_class = TOKENIZER_MAPPING[tokenizer_type]
        return tokenizer_class(config, tokenizer_dir, *args, **kwargs)

    @staticmethod
    def _download_from_hub(repo_id: str) -> str:
        """
        Download tokenizer repository from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID

        Returns:
            Path to downloaded tokenizer directory

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
            raise FileNotFoundError(f"Could not download tokenizer from {repo_id}: {e}")
