"""
Auden Models Module

This module provides model implementations for audio & multimodal understanding tasks.
It includes both general-purpose core models and task-specific example models.

Structure:
    - base/: Base classes for model configuration and implementation
    - core/: General-purpose models (zipformer, transformer, conformer, etc.)
    - examples/: Task-specific models (ASR, audio caption, speaker verification, speech-LLM, etc.)

Example:
    ```python
    from auden.models import BaseModel, BaseConfig
    from auden.models.zipformer import ZipformerModel, ZipformerConfig

    # Create a model
    config = ZipformerConfig()
    model = ZipformerModel(config)
    ```
"""

from .base.model_config import BaseConfig

__all__ = [
    "BaseConfig",
]
