"""
Whisper Encoder Models

This subpackage contains a wrapper around HuggingFace's Whisper encoder
for integration with the auden framework.

Exports:
- WhisperEncoderConfig: Model configuration class
- WhisperEncoderModel: Encoder model class
"""

from .model import WhisperEncoderModel
from .model_config import WhisperEncoderConfig

__all__ = [
    "WhisperEncoderConfig",
    "WhisperEncoderModel",
]
