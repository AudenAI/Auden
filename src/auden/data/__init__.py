"""
Auden Data Module

This module provides data loading and processing functionality for audio & multimodal understanding tasks.
It includes base classes for creating task-specific datamodules with Lhotse integration.

Main Components:
    - BaseLhotseDatamodule: Abstract base class for Lhotse-based audio datamodules
    - _SeedWorkers: Utility class for reproducible data loading

Example:
    ```python
    from auden.data import BaseLhotseDatamodule

    class MyDatamodule(BaseLhotseDatamodule):
        def setup_train(self):
            # Implement training data setup
            pass

        def setup_valid(self):
            # Implement validation data setup
            pass
    ```
"""

from .lhotse_datamodule import BaseLhotseDatamodule, _SeedWorkers

__all__ = [
    "BaseLhotseDatamodule",
    "_SeedWorkers",
]
