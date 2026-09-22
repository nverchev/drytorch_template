"""Dataset split wrapper with optional runtime augmentation."""

from __future__ import annotations

from collections.abc import Callable
import enum
from typing import Final

import torch
from torch.utils.data import Dataset

from src.data.augmentations import get_augmenter
from src.data.structures import Inputs, Targets

__all__ = [
    'DataSplit',
    'Partitions',
]


class Partitions(enum.Enum):
    """Named dataset splits."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class DataSplit(Dataset[tuple[Inputs, Targets]]):
    """Dataset split wrapper applying runtime augmentations."""

    augment: Final[Callable[[torch.Tensor], torch.Tensor]]
    features: Final[torch.Tensor]
    labels: Final[torch.Tensor]

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()
        self.augment = get_augmenter()
        self.features = features
        self.labels = labels
        return

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        features = self.features[index]
        if not torch.is_inference_mode_enabled():
            features = self.augment(features)

        return Inputs(features=features), Targets(label=self.labels[index])
