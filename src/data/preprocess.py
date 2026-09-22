"""Preprocessing utilities run once before dataset construction."""

from typing import Final

import torch

from src.config.experiment import Experiment

__all__ = [
    'Preprocessor',
    'get_preprocessor',
]


class Preprocessor:
    """Offline preprocessor scaling features with the training statistics.

    The statistics come from the training data only, and the other splits
    reuse them, so that evaluation sees the same scaling as training.

    Attributes:
        mean: Mean of each feature of the training data.
        normalize: Whether the features are scaled.
        std: Standard deviation of each feature of the training data.
    """

    normalize: Final[bool]
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.normalize = cfg.data.preprocess.normalize
        self.mean = torch.zeros(1)
        self.std = torch.ones(1)
        return

    def __call__(self, features: torch.Tensor, /) -> torch.Tensor:
        """Scale the features with the statistics of the training data.

        Args:
            features: features of one split.

        Returns:
            The scaled features, or the original ones when not normalizing.
        """
        if not self.normalize:
            return features

        return (features - self.mean) / (self.std + 1e-8)

    def fit(self, features: torch.Tensor, /) -> None:
        """Take the statistics from the training features.

        Args:
            features: features of the training data.
        """
        self.mean = features.mean(dim=0, keepdim=True)
        self.std = features.std(dim=0, keepdim=True)
        return


def get_preprocessor() -> Preprocessor:
    """Build and return a Preprocessor from configuration."""
    return Preprocessor()
