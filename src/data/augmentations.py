"""Online data augmentation applied per-sample during training."""

from collections.abc import Callable
from typing import Final

import torch

from src.config.experiment import Experiment

__all__ = [
    'Augmenter',
    'get_augmenter',
    'random_noise',
]


def random_noise(*, clip: float = 0.05, sigma: float = 0.01) -> Callable[[torch.Tensor], torch.Tensor]:
    """Define random noise addition to be applied to features.

    Args:
        clip: maximum absolute value for noise clipping.
        sigma: standard deviation of Gaussian noise.

    Returns:
        closure that applies clipped Gaussian noise to a tensor.
    """

    def _add_noise(features: torch.Tensor) -> torch.Tensor:
        new_features = features.clone()
        noise = torch.randn_like(new_features) * sigma
        new_features += torch.clamp(noise, -clip, clip)
        return new_features

    return _add_noise


class Augmenter:
    """Apply feature augmentations to training samples.

    Attributes:
        add_noise: Callable adding Gaussian noise to features.
        noise: Noise standard deviation.
    """

    add_noise: Final[Callable[[torch.Tensor], torch.Tensor]]
    noise: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        cfg = Experiment.get_config()
        self.noise = cfg.data.augmentation.noise
        self.add_noise = random_noise(sigma=self.noise)
        return

    def __call__(self, features: torch.Tensor, /) -> torch.Tensor:
        """Apply noise augmentation to features when noise is enabled.

        Args:
            features: input feature tensor.

        Returns:
            augmented or unchanged feature tensor.
        """
        if self.noise:
            features = self.add_noise(features)

        return features


def get_augmenter() -> Augmenter:
    """Build and return an Augmenter from configuration.

    Returns:
        configured Augmenter instance.
    """
    return Augmenter()
