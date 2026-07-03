"""Online data augmentation applied per-sample during training."""

from collections.abc import Callable

import torch

from src.config.experiment import Experiment


def random_noise(sigma: float = 0.01, clip: float = 0.05) -> Callable[[torch.Tensor], torch.Tensor]:
    """Define random noise addition to be applied to features."""

    def _add_noise(features: torch.Tensor) -> torch.Tensor:
        new_features = features.clone()
        noise = torch.randn_like(new_features) * sigma
        new_features += torch.clamp(noise, -clip, clip)
        return new_features

    return _add_noise


class Augmenter:
    """Augmentation class for feature transformations."""

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.noise = cfg.data.augmentation.noise
        self.add_noise = random_noise(sigma=self.noise)

        return

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.noise:
            features = self.add_noise(features)

        return features


def get_augmenter() -> Augmenter:
    """Build and return an Augmenter from configuration."""
    return Augmenter()
