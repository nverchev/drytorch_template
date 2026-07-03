"""Preprocessing utilities run once before dataset construction."""

import torch

from src.config.experiment import Experiment


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize feature columns to zero mean and unit variance."""
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    return (features - mean) / (std + 1e-8)


class Preprocessor:
    """Offline preprocessor for dataset features."""

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.normalize = cfg.data.preprocess.normalize
        return

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            features = normalize_features(features)

        return features


def get_preprocessor() -> Preprocessor:
    """Build and return a Preprocessor from configuration."""
    return Preprocessor()
