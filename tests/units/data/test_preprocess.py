"""Tests for feature preprocessing module."""

import torch

from src.config import experiment, hydra
from src.data.preprocess import Preprocessor


class TestPreprocessor:
    """Test Preprocessor normalization and statistics application."""

    def test_fit_and_call_centres_features(self) -> None:
        """Centre and scale features to zero mean and unit variance."""
        tolerance = 1e-4
        features = torch.randn(200, 16) * 5.0 + 3.0
        preprocessor = Preprocessor()

        preprocessor.fit(features)
        scaled = preprocessor(features)
        mean = scaled.mean(dim=0)
        std = scaled.std(dim=0)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=tolerance)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    def test_second_tensor_uses_first_tensor_statistics(self) -> None:
        """Leave second tensor uncentred when using statistics fitted on first tensor."""
        tolerance = 1e-4
        tensor1 = torch.randn(200, 16)
        tensor2 = tensor1 + 10.0
        preprocessor = Preprocessor()

        preprocessor.fit(tensor1)
        scaled2 = preprocessor(tensor2)
        mean2 = scaled2.mean(dim=0)

        assert (mean2.abs() > tolerance).all()

    def test_normalize_false_returns_unchanged(self, mocker) -> None:
        """Return features unmodified when normalize setting is disabled."""
        cfg_no_norm = hydra.get_experiment_config(overrides=['data.preprocess.normalize=false'])
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg_no_norm)
        preprocessor = Preprocessor()
        features = torch.randn(50, 16)

        output = preprocessor(features)

        assert torch.equal(output, features)
