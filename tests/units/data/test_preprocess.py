"""Tests for feature preprocessing module."""

import copy

import torch
from pytest_mock import MockerFixture

from src.config.experiment import Experiment
from src.config.specs import AllConfig
from src.data.preprocess import Preprocessor


class TestPreprocessor:
    """Test Preprocessor normalization and statistics application."""

    def test_fit_and_call_centres_features(self) -> None:
        """Test that fit followed by call on the same tensor centres and scales it."""
        tolerance = 1e-4
        features = torch.randn(200, 16) * 5.0 + 3.0
        preprocessor = Preprocessor()
        preprocessor.fit(features)
        scaled = preprocessor(features)
        mean = scaled.mean(dim=0)
        std = scaled.std(dim=0)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=tolerance)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
        return

    def test_second_tensor_uses_first_tensor_statistics(self) -> None:
        """Test that a second tensor transformed with first tensor statistics is not centred."""
        tolerance = 1e-4
        tensor1 = torch.randn(200, 16)
        tensor2 = tensor1 + 10.0
        preprocessor = Preprocessor()
        preprocessor.fit(tensor1)
        scaled2 = preprocessor(tensor2)
        mean2 = scaled2.mean(dim=0)
        assert (mean2.abs() > tolerance).all()
        return

    def test_normalize_false_returns_unchanged(
        self,
        default_config: AllConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that call returns input unchanged when normalize is false."""
        cfg_no_norm = copy.deepcopy(default_config)
        cfg_no_norm.data.preprocess.normalize = False
        mocker.patch.object(Experiment, 'get_config', return_value=cfg_no_norm)
        preprocessor = Preprocessor()
        features = torch.randn(50, 16)
        output = preprocessor(features)
        assert torch.equal(output, features)
        return
