"""Tests for train objectives module."""

import pytest
import torch

from src.data import structures
from src.train.objectives import (
    AggregatedAccuracy,
    get_accuracy,
    get_autoencoder_loss,
    get_classification_loss,
)


class TestLossesAndMetrics:
    """Test loss functions and per-sample metric computation."""

    def test_calculate_loss(self) -> None:
        """Produce expected loss entries for classification."""
        loss_fn = get_classification_loss()
        outputs = torch.randn(4, 5)
        targets = structures.Targets(label=torch.tensor([0, 1, 2, 3]), features=torch.zeros(4, 16))

        results = loss_fn.calculate(outputs, targets)

        assert 'CrossEntropy' in results
        assert results['CrossEntropy'].shape == (4,)

    def test_reconstruction_loss(self) -> None:
        """Produce expected mean squared error for reconstruction."""
        loss_fn = get_autoencoder_loss()
        outputs = structures.AutoencodingOutputs(torch.zeros(4, 1, 16), torch.zeros(4, 8))
        targets = structures.Targets(label=torch.tensor([0, 1, 2, 3]), features=torch.zeros(4, 1, 16))

        results = loss_fn.calculate(outputs, targets)

        assert 'MSE' in results
        assert results['MSE'].mean().item() == pytest.approx(0.0)

    def test_get_accuracy_metric(self) -> None:
        """Compute top-1 scalar accuracy for correct predictions."""
        accuracy_metric = get_accuracy()
        outputs = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        targets = structures.Targets(label=torch.tensor([0, 1]), features=torch.zeros(2, 16))

        res = accuracy_metric.calculate(outputs, targets)

        assert res['Accuracy'].mean().item() == pytest.approx(1.0)


class TestAggregatedAccuracy:
    """Test AggregatedAccuracy metric computation."""

    def test_accuracy_calculation(self) -> None:
        """Compute macro accuracy from accumulated batch predictions."""
        metric = AggregatedAccuracy(n_classes=3)
        outputs = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = structures.Targets(label=torch.tensor([0, 1, 2]), features=torch.zeros(3, 16))

        metric.update(outputs, targets)
        res = metric.compute()

        assert res['MacroAccuracy'].item() == pytest.approx(1.0)
