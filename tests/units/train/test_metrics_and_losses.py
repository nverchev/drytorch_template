"""Tests for train metrics and loss modules."""

import pytest
import torch

from src.data.structures import Outputs, Targets
from src.train.metrics_and_losses import AggregatedAccuracy, get_accuracy, get_loss


class TestLossesAndMetrics:
    """Test loss functions and per-sample metric computation."""

    def test_calculate_loss(self) -> None:
        """Test loss calculation returns expected loss dict."""
        loss_fn = get_loss()
        outputs = Outputs(logits=torch.randn(4, 5))
        targets = Targets(label=torch.tensor([0, 1, 2, 3]))
        results = loss_fn.calculate(outputs, targets)
        assert 'CrossEntropy' in results
        assert results['CrossEntropy'].shape == (4,)
        return

    def test_get_accuracy_metric(self) -> None:
        """Test get_accuracy scalar metric computation."""
        accuracy_metric = get_accuracy()
        outputs = Outputs(logits=torch.tensor([[10.0, 0.0], [0.0, 10.0]]))
        targets = Targets(label=torch.tensor([0, 1]))
        res = accuracy_metric.calculate(outputs, targets)
        assert res['Accuracy'].mean().item() == pytest.approx(1.0)
        return


class TestAggregatedAccuracy:
    """Test AggregatedAccuracy metric computation."""

    def test_accuracy_calculation(self) -> None:
        """Test accuracy metric evaluation."""
        metric = AggregatedAccuracy(n_classes=3)
        outputs = Outputs(logits=torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]))
        targets = Targets(label=torch.tensor([0, 1, 2]))
        metric.update(outputs, targets)
        res = metric.compute()
        assert res['MacroAccuracy'].item() == pytest.approx(1.0)
        return
