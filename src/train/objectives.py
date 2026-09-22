"""Objective specifications for training and evaluation.

This module illustrates two primary patterns for evaluation objectives:

1. **Simple Scalar Metrics** (:func:`get_accuracy`):
   Per-sample scalar metrics wrapped with :class:`drytorch.lib.objectives.Metric`.
   These evaluate batch-by-batch and can be attached directly to training loss via
   ``join(loss, metric)``.

2. **Stateful Cross-Batch Aggregation** (:class:`ConfusionMatrixAccumulator`, :class:`AggregatedAccuracy`):
   Advanced evaluation metrics requiring global state across full evaluation sets
   (e.g., confusion matrices, AUROC, FID scores). Built using:
   - :class:`~drytorch.lib.accumulators.AbstractAccumulator`: holds partial state, handles DDP sync and reduction.
   - :class:`~drytorch.lib.objectives.Objective`: drives calculation, update, compute, and reset cycles.
"""

from __future__ import annotations

from typing import ClassVar, Final, override

import torch
from torch import Tensor

from drytorch.lib.accumulators import AbstractAccumulator
from drytorch.lib.objectives import Loss, LossWithMetrics, Metric, Objective, join

from src.data.structures import Outputs, Targets

__all__ = [
    'AggregatedAccuracy',
    'ConfusionMatrixAccumulator',
    'get_accuracy',
    'get_cross_entropy',
    'get_loss',
]


# ---------------------------------------------------------------------------
# Easy Pattern: Per-sample scalar metrics
# ---------------------------------------------------------------------------


def get_accuracy() -> Metric[Outputs, Targets]:
    """Return top-1 per-sample accuracy metric."""

    def _accuracy(outputs: Outputs, targets: Targets) -> torch.Tensor:
        return (outputs.logits.argmax(dim=-1) == targets.label).float()

    return Metric(_accuracy, name='Accuracy')


def get_cross_entropy() -> LossWithMetrics[Outputs, Targets]:
    """Return cross-entropy loss objective."""
    ce = torch.nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: Outputs, targets: Targets) -> torch.Tensor:
        return ce(outputs.logits, targets.label)

    return Loss(_cross_entropy, name='CrossEntropy')


def get_loss() -> LossWithMetrics[Outputs, Targets]:
    """Return main loss objective watching per-sample accuracy during training."""
    return join(get_cross_entropy(), get_accuracy())


# ---------------------------------------------------------------------------
# Advanced Pattern: Stateful cross-batch aggregation
# ---------------------------------------------------------------------------


class ConfusionMatrixAccumulator(AbstractAccumulator[Tensor, Tensor]):
    """Custom accumulator for confusion matrices.

    Attributes:
        matrix: total counts by label (rows) and prediction (columns).
    """

    matrix: Tensor

    def __init__(self, matrix: Tensor) -> None:
        """Initialize.

        Args:
            matrix: counts of a batch, with labels on the rows.
        """
        self.matrix = matrix
        return

    @classmethod
    @override
    def from_value(cls, value: Tensor) -> ConfusionMatrixAccumulator:
        return cls(value.clone())

    @override
    def merge(self, other: ConfusionMatrixAccumulator) -> None:
        self.matrix += other.matrix
        return

    @override
    def sync(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.matrix)

        return

    @override
    def reduce(self) -> Tensor:
        return self.matrix.float()


class AggregatedAccuracy(Objective[Outputs, Targets]):
    """Stateful macro and per-class accuracy computed over full evaluation sets.

    Attributes:
        n_classes: Total number of target classes.
    """

    _KEY: ClassVar[str] = 'confusion_matrix'
    n_classes: Final[int]
    _matrix: Tensor | None

    def __init__(self, n_classes: int) -> None:
        """Initialize.

        Args:
            n_classes: total number of target classes.
        """
        super().__init__()
        self.n_classes = n_classes
        self._matrix = None
        return

    @property
    def confusion_matrix(self) -> Tensor:
        """Counts of the last computed evaluation, labels on the rows.

        Raises:
            ValueError: if the metrics have not been computed yet.
        """
        if self._matrix is None:
            msg = 'The confusion matrix is available after computing the metrics.'
            raise ValueError(msg)

        return self._matrix

    @classmethod
    @override
    def _get_accumulator_cls(cls) -> type[ConfusionMatrixAccumulator]:
        return ConfusionMatrixAccumulator

    @override
    def calculate(self, outputs: Outputs, targets: Targets) -> dict[str, Tensor]:
        matrix = self._calculate_confusion_matrix(outputs, targets)
        return {self._KEY: matrix}

    def _calculate_confusion_matrix(self, outputs: Outputs, targets: Targets) -> Tensor:
        preds = outputs.logits.argmax(dim=-1)
        labels = targets.label
        indices = labels * self.n_classes + preds
        counts = torch.bincount(indices, minlength=self.n_classes**2)
        return counts.view(self.n_classes, self.n_classes)

    @override
    def _compute(self) -> dict[str, Tensor]:
        if self._KEY not in self._aggregator.accumulators:
            return {}

        cm = self._aggregator.accumulators[self._KEY].reduce()
        self._matrix = cm
        return self._compute_accuracies(cm)

    def _compute_accuracies(self, cm: Tensor) -> dict[str, Tensor]:
        total_by_class = cm.sum(dim=1)
        per_class = cm.diag() / total_by_class.clamp(min=1)
        support = total_by_class > 0  # exclude classes with no samples
        metrics = {'MacroAccuracy': per_class[support].mean()}
        for i in range(self.n_classes):
            metrics[f'Accuracy_Class_{i}'] = per_class[i]

        return metrics
