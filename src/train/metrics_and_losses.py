"""Evaluation metrics and loss specifications.

This module illustrates two primary patterns for evaluation objectives:

1. **Simple Scalar Metrics** (:func:`get_accuracy`):
   Per-sample scalar metrics wrapped with :class:`drytorch.lib.objectives.Metric`.
   These evaluate batch-by-batch and can be attached directly to training loss via
   ``loss.watch(metric)``.

2. **Stateful Cross-Batch Aggregation** (:class:`ConfusionMatrixAccumulator`, :class:`AggregatedAccuracy`):
   Advanced evaluation metrics requiring global state across full evaluation sets
   (e.g., confusion matrices, AUROC, FID scores). Built using:
   - :class:`~drytorch.lib.aggregators.AbstractAccumulator`: holds partial state, handles DDP sync and reduction.
   - :class:`~drytorch.lib.aggregators.AbstractAggregator`: maps keys to accumulator instances.
   - :class:`~drytorch.core.protocols.ObjectiveProtocol`: drives update, compute, and reset cycles.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor

from drytorch.core import protocols as p
from drytorch.lib.aggregators import AbstractAccumulator, AbstractAggregator
from drytorch.lib.objectives import Loss, LossBase, Metric

from src.data.structures import Outputs, Targets


# ---------------------------------------------------------------------------
# Easy Pattern: Per-sample scalar metrics
# ---------------------------------------------------------------------------


def get_accuracy() -> Metric[Outputs, Targets]:
    """Return top-1 per-sample accuracy metric."""

    def _accuracy(outputs: Outputs, targets: Targets) -> torch.Tensor:
        return (outputs.logits.argmax(dim=-1) == targets.label).float()

    return Metric(_accuracy, name='Accuracy')


def get_cross_entropy() -> LossBase[Outputs, Targets]:
    """Return cross-entropy loss objective."""
    ce = torch.nn.CrossEntropyLoss(reduction='none')

    def _cross_entropy(outputs: Outputs, targets: Targets) -> torch.Tensor:
        return ce(outputs.logits, targets.label)

    return Loss(_cross_entropy, name='CrossEntropy')


def get_loss() -> LossBase[Outputs, Targets]:
    """Return main loss objective watching per-sample accuracy during training."""
    loss = get_cross_entropy()
    loss.watch(get_accuracy())
    return loss


# ---------------------------------------------------------------------------
# Advanced Pattern: Stateful cross-batch aggregation
# ---------------------------------------------------------------------------


class ConfusionMatrixAccumulator(AbstractAccumulator[tuple[Tensor, Tensor], Tensor]):
    """Stateful accumulator for confusion matrix count matrices.

    Attributes:
        n_classes: Total number of classes.
        matrix: Running ``(n_classes, n_classes)`` count matrix.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.matrix = torch.zeros(n_classes, n_classes, dtype=torch.long)
        return

    @classmethod
    @override
    def from_value(cls, value: tuple[Tensor, Tensor]) -> ConfusionMatrixAccumulator:
        preds, labels = value
        n_classes = int(preds.max().item()) + 1 if preds.numel() > 0 else 1
        inst = cls(n_classes)
        for p_i, l_i in zip(preds.tolist(), labels.tolist(), strict=True):
            inst.matrix[int(l_i), int(p_i)] += 1

        return inst

    @override
    def merge(self, other: AbstractAccumulator) -> None:
        assert isinstance(other, ConfusionMatrixAccumulator)
        if other.matrix.shape[0] > self.matrix.shape[0]:
            new_matrix = torch.zeros_like(other.matrix)
            new_matrix[: self.matrix.shape[0], : self.matrix.shape[1]] = self.matrix
            self.matrix = new_matrix

        self.matrix[: other.matrix.shape[0], : other.matrix.shape[1]] += other.matrix
        return

    @override
    def sync(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.matrix)

        return

    @override
    def reduce(self) -> Tensor:
        return self.matrix.float()


class ConfusionMatrixAggregator(AbstractAggregator[tuple[Tensor, Tensor], Tensor]):
    """Aggregator wrapping :class:`ConfusionMatrixAccumulator`."""

    accumulator_cls = ConfusionMatrixAccumulator


class AggregatedAccuracy(p.ObjectiveProtocol[Outputs, Targets]):
    """Stateful macro and per-class accuracy computed over full evaluation sets.

    Attributes:
        n_classes: Total number of target classes.
    """

    __slots__ = ('_aggregator', 'n_classes')
    name: str = 'AggregatedAccuracy'

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self._aggregator: ConfusionMatrixAggregator = ConfusionMatrixAggregator()
        return

    @override
    def update(self, outputs: Outputs, targets: Targets) -> dict[str, Tensor]:
        preds = outputs.logits.argmax(dim=-1)
        self._aggregator += {'cm': (preds, targets.label)}
        return self.compute()

    @override
    def compute(self) -> dict[str, Tensor]:
        reduced = self._aggregator.reduce()
        if not reduced or 'cm' not in reduced:
            return {}

        cm = reduced['cm']
        if cm.sum() == 0:
            return {}

        per_class = cm.diag() / cm.sum(dim=1).clamp(min=1)
        metrics = {'MacroAccuracy': per_class.mean()}
        for i in range(self.n_classes):
            metrics[f'Accuracy_Class_{i}'] = per_class[i]

        return metrics

    @override
    def reset(self) -> None:
        self._aggregator.clear()
        return
