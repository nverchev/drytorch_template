"""Typed I/O structures for the model, loss and metrics.

Every drytorch project defines:
  ``Inputs``  — ``NamedTuple`` of tensors fed to the model.
  ``Targets`` — ``NamedTuple`` of tensors consumed by the loss.
  ``Outputs`` — Custom objectproduced by the model's forward pass.
"""

import dataclasses
from typing import NamedTuple, Self

import torch


# ---------------------------------------------------------------------------
# Inputs: tensors consumed by the model's forward() method
# ---------------------------------------------------------------------------


class Inputs(NamedTuple):
    """Flattened feature tensor fed to the classifier.

    Attributes:
        features: Float tensor ``(B, in_features)``.
    """

    features: torch.Tensor


# ---------------------------------------------------------------------------
# Targets: tensors consumed by the loss and evaluation metric
# ---------------------------------------------------------------------------


class Targets(NamedTuple):
    """Ground-truth labels consumed by the loss.

    Attributes:
        label: Long tensor ``(B,)`` with class indices in ``[0, n_classes)``.
    """

    label: torch.Tensor


# ---------------------------------------------------------------------------
# Outputs: tensors produced by the model's forward() method
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Outputs:
    """Model outputs consumed by the loss and metrics.

    Attributes:
        logits: Raw class scores ``(B, n_classes)``.
    """

    logits: torch.Tensor

    def update(self, other: Self) -> None:
        """Update outputs."""
        for attribute in other.__slots__:
            try:
                setattr(self, attribute, getattr(other, attribute))
            except AttributeError:
                pass

        return

    def __repr__(self) -> str:
        parts = []
        for name in self.__slots__:
            if hasattr(self, name):
                try:
                    parts.append(f'{name}={getattr(self, name)!r}')
                except Exception:
                    parts.append(f'{name}=<error>')
            else:
                parts.append(f'{name}=<uninitialized>')

        return f'{self.__class__.__name__}({", ".join(parts)})'
