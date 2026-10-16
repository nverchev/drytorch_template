"""Typed I/O structures for the model, loss and metrics.

Every drytorch project defines:
  ``Inputs``  — ``NamedTuple`` of tensors fed to the model.
  ``Targets`` — ``NamedTuple`` of tensors consumed by the loss.
  ``Outputs`` — Custom objectproduced by the model's forward pass.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple, override

import torch

__all__ = [
    'Inputs',
    'Outputs',
    'Targets',
]


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
        features: Target features for autoencoder reconstruction.
    """

    label: torch.Tensor
    features: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Outputs: tensors produced by the model's forward() method
# ---------------------------------------------------------------------------


@dataclasses.dataclass(init=False, slots=True)
class Outputs:
    """Model outputs consumed by the loss and metrics.

    The fields a model does not produce stay unset, so an objective that reads
    one of them fails where the model and the objective disagree.

    Attributes:
        logits: Raw class scores ``(B, n_classes)``.
        reconstruction: Reconstructed features.
        encodings: Latent representations ``(B, latent_dim)``.
    """

    logits: torch.Tensor
    reconstruction: torch.Tensor
    encodings: torch.Tensor

    def __init__(
        self,
        *,
        logits: torch.Tensor | None = None,
        reconstruction: torch.Tensor | None = None,
        encodings: torch.Tensor | None = None,
    ) -> None:
        """Initialize.

        Args:
            logits: raw class scores ``(B, n_classes)``.
            reconstruction: reconstructed features.
            encodings: latent representations ``(B, latent_dim)``.
        """
        if logits is not None:
            self.logits = logits

        if reconstruction is not None:
            self.reconstruction = reconstruction

        if encodings is not None:
            self.encodings = encodings

        return

    def update(self, other: Outputs, /) -> None:
        """Update outputs with attributes from another Outputs instance.

        Args:
            other: instance to copy attributes from.
        """
        for attribute in other.__slots__:
            try:
                setattr(self, attribute, getattr(other, attribute))
            except AttributeError:
                pass

        return

    @override
    def __repr__(self) -> str:
        parts = []
        for name in self.__slots__:
            try:
                parts.append(f'{name}={getattr(self, name)!r}')
            except AttributeError:
                parts.append(f'{name}=<uninitialized>')

        return f'{self.__class__.__name__}({", ".join(parts)})'
