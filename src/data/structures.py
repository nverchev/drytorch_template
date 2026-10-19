"""Typed I/O structures for the model, loss and metrics.

Every drytorch project defines:
  ``Inputs``  — ``NamedTuple`` of tensors fed to the model.
  ``Targets`` — ``NamedTuple`` of tensors consumed by the loss.
  ``AutoencodingOutputs`` — dataclass of tensors produced by the autoencoder.
"""

import dataclasses
from typing import NamedTuple

import torch

__all__ = [
    'AutoencodingOutputs',
    'Inputs',
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
    features: torch.Tensor


# ---------------------------------------------------------------------------
# Outputs: tensors produced by the autoencoder's forward() method
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class AutoencodingOutputs:
    """Autoencoder outputs consumed by the reconstruction loss.

    Attributes:
        reconstruction: Reconstructed features.
        encodings: Latent representations ``(B, latent_dim)``.
    """

    reconstruction: torch.Tensor
    encodings: torch.Tensor
