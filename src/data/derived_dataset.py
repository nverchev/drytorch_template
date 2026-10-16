"""Dataset of representations produced by a trained autoencoder."""

from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Final

import torch
from drytorch.utils.apply_ops import apply_to
from torch.utils import data

from src.data.structures import Inputs, Targets

if TYPE_CHECKING:
    from src.train.models import AutoencoderModel

__all__ = ['EncodedDataset']


class EncodedDataset(data.Dataset[tuple[Inputs, Targets]]):
    """Dataset wrapping a split and returning autoencoder latent representations.

    At each access, features from the underlying split are passed through the
    AutoencoderModel under inference mode, preserving any runtime data
    augmentations applied during training access.

    Attributes:
        device: Device on which the autoencoder model resides.
        model: Autoencoder model producing latent representations.
        n_samples: Number of samples in the split.
        split: Underlying dataset split.
    """

    device: Final[torch.device]
    model: Final[AutoencoderModel]
    n_samples: Final[int]
    split: Final[data.Dataset[tuple[Inputs, Targets]]]

    def __init__(
        self,
        split: data.Dataset[tuple[Inputs, Targets]],
        model: AutoencoderModel,
        /,
    ) -> None:
        """Initialize.

        Args:
            split: dataset split to encode.
            model: drytorch model producing representations.

        Raises:
            TypeError: if the split does not have a known length.
        """
        super().__init__()
        if not isinstance(split, Sized):
            msg = 'The encoded dataset needs a split of known length.'
            raise TypeError(msg)

        model.module.eval()
        self.model = model
        self.device = model.device
        self.n_samples = len(split)
        self.split = split
        return

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        """Retrieve an encoded sample at the given index.

        Args:
            index: sample index.

        Returns:
            tuple containing encoded input features and original targets.
        """
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[tuple[Inputs, Targets]]:
        """Retrieve a batch of encoded items.

        Args:
            indices: indices to retrieve.

        Returns:
            list of tuples containing encoded inputs and target labels.
        """
        batch = [self.split[idx] for idx in indices]
        batch_inputs = Inputs(features=torch.stack([inp.features for inp, _ in batch]))
        codes = self._encode(batch_inputs)
        return [(Inputs(features=code), label) for code, (_, label) in zip(codes, batch, strict=True)]

    @torch.inference_mode()
    def _encode(self, inputs: Inputs) -> torch.Tensor:
        x = apply_to(inputs, self.device)
        return self.model.encode(x)
