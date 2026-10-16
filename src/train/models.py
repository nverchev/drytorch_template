"""Model and EMAModel wrappers for training phases."""

from typing import override

import torch
from drytorch.lib.models import EMAModel, Model

from src.data.structures import Inputs, Outputs
from src.module.autoencoders import BaseAutoencoder
from src.module.classifiers import BaseClassifier

__all__ = [
    'AutoencoderEMAModel',
    'AutoencoderModel',
    'ClassifierEMAModel',
    'ClassifierModel',
]


class ClassifierModel(Model[Inputs, Outputs]):
    """drytorch Model wrapper for classification architectures.

    Attributes:
        module: classification architecture module.
    """

    module: BaseClassifier


class ClassifierEMAModel(EMAModel[Inputs, Outputs], ClassifierModel):
    """Classifier model wrapper maintaining exponential moving average weights."""

    module: BaseClassifier


class AutoencoderModel(Model[Inputs, Outputs]):
    """drytorch Model wrapper for autoencoder architectures.

    Attributes:
        module: autoencoder architecture module.
    """

    module: BaseAutoencoder

    @torch.inference_mode()
    def encode(self, inputs: Inputs, /) -> torch.Tensor:
        """Compute bottleneck latent encodings under inference mode, in the precision of the model.

        Args:
            inputs: input batch containing signal features.

        Returns:
            latent representation tensor, in full precision.
        """
        with self._autocast():
            return self.module.encode(inputs).float()

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=self.mixed_precision, enabled=self.mixed_precision is not None)


class AutoencoderEMAModel(EMAModel[Inputs, Outputs], AutoencoderModel):
    """Autoencoder model wrapper maintaining exponential moving average weights."""

    module: BaseAutoencoder
    averaged_module: BaseAutoencoder

    @override
    @torch.inference_mode()
    def encode(self, inputs: Inputs, /) -> torch.Tensor:
        """Compute bottleneck latent encodings using averaged module weights.

        Args:
            inputs: input batch containing signal features.

        Returns:
            latent representation tensor.
        """
        with self._autocast():
            return self.averaged_module.encode(inputs).float()
