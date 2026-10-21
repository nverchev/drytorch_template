"""Model and EMAModel wrappers for training phases."""

from typing import Any, override

import torch
from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib.models import EMAModel, Model

from src.config import Experiment
from src.data.structures import AutoencodingOutputs, Inputs
from src.module.autoencoders import BaseAutoencoder
from src.module.classifiers import BaseClassifier

__all__ = [
    'AutoencoderEMAModel',
    'AutoencoderModel',
    'ClassifierEMAModel',
    'ClassifierModel',
    'load_trained_state',
]


class ClassifierModel(Model[Inputs, torch.Tensor]):
    """drytorch Model wrapper for classification architectures.

    Attributes:
        module: classification architecture module.
    """

    module: BaseClassifier


class ClassifierEMAModel(EMAModel[Inputs, torch.Tensor], ClassifierModel):
    """Classifier model wrapper maintaining exponential moving average weights."""

    module: BaseClassifier


class AutoencoderModel(Model[Inputs, AutoencodingOutputs]):
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


class AutoencoderEMAModel(EMAModel[Inputs, AutoencodingOutputs], AutoencoderModel):
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


def load_trained_state(model: Model[Any, Any], /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.par_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return
