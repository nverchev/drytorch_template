"""Classification architectures built from layer primitives."""

import abc
import itertools

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import Classifiers
from src.data.dataset import get_databuilder
from src.data.structures import Inputs
from src.module.layers import LinearLayer

__all__ = [
    'BaseClassifier',
    'LinearClassifier',
    'MLPClassifier',
    'get_classifier',
]


class BaseClassifier(nn.Module, abc.ABC):
    """Abstract base for all classification model architectures.

    Attributes:
        in_dim: Number of input features per sample.
        n_classes: Number of target classes to predict.
    """

    in_dim: int
    n_classes: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        exp_cfg = Experiment.get_config()
        self.in_dim = exp_cfg.data.dataset.in_features
        self.n_classes = get_databuilder().n_classes
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Run the forward pass.

        Args:
            inputs: input batch containing features.

        Returns:
            model outputs ``(B, n_classes)``.
        """


class MLPClassifier(BaseClassifier):
    """Feedforward MLP: in_features → hidden_dims → n_classes.

    Attributes:
        layers: Sequence of linear layers and optional dropout.
    """

    layers: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        arch = cfg.classification.model.architecture
        dims = (self.in_dim, *arch.hidden_dims)
        self.layers = nn.Sequential()
        for in_dim, out_dim in itertools.pairwise(dims):
            self.layers.append(LinearLayer(in_dim, out_dim, act_cls=arch.act_cls, norm_cls=arch.norm_cls))
            if arch.dropout_rate > 0:
                self.layers.append(nn.Dropout(arch.dropout_rate))

        self.layers.append(LinearLayer(dims[-1], self.n_classes))
        return

    @override
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Forward pass through the hidden layers and the output layer.

        Args:
            inputs: input batch containing features.

        Returns:
            model outputs containing logits ``(B, n_classes)``.
        """
        return self.layers(inputs.features)


class LinearClassifier(BaseClassifier):
    """Linear classifier on learned latent encodings.

    Attributes:
        norm: layer normalization before linear projection.
        head: linear projection layer from encodings to class logits.
    """

    norm: Final[nn.LayerNorm]
    head: Final[LinearLayer]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        exp_cfg = Experiment.get_config()
        assert exp_cfg.autoencoding is not None

        self.in_dim = exp_cfg.autoencoding.model.architecture.latent_dim
        self.norm = nn.LayerNorm(self.in_dim)
        self.head = LinearLayer(self.in_dim, self.n_classes)
        return

    @override
    def forward(self, inputs: Inputs, /) -> torch.Tensor:
        """Forward pass through normalization and linear head.

        Args:
            inputs: input batch containing features.

        Returns:
            model outputs containing class logits ``(B, n_classes)``.
        """
        return self.head(self.norm(inputs.features))


def get_classifier() -> BaseClassifier:
    """Instantiate a classification architecture from configuration.

    Returns:
        instantiated classifier architecture.

    Raises:
        ValueError: if the configured name is not registered.
    """
    exp_cfg = Experiment.get_config()
    cfg = exp_cfg.classification.model
    model_map: dict[Classifiers, type[BaseClassifier]] = {
        Classifiers.MLP_CLASSIFIER: MLPClassifier,
        Classifiers.LINEAR_CLASSIFIER: LinearClassifier,
    }

    cls = model_map.get(cfg.name)
    if cls is None:
        raise ValueError(f'Unknown classifier model class: {cfg.name!r}. Register it in get_classifier().')

    return cls()
