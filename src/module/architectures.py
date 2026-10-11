"""Classifier architectures built from the layer primitives in layers.py."""

import abc
import itertools

from typing import Final

import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import Models
from src.config.torch import ActClass, NormClass
from src.data.dataset import get_databuilder
from src.data.structures import Inputs, Outputs
from src.module.layers import LinearLayer, LinearResBlock

__all__ = [
    'BaseArchitecture',
    'MLPClassifier',
    'MLPResClassifier',
    'get_model',
]


class BaseArchitecture(nn.Module, abc.ABC):
    """Abstract base for all model architectures, storing the settings they share.

    Attributes:
        in_dim: features of every sample.
        hidden_dims: dimensions of the hidden layers.
        n_classes: classes to predict.
        act_cls: activation of the hidden layers.
        norm_cls: normalization of the hidden layers.
        dropout_rate: dropout probability after the hidden layers.
    """

    in_dim: Final[int]
    hidden_dims: Final[tuple[int, ...]]
    n_classes: Final[int]
    act_cls: Final[ActClass]
    norm_cls: Final[NormClass]
    dropout_rate: Final[float]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        exp_cfg = Experiment.get_config()
        arch = exp_cfg.classification.model.architecture
        self.in_dim = exp_cfg.data.dataset.in_features
        self.hidden_dims = arch.hidden_dims
        self.n_classes = get_databuilder().n_classes
        self.act_cls = arch.act_cls
        self.norm_cls = arch.norm_cls
        self.dropout_rate = arch.dropout_rate
        return

    @abc.abstractmethod
    def forward(self, inputs: Inputs, /) -> Outputs:
        """Run the forward pass."""


class MLPClassifier(BaseArchitecture):
    """Feedforward MLP: in_features → hidden_dims → n_classes."""

    layers: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        dims = (self.in_dim, *self.hidden_dims)
        self.layers = nn.Sequential()
        for in_dim, out_dim in itertools.pairwise(dims):
            self.layers.append(LinearLayer(in_dim, out_dim, act_cls=self.act_cls, norm_cls=self.norm_cls))
            if self.dropout_rate > 0:
                self.layers.append(nn.Dropout(self.dropout_rate))

        self.layers.append(LinearLayer(dims[-1], self.n_classes))
        return

    def forward(self, inputs: Inputs, /) -> Outputs:
        """Forward pass through the hidden layers and the output layer."""
        return Outputs(logits=self.layers(inputs.features))


class MLPResClassifier(BaseArchitecture):
    """MLP with residual blocks: in_features → hidden_dims → n_classes."""

    layers: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        dims = [self.in_dim, *self.hidden_dims]
        self.layers = nn.Sequential(LinearResBlock(dims, act_cls=self.act_cls, norm_cls=self.norm_cls))
        if self.dropout_rate > 0:
            self.layers.append(nn.Dropout(self.dropout_rate))

        self.layers.append(LinearLayer(dims[-1], self.n_classes))
        return

    def forward(self, inputs: Inputs, /) -> Outputs:
        """Forward pass through the residual block and the output layer."""
        return Outputs(logits=self.layers(inputs.features))


def get_model() -> BaseArchitecture:
    """Instantiate an architecture from its configuration."""
    cfg = Experiment.get_config().classification.model
    _REGISTRY: dict[Models, type[BaseArchitecture]] = {
        Models.MLP_CLASSIFIER: MLPClassifier,
        Models.MLP_RES_CLASSIFIER: MLPResClassifier,
    }

    cls = _REGISTRY.get(cfg.class_name)
    if cls is None:
        raise ValueError(f'Unknown model class: {cfg.class_name!r}. Register it in get_model().')

    return cls()
