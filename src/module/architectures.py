"""Classifier architectures built from the layer primitives in layers.py."""

import abc
import itertools

import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import Models
from src.data.dataset import get_databuilder
from src.data.structures import Inputs, Outputs
from src.module.layers import LinearLayer, LinearResBlock


class BaseArchitecture(nn.Module, abc.ABC):
    """Abstract base for all model architectures."""

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Run the forward pass."""


class MLPClassifier(BaseArchitecture):
    """Feedforward MLP: in_features → hidden_dims → n_classes."""

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        cfg = exp_cfg.classification.model
        arch = cfg.architecture
        dims = (exp_cfg.data.dataset.in_features, *arch.hidden_dims)
        self.hidden = nn.Sequential()
        for in_dim, out_dim in itertools.pairwise(dims):
            self.hidden.append(LinearLayer(in_dim, out_dim, act_cls=arch.act_cls, norm_cls=arch.norm_cls))
            if arch.dropout_rate > 0:
                self.hidden.append(nn.Dropout(arch.dropout_rate))

        n_classes = get_databuilder().n_classes
        self.head = LinearLayer(dims[-1], n_classes)
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass through hidden layers then output head."""
        x = self.hidden(inputs.features)
        return Outputs(logits=self.head(x))


class MLPResClassifier(BaseArchitecture):
    """MLP with residual blocks: in_features → hidden_dims → n_classes."""

    def __init__(self) -> None:
        super().__init__()
        exp_cfg = Experiment.get_config()
        cfg = exp_cfg.classification.model
        arch = cfg.architecture
        dims = (exp_cfg.data.dataset.in_features, *arch.hidden_dims)
        self.res_block = LinearResBlock(dims=list(dims), act_cls=arch.act_cls, norm_cls=arch.norm_cls)
        self.dropout = nn.Dropout(arch.dropout_rate) if arch.dropout_rate > 0 else nn.Identity()
        n_classes = get_databuilder().n_classes
        self.head = LinearLayer(dims[-1], n_classes)
        return

    def forward(self, inputs: Inputs) -> Outputs:
        """Forward pass through residual block then output head."""
        x = self.res_block(inputs.features)
        return Outputs(logits=self.head(self.dropout(x)))


def get_model() -> BaseArchitecture:
    """Instantiate an architecture from its configuration."""
    cfg = Experiment.get_config().classification.model
    _REGISTRY: dict[Models, type[BaseArchitecture]] = {
        Models.MLPClassifier: MLPClassifier,
        Models.MLPResClassifier: MLPResClassifier,
    }

    cls = _REGISTRY.get(cfg.class_name)
    if cls is None:
        raise ValueError(f'Unknown model class: {cfg.class_name!r}. Register it in get_model().')

    return cls()
