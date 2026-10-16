"""Autoencoder architectures built from layer primitives."""

import abc

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import Autoencoders
from src.data.structures import Inputs, Outputs
from src.module.layers import Conv1DResBlock, LinearLayer

__all__ = [
    'BaseAutoencoder',
    'CNNResAutoencoder',
    'get_autoencoder',
]


class BaseAutoencoder(nn.Module, abc.ABC):
    """Abstract base for all autoencoder model architectures.

    Attributes:
        in_dim: Number of input signal channels.
        sequence_length: Length of the signal sequence.
        latent_dim: Bottleneck dimension of the latent encodings.
    """

    in_dim: Final[int]
    sequence_length: Final[int]
    latent_dim: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        exp_cfg = Experiment.get_config()
        if exp_cfg.autoencoding is None:
            raise ValueError('Autoencoding configuration is missing.')

        self.in_dim = exp_cfg.data.dataset.in_features
        self.sequence_length = exp_cfg.data.dataset.sequence_length
        self.latent_dim = exp_cfg.autoencoding.model.architecture.latent_dim
        return

    @abc.abstractmethod
    def encode(self, inputs: Inputs | torch.Tensor, /) -> torch.Tensor:
        """Encode input signals into latent representations."""

    @abc.abstractmethod
    def decode(self, z: torch.Tensor, /) -> torch.Tensor:
        """Decode latent representations back into signal waveforms."""

    @abc.abstractmethod
    def forward(self, inputs: Inputs, /) -> Outputs:
        """Run forward pass."""


class CNNResAutoencoder(BaseAutoencoder):
    """1D CNN Autoencoder with residual blocks for representation learning.

    Attributes:
        encoder: Encoder module producing latent representations.
        decoder: Decoder module reconstructing signal waveforms.
    """

    encoder: Final[nn.Sequential]
    decoder: Final[nn.Sequential]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        exp_cfg = Experiment.get_config()
        if exp_cfg.autoencoding is None:
            raise ValueError('Autoencoding configuration is missing.')

        arch = exp_cfg.autoencoding.model.architecture
        enc_dims = [self.in_dim, *arch.hidden_dims]
        self.encoder = nn.Sequential(
            Conv1DResBlock(enc_dims, act_cls=arch.act_cls, norm_cls=arch.norm_cls),
            nn.Flatten(),
            LinearLayer(enc_dims[-1] * self.sequence_length, self.latent_dim),
        )
        dec_dims = [*reversed(arch.hidden_dims), self.in_dim]
        self.decoder = nn.Sequential(
            LinearLayer(self.latent_dim, arch.hidden_dims[-1] * self.sequence_length),
            nn.Unflatten(1, (arch.hidden_dims[-1], self.sequence_length)),
            Conv1DResBlock(dec_dims, act_cls=arch.act_cls, norm_cls=arch.norm_cls),
        )
        return

    @override
    def encode(self, inputs: Inputs | torch.Tensor, /) -> torch.Tensor:
        """Encode input signals into latent representations.

        Args:
            inputs: input batch containing signal features or raw feature tensor.

        Returns:
            latent representation tensor.
        """
        features = inputs.features if isinstance(inputs, Inputs) else inputs
        if features.ndim == 2:
            features = features.unsqueeze(1)

        return self.encoder(features)

    @override
    def decode(self, z: torch.Tensor, /) -> torch.Tensor:
        """Decode latent representations back into signal waveforms.

        Args:
            z: latent representation tensor.

        Returns:
            reconstructed signal waveform tensor.
        """
        return self.decoder(z)

    @override
    def forward(self, inputs: Inputs, /) -> Outputs:
        """Reconstruct input signals and return latent encodings.

        Args:
            inputs: input batch containing signal features.

        Returns:
            model outputs containing reconstruction and latent encodings.
        """
        z = self.encode(inputs)
        recon = self.decode(z)
        return Outputs(reconstruction=recon, encodings=z)


def get_autoencoder() -> BaseAutoencoder:
    """Instantiate an autoencoder architecture from configuration.

    Returns:
        instantiated autoencoder architecture.

    Raises:
        ValueError: if autoencoder configuration is missing or unregistered.
    """
    exp_cfg = Experiment.get_config()
    if exp_cfg.autoencoding is None:
        raise ValueError('Autoencoding configuration is missing.')

    cfg = exp_cfg.autoencoding.model
    model_map: dict[Autoencoders, type[BaseAutoencoder]] = {
        Autoencoders.CNN_RES_AUTOENCODER: CNNResAutoencoder,
    }

    cls = model_map.get(cfg.class_name)
    if cls is None:
        raise ValueError(f'Unknown autoencoder model class: {cfg.class_name!r}. Register it in get_autoencoder().')

    return cls()
