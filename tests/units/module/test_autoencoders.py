"""Tests for the autoencoders module."""

import pytest
import torch

from src.data import structures
from src.module.autoencoders import CNNResAutoencoder, get_autoencoder


@pytest.mark.usefixtures('signal_classifier_config')
class TestCNNResAutoencoder:
    """Test the residual CNN autoencoder."""

    @pytest.fixture
    def autoencoder(self) -> CNNResAutoencoder:
        """Return the autoencoder of the configuration with an autoencoding phase."""
        return CNNResAutoencoder()

    def test_forward_returns_reconstruction_and_encodings(self, autoencoder: CNNResAutoencoder) -> None:
        """Reconstruct the input shape and encode each sample in the latent space."""
        features = torch.randn(8, autoencoder.in_dim, autoencoder.sequence_length)
        inputs = structures.Inputs(features=features)

        out = autoencoder(inputs)

        assert out.reconstruction.shape == features.shape
        assert out.encodings.shape == (8, autoencoder.latent_dim)


@pytest.mark.usefixtures('signal_classifier_config')
def test_get_autoencoder() -> None:
    """Build the configured autoencoder."""
    autoencoder = get_autoencoder()

    assert isinstance(autoencoder, CNNResAutoencoder)
