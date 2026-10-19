"""Tests for module architectures and model factories."""

import torch

from src.config import experiment, hydra
from src.data import structures
from src.module import (
    CNNResAutoencoder,
    LinearClassifier,
    MLPClassifier,
    get_autoencoder,
    get_classifier,
)


class TestMLPClassifier:
    """Test MLPClassifier forward pass and output dimensions."""

    def test_mlp_classifier_forward(self, sample_batch, n_classes) -> None:
        """Produce logits matching expected class dimensions."""
        inputs, _ = sample_batch
        classifier = MLPClassifier()

        out = classifier(inputs)

        assert out.shape == (8, n_classes)


class TestCNNResAutoencoder:
    """Test CNNResAutoencoder forward pass and reconstruction."""

    def test_cnn_res_autoencoder_forward(self, mocker) -> None:
        """Produce sequence reconstruction and bottleneck encodings."""
        cfg = hydra.get_experiment_config('signal_classifier')
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)
        autoencoder = CNNResAutoencoder()
        ds_cfg = cfg.data.dataset
        inputs = structures.Inputs(features=torch.randn(8, ds_cfg.in_features, ds_cfg.sequence_length))

        out = autoencoder(inputs)

        assert out.reconstruction.shape == (8, ds_cfg.in_features, ds_cfg.sequence_length)
        assert out.encodings.shape == (8, autoencoder.latent_dim)


class TestLinearClassifier:
    """Test LinearClassifier on encodings."""

    def test_linear_classifier_forward(self, mocker) -> None:
        """Project encoder bottleneck representations to class logits."""
        cfg = hydra.get_experiment_config('signal_classifier')
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)
        classifier = LinearClassifier()
        inputs = structures.Inputs(features=torch.randn(8, classifier.in_dim))

        out = classifier(inputs)

        assert out.shape == (8, 4)


class TestFactories:
    """Test phase factory functions get_classifier and get_autoencoder."""

    def test_get_classifier_builds_configured_class(self) -> None:
        """Instantiate configured classifier PyTorch module."""
        classifier = get_classifier()

        assert isinstance(classifier, MLPClassifier)

    def test_get_autoencoder_builds_configured_class(self, mocker) -> None:
        """Instantiate configured autoencoder PyTorch module."""
        cfg = hydra.get_experiment_config('signal_classifier')
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)

        autoencoder = get_autoencoder()

        assert isinstance(autoencoder, CNNResAutoencoder)
