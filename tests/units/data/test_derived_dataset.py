"""Tests for EncodedDataset derived dataset wrapper."""

import pytest
import torch

from src.config import experiment, hydra
from src.data.derived_dataset import EncodedDataset
from src.data.split import DataSplit
from src.module.autoencoders import CNNResAutoencoder
from src.lib.models import AutoencoderEMAModel


class TestEncodedDataset:
    """Test EncodedDataset latent code shape and evaluation mode persistence."""

    def test_encoded_dataset_codes_have_latent_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return sample features matching encoder latent dimension."""
        cfg = hydra.get_experiment_config('signal_classifier')
        monkeypatch.setattr(experiment.Experiment, 'get_config', lambda: cfg)
        autoencoder = CNNResAutoencoder()
        model = AutoencoderEMAModel(autoencoder, device=torch.device('cpu'), torch_compile=False)
        features = torch.randn(4, 1, cfg.data.dataset.sequence_length)
        labels = torch.tensor([0, 1, 2, 3])
        split = DataSplit(features, labels)
        dataset = EncodedDataset(split, model)

        inputs, targets = dataset[0]

        assert inputs.features.shape == (autoencoder.latent_dim,)
        assert targets.features.shape == (1, cfg.data.dataset.sequence_length)
        assert targets.label == labels[0]

    def test_encoder_placed_in_evaluation_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure autoencoder module is placed in evaluation mode upon building the dataset."""
        cfg = hydra.get_experiment_config('signal_classifier')
        monkeypatch.setattr(experiment.Experiment, 'get_config', lambda: cfg)
        autoencoder = CNNResAutoencoder()
        model = AutoencoderEMAModel(autoencoder, device=torch.device('cpu'), torch_compile=False)
        model.module.train()
        features = torch.randn(4, 1, cfg.data.dataset.sequence_length)
        labels = torch.tensor([0, 1, 2, 3])
        split = DataSplit(features, labels)

        _ = EncodedDataset(split, model)

        assert not model.module.training
        assert not model.averaged_module.training

    def test_training_augmentations_apply_before_encoding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Apply data augmentations to input features before passing to encoder."""
        cfg = hydra.get_experiment_config('signal_classifier')
        monkeypatch.setattr(experiment.Experiment, 'get_config', lambda: cfg)
        autoencoder = CNNResAutoencoder()
        model = AutoencoderEMAModel(autoencoder, device=torch.device('cpu'), torch_compile=False)

        def _offset(x: torch.Tensor) -> torch.Tensor:
            return x + 50.0

        monkeypatch.setattr('src.data.split.get_augmenter', lambda: _offset)
        features = torch.zeros(2, 1, cfg.data.dataset.sequence_length)
        labels = torch.tensor([0, 1])
        split = DataSplit(features, labels)
        dataset = EncodedDataset(split, model)

        inputs_train, _ = dataset[0]
        with torch.inference_mode():
            inputs_eval, _ = dataset[0]

        assert not torch.allclose(inputs_train.features, inputs_eval.features)
