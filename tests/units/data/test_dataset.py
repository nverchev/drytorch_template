"""Tests for the data modules."""

import torch

from src.config import experiment, hydra
from src.data import split
from src.data.databuilder import DataBuilder
from src.data.dataset import get_databuilder, get_splits
from src.data.signal_dataset import SignalDataBuilder
from src.data.synthetic_dataset import SyntheticDataBuilder


class TestDataBuilder:
    """Test DataBuilder factory and data structure instantiation."""

    def test_get_databuilder_returns_instance(self) -> None:
        """Instantiate a DataBuilder object."""
        builder = get_databuilder()

        assert isinstance(builder, DataBuilder)

    def test_get_splits(self) -> None:
        """Build non-empty training and evaluation splits."""
        train_split, eval_split = get_splits()

        assert len(train_split)
        assert len(eval_split)


class TestSyntheticDataBuilder:
    """Test SyntheticDataBuilder initialization and sample indexing."""

    def test_synthetic_split_sample_shape(self) -> None:
        """Generate samples matching configured dimensions and label types."""
        cfg = hydra.get_experiment_config()
        databuilder = SyntheticDataBuilder()

        train_split = databuilder.get_split(split.Partitions.train)
        inputs, targets = train_split[0]

        assert len(train_split) == cfg.data.dataset.n_train_samples
        assert inputs.features.shape[-1] == cfg.data.dataset.in_features
        assert targets.label is not None

    def test_partition_statistics_non_final(self) -> None:
        """Centre training split while preserving uncentred test split in non-final runs."""
        tolerance = 1e-4
        databuilder = SyntheticDataBuilder()

        train_features = databuilder.split(split.Partitions.train).features
        test_features = databuilder.split(split.Partitions.test).features
        train_mean = train_features.mean(dim=0)
        train_std = train_features.std(dim=0)
        test_mean = test_features.mean(dim=0)

        assert torch.allclose(train_mean, torch.zeros_like(train_mean), atol=tolerance)
        assert torch.allclose(train_std, torch.ones_like(train_std), atol=1e-2)
        assert not torch.allclose(test_mean, torch.zeros_like(test_mean), atol=tolerance)


class TestSignalDataBuilder:
    """Test SignalDataBuilder initialization and sample indexing."""

    def test_signal_split_sample_shape(self, mocker) -> None:
        """Generate multichannel 1D sequence samples with corresponding labels."""
        cfg = hydra.get_experiment_config('signal_classifier')
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)
        databuilder = SignalDataBuilder()

        train_split = databuilder.get_split(split.Partitions.train)
        inputs, targets = train_split[0]

        assert len(train_split) == cfg.data.dataset.n_train_samples
        assert inputs.features.shape == (cfg.data.dataset.in_features, cfg.data.dataset.sequence_length)
        assert targets.label is not None
