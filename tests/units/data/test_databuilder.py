"""Tests for the DataBuilder caching metaclass."""

from collections.abc import Generator

import pytest

from src.config import experiment, hydra, specs
from src.data import synthetic_dataset
from src.data.databuilder import ConfigCached


class TestConfigCached:
    """Test ConfigCached instance reuse and cache key behavior."""

    @pytest.fixture(autouse=True)
    def clean_instances_cache(self) -> Generator[None]:
        """Clear ConfigCached instances cache before and after test execution."""
        ConfigCached._instances.clear()
        yield
        ConfigCached._instances.clear()
        return

    def test_same_configuration_returns_same_instance(self) -> None:
        """Reuse cached instance under identical configuration."""
        instance1 = synthetic_dataset.SyntheticDataBuilder()
        instance2 = synthetic_dataset.SyntheticDataBuilder()

        assert instance1 is instance2

    def test_changed_data_configuration_returns_different_instance(
        self,
        default_config: specs.ExperimentConfig,
        mocker,
    ) -> None:
        """Create new instance when data configuration changes."""
        instance1 = synthetic_dataset.SyntheticDataBuilder()
        n_train_samples = default_config.data.dataset.n_train_samples + 10
        cfg_changed = hydra.get_experiment_config(overrides=[f'data.dataset.n_train_samples={n_train_samples}'])
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg_changed)

        instance2 = synthetic_dataset.SyntheticDataBuilder()

        assert instance1 is not instance2

    def test_changed_non_data_configuration_returns_same_instance(
        self,
        default_config: specs.ExperimentConfig,
        mocker,
    ) -> None:
        """Reuse cached instance when non-data configuration changes."""
        instance1 = synthetic_dataset.SyntheticDataBuilder()
        n_epochs = default_config.classification.train.n_epochs + 5
        cfg_changed = hydra.get_experiment_config(overrides=[f'classification.train.n_epochs={n_epochs}'])
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg_changed)

        instance2 = synthetic_dataset.SyntheticDataBuilder()

        assert instance1 is instance2
