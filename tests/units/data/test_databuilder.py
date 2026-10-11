"""Tests for the DataBuilder caching metaclass."""

from collections.abc import Generator

import pytest
from pytest_mock import MockerFixture

from src.config.experiment import Experiment
from src.config.hydra import get_config_all
from src.config.specs import AllConfig
from src.data.databuilder import ConfigCached
from src.data.synthetic_dataset import SyntheticDataset


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
        """Test that two constructions under the same configuration return the same instance."""
        instance1 = SyntheticDataset()
        instance2 = SyntheticDataset()
        assert instance1 is instance2
        return

    def test_changed_data_configuration_returns_different_instance(
        self,
        default_config: AllConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that construction under a changed data configuration returns a different instance."""
        instance1 = SyntheticDataset()
        n_train_samples = default_config.data.dataset.n_train_samples + 10
        cfg_changed = get_config_all(overrides=[f'data.dataset.n_train_samples={n_train_samples}'])
        mocker.patch.object(Experiment, 'get_config', return_value=cfg_changed)
        instance2 = SyntheticDataset()
        assert instance1 is not instance2
        return

    def test_changed_non_data_configuration_returns_same_instance(
        self,
        default_config: AllConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that construction with changes outside data configuration reuses the instance."""
        instance1 = SyntheticDataset()
        n_epochs = default_config.classification.train.n_epochs + 5
        cfg_changed = get_config_all(overrides=[f'classification.train.n_epochs={n_epochs}'])
        mocker.patch.object(Experiment, 'get_config', return_value=cfg_changed)
        instance2 = SyntheticDataset()
        assert instance1 is instance2
        return
