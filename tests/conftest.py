"""Configuration module with shared test fixtures."""

from collections.abc import Generator
import pathlib

import pytest
import torch

from src.config import ExperimentConfig, Experiment, get_experiment_config
from src.data.dataset import get_databuilder
from src.data.structures import Inputs, Targets


@pytest.fixture(scope='package')
def default_config() -> ExperimentConfig:
    """Return the configuration without overrides."""
    return get_experiment_config()


@pytest.fixture
def overrides() -> list[str]:
    """Return the overrides of the configuration; a test parametrizes it to change settings."""
    return []


@pytest.fixture(autouse=True)
def active_experiment(
    overrides: list[str], default_config: ExperimentConfig, tmp_path: pathlib.Path
) -> Generator[Experiment]:
    """Provide an active run of the configuration with the overrides, for every test."""
    config = get_experiment_config(overrides=overrides) if overrides else default_config
    exp = Experiment(config, name='test_exp', par_dir=tmp_path)
    with exp.create_run(record=False):
        yield exp

    return


@pytest.fixture
def n_classes() -> int:
    """Return the number of classes of the configured dataset."""
    return get_databuilder().n_classes


@pytest.fixture(scope='package')
def sample_batch() -> tuple[Inputs, Targets]:
    """Return sample batch of inputs and targets."""
    features = torch.randn(8, 64)
    inputs = Inputs(features=features)
    targets = Targets(label=torch.randint(0, 6, (8,)), features=features)
    return inputs, targets
