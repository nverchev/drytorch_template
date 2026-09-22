"""Configuration module with shared test fixtures."""

from collections.abc import Generator

import pytest
import torch

from src.config import AllConfig, Experiment, get_config_all
from src.data.dataset import get_databuilder
from src.data.structures import Inputs, Targets


@pytest.fixture(scope='package')
def default_config() -> AllConfig:
    """Return default AllConfig object."""
    return get_config_all()


@pytest.fixture(autouse=True)
def active_experiment(default_config: AllConfig) -> Generator[Experiment]:
    """Fixture ensuring an active Experiment run is present for all tests."""
    exp = Experiment(default_config, name='test_exp', par_dir=default_config.user.path.version_dir)
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
    inputs = Inputs(features=torch.randn(8, 64))
    targets = Targets(label=torch.randint(0, 6, (8,)))
    return inputs, targets
