"""Configuration module with the fixtures of the module tests."""

import pytest
import pytest_mock

from src.config import ExperimentConfig, experiment, hydra


@pytest.fixture
def signal_classifier_config(mocker: pytest_mock.MockerFixture) -> ExperimentConfig:
    """Return the configuration with an autoencoding phase, which the experiment provides."""
    cfg = hydra.get_experiment_config('signal_classifier')
    mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)
    return cfg
