"""Package for configuring the experiment."""

from src.config.specs import (
    AutoencodingConfig,
    ClassificationConfig,
    ExperimentConfig,
    PhaseConfig,
)
from src.config.hydra import hydra_main, get_experiment_config
from src.config.experiment import Experiment
from src.config.environment import PROJECT_NAME, VERSION, ConfigPath
from src.config.torch import ActClass, NormClass

__all__ = [
    'PROJECT_NAME',
    'VERSION',
    'ActClass',
    'AutoencodingConfig',
    'ClassificationConfig',
    'ConfigPath',
    'Experiment',
    'ExperimentConfig',
    'NormClass',
    'PhaseConfig',
    'get_experiment_config',
    'hydra_main',
]
