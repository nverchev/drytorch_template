"""Tests for the config specs module."""

import dataclasses
import pathlib
import pytest
import torch

from src.config import environment, hydra
from src.config.specs import (
    ExperimentConfig,
    AutoencodingConfig,
    AutoencoderModelConfig,
    PhaseConfig,
    LinearArchitectureConfig,
    PathSpecs,
    TrainingConfig,
)


class TestLinearArchitectureConfig:
    """Test LinearArchitectureConfig initialization and defaults."""

    def test_default_activation_and_norm(self) -> None:
        """Resolve default activation and normalization classes."""
        arch = LinearArchitectureConfig()

        assert arch.act_cls is not None
        assert arch.norm_cls is not None

    def test_custom_activation_and_norm(self) -> None:
        """Resolve custom activation and normalization class names."""
        arch = LinearArchitectureConfig(act_cls='GELU', norm_cls='LayerNorm')

        assert arch.act_cls.__name__ == 'GELU'
        assert arch.norm_cls.__name__ == 'LayerNorm'


class TestTrainingConfig:
    """Test TrainingConfig batch size calculation and validation."""

    def test_batch_size_per_device_single(self, default_config: ExperimentConfig) -> None:
        """Compute batch size per device when no subprocesses are configured."""
        train_cfg = default_config.classification.train

        assert train_cfg.batch_size_per_device == train_cfg.batch_size

    def test_check_batch_divisibility_raises(self, default_config: ExperimentConfig) -> None:
        """Raise ValueError when batch size is not divisible by subprocess count."""
        train_cfg = default_config.classification.train

        with pytest.raises(ValueError, match='Global batch size'):
            TrainingConfig(
                batch_size=17,
                learn=train_cfg.learn,
                n_epochs=10,
                early_stopping=train_cfg.early_stopping,
                _n_subprocesses=4,
            )

    def test_mixed_precision_by_name(self) -> None:
        """Convert the name of the precision in the configuration files to a dtype."""
        cfg = hydra.get_experiment_config(overrides=['classification.train.mixed_precision=float16'])

        assert cfg.classification.train.mixed_precision is torch.float16

    def test_mixed_precision_survives_replace(self, default_config: ExperimentConfig) -> None:
        """Keep the dtype when a copy of the configuration validates it again."""
        train_cfg = dataclasses.replace(default_config.classification.train, mixed_precision=torch.bfloat16)

        replaced = dataclasses.replace(train_cfg, n_epochs=1)

        assert replaced.mixed_precision is torch.bfloat16


class TestPathSpecs:
    """Test PathSpecs version directory computation."""

    def test_version_dir(self) -> None:
        """Construct version directory path including project version."""
        specs = PathSpecs(root_exp_dir=pathlib.Path('./exp_test'))

        assert specs.version_dir == pathlib.Path(f'./exp_test/v{environment.VERSION}')


class TestExperimentConfig:
    """Test ExperimentConfig properties and variation handling."""

    def test_resolved_variation_fallback(self, mocker) -> None:
        """Fall back to git branch name when variation is None."""
        mocker.patch('src.config.specs.get_current_branch', return_value='main')

        cfg = dataclasses.replace(hydra.get_experiment_config(), variation=None)

        assert cfg.resolved_variation == 'main'

    def test_resolved_variation_explicit(self) -> None:
        """Return explicit variation string when provided."""
        cfg = dataclasses.replace(hydra.get_experiment_config(), variation='custom_var')

        assert cfg.resolved_variation == 'custom_var'

    def test_name_property_with_final_flag(self) -> None:
        """Append final suffix to experiment name when final flag is set."""
        cfg = dataclasses.replace(hydra.get_experiment_config(), variation='test_run', final=True)

        assert cfg.name == 'test_run_final'


class TestAutoencodingConfig:
    """Test AutoencodingConfig subclassing and attributes."""

    def test_subclasses_phase_config(self) -> None:
        """Inherit phase fields from PhaseConfig."""
        cfg = hydra.get_experiment_config()
        ae_cfg = AutoencodingConfig(
            name='autoencoder',
            model=AutoencoderModelConfig(),
            train=cfg.classification.train,
        )

        assert isinstance(ae_cfg, PhaseConfig)
        assert ae_cfg.name == 'autoencoder'
