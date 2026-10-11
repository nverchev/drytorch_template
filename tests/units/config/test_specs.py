"""Tests for the config specs module."""

import dataclasses
import pathlib
import pytest

from src.config.environment import VERSION
from src.config.specs import AllConfig, LinearArchitectureConfig, TrainingConfig, PathSpecs
from src.config import get_config_all


class TestLinearArchitectureConfig:
    """Test LinearArchitectureConfig initialization and defaults."""

    def test_default_activation_and_norm(self) -> None:
        """Test default activation and normalization resolution."""
        arch = LinearArchitectureConfig()
        assert arch.act_cls is not None
        assert arch.norm_cls is not None
        return

    def test_custom_activation_and_norm(self) -> None:
        """Test resolving custom activation and normalization class names."""
        arch = LinearArchitectureConfig(act_cls='GELU', norm_cls='LayerNorm')
        assert arch.act_cls.__name__ == 'GELU'
        assert arch.norm_cls.__name__ == 'LayerNorm'
        return


class TestTrainingConfig:
    """Test TrainingConfig batch size calculation and validation."""

    def test_batch_size_per_device_single(self, default_config: AllConfig) -> None:
        """Test batch size calculation when no subprocesses are configured."""
        train_cfg = default_config.classification.train
        assert train_cfg.batch_size_per_device == train_cfg.batch_size
        return

    def test_check_batch_divisibility_raises(self, default_config: AllConfig) -> None:
        """Test error raised when batch size is not divisible by n_subprocesses."""
        train_cfg = default_config.classification.train
        with pytest.raises(ValueError, match='Global batch size'):
            TrainingConfig(
                batch_size=17,
                learn=train_cfg.learn,
                n_epochs=10,
                early_stopping=train_cfg.early_stopping,
                _n_subprocesses=4,
            )

        return


class TestPathSpecs:
    """Test PathSpecs version directory computation."""

    def test_version_dir(self) -> None:
        """Test version_dir path construction."""
        specs = PathSpecs(root_exp_dir=pathlib.Path('./exp_test'))
        assert specs.version_dir == pathlib.Path(f'./exp_test/v{VERSION}')
        return


class TestAllConfig:
    """Test AllConfig properties and variation handling."""

    def test_resolved_variation_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolved_variation uses git_branch if variation is None."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        cfg = dataclasses.replace(get_config_all(), variation=None)
        assert cfg.resolved_variation == 'main'
        return

    def test_resolved_variation_explicit(self) -> None:
        """Test resolved_variation returns explicit variation string."""
        cfg = dataclasses.replace(get_config_all(), variation='custom_var')
        assert cfg.resolved_variation == 'custom_var'
        return

    def test_name_property_with_final_flag(self) -> None:
        """Test experiment name suffix when final flag is set."""
        cfg = dataclasses.replace(get_config_all(), variation='test_run', final=True)
        assert cfg.name == 'test_run_final'
        return
