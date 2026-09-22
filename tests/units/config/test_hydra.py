"""Tests for the Hydra integration module."""

import pytest

from src.config.hydra import get_config_all
from src.config.experiment import update_variation


class TestHydraHelpers:
    """Test Hydra config composition and override tracking."""

    def test_get_config_all_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting default typed configuration."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        cfg = get_config_all()
        assert cfg.classification.name == 'classification'
        assert isinstance(cfg.user.cpu, bool)
        return

    def test_get_config_all_with_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying overrides to get_config_all."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        overrides = ['classification.train.n_epochs=25']
        cfg = get_config_all(overrides=overrides)
        assert cfg.classification.train.n_epochs == 25
        assert cfg.tags == ['classification.train.n_epochs=25']
        return

    def test_update_variation_filters_user_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test update_variation ignores user-specific overrides."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        cfg = get_config_all()
        overrides = ['user.cpu=true', 'classification.train.n_epochs=50']
        update_variation(cfg, overrides)
        assert cfg.tags == ['classification.train.n_epochs=50']
        return
