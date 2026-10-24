"""Tests for the Hydra integration module."""

from src.config import experiment, options, specs
from src.config.hydra import get_experiment_config


class TestHydraHelpers:
    """Test Hydra config composition and override tracking."""

    def test_get_experiment_config_default(self, mocker) -> None:
        """Compose default typed configuration."""
        mocker.patch('src.config.specs.get_current_branch', return_value='main')

        cfg = get_experiment_config()

        assert cfg.classification.model.name == 'mlp_classifier'
        assert cfg.user.device == 'auto'

    def test_get_experiment_config_with_overrides(self, mocker) -> None:
        """Apply overrides to typed configuration."""
        mocker.patch('src.config.specs.get_current_branch', return_value='main')
        overrides = ['classification.train.n_epochs=25']

        cfg = get_experiment_config(overrides=overrides)

        assert cfg.classification.train.n_epochs == 25
        assert cfg.tags == ['classification.train.n_epochs=25']

    def test_get_experiment_config_starts_from_the_named_file(self, mocker) -> None:
        """The named entry configuration is composed and recorded, and its runs get their own folder."""
        mocker.patch('src.config.specs.get_current_branch', return_value='main')

        cfg = get_experiment_config('signal_classifier')

        assert cfg.autoencoding is not None
        assert isinstance(cfg.autoencoding, specs.AutoencodingConfig)
        assert cfg.autoencoding.model.name == options.Autoencoders.CNN_RES_AUTOENCODER
        assert cfg.classification.model.name == options.Classifiers.LINEAR_CLASSIFIER
        assert cfg.user.hydra.config_name == 'signal_classifier'
        assert cfg.par_dir == cfg.user.path.version_dir / 'signal_classifier'

    def test_get_variation_filters_user_keys(self, mocker) -> None:
        """Filter user-specific overrides from variation string."""
        mocker.patch('src.config.experiment.get_current_branch', return_value='main')
        overrides = ['user.device=cpu', 'classification.train.n_epochs=50']

        variation, tags = experiment.get_variation(None, overrides)

        assert variation == 'main_classification.train.n_epochs=50'
        assert tags == ['classification.train.n_epochs=50']
