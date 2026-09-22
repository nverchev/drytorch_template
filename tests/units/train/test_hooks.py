"""Tests for training hooks registration."""

from unittest.mock import MagicMock

import torch

from drytorch import DataLoader, Trainer
from drytorch.contrib.optuna import TrialCallback
from drytorch.lib.hooks import EarlyStopping, SavingHook
from src.config.experiment import Experiment
from src.data.dataset import get_datasets
from src.module import get_model
from src.train import get_learning_schema, get_loss
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.train.models import ClassifierModel


def _get_test_builder():
    cfg = Experiment.get_config()
    model = ClassifierModel(
        get_model(),
        device=torch.device('cpu'),
        torch_compile=False,
        distribute=False,
    )
    dataset, _ = get_datasets()
    loader = DataLoader(dataset, batch_size=4)
    schema = get_learning_schema(cfg.classification)
    builder = Trainer.builder(
        model,
        name='test_trainer',
        loader=loader,
        loss=get_loss(),
        learning_schema=schema,
    )
    builder.add_validation(loader, val_name='test_val')
    return builder


class TestHooks:
    """Test hook registration functions."""

    def test_register_checkpointing(self) -> None:
        """Test checkpointing hook registration."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)
        register_checkpointing(builder, checkpoint_every=None)
        assert len(builder._post_epoch_hooks) == initial_count
        register_checkpointing(builder, checkpoint_every=2)
        assert len(builder._post_epoch_hooks) == initial_count + 1
        assert isinstance(builder._post_epoch_hooks[-1], SavingHook)
        return

    def test_register_early_stopping(self) -> None:
        """Test early stopping hook registration."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)
        register_early_stopping(builder, window=5, patience=3)
        assert len(builder._post_epoch_hooks) == initial_count + 1
        assert isinstance(builder._post_epoch_hooks[-1], EarlyStopping)
        return

    def test_register_pruning(self) -> None:
        """Test pruning hook registration with trial."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)
        mock_trial = MagicMock()
        register_pruning(builder, trial=mock_trial)
        assert len(builder._post_epoch_hooks) == initial_count + 1
        assert isinstance(builder._post_epoch_hooks[-1], TrialCallback)
        return
