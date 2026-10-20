"""Tests for training hooks registration."""

from typing import Any

import torch

from drytorch import DataLoader, Trainer
from drytorch.contrib.optuna import TrialCallback
from drytorch.lib.hooks import EarlyStopping, SavingHook
from drytorch.lib.training import TrainerBuilder
from src import module
from src.config import experiment
from src.data import dataset
from src.train import learning_schema, models, objectives
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning


def _get_test_builder() -> TrainerBuilder[Any, Any, Any]:
    cfg = experiment.Experiment.get_config()
    model = models.ClassifierModel(
        module.get_classifier(),
        device=torch.device('cpu'),
        torch_compile=False,
        distribute=False,
    )
    data, _ = dataset.get_splits()
    loader = DataLoader(data, batch_size=4)
    schema = learning_schema.get_learning_schema(cfg.classification)
    builder = Trainer.builder(
        model,
        name='test_trainer',
        loader=loader,
        loss=objectives.get_classification_loss(),
        learning_schema=schema,
    )
    builder.add_validation(loader, val_name='test_val')
    return builder


class TestHooks:
    """Test hook registration functions."""

    def test_register_checkpointing_noop_when_zero(self) -> None:
        """Skip registering checkpointing hook when interval is zero."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)

        register_checkpointing(builder, checkpoint_every=0)

        assert len(builder._post_epoch_hooks) == initial_count

    def test_register_checkpointing_when_specified(self) -> None:
        """Register model checkpointing hook when interval is specified."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)

        register_checkpointing(builder, checkpoint_every=2)

        assert len(builder._post_epoch_hooks) == initial_count + 1
        assert isinstance(builder._post_epoch_hooks[-1], SavingHook)

    def test_register_early_stopping(self) -> None:
        """Register early stopping hook with configured window and patience."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)

        register_early_stopping(builder, window=5, patience=3)

        assert len(builder._post_epoch_hooks) == initial_count + 1
        assert isinstance(builder._post_epoch_hooks[-1], EarlyStopping)

    def test_register_pruning(self, mocker) -> None:
        """Register trial callback hook for hyperparameter pruning."""
        builder = _get_test_builder()
        initial_count = len(builder._post_epoch_hooks)
        mock_trial = mocker.Mock()

        register_pruning(builder, trial=mock_trial, window=1)

        assert len(builder._post_epoch_hooks) == initial_count + 1
        assert isinstance(builder._post_epoch_hooks[-1], TrialCallback)
