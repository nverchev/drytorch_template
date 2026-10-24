"""Classifier training entry point.

Usage::

    uv run train_classifier.py
    uv run train_classifier.py --config-name signal_classifier
"""

from typing import TYPE_CHECKING, Any

from drytorch import Trainer

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.data.dataset import get_splits
from src.data.derived_dataset import EncodedDataset
from src.module import get_classifier
from src.lib import (
    ClassifierModel,
    get_classification_loss,
    get_encoded_loaders,
    get_learning_schema,
    get_loaders,
)
from src.lib.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.lib.models import get_classifier_ema_model, load_autoencoder_model
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_classification(trial: Trial | None = None) -> None:
    """Set up model, data, and trainer then run training for classification.

    With an autoencoding phase, the classifier learns from the encodings of the trained autoencoder.

    Args:
        trial: optional Optuna trial for pruning.
    """
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    class_cfg = cfg.classification
    if trial is None:
        model = get_classifier_ema_model()
    else:
        model = ClassifierModel(
            get_classifier(),
            device=user_cfg.device,
            name=class_cfg.model.name,
            mixed_precision=class_cfg.train.mixed_precision,
            torch_compile=user_cfg.torch_compile,
        )

    train_split, eval_split = get_splits()
    autoencoder = None if cfg.autoencoding is None else load_autoencoder_model()
    if autoencoder is not None:
        encoded_train = EncodedDataset(train_split, autoencoder)
        encoded_eval = EncodedDataset(eval_split, autoencoder)
        train_loader, eval_loader = get_encoded_loaders(
            encoded_train,
            encoded_eval,
            batch_size=class_cfg.train.batch_size_per_device,
        )
    else:
        train_loader, eval_loader = get_loaders(
            batch_size=class_cfg.train.batch_size_per_device,
            n_workers=user_cfg.n_workers,
        )

    loss = get_classification_loss()
    schema = get_learning_schema(class_cfg)
    builder = Trainer.builder(
        model,
        learning_schema=schema,
        loader=train_loader,
        loss=loss,
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    if not cfg.final and class_cfg.train.early_stopping.active:
        cfg_early = class_cfg.train.early_stopping
        register_early_stopping(builder, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(builder, trial, window=class_cfg.train.early_stopping.window)
    else:
        register_checkpointing(builder, checkpoint_every=user_cfg.checkpoint_every)

    trainer = builder.build()
    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    trainer.train_until(class_cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Initialise the experiment run and launch classifier training."""
    trackers = get_trackers(cfg)
    has_autoencoder = cfg.autoencoding is not None
    resume = has_autoencoder or cfg.user.load != 0 or cfg.user.run_id is not None
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_classification()

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point — spawns DDP workers when ``n_subprocesses > 0``."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)

    return


if __name__ == '__main__':
    main()
