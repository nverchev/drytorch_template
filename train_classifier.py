"""Training entry point.

Usage::

    uv run train_classifier.py
    uv run train_classifier.py classification.train.learn.learning_rate=1e-4
    uv run train_classifier.py final=true
"""

from src.train.metrics_and_losses import AggregatedAccuracy

import sys

from typing import TYPE_CHECKING, Any
from drytorch import Test, Trainer

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.data.dataset import get_databuilder
from src.module import get_model
from src.train import get_learning_schema, get_loaders, get_loss
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.train.models import ClassifierModel, ClassifierEMAModel
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


DEBUG_MODE = any(module in sys.modules for module in ('pydevd', 'debugpy', 'pdb')) or sys.gettrace() is not None


def train_classification(trial: Trial | None = None) -> None:
    """Set up model, data, and trainer then run training for classification."""
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    class_cfg = cfg.classification

    architecture = get_model()
    if trial is None:
        model = ClassifierEMAModel(architecture, name=class_cfg.name, device=user_cfg.device)

    else:
        model = ClassifierModel(architecture, name=class_cfg.name, device=user_cfg.device)

    train_loader, eval_loader = get_loaders(
        batch_size=class_cfg.train.batch_size_per_device, n_workers=user_cfg.n_workers
    )

    n_classes = get_databuilder().n_classes
    loss = get_loss()
    accuracy = AggregatedAccuracy(n_classes)
    schema = get_learning_schema(class_cfg)

    trainer = Trainer(model, name=f'Trainer_{class_cfg.name}', loader=train_loader, loss=loss, learning_schema=schema)
    test = Test(model, name=f'Test_{class_cfg.name}', loader=eval_loader, metric=accuracy)

    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    if not cfg.final:
        trainer.add_validation(eval_loader, f'Validation_{class_cfg.name}')

    if not cfg.final and class_cfg.train.early_stopping.active:
        cfg_early = class_cfg.train.early_stopping
        register_early_stopping(trainer, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(trainer, trial)

    else:
        register_checkpointing(trainer, user_cfg.checkpoint_every or 1)

    trainer.train_until(class_cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    test()
    return


def setup_and_train(cfg: AllConfig) -> None:
    """Initialise the experiment run and launch training."""
    trackers = get_trackers(cfg, DEBUG_MODE)
    resume = cfg.user.load != 0
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, record=not DEBUG_MODE):
        train_classification()

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Entry point — spawns DDP workers when ``n_subprocesses > 0``."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        DistributedWorker(setup_and_train, n_processes).spawn(cfg)

    else:
        setup_and_train(cfg)

    return


if __name__ == '__main__':
    main()
