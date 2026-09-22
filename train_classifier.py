"""Training entry point.

Usage::

    uv run train_classifier.py
    uv run train_classifier.py classification.train.learn.lr=1e-4
    uv run train_classifier.py final=true
"""

import sys

from typing import TYPE_CHECKING, Any, cast

from drytorch import Test, Trainer, Validation

from src.config import AllConfig, Experiment, get_trackers, hydra_main
from src.config.events import ConfusionMatrixEvent
from src.data.dataset import get_databuilder
from src.module import get_model
from src.train import get_learning_schema, get_loaders, get_loss
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.train.models import ClassifierModel, ClassifierEMAModel
from src.train.objectives import AggregatedAccuracy
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


DEBUG_MODE = any(module in sys.modules for module in ('pydevd', 'debugpy')) or sys.gettrace() is not None


def train_classification(trial: Trial | None = None) -> None:
    """Set up model, data, and trainer then run training for classification."""
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    class_cfg = cfg.classification
    architecture = get_model()
    if trial is None:
        model = ClassifierEMAModel(
            architecture,
            device=user_cfg.device,
            name=class_cfg.name,
            torch_compile=user_cfg.torch_compile,
        )

    else:
        model = ClassifierModel(
            architecture,
            device=user_cfg.device,
            name=class_cfg.name,
            torch_compile=user_cfg.torch_compile,
        )

    train_loader, eval_loader = get_loaders(
        batch_size=class_cfg.train.batch_size_per_device, n_workers=user_cfg.n_workers
    )
    n_classes = get_databuilder().n_classes
    loss = get_loss()
    accuracy = AggregatedAccuracy(n_classes)
    schema = get_learning_schema(class_cfg)
    builder = Trainer.builder(
        model,
        learning_schema=schema,
        loader=train_loader,
        loss=loss,
    )
    if not cfg.final:
        builder.add_validation(eval_loader, val_name=f'LossValidation')

    if not cfg.final and class_cfg.train.early_stopping.active:
        cfg_early = class_cfg.train.early_stopping
        register_early_stopping(builder, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(builder, trial)

    else:
        register_checkpointing(builder, user_cfg.checkpoint_every or 1)

    trainer = builder.build()
    if cfg.final:
        evaluation = Test(model, loader=eval_loader, metric=accuracy)
    else:
        evaluation = Validation(model, loader=eval_loader, metric=accuracy)

    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    trainer.train_until(class_cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    evaluation()

    # use a custom event to send non-scalar metrics to trackers
    evaluated_accuracy = cast(AggregatedAccuracy, evaluation.objective)
    ConfusionMatrixEvent(
        source_name=evaluation.name,
        model_name=model.name,
        epoch=model.epoch,
        matrix=evaluated_accuracy.confusion_matrix,
    )
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
