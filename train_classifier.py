"""Classifier training entry point.

Usage::

    uv run train_classifier.py
    uv run train_classifier.py --config-name signal_classifier
"""

from typing import TYPE_CHECKING, Any, cast

from drytorch import Test, Trainer, Validation

from src.config import ExperimentConfig, Experiment, get_trackers, hydra_main
from src.config.environment import DEBUG_MODE
from src.config.events import ConfusionMatrixEvent
from src.data.dataset import get_databuilder, get_datasets
from src.data.derived_dataset import EncodedDataset
from src.module import get_autoencoder, get_classifier
from src.train import (
    ClassifierEMAModel,
    ClassifierModel,
    get_classification_loss,
    get_encoded_loaders,
    get_learning_schema,
    get_loaders,
)
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.train.models import AutoencoderEMAModel
from src.train.objectives import AggregatedAccuracy
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def load_autoencoder() -> AutoencoderEMAModel:
    """Load the trained autoencoder with the weights it was evaluated with."""
    cfg = Experiment.get_config()
    if cfg.autoencoding is None:
        raise ValueError('Autoencoding configuration is missing.')

    ae_cfg = cfg.autoencoding
    autoencoder = get_autoencoder()
    model = AutoencoderEMAModel(
        autoencoder,
        decay=ae_cfg.model.ema_decay,
        device=cfg.user.device,
        name=ae_cfg.name,
        torch_compile=cfg.user.torch_compile,
    )
    model.load_state(cfg.user.load or -1)
    return model


def train_classification(
    autoencoder: AutoencoderEMAModel | None = None,
    trial: Trial | None = None,
) -> None:
    """Set up model, data, and trainer then run training for classification.

    Args:
        autoencoder: optional pretrained autoencoder for encoded representations.
        trial: optional Optuna trial for pruning.
    """
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    class_cfg = cfg.classification
    classifier = get_classifier()
    if trial is None:
        model = ClassifierEMAModel(
            classifier,
            decay=class_cfg.model.ema_decay,
            device=user_cfg.device,
            name=class_cfg.name,
            torch_compile=user_cfg.torch_compile,
        )
    else:
        model = ClassifierModel(
            classifier,
            device=user_cfg.device,
            name=class_cfg.name,
            torch_compile=user_cfg.torch_compile,
        )

    train_dataset, eval_dataset = get_datasets()
    if autoencoder is not None:
        encoded_train = EncodedDataset(train_dataset, autoencoder)
        encoded_eval = EncodedDataset(eval_dataset, autoencoder)
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

    n_classes = get_databuilder().n_classes
    loss = get_classification_loss()
    accuracy = AggregatedAccuracy(n_classes)
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
        register_pruning(builder, trial)
    else:
        register_checkpointing(builder, user_cfg.checkpoint_every or 1)

    trainer = builder.build()
    if cfg.final:
        evaluation = Test(model, loader=eval_loader, metric=accuracy, name='AccuracyTest')
    else:
        evaluation = Validation(model, loader=eval_loader, metric=accuracy, name='AccuracyValidation')

    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    trainer.train_until(class_cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    evaluation()

    evaluated_accuracy = cast(AggregatedAccuracy, evaluation.objective)
    ConfusionMatrixEvent(
        source_name=evaluation.name,
        model_name=model.name,
        epoch=model.epoch,
        matrix=evaluated_accuracy.confusion_matrix,
    )
    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Initialise the experiment run and launch classifier training."""
    trackers = get_trackers(cfg)
    has_autoencoder = cfg.autoencoding is not None
    resume = True if has_autoencoder else cfg.user.load != 0
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, record=not DEBUG_MODE):
        autoencoder = load_autoencoder() if has_autoencoder else None
        train_classification(autoencoder=autoencoder)

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
