"""Autoencoder training entry point.

Usage::

    uv run train_autoencoder.py --config-name signal_classifier
"""

from typing import TYPE_CHECKING, Any

from drytorch import Test, Trainer, Validation

from src.config import ExperimentConfig, Experiment, hydra_main
from src.trackers import get_trackers, save_learning_curves
from src.config.environment import DEBUG_MODE
from src.module import get_autoencoder
from src.train import get_autoencoder_loss, get_learning_schema, get_loaders
from src.train.hooks import register_checkpointing, register_early_stopping, register_pruning
from src.train.models import AutoencoderEMAModel, AutoencoderModel
from src.utils.parallel import DistributedWorker

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_autoencoder(trial: Trial | None = None) -> None:
    """Set up autoencoder model, data, and trainer then run representation training.

    Args:
        trial: optional Optuna trial for pruning.
    """
    cfg = Experiment.get_config()
    user_cfg = cfg.user
    if cfg.autoencoding is None:
        raise ValueError('Autoencoding configuration is missing.')

    ae_cfg = cfg.autoencoding
    autoencoder = get_autoencoder()
    if trial is None:
        model = AutoencoderEMAModel(
            autoencoder,
            decay=ae_cfg.model.ema_decay,
            device=user_cfg.device,
            name=ae_cfg.name,
            mixed_precision=ae_cfg.train.mixed_precision,
            torch_compile=user_cfg.torch_compile,
        )
    else:
        model = AutoencoderModel(
            autoencoder,
            device=user_cfg.device,
            name=ae_cfg.name,
            mixed_precision=ae_cfg.train.mixed_precision,
            torch_compile=user_cfg.torch_compile,
        )

    train_loader, eval_loader = get_loaders(batch_size=ae_cfg.train.batch_size_per_device, n_workers=user_cfg.n_workers)
    loss = get_autoencoder_loss()
    schema = get_learning_schema(ae_cfg)
    builder = Trainer.builder(
        model,
        learning_schema=schema,
        loader=train_loader,
        loss=loss,
    )
    if not cfg.final:
        builder.add_validation(eval_loader)

    if not cfg.final and ae_cfg.train.early_stopping.active:
        cfg_early = ae_cfg.train.early_stopping
        register_early_stopping(builder, window=cfg_early.window, patience=cfg_early.patience)

    if trial is not None:
        register_pruning(builder, trial)
    else:
        register_checkpointing(builder, user_cfg.checkpoint_every or 1)

    trainer = builder.build()
    if cfg.final:
        evaluation = Test(model, loader=eval_loader, metric=loss, name='MSETest')
    else:
        evaluation = Validation(model, loader=eval_loader, metric=loss, name='MSEValidation')

    if user_cfg.load:
        trainer.load_checkpoint(user_cfg.load)

    trainer.train_until(ae_cfg.train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    evaluation()
    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Initialise the experiment run and launch autoencoder training."""
    trackers = get_trackers(cfg)
    resume = cfg.user.load != 0 or cfg.user.run_id is not None
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_autoencoder()
        if cfg.autoencoding is not None:
            save_learning_curves(trackers, cfg.autoencoding.name, show=cfg.user.plot.show)

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
