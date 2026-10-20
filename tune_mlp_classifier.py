"""Hyperparameter tuning entry point using Optuna.

Usage::

    uv run tune_mlp_classifier.py
    uv run tune_mlp_classifier.py tune=learn
"""

import pathlib
import warnings

import hydra
import optuna
import torch

from collections.abc import Callable
from omegaconf import DictConfig

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError

from src.config import ConfigPath, Experiment, get_experiment_config
from src.trackers import set_tuning_logging
from src.utils.tuning import impute_failed_trial, get_study_name, log_study_settings

from train_classifier import train_classification


def set_objective(tune_cfg: DictConfig) -> Callable[[optuna.Trial], float]:
    """Set the objective function following the study configuration."""

    def objective(trial: optuna.Trial) -> float:
        """Set up the experiment and launch the training cycle."""
        overrides = suggest_overrides(tune_cfg, trial)
        seed = 'null' if tune_cfg.seed is None else tune_cfg.seed
        overrides += [f'classification.train.n_epochs={tune_cfg.n_epochs}', f'user.seed={seed}']
        if tune_cfg.mixed_precision is not None:
            overrides.append(f'classification.train.mixed_precision={tune_cfg.mixed_precision}')

        trial_cfg = get_experiment_config(tune_cfg.config_name, overrides=overrides)

        exp = Experiment(trial_cfg, name=trial_cfg.name, par_dir=trial_cfg.par_dir, tags=trial_cfg.tags)
        with exp.create_run(record=False):
            try:
                train_classification(trial=trial)
            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNING.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig) -> None:
    """Set up the study and launch optimization."""
    base_cfg = get_experiment_config(tune_cfg.config_name)
    if base_cfg.autoencoding is not None:
        msg = 'Tuning supports single-phase configurations; projects with phases use one tuning script per phase.'
        raise ValueError(msg)

    set_tuning_logging()
    pathlib.Path(tune_cfg.db_location).mkdir(exist_ok=True)

    if tune_cfg.tune.use_pruner:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=tune_cfg.tune.min_resource,
            max_resource=tune_cfg.tune.max_resource,
            reduction_factor=tune_cfg.tune.reduction_factor,
        )

    else:
        pruner = optuna.pruners.NopPruner()

    with warnings.catch_warnings():
        # multivariate TPE and the constant liar are stable in practice; Optuna still labels them experimental
        warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True)

    study_name = get_study_name(tune_cfg.config_name, tune_cfg.tune.study_name, tune_cfg.overrides)
    study = optuna.create_study(
        study_name=study_name, storage=tune_cfg.storage, sampler=sampler, pruner=pruner, load_if_exists=True
    )
    log_study_settings(study, tune_cfg)
    # the trials of the study, from every worker and every run, stop at n_trials
    max_trials = optuna.study.MaxTrialsCallback(tune_cfg.tune.n_trials, states=None)
    study.optimize(set_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials, callbacks=[max_trials])
    return


if __name__ == '__main__':
    main()
