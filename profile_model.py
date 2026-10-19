"""Model profiling entry point.

Usage::

    uv run profile.py
"""

import pathlib
import torch

from src.config import ExperimentConfig, Experiment, hydra_main
from src.data.dataset import get_splits
from src.data.structures import Inputs, Targets
from src.module import get_classifier
from src.train import get_classification_loss
from src.utils.profiling import profile_backward, profile_forward, log_parameter_counts


def run_profiling(cfg: ExperimentConfig) -> None:
    """Set up model, create inputs, and profile forward and backward passes."""
    device = cfg.user.device
    if device is None:
        device = torch.accelerator.current_accelerator(check_available=True) or torch.device('cpu')

    train_split, _ = get_splits()
    inputs, targets = train_split[0]
    batch_inputs = Inputs(features=inputs.features.unsqueeze(0).to(device))
    label = targets.label.unsqueeze(0).to(device)
    batch_targets = Targets(label=label, features=targets.features.unsqueeze(0).to(device))

    classifier = get_classifier()
    classifier.to(device)
    profile_dir = _get_profiling_dir()
    log_parameter_counts(classifier, profile_dir)

    loss_fn = get_classification_loss()

    outputs = profile_forward(classifier, batch_inputs, device=device, profile_dir=profile_dir)
    loss = loss_fn.calculate(outputs, batch_targets)['CrossEntropy'].sum()
    profile_backward(loss, device=device, profile_dir=profile_dir)
    return


def get_databuilder_classes() -> int:
    from src.data.dataset import get_databuilder

    return get_databuilder().n_classes


def setup_and_profile(cfg: ExperimentConfig) -> None:
    """Initialise experiment run and launch profiling."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        run_profiling(cfg)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Profiling entry point."""
    setup_and_profile(cfg)
    return


def _get_profiling_dir() -> pathlib.Path:
    exp = Experiment.get_current()
    if '@' in exp.run.id:
        day, time = exp.run.id.split('@')
        run_subdir = pathlib.Path(day) / time
    else:
        run_subdir = pathlib.Path(exp.run.id)

    profile_dir = exp.par_dir / 'profile' / exp.name / run_subdir
    profile_dir.mkdir(exist_ok=True, parents=True)
    return profile_dir


if __name__ == '__main__':
    main()
