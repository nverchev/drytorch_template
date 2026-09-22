"""Model profiling entry point.

Usage::

    uv run profile.py
"""

import pathlib
import torch

from src.config import AllConfig, Experiment, hydra_main
from src.data.dataset import get_datasets
from src.data.structures import Inputs, Targets
from src.module import get_model
from src.train import get_loss
from src.utils.profiling import profile_backward, profile_forward, save_parameters


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


def run_profiling(cfg: AllConfig) -> None:
    """Set up model, create inputs, and profile forward and backward passes."""
    device = cfg.user.device
    if device is None:
        device = torch.accelerator.current_accelerator(check_available=True) or torch.device('cpu')

    train_split, _ = get_datasets()
    inputs, targets = train_split[0]
    batch_inputs = Inputs(features=inputs.features.unsqueeze(0).to(device))
    batch_targets = Targets(label=targets.label.unsqueeze(0).to(device))

    model = get_model()
    model.to(device)
    profile_dir = _get_profiling_dir()
    save_parameters(model, profile_dir)

    loss_fn = get_loss()

    outputs = profile_forward(model, batch_inputs, device=device, profile_dir=profile_dir)
    loss = loss_fn.calculate(outputs, batch_targets)['CrossEntropy'].sum()
    profile_backward(loss, device=device, profile_dir=profile_dir)
    return


def get_databuilder_classes() -> int:
    from src.data.dataset import get_databuilder

    return get_databuilder().n_classes


def setup_and_profile(cfg: AllConfig) -> None:
    """Initialise experiment run and launch profiling."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, record=False):
        run_profiling(cfg)

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Profiling entry point."""
    setup_and_profile(cfg)
    return


if __name__ == '__main__':
    main()
