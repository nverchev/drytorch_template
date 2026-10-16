"""Clone an experiment under the name that overrides form, to reuse its earlier phases.

The run name includes every override, so a later phase run with new overrides
looks for the earlier phases under a new name. Cloning the experiment that holds
them under that name lets the later phase resume it without training them again.

Usage::

    uv run clone_experiment.py --config-name signal_classifier classification.train.learn.lr=1e-4
    uv run clone_experiment.py --config-name signal_classifier --from main_v2 classification.train.learn.lr=1e-4
    uv run clone_experiment.py --config-name signal_classifier --from other_branch
"""

import argparse

from drytorch.utils import local_ops

from src.config import get_experiment_config
from src.config.hydra import DEFAULT_CONFIG


def main() -> None:
    """Clone an experiment under the name of the configuration with the overrides."""
    parser = argparse.ArgumentParser(description='Clone an experiment under the name that overrides form.')
    parser.add_argument(
        '--config-name', default=DEFAULT_CONFIG, help='entry configuration, as for the training scripts'
    )
    parser.add_argument(
        '--from', dest='from_', metavar='NAME', help='experiment to clone; by default, the current variation'
    )
    parser.add_argument('overrides', nargs='*', help='overrides of the new experiment, as for the training scripts')
    args = parser.parse_args()
    cfg = get_experiment_config(args.config_name, overrides=args.overrides)
    from_name = args.from_
    if from_name is None:
        if cfg.final:
            parser.error('a final experiment needs --from: name the final experiment to clone')

        from_name = get_experiment_config(args.config_name).name

    if from_name.endswith('_final') != cfg.final:
        parser.error(f'{from_name} and {cfg.name} mix a final and a development experiment')

    if from_name == cfg.name:
        parser.error(f'nothing to clone: {from_name} is already the experiment of these settings')

    print(f'Cloning {from_name} as {cfg.name} in {cfg.par_dir}...')
    local_ops.clone_experiment_data(cfg.par_dir, from_name, cfg.name)
    print('Done.')
    return


if __name__ == '__main__':
    main()
