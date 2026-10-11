"""Hydra integration layer.

This module wires up Hydra's configuration composition with the project's
typed config hierarchy.  Two public helpers are exposed:

* :func:`hydra_main` — decorator that converts the Hydra ``DictConfig`` into a
  fully typed :class:`~src.config.specs.AllConfig` and passes it to the
  decorated entry-point function.

* :func:`get_config_all` — constructs an :class:`~src.config.specs.AllConfig`
  programmatically (without starting a Hydra run), used by the tuning script
  to instantiate fresh trial configs.
"""

import dataclasses
import functools
import pathlib
from typing import Any, Final, cast
from collections.abc import Callable

import hydra
from hydra.conf import HydraConf
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from pydantic import TypeAdapter

from src.config.environment import ConfigPath
from src.config.experiment import get_variation
from src.config.specs import AllConfig, HydraSettings

__all__ = [
    'get_config_all',
    'get_hydra_settings',
    'hydra_main',
]

DEFAULT_CONFIG: Final = 'mlp_classifier'


def hydra_main(func: Callable[[AllConfig], None], /) -> Callable[[], None]:
    """Decorator that starts a Hydra run and passes a typed config to *func*.

    Usage::

        @hydra_main
        def main(cfg: AllConfig) -> None: ...

    The decorator:

    1. Reads ``configs/experiment/<config_name>.yaml`` and composes the full config:
       ``mlp_classifier`` unless ``--config-name`` selects another file.
    2. Adds the Hydra runtime settings and embeds the active overrides into the
       variation, with :func:`~src.config.experiment.get_variation`.
    3. Validates the result into a frozen :class:`AllConfig` with pydantic.

    Args:
        func: Entry-point function that accepts an :class:`AllConfig`.

    Returns:
        A no-argument callable suitable for use as ``__main__``.
    """

    @hydra.main(version_base=None, config_path=str(ConfigPath.EXPERIMENT.absolute()), config_name=DEFAULT_CONFIG)
    @functools.wraps(func)
    def wrapper(dict_cfg: DictConfig) -> None:
        """Convert configuration to the stored object."""
        hydra_dict_cfg = HydraConfig.get()
        settings = get_hydra_settings(hydra_dict_cfg)
        return func(_validate(dict_cfg, settings, list(hydra_dict_cfg.overrides.task)))

    return wrapper.__call__


def get_config_all(config_name: str = DEFAULT_CONFIG, *, overrides: list[str] | None = None) -> AllConfig:
    """Compose and return an :class:`AllConfig` without starting a Hydra run.

    This is the recommended way to construct a config inside the tuning
    objective function, where each trial may use different overrides.

    Args:
        config_name: name of the file in ``configs/experiment`` to start from.
        overrides: List of Hydra override strings, e.g.
            ``['classification.train.learn.lr=0.001']``.

    Returns:
        A fully typed and validated :class:`AllConfig`.
    """
    GlobalHydra.instance().clear()

    with hydra.initialize(version_base=None, config_path=ConfigPath.EXPERIMENT.relative()):
        dict_cfg = hydra.compose(config_name=config_name, overrides=overrides)
        try:
            hydra_settings = get_hydra_settings(HydraConfig.get())
        except ValueError:
            hydra_settings = HydraSettings(job_logging=OmegaConf.create())

        # a running Hydra job, such as a tuning script, has its own configuration name
        hydra_settings = dataclasses.replace(hydra_settings, config_name=config_name)
        return _validate(dict_cfg, hydra_settings, overrides)


def get_hydra_settings(dict_cfg: HydraConf, /) -> HydraSettings:
    """Get subset of hydra settings."""
    return HydraSettings(
        output_dir=pathlib.Path(dict_cfg.runtime.output_dir),
        job_logging=cast(DictConfig, dict_cfg.job_logging),
        config_name=str(dict_cfg.job.config_name),
    )


def _validate(dict_cfg: DictConfig, settings: HydraSettings, overrides: list[str] | None) -> AllConfig:
    # Hydra composes and resolves the files, pydantic converts and checks the values
    values = cast(dict[str, Any], OmegaConf.to_container(dict_cfg, resolve=True))
    values['user']['hydra'] = settings
    if overrides is not None:
        values['variation'], values['tags'] = get_variation(values.get('variation'), overrides)

    return TypeAdapter(AllConfig).validate_python(values, extra='forbid')
