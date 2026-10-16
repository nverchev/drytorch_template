"""Abstract base for datasets requiring preprocessing and splitting."""

import abc
import dataclasses
import json
import pathlib
from typing import Any, ClassVar, Final

import torch

from src.config.experiment import Experiment
from src.data.split import DataSplit, Partitions

__all__ = [
    'ConfigCached',
    'DataBuilder',
]


class ConfigCached(type):
    """Metaclass reusing one instance per class and data configuration."""

    _instances: ClassVar[dict[tuple[type, str], Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        key = (cls, cls._get_config_key())
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)

        return cls._instances[key]

    @staticmethod
    def _get_config_key() -> str:
        """Return the data configuration of the current experiment as a key."""
        data_config = Experiment.get_config().data
        return json.dumps(dataclasses.asdict(data_config), sort_keys=True, default=str)


class DataBuilder(metaclass=ConfigCached):
    """Abstract dataset builder: (download →) preprocess → split.

    Attributes:
        n_classes: Number of distinct target classes.
        data_dir: Directory path for raw or cached dataset files.
    """

    n_classes: ClassVar[int]
    data_dir: Final[pathlib.Path]

    def __init__(self) -> None:
        """Initialize."""
        cfg = Experiment.get_config()
        self.data_dir = cfg.user.path.data_dir
        self._download()
        self._preprocess()
        return

    def split(self, partition: Partitions, /) -> DataSplit:
        """Load partition and return a DataSplit.

        Args:
            partition: dataset partition to retrieve.

        Returns:
            split dataset containing partition features and labels.
        """
        features, labels = self._get_partition_data(partition)
        return DataSplit(features, labels)

    get_split = split

    def _download(self) -> None:
        """Override to fetch raw data from a remote source before preprocessing."""
        return

    @abc.abstractmethod
    def _preprocess(self) -> None:
        """Preprocess dataset features prior to splitting."""

    @abc.abstractmethod
    def _get_partition_data(self, partition: Partitions, /) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, labels) tensors for partition."""
