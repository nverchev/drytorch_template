"""Abstract base for datasets requiring preprocessing and splitting."""

import abc
from typing import Any, ClassVar

import torch

from src.config.experiment import Experiment
from src.data.split import DataSplit, Partitions


class Singleton(type):
    """Metaclass that creates at most one instance per class."""

    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class DataBuilder(metaclass=Singleton):
    """Abstract dataset builder: (download →) preprocess → split."""

    n_classes: ClassVar[int]

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.data_dir = cfg.user.path.root_exp_dir / 'data'
        self._download()
        self._preprocess()
        return

    def split(self, partition: Partitions) -> DataSplit:
        """Load partition and return a DataSplit."""
        features, labels = self._get_partition_data(partition)
        return DataSplit(features=features, labels=labels)

    get_split = split

    def _download(self) -> None:
        """Override to fetch raw data from a remote source before preprocessing."""
        return

    @abc.abstractmethod
    def _preprocess(self) -> None:
        """Run offline preprocessing (skip if the processed cache already exists)."""

    @abc.abstractmethod
    def _get_partition_data(self, partition: Partitions) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, labels) tensors for partition."""
