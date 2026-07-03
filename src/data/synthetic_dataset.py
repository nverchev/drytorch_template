"""Synthetic dataset implementation."""

from typing import ClassVar

import torch

from src.config.experiment import Experiment
from src.data.databuilder import DataBuilder
from src.data.preprocess import get_preprocessor
from src.data.split import Partitions


class SyntheticDataset(DataBuilder):
    """Synthetic dataset builder generating deterministic non-random samples."""

    n_classes: ClassVar[int] = 10

    def __init__(self) -> None:
        cfg_dataset = Experiment.get_config().data.dataset
        self.in_features = cfg_dataset.in_features
        self.n_train_samples = cfg_dataset.n_train_samples
        self.n_val_samples = cfg_dataset.n_val_samples
        self.n_test_samples = cfg_dataset.n_test_samples
        self.partition_data = self._create_splits()
        super().__init__()
        return

    def _create_splits(self) -> dict[Partitions, tuple[torch.Tensor, torch.Tensor]]:
        class_indices = torch.arange(self.n_classes, dtype=torch.float32).unsqueeze(1)
        feature_indices = torch.arange(1, self.in_features + 1, dtype=torch.float32).unsqueeze(0)
        centers = torch.sin(class_indices * feature_indices)

        partitions_info = [
            (Partitions.train, self.n_train_samples),
            (Partitions.val, self.n_val_samples),
            (Partitions.test, self.n_test_samples),
        ]
        partition_data: dict[Partitions, tuple[torch.Tensor, torch.Tensor]] = {}
        for partition, n_samples in partitions_info:
            labels = torch.arange(n_samples, dtype=torch.long) % self.n_classes
            features = centers[labels]
            partition_data[partition] = (features, labels)

        return partition_data

    def _preprocess(self) -> None:
        preprocessor = get_preprocessor()
        for partition, (features, labels) in self.partition_data.items():
            self.partition_data[partition] = (preprocessor(features), labels)

        return

    def _get_partition_data(self, partition: Partitions) -> tuple[torch.Tensor, torch.Tensor]:
        if partition == Partitions.train_val:
            train_f, train_l = self.partition_data[Partitions.train]
            val_f, val_l = self.partition_data[Partitions.val]
            return torch.cat([train_f, val_f]), torch.cat([train_l, val_l])

        return self.partition_data[partition]
