"""1D synthetic signal dataset implementation."""

from typing import ClassVar, Final, override

import torch

from src.config.experiment import Experiment
from src.data.databuilder import DataBuilder
from src.data.preprocess import Preprocessor, get_preprocessor
from src.data.split import Partitions

__all__ = ['SignalDataset']


class SignalDataset(DataBuilder):
    """Synthetic 1D signal dataset builder with frequency-modulated waveforms.

    Attributes:
        n_classes: Number of distinct target classes.
        final: Whether the final evaluation protocol is active.
        in_features: Number of signal channels or features.
        sequence_length: Length of each signal sequence.
        preprocessor: Feature scaling preprocessor.
        n_train_samples: Number of training samples.
        n_val_samples: Number of validation samples.
        n_test_samples: Number of test samples.
        partition_data: Mapping from partition to feature and label tensors.
    """

    n_classes: ClassVar[int] = 4
    final: Final[bool]
    in_features: Final[int]
    sequence_length: Final[int]
    preprocessor: Preprocessor
    n_train_samples: Final[int]
    n_val_samples: Final[int]
    n_test_samples: Final[int]
    partition_data: Final[dict[Partitions, tuple[torch.Tensor, torch.Tensor]]]

    def __init__(self) -> None:
        """Initialize."""
        cfg = Experiment.get_config()
        cfg_dataset = cfg.data.dataset
        self.final = cfg.final
        self.in_features = cfg_dataset.in_features
        self.sequence_length = cfg_dataset.sequence_length
        self.n_train_samples = cfg_dataset.n_train_samples
        self.n_val_samples = cfg_dataset.n_val_samples
        self.n_test_samples = cfg_dataset.n_test_samples
        self.partition_data = self._create_splits()
        super().__init__()
        return

    def _create_splits(self) -> dict[Partitions, tuple[torch.Tensor, torch.Tensor]]:
        t = torch.linspace(0.0, 1.0, self.sequence_length)
        frequencies = [2.0, 5.0, 9.0, 14.0]
        base_signals = torch.stack([torch.sin(2.0 * torch.pi * f * t) for f in frequencies])
        base_signals = base_signals.unsqueeze(1).repeat(1, self.in_features, 1)
        partitions_info = [
            (Partitions.train, self.n_train_samples),
            (Partitions.val, self.n_val_samples),
            (Partitions.test, self.n_test_samples),
        ]
        partition_data: dict[Partitions, tuple[torch.Tensor, torch.Tensor]] = {}
        for partition, n_samples in partitions_info:
            labels = torch.arange(n_samples, dtype=torch.long) % self.n_classes
            features = base_signals[labels]
            partition_data[partition] = (features, labels)

        return partition_data

    @override
    def _preprocess(self) -> None:
        preprocessor = get_preprocessor()
        train_partition = Partitions.train_val if self.final else Partitions.train
        train_features, _ = self._get_partition_data(train_partition)
        preprocessor.fit(train_features)
        for partition, (features, labels) in self.partition_data.items():
            self.partition_data[partition] = (preprocessor(features), labels)

        self.preprocessor = preprocessor
        return

    @override
    def _get_partition_data(self, partition: Partitions, /) -> tuple[torch.Tensor, torch.Tensor]:
        if partition == Partitions.train_val:
            train_f, train_l = self.partition_data[Partitions.train]
            val_f, val_l = self.partition_data[Partitions.val]
            return torch.cat([train_f, val_f]), torch.cat([train_l, val_l])

        return self.partition_data[partition]
