"""Dataset classes and factory for the training/eval split."""

from src.config.experiment import Experiment
from src.config.options import Datasets
from src.data.databuilder import DataBuilder
from src.data.signal_dataset import SignalDataBuilder
from src.data.split import DataSplit, Partitions
from src.data.synthetic_dataset import SyntheticDataBuilder

__all__ = [
    'get_databuilder',
    'get_splits',
]


def get_databuilder(name: Datasets | None = None, /) -> DataBuilder:
    """Return the dataset builder instance corresponding to dataset name.

    Args:
        name: dataset enum identifier.
    """
    if name is None:
        name = Experiment.get_config().data.dataset.name

    databuilder_dict: dict[Datasets, type[DataBuilder]] = {
        Datasets.SYNTHETIC: SyntheticDataBuilder,
        Datasets.SIGNAL: SignalDataBuilder,
    }
    return databuilder_dict[name]()


def _get_splits() -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    databuilder = get_databuilder()
    train_split = databuilder.get_split(train_partition)
    eval_split = databuilder.get_split(eval_partition)
    return train_split, eval_split


def get_splits() -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    splits: tuple[DataSplit, DataSplit] | None = None
    if cfg.user.n_subprocesses:
        import torch.distributed as dist

        rank = dist.get_rank()
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                splits = _get_splits()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])
    else:
        splits = _get_splits()

    if splits is None:
        raise RuntimeError('Splits could not be created.')

    return splits
