"""Dataset classes and factory for the training/eval split."""

from src.config.experiment import Experiment
from src.config.options import Datasets
from src.data.databuilder import DataBuilder
from src.data.split import DataSplit, Partitions
from src.data.synthetic_dataset import SyntheticDataset

__all__ = [
    'get_databuilder',
    'get_datasets',
]


def get_databuilder(name: Datasets | None = None, /) -> DataBuilder:
    """Return the dataset builder instance corresponding to dataset name.

    Args:
        name: dataset enum identifier.
    """
    if name is None:
        name = Experiment.get_config().data.dataset.name

    databuilder_dict: dict[Datasets, type[DataBuilder]] = {
        Datasets.SYNTHETIC: SyntheticDataset,
    }
    return databuilder_dict[name]()


def _get_datasets() -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets."""
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    databuilder = get_databuilder()
    train_dataset = databuilder.get_split(train_partition)
    eval_dataset = databuilder.get_split(eval_partition)
    return train_dataset, eval_dataset


def get_datasets() -> tuple[DataSplit, DataSplit]:
    """Get training and evaluation datasets in a multiprocess-safe way."""
    cfg = Experiment.get_config()
    splits: tuple[DataSplit, DataSplit] | None = None
    if cfg.user.n_subprocesses:
        import torch.distributed as dist

        rank = dist.get_rank()
        for i in range(cfg.user.n_subprocesses):
            if rank == i:
                splits = _get_datasets()

            dist.barrier() if cfg.user.cpu else dist.barrier(device_ids=[rank])
    else:
        splits = _get_datasets()

    if splits is None:
        raise RuntimeError('Splits could not be created.')

    return splits
