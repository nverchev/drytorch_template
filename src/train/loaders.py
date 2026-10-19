"""DataLoader factory for training and evaluation."""

from drytorch import DataLoader

from src.data.dataset import get_splits
from src.data.derived_dataset import EncodedDataset

__all__ = [
    'get_encoded_loader',
    'get_encoded_loaders',
    'get_loaders',
]


def get_loaders(
    *,
    batch_size: int,
    n_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, eval_loader) for the current experiment phase.

    Args:
        batch_size: number of samples per batch.
        n_workers: number of worker subprocesses.

    Returns:
        A tuple of (train_loader, eval_loader).
    """
    train_split, eval_split = get_splits()
    train_loader = DataLoader(train_split, batch_size=batch_size, n_workers=n_workers)
    eval_loader = DataLoader(eval_split, batch_size=batch_size, n_workers=n_workers)
    return train_loader, eval_loader


def get_encoded_loader(
    split: EncodedDataset,
    *,
    batch_size: int,
) -> DataLoader:
    """Return a DataLoader for an encoded dataset partition.

    The codes come out of the autoencoder already on the device, so they
    are not pinned and not loaded in another process.

    Args:
        split: encoded dataset partition.
        batch_size: number of samples per batch.

    Returns:
        DataLoader drawing batches without memory pinning or worker processes.
    """
    return DataLoader(split, batch_size=batch_size, n_workers=0, pin_memory=False)


def get_encoded_loaders(
    encoded_train: EncodedDataset,
    encoded_eval: EncodedDataset,
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, eval_loader) for encoded dataset representations.

    Args:
        encoded_train: encoded training split.
        encoded_eval: encoded evaluation split.
        batch_size: number of samples per batch.

    Returns:
        A tuple of (train_loader, eval_loader).
    """
    train_loader = get_encoded_loader(encoded_train, batch_size=batch_size)
    eval_loader = get_encoded_loader(encoded_eval, batch_size=batch_size)
    return train_loader, eval_loader
