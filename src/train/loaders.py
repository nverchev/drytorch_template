"""DataLoader factory for training and evaluation."""

from drytorch import DataLoader

from src.data.dataset import get_datasets

__all__ = ['get_loaders']


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
    train_dataset, eval_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, n_workers=n_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, n_workers=n_workers)
    return train_loader, eval_loader
