"""DataLoader factory for training and evaluation."""

from drytorch import DataLoader

from src.data.dataset import get_datasets


def get_loaders(
    batch_size: int,
    n_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, eval_loader) for the current experiment phase."""
    train_dataset, eval_dataset = get_datasets()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, n_workers=n_workers)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, n_workers=n_workers)
    return train_loader, eval_loader
