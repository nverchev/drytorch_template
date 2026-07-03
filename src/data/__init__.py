"""Subpackage for data loading, preprocessing and augmentation."""

from src.data.dataset import get_datasets
from src.data.split import Partitions
from src.data.structures import Inputs, Outputs, Targets
from src.data.synthetic_dataset import SyntheticDataset

__all__ = [
    'Inputs',
    'Outputs',
    'Partitions',
    'SyntheticDataset',
    'Targets',
    'get_datasets',
]
