"""Subpackage for training, evaluation and losses."""

from src.train.learning_schema import get_learning_schema
from src.train.loaders import get_encoded_loader, get_encoded_loaders, get_loaders
from src.train.models import (
    AutoencoderEMAModel,
    AutoencoderModel,
    ClassifierEMAModel,
    ClassifierModel,
)
from src.train.objectives import get_accuracy, get_autoencoder_loss, get_classification_loss

__all__ = [
    'AutoencoderEMAModel',
    'AutoencoderModel',
    'ClassifierEMAModel',
    'ClassifierModel',
    'get_accuracy',
    'get_autoencoder_loss',
    'get_classification_loss',
    'get_encoded_loader',
    'get_encoded_loaders',
    'get_learning_schema',
    'get_loaders',
]
