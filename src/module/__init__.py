"""Subpackage for neural network modules."""

from src.module.architectures import BaseArchitecture, MLPClassifier, MLPResClassifier, get_model

__all__ = [
    'BaseArchitecture',
    'MLPClassifier',
    'MLPResClassifier',
    'get_model',
]
