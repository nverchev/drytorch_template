"""Subclasses for the Model and EMAModel classes."""

from drytorch.lib.models import Model, EMAModel

from src.data.structures import Inputs, Outputs


ClassifierModel = Model[Inputs, Outputs]
ClassifierEMAModel = EMAModel[Inputs, Outputs]

__all__ = [
    'ClassifierEMAModel',
    'ClassifierModel',
]
