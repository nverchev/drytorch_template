"""Subclasses for the Model and EMAModel classes."""

from drytorch.lib.models import EMAModel, Model

from src.data.structures import Inputs, Outputs

__all__ = [
    'ClassifierEMAModel',
    'ClassifierModel',
]

ClassifierModel = Model[Inputs, Outputs]
ClassifierEMAModel = EMAModel[Inputs, Outputs]
