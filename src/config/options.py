"""Defines all enumerated options used in configuration files."""

import enum

__all__ = [
    'Autoencoders',
    'Classifiers',
    'ClipCriterion',
    'Datasets',
    'GradOp',
    'Schedulers',
]


# ---------------------------------------------------------------------------
# Project Specific: Add choices for your project
# ---------------------------------------------------------------------------
class Datasets(enum.StrEnum):
    """Dataset choices."""

    SYNTHETIC = enum.auto()
    SIGNAL = enum.auto()


class Classifiers(enum.StrEnum):
    """Classifier architectures."""

    MLP_CLASSIFIER = enum.auto()
    LINEAR_CLASSIFIER = enum.auto()


class Autoencoders(enum.StrEnum):
    """Autoencoder architectures."""

    CNN_RES_AUTOENCODER = enum.auto()


# ---------------------------------------------------------------------------
# Generic options available in drytorch
# ---------------------------------------------------------------------------


class GradOp(enum.StrEnum):
    """Gradient operation names."""

    GRAD_PARAM_NORMALIZER = enum.auto()
    GRAD_Z_SCORE_NORMALIZER = enum.auto()
    GRAD_NORM_CLIPPER = enum.auto()
    GRAD_VALUE_CLIPPER = enum.auto()
    HIST_CLIPPER = enum.auto()
    PARAM_HIST_CLIPPER = enum.auto()


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names for histogram-based gradient clippers."""

    Z_STAT = enum.auto()
    EMA = enum.auto()


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    CONSTANT = enum.auto()
    COSINE = enum.auto()
    EXPONENTIAL = enum.auto()
    POLYNOMIAL = enum.auto()
    STEP = enum.auto()
