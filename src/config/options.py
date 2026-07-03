"""Defines all enumerated options used in configuration files.

When adapting this template to a new project, add new enum members to describe
the concrete choices available (e.g. additional model architectures, datasets).
The string values are used as YAML keys so prefer lowercase underscore names.
"""

import enum


# ---------------------------------------------------------------------------
# Project Specific: Add choices for your project
# ---------------------------------------------------------------------------
class Datasets(enum.StrEnum):
    """Dataset choices."""

    Synthetic = enum.auto()


class Models(enum.StrEnum):
    """Model architectures."""

    MLPClassifier = enum.auto()
    MLPResClassifier = enum.auto()


# ---------------------------------------------------------------------------
# Generic options available in drytorch
# ---------------------------------------------------------------------------


class GradOp(enum.StrEnum):
    """Gradient operation names."""

    GradParamNormalizer = enum.auto()
    GradZScoreNormalizer = enum.auto()
    GradNormClipper = enum.auto()
    GradValueClipper = enum.auto()
    HistClipper = enum.auto()
    ParamHistClipper = enum.auto()


class ClipCriterion(enum.StrEnum):
    """Clipping criterion names for histogram-based gradient clippers."""

    ZStat = enum.auto()
    EMA = enum.auto()


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    Constant = enum.auto()
    Cosine = enum.auto()
    Exponential = enum.auto()
    Polynomial = enum.auto()
    Step = enum.auto()
