"""Trackers handling the events of this project.

The trackers for optional dependencies live in their own modules, imported
only when the dependency is installed.
"""

from src.trackers.logging import ClassificationLogger


__all__ = ['ClassificationLogger']
