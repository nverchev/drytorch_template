"""Tests for torch config utilities."""

import torch.nn as nn
import torch.optim as optim

from src.config.torch import GroupedNorm, get_activation_cls, get_norm_cls, get_optim_cls


class TestGroupedNorm:
    """Test GroupedNorm subclass of nn.GroupNorm."""

    def test_grouped_norm_calculates_groups(self) -> None:
        """Calculate num_groups automatically for composite channel sizes."""
        norm = GroupedNorm(num_channels=64)

        assert isinstance(norm, nn.GroupNorm)
        assert norm.num_groups == 8
        assert norm.num_channels == 64

    def test_grouped_norm_prime_channels(self) -> None:
        """Calculate single group for prime channel sizes."""
        norm = GroupedNorm(num_channels=17)

        assert norm.num_groups == 1


class TestClassResolvers:
    """Test activation, normalization, and optimizer class resolvers."""

    def test_get_activation_cls(self) -> None:
        """Resolve PyTorch activation class by name."""
        cls = get_activation_cls('ReLU')

        assert cls is nn.ReLU

    def test_get_norm_cls_grouped_norm(self) -> None:
        """Resolve custom GroupedNorm class by name."""
        cls = get_norm_cls('GroupedNorm')

        assert cls is GroupedNorm

    def test_get_optim_cls(self) -> None:
        """Resolve AdamW optimizer class by name."""
        cls = get_optim_cls('AdamW')

        assert cls is optim.AdamW
