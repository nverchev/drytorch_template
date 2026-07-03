"""Tests for torch config utilities."""

import torch.nn as nn

from src.config.torch import GroupedNorm, get_activation_cls, get_norm_cls, get_optim_cls


class TestGroupedNorm:
    """Test GroupedNorm subclass of nn.GroupNorm."""

    def test_grouped_norm_calculates_groups(self) -> None:
        """Test automatic calculation of num_groups."""
        norm = GroupedNorm(num_channels=64)
        assert isinstance(norm, nn.GroupNorm)
        assert norm.num_groups == 8
        assert norm.num_channels == 64
        return

    def test_grouped_norm_prime_channels(self) -> None:
        """Test automatic calculation of num_groups for prime channel sizes."""
        norm = GroupedNorm(num_channels=17)
        assert norm.num_groups == 1
        return


class TestClassResolvers:
    """Test activation, normalization, and optimizer class resolvers."""

    def test_get_activation_cls(self) -> None:
        """Test get_activation_cls resolves PyTorch activation."""
        cls = get_activation_cls('ReLU')
        assert cls is nn.ReLU
        return

    def test_get_norm_cls_grouped_norm(self) -> None:
        """Test get_norm_cls resolves custom GroupedNorm."""
        cls = get_norm_cls('GroupedNorm')
        assert cls is GroupedNorm
        return

    def test_get_optim_cls(self) -> None:
        """Test get_optim_cls resolves AdamW."""
        import torch.optim as optim

        cls = get_optim_cls('AdamW')
        assert cls is optim.AdamW
        return
