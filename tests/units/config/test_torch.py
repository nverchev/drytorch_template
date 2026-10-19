"""Tests for torch config utilities."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.config.torch import (
    GroupedNorm,
    get_activation_cls,
    get_device,
    get_mixed_precision,
    get_norm_cls,
    get_optim_cls,
)


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

    @pytest.mark.parametrize('device', ['cpu', torch.device('cpu')])
    def test_get_device(self, device: str | torch.device) -> None:
        """Resolve the device by name or as a device."""
        resolved = get_device(device)

        assert resolved == torch.device('cpu')

    def test_get_device_auto(self) -> None:
        """Keep auto, which Model resolves to the accelerator when available."""
        resolved = get_device('auto')

        assert resolved == 'auto'

    def test_get_device_raises(self) -> None:
        """Raise ValueError for a name that is not a device."""
        with pytest.raises(ValueError, match='not auto or the name of a device'):
            get_device('gpu')

    @pytest.mark.parametrize('precision', ['bfloat16', torch.bfloat16])
    def test_get_mixed_precision(self, precision: str | torch.dtype) -> None:
        """Resolve the autocast precision by name or as a dtype."""
        dtype = get_mixed_precision(precision)

        assert dtype is torch.bfloat16

    def test_get_mixed_precision_none(self) -> None:
        """Keep None, which trains in full precision."""
        dtype = get_mixed_precision(None)

        assert dtype is None

    @pytest.mark.parametrize('precision', ['float32', torch.float32, 'half'])
    def test_get_mixed_precision_raises(self, precision: str | torch.dtype) -> None:
        """Raise ValueError for a precision other than float16 and bfloat16."""
        with pytest.raises(ValueError, match='not float16 or bfloat16'):
            get_mixed_precision(precision)

    def test_get_norm_cls_grouped_norm(self) -> None:
        """Resolve custom GroupedNorm class by name."""
        cls = get_norm_cls('GroupedNorm')

        assert cls is GroupedNorm

    def test_get_optim_cls(self) -> None:
        """Resolve AdamW optimizer class by name."""
        cls = get_optim_cls('AdamW')

        assert cls is optim.AdamW
