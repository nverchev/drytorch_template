"""Neural network layer building blocks.

This module provides reusable, type-safe layer primitives that can be combined
to build any feedforward or attention-based architecture.

Key abstractions
----------------
:class:`BaseLayer`
    Abstract layer wrapper with configurable activation, normalisation and
    weight initialisation.  Subclass this to create new layer types.
:class:`LinearLayer`
    Fully-connected layer (``nn.Linear``) with the :class:`BaseLayer` API.
:class:`PointsConvLayer`
    Point-wise 1-D convolution (``nn.Conv1d``), useful for set-valued inputs
    such as point clouds or token sequences.
:class:`BaseResBlock`
    Abstract residual block over any :class:`BaseLayer` subclass.
:class:`LinearResBlock`
    Residual MLP block built from :class:`LinearLayer` layers.

Utilities
---------
:func:`reset_parameters_recursive`
    Recursively reinitialise all submodule parameters.
"""

import abc
import functools
import itertools
import math
import sys
import warnings

from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar, cast, override, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from src.config.torch import ActClass, NormClass

DEBUG_MODE = any(module in sys.modules for module in ('pydevd', 'debugpy')) or sys.gettrace() is not None

__all__ = [
    'DEBUG_MODE',
    'BaseLayer',
    'BaseResBlock',
    'HasInplace',
    'HasResetParamPrivate',
    'HasResetParamPublic',
    'Layer',
    'LinearLayer',
    'LinearResBlock',
    'ProjectionLayer',
    'View',
    'Weighted',
    'debug_check',
    'reset_parameters_recursive',
]

type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor


@runtime_checkable
class HasInplace(Protocol):
    """Protocol for classes that have an inplace option."""

    inplace: bool


@runtime_checkable
class HasResetParamPublic(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""


@runtime_checkable
class HasResetParamPrivate(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def _reset_parameters(self) -> None:
        """Reset the parameters of the module."""


@runtime_checkable
class Weighted(Protocol):
    """Protocol for classes that have a weight attribute."""

    weight: torch.Tensor
    bias: torch.Tensor | None


class View(nn.Module):
    """A simple module that reshapes the input tensor according to the specified shape."""

    shape: tuple[int, ...]

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.view(self.shape)


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """A class that includes a layer followed by an optional activation and grouped normalization."""

    in_dim: int
    out_dim: int
    n_groups_layer: int
    use_bias: bool
    use_trunc_init: bool
    init_weight_fn: Callable[[torch.Tensor], torch.Tensor]
    init_bias_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups_layer = n_groups_layer
        self.use_bias = True if norm_cls is None else False
        self.module: nn.Module = self.get_module()
        self.norm: nn.Module | None = self.get_norm_layer(norm_cls, out_dim)
        self.act: nn.Module | None = self.get_activation(act_cls)
        self.use_trunc_init = use_trunc_init
        self.init_weight_fn = self.get_init_weight(self.act, use_trunc_init)
        self.init_bias_fn = self.get_init_bias(use_trunc_init)
        self.reset_parameters()
        if DEBUG_MODE:
            warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    def reset_parameters(self) -> None:
        """(Re)-Initialize the weights and bias of the layer."""
        self.init_weight_fn(cast(torch.Tensor, self.module.weight))
        if self.norm is None:
            self.init_bias_fn(cast(torch.Tensor, self.module.bias))
            return

        _reset_parameters_or_warn(self.norm)
        self.init_bias_fn(cast(torch.Tensor, self.norm.bias))
        return

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Get the wrapped layer"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        return y

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        """Get the activation function."""
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace):
            act.inplace = True

        return act

    @staticmethod
    def get_init_weight(act: nn.Module | None, use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization for the weight strategy according to the type of activation."""
        if use_trunc_init:
            return functools.partial(nn.init.trunc_normal_, mean=0.0, std=0.02, a=-0.04, b=0.04)

        if act is None:
            return functools.partial(nn.init.xavier_normal_, gain=1)

        if isinstance(act, nn.Hardtanh):
            return functools.partial(nn.init.xavier_normal_, gain=nn.init.calculate_gain('tanh'))

        if isinstance(act, nn.ReLU):
            return functools.partial(nn.init.kaiming_normal_, nonlinearity='relu')

        if isinstance(act, nn.LeakyReLU):
            return functools.partial(nn.init.kaiming_normal_, a=act.negative_slope, nonlinearity='leaky_relu')

        if isinstance(act, nn.GELU):
            return functools.partial(nn.init.xavier_normal_, gain=1)

        return functools.partial(nn.init.xavier_normal_, gain=1)

    @staticmethod
    def get_init_bias(use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization strategy for the bias according to the type of activation."""
        return torch.nn.init.zeros_ if use_trunc_init else lambda x: x

    @staticmethod
    def get_norm_layer(norm_cls: NormClass | None, out_dim: int) -> nn.Module | None:
        """Get the normalization layer"""
        if norm_cls is None:
            return None

        return norm_cls(out_dim)


class LinearLayer(BaseLayer):
    """Linear layer block."""

    @override
    def get_module(self) -> nn.Module:
        if self.n_groups_layer > 1:
            raise NotImplementedError('nn.Linear has no option for groups')

        return nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)


Layer = TypeVar('Layer', bound=BaseLayer)


class ProjectionLayer(nn.Module):
    """A layer that projects the input to a different dimension."""

    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the projection layer."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        return

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.shape[1] != self.in_dim:
            raise ValueError(f'Input dimension {x.shape[1]} does not match expected dimension {self.in_dim}')

        if self.in_dim == self.out_dim:
            return x

        if self.in_dim > self.out_dim:
            out = torch.cat([t.mean(1, keepdim=True) for t in torch.chunk(x, self.out_dim, dim=1)], dim=1)
        else:
            out = x.repeat_interleave(self.out_dim // self.in_dim + 1, 1)[:, : self.out_dim, ...]

        var_adjust = math.sqrt(self.in_dim / self.out_dim)
        return out * var_adjust


class BaseResBlock(nn.Module, Generic[Layer], abc.ABC):
    """A block of a neural network consisting of a sequence of layers."""

    dims: Sequence[int]
    act_cls: ActClass | None
    norm_cls: NormClass | None
    n_groups_layer: int

    def __init__(
        self,
        dims: Sequence[int],
        /,
        *,
        act_cls: ActClass | None = None,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.act_cls = act_cls
        self.norm_cls = norm_cls
        self.n_groups_layer = n_groups_layer
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        for in_dim, out_dim in itertools.pairwise(dims):
            layer = self.get_layer(in_dim, out_dim, act_cls=act_cls, n_groups_layer=n_groups_layer, norm_cls=norm_cls)
            self.layers.append(layer)
            if in_dim == out_dim:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(self.get_projection(in_dim, out_dim))

        self._adjust_variance_init()
        return

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = self.projections[i](x) + layer(x)

        return x

    def reset_parameters(self) -> None:
        """(Re)-Initialize the weights and bias of the layer."""
        for layer in self.layers:
            _reset_parameters_or_warn(layer)

        self._adjust_variance_init()
        return

    @classmethod
    @abc.abstractmethod
    def get_layer(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> Layer:
        """Get the layer instance."""

    @classmethod
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        """Get a projection layer"""
        return ProjectionLayer(in_dim, out_dim)

    def _adjust_variance_init(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer.module, Weighted):
                    layer.module.weight.mul_(1 / math.sqrt(len(self.dims) - 1))
                    if layer.module.bias is not None:
                        layer.module.bias.mul_(0)

        return


class LinearResBlock(BaseResBlock[LinearLayer]):
    """A block of a neural network consisting of a sequence of linear layers."""

    @classmethod
    @override
    def get_layer(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> LinearLayer:
        return LinearLayer(in_dim, out_dim, act_cls=act_cls, n_groups_layer=n_groups_layer, norm_cls=norm_cls)


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """This function is used for debugging purposes during the training process."""
    if isinstance(tensor_out, tuple):
        tensor = tensor_out[0]
    else:
        tensor = tensor_out

    if not torch.isfinite(tensor).all():
        breakpoint()

    return


def _reset_parameters_or_warn(module: nn.Module) -> None:
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    if isinstance(module, nn.Module):
        warnings.warn(f'Module {module.__class__.__name__} does not implement reset_parameters().', stacklevel=3)

    return


def reset_parameters_recursive(module: nn.Module, /, *, warn: bool = False) -> None:
    """Recursively reset parameters for all submodules.

    Args:
        module: The module to reset parameters for, along with all its submodules
        warn: Whether to warn about modules without reset methods
    """
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    has_params = any(True for _ in module.parameters(recurse=False))
    has_children = any(True for _ in module.children())
    if has_params and not has_children and warn:
        warnings.warn(
            f'Leaf module {module.__class__.__name__} has parameters but does not implement reset_parameters().',
            stacklevel=2,
        )
    for child in module.children():
        reset_parameters_recursive(child, warn=warn)

    return
