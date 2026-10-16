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
:class:`Conv1DLayer`
    1D Convolutional layer (``nn.Conv1d``) with the :class:`BaseLayer` API.
:class:`Proj1DLayer`
    1D projection layer (``nn.Conv1d``) initialized to identity with ``eye``.
:class:`BaseResBlock`
    Abstract residual block: a sequence of :class:`BaseLayer` layers added to its input.
:class:`Conv1DResBlock`
    Residual 1D CNN block built from :class:`Conv1DLayer` layers.

Utilities
---------
:func:`reset_parameters_recursive`
    Recursively reinitialise all submodule parameters.
"""

import abc
import functools
import itertools
import warnings

from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar, cast, override, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from src.config.environment import DEBUG_MODE
from src.config.torch import ActClass, NormClass

__all__ = [
    'BaseLayer',
    'BaseResBlock',
    'Conv1DLayer',
    'Conv1DResBlock',
    'HasInplace',
    'HasResetParamPrivate',
    'HasResetParamPublic',
    'Layer',
    'LinearLayer',
    'Proj1DLayer',
    'View',
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


class View(nn.Module):
    """Reshape the input tensor according to the specified shape.

    Attributes:
        shape: Target dimensions for reshaping.
    """

    shape: tuple[int, ...]

    def __init__(self, *shape: int) -> None:
        """Initialize.

        Args:
            *shape: target dimensions for reshaping.
        """
        super().__init__()
        self.shape = shape
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reshaping input to configured target shape.

        Args:
            x: input tensor.

        Returns:
            reshaped tensor.
        """
        return x.view(self.shape)


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """Layer followed by optional activation and grouped normalization.

    Attributes:
        in_dim: Input dimension of the layer.
        out_dim: Output dimension of the layer.
        n_groups_layer: Number of groups in the layer.
        use_bias: Whether the layer uses a bias.
        use_trunc_init: Whether truncated normal initialization is used.
        init_weight_fn: Initialization function for weights.
        init_bias_fn: Initialization function for biases.
    """

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
        """Initialize.

        Args:
            in_dim: input dimension of the layer.
            out_dim: output dimension of the layer.
            act_cls: activation class to apply.
            n_groups_layer: number of groups in the layer.
            norm_cls: normalization class to apply.
            use_trunc_init: whether to use truncated normal initialization.
        """
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
        """Get the wrapped layer.

        Returns:
            wrapped PyTorch module.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer, normalization, and activation.

        Args:
            x: input tensor.

        Returns:
            transformed output tensor.
        """
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        return y

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        """Get the activation function.

        Args:
            act_cls: activation class or None.

        Returns:
            instantiated activation module, or None.
        """
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace):
            act.inplace = True

        return act

    @staticmethod
    def get_init_weight(act: nn.Module | None, use_trunc_init: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization strategy for the weight according to the type of activation.

        Args:
            act: activation module.
            use_trunc_init: whether to use truncated normal initialization.

        Returns:
            callable that initializes a weight tensor.
        """
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
        """Get initialization strategy for the bias.

        Args:
            use_trunc_init: whether to initialize with zeros or identity.

        Returns:
            callable that initializes a bias tensor.
        """
        return torch.nn.init.zeros_ if use_trunc_init else lambda x: x

    @staticmethod
    def get_norm_layer(norm_cls: NormClass | None, out_dim: int) -> nn.Module | None:
        """Get the normalization layer.

        Args:
            norm_cls: normalization class or None.
            out_dim: output dimension of the layer.

        Returns:
            instantiated normalization module, or None.
        """
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


class Conv1DLayer(BaseLayer):
    """1D Convolutional layer block."""

    @override
    def get_module(self) -> nn.Module:
        return nn.Conv1d(
            self.in_dim,
            self.out_dim,
            kernel_size=3,
            padding=1,
            groups=self.n_groups_layer,
            bias=self.use_bias,
        )


class Proj1DLayer(nn.Conv1d):
    """1D convolutional layer that starts as the identity projection, cutting or padding channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        bias: bool = True,
    ) -> None:
        """Initialize.

        Args:
            in_channels: number of channels in the input signal.
            out_channels: number of channels produced by the convolution.
            kernel_size: size of the convolving kernel.
            bias: whether to add a learnable bias to the output.
        """
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        return

    @override
    def reset_parameters(self) -> None:
        """Initialize the weights as the identity and the bias as zeros."""
        nn.init.eye_(self.weight.squeeze(-1))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        return


Layer = TypeVar('Layer', bound=BaseLayer)


class BaseResBlock(nn.Module, Generic[Layer], abc.ABC):
    """A sequence of layers whose output is added to the input of the block.

    The skip connection is the identity, or a projection initialized close to it when the
    first and last dimensions differ. The last layer starts from a truncated Gaussian with a
    small deviation, so the block starts close to its skip connection.

    Attributes:
        dims: the dimensions of the input and of the output of each layer.
        act_cls: the class of the activation.
        norm_cls: the class of the normalization.
        n_groups_layer: the number of groups of each layer.
        layers: the layers, in order.
        skip: the connection carrying the input to the output of the layers.
    """

    dims: Sequence[int]
    act_cls: ActClass | None
    norm_cls: NormClass | None
    n_groups_layer: int
    layers: nn.Sequential
    skip: nn.Module

    def __init__(
        self,
        dims: Sequence[int],
        /,
        *,
        act_cls: ActClass | None = None,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> None:
        """Initialize.

        Args:
            dims: the dimensions of the input and of the output of each layer.
            act_cls: the class of the activation.
            n_groups_layer: the number of groups of each layer.
            norm_cls: the class of the normalization.
        """
        super().__init__()
        self.dims = dims
        self.act_cls = act_cls
        self.norm_cls = norm_cls
        self.n_groups_layer = n_groups_layer
        self.layers = nn.Sequential()
        n_layers = len(dims) - 1
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims), start=1):
            layer = self.get_layer(
                in_dim,
                out_dim,
                act_cls=act_cls,
                n_groups_layer=n_groups_layer,
                norm_cls=norm_cls,
                use_trunc_init=i == n_layers,
            )
            self.layers.append(layer)

        self.skip = nn.Identity() if dims[0] == dims[-1] else self.get_projection(dims[0], dims[-1])
        return

    def forward(self, x: Tensor) -> Tensor:
        """Add the input, carried by the skip connection, to the output of the layers.

        Args:
            x: input tensor.

        Returns:
            output tensor after adding residual skip connection.
        """
        return self.skip(x) + self.layers(x)

    def reset_parameters(self) -> None:
        """(Re)-Initialize the weights and bias of the layers and of the skip connection."""
        for layer in self.layers:
            _reset_parameters_or_warn(layer)

        if not isinstance(self.skip, nn.Identity):
            _reset_parameters_or_warn(self.skip)

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
        use_trunc_init: bool = False,
    ) -> Layer:
        """Get the layer instance.

        Args:
            in_dim: input dimension.
            out_dim: output dimension.
            act_cls: activation class.
            n_groups_layer: number of layer groups.
            norm_cls: normalization class.
            use_trunc_init: whether to use truncated normal initialization.

        Returns:
            instantiated layer block.
        """

    @classmethod
    @abc.abstractmethod
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        """Get the projection carrying the input when the dimensions differ.

        Args:
            in_dim: input dimension.
            out_dim: output dimension.

        Returns:
            projection module.
        """


class Conv1DResBlock(BaseResBlock[Conv1DLayer]):
    """A block of a neural network consisting of a sequence of 1D convolutional layers."""

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
        use_trunc_init: bool = False,
    ) -> Conv1DLayer:
        return Conv1DLayer(
            in_dim,
            out_dim,
            act_cls=act_cls,
            n_groups_layer=n_groups_layer,
            norm_cls=norm_cls,
            use_trunc_init=use_trunc_init,
        )

    @classmethod
    @override
    def get_projection(cls, in_dim: int, out_dim: int) -> Proj1DLayer:
        return Proj1DLayer(in_dim, out_dim)


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
