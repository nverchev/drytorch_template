"""Tests for the layer building blocks."""

import pytest
import torch

from src.module.layers import IdentityLinear, LinearLayer, LinearResBlock


class TestLinearResBlock:
    """Test the residual block."""

    @pytest.fixture
    def block(self) -> LinearResBlock:
        """Provide a block changing the dimension from 4 to 3."""
        return LinearResBlock([4, 6, 3])

    def test_the_input_is_added_to_the_output_of_the_layers(self, block: LinearResBlock) -> None:
        """The output is the projected input plus the output of the layers."""
        x = torch.randn(5, 4)

        out = block(x)

        torch.testing.assert_close(out, block.skip(x) + block.layers(x))
        assert out.shape == (5, 3)

    def test_the_projection_starts_as_the_identity(self, block: LinearResBlock) -> None:
        """After a reset, the projection is again the identity cut to the output dimension."""
        skip = block.skip
        assert isinstance(skip, IdentityLinear)
        with torch.no_grad():
            skip.weight.add_(1)

        block.reset_parameters()

        torch.testing.assert_close(skip.weight, torch.eye(3, 4))
        torch.testing.assert_close(skip.bias, torch.zeros(3))

    def test_only_the_last_layer_starts_from_a_truncated_gaussian(self, block: LinearResBlock) -> None:
        """The last layer uses the truncated initialization, the others their own."""
        layers = [layer for layer in block.layers if isinstance(layer, LinearLayer)]

        assert [layer.use_trunc_init for layer in layers] == [False, True]

    def test_equal_dimensions_skip_with_the_identity(self) -> None:
        """Without a change of dimension, the skip connection is the identity."""
        block = LinearResBlock([3, 5, 3])

        assert isinstance(block.skip, torch.nn.Identity)
