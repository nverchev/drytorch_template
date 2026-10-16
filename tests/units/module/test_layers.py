"""Tests for the layer building blocks."""

import pytest
import torch

from src.module.layers import Conv1DLayer, Conv1DResBlock, Proj1DLayer


class TestProj1DLayer:
    """Test the 1D projection layer."""

    def test_forward_acts_as_identity_projection(self) -> None:
        """The projection preserves input channels up to min(in_channels, out_channels)."""
        layer_down = Proj1DLayer(4, 3)
        layer_up = Proj1DLayer(3, 5)
        x = torch.randn(2, 4, 10)

        out_down = layer_down(x)
        out_up = layer_up(x[:, :3, :])

        torch.testing.assert_close(out_down, x[:, :3, :])
        torch.testing.assert_close(out_up[:, :3, :], x[:, :3, :])
        torch.testing.assert_close(out_up[:, 3:, :], torch.zeros(2, 2, 10))

    def test_reset_parameters_initializes_identity_and_zero_bias(self) -> None:
        """Parameters reset to eye on the weight matrix and zeros on the bias."""
        layer = Proj1DLayer(4, 3)
        with torch.no_grad():
            layer.weight.add_(1.0)
            layer.bias.add_(1.0)

        layer.reset_parameters()

        expected_weight = torch.zeros(3, 4, 1)
        expected_weight[:3, :3, 0] = torch.eye(3)
        torch.testing.assert_close(layer.weight, expected_weight)
        torch.testing.assert_close(layer.bias, torch.zeros(3))


class TestConv1DResBlock:
    """Test the residual block."""

    @pytest.fixture
    def block(self) -> Conv1DResBlock:
        """Provide a block changing channel dimensions from 4 to 3."""
        return Conv1DResBlock([4, 6, 3])

    def test_the_input_is_added_to_the_output_of_the_layers(self, block: Conv1DResBlock) -> None:
        """The output is the projected input plus the output of the layers."""
        x = torch.randn(5, 4, 16)

        out = block(x)

        torch.testing.assert_close(out, block.skip(x) + block.layers(x))
        assert out.shape == (5, 3, 16)

    def test_the_projection_starts_as_the_identity(self, block: Conv1DResBlock) -> None:
        """After a reset, the projection is again the identity cut to the output dimension."""
        skip = block.skip
        assert isinstance(skip, Proj1DLayer)
        with torch.no_grad():
            skip.weight.add_(1.0)

        block.reset_parameters()

        expected_weight = torch.zeros(3, 4, 1)
        expected_weight[:3, :3, 0] = torch.eye(3)
        torch.testing.assert_close(skip.weight, expected_weight)
        torch.testing.assert_close(skip.bias, torch.zeros(3))

    def test_only_the_last_layer_starts_from_a_truncated_gaussian(self, block: Conv1DResBlock) -> None:
        """The last layer uses the truncated initialization, the others their own."""
        layers = [layer for layer in block.layers if isinstance(layer, Conv1DLayer)]

        assert [layer.use_trunc_init for layer in layers] == [False, True]

    def test_equal_dimensions_skip_with_the_identity(self) -> None:
        """Without a change of dimension, the skip connection is the identity."""
        block = Conv1DResBlock([3, 5, 3])

        assert isinstance(block.skip, torch.nn.Identity)
