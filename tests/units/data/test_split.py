"""Tests for dataset split and runtime augmentation."""

import torch
from pytest_mock import MockerFixture

from src.data.split import DataSplit


def _offset_transform(features: torch.Tensor) -> torch.Tensor:
    return features + 100.0


class TestDataSplit:
    """Test DataSplit sample retrieval and conditional augmentation."""

    def test_augmentation_only_outside_inference_mode(self, mocker: MockerFixture) -> None:
        """Test that DataSplit augments features only when inference mode is off."""
        mocker.patch('src.data.split.get_augmenter', return_value=_offset_transform)
        stored_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        stored_labels = torch.tensor([0, 1])
        data_split = DataSplit(stored_features, stored_labels)
        inputs_train, _ = data_split[0]
        with torch.inference_mode():
            inputs_eval, _ = data_split[0]

        assert torch.equal(inputs_eval.features, stored_features[0])
        assert torch.equal(inputs_train.features, _offset_transform(stored_features[0]))
        return
