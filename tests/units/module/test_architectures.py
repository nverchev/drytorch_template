"""Tests for module architectures and model factory."""

import torch

from src.config.experiment import Experiment
from src.config.options import Models
from src.module import get_model
from src.module.architectures import MLPClassifier, MLPResClassifier


class TestMLPClassifier:
    """Test MLPClassifier forward pass and output dimensions."""

    def test_mlp_classifier_forward(self, sample_batch, n_classes) -> None:
        """Test MLPClassifier forward pass output shape."""
        inputs, _ = sample_batch
        cfg = Experiment.get_config()
        cfg.classification.model.class_name = Models.MLPClassifier
        model = MLPClassifier()
        out = model(inputs)
        assert out.logits.shape == (8, n_classes)
        return


class TestMLPResClassifier:
    """Test MLPResClassifier forward pass and residual connections."""

    def test_mlp_res_classifier_forward(self, sample_batch, n_classes) -> None:
        """Test MLPResClassifier forward pass output shape."""
        inputs, _ = sample_batch
        cfg = Experiment.get_config()
        cfg.classification.model.class_name = Models.MLPResClassifier
        model = MLPResClassifier()
        out = model(inputs)
        assert out.logits.shape == (8, n_classes)
        return


class TestGetModel:
    """Test model factory function get_model."""

    def test_get_model_instantiates_classifier(self) -> None:
        """Test get_model instantiates correct model subclass."""
        model = get_model()
        assert isinstance(model, torch.nn.Module)
        return
