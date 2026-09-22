"""Tests for ClassifierModel and ClassifierEMAModel compilation and execution."""

import copy

import torch

from src.config.experiment import Experiment
from src.data.structures import Inputs
from src.module import get_model
from src.train.models import ClassifierEMAModel, ClassifierModel


class TestClassifierModels:
    """Test ClassifierModel and ClassifierEMAModel execution with torch.compile."""

    def test_classifier_model_compiled_forward(self, sample_batch, n_classes) -> None:
        """Test forward pass of compiled ClassifierModel in train and inference modes."""
        inputs, _ = sample_batch
        user_cfg = Experiment.get_config().user
        arch = get_model()
        model = ClassifierModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=user_cfg.torch_compile,
            distribute=False,
        )
        train_out = model(inputs)
        assert train_out.logits.shape == (8, n_classes)
        assert torch.isfinite(train_out.logits).all()
        with torch.inference_mode():
            eval_out = model(inputs)
            assert eval_out.logits.shape == (8, n_classes)
            assert torch.isfinite(eval_out.logits).all()
            diff_batch_out = model(Inputs(features=torch.randn(3, 64)))
            assert diff_batch_out.logits.shape == (3, n_classes)

        return

    def test_classifier_ema_model_compiled_forward(self, sample_batch, n_classes) -> None:
        """Test forward pass of compiled ClassifierEMAModel across modes and batch sizes."""
        inputs, _ = sample_batch
        user_cfg = Experiment.get_config().user
        arch = get_model()
        model = ClassifierEMAModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=user_cfg.torch_compile,
            distribute=False,
        )
        train_out = model(inputs)
        assert train_out.logits.shape == (8, n_classes)
        assert torch.isfinite(train_out.logits).all()
        with torch.inference_mode():
            eval_out = model(inputs)
            assert eval_out.logits.shape == (8, n_classes)
            assert torch.isfinite(eval_out.logits).all()
            diff_batch_out = model(Inputs(features=torch.randn(3, 64)))
            assert diff_batch_out.logits.shape == (3, n_classes)

        return

    def test_compilation_numerical_consistency(self, sample_batch) -> None:
        """Test that compiled and uncompiled models produce identical results."""
        inputs, _ = sample_batch
        arch = get_model()
        arch_copy = copy.deepcopy(arch)
        uncompiled = ClassifierModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=False,
            distribute=False,
        )
        compiled = ClassifierModel(
            arch_copy,
            device=torch.device('cpu'),
            torch_compile=True,
            distribute=False,
        )
        with torch.inference_mode():
            uncompiled_out = uncompiled(inputs)
            compiled_out = compiled(inputs)
            assert torch.allclose(uncompiled_out.logits, compiled_out.logits, atol=1e-5)

        return
