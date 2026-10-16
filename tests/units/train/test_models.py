"""Tests for model and EMA model wrappers across phases."""

import copy

import torch

from src import module
from src.config import experiment, hydra
from src.data import structures
from src.train.models import (
    AutoencoderEMAModel,
    AutoencoderModel,
    ClassifierEMAModel,
    ClassifierModel,
)


class TestClassifierModels:
    """Test ClassifierModel and ClassifierEMAModel execution with torch.compile."""

    def test_classifier_model_compiled_forward(self, sample_batch, n_classes) -> None:
        """Execute compiled ClassifierModel in train and inference modes."""
        inputs, _ = sample_batch
        user_cfg = experiment.Experiment.get_config().user
        arch = module.get_classifier()
        model = ClassifierModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=user_cfg.torch_compile,
            distribute=False,
        )

        train_out = model(inputs)
        with torch.inference_mode():
            eval_out = model(inputs)
            diff_batch_out = model(structures.Inputs(features=torch.randn(3, 64)))

        assert train_out.logits.shape == (8, n_classes)
        assert torch.isfinite(train_out.logits).all()
        assert eval_out.logits.shape == (8, n_classes)
        assert torch.isfinite(eval_out.logits).all()
        assert diff_batch_out.logits.shape == (3, n_classes)

    def test_classifier_ema_model_compiled_forward(self, sample_batch, n_classes) -> None:
        """Execute compiled ClassifierEMAModel across modes and batch sizes."""
        inputs, _ = sample_batch
        user_cfg = experiment.Experiment.get_config().user
        arch = module.get_classifier()
        model = ClassifierEMAModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=user_cfg.torch_compile,
            distribute=False,
        )

        train_out = model(inputs)
        with torch.inference_mode():
            eval_out = model(inputs)
            diff_batch_out = model(structures.Inputs(features=torch.randn(3, 64)))

        assert train_out.logits.shape == (8, n_classes)
        assert torch.isfinite(train_out.logits).all()
        assert eval_out.logits.shape == (8, n_classes)
        assert torch.isfinite(eval_out.logits).all()
        assert diff_batch_out.logits.shape == (3, n_classes)

    def test_compilation_numerical_consistency(self, sample_batch) -> None:
        """Produce identical numerical results between compiled and uncompiled models."""
        inputs, _ = sample_batch
        arch = module.get_classifier()
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


class TestAutoencoderModels:
    """Test AutoencoderModel and AutoencoderEMAModel execution and encoding."""

    def test_autoencoder_model_encode(self, mocker) -> None:
        """Compute latent encodings matching latent dimension without running decoder."""
        cfg = hydra.get_experiment_config('signal_classifier')
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)
        arch = module.get_autoencoder()
        model = AutoencoderModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=False,
            distribute=False,
        )
        inputs = structures.Inputs(features=torch.randn(4, arch.in_dim, arch.sequence_length))

        codes = model.encode(inputs)

        assert codes.shape == (4, arch.latent_dim)
        assert torch.isfinite(codes).all()

    def test_autoencoder_ema_model_encode_uses_averaged_module(self, mocker) -> None:
        """Compute latent encodings using averaged module parameters."""
        cfg = hydra.get_experiment_config('signal_classifier')
        mocker.patch.object(experiment.Experiment, 'get_config', return_value=cfg)
        arch = module.get_autoencoder()
        model = AutoencoderEMAModel(
            arch,
            device=torch.device('cpu'),
            torch_compile=False,
            distribute=False,
        )
        for param in model.averaged_module.parameters():
            param.data.add_(10.0)

        inputs = structures.Inputs(features=torch.randn(4, arch.in_dim, arch.sequence_length))
        raw_codes = model.module.encode(inputs)
        avg_codes = model.averaged_module.encode(inputs)

        ema_codes = model.encode(inputs)

        assert torch.allclose(ema_codes, avg_codes)
        assert not torch.allclose(ema_codes, raw_codes)
        assert ema_codes.shape == (4, arch.latent_dim)
        assert torch.isfinite(ema_codes).all()
