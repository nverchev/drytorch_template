"""Tests for the classifiers module."""

import pytest
import torch

from src.data import structures
from src.module.classifiers import LinearClassifier, MLPClassifier, get_classifier


class TestMLPClassifier:
    """Test the MLP classifier."""

    @pytest.fixture
    def classifier(self) -> MLPClassifier:
        """Return the classifier of the default configuration."""
        return MLPClassifier()

    def test_forward_returns_logits(
        self,
        classifier: MLPClassifier,
        sample_batch: tuple[structures.Inputs, structures.Targets],
        n_classes: int,
    ) -> None:
        """Return a logit for each class of each sample."""
        inputs, _ = sample_batch

        out = classifier(inputs)

        assert out.shape == (len(inputs.features), n_classes)


@pytest.mark.usefixtures('signal_classifier_config')
class TestLinearClassifier:
    """Test the linear classifier on the encodings."""

    @pytest.fixture
    def classifier(self) -> LinearClassifier:
        """Return the classifier of the configuration with an autoencoding phase."""
        return LinearClassifier()

    def test_forward_returns_logits(self, classifier: LinearClassifier) -> None:
        """Return a logit for each class of each encoding."""
        inputs = structures.Inputs(features=torch.randn(8, classifier.in_dim))

        out = classifier(inputs)

        assert out.shape == (8, classifier.n_classes)


def test_get_classifier() -> None:
    """Build the configured classifier."""
    classifier = get_classifier()

    assert isinstance(classifier, MLPClassifier)
