"""Tests for learning schema and scheduler composition."""

from drytorch.lib import schedulers
from src.config import options, specs
from src.lib.learning_schema import get_scheduler


class TestLearningSchema:
    """Test scheduler instantiation and composition."""

    def test_get_scheduler_composition(self) -> None:
        """Compose learning rate scheduler with warmup and restarts."""
        config = specs.SchedulerConfig(
            function=options.Schedulers.COSINE,
            settings={'decay_steps': 10},
            restart_interval=5,
            restart_fraction=0.5,
            warmup_steps=3,
        )

        scheduler = get_scheduler(config)

        assert isinstance(scheduler, schedulers.AbstractScheduler)
