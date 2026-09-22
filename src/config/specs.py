"""Specification for the configuration files.

Root config hierarchy::

    AllConfig
    ├── classification: ExperimentConfig
    │   ├── name
    │   ├── model: ModelConfig
    │   │   └── architecture: LinearArchitectureConfig
    │   └── train: TrainingConfig
    │       ├── learn: LearningConfig
    │       │   └── scheduler: SchedulerConfig
    │       └── early_stopping: EarlyStoppingConfig
    ├── data: DataConfig
    └── user: UserSettings
        ├── trackers: TrackerList
        ├── path: PathSpecs          ← populated from .env
        └── hydra: HydraSettings     ← populated at runtime

For projects with multiple subexperiments, add one ExperimentConfig per subexperiment directly to AllConfig.
"""

import dataclasses
import pathlib
from functools import cached_property

from typing import Any, Annotated, Self

import torch

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from src.config.environment import EnvSettings, PROJECT_NAME, VERSION, get_current_branch
from src.config.torch import get_activation_cls, get_norm_cls, get_optim_cls, set_seed, DEFAULT_ACT, DEFAULT_NORM
from src.config.options import GradOp, ClipCriterion, Schedulers, Models, Datasets


PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


# ---------------------------------------------------------------------------
# Model configs: keep base class, adapt concrete subclasses
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BaseArchitectureConfig:
    """Base specification for an architecture with activation and normalisation.

    Attributes:
        norm_name: Name of a PyTorch normalisation class, e.g. ``'BatchNorm1d'``.
            Leave empty to use the project default (see :data:`DEFAULT_NORM`).
        act_name: Name of a PyTorch activation class, e.g. ``'ReLU'``.
            Leave empty to use the project default (see :data:`DEFAULT_ACT`).
    """

    norm_name: str = ''
    act_name: str = ''

    def __post_init__(self) -> None:
        """Resolve activation and normalisation classes from their string names."""
        self.act_cls = get_activation_cls(self.act_name) if self.act_name else DEFAULT_ACT
        self.norm_cls = get_norm_cls(self.norm_name) if self.norm_name else DEFAULT_NORM
        return


@dataclass(kw_only=True)
class LinearArchitectureConfig(BaseArchitectureConfig):
    """Feedforward (linear) architecture parameters.

    Attributes:
        hidden_dims: Hidden layer dimensions inserted before the output projection.
        dropout_rate: Dropout probability applied between hidden layers.
    """

    hidden_dims: tuple[StrictlyPositiveInt, ...] = (256, 128)
    dropout_rate: PositiveFloat = 0.0


@dataclass
class ModelConfig:
    """Model class selector and non-architectural settings.

    Attributes:
        class_name: Which architecture class to instantiate.
        architecture: Architecture-specific hyperparameters.
        ema_decay: Exponential moving average decay for the EMA model copy.
    """

    class_name: Models = Models.MLPClassifier
    architecture: LinearArchitectureConfig = dataclasses.field(default_factory=LinearArchitectureConfig)
    ema_decay: float = 0.999


# ---------------------------------------------------------------------------
# Data configs: change attributes as needed
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Dataset identity and class filter.

    Attributes:
        name: Name of the dataset.
        selected_classes: Subset of class names to include (empty uses all).
        in_features: Feature dimension for each sample.
        n_train_samples: Number of training split samples.
        n_val_samples: Number of validation split samples.
        n_test_samples: Number of test split samples.
    """

    name: Datasets = Datasets.Synthetic
    selected_classes: list[str] = dataclasses.field(default_factory=list)
    in_features: StrictlyPositiveInt = 64
    n_train_samples: StrictlyPositiveInt = 1000
    n_val_samples: StrictlyPositiveInt = 200
    n_test_samples: StrictlyPositiveInt = 200


@dataclass
class PreprocessConfig:
    """Preprocessing settings.

    Attributes:
        normalize: Whether to normalize input features to zero mean and unit variance.
    """

    normalize: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation flags.

    Attributes:
        noise: Noise quantity added to features (0.0 means no noise).
    """

    noise: PositiveFloat = 0.0


@dataclass
class DataConfig:
    """Full data configuration.

    Attributes:
        dataset: Dataset selection and class parameters.
        preprocess: Preprocessing settings.
        augmentation: Data augmentation flags.
    """

    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = dataclasses.field(default_factory=PreprocessConfig)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)


# ---------------------------------------------------------------------------
# Training configs — generic, reuse as-is
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Specification for the learning-rate scheduler.

    Attributes:
        function: Which scheduler to use (see :class:`~src.config.options.Schedulers`).
        restart_interval: Epochs between cosine restarts (0 = no restart).
        restart_fraction: Fraction of base LR applied at each restart.
        warmup_steps: Linear warm-up epochs at the start of training.
        settings: Additional keyword arguments forwarded to the scheduler
            constructor (e.g. ``decay_steps``, ``min_decay``).
    """

    function: Schedulers
    restart_interval: PositiveInt
    restart_fraction: PositiveFloat
    warmup_steps: PositiveInt
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer_name: Name of a PyTorch optimiser, e.g. ``'AdamW'``.
        lr: Global learning rate (or per-parameter group base LR).
        scheduler: Learning-rate scheduler configuration.
        grad_op: Optional gradient pre-processing operation applied before
            each optimiser step (see :class:`~src.config.options.GradOp`).
        clip_criterion: Criterion for histogram-based clippers.
        opt_settings: Extra keyword arguments forwarded to the optimiser
            constructor (e.g. ``weight_decay``).
    """

    optimizer_name: str
    lr: PositiveFloat
    scheduler: SchedulerConfig
    grad_op: GradOp | None = None
    clip_criterion: ClipCriterion | None = None
    opt_settings: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve optimizer class from name."""
        self.optimizer_cls = get_optim_cls(self.optimizer_name)
        return

    @model_validator(mode='after')
    def _check_clip_criterion(self) -> Self:
        if self.clip_criterion is None and self.grad_op in (GradOp.HistClipper, GradOp.ParamHistClipper):
            msg = 'Must specify clip_criterion when using HistClipper or ParamHistClipper.'
            raise ValueError(msg)

        return self


@dataclass
class EarlyStoppingConfig:
    """Specification for early stopping.

    Attributes:
        active: Whether to enable early stopping.
        window: Number of recent validation metrics to average.
        patience: Epochs to wait for improvement before stopping.
    """

    active: bool
    window: int = 1
    patience: int = 10


@dataclass
class TrainingConfig:
    """Specification for the training loop.

    Attributes:
        batch_size: Total batch size across all devices.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig
    _n_subprocesses: PositiveInt

    @model_validator(mode='after')
    def _check_batch_divisibility(self) -> Self:
        if self._n_subprocesses and self.batch_size % self._n_subprocesses != 0:
            msg = 'Global batch size {} not divisible by number of devices {}.'
            raise ValueError(msg.format(self.batch_size, self._n_subprocesses))

        return self

    @property
    def batch_size_per_device(self) -> int:
        """Batch size assigned to each individual device."""
        if self._n_subprocesses == 0:
            return self.batch_size

        return self.batch_size // self._n_subprocesses


# ---------------------------------------------------------------------------
# ADAPT: Add an ExperimentConfig subclass when you need per-stage configs
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Specification for a single subexperiment."""

    name: str
    model: ModelConfig
    train: TrainingConfig


# ---------------------------------------------------------------------------
# User / environment settings — generic, reuse as-is
# ---------------------------------------------------------------------------


@dataclass
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        csv: Appends metrics to a CSV file in the experiment directory.
        tensorboard: Streams metrics to a local TensorBoard server.
        wandb: Uploads metrics to Weights & Biases (requires ``track_all`` extra).
        sqlalchemy: Persists metrics in a SQLite database (requires ``track_all`` extra).
    """

    hydra: bool
    csv: bool
    tensorboard: bool
    wandb: bool
    sqlalchemy: bool


@dataclasses.dataclass
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
    """

    output_dir: pathlib.Path = dataclasses.field(default_factory=pathlib.Path, init=False)
    job_logging: Any = dataclasses.field(default=None, init=False)


@dataclass
class PathSpecs:
    """Filesystem path specifications populated from environment variables.

    Attributes:
        data_dir: Root directory holding the datasets, shared by all versions.
        root_exp_dir: Root directory under which all experiment artefacts are stored.

    Note:
        Paths are loaded from the ``.env`` file via :class:`~src.config.environment.EnvSettings`.
    """

    _env = EnvSettings()
    data_dir: pathlib.Path = _env.data_dir
    root_exp_dir: pathlib.Path = _env.root_exp_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """Version-scoped experiment directory: ``<root_exp_dir>/v<VERSION>``."""
        return self.root_exp_dir / f'v{VERSION}'


@dataclass
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        cpu: Force all computations onto CPU (useful for debugging).
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        n_subprocesses: Number of DDP processes. ``0`` disables distributed training.
        trackers: Active experiment-tracking backends.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        checkpoint_every: Save a model checkpoint every *n* epochs.
        torch_compile: Whether to compile the model with torch.compile.
        load: Checkpoint epoch to resume from:``0`` skip; ``-1`` loads last checkpoint.
    """

    cpu: bool
    n_workers: PositiveInt
    n_subprocesses: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    torch_compile: bool = True
    load: int = 0
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return

    @property
    def device(self) -> torch.device | None:
        """Compute device: ``cpu`` when forced, ``None`` to use the default accelerator."""
        return torch.device('cpu') if self.cpu else None


# ---------------------------------------------------------------------------
# Root config — main entry point for the configuration
# ---------------------------------------------------------------------------


@dataclass
class AllConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        classification: Classification subexperiment configuration.
        user: User-specific options and runtime preferences.
        data: Data pre-processing and dataset configuration.
        final: If True, uses validation data for training and test data for evaluation.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    classification: ExperimentConfig
    user: UserSettings
    data: DataConfig
    final: bool
    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in defaults.yaml."
            raise ValueError(msg) from ve

    @property
    def resolved_variation(self) -> str:
        """variation if set, otherwise the current git branch."""
        return self.variation or self.git_branch

    @property
    def name(self) -> str:
        """Full experiment name embedded into tracker runs."""
        suffix = '_final' if self.final else ''
        return self.resolved_variation[:249] + suffix

    @property
    def project(self) -> str:
        """Tracker project name, one per version."""
        return f'{PROJECT_NAME}-{self.version}'
