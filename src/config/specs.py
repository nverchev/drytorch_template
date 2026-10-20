"""Specification for the configuration files.

Root config hierarchy::

    ExperimentConfig
    ├── classification: ClassificationConfig
    │   ├── name
    │   ├── model: ClassifierModelConfig
    │   │   └── architecture: LinearArchitectureConfig
    │   └── train: TrainingConfig
    │       ├── learn: LearningConfig
    │       │   └── scheduler: SchedulerConfig
    │       └── early_stopping: EarlyStoppingConfig
    ├── autoencoding: AutoencodingConfig | None
    │   ├── name
    │   ├── model: AutoencoderModelConfig
    │   │   └── architecture: LinearArchitectureConfig
    │   └── train: TrainingConfig
    │       ├── learn: LearningConfig
    │       │   └── scheduler: SchedulerConfig
    │       └── early_stopping: EarlyStoppingConfig
    ├── data: DataConfig
    └── user: UserSettings
        ├── trackers: TrackerList
        ├── plot: PlottingSettings
        ├── path: PathSpecs          ← populated from .env
        └── hydra: HydraSettings     ← populated at runtime

For projects with multiple phases, add one PhaseConfig per phase directly to ExperimentConfig.
"""

import dataclasses
import pathlib
from functools import cached_property

from typing import Any, Annotated, Self, Literal

import torch
import torch.nn as nn

from pydantic import BeforeValidator, Field, PlainValidator, model_validator
from pydantic.dataclasses import dataclass

from src.config.environment import EnvSettings, PROJECT_NAME, VERSION, get_current_branch
from src.config.torch import (
    ActClass,
    GroupedNorm,
    NormClass,
    get_activation_cls,
    get_device,
    get_mixed_precision,
    get_norm_cls,
    get_optim_cls,
    set_seed,
)
from src.config.options import Autoencoders, Classifiers, ClipCriterion, Datasets, GradOp, Schedulers


ActivationClass = Annotated[ActClass, BeforeValidator(get_activation_cls)]
Device = Annotated[torch.device | Literal['auto'], PlainValidator(get_device)]
NormalizationClass = Annotated[NormClass, BeforeValidator(get_norm_cls)]
MixedPrecision = Annotated[torch.dtype | None, PlainValidator(get_mixed_precision)]
OptimizerClass = Annotated[type[torch.optim.Optimizer], BeforeValidator(get_optim_cls)]
PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0)]


# ---------------------------------------------------------------------------
# Model configs: keep base class, adapt concrete subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BaseArchitectureConfig:
    """Base specification for an architecture with activation and normalisation.

    Attributes:
        norm_cls: Normalization of the layers, written by its name, e.g. ``BatchNorm1d``.
        act_cls: Activation of the layers, written by its name, e.g. ``ReLU``.
    """

    norm_cls: NormalizationClass = GroupedNorm
    act_cls: ActivationClass = nn.ELU


@dataclass(frozen=True, kw_only=True)
class LinearArchitectureConfig(BaseArchitectureConfig):
    """Feedforward (linear) architecture parameters.

    Attributes:
        hidden_dims: Hidden layer dimensions inserted before the output projection.
        dropout_rate: Dropout probability applied between hidden layers.
    """

    hidden_dims: tuple[StrictlyPositiveInt, ...] = (256, 128)
    dropout_rate: PositiveFloat = 0.0
    latent_dim: StrictlyPositiveInt = 32


@dataclass(frozen=True)
class ClassifierModelConfig:
    """Classifier model class selector and non-architectural settings.

    Attributes:
        class_name: Which classifier architecture class to instantiate.
        architecture: Architecture-specific hyperparameters.
        ema_decay: Exponential moving average decay for the EMA model copy.
    """

    class_name: Classifiers = Classifiers.MLP_CLASSIFIER
    architecture: LinearArchitectureConfig = dataclasses.field(default_factory=LinearArchitectureConfig)
    ema_decay: float = 0.999


@dataclass(frozen=True)
class AutoencoderModelConfig:
    """Autoencoder model class selector and non-architectural settings.

    Attributes:
        class_name: Which autoencoder architecture class to instantiate.
        architecture: Architecture-specific hyperparameters.
        ema_decay: Exponential moving average decay for the EMA model copy.
    """

    class_name: Autoencoders = Autoencoders.CNN_RES_AUTOENCODER
    architecture: LinearArchitectureConfig = dataclasses.field(default_factory=LinearArchitectureConfig)
    ema_decay: float = 0.999


# ---------------------------------------------------------------------------
# Data configs: change attributes as needed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset identity and class filter.

    Attributes:
        name: Name of the dataset.
        selected_classes: Subset of class names to include (empty uses all).
        in_features: Feature dimension for each sample.
        sequence_length: Length of sequence or signal samples.
        n_train_samples: Number of training split samples.
        n_val_samples: Number of validation split samples.
        n_test_samples: Number of test split samples.
    """

    name: Datasets = Datasets.SYNTHETIC
    selected_classes: list[str] = dataclasses.field(default_factory=list)
    in_features: StrictlyPositiveInt = 64
    sequence_length: StrictlyPositiveInt = 128
    n_train_samples: StrictlyPositiveInt = 1000
    n_val_samples: StrictlyPositiveInt = 200
    n_test_samples: StrictlyPositiveInt = 200


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing settings.

    Attributes:
        normalize: Whether to normalize input features to zero mean and unit variance.
    """

    normalize: bool = True


@dataclass(frozen=True)
class AugmentationConfig:
    """Data augmentation flags.

    Attributes:
        noise: Noise quantity added to features (0.0 means no noise).
    """

    noise: PositiveFloat = 0.0


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer: PyTorch optimizer, written by its name, e.g. ``AdamW``.
        lr: Global learning rate (or per-parameter group base LR).
        scheduler: Learning-rate scheduler configuration.
        grad_op: Optional gradient pre-processing operation applied before
            each optimiser step (see :class:`~src.config.options.GradOp`).
        clip_criterion: Criterion for histogram-based clippers.
        opt_settings: Extra keyword arguments forwarded to the optimiser
            constructor (e.g. ``weight_decay``).
    """

    optimizer: OptimizerClass
    lr: PositiveFloat
    scheduler: SchedulerConfig
    grad_op: GradOp | None = None
    clip_criterion: ClipCriterion | None = None
    opt_settings: dict[str, Any] = dataclasses.field(default_factory=dict)

    @model_validator(mode='after')
    def _check_clip_criterion(self) -> Self:
        if self.clip_criterion is None and self.grad_op in (GradOp.HIST_CLIPPER, GradOp.PARAM_HIST_CLIPPER):
            msg = 'Must specify clip_criterion when using HistClipper or ParamHistClipper.'
            raise ValueError(msg)

        return self


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TrainingConfig:
    """Specification for the training loop.

    Attributes:
        batch_size: Total batch size across all devices.
        learn: Optimiser and scheduler configuration.
        n_epochs: Maximum number of training epochs.
        early_stopping: Early stopping configuration.
        mixed_precision: Autocast precision, ``float16`` (with gradient scaling) or ``bfloat16``; empty for none.
    """

    batch_size: StrictlyPositiveInt
    learn: LearningConfig
    n_epochs: StrictlyPositiveInt
    early_stopping: EarlyStoppingConfig
    _n_subprocesses: PositiveInt
    mixed_precision: MixedPrecision = None

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
# ADAPT: Add a PhaseConfig subclass for each phase
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseConfig:
    """Specification for a single phase.

    Attributes:
        name: Name of the phase.
        model: Model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    name: str
    model: ClassifierModelConfig | AutoencoderModelConfig
    train: TrainingConfig


@dataclass(frozen=True)
class ClassificationConfig(PhaseConfig):
    """Specification for the classifier phase.

    Attributes:
        name: Name of the phase.
        model: Classifier model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    model: ClassifierModelConfig


@dataclass(frozen=True)
class AutoencodingConfig(PhaseConfig):
    """Specification for the autoencoder phase.

    Attributes:
        name: Name of the phase.
        model: Autoencoder model and architecture configuration.
        train: Training loop and learning schedule configuration.
    """

    model: AutoencoderModelConfig


# ---------------------------------------------------------------------------
# User / environment settings — generic, reuse as-is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        csv: Appends metrics to a CSV file in the experiment directory.
        tensorboard: Streams metrics to a local TensorBoard server.
        wandb: Uploads metrics to Weights & Biases (requires ``track_all`` extra).
        sqlalchemy: Persists metrics in a SQLite database (requires ``track_all`` extra).
        matplotlib: Saves a PNG figure of each metric after training (requires ``track_all`` extra).
        plotly: Saves an HTML figure of each metric after training (requires ``track_all`` extra).
    """

    hydra: bool
    csv: bool
    tensorboard: bool
    wandb: bool
    sqlalchemy: bool
    matplotlib: bool
    plotly: bool


@dataclasses.dataclass(frozen=True)
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
        config_name: Name of the file in ``configs/experiment`` the configuration starts from.
    """

    output_dir: pathlib.Path = pathlib.Path('outputs')
    job_logging: Any = None
    config_name: str = ''


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PlottingSettings:
    """Options for visual inspection.

    Attributes:
        show: Also display the learning curves of the plotting trackers after training (needs a display).
    """

    show: bool = False


@dataclass(frozen=True)
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        device: Device where the models run: ``auto`` for the accelerator when available, or a name such as
            ``cpu`` or ``cuda:1``.
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        n_subprocesses: Number of DDP processes. ``0`` disables distributed training.
        trackers: Active experiment-tracking backends.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        checkpoint_every: Save a model checkpoint every *n* epochs.
        torch_compile: Whether to compile the model with torch.compile.
        load: Checkpoint epoch to resume from:``0`` skip; ``-1`` loads last checkpoint.
        run_id: Run to resume. Defaults to the last run of the experiment.
        plot: Options for visual inspection.
    """

    device: Device
    n_workers: PositiveInt
    n_subprocesses: PositiveInt
    trackers: TrackerList
    seed: int | None
    checkpoint_every: PositiveInt
    torch_compile: bool = False
    load: int = 0
    run_id: str | None = None
    plot: PlottingSettings = dataclasses.field(default_factory=PlottingSettings)
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return


# ---------------------------------------------------------------------------
# Root config — main entry point for the configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        classification: Classification phase configuration.
        autoencoding: Autoencoding phase configuration.
        user: User-specific options and runtime preferences.
        data: Data pre-processing and dataset configuration.
        final: If True, uses validation data for training and test data for evaluation.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    classification: ClassificationConfig
    user: UserSettings
    data: DataConfig
    final: bool
    autoencoding: AutoencodingConfig | None = None
    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in the configuration file."
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
    def par_dir(self) -> pathlib.Path:
        """Directory of the experiments of this configuration: ``<version_dir>/<config_name>``."""
        return self.user.path.version_dir / self.user.hydra.config_name

    @property
    def project(self) -> str:
        """Tracker project name, one per version."""
        return f'{PROJECT_NAME}-{self.version}'
