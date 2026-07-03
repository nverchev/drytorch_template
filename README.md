# drytorch-template

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A project template for the [drytorch](https://github.com/nverchev/drytorch) ML experimentation library.

Out of the box it runs a synthetic tabular classification task. The entire
stack — config, data, model, training loop, evaluation, checkpointing,
hyperparameter tuning — is wired and ready; you adapt it to your task.

## Key Features

### Architecture & Design Patterns

- **Modular Layer Primitives**: Extensible `BaseLayer` abstraction (`src/module/layers.py`) pairing linear/convolutional operations (`LinearLayer`, `PointsConvLayer`) with configurable normalization, activations, activation-aware parameter initializations (Kaiming, Xavier, Truncated Normal), forward/backward debug hooks, and residual block composition (`LinearResBlock`).
- **Typed & Validated Configuration**: Pydantic-backed dataclasses (`AllConfig`) paired with Hydra YAML composition, incorporating runtime validation, git branch variation tracking, and automatic experiment naming from CLI overrides.
- **Structured Data Subpackage**: Clean separation of dataset construction (`DataBuilder`), offline feature preprocessing (`src/data/preprocess.py`), online runtime augmentations (`src/data/augmentations.py`), and multiprocess-safe dataset splitting.
- **Optuna Hyperparameter Tuning**: First-class Optuna optimization pipeline (`tune.py`) featuring Gaussian Process sampling, trial pruners, and robust imputation for pruned or failed trials.

### DryTorch Integration & Experiment Management

- **Experiment & Run Lifecycle**: Managed run lifecycle via `Experiment(cfg)`, handling version-scoped directory organization (`v<VERSION>`) and run metadata recording.
- **Multi-Backend Tracking Subscriptions**: Multi-tracker subscriptions (Hydra config logs, CSV, TensorBoard, Weights & Biases, SQLite/SQLAlchemy) configured via `get_trackers(cfg)`, with automatic debug-mode detection.
- **Loss Watching & Auxiliary Metrics**: `drytorch.lib.objectives.Loss` wrappers with metric binding (`loss.watch(metric)`), logging secondary evaluation metrics live alongside loss without extra validation passes.
- **Stateful Metric Accumulation & Reduction**: Custom batch accumulators (`AbstractAccumulator`, `AbstractAggregator`) and evaluation objectives (`ObjectiveProtocol` in `src/train/metrics_and_losses.py`) for cross-batch matrix reduction (e.g. confusion matrices, macro metrics) with automatic DDP rank synchronization (`sync()`).
- **Learning Schemas & Schedulers**: Flexible training loop orchestration using `drytorch.Trainer` paired with configurable learning schemas (optimizers, gradient clipping, LR schedulers like Cosine, Step, Warmup).
- **Training Hooks & Callbacks**: Automated checkpointing (`register_checkpointing`), early stopping (`register_early_stopping`), and Optuna trial pruning (`register_pruning`).
- **Distributed Execution**: Native DDP worker spawning via `DistributedWorker` for multi-GPU execution.

## Setup

```console
uv sync
```

Optional extras:

```console
uv sync --extra track_all   # TensorBoard, W&B, SQLAlchemy
uv sync --extra finetune    # Optuna hyperparameter tuning
```

## Configuration

Configuration is managed with [Hydra](https://hydra.cc/). The config
hierarchy lives under `configs/experiment/`:

- `defaults.yaml` — top-level composition
- `experiment/` — model and training settings
- `data/` — dataset source, preprocessing, augmentation
- `user/` — user-specific paths and tracker preferences

Any value can be overridden on the command line:

```console
uv run train_classifier.py experiment.train.learn.learning_rate=1e-4
uv run train_classifier.py final=true
```

## Usage

### Train

```console
uv run train_classifier.py
```

### Profile

```console
uv run profile_model.py
```

### Hyperparameter tuning

```console
uv run tune_classifier.py
uv run tune_classifier.py tune=learn
uv run tune_classifier.py tune=architecture
```

Requires the `finetune` optional dependency group (`uv sync --extra finetune`).
Add parameter search spaces under `configs/tuning/experiment/tune/`.

## Adapting to a new project

1. **`src/data/structures.py`** — redefine `Inputs`, `Targets`, `Outputs`.
2. **`src/config/options.py`** — add `Datasets` / `Models` enum values.
3. **`src/config/specs.py`** — extend `ModelConfig`, `DataConfig`, etc.
4. **`configs/experiment/`** — add YAML configs for new options.
5. **`src/data/synthetic_dataset.py`** — replace `SyntheticDataset` (or subclass `DataBuilder`).
6. **`src/data/augmentations.py`** — implement `get_augmenter`.
7. **`src/module/architectures.py`** — add new `BaseArchitecture` subclasses.
8. **`src/train/loss.py`** — replace `get_loss` / `get_accuracy`.
9. **`train.py`** — adjust the entry point to match your pipeline.
