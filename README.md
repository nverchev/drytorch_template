# drytorch-template

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A project template for the [DRYTorch](https://github.com/nverchev/drytorch) ML experimentation library. Out of the box, it provides both a simple baseline (`mlp_classifier`) and a multi-phase pipeline (`signal_classifier`, consisting of autoencoding pretraining followed by classification on learned representations), showing how the structure extends to complex, multi-phase workflows.

This file explains how the template works and how to use it; for recommended
best practices on writing code and configuration files, see also
[GUIDELINES.md](GUIDELINES.md).

## Layout

Package organization and how to adapt it for your own task. Proceed with the changes in the following order.

| Area | Files | This template | You change |
| --- | --- | --- | --- |
| Project | `pyproject.toml` | Experiment metadata, dependencies | rename it and adapt dependencies, including extras for trackers |
| Typed I/O | `src/data/structures.py` | Classes for inputs, outputs, and targets | modify structures for your sample data; all modules consume these |
| Options | `src/config/options.py`, `src/config/torch.py` | Option enums and PyTorch defaults | add members for your models and datasets; factories dispatch on them |
| Configuration | `src/config/specs.py`, `configs/experiment/` | Pydantic schemas, Hydra YAMLs, entry configs | modify fields and YAMLs; add entry configs and phases (see [Configuration](#configuration)) |
| Data | `src/data/` | Dataset builder, splits, preprocessing, augmentations | modify the builder for your source; add further datasets in separate modules |
| Models | `src/module/` | Layer primitives, model architectures | replace with your own network architectures; reusable layers can stay |
| Training | `src/train/` | Objectives, learning schema, loaders, hooks, model wrappers, events | modify the loss, metrics, loaders, and optimizer schema; hooks and wrappers can stay |
| Entry points | `train_*.py`, `profile_classifier.py`, `clone_experiment.py` | CLI entry points for training phases, profiling and reusing a phase | adapt or add one training script per phase; profiling script can stay |
| Tuning | `tune_mlp_classifier.py`, `configs/tuning/` | Optuna study, search space configurations | modify search spaces under `configs/tuning/tune/`; write one tuning script per phase |
| Tracking & Utils | `src/trackers/`, `src/utils/parallel.py` | Custom event handlers, tracker setup (`get_trackers`), learning curves (`save_learning_curves`), DDP multi-process spawning | usually stay unchanged unless tracking custom artifacts or adapting DDP |

## Starting a project

### Copying the template

Start the project from a copy of the template with a history of its own, with
GitHub's "Use this template" or:

```console
git clone --depth 1 https://github.com/nverchev/drytorch_template.git my_project
cd my_project
rm -rf .git
git init
```

### Setup

```console
uv sync --no-dev
```

for minimal support, or

```console
uv sync --all-extras
```

for full tracker, tuning and development support.

Paths come from a `.env`, gitignored and created by you: `DATA_DIR` for the
datasets (default `./data`) and `ROOT_EXP_DIR` for the experiments (default
`./experiments`).

Install the hooks with `uv run pre-commit install`: `.pre-commit-config.yaml`
runs ruff and basedpyright on every commit, following the rules in `ruff.toml`,
which you adapt to your own conventions.

## Configuration

[Hydra](https://hydra.cc/) composes the configuration from the YAML files in
`configs/experiment/`, and pydantic checks the result against the dataclasses
in `src/config/specs.py`. Unknown keys and invalid values stop the run before it
starts. The result is a frozen `ExperimentConfig`, which the code reads with
`Experiment.get_config()` and DRYTorch records with the run.

### Groups and setting names

Following standard Hydra mechanics (see the
[Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/)),
folders under `configs/experiment/` are *groups* and files are *options*,
composed via the `defaults:` list.

The YAML hierarchy is symmetrical to the Pydantic dataclasses in
`src/config/specs.py`. A setting's dotted path matches its attribute path on
`ExperimentConfig`:

| YAML folder | Pydantic attribute | Example setting |
| --- | --- | --- |
| `classification/` | `ExperimentConfig.classification` | `classification.name` |
| `classification/model/` | `classification.model` | `classification.model.class_name` |
| `classification/train/learn/` | `classification.train.learn` | `classification.train.learn.lr` |

### Entry configurations

The files directly in `configs/experiment/` are the starting points of a run.
The template has two:

- `mlp_classifier.yaml`, default tabular feature classification with an MLP;
- `signal_classifier.yaml`, a two-phase pipeline on 1D signals (`autoencoding` pretraining phase followed by `classification` on encodings).

Select one with `--config-name`:

```console
uv run train_classifier.py                                    # mlp_classifier.yaml
uv run train_classifier.py --config-name signal_classifier
```

Entry configurations define baseline setups by selecting top-level defaults across the
hierarchy. Switching configuration files changes whole components at once—such as the
model, dataset, or training pipeline. Each entry configuration keeps its runs in its own
folder (see [Experiments](#experiments)).

### Phases

A *phase* encapsulates the settings for one model and its training scheme (`PhaseConfig`). An experiment can define multiple phases executed in sequence (e.g. self-supervised pretraining followed by downstream probing on representations). Settings common to the whole experiment (`data` and `user`) are shared across phases.

In this template:
- `mlp_classifier.yaml` defines a single phase (`classification`): an MLP on the tabular `synthetic` dataset.
- `signal_classifier.yaml` defines a two-phase pipeline on the 1D `signal` dataset, with two phases: `autoencoding` (residual 1D CNN autoencoder) and `classification` (linear probe on the learned bottleneck encodings).

### Overrides

Any setting can be changed from the command line, in two ways:

```console
uv run train_classifier.py classification.train.learn.lr=1e-4   # a value: its path, with dots
uv run train_classifier.py data/dataset=synthetic               # an option: its group, with slashes, and the file name
```

The `user` group holds the settings of the machine and of convenience:
workers, trackers, checkpoint frequency, seed. Overrides of `user.*` do not
enter the name of the run, so they should not change its results.

The `final` flag determines which partitions a run uses:

- **`final=false`** *(default)*: trains on `train`, evaluates on `val`
  (tuning and comparisons);
- **`final=true`**: trains on `train` and `val`, evaluates on `test` (final
  evaluation).

## Experiments

### Run names

A run is named after its *variation*, followed by its overrides, sorted, and
by `_final` when `final=true`. The variation is the `variation:` value of the
entry configuration, or the current git branch when it is empty. On the branch
`main`:

```console
uv run train_classifier.py                                        # main
uv run train_classifier.py classification.train.learn.lr=1e-4     # main_classification.train.learn.lr=1e-4
uv run train_classifier.py final=true                             # main_final
```

### Folders

Runs are stored under `<ROOT_EXP_DIR>/v<version>/<config_name>/`, where
`version` comes from `pyproject.toml` and `config_name` is the entry
configuration. Each folder holds the checkpoints, the metrics and the metadata
of the runs:

```
experiments/v0.1/mlp_classifier/checkpoints/main/
experiments/v0.1/mlp_classifier/checkpoints/main_classification.train.learn.lr=1e-4/
experiments/v0.1/signal_classifier/checkpoints/main/
```

See [GUIDELINES.md](GUIDELINES.md#experiments) for best practices on versioning,
comparing variations, and organizing models.

### Reusing a phase

The phases of `signal_classifier` share the run name, which includes every
override. An override of the classifier alone therefore changes the name under
which the classifier looks for the autoencoder. To try other classifier settings
without training the autoencoder again, clone its experiment under the new name
first, with the same overrides:

```console
uv run train_autoencoder.py --config-name signal_classifier                                # main
uv run clone_experiment.py --config-name signal_classifier classification.train.learn.lr=1e-4
uv run train_classifier.py --config-name signal_classifier classification.train.learn.lr=1e-4
```

`clone_experiment.py` clones the experiment of the current variation (`main`
here), or the one named by `--from`. A final experiment (`final=true`) is only
cloned from another final experiment, which `--from` must name. The clone
keeps the checkpoints, metrics and metadata, so the new experiment also records
how its autoencoder was trained.

## Usage

### Train

```console
uv run train_classifier.py
uv run train_classifier.py user.n_subprocesses=2   # DDP
```

### Final pipeline

The final models train on `train` and `val` and are evaluated on `test`. Run every phase of
a configuration with the same overrides:

```console
uv run train_classifier.py final=true                                  # mlp_classifier
uv run train_autoencoder.py --config-name signal_classifier final=true # signal_classifier
uv run train_classifier.py --config-name signal_classifier final=true
```

### Profile

```console
uv run profile_classifier.py
```

### Hyperparameter tuning

```console
uv run tune_mlp_classifier.py
uv run tune_mlp_classifier.py tune=learn
```

This template only includes tuning for single-phase configurations (`mlp_classifier`). In multi-phase projects, tune each phase independently with its own tuning script.
Requires the `finetune` extra dependency (see Setup). The tuning configuration in
`configs/tuning/` is separate from the experiment configuration:

- `tune=<file>` chooses the search space, one file per study under
  `configs/tuning/tune/`;
- `config_name` is the entry configuration the trials start from;
- `overrides` lists the experiment overrides every trial starts from;
- `n_epochs` is the length of each trial (typically set to a fraction of the full training length).
- `mixed_precision` optionally sets the autocast precision of the trials; with `float16`, the DRYTorch trainer also scales the gradients.

Trials are stored in `optimization_db/`. To plot a study, pass
`plot_tuning_results.py` the same `tune`, `config_name` and `overrides` as the
tuning run, since they form the study name. `renderer` chooses where the plots
open: by default `browser`, the system's default browser, or any other plotly
renderer, such as `firefox`.

```console
uv run plot_tuning_results.py                   # the default study
uv run plot_tuning_results.py tune=learn        # the study of tune=learn
uv run plot_tuning_results.py renderer=firefox
```

#### Overriding settings when tuning

Tuning settings overrides use the standard Hydra syntax (`tune.n_trials=20`). 
To tune on overridden experimental settings, use the `overrides` parameter and pass them as a quoted list of strings:

```console
uv run tune_mlp_classifier.py "overrides=['classification.train.batch_size=32']"
```


### Debugging

Running an entry point under a debugger sets `DEBUG_MODE`, defined in
`src/config/environment.py`. It is the default of `get_trackers`, which skips
the trackers and leaves the run unrecorded, so no metrics or metadata are
stored. Checkpoints are still saved, which can be useful for debugging.

For example, you can debug the training script under `pdb` as follows:

```console
uv run -m pdb -c c train_classifier.py
```

### Testing

The tests use [pytest](https://docs.pytest.org):

```console
uv run pytest
uv run pytest tests/units/data/test_split.py::TestDataSplit::test_augmentation_only_outside_inference_mode
```

Every test runs inside a DRYTorch run, which the autouse fixture in
`tests/conftest.py` opens in a temporary folder. 

## Example workflow

A typical workflow for setting up a baseline, tuning hyperparameters, and
tracking experiment variations:

> [!TIP]
> Commit before every recorded run so DRYTorch links the run to the exact code.

Set up the baseline:

```console
git switch -c development                     # the branch names the runs
uv run pytest                                 # the code works
uv run -m pdb -c c train_classifier.py        # under the debugger: records nothing
uv run tune_mlp_classifier.py tune=learn      # search the learning settings
uv run plot_tuning_results.py tune=learn      # look at the study
```

Write the best values into the YAML files, bump the version, and train the
baseline:

```console
uv run train_classifier.py
```

Try a change, either as an override, tuned with the same override:

```console
uv run train_classifier.py classification.model.architecture.hidden_dims=[512,256]
uv run tune_mlp_classifier.py tune=learn "overrides=['classification.model.architecture.hidden_dims=[512,256]']"
```

or by starting fresh on a new branch describing the change:

```console
git switch -c wider_layers
uv run train_classifier.py
uv run tune_mlp_classifier.py tune=learn
```

Repeat the training steps for each entry configuration, one script per phase
(tuning covers single-phase configurations only):

```console
uv run train_autoencoder.py --config-name signal_classifier
uv run train_classifier.py --config-name signal_classifier
```

Finally, train each baseline on the training and validation splits and
evaluate it on the test split. Set the seed, so that the final runs can be
repeated, and, if the validation runs stopped early, set `n_epochs` to the epoch
they stopped at:

```console
uv run train_classifier.py final=true user.seed=0
uv run train_autoencoder.py --config-name signal_classifier final=true user.seed=0
uv run train_classifier.py --config-name signal_classifier final=true user.seed=0
```

## Projects using this template

- **[PCGen](https://github.com/nverchev/PCGen)**: point cloud generation with two models trained in sequence, and visualization through training hooks and custom events.
- **[SurfaceVAE](https://github.com/nverchev/SurfaceVAE)**: mesh autoencoder with an annealed loss.
- **[VAEX](https://github.com/nverchev/VAEX)**: visual counterfactuals with a pipeline for complex evaluation in explainable AI (XAI).
