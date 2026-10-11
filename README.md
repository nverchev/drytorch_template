# drytorch-template

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A project template for the [drytorch](https://github.com/nverchev/drytorch) ML experimentation library. Out of the box, it implements a simple classification project but can be easily extended to other tasks.

This file explains how the template works and how to use it; for recommended
best practices on writing code and configuration files, see also
[GUIDELINES.md](GUIDELINES.md).

## Layout

Package organization and how to adapt it for your own task. Proceed with the changes in the following order.

| Area | Files | This template | You change |
| --- | --- | --- | --- |
| Project | `pyproject.toml` | Experiment metadata, dependencies | rename it and adapt the dependencies, including the extras that supply the trackers |
| Typed I/O | `src/data/structures.py` | Classes for inputs, outputs, and targets | modify the structures according to your sample data |
| Options | `src/config/options.py`, `src/config/torch.py` | Option enums and PyTorch defaults | add members; the factories dispatch on them |
| Configuration | `src/config/specs.py`, `configs/experiment/` | Pydantic schema, Hydra YAMLs, entry configurations | modify the fields and their YAML; add entry configurations and subexperiments when needed (see [Configuration](#configuration)) |
| Data | `src/data/` | Dataset builder, splits, preprocessing, augmentation | modify the builder for your source; add each further dataset in its own module |
| Model | `src/module/` | Layer primitives, baseline architectures | replace with your own architectures; this part is largely project dependent |
| Objectives | `src/train/objectives.py` | Loss function, metrics, stateful accumulator | modify the loss and the metrics; write a custom objective when needed |
| Training | `src/train/`, `train_classifier.py` | Learning schema, training hooks, CLI entry point | modify the entry point and the schema; the hook helpers usually stay |
| Tracking | `src/config/experiment.py`, `src/config/events.py`, `src/trackers/` | Tracker setup, custom event and handlers | delete the subpackage unless your project has an artefact the trackers do not already record |
| Tuning | `tune_classifier.py`, `configs/tuning/` | Optuna study, search space configurations | modify the search spaces under `configs/tuning/tune/`, one file per study (see [Hyperparameter tuning](#hyperparameter-tuning)) |
| Distributed | `src/utils/parallel.py` | DDP multi-process spawning | usually unchanged; the init method and the backend need adapting on Windows |

## Setup

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
starts. The result is a frozen `AllConfig`, which the code reads with
`Experiment.get_config()` and drytorch records with the run.

### Groups and setting names

Following standard Hydra mechanics (see the
[Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/)),
folders under `configs/experiment/` are *groups* and files are *options*,
composed via the `defaults:` list.

The YAML hierarchy is symmetrical to the Pydantic dataclasses in
`src/config/specs.py`. A setting's dotted path matches its attribute path on
`AllConfig`:

| YAML folder | Pydantic attribute | Example setting |
| --- | --- | --- |
| `classification/` | `AllConfig.classification` (`ExperimentConfig`) | `classification.name` |
| `classification/model/` | `classification.model` (`ModelConfig`) | `classification.model.class_name` |
| `classification/train/learn/` | `classification.train.learn` (`LearningConfig`) | `classification.train.learn.lr` |
| `data/dataset/` | `AllConfig.data.dataset` | `data.dataset.n_train_samples` |

### Subexperiments

`classification` is an *experimental part* or *subexperiment*, containing the settings for one model and its training. In this template, it is the only such part, but an experiment may have different subexperiments, each defining its own model and training scheme. Common settings (here `data` and `user`) are shared across subconfigurations.

### Entry configurations

The files directly in `configs/experiment/` are the starting points of a run.
The template has two:

- `mlp_classifier.yaml`, the default;
- `mlp_res_classifier.yaml`, the same setup with the residual model.

Select one with `--config-name`:

```console
uv run train_classifier.py                                    # mlp_classifier.yaml
uv run train_classifier.py --config-name mlp_res_classifier
```

Entry configurations select top-level defaults that propagate down the
hierarchy. A single file can switch entire sub-trees at once, such as the model,
the dataset, or the training scheme.

Because changes can span multiple components, entry configurations are
typically used for distinct experiments rather than direct comparisons. Each
one stores its runs in its own folder (see [Experiments](#experiments)).

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
experiments/v0.1/mlp_res_classifier/checkpoints/main/
```

Bump the version whenever the baseline improves or changes:

- **the minor part** (`0.1` to `0.2`) after tuning the hyperparameters or after
  a small architectural change;
- **the major part** (`0.2` to `1.0`) after a larger change, such as a change in
  the specifications that breaks backward compatibility.

## Usage

### Train

```console
uv run train_classifier.py
uv run train_classifier.py user.n_subprocesses=2   # DDP
```

### Profile

```console
uv run profile_model.py
```

### Hyperparameter tuning

```console
uv run tune_classifier.py
uv run tune_classifier.py tune=learn
uv run tune_classifier.py tune=architecture config_name=mlp_res_classifier
```

Requires the `finetune` extra dependency (see Setup). The tuning configuration in
`configs/tuning/` is separate from the experiment configuration:

- `tune=<file>` chooses the search space, one file per study under
  `configs/tuning/tune/`;
- `config_name` is the entry configuration the trials start from;
- `subexperiment` is the subexperiment whose settings the search space changes:
  a parameter `subexperiment.train.learn.lr` becomes
  `classification.train.learn.lr`;
- `overrides` lists the experiment overrides every trial starts from;
- `n_epochs` is the length of each trial (typically set to a fraction of the full training length).

Trials are stored in `optimization_db/` and plotted with:

```console
uv run plot_tuning_results.py
```

#### Overriding settings when tuning

Tuning settings overrides use the standard Hydra syntax (`tune.n_trials=20`). 
To tune on overridden experimental settings, use the `overrides` parameter and pass them as a quoted list of strings:

```console
uv run tune_classifier.py "overrides=['classification.train.batch_size=32']"
```

The overrides are part of the study name, so `plot_tuning_results.py` needs the same ones.


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

```console
uv run pytest
```

## Example workflow

A typical workflow for setting up a baseline, tuning hyperparameters, and
tracking experiment variations:

> [!TIP]
> Commit before every recorded run so drytorch links the run to the exact code.

Set up the baseline:

```console
git switch -c development                     # the branch names the runs
uv run pytest                                 # the code works
uv run -m pdb -c c train_classifier.py        # under the debugger: records nothing
uv run tune_classifier.py tune=learn          # search the learning settings
```

Write the best values into the YAML files, bump the version, and train the
baseline:

```console
uv run train_classifier.py
```

Try a change, either as an override, tuned with the same override:

```console
uv run train_classifier.py classification.model.architecture.hidden_dims=[512,256]
uv run tune_classifier.py tune=learn "overrides=['classification.model.architecture.hidden_dims=[512,256]']"
```

or by starting fresh on a new branch describing the change:

```console
git switch -c wider_layers
uv run train_classifier.py
uv run tune_classifier.py tune=learn
```

Repeat the same steps for each entry configuration:

```console
uv run train_classifier.py --config-name mlp_res_classifier
uv run tune_classifier.py config_name=mlp_res_classifier tune=learn
```

Finally, train each baseline on the training and validation splits and
evaluate it on the test split. Set the seed, so that the final runs can be
repeated, and, if the validation runs stopped early, set `n_epochs` to the epoch
they stopped at:

```console
uv run train_classifier.py final=true user.seed=0
uv run train_classifier.py --config-name mlp_res_classifier final=true user.seed=0
```

## Projects using this template

- **[PCGen](https://github.com/nverchev/PCGen)**: point cloud generation with two models trained in sequence, and visualization through training hooks and custom events.
- **[SurfaceVAE](https://github.com/nverchev/SurfaceVAE)**: mesh autoencoder with an annealed loss.
- **[VAEX](https://github.com/nverchev/VAEX)**: visual counterfactuals with a pipeline for complex evaluation in explainable AI (XAI).
