# drytorch-template

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A project template for the [drytorch](https://github.com/nverchev/drytorch) ML experimentation library. Out of the box, it implements a simple classification project but can be easily extended to other tasks.

## Layout

Package organization and how to adapt it for your own task. Proceed with the changes in the following order.

| Area | Files | In the box | You change |
| --- | --- | --- | --- |
| Project | `pyproject.toml` | the name and the version identifying your experiments, and the dependencies | rename it and adapt the dependencies, including the extras that supply the trackers |
| Typed I/O | `src/data/structures.py` | the typed structures the model consumes and produces | modify the structures according to your sample data |
| Options | `src/config/options.py`, `src/config/torch.py` | definitions and defaults of the choices the configuration names | add members; the factories dispatch on them |
| Configuration | `src/config/specs.py`, `configs/experiment/` | the typed configuration tree and the YAML composing it | modify the fields and their YAML; add subexperiments when needed |
| Data | `src/data/` | a dataset builder, its splits, preprocessing and augmentation | modify the builder for your source; add each further dataset in its own module |
| Model | `src/module/` | layer primitives and the architectures built from them | replace with your own architectures; this part is largely project dependent |
| Objectives | `src/train/objectives.py` | the loss and the metrics, with a stateful accumulator | modify the loss and the metrics; write a custom objective when needed |
| Training | `src/train/`, `train_classifier.py` | the learning schema, the hooks and the entry point wiring them | modify the entry point and the schema; the hook helpers usually stay |
| Tracking | `src/config/experiment.py`, `src/config/events.py`, `src/trackers/` | the tracker selection, and a custom event with its handlers as an example | delete the subpackage unless your project has an artefact the trackers do not already record |
| Tuning | `tune_classifier.py`, `configs/tuning/` | an Optuna study searching over the configuration | modify the search spaces under `configs/tuning/tune/`, one file per study |
| Distributed | `src/utils/parallel.py` | spawning of the DDP processes | usually unchanged; the init method and the backend need adapting on Windows |

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

Configuration is managed with [Hydra](https://hydra.cc/). The config
hierarchy lives under `configs/experiment/`.

Any value can be overridden on the command line.
The experiment will be automatically named according to the override and the
`git` branch name. Overrides in `configs/experiment/user.yaml` do not change the experiment name and should not affect the results of a run.

```console
uv run train_classifier.py classification.train.learn.lr=1e-4
uv run train_classifier.py final=true
```
The `final` flag determines which partitions a run uses:
- **`final=false`** *(default)*: trains on `train` → evaluates on `val` (tuning and comparisons)
- **`final=true`**: trains on `train` + `val` → evaluates on `test` (final evaluation)

## Versioning

`version` in `pyproject.toml` determines the parent directory of the experiment data, containing the checkpoints, the metrics and the metadata of the baseline and its variations:

```
experiments/v0.1/checkpoints/main/
experiments/v0.1/checkpoints/main_classification.train.learn.lr=1e-4/
experiments/v0.1/checkpoints/my_variation/
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
uv run tune_classifier.py tune=architecture
```

Requires the `finetune` extra (see Setup).
Add parameter search spaces under `configs/tuning/tune/`. Trials are stored in
`optimization_db/` and plotted with:

```console
uv run plot_tuning_results.py
```

### Debugging

Running an entry point under a debugger sets `DEBUG_MODE`, which skips the
trackers and leaves the run unrecorded, so a debugging session writes no
checkpoints, metrics or metadata.

### Testing

```console
uv run pytest
```
