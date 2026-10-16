# Guidelines

Recommendations for projects built on this template. Follow them unless there
is a good reason not to. The [README](README.md) explains how the template works:
the configuration, the run names and where runs are stored.

## Models

DRYTorch records the representation of the module as the architecture of the
model (`architecture.yaml` in the run's metadata). The representation lists the
modules, so a model documents itself only as far as it is made of modules.

- Build architectures from `nn.Sequential` and `nn.ModuleList`, not from layers
  stored one by one as attributes. A sequence states the order of its layers in
  one place, and its representation lists them in that order.
- Write every operation as a module, not as a call inside `forward`: reshaping,
  activations, normalization, pooling. When `torch.nn` has no module for an
  operation, write a small one, as `View` in `src/module/layers.py` does for a
  reshape.
- Give a module with settings an `extra_repr` that returns them, as
  `torch.nn.Linear` shows its dimensions: without it, the representation
  shows the module's name but not, for example, its temperature or the
  dimension it pools.
- Keep `forward` to the flow of the data: which module receives which tensor,
  and how branches are joined.

For example, prefer

```python
self.head = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_dim, h_dim),
    nn.ReLU(),
    nn.Linear(h_dim, n_classes),
)
```

to two linear layers stored separately, with the flattening and the activation
written in `forward`: there, the representation would show the two layers but
not the flattening or the activation between them.

### Model settings

- Give every role in the model (an encoder, a decoder, a classifier) an
  abstract base class, such as `BaseDecoder`, that reads the settings all its
  implementations use and stores them as public `Final` attributes. An
  implementation, such as `MyDecoder`, reads only the settings it adds and
  builds its layers from the stored ones. `BaseArchitecture` in
  `src/module/architectures.py` does this for the classifiers.
- Layers do not read the configuration: the module that builds them passes
  their settings, the activation and the normalization included, as
  arguments.
- Keep general modules in `src/module/layers.py`. A module that only one class
  uses, for a task specific to it, stays in the module of that class.

## Configuration

The code must be as configurable as possible, so that the settings alone can
document the experiment.

### Settings

- Write the value of every setting in the YAML files, even when the dataclass has a default.
- Make the dataclasses of the specifications frozen.
- Use `enum.StrEnum` for string options: `enum.auto()` maps upper-case members
  to lower-case YAML values (`SYNTHETIC = enum.auto()`). Set explicit values
  only when matching external names.
- Make sure that settings in `user` do not impact the experiment. For example,
  keep the total `batch_size` in the training configuration and divide it across
  `n_subprocesses`, so changing execution resources does not alter training dynamics.
- Leave the seed unset by default (`seed:` in `user/user.yaml`), and set it for
  the runs that must be repeatable.

### File names

- Give each configuration file a descriptive name that is clear on its own:
  editor tabs, defaults lists, and overrides show only the filename. Avoid
  generic names like `default.yaml` or `standard.yaml`; prefer file naming
  patterns (e.g., `train_vae.yaml`, `model_unet.yaml`) and use single-file
  groups for obvious cases (e.g., `train/train.yaml`).


## Experiments

- **Isolate one change at a time:** Keep data splits and augmentations
  identical across compared runs so differences reflect only the model or
  hyperparameter change. Keep the seed random when developing the project
  and set it for the final evaluation of the selected model.
- **Compare variations on the same setup with overrides:** When exploring
  hyperparameters, activations, or model architectures on the same dataset
  and task, use overrides on the same entry configuration. Overrides are
  recorded in the run name and share the same experiment folder, making them
  directly comparable side-by-side in trackers.
- **Separate non-comparable setups into entry configurations:** Use distinct
  entry configurations for setups that cannot be directly compared (i.e.
  different datasets paired with their own network architectures and training
  schemes). Each entry configuration switches the entire configuration tree and
  keeps its runs in an isolated folder (`experiments/v<version>/<config_name>/`).
- **Bump versions when baselines improve:** In `pyproject.toml`, bump the minor
  part (`0.1` → `0.2`) after hyperparameter tuning or adopting a better model,
  and the major part (`0.2` → `1.0`) after breaking specification changes. This
  archives earlier experiments cleanly without overwriting past runs.
- **Compare models on validation (`final=false`):** Compare models and
  variations strictly under `final=false` on the validation split. Reserve
  `final=true` only for the final benchmark evaluation of the selected model on
  the test set.
- **Treat tuning trials as a proxy:** Trials train for fewer epochs and in
  mixed precision to run faster. Check the selected settings with a full run.


## Pipelines

Some projects run several training phases or train multiple models in sequence,
each phase using the ones before it.

- Give each phase or model its own configuration section (see [Phases](README.md#phases)).
  In this template, `signal_classifier` demonstrates this pattern with an `autoencoding`
  phase followed by a `classification` probe.
- Give each phase its own training script. A later phase resumes the run of
  the earlier one and loads its checkpoint.
- Run every phase with the same overrides, apart from `user.*`: the overrides
  form the run name, and a later phase finds the earlier checkpoint under it.
- To change only a later phase, clone the experiment of the earlier phase under
  the new name with `clone_experiment.py` instead of training it again (see
  [Reusing a phase](README.md#reusing-a-phase)).
- Use one tuning script per phase in projects with several phases.

## Visualization

- Choose the samples to show with
  `drytorch.lib.loading.take_from_dataset(dataset, indices=...)`, with the
  indices in the configuration, so that every run shows the same samples and
  the samples arrive already batched and on the device.

## Style

- This template follows the style from DRYTorch
  [STYLE](https://github.com/nverchev/drytorch/blob/main/STYLE.md), but others
  are also possible.
- Tests maintain a simplified, flattened structure in `tests/units/` and adhere
  to STYLE rules: Arrange-Act-Assert layout, tests concluding directly on
  assertions rather than returning, and behavior-focused docstrings.
- Maintain a consistent coding and documentation style across the project.

### Tests

- Every test runs inside the run of the autouse `active_experiment` fixture in
  [tests/conftest.py](tests/conftest.py). Tests never create an `Experiment` or
  call `create_run`.
- A test that needs other settings uses `@pytest.mark.parametrize('overrides', [[...], ...])`.
  pytest replaces the `overrides` fixture with each value. Never loop over
  settings inside a test.
- A test reads the configuration through the `active_experiment` fixture
  (`active_experiment.config`) or `Experiment.get_config()`.
