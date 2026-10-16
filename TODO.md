# TODO once drytorch is released

CI cannot pass while the project depends on a local checkout of drytorch, so
the push waits for the release. When the stable version is on PyPI:

1. Remove the local source from `pyproject.toml`:

   ```toml
   [tool.uv.sources]
   drytorch = { path = "../drytorch", editable = true }
   ```

   and set the released version in `dependencies`.
2. Rebuild from scratch and check the template against the published package:

   ```console
   rm -rf .venv uv.lock
   uv sync --all-extras --dev
   uv run ruff check . && uv run ruff format --check . && uv run basedpyright
   uv run pytest
   uv run train_classifier.py classification.train.n_epochs=1
   ```
3. Commit `.github/workflows/ci.yaml`, which runs on `main`.
4. Merge `development` into `main` and push. On GitHub, make `main` the default
   branch and delete `master`.
5. Do the same in SurfaceVAE: drop its `[tool.uv.sources]` entry and merge its
   `development` branch into `main`.
6. Delete this file.
