# Code

## Names

- Every file and folder name is lowercase, words joined by underscores: `import_runs.py`, `e10_prevalence_shift.yaml`, `for_by_window.png`. The project name `forge-nids` is the one hyphenated name.
- Python follows PEP 8: `snake_case` functions and variables, `CapWords` classes, `UPPER_CASE` constants.
- Short mathematical names are fine when they match the paper: `pi`, `eps`, `delta`, `t`.

## Paths and personal data

- No absolute path appears in code, configs or notebooks, except the fixed Colab mount point `/content/drive/MyDrive/forge-nids` in `paths.py` and the runner notebook.
- Code finds data and runs through `forge.paths` only. `FORGE_STORAGE` overrides the storage root for a single shell when needed.
- No username, email, machine name, token or Drive file id appears in any committed file. GitHub details reach Colab through Colab Secrets.
- Local runs, temporary files and caches go under `drive/` or the system temp folder, never next to the code.

## Configs

- Every setting that changes a result lives in a yaml config, not in code. Code reads it from the `cfg` dictionary it receives.
- Experiment configs inherit `base.yaml` and, when they use a dataset, the dataset config, through `inherit: [../base.yaml, ../data/litnet2020.yaml]`.
- Quick variations use overrides on the command line, `--set seed=1 smoke.epsilon=0.005`. The resolved config is saved in the run folder, so overrides are never lost.

## Randomness

Every random draw takes a generator built from `cfg["seed"]`, for example `np.random.default_rng(cfg["seed"])`. Repeated runs over seeds are separate runs with `--set seed=<n>`, and analysis groups them by experiment.

## Dependencies

Dependencies are listed in `pyproject.toml` with lower bounds. Colab already ships most of them, so `pip install -e .` adds only what is missing. The exact versions used by a run are recorded in its `meta.json`. A new dependency is added to `pyproject.toml` in the same commit as the code that needs it.

## Tests

`tests/` covers what a silent error would corrupt: bounds, metrics, splits, config merging and figure export. A fast test runs the smoke experiment end to end. `pytest -q` runs before every push.
