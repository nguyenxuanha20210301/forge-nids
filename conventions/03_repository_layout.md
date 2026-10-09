# Repository layout

```
forge-nids/
├── readme.md
├── pyproject.toml            package metadata and dependencies, installed with pip install -e .
├── .gitignore
├── conventions/              these rules
├── docs/                     proposal.md, plan.md, notes that outlive a single run
├── configs/
│   ├── base.yaml             defaults every experiment inherits
│   ├── data/                 one file per dataset, <dataset>.yaml
│   └── experiments/          one file per experiment, e<nn>_<short_name>.yaml
├── src/forge/                the importable package, all logic lives here
│   ├── paths.py              where data and runs are stored, resolved at runtime
│   ├── config.py             yaml loading with inherit and overrides
│   ├── runs.py               run folders, meta.json, saving and loading results
│   ├── data/                 loaders, cleaning, temporal splits, one module per dataset
│   ├── models/               base scorers and ensembles
│   ├── certify/              offline certification of dismiss thresholds
│   ├── monitor/              post-deployment audits and confidence sequences
│   ├── eval/                 metrics computed on saved predictions
│   ├── viz/                  style.py, export.py, shared plotting helpers
│   └── experiments/          one module per experiment, run() and FIGURES
├── scripts/                  command line entry points
│   ├── run.py                runs one experiment config
│   ├── plot.py               redraws figures of saved runs
│   └── import_runs.py        moves runs downloaded from Drive into drive/runs
├── notebooks/                Colab runners, thin, uploaded to Drive
├── analysis/                 local scripts and notebooks that compare runs
├── results/                  small tables and final figures used in the paper
├── tests/
└── drive/                    local mirror of the Drive folder, never committed
```

Folders that do not exist yet (`models/`, `monitor/`, `analysis/`) are created when their first file is written.

## Where things go

- **New method code** goes in the matching package under `src/forge/`. A module never reads a path directly. It takes arrays or data frames and returns them.
- **A new experiment** is two files: `configs/experiments/e<nn>_<name>.yaml` and `src/forge/experiments/<name>.py`. The module defines `run(cfg, run)` and `FIGURES`, see [07_results_and_figures.md](07_results_and_figures.md). Experiment numbers follow the research questions in `docs/plan.md`, for example `e10` to `e19` for RQ1.
- **A new dataset** is `configs/data/<dataset>.yaml` plus `src/forge/data/<dataset>.py`, with a function that reads from `paths.data_dir("raw")` and writes a versioned folder under `paths.data_dir("processed")`.
- **A helper used by two experiments** moves out of the experiment module into the package it belongs to.
- **Throwaway exploration** goes in `analysis/` and is deleted or promoted once it has answered its question.

## Tracing a number back

Every figure and table in the paper traces back the same way. The file in `results/` names its run id. The run folder on Drive holds `config.yaml` and `meta.json`, and `meta.json` holds the git commit. `git checkout <commit>` and the config reproduce the run.
