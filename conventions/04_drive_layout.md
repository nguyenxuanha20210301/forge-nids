# Drive layout

```
MyDrive/forge-nids/
├── notebooks/                       uploaded copies of notebooks/*.ipynb
├── data/
│   ├── raw/<dataset>/               original downloads, never modified
│   └── processed/<dataset>/<version>/   cleaned tables, v1, v2 and so on
├── runs/<experiment>/<run_id>/      one folder per run, written once
│   ├── config.yaml
│   ├── meta.json
│   ├── log.txt
│   ├── results/
│   └── figures/
└── exports/                         anything packed for sharing
```

The runner notebook creates these folders on first use. The local `drive/` folder in the repository has the same layout and holds whatever was pulled down.

## Run folders

- The run id is `<yyyymmdd>-<hhmmss>_<commit>`, in UTC, for example `20261010-091500_ab12cd3`. Sorting by name sorts by time.
- A run folder is never edited after the run ends. A rerun, a fix or a new seed makes a new run folder. Failed runs stay, with `"status": "failed"` in `meta.json`, and are deleted by hand once understood.
- Redrawn figures do not go back into the run folder. They go to `redrawn/<experiment>/<run_id>/` next to `runs/`.

## Processed data versions

A processed dataset folder is never overwritten. A change to cleaning or splitting makes a new version folder, and the dataset config names the version in `processed_version`. Each version folder holds a `readme.md` saying what changed and the commit that produced it.

## Size

Drive has room to keep everything, so nothing is deleted to save space. Large intermediate arrays are saved as compressed `npz` or `parquet`, never as pickles.
