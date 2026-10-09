# Notebooks

## Thin runners

Notebooks in `notebooks/` do four things only: mount Drive, pull the code from GitHub, install it, and call a script with a config. Any logic longer than a few lines moves into `src/forge/`, where it is tested, versioned and importable from every notebook. This keeps a notebook on Drive valid across code changes, since it pulls the latest code each time it runs.

| Notebook | Purpose |
|---|---|
| `run_experiment.ipynb` | runs one experiment config and shows its figures |

Data preparation gets its own runner, `prepare_<dataset>.ipynb`, when the first dataset module exists.

## Names

Lowercase, verb first: `run_experiment.ipynb`, `prepare_litnet2020.ipynb`. Local analysis notebooks in `analysis/` follow the same rule.

## Uploading to Drive

1. Open `drive.google.com`, go to `My Drive`, then `forge-nids`, then `notebooks`.
2. Drag the `.ipynb` file from `~/projects/forge-nids/notebooks/` into the browser window, or use `New`, then `File upload`.
3. If a file with the same name exists, Drive asks whether to replace it or keep both. Choose replace, so links and the Colab history stay on one file.
4. Double-click the file and choose `Open with`, then `Google Colaboratory`.

## Outputs

Notebooks are committed with outputs cleared, so diffs show code changes only and no output from a run leaks into the repository:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

The copy on Drive keeps its outputs from the last Colab session. That copy is a working copy, and the version in git is the reference.
