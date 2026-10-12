# Workflow

## Three places, one name

The project is called `forge-nids` in all three places.

| Place | Location | Holds | Used for |
|---|---|---|---|
| Local machine | `~/projects/forge-nids` | the git working copy, `docs/` with the proposal and plan, and `drive/`, a local mirror of runs pulled from Drive | writing code, configs and notebooks, analysing downloaded results, committing and pushing |
| GitHub | `<github-user>/forge-nids`, private | code, configs, notebooks, conventions, small paper results | the single source of code for Colab |
| Google Drive | `MyDrive/forge-nids` | raw and processed data, every run folder, uploaded notebooks | everything Colab reads and writes |

Colab is where computation happens. It never holds anything that matters after the session ends: the code is cloned fresh into `/content/forge-nids` and every output is written to Drive.

## The loop

1. **Write locally.** Edit code and configs in `~/projects/forge-nids`. Run `pytest` and, when useful, a small local run with the smoke config.
2. **Push.** Commit and push to GitHub yourself, see [02_git.md](02_git.md).
3. **Run on Colab.** Open `MyDrive/forge-nids/notebooks/run_experiment.ipynb` in Colab, set `CONFIG`, run all cells. The notebook pulls the branch from GitHub, installs the package, runs `scripts/run.py` and writes a new run folder under `MyDrive/forge-nids/runs/<experiment>/`.
4. **Pull results.** Download the run folder from Drive. It lands in `~/Downloads` as one or more zip files. Run `python scripts/import_runs.py ~/Downloads/<name>*.zip --delete` to move it into `drive/runs/` locally.
5. **Analyse.** Work on the local mirror with scripts in `analysis/` or `scripts/plot.py`. Copy the tables and figures that go into the paper into `results/` and commit them.

Code reaches Colab only through GitHub. Nothing is edited on Colab or on Drive by hand, apart from uploading notebooks.

## When the notebook itself changes

The runner notebook rarely changes because the logic lives in `src/`. When it does change, push it as usual and upload the new file to `MyDrive/forge-nids/notebooks/`, replacing the old one. See [08_notebooks.md](08_notebooks.md).

## Secrets on Colab

The runner reads two values from Colab Secrets (the key icon in the left bar of Colab), so no notebook contains them:

| Secret | Value |
|---|---|
| `GITHUB_REPO` | `<github-user>/forge-nids` |
| `GITHUB_TOKEN` | a fine-grained GitHub token with read-only access to the contents of this one repository |
