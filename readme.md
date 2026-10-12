# forge-nids

FORGE, false omission rate guarantees under evolving traffic. An alert triage method for network intrusion detection that dismisses part of the traffic automatically, with a finite-sample bound on the share of attacks inside the dismissed part, valid under changes in attack prevalence and monitored after deployment through random audits and confidence sequences.


## Quick start

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run.py configs/experiments/e00_smoke.yaml
```

The smoke run writes a run folder under `drive/runs/e00_smoke/` locally, or under the project folder on Drive when it runs on Colab. Experiments run on Colab through `notebooks/run_experiment.ipynb`.
