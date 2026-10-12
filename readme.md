# forge-nids

FORGE, false omission rate guarantees under evolving traffic. An alert triage method for network intrusion detection that dismisses part of the traffic automatically, with a finite-sample bound on the share of attacks inside the dismissed part, valid under changes in attack prevalence and monitored after deployment through random audits and confidence sequences.


## Quick start

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run.py configs/experiments/e00_smoke.yaml
```

The smoke run writes a run folder under `drive/runs/e00_smoke/` locally, or under the project folder on Drive when it runs on Colab. Experiments run on Colab through `notebooks/run_experiment.ipynb`.

## Data

No dataset is stored in this repository. `python scripts/prepare.py configs/data/litnet2020.yaml` downloads the twelve LITNET-2020 capture archives from the authors' repository, [Grigaliunas/electronics9050800](https://github.com/Grigaliunas/electronics9050800), at commit `7b366c8`, keeps each file only if its size and sha256 match the values in `src/forge/data/litnet2020.py`, and writes a processed version whose `readme.md` lists the source, the processing steps, the counts per capture and a fingerprint of the table. On Colab, `notebooks/prepare_litnet2020.ipynb` does the same on Drive.

`python scripts/prepare.py configs/data/anoshift.yaml` streams the AnoShift tar, the Kyoto 2006+ sessions of 2006 to 2015 as preprocessed by the authors of [bit-ml/AnoShift](https://github.com/bit-ml/AnoShift), from Google storage once, hashes all of it, since no hash is published, and keeps only the ten yearly test files, each a random 10% of every label class of its year. Its processed version records the address, size and sha256 of the tar and of each kept file, the processing steps, the counts per year and a fingerprint of the table. On Colab, `notebooks/prepare_anoshift.ipynb` does the same on Drive.
