"""the LAND attacks of the calibration part that score at or below the LTT threshold of an e12 run, against the other
LAND attacks of the calibration, selection and test parts, by time and by every flow feature of the processed table.
it reads the run folder and the processed table where forge finds them, so on Colab it runs after the cells that mount
Drive and install the code.

    python analysis/e12_land_episodes.py <run_id>
"""

import sys
import time

import numpy as np
import pandas as pd

from forge import paths, runs
from forge.config import load_config
from forge.data import litnet2020

run_id = sys.argv[1]
run_dir = paths.runs_dir() / "e12_litnet_prevalence" / run_id
t0 = time.time()
scores = runs.load_arrays(run_dir, "scores")
grid = runs.load_table(run_dir, "grid")
ltt = runs.load_arrays(run_dir, "draws")["ltt_iid_index"]
threshold = float(grid.loc[int(np.median(ltt[ltt >= 0])), "threshold"])
land = scores["family"] == list(scores["families"]).index("land")
rows, score, part = scores["row"][land], scores["score"][land], scores["part"][land]
cfg = load_config(paths.configs_dir() / "data" / "litnet2020.yaml")
features = ["td", "proto", "tcp_flags", "stos", "ipkt", "ibyt", "sp", "dp", "in_if", "out_if", "sas", "das", "exid"]
flows = litnet2020.load(cfg, columns=["te", "row_id", "label", "family", "capture", *features]).iloc[rows]
flows = flows.assign(score=score, part=part, sp_is_dp=(flows["sp"] == flows["dp"]).to_numpy(),
                     sp_range=litnet2020.port_range(flows["sp"].to_numpy()))
groups = {"episodes": (part == 2) & (score <= threshold), "other calibration": (part == 2) & (score > threshold),
          "selection": part == 1, "test": part == 3}
print(f"run {run_id}, LTT threshold score {threshold:.4f}, {land.sum()} LAND attacks scored, {time.time() - t0:.0f} s")


def top(values, k=4):
    # the k most frequent values with their share
    counts = pd.Series(values).value_counts(normalize=True).head(k)
    return " ".join(f"{v} {s:.3f}" for v, s in counts.items())


for name, mask in groups.items():
    g = flows[mask]
    print(f"\n{name}, {len(g)} attacks")
    if g.empty:
        continue
    seconds = g["te"].dt.floor("s").value_counts().sort_index()
    print(f"  te {g['te'].min()} to {g['te'].max()}, {len(seconds)} distinct seconds, busiest "
          + " ".join(f"{str(t)[11:19]} {n}" for t, n in seconds.sort_values(ascending=False).head(4).items()))
    print("  score quantiles 0.1 0.5 0.9 " + " ".join(f"{v:.3f}" for v in np.quantile(g["score"], [0.1, 0.5, 0.9])))
    for column in ["td", "ipkt", "ibyt"]:
        print(f"  {column} quantiles 0.1 0.5 0.9 " + " ".join(f"{v:g}" for v in np.quantile(g[column], [0.1, 0.5, 0.9])))
    for column in ["proto", "tcp_flags", "stos", "dp", "sp", "sp_range", "sp_is_dp", "in_if", "out_if", "sas", "das",
                   "exid"]:
        print(f"  {column} {top(g[column].astype(str))}")
# the episodes in time, every second of the calibration part of LAND with its attacks and how many of them are low
cal = flows[groups["episodes"] | groups["other calibration"]]
per = cal.assign(second=cal["te"].dt.floor("s"), low=cal["score"] <= threshold).groupby("second")["low"].agg(["size", "sum"])
print("\nseconds of the calibration part, attacks and low attacks " + ", ".join(
    f"{str(t)[11:19]} {n} {k}" for t, (n, k) in per.iterrows()))
