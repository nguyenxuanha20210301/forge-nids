import numpy as np
import pandas as pd

from forge import experiments, runs
from forge.config import load_config
from forge.data import litnet2020
from forge.paths import configs_dir
from forge.runs import Run


def test_e01_tables_and_figures(litnet_raw, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e01_litnet.yaml")
    litnet2020.prepare(cfg, files=litnet_raw.files, duplicates=litnet_raw.duplicates, log=lambda m: None)
    run = Run(cfg, root=tmp_path / "runs")
    experiments.get("litnet_overview").run(cfg, run)
    run.finish()
    truth = pd.DataFrame(litnet_raw.rows)
    a, b = truth[truth["capture"] == "a"], truth[truth["capture"] == "b"]

    windows = runs.load_table(run.dir, "windows")
    assert windows["start"].dt.strftime("%Y-%m-%d").tolist() == ["2019-12-30", "2020-01-06", "2020-01-13"]
    assert windows["n_flows"].tolist() == [len(b), 0, len(a)]
    assert windows["n_attacks"].tolist() == [b["attack_a"].sum(), 0, a["attack_a"].sum()]
    assert windows["prevalence"].iloc[0] == b["attack_a"].sum() / len(b)
    assert np.isnan(windows["prevalence"].iloc[1])
    assert windows["captures"].fillna("").tolist() == ["b", "", "a"]

    mix = runs.load_table(run.dir, "family_mix")
    assert len(mix) == 3 * len(litnet2020.ATTACK_FAMILIES)
    shares = mix.dropna(subset=["share"]).groupby("window")["share"].sum()
    assert shares.index.tolist() == [0, 2] and np.allclose(shares, 1)
    last = mix[mix["window"] == 2].set_index("family")["n_attacks"]
    assert last["smurf"] == (a["attack_t"] == "icmp_smf").sum()
    assert last["icmp_flood"] == (a["attack_t"] == "icmp_f").sum()

    captures = runs.load_table(run.dir, "captures")
    assert captures["capture"].tolist() == ["b", "a"]
    assert captures["families"].tolist() == ["udp_flood", "smurf+icmp_flood"]

    summary = runs.load_json(run.dir, "summary")
    assert summary["fingerprint"] == litnet2020.read_manifest(cfg)["fingerprint"]
    assert summary["windows_with_flows"] == 2
    names = {p.name for p in run.figures.iterdir()}
    for figure in experiments.get("litnet_overview").FIGURES:
        assert {f"{figure}.png", f"{figure}.pdf", f"{figure}_a4.pdf"} <= names
