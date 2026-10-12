import numpy as np
import pandas as pd

from forge import experiments, runs
from forge.config import load_config
from forge.data import litnet2020
from forge.experiments import litnet_overview
from forge.paths import configs_dir
from forge.runs import Run

from conftest import PUBLISHED


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


def test_e01_on_the_layout_of_the_published_data(litnet_like, tmp_path):
    # eleven captures and twelve families, more categories than the small fixtures, on the dates of the real captures
    cfg = load_config(configs_dir() / "experiments" / "e01_litnet.yaml")
    litnet2020.prepare(cfg, files=litnet_like.files, duplicates=litnet_like.duplicates, log=lambda m: None)
    run = Run(cfg, root=tmp_path / "runs")
    experiments.get("litnet_overview").run(cfg, run)
    run.finish()

    windows = runs.load_table(run.dir, "windows")
    assert len(windows) == 48
    assert windows.loc[windows["n_flows"] > 0, "captures"].tolist() == [
        "udp_flood", "reaper_worm", "fragmentation", "syn_flood", "spam", "http_flood",
        "scanning_spread+blaster_worm", "land_attack+red_worm+icmp_flood",
    ]
    captures = runs.load_table(run.dir, "captures")
    assert captures["capture"].tolist() == [litnet2020.capture_name(name) for name, _, _ in PUBLISHED]
    assert captures.set_index("capture").loc["icmp_flood", "families"] == "smurf+icmp_flood"
    mix = runs.load_table(run.dir, "family_mix")
    assert mix.groupby("family")["n_attacks"].sum().gt(0).all()


def test_capture_table_counts_beyond_int8_codes():
    # category codes are int8 below 128 categories, so arithmetic on them must not stay in int8
    names = [f"c{i:02d}" for i in range(12)]
    flows = pd.DataFrame({
        "te": pd.date_range("2020-01-01", periods=len(names), freq="h"),
        "label": np.ones(len(names), dtype=np.int8),
        "family": pd.Categorical(["fragmentation"] * len(names), categories=litnet2020.FAMILY_ORDER),
        "capture": pd.Categorical(names, categories=names),
    })
    table = litnet_overview.capture_table(flows, "te")
    assert table["capture"].tolist() == names
    assert (table["families"] == "fragmentation").all()
