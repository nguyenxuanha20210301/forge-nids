import numpy as np
import pandas as pd

from forge import experiments, runs
from forge.config import load_config
from forge.data import ugr16
from forge.experiments import ugr16_overview
from forge.paths import configs_dir
from forge.runs import Run

from conftest import UGR16_WEEKS


def kept(rows):
    # the rows the fixture keeps at every third row
    return [r for i, r in enumerate(rows) if i % 3 == 0]


def test_e02_tables_and_figures(ugr16_mirror, tmp_path):
    m = ugr16_mirror
    ugr16.prepare(m.cfg, files=m.files, url=m.url, scratch=m.scratch, log=lambda s: None)
    cfg = load_config(configs_dir() / "experiments" / "e02_ugr16.yaml")
    cfg["dataset"].update(m.cfg["dataset"])
    run = Run(cfg, root=tmp_path / "runs")
    module = experiments.get("ugr16_overview")
    module.run(cfg, run)
    run.finish()
    july, august = (kept(m.truth[w]) for w in UGR16_WEEKS)

    # the two weeks start five days apart at midnight, so 21 windows of six hours with the first and the last filled
    windows = runs.load_table(run.dir, "windows")
    assert len(windows) == 21 and windows["window"].tolist() == list(range(21))
    assert windows["start"].iloc[0] == pd.Timestamp("2016-07-27 00:00") and \
        windows["start"].iloc[20] == pd.Timestamp("2016-08-01 00:00")
    assert windows["n_flows"].tolist() == [len(july)] + [0] * 19 + [len(august)]
    assert windows["weeks"].tolist() == ["july_week5"] + [""] * 19 + ["august_week1"]
    for i, rows in [(0, july), (20, august)]:
        w = windows.iloc[i]
        labels = pd.Series([ugr16.family_name(r["label"]) for r in rows])
        attacks = labels[labels != "benign"]
        assert w["n_attacks"] == len(attacks) and w["prevalence"] == len(attacks) / len(rows)
        assert w["n_launched"] == attacks.isin(ugr16.INJECTED_FAMILIES).sum()
        assert w["n_blacklist"] == (attacks == "blacklist").sum()
        assert w["n_found"] == len(attacks) - w["n_launched"] - w["n_blacklist"]
        assert w["prevalence_without_blacklist"] == (len(attacks) - w["n_blacklist"]) / (len(rows) - w["n_blacklist"])
        counts = attacks.value_counts()
        assert w["families"] == ", ".join(f"{f} {counts[f]}" for f in ugr16.FAMILY_ORDER + ["anomaly_newkind"]
                                          if f in counts.index)
    assert np.isnan(windows["prevalence"].iloc[1]) and windows["families"].iloc[1] == ""

    # every attack family of the table has a row in every window, the unknown label of the fixture included
    mix = runs.load_table(run.dir, "family_mix")
    families = ugr16.ATTACK_FAMILIES + ["anomaly_newkind"]
    assert len(mix) == 21 * len(families) and mix["family"].iloc[:len(families)].tolist() == families
    shares = mix.dropna(subset=["share"]).groupby("window")["share"].sum()
    assert shares.index.tolist() == [0, 20] and np.allclose(shares, 1)
    first = mix[mix["window"] == 0].set_index("family")["n_attacks"]
    assert first["anomaly_newkind"] == sum(r["label"] == "anomaly-newkind" for r in july)
    assert first["dos"] == sum(r["label"] == "dos" for r in july)
    assert mix.groupby("window")["n_attacks"].sum().tolist() == windows["n_attacks"].tolist()

    weeks = runs.load_table(run.dir, "weeks")
    assert weeks["week"].tolist() == UGR16_WEEKS and weeks["n_flows"].tolist() == [len(july), len(august)]
    assert weeks["start"].tolist() == [min(r["te"] for r in july), min(r["te"] for r in august)]
    assert weeks["n_attacks"].tolist() == windows["n_attacks"].iloc[[0, 20]].tolist()
    assert weeks["families"].tolist() == windows["families"].iloc[[0, 20]].tolist()

    summary = runs.load_json(run.dir, "summary")
    assert summary["fingerprint"] == ugr16.read_manifest(m.cfg)["fingerprint"]
    assert summary["windows"] == 21 and summary["windows_with_flows"] == 2 and summary["window_length"] == "6h"
    assert summary["attacks"] == int(windows["n_attacks"].sum()) and summary["weeks"] == UGR16_WEEKS
    assert sum(summary["groups"].values()) == summary["attacks"]
    names = {p.name for p in run.figures.iterdir()}
    for figure in module.FIGURES:
        assert {f"{figure}.png", f"{figure}.pdf", f"{figure}_a4.pdf"} <= names
    log = (run.dir / "log.txt").read_text()
    assert "window 2016-07-27 06:00, no flows" in log and "2 of 21 windows of 6h hold flows" in log


def test_groups_cover_every_family():
    assert [ugr16_overview.group_of(f) for f in ugr16.ATTACK_FAMILIES] == \
        ["launched"] * 4 + ["found"] * 3 + ["blacklist"]
    assert ugr16_overview.group_of("anomaly_newkind") == "found"
