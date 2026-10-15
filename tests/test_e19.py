import numpy as np

from forge import experiments, runs
from forge.batch import overrides
from forge.config import load_config
from forge.eval import variants
from forge.paths import configs_dir
from forge.runs import Run

from test_litnet_prevalence import OVERRIDES

E19 = configs_dir() / "experiments" / "e19_ablations.yaml"


def run_variant(name, root, extra=()):
    cfg = load_config(E19)
    entry = [v for v in cfg["sweep"] if v["variant"] == name][0]
    cfg = load_config(E19, OVERRIDES + [f"variant={name}", *overrides(entry.get("set") or {}), *extra])
    run = Run(cfg, root=root)
    experiments.get(cfg["experiment"]["module"]).run(cfg, run)
    run.finish()
    return run, cfg


def test_e19_variants_resolve_and_the_feature_sets_run_on_the_e12_pipeline(litnet_signal, tmp_path):
    cfg = load_config(E19)
    names = [v["variant"] for v in cfg["sweep"]]
    assert cfg["experiment"] == {"name": "e19_ablations", "module": "litnet_prevalence"} and cfg["variant"] == "base"
    assert len(set(names)) == len(names) and names[0] == "base"
    # every variant moves one setting of e12, or none, and the settings it moves exist in the e12 config
    e12 = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml")
    for v in cfg["sweep"]:
        moved = load_config(E19, [f"variant={v['variant']}", *overrides(v.get("set") or {})])
        for key in (v.get("set") or {}):
            node = e12
            for part in key.split("."):
                assert part in node, f"{key} is no setting of e12"
                node = node[part]
        assert len(v.get("set") or {}) <= 1
        if v["variant"].startswith("split"):
            assert np.isclose(sum(moved["split"]["shares"]), 1) and moved["split"]["shares"][1:3] == [0.1, 0.2]
        if v["variant"].startswith("features"):
            assert "sp_range" not in moved["scorer"]["features"]
    assert cfg["scorer"]["categorical"] == ["proto", "tcp_flags", "in_if", "out_if", "exid"]
    root = tmp_path / "runs"
    base, base_cfg = run_variant("base", root)
    without, _ = run_variant("features_no_ports", root)
    every, every_cfg = run_variant("features_all", root)
    assert runs.load_json(base.dir, "scorer")["features"] == e12["scorer"]["features"]
    assert runs.load_json(every.dir, "scorer")["encoder"]["categorical"] == ["proto", "tcp_flags", "in_if", "out_if",
                                                                            "exid"]
    assert set(runs.load_json(every.dir, "scorer")["encoder"]["numeric"]) == {"td", "stos", "ipkt", "ibyt", "dp",
                                                                              "sp", "sas", "das"}
    assert runs.load_json(without.dir, "scorer")["encoder"]["numeric"] == ["td", "stos", "ipkt", "ibyt"]
    assert [runs.load_run_config(d)["variant"] for d in runs.run_dirs("e19_ablations", root)] == \
        ["base", "features_no_ports", "features_all"]
    table = variants.e19_table(root)
    assert table["variant"].tolist() == ["base", "features_no_ports", "features_all"]
    assert table.loc[0, "features"] == " ".join(e12["scorer"]["features"])
    assert table.loc[1, "features"] == "td proto tcp_flags stos ipkt ibyt"
    assert table.loc[0, "split"] == "0.35 0.1 0.2 0.35" and (table["grid_thresholds"] > 0).all()
    for col in ["forge_simplex_dismissed", "ltt_iid_violation", "best_up_to_pi_max", "families", "selection_auc",
                "forge_simplex_largest_violation", "forge_simplex_refused"]:
        assert col in table
    # the fixture has no LAND capture, so its columns are empty
    assert table["land_reference_ltt_iid"].isna().all() and table["land_calibration_attacks"].isna().all()
    # a later run of a variant replaces the earlier one in the table
    again, _ = run_variant("base", root, ["certify.grid_size=20"])
    table = variants.e19_table(root)
    assert len(table) == 3 and table.loc[table["variant"] == "base", "run_id"].iloc[0] == again.run_id
    assert table.loc[table["variant"] == "base", "grid_size"].iloc[0] == 20
    assert variants.blaster_table(root).empty
