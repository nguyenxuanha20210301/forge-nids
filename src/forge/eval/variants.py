"""the variants of a sweep set side by side, one row per variant from the latest finished run of each, for the
ablations of E9 on the e12 pipeline and for the Blaster worm held out under its feature sets."""

import numpy as np
import pandas as pd

from forge import runs
from forge.eval.seeds import seed_runs

E19 = "e19_ablations"
E19_BLASTER = "e19_blaster_unseen"
E23 = "e23_unseen_family"
# the window whose price the split shares move and the family that only a scorer with the ports escalates unseen
LAND, BLASTER = "LAND", "blaster_worm"


def variant_runs(experiment: str, root=None) -> pd.DataFrame:
    # the latest finished run of every variant, in the order of the variants' first runs
    table = seed_runs(experiment, root)
    done = table[(table["status"] == "done") & table["variant"].notna()].sort_values("run_id")
    return done.drop_duplicates("variant", keep="last").reset_index(drop=True)


def _rate_of(claims: pd.DataFrame, arm: str, segment: str, pi: float, method: str) -> float:
    g = claims[(claims["arm"] == arm) & (claims["segment"] == segment) & (claims["prevalence"] == pi)
               & (claims["mix"] == "window") & (claims["method"] == method)]
    return float(g["rate"].iloc[0]) if len(g) else np.nan


def e19_table(root=None) -> pd.DataFrame:
    """one row per variant of e19, with its settings, the scorer, the families, the certificates in the calibration
    parts, every method on the reference windows inside the claim at the main prevalence, the three yardsticks, the
    violation shares on the LAND window and where the LAND attacks fall in the split."""
    rows = []
    for r in variant_runs(E19, root).itertuples():
        cfg, summary = r.config, runs.load_json(r.dir, "summary")
        by_prevalence = runs.load_table(r.dir, "by_prevalence")
        claims = runs.load_table(r.dir, "claims")
        cuts = runs.load_table(r.dir, "cuts")
        main, c, forge_cal = summary["main_prevalence"], cfg["certify"], summary["forge_calibration"]
        ref = by_prevalence[(by_prevalence["arm"] == "reference") & (by_prevalence["subset"] == "in_claim")
                            & (by_prevalence["mix"] == "window") & (by_prevalence["prevalence"] == main)]
        ref = ref.set_index("method")
        largest = summary["largest_violation_up_to_pi_max"].get("reference", {})
        land = cuts[cuts["families"] == "land"].set_index("part")
        row = {
            "variant": cfg.get("variant"), "run_id": r.run_id, "commit": r.commit, "seconds": r.seconds,
            "features": " ".join(cfg["scorer"]["features"]), "split": " ".join(str(s) for s in cfg["split"]["shares"]),
            "delta_benign": c["delta_benign"], "grid_size": c["grid_size"], "pi_max": c["pi_max"],
            "start_misses": c["start_misses"], "start_slack": c["start_slack"],
            "grid_thresholds": summary["grid_thresholds"], "family_minimum": summary["family_minimum"],
            "families": ", ".join("+".join(g) for g in summary["families"]),
            "calibration_sessions": summary["calibration_sessions"], "best_iteration": runs.load_json(
                r.dir, "scorer")["best_iteration"],
            "selection_auc": summary["selection_auc"], "calibration_auc": summary["calibration_auc"],
            "forge_benign_dismissed": forge_cal["benign_dismissed"],
            "forge_miss_land": forge_cal["miss"].get("land", np.nan),
            **{f"{m}_refused": v for m, v in summary["offline_refused"].items()},
            **{f"{m}_violation": float(ref.loc[m, "violation_rate"]) for m in ref.index},
            **{f"{m}_dismissed": float(ref.loc[m, "dismissed_mean"]) for m in ref.index},
            **{f"{m}_largest_violation": largest.get(m, {}).get("window", {}).get("rate", np.nan)
               for m in summary["methods"]},
            "best_up_to_pi_max": float(ref["best_up_to_pi_max_dismissed"].iloc[0]) if len(ref) else np.nan,
            "best_every_mix": float(ref["best_every_mix_dismissed"].iloc[0]) if "best_every_mix_dismissed" in ref
            and len(ref) else np.nan,
            **{f"land_reference_{m}": _rate_of(claims, "reference", LAND, main, m)
               for m in ("forge_simplex", "forge_pooled", "ltt_iid", "conformal_fnr")},
            **{f"land_real_{m}": _rate_of(claims, "real", LAND, main, m)
               for m in ("forge_simplex", "forge_pooled", "ltt_iid", "conformal_fnr")},
            **{f"land_{part}_attacks": int(land.loc[part, "attacks"]) if part in land.index else np.nan
               for part in ("train", "selection", "calibration", "test")},
            "land_calibration_first_row": int(land.loc["calibration", "first_row_id"]) if "calibration" in land.index
            else np.nan,
            "land_calibration_last_row": int(land.loc["calibration", "last_row_id"]) if "calibration" in land.index
            else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def blaster_table(root=None, base_run=None) -> pd.DataFrame:
    """one row per feature set under which the Blaster worm was held out, the e19_blaster_unseen runs and, as the
    base, an e23 run, the latest finished one with the seed of its source unless base_run names one. the AUC, the
    benign share dismissed and the miss rate of the Blaster window with the family unseen against known, and the FOR
    and the alarms of the monitor on the offline FORGE and the fixed threshold streams."""
    sources = [(r.config.get("variant"), r.dir, r.run_id) for r in variant_runs(E19_BLASTER, root).itertuples()]
    e23 = seed_runs(E23, root)
    e23 = e23[e23["status"] == "done"]
    if base_run is not None:
        e23 = e23[e23["run_id"] == base_run]
    else:
        e23 = e23[[c.get("seed", 0) == runs.load_json(d, "summary").get("source_seed", c.get("seed", 0))
                   for c, d in zip(e23["config"], e23["dir"])]]
    if len(e23):
        last = e23.sort_values("run_id").iloc[-1]
        sources.insert(0, ("base", last["dir"], last["run_id"]))
    rows = []
    for variant, run_dir, run_id in sources:
        summary = runs.load_json(run_dir, "summary")
        windows = runs.load_table(run_dir, "windows")
        claims = runs.load_table(run_dir, "claims")
        units = [u for u in summary["units"] if u["name"] == BLASTER]
        if not units:
            continue
        w = units[0]["window"]
        own = windows[(windows["unit"] == BLASTER) & windows["unseen"]].iloc[0]
        known = windows[(windows["unit"] == summary["control"]) & (windows["position"] == w)].iloc[0]
        cfg = runs.load_run_config(run_dir)
        source_dirs = [d for d in runs.run_dirs(cfg["source"]["experiment"], root) if d.name == cfg["source"]["run"]]
        # the features come from the source run, which may not be in the runs folder
        source_cfg = runs.load_run_config(source_dirs[0]) if source_dirs else {}
        row = {"variant": variant, "run_id": run_id, "source_run": cfg["source"]["run"],
               "features": " ".join(source_cfg.get("scorer", {}).get("features", [])),
               "auc_unseen": own["auc"], "auc_known": known["auc"],
               "benign_dismissed_unseen": own["forge_benign_dismissed"],
               "benign_dismissed_known": known["forge_benign_dismissed"],
               "miss_unseen": own[f"forge_miss_{BLASTER}"], "miss_known": known[f"forge_miss_{BLASTER}"]}
        for policy in ("forge_simplex", "fixed_sel"):
            g = claims[(claims["unit"] == BLASTER) & (claims["case"] == "unseen")
                       & (claims["composition"] == "typical") & (claims["policy"] == policy)
                       & (claims["arm"] == "monitor")]
            if len(g):
                g = g.iloc[0]
                row.update({f"{policy}_for": g["unseen_for_true_mean"], f"{policy}_miss": g["unseen_miss_mean"],
                            f"{policy}_violated": g["unseen_violated_rate"],
                            f"{policy}_alarm_inside": g["alarm_in_unseen_rate"],
                            f"{policy}_leak": g["unseen_leak_median"]})
        rows.append(row)
    return pd.DataFrame(rows)
