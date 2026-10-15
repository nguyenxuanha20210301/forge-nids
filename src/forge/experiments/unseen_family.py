"""e23, RQ4 on LITNET-2020, FORGE-Monitor against an attack family the scorer and the certificate never saw, on the
test parts of the e12 split in capture order.

for each family of a capture with benign flows, one unit, the scorer of e12 is refit on the training parts without
its attacks and stopped on the selection parts without them, the grid and the certificates of e12 are built on the
selection and calibration parts without them, and the test parts of the captures with benign flows are deployed in
capture order, each thinned to a prevalence as a real window of e12, the known windows at one prevalence and the
window of the held out family at another. the same windows under the scorer and the certificates with every family
known are the control, the case known against the case unseen. every unit draws its calibration sets and its windows
with the seeds of the e12 run, so a unit and its control differ in the scorer and the certificate only, and the
control is checked against the thresholds and window sizes the e12 run saved.

the streams, the threshold policies and the monitors are those of e21 with uniform audits. a stream is the dismissed
flows of one threshold policy through the windows, every dismissed flow an attack with the FOR of the window's
population at the threshold and audited with probability q. a window holds attacks of its own family only, so the
attacks of the held out family dismissed until the switch off are the count inside its window at the alarm.

monitors, the arms on every stream.
    monitor          FORGE-Monitor with one sequence for the deployment
    monitor_window   a new sequence at every window at delta over the windows, the counterpart of the per window audit
    monitor_any      a sequence started at every window at delta over the windows and the first alarm of any of them,
                     the any start rule of e22, since LITNET has no natural period
    none             the same stream without an alarm, so its leakage is the whole stream's
    window_audit     the per window audit, a Clopper-Pearson bound at delta over the windows read at the window's end
    naive            the running Horvitz Thompson estimate against epsilon times the dismissed count, no bound

counted per unit, case, composition, policy and arm as in e21, and for the window of the held out family the share of
draws alarmed inside it, before it and after it, the dismissed sessions of the window until the alarm, the attacks of
the family dismissed until the switch off against all of them, and the population FOR and the miss rate of the family
at the policy's threshold. per unit the scorer and the certificates, and per unit and window the AUC and the rates at
the offline FORGE thresholds. figures bounds_over_time, for_by_family, alarm_by_family, leakage_by_family and
leakage_by_arm.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from forge import paths, runs
from forge.certify.certificate import (certify, conformal_fnr, family_groups, fixed_sequence, for_from_rates,
                                       grid_from, group_rows, ltt_iid, min_family_size, online_certifier)
from forge.config import load_config
from forge.data import litnet2020
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge
from forge.eval.metrics import auc, share_with_interval
from forge.experiments.litnet_prevalence import ARMS as E12_ARMS
from forge.experiments.litnet_prevalence import (calibration_shares, fit_scorer, in_claim, load_flows, score_rows,
                                                 segment_table, split_flows)
from forge.experiments.monitor_streams import (OFFLINE, POLICIES, POLICY_LABELS, STATE_CODES, _policy_line,
                                               _quantiles, _with_prefix, audited_stream, judge_stream, recalibrate)
from forge.experiments.monitor_sweep import any_start_monitor
from forge.monitor import rule
from forge.monitor.sequence import KINDS, Sequence, Tables
from forge.viz import layout, style
from forge.viz.export import save_figure

CONTROL = "all_known"
CASES = ["unseen", "known"]
CASE_LABELS = {"unseen": "family unseen", "known": "family known"}
ARMS = ["monitor", "monitor_window", "monitor_any", "none", "window_audit", "naive"]
MONITORS = ["monitor", "monitor_window", "monitor_any"]
ARM_LABELS = {"monitor": "FORGE-Monitor", "monitor_window": "FORGE-Monitor, a sequence per window",
              "monitor_any": "FORGE-Monitor, a sequence from every window", "none": "no monitoring",
              "window_audit": "per window audit, Clopper-Pearson", "naive": "running estimate, no bound"}
ARM_SHORT = {"monitor": "FORGE-Monitor", "monitor_window": "FORGE-Monitor per window",
             "monitor_any": "FORGE-Monitor from every window", "window_audit": "per window audit",
             "naive": "running estimate", "none": "no monitoring"}
ARM_MARKERS = {"monitor": "o", "monitor_window": "v", "monitor_any": "D", "window_audit": "s", "naive": "^",
               "none": "x"}
ARM_COLORS = {"monitor": style.COLORS[0], "monitor_window": style.SEQUENTIAL[1], "monitor_any": style.SEQUENTIAL[8],
              "window_audit": style.INK, "naive": style.GRAY, "none": style.GRAY}


def load_source(cfg: dict) -> dict:
    # the e12 run whose split, scorer, certificates, seeds and real windows the units follow, in the runs folder of
    # the storage root
    src = cfg["source"]
    run_dir = paths.runs_dir() / src["experiment"] / src["run"]
    if not (run_dir / "results").is_dir():
        raise FileNotFoundError(f"no results in {run_dir}, the source run must be in the runs folder")
    return {"dir": run_dir, "cfg": load_config(run_dir / "config.yaml"), "summary": runs.load_json(run_dir, "summary"),
            "draws": runs.load_arrays(run_dir, "draws"), "claims": runs.load_table(run_dir, "claims"),
            "cuts": runs.load_table(run_dir, "cuts")}


def window_size(benign: int, attacks, pi: float, shares, subsample: bool) -> tuple[int, bool]:
    """the sessions of a real window of e12 at prevalence pi and whether its benign flows are subsampled.

    the window keeps all benign flows of the test part with the attacks of each family thinned to the mix, unless the
    test part lacks the attacks a family would need, in which case it keeps all of them and has its benign flows
    subsampled, or is left out, size 0, when subsampling is off or a family of the mix has no attack.
    """
    attacks, shares = np.asarray(attacks, dtype=float), np.asarray(shares, dtype=float)
    full = int(round(benign / (1 - pi)))
    need = shares > 0
    if np.any(attacks < full * pi * shares):
        if not subsample or (attacks[need] == 0).any():
            return 0, False
        return int(np.floor((attacks[need] / (pi * shares[need])).min())), True
    return full, False


def e12_seeds(cfg12: dict, n_segments: int):
    # the stream of seeds of the e12 run, one for the calibration sets, one per window, prevalence, mix and arm, and
    # one per natural window
    dep = cfg12["deployment"]
    cells = n_segments * len(dep["prevalences"]) * len(dep["mixes"])
    return np.random.SeedSequence(cfg12["seed"]).spawn(1 + len(E12_ARMS) * cells + n_segments)


def e12_window_seed(cfg12: dict, n_segments: int, i: int, pi: float) -> int:
    # the index of the seed e12 used for the real window of segment i at this prevalence with the window's own mix
    dep = cfg12["deployment"]
    prevalences, mixes = dep["prevalences"], dep["mixes"]
    if pi not in prevalences or "window" not in mixes:
        raise ValueError(f"prevalence {pi} with the window's own mix has no real window in the source run, which has "
                         f"prevalences {prevalences} and mixes {mixes}")
    cells = n_segments * len(prevalences) * len(mixes)
    return 1 + E12_ARMS.index("real") * cells + (i * len(prevalences) + prevalences.index(pi)) * len(mixes) \
        + mixes.index("window")


def units_of(names, segments: pd.DataFrame) -> list[dict]:
    # one unit per name, the families of one capture with benign flows joined as the cuts table of e12 joins them
    out = []
    for name in names:
        match = segments.index[(segments["families"] == name) & ~segments["filtered"]].tolist()
        if len(match) != 1:
            windows = ", ".join(segments.loc[~segments["filtered"], "families"])
            raise ValueError(f"unit {name} names no window of a capture with benign flows, the windows are {windows}")
        out.append({"name": name, "families": name.split("+"), "segment": int(match[0])})
    if len({u["name"] for u in out}) != len(out):
        raise ValueError("a unit is listed twice")
    return out


def file_name(unit: str) -> str:
    return unit.replace("+", "_")


def unseen_metrics(r: dict, stream: dict, w: int) -> dict:
    """what the alarm of an arm means for the window w of the held out family, per draw.

    the attacks of the family dismissed until the switch off are those of its window up to the alarm, all of them
    without an alarm or with one after the window, none with one before it. the sessions of the window deployed
    before the switch off follow the same rule.
    """
    offsets, attacks = stream["offsets"], stream["attacks"]
    draws = attacks.shape[1]
    cum = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(attacks, axis=0)])
    start, length, before, own = offsets[w], offsets[w + 1] - offsets[w], cum[w], attacks[w]
    alarmed, block = r["alarmed"], r["alarm_block"]
    inside = alarmed & (block == w)
    deployed = np.clip(r["sessions"] - start, 0, length)
    return {"inside": inside, "before": alarmed & (block < w), "after": alarmed & (block > w),
            "detect": np.where(inside, deployed, -1), "leak": np.clip(r["leak"] - before, 0, own), "attacks": own,
            "length": length, "share": np.divide(deployed, length, out=np.ones(draws, dtype=float), where=length > 0)}


def fit_unit(unit: dict, flows: pd.DataFrame, split: dict, segments: pd.DataFrame, cfg12: dict, cal_seed,
             draws: int, run: runs.Run) -> dict:
    """the scorer, the grid, the calibration sets and the certificates of e12 without the attacks of the unit's
    families, and the counts of every window at the grid."""
    sc, c, cc = cfg12["scorer"], cfg12["certify"], cfg12["calibration"]
    eps, delta, pi_max, part_b = c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"]
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    pi_cal = cc["prevalence"]
    label, segment, part, kept, families, code = (split[k] for k in ("label", "segment", "part", "kept", "families",
                                                                     "code"))
    held = np.array([f in unit["families"] for f in families])
    # the attacks of the held out families leave the training, selection and calibration parts and stay in the test
    # parts, the benign flows of their capture stay everywhere
    drop = np.isin(code, np.flatnonzero(held) + 1)
    train = np.flatnonzero((part == 0) & kept & ~drop)
    sel = np.flatnonzero((part == 1) & kept & ~drop)
    booster, enc, sel_loss = fit_scorer(flows, label, train, sel, sc, cfg12["seed"])
    booster.save_model(str(run.results / "models" / f"{file_name(unit['name'])}.txt"),
                       num_iteration=booster.best_iteration)
    used = np.flatnonzero((part >= 1) & kept)
    score = score_rows(booster, enc, flows, used)
    u_part, u_segment, u_code, u_label, u_known = part[used], segment[used], code[used], label[used], ~drop[used]
    sel_mask, cal_mask = (u_part == 1) & u_known, (u_part == 2) & u_known
    sel_auc, cal_auc = auc(score[sel_mask], u_label[sel_mask]), auc(score[cal_mask], u_label[cal_mask])
    grid = grid_from(score[(u_part == 1) & (u_code == 0)], c["grid_size"], c.get("grid_tail", []))

    def counts_of(mask):
        return judge.class_counts(score[mask], u_code[mask], grid, len(families))

    sel_counts, cal_counts = counts_of(sel_mask), counts_of(cal_mask)
    windows = [counts_of((u_part == 3) & (u_segment == g)) for g in segments["segment"]]
    with np.errstate(invalid="ignore"):
        test_auc = [auc(score[(u_part == 3) & (u_segment == g)], u_label[(u_part == 3) & (u_segment == g)])
                    for g in segments["segment"]]
    b_sel, m_sel_family, _, _ = judge.rates(sel_counts)
    b_cal, m_cal_family, _, _ = judge.rates(cal_counts)

    # the calibration sets of e12 over the known families, the held out ones have no share
    n_benign_cal, n_attack_cal = int(cal_counts["sizes"][0]), cal_counts["sizes"][1:]
    shares_cal = calibration_shares(n_attack_cal, pi_cal / (1 - pi_cal) * n_benign_cal, cc["mix"])
    n_cal = int(round(n_benign_cal / (1 - pi_cal)))
    expected = n_cal * pi_cal * shares_cal
    minimum = min_family_size(eps, pi_max, delta - part_b, c["family_floor"])
    groups = np.full(len(families), -1)
    groups[~held] = family_groups(expected[~held], minimum)
    merged = [[f for f, g in zip(families, groups) if g == k] for k in range(groups.max() + 1)]
    if (sel_counts["sizes"][1:][shares_cal > 0] == 0).any():
        raise ValueError("a family of the calibration mix has no attack in the selection parts")
    m_sel = shares_cal @ m_sel_family
    fixed = int(fixed_sequence(for_from_rates(pi_cal, m_sel, b_sel) <= eps, 0))
    beta = float(m_sel[fixed]) if fixed >= 0 else 0.0
    cal = draw_counts(np.random.default_rng(cal_seed), n_cal, pi_cal, shares_cal, b_cal, m_cal_family, draws)
    grouped_below, grouped_n = group_rows(cal.family_below, cal.n_family, groups)
    pooled_below, pooled_n = cal.family_below.sum(1), cal.n_family.sum(1)
    args = (eps, delta, pi_max, part_b)
    ltt_start = {"n": cal.n_benign + pooled_n, **start} if c.get("ltt_guarded_start", False) else {}
    offline = {
        "forge_simplex": certify(cal.benign_below, cal.n_benign, grouped_below, grouped_n, *args, **start)["index"],
        "ltt_iid": ltt_iid(cal.benign_below, pooled_below, eps, delta, **ltt_start)["index"],
        "conformal_fnr": conformal_fnr(pooled_below, pooled_n, beta),
        "fixed_sel": np.full(draws, fixed),
    }
    offline = {p: np.asarray(index, dtype=np.int64) for p, index in offline.items()}
    forge_index = offline["forge_simplex"]
    forge_hit = forge_index >= 0

    def at_forge(rate: np.ndarray) -> np.ndarray:
        if not forge_hit.any():
            return np.full(rate.shape[:-1], np.nan)
        return rate[..., forge_index[forge_hit]].mean(-1)

    return {
        "name": unit["name"], "families": unit["families"], "held": held, "segment": unit.get("segment", -1),
        "train_flows": int(len(train)), "train_attacks": int(label[train].sum()),
        "best_iteration": int(booster.best_iteration), "selection_loss": sel_loss, "selection_auc": float(sel_auc),
        "calibration_auc": float(cal_auc), "encoder": enc.to_dict(), "grid": grid, "sel_counts": sel_counts,
        "cal_counts": cal_counts, "windows": windows, "test_auc": test_auc, "shares_cal": shares_cal, "n_cal": n_cal,
        "expected": expected, "minimum": minimum, "groups": groups, "merged": merged, "fixed": fixed, "beta": beta,
        "fixed_benign": float(b_sel[fixed]) if fixed >= 0 else 0.0, "offline": offline,
        "certify_window": online_certifier(grouped_below, grouped_n, *args, **start),
        "offline_benign": {p: float(np.where(index >= 0, b_cal[np.maximum(index, 0)], 0.0).mean())
                           for p, index in offline.items()},
        "forge_benign": float(at_forge(b_cal)),
        "forge_miss": {f: float(v) if not h else np.nan for f, v, h in zip(families, at_forge(m_cal_family), held)},
        "at_forge": at_forge, "counts": np.column_stack([cal.n_benign, cal.n_family]), "cache": {},
    }


def run(cfg: dict, run: runs.Run) -> None:
    m, au, sim = cfg["monitor"], cfg["audits"], cfg["simulation"]
    eps, delta, q = m["epsilon"], m["delta"], au["rate"]
    delta_lower = m["delta_lower"]
    if not 0 < delta_lower < delta:
        raise ValueError(f"delta_lower is {delta_lower}, expected a value between 0 and delta {delta}")
    if m["kind"] not in KINDS:
        raise ValueError(f"unknown kind {m['kind']}")
    if not 0 < q <= 1:
        raise ValueError(f"the audit rate is {q}, expected a probability above 0")
    policies, arms, compositions = list(cfg["policies"]), list(cfg["arms"]), list(cfg["compositions"])
    unknown = (set(policies) - set(POLICIES)) | (set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown policies or arms {', '.join(sorted(unknown))}")
    names = [co["name"] for co in compositions]
    if len(set(names)) != len(names):
        raise ValueError("every composition needs its own name")
    trace = sim["trace"]
    if trace["unit"] not in cfg["units"] or trace["policy"] not in policies or trace["composition"] not in names:
        raise ValueError("the trace names a unit, policy or composition that is not run")

    source = load_source(cfg)
    cfg12, ssum, sdraws, sclaims = (source[key] for key in ("cfg", "summary", "draws", "claims"))
    d, sc, c, cc, dep12 = cfg12["dataset"], cfg12["scorer"], cfg12["certify"], cfg12["calibration"], \
        cfg12["deployment"]
    if c["epsilon"] != eps:
        raise ValueError(f"the monitor's epsilon {eps} differs from the certificate's {c['epsilon']}")
    missing = set(policies) - set(ssum["methods"]) - {"recalibration"}
    if missing:
        raise ValueError(f"the source run has no thresholds of {', '.join(sorted(missing))}")
    draws, pi_cal, shares_split = int(ssum["draws"]), cc["prevalence"], cfg12["split"]["shares"]
    check = bool(cfg["source"].get("check", True))
    filtered = set(d.get("filtered_captures", []))
    manifest = litnet2020.read_manifest(cfg12)
    if manifest["fingerprint"] != ssum["fingerprint"]:
        raise ValueError("the processed table differs from the one of the source run")
    flows = load_flows(cfg12, sc["features"])
    split = split_flows(flows, shares_split, filtered)
    label, segment, part, families = split["label"], split["segment"], split["part"], split["families"]
    run.log(f"source {cfg['source']['experiment']} run {cfg['source']['run']}, {d['name']} {d['processed_version']}, "
            f"fingerprint {manifest['fingerprint']}, {len(flows)} flows, {int(label.sum())} attacks, "
            f"{flows['capture'].nunique()} captures, {draws} draws, families {', '.join(families)}")
    cuts = segment_table(flows, segment, part, filtered)
    for col in ["flows", "benign", "attacks"]:
        if not np.array_equal(cuts[col].to_numpy(), source["cuts"][col].to_numpy()):
            raise ValueError("the split of the source run could not be reproduced")
    run.save_table(cuts, "cuts")
    segments = cuts.drop_duplicates("segment").reset_index(drop=True)
    window_segments = [int(i) for i, g in enumerate(segments.itertuples()) if not g.filtered]
    w_count = len(window_segments)
    labels = [segments["label"][i] for i in window_segments]
    run.log(f"{w_count} windows in capture order, " + ", ".join(labels) + ", the captures "
            + ", ".join(segments.loc[segments["filtered"], "label"]) + " have no benign flows and no window")
    units = units_of(cfg["units"], segments)
    positions = {u["name"]: window_segments.index(u["segment"]) for u in units}
    seeds12 = e12_seeds(cfg12, len(segments))
    a_real, k_window = E12_ARMS.index("real"), dep12["mixes"].index("window")
    real_claims = sclaims[(sclaims["arm"] == "real") & (sclaims["mix"] == "window")]

    seq = Sequence(m["kind"], m["lower_ratios"], m["upper_ratios"], q, q, delta_lower, delta - delta_lower)
    tables = Tables(seq, q, q)
    per_window = Sequence(m["kind"], m["lower_ratios"], m["upper_ratios"], q, q, delta_lower / w_count,
                          (delta - delta_lower) / w_count)
    per_tables = Tables(per_window, q, q)
    restarts = {"monitor_window": (per_tables, np.arange(w_count))}

    # the prevalence of every window in each cell, the window of the unit at the unseen prevalence
    def vector_of(unit, co):
        return tuple(co["unseen"] if w == positions[unit["name"]] else co["known"] for w in range(w_count))

    stream_keys = []
    for unit in units:
        for co in compositions:
            for scorer in (CONTROL, unit["name"]):
                if (scorer, vector_of(unit, co)) not in stream_keys:
                    stream_keys.append((scorer, vector_of(unit, co)))
    stream_seeds = np.random.SeedSequence(cfg["seed"]).spawn(len(stream_keys) * len(policies))
    (run.results / "models").mkdir()

    def window_draw(unit_fit, i, pi):
        # the real window of segment i at prevalence pi under the unit's scorer, drawn with the seed of e12, with the
        # online threshold of every draw, checked against the source run for the control
        key = (i, pi)
        if key in unit_fit["cache"]:
            return unit_fit["cache"][key]
        win = unit_fit["windows"][i]
        b_w, m_w_family, _, own = judge.rates(win)
        test_attacks = win["sizes"][1:]
        n_w, subsampled = window_size(int(win["sizes"][0]), test_attacks, pi, own, dep12["subsample_benign"])
        if n_w == 0:
            raise ValueError(f"the test part of {segments['label'][i]} lacks the attacks of a window at prevalence "
                             f"{pi}")
        rng = np.random.default_rng(seeds12[e12_window_seed(cfg12, len(segments), i, pi)])
        drawn = draw_counts(rng, n_w, pi, own, b_w, m_w_family, draws)
        attack_below = drawn.family_below.sum(1)
        window_below = drawn.benign_below + attack_below
        online = np.asarray(unit_fit["certify_window"](window_below, np.full(draws, n_w))["index"], dtype=np.int64)
        if unit_fit["name"] == CONTROL and check:
            j = dep12["prevalences"].index(pi)
            saved = real_claims[(real_claims["window"] == i) & (real_claims["prevalence"] == pi)]
            if not np.array_equal(online, sdraws["online_index"][a_real, i, j, k_window].astype(np.int64)) \
                    or saved["window_sessions"].nunique() != 1 or int(saved["window_sessions"].iloc[0]) != n_w:
                raise ValueError(f"the real window of {segments['label'][i]} at prevalence {pi} of the source run "
                                 "could not be reproduced")
        m_mix = own @ m_w_family
        out = {"n_w": n_w, "subsampled": subsampled, "window_below": window_below, "attack_below": attack_below,
               "n_attacks": drawn.n_family.sum(1), "curve": for_from_rates(pi, m_mix, b_w), "miss": m_mix,
               "benign": b_w, "online": online}
        unit_fit["cache"][key] = out
        return out

    def monitor_vector(unit_fit, vector, seed_base):
        # the thresholds, the streams and the arms of every policy through the windows at the prevalences of vector
        index = {p: np.full((w_count, draws), -1, dtype=np.int64) for p in policies}
        dismissed = {p: np.zeros((w_count, draws), dtype=np.int64) for p in policies}
        rate = {p: np.zeros((w_count, draws)) for p in policies}
        miss = {p: np.zeros((w_count, draws)) for p in policies}
        benign = {p: np.zeros((w_count, draws)) for p in policies}
        n_window = np.zeros(w_count, dtype=np.int64)
        subsampled = np.zeros(w_count, dtype=bool)
        recal = np.full(draws, unit_fit["fixed"], dtype=np.int64)
        for w, (i, pi) in enumerate(zip(window_segments, vector)):
            dw = window_draw(unit_fit, i, pi)
            n_window[w], subsampled[w] = dw["n_w"], dw["subsampled"]
            for p in policies:
                if p == "forge_online":
                    t = dw["online"]
                elif p == "recalibration":
                    t = recal
                else:
                    t = unit_fit["offline"][p]
                hit, safe = t >= 0, np.maximum(t, 0)
                index[p][w] = t
                dismissed[p][w] = np.where(hit, np.take_along_axis(dw["window_below"], safe[:, None], 1)[:, 0], 0)
                rate[p][w] = np.where(hit, dw["curve"][safe], 0.0)
                miss[p][w] = np.where(hit, dw["miss"][safe], 0.0)
                benign[p][w] = np.where(hit, dw["benign"][safe], 0.0)
            if "recalibration" in policies:
                recal = recalibrate(dw["window_below"], dw["attack_below"], dw["n_w"], dw["n_attacks"], recal, pi_cal,
                                    eps)
        results = {"n_window": n_window, "subsampled": subsampled}
        for pj, p in enumerate(policies):
            rng = np.random.default_rng(stream_seeds[seed_base + pj])
            stream = audited_stream(rng, dismissed[p], rate[p], q)
            out = judge_stream(tables, stream, rate[p], eps, delta, q, restarts)
            if "monitor_any" in arms:
                out["monitor_any"] = any_start_monitor(per_tables, stream, eps)
            results[p] = {"stream": stream, "out": out, "index": index[p], "dismissed": dismissed[p],
                          "rate": rate[p], "miss": miss[p], "benign": benign[p]}
        return results

    unit_rows, window_rows, claim_rows, block_rows, saved, hist = [], [], [], [], {}, {}
    kept_trace = None
    fits = {}
    computed = {}

    def fit_and_log(unit):
        fit = fit_unit(unit, flows, split, segments, cfg12, seeds12[0], draws, run)
        fits[unit["name"]] = fit
        held_out = ", ".join(unit["families"]) if unit["families"] else "nothing"
        run.log(f"unit {unit['name']}, held out {held_out}, trained LightGBM on {fit['train_flows']} flows with "
                f"{fit['train_attacks']} attacks, best round {fit['best_iteration']}, selection loss "
                f"{fit['selection_loss']:.5f}, selection AUC {fit['selection_auc']:.5f}, calibration AUC "
                f"{fit['calibration_auc']:.5f}, grid of {len(fit['grid'])} thresholds, calibration sets of "
                f"{fit['n_cal']} flows at prevalence {pi_cal}, families "
                + ", ".join("+".join(f) for f in fit["merged"]) + ", refused "
                + ", ".join(f"{p} {np.mean(index < 0):.3f}" for p, index in fit["offline"].items())
                + ", benign share of the calibration parts dismissed "
                + ", ".join(f"{p} {v:.4f}" for p, v in fit["offline_benign"].items())
                + f", fixed threshold benign share {fit['fixed_benign']:.4f} of the selection parts with miss rate "
                f"{fit['beta']:.5f}, which conformal FNR takes as beta")
        if unit["name"] == CONTROL and check:
            for p in OFFLINE:
                if p in policies and not np.array_equal(fit["offline"][p], sdraws[f"{p}_index"].astype(np.int64)):
                    raise ValueError(f"the {p} thresholds of the source run could not be reproduced")
            run.log("control, offline thresholds of the source run reproduced in every calibration set")
        unit_rows.append({
            "unit": unit["name"], "held_out": "+".join(unit["families"]), "unseen_window": positions.get(unit["name"],
                                                                                                        -1),
            "train_flows": fit["train_flows"], "train_attacks": fit["train_attacks"],
            "best_iteration": fit["best_iteration"], "selection_loss": fit["selection_loss"],
            "selection_auc": fit["selection_auc"], "calibration_auc": fit["calibration_auc"],
            "grid_thresholds": len(fit["grid"]), "calibration_sessions": fit["n_cal"],
            "families": ", ".join("+".join(f) for f in fit["merged"]), "fixed_index": fit["fixed"],
            "beta": fit["beta"], "fixed_benign_dismissed": fit["fixed_benign"],
            **{f"{p}_refused": float(np.mean(index < 0)) for p, index in fit["offline"].items()},
            **{f"{p}_benign_dismissed": v for p, v in fit["offline_benign"].items()},
            "forge_benign_dismissed": fit["forge_benign"],
            **{f"forge_miss_{f}": v for f, v in fit["forge_miss"].items()},
        })
        for i, g in enumerate(segments.itertuples()):
            win = fit["windows"][i]
            test_attacks = win["sizes"][1:]
            if g.filtered:
                b_w, m_w_family, own = np.full(len(fit["grid"]), np.nan), None, test_attacks / test_attacks.sum()
            else:
                b_w, m_w_family, _, own = judge.rates(win)
            miss = (np.where(test_attacks > 0, fit["at_forge"](m_w_family), np.nan) if m_w_family is not None
                    else np.full(len(families), np.nan))
            window_rows.append({
                "unit": unit["name"], "window": i, "position": window_segments.index(i) if not g.filtered else -1,
                "segment": g.label, "families": g.families, "filtered": bool(g.filtered),
                "unseen": g.families == unit["name"], "in_claim": in_claim(own, fit["shares_cal"], fit["groups"]),
                "test_flows": int(win["sizes"].sum()), "benign": int(win["sizes"][0]),
                "attacks": int(test_attacks.sum()), "auc": float(fit["test_auc"][i]),
                "forge_benign_dismissed": float(fit["at_forge"](b_w)) if not g.filtered else np.nan,
                **{f"forge_miss_{f}": float(v) for f, v in zip(families, miss)},
            })
        prefix = f"{file_name(unit['name'])}|"
        hist[prefix + "grid"] = fit["grid"]
        hist[prefix + "shares"] = fit["shares_cal"]
        hist[prefix + "groups"] = fit["groups"]
        for part_name, counts in [("selection", fit["sel_counts"]), ("calibration", fit["cal_counts"])]:
            for key in ("benign_below", "family_below", "sizes"):
                hist[f"{prefix}{part_name}_{key}"] = counts[key]
        for key in ("benign_below", "family_below", "sizes"):
            hist[f"{prefix}window_{key}"] = np.stack([w[key] for w in fit["windows"]])
        for p, index in fit["offline"].items():
            hist[f"{prefix}{p}_index"] = index
        hist[prefix + "counts"] = fit["counts"]
        return fit

    def cell_rows(res, unit, case, co, vector):
        # the rows of claims and blocks of one unit, case and composition, and the saved arrays of every policy
        w_u = positions[unit["name"]]
        n_window, subsampled = res["n_window"], res["subsampled"]
        total_windows = int(n_window.sum())
        for p in policies:
            r_p = res[p]
            stream, out = r_p["stream"], r_p["out"]
            dismissed, rate, miss = r_p["dismissed"], r_p["rate"], r_p["miss"]
            n_end = stream["offsets"][-1]
            held = dismissed > 0
            for w in range(w_count):
                h = held[w]
                block_rows.append({
                    "unit": unit["name"], "case": case, "composition": co["name"], "policy": p, "window": w,
                    "segment": labels[w], "unseen": w == w_u, "prevalence": vector[w],
                    "window_sessions": int(n_window[w]), "benign_subsampled": bool(subsampled[w]),
                    "refused_rate": float(np.mean(~h)),
                    "threshold_mean": float(r_p["index"][w][h].mean()) if h.any() else np.nan,
                    "dismissed_share_mean": float((dismissed[w] / n_window[w]).mean()),
                    "for_true_mean": float(rate[w][h].mean()) if h.any() else np.nan,
                    "for_true_median": float(np.median(rate[w][h])) if h.any() else np.nan,
                    "miss_mean": float(miss[w][h].mean()) if h.any() else np.nan,
                    "benign_dismissed_mean": float(r_p["benign"][w][h].mean()) if h.any() else np.nan,
                    "violated_rate": float(np.mean(h & (rate[w] > eps))),
                    "attacks_mean": float(stream["attacks"][w].mean()),
                    "realized_for_mean": float(np.mean(stream["attacks"][w][h] / dismissed[w][h])) if h.any()
                    else np.nan,
                    **{f"state_{name}": float(np.mean(out["monitor"]["state"][w] == code))
                       for name, code in STATE_CODES.items()},
                    "window_audit_alarm_rate": float(out["window_audit"]["block_alarm"][w].mean()),
                    "window_audit_certified_rate": float(out["window_audit"]["block_certified"][w].mean()),
                })
            onset = {"onset_true_rate": float(np.mean(out["onset_true"] >= 0)),
                     **_with_prefix("onset_true_window", _quantiles(out["onset_true"] + 1, out["onset_true"] >= 0)),
                     "onset_realized_rate": float(np.mean(out["onset_realized"] >= 0)),
                     **_with_prefix("onset_realized_window",
                                    _quantiles(out["onset_realized"] + 1, out["onset_realized"] >= 0))}
            ones = np.ones(draws, dtype=bool)
            h_u = held[w_u]
            unseen = {"unseen_segment": labels[w_u], "unseen_sessions": int(n_window[w_u]),
                      "unseen_subsampled": bool(subsampled[w_u]),
                      **_with_prefix("unseen_dismissed", _quantiles(dismissed[w_u], ones)),
                      "unseen_for_true_mean": float(rate[w_u][h_u].mean()) if h_u.any() else np.nan,
                      "unseen_miss_mean": float(miss[w_u][h_u].mean()) if h_u.any() else np.nan,
                      "unseen_violated_rate": float(np.mean(h_u & (rate[w_u] > eps)))}
            key = f"{unit['name']}|{case}|{co['name']}|{p}|"
            saved[key + "attacks"] = stream["attacks"]
            saved[key + "dismissed"] = dismissed
            saved[key + "for_true"] = rate.astype(np.float32)
            for arm in arms:
                r = out[arm]
                alarmed = r["alarmed"]
                um = unseen_metrics(r, stream, w_u)
                row = {
                    "unit": unit["name"], "case": case, "composition": co["name"], "policy": p, "arm": arm,
                    "draws": draws, "refused_rate": float(np.mean(n_end == 0)),
                    "stream_sessions_mean": float(n_end.mean()), "alarm_rate": float(alarmed.mean()),
                    "false_alarm_rate": float(r["false_alarm"].mean()),
                    **_with_prefix("alarm_window", _quantiles(r["alarm_block"] + 1, alarmed)),
                    **_with_prefix("alarm_session", _quantiles(r["alarm_pos"], alarmed)),
                    **onset,
                    **_with_prefix("leak", _quantiles(r["leak"], ones)), "leak_mean": float(r["leak"].mean()),
                    **_with_prefix("excess", _quantiles(r["leak"] - eps * r["sessions"], ones)),
                    **_with_prefix("sessions", _quantiles(r["sessions"], ones)),
                    "dismissed_share_mean": float((r["sessions"] / total_windows).mean()),
                    **_with_prefix("realized_for", _quantiles(np.divide(r["leak"], r["sessions"],
                                                                        out=np.zeros(draws), where=r["sessions"] > 0),
                                                              r["sessions"] > 0)),
                    "audits_mean": float(out["audits"].mean()),
                    **unseen,
                    "alarm_in_unseen_rate": float(um["inside"].mean()),
                    "alarm_before_unseen_rate": float(um["before"].mean()),
                    "alarm_after_unseen_rate": float(um["after"].mean()),
                    **_with_prefix("detect_sessions", _quantiles(um["detect"], um["inside"])),
                    **_with_prefix("unseen_leak", _quantiles(um["leak"], ones)),
                    "unseen_leak_mean": float(um["leak"].mean()),
                    **_with_prefix("unseen_attacks", _quantiles(um["attacks"], ones)),
                    "unseen_deployed_share_mean": float(um["share"].mean()),
                }
                if arm in MONITORS:
                    either = r["miss_lower"] | r["miss_upper"]
                    row.update({
                        "miscover": int(either.sum()), **share_with_interval(int(either.sum()), draws, "miscover_"),
                        "miscover_lower_rate": float(r["miss_lower"].mean()),
                        "miscover_upper_rate": float(r["miss_upper"].mean()),
                        "sequences": int(r.get("sequences", 1)),
                        "miscover_sequence_rate": float(r["failed_sequences"].mean()) if "failed_sequences" in r
                        else float(either.mean()),
                        **_with_prefix("bound", _quantiles(r["bound"], alarmed & (r["bound"] >= 0))),
                        "over_bound_rate": float(r["over_bound"].sum() / max(alarmed.sum(), 1)),
                        **{f"windows_{name}_mean": float((r["state"] == code).sum(0).mean())
                           for name, code in STATE_CODES.items()},
                    })
                if arm == "window_audit":
                    row.update({"window_false_alarm_rate": float(r["block_false_alarm"].mean()),
                                "windows_certified_mean": float(r["block_certified"].sum(0).mean())})
                claim_rows.append(row)
                saved[key + f"{arm}_alarm_pos"] = r["alarm_pos"]
                saved[key + f"{arm}_leak"] = r["leak"]
                saved[key + f"{arm}_unseen_leak"] = um["leak"]
                if arm in MONITORS:
                    saved[key + f"{arm}_state"] = r["state"]
                    saved[key + f"{arm}_bound"] = r["bound"]
            by_arm = {x["arm"]: x for x in claim_rows[-len(arms):]}
            line = (f"{unit['name']} {case}, composition {co['name']}, {p}, the window of the family at prevalence "
                    f"{vector[w_u]} with {unseen['unseen_sessions']} sessions, FOR "
                    f"{unseen['unseen_for_true_mean']:.5f} and miss rate {unseen['unseen_miss_mean']:.4f} at the "
                    "threshold, "
                    f"{np.median(stream['attacks'][w_u]):.0f} of its attacks dismissed without monitoring, "
                    f"population FOR above epsilon in a first window for {onset['onset_true_rate']:.3f} of draws at "
                    f"window {onset['onset_true_window_median']:.0f}")
            for arm in arms:
                if arm == "none":
                    continue
                x = by_arm[arm]
                when = (f"after {x['detect_sessions_median']:.0f} dismissed sessions of the window"
                        if x["alarm_in_unseen_rate"] > 0 else "never inside it")
                line += (f", {ARM_SHORT[arm]} alarm inside the window in {x['alarm_in_unseen_rate']:.4f} of draws, "
                         f"before it in {x['alarm_before_unseen_rate']:.4f}, false in {x['false_alarm_rate']:.4f}, "
                         f"{when}, {x['unseen_leak_median']:.0f} of its attacks dismissed, leakage median "
                         f"{x['leak_median']:.0f}")
                if arm in MONITORS:
                    line += (f", bounds left in {x['miscover_rate']:.4f}, above the bound in "
                             f"{x['over_bound_rate']:.4f} of alarms")
            run.log(line)

    control = fit_and_log({"name": CONTROL, "families": []})
    for scorer, vector in stream_keys:
        if scorer == CONTROL:
            computed[(scorer, vector)] = monitor_vector(control, vector, stream_keys.index((scorer, vector))
                                                        * len(policies))
    control["cache"].clear()
    for unit in units:
        fit = fit_and_log(unit)
        w_u = positions[unit["name"]]
        i_u = unit["segment"]
        own_known = [f for f in unit["families"]]
        run.log(f"{labels[w_u]} under the scorer without {', '.join(own_known)}, AUC {fit['test_auc'][i_u]:.4f} "
                f"against {control['test_auc'][i_u]:.4f} with it, at the offline FORGE thresholds benign share "
                f"dismissed {fit['at_forge'](judge.rates(fit['windows'][i_u])[0]):.4f} against "
                f"{control['at_forge'](judge.rates(control['windows'][i_u])[0]):.4f}, miss rate "
                + ", ".join(f"{f} {fit['at_forge'](judge.rates(fit['windows'][i_u])[1][families.index(f)]):.4f} "
                            f"against "
                            f"{control['at_forge'](judge.rates(control['windows'][i_u])[1][families.index(f)]):.4f}"
                            for f in own_known) + " when known")
        for co in compositions:
            vector = vector_of(unit, co)
            key = (unit["name"], vector)
            computed[key] = monitor_vector(fit, vector, stream_keys.index(key) * len(policies))
            cell_rows(computed[key], unit, "unseen", co, vector)
            cell_rows(computed[(CONTROL, vector)], unit, "known", co, vector)
            if unit["name"] == trace["unit"] and co["name"] == trace["composition"]:
                res = computed[key]
                kept_trace = {**res[trace["policy"]]["out"]["trace"], "window_sessions": res["n_window"],
                              "unseen": w_u}
            computed.pop(key)
        fit["cache"].clear()

    claims = pd.DataFrame(claim_rows)
    run.save_table(claims, "claims")
    run.save_table(pd.DataFrame(block_rows), "blocks")
    run.save_table(pd.DataFrame(unit_rows), "units")
    run.save_table(pd.DataFrame(window_rows), "windows")
    run.save_arrays("draws", **saved)
    run.save_arrays("histograms", **hist)
    run.save_arrays("trace", **kept_trace)
    exceeds = {}
    for arm in [a for a in MONITORS if a in arms]:
        g = claims[claims["arm"] == arm]
        worst = g.loc[g["miscover_rate"].idxmax()]
        exceeds[arm] = bool((g["miscover_low"] > delta).any())
        run.log(f"{ARM_SHORT[arm]}, largest share of draws leaving the bounds {worst['miscover_rate']:.4f} on "
                f"{worst['policy']} with {worst['unit']} {worst['case']} at composition {worst['composition']}, 95% "
                f"interval {worst['miscover_low']:.4f} to {worst['miscover_high']:.4f}, delta {delta}, above delta "
                f"beyond simulation noise {exceeds[arm]}, largest share of alarms above the bound of Theorem 3 "
                f"{g['over_bound_rate'].max():.4f}")
    run.save_json({
        "source": {**cfg["source"], "fingerprint": ssum["fingerprint"], "dataset": ssum["dataset"],
                   "processed_version": ssum["processed_version"], "draws": draws, "segments": ssum["segments"],
                   "families": families, "split_shares": shares_split},
        "epsilon": eps, "delta": delta, "delta_lower": delta_lower, "kind": m["kind"],
        "lower_ratios": m["lower_ratios"], "upper_ratios": m["upper_ratios"], "audit_rate": q,
        "alarm_floor": rule.alarm_floor(eps), "calibration_prevalence": pi_cal, "windows": labels,
        "units": [{"name": u["name"], "families": u["families"], "window": positions[u["name"]],
                   "label": labels[positions[u["name"]]]} for u in units],
        "control": CONTROL, "cases": CASES, "policies": policies, "arms": arms, "compositions": compositions,
        "trace": trace, "source_checked": check, "miscoverage_exceeds_delta": exceeds,
        "restart_sequences": {"monitor_window": w_count, "monitor_any": w_count},
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _arm_line(arm: str) -> dict:
    return {"color": ARM_COLORS[arm], "marker": ARM_MARKERS[arm], "label": ARM_LABELS[arm]}


def _unit_axis(ax, summary: dict) -> None:
    units = summary["units"]
    ax.set_xticks(np.arange(len(units)))
    ax.set_xticklabels([u["label"] for u in units], rotation=30, ha="right")


def _per_unit(table: pd.DataFrame, summary: dict, column: str) -> np.ndarray:
    # the value of one column in the order of the units of the summary, nan where a unit has no row
    return table.set_index("unit").reindex([u["name"] for u in summary["units"]])[column].to_numpy(dtype=float)


def fig_bounds_over_time(run_dir) -> plt.Figure:
    style.use()
    t = runs.load_arrays(run_dir, "trace")
    summary = runs.load_json(run_dir, "summary")
    eps = summary["epsilon"]
    scale = 1e6
    color = style.color_for("forge_simplex")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.6))
    # the count is known at the audited attacks and at the window ends, the bounds move at the audited attacks only,
    # and the window of the held out family carries a band
    x = np.concatenate([t["ev_pos"], t["ends"]]) / scale
    y = np.concatenate([t["ev_count"], t["cum_attacks"]])
    order = np.argsort(x, kind="stable")
    ax.plot(x[order], y[order], color=style.INK, linewidth=1, label="attacks among the dismissed flows")
    pos = t["ev_pos"] / scale
    n_end = t["n_end"] / scale
    ax.step(np.r_[pos, n_end], np.r_[t["lcb"], t["lcb"][-1]], where="post", color=color, linewidth=1.2,
            label="lower bound")
    ax.step(np.r_[pos, n_end], np.r_[t["ucb"], t["ucb"][-1]], where="post", color=color, linestyle="--",
            linewidth=1.2, label="upper bound")
    ax.plot([0, n_end], [0, eps * t["n_end"]], color=style.GRAY, linestyle=":", linewidth=1,
            label=r"$\varepsilon$ times the dismissed flows")
    ends = t["ends"] / scale
    for e in ends[:-1]:
        ax.axvline(e, color=style.GRAY, linestyle=":", linewidth=0.6)
    w = int(t["unseen"])
    band = ax.axvspan(0 if w == 0 else ends[w - 1], ends[w], color=style.GRAY, alpha=0.15, linewidth=0)
    if t["alarm_pos"] >= 0:
        ax.axvline(t["alarm_pos"] / scale, color=color, linestyle="-.", linewidth=0.8)
    ax.set_xlim(0, n_end)
    ax.set_ylim(0, 1.1 * max(float(t["ucb"].max()), float(y.max()), 1.0))
    ax.set_xlabel("millions of dismissed flows")
    ax.set_ylabel("attacks")
    layout.legend_below(fig, ax, ["attacks among the dismissed flows", "lower bound", "upper bound"],
                        [(band, "window of the held out family")], ncol=2)
    return fig


def fig_for_by_family(run_dir) -> plt.Figure:
    style.use()
    blocks = runs.load_table(run_dir, "blocks")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    fig, axes = plt.subplots(1, len(compositions), figsize=(style.WIDTH_DOUBLE, 2.8), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]
    x = np.arange(len(summary["units"]))
    floor = summary["epsilon"] / 100
    # the population FOR of the family's window at the policy's threshold with the family unseen, filled, and known,
    # hollow, a dismissed share of zero has no rate
    data = blocks[blocks["unseen"]]
    for ax, co, letter in zip(axes, compositions, "abcdef"):
        g = data[data["composition"] == co["name"]]
        for k, p in enumerate(policies):
            line = _policy_line(p)
            xs = layout.dodge(x, k, len(policies), 0.1)
            for case, face in (("unseen", line["color"]), ("known", "white")):
                h = g[(g["policy"] == p) & (g["case"] == case)]
                ax.plot(xs, np.maximum(_per_unit(h, summary, "for_true_mean"), floor), linestyle="none",
                        markersize=4, markerfacecolor=face, **line)
        ax.axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
        ax.set_yscale("log")
        ax.set_xlabel(f"held out family, its window at prevalence {style.percent(co['unseen'])}")
        _unit_axis(ax, summary)
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("FOR of the window at the threshold")
    hollow = Line2D([], [], color=style.GRAY, marker="o", markerfacecolor="white", linestyle="none")
    layout.legend_below(fig, axes[0], [POLICY_LABELS[p] for p in policies], [(hollow, "family known")])
    return fig


def fig_alarm_by_family(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    fig, axes = plt.subplots(2, len(compositions), figsize=(style.WIDTH_DOUBLE, 4.4), sharex=True, squeeze=False)
    x = np.arange(len(summary["units"]))
    data = claims[(claims["arm"] == "monitor") & (claims["case"] == "unseen")]
    # top, the share of draws with the alarm inside the window of the held out family, filled, and before it, hollow.
    # bottom, the dismissed sessions of that window until the alarm with their 5% to 95% range, and the dismissed
    # sessions of the whole window
    for col, co in enumerate(compositions):
        g = data[data["composition"] == co["name"]]
        for k, p in enumerate(policies):
            h = g[g["policy"] == p]
            xs = layout.dodge(x, k, len(policies), 0.1)
            line = _policy_line(p)
            axes[0, col].plot(xs, _per_unit(h, summary, "alarm_in_unseen_rate"), linestyle="none", markersize=4,
                              **line)
            axes[0, col].plot(xs, _per_unit(h, summary, "alarm_before_unseen_rate"), linestyle="none",
                              markersize=4, markerfacecolor="white", color=line["color"], marker=line["marker"])
            layout.points(axes[1, col], xs, np.maximum(_per_unit(h, summary, "detect_sessions_median"), 1),
                          np.maximum(_per_unit(h, summary, "detect_sessions_low"), 1),
                          np.maximum(_per_unit(h, summary, "detect_sessions_high"), 1), linestyle="none", **line)
            axes[1, col].plot(xs, np.maximum(_per_unit(h, summary, "unseen_dismissed_median"), 1),
                              linestyle="none", marker="_", markersize=7, color=style.INK)
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel(f"held out family, its window at prevalence {style.percent(co['unseen'])}")
        _unit_axis(axes[1, col], summary)
    axes[0, 0].set_ylabel("share of draws with an alarm\ninside the window")
    axes[1, 0].set_ylabel("dismissed sessions of the window\nuntil the alarm")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
    hollow = Line2D([], [], color=style.GRAY, marker="o", markerfacecolor="white", linestyle="none")
    tick = Line2D([], [], color=style.INK, marker="_", markersize=7, linestyle="none")
    layout.legend_below(fig, axes[0, 0], [POLICY_LABELS[p] for p in policies],
                        [(hollow, "alarm before the window"), (tick, "dismissed sessions of the whole window")])
    return fig


def fig_leakage_by_family(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    fig, axes = plt.subplots(1, len(compositions), figsize=(style.WIDTH_DOUBLE, 2.8), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(summary["units"]))
    data = claims[claims["arm"] == "monitor"]
    # the attacks of the held out family dismissed until the switch off with the family unseen, filled, and known,
    # hollow, and all of them, the leakage without monitoring, as a tick
    for ax, co, letter in zip(axes, compositions, "abcdef"):
        g = data[data["composition"] == co["name"]]
        for k, p in enumerate(policies):
            line = _policy_line(p)
            xs = layout.dodge(x, k, len(policies), 0.1)
            for case, face in (("unseen", line["color"]), ("known", "white")):
                h = g[(g["policy"] == p) & (g["case"] == case)]
                layout.points(ax, xs, np.maximum(_per_unit(h, summary, "unseen_leak_median"), 1),
                              np.maximum(_per_unit(h, summary, "unseen_leak_low"), 1),
                              np.maximum(_per_unit(h, summary, "unseen_leak_high"), 1), linestyle="none",
                              markerfacecolor=face, **line)
            h = g[(g["policy"] == p) & (g["case"] == "unseen")]
            ax.plot(xs, np.maximum(_per_unit(h, summary, "unseen_attacks_median"), 1), linestyle="none", marker="_",
                    markersize=7, color=line["color"])
        ax.set_yscale("log")
        ax.set_xlabel(f"held out family, its window at prevalence {style.percent(co['unseen'])}")
        _unit_axis(ax, summary)
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("attacks of the family dismissed\nuntil the switch off")
    hollow = Line2D([], [], color=style.GRAY, marker="o", markerfacecolor="white", linestyle="none")
    tick = Line2D([], [], color=style.GRAY, marker="_", markersize=7, linestyle="none")
    layout.legend_below(fig, axes[0], [POLICY_LABELS[p] for p in policies],
                        [(hollow, "family known"), (tick, "without monitoring")])
    return fig


def fig_leakage_by_arm(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    arms = [a for a in ARMS if a in summary["arms"]]
    compositions = summary["compositions"]
    policy = summary["trace"]["policy"]
    fig, axes = plt.subplots(1, len(compositions), figsize=(style.WIDTH_DOUBLE, 2.8), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(summary["units"]))
    data = claims[(claims["case"] == "unseen") & (claims["policy"] == policy)]
    # the attacks of the held out family dismissed until the switch off under every arm, for one threshold policy
    for ax, co, letter in zip(axes, compositions, "abcdef"):
        g = data[data["composition"] == co["name"]]
        for k, arm in enumerate(arms):
            h = g[g["arm"] == arm]
            xs = layout.dodge(x, k, len(arms), 0.12)
            layout.points(ax, xs, np.maximum(_per_unit(h, summary, "unseen_leak_median"), 1),
                          np.maximum(_per_unit(h, summary, "unseen_leak_low"), 1),
                          np.maximum(_per_unit(h, summary, "unseen_leak_high"), 1), linestyle="none",
                          **_arm_line(arm))
        ax.set_yscale("log")
        ax.set_xlabel(f"held out family, its window at prevalence {style.percent(co['unseen'])}")
        _unit_axis(ax, summary)
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel(f"attacks of the family dismissed until\nthe switch off, {POLICY_LABELS[policy].split(',')[0]}")
    layout.legend_below(fig, axes[0], [ARM_LABELS[a] for a in arms])
    return fig


FIGURES = {
    "bounds_over_time": fig_bounds_over_time,
    "for_by_family": fig_for_by_family,
    "alarm_by_family": fig_alarm_by_family,
    "leakage_by_family": fig_leakage_by_family,
    "leakage_by_arm": fig_leakage_by_arm,
}
