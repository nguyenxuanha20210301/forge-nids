import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forge import experiments, runs
from forge.certify.certificate import (certify, certify_online, conformal_fnr, counts_below, fixed_sequence,
                                       for_from_rates, group_rows, ltt_iid)
from forge.config import load_config
from forge.data import splits
from forge.data.synthetic import draw_counts
from forge.eval import seeds
from forge.eval.metrics import auc
from forge.experiments import ugr16_prevalence as e13
from forge.paths import configs_dir
from forge.runs import Run

from conftest import UGR16_SIGNAL_WINDOWS

# a looser epsilon and pi_max that a few thousand flows can certify, a short tail of the grid, a floor on the attacks
# of a family that some windows of the fixture reach, the last prevalence above pi_max, and the roles cut at the
# clock of the fixture, four windows of training, two of selection, two of calibration and eight of deployment
CUTS = ["2016-07-31 12:00", "2016-08-01 00:00", "2016-08-01 12:00"]
OVERRIDES = [
    "scorer.rounds=60", "scorer.early_stopping=10", "scorer.params.min_data_in_leaf=20", "certify.epsilon=0.05",
    "certify.pi_max=0.3", "certify.grid_size=50", "certify.grid_tail=[0.995, 0.999]", "calibration.prevalence=0.1",
    "deployment.prevalences=[0.05, 0.2, 0.4]", "deployment.main_prevalence=0.2", "deployment.min_family_attacks=300",
    "simulation.draws=200", f'split.selection_start="{CUTS[0]}"', f'split.calibration_start="{CUTS[1]}"',
    f'split.deployment_start="{CUTS[2]}"',
]
EPS, PI_MAX, PI_CAL, DRAWS, MIN_ATTACKS = 0.05, 0.3, 0.1, 200, 300
ARMS = ["reference", "s2", "real"]
FAMILIES = ["dos", "scan11", "scan44", "nerisbotnet", "anomaly_spam"]
KEPT = ["benign", *FAMILIES]
STARTS = pd.date_range("2016-08-01 12:00", periods=8, freq="6h")


def run_e13(cfg, root):
    run = Run(cfg, root=root)
    module = experiments.get("ugr16_prevalence")
    module.run(cfg, run)
    run.finish()
    return run, module


def cdfs(benign_below, family_below, sizes):
    # cdfs of the benign class and of each family at the grid, and the mix of the families, zeros for a window
    # without attacks
    benign = benign_below / sizes[0]
    family = np.divide(family_below, sizes[1:, None], out=np.zeros(family_below.shape), where=sizes[1:, None] > 0)
    own = sizes[1:] / sizes[1:].sum() if sizes[1:].sum() else np.zeros(len(sizes) - 1)
    return benign, family, own


def roles_of(frame):
    cuts = np.array([np.datetime64(c) for c in CUTS]).astype("datetime64[ms]")
    return np.searchsorted(cuts, frame["te"].to_numpy().astype("datetime64[ms]"), side="right")


def test_role_cuts_assign_roles_and_deployment_windows():
    split = {"selection_start": "2016-08-01 00:00", "calibration_start": "2016-08-04 12:00",
             "deployment_start": "2016-08-08 00:00"}
    cuts = e13.role_cuts(split, "6h")
    assert cuts.dtype == np.dtype("datetime64[ms]") and cuts.tolist() == [
        pd.Timestamp(v).to_pydatetime() for v in split.values()]
    te = pd.to_datetime(["2016-07-27 13:43:21", "2016-07-31 23:59:59", "2016-08-01 00:00:00", "2016-08-04 11:59:59",
                         "2016-08-04 12:00:00", "2016-08-07 23:59:59", "2016-08-08 00:00:00", "2016-08-29 09:27:51"])
    assert e13.assign_roles(te.to_numpy(), cuts).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError, match="not the start of a window"):
        e13.role_cuts({**split, "calibration_start": "2016-08-04 13:00"}, "6h")
    with pytest.raises(ValueError, match="must come after"):
        e13.role_cuts({**split, "deployment_start": "2016-08-04 12:00"}, "6h")
    # the deployment windows that hold a kept flow, numbered from 0, an empty window of the span left out and counted,
    # and the window of every deployment flow, the left out flows of a held window included
    te = pd.to_datetime(["2016-08-07 18:00", "2016-08-08 01:00", "2016-08-08 02:00", "2016-08-08 13:00",
                         "2016-08-08 20:00", "2016-08-08 21:00"])
    ids, table = splits.fixed_windows(te, "6h"), splits.fixed_window_table(te, "6h")
    deployed = np.array([False, True, True, True, True, True])
    kept = np.array([True, True, False, True, False, False])
    window, windows, empty = e13.deployment_windows(ids, table, deployed, kept)
    assert window.tolist() == [-1, 0, 0, 1, -1, -1] and empty == 2
    assert windows["window"].tolist() == [0, 1] and windows["start"].tolist() == [pd.Timestamp("2016-08-08 00:00"),
                                                                                   pd.Timestamp("2016-08-08 12:00")]
    with pytest.raises(ValueError, match="no kept flow falls in the deployment"):
        e13.deployment_windows(ids, table, deployed, np.zeros(6, dtype=bool))


def test_family_codes_keep_the_families_with_calibration_attacks():
    family = pd.Series(pd.Categorical(["benign", "dos", "blacklist", "anomaly_udpscan", "dos", "benign", "scan11"],
                                      categories=["benign", "dos", "scan11", "anomaly_udpscan", "blacklist"]))
    role = np.array([0, 1, 2, 1, 2, 3, 3])
    fam = e13.family_codes(family, role, {"blacklist"})
    # scan11 has no calibration attack and leaves like the blacklist, the udp scan too
    assert fam["families"] == ["dos"] and fam["code"].tolist() == [0, 1, -1, -1, 1, 0, -1]
    assert fam["left"] == {"scan11": "no calibration attack", "anomaly_udpscan": "no calibration attack",
                           "blacklist": "left out"}
    assert e13.unseen_name("anomaly_udpscan") == "unseen_udpscan" and e13.unseen_name("newkind") == "unseen_newkind"
    with pytest.raises(ValueError, match="no labelled family has a calibration attack"):
        e13.family_codes(family, np.zeros(7, dtype=int), {"blacklist"})


def test_e13_tables_figures_and_window_rates(ugr16_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml", OVERRIDES)
    run, module = run_e13(cfg, tmp_path / "runs")
    frame = ugr16_signal.frame
    dep, c, methods = cfg["deployment"], cfg["certify"], cfg["simulation"]["methods"]
    prevalences, mixes = dep["prevalences"], dep["mixes"]
    summary = runs.load_json(run.dir, "summary")
    hist = runs.load_arrays(run.dir, "histograms")
    saved = runs.load_arrays(run.dir, "draws")
    windows = runs.load_table(run.dir, "windows")
    claims = runs.load_table(run.dir, "claims")
    natural = runs.load_table(run.dir, "natural")
    roles = runs.load_table(run.dir, "roles")
    family = frame["family"].astype(str).to_numpy()
    role = roles_of(frame)
    kept = np.isin(family, KEPT)

    # the roles by clock, four windows of training, two of selection and calibration each and eight of deployment,
    # with the kept flows of every class and the flows of every family left out
    assert roles["role"].tolist() == module.ROLES and roles["windows"].tolist() == [4, 2, 2, 8]
    assert roles["start"].tolist() == [pd.Timestamp("2016-07-30 12:00"), *(pd.Timestamp(v) for v in CUTS)]
    assert roles["end"].tolist() == [*(pd.Timestamp(v) for v in CUTS), pd.Timestamp("2016-08-03 12:00")]
    for r, row in enumerate(roles.itertuples()):
        mine = role == r
        assert row.flows == int((mine & kept).sum()) and row.benign == int((mine & (family == "benign")).sum())
        assert row.first_te == frame.loc[mine, "te"].min() and row.last_te == frame.loc[mine, "te"].max()
        for f in [*FAMILIES, "blacklist", "anomaly_udpscan", "anomaly_sshscan"]:
            assert getattr(row, f) == int((mine & (family == f)).sum())
    assert roles.loc[1, "anomaly_udpscan"] == 200 and roles.loc[3, "anomaly_sshscan"] == 1
    assert summary["families_left_out"] == {"anomaly_sshscan": "no calibration attack",
                                            "anomaly_udpscan": "no calibration attack", "blacklist": "left out"}
    assert summary["families"] == [["dos", "scan11"], ["scan44"], ["nerisbotnet"], ["anomaly_spam"]]
    assert summary["roles"]["deployment"] == {"start": "2016-08-01 12:00", "end": "2016-08-03 12:00", "windows": 8,
                                              "flows": int(roles.loc[3, "flows"]), "benign": int(roles.loc[3, "benign"]),
                                              "attacks": int(roles.loc[3, "attacks"])}

    # the deployment windows in time order with the attacks of every family and the flows left out, the last two
    # without attacks, the two windows mixing dos and scan11 outside the claim of the merged group
    assert windows["window"].tolist() == list(range(8)) and windows["start"].tolist() == STARTS.tolist()
    assert (windows["end"] - windows["start"] == pd.Timedelta("6h")).all()
    assert runs.load_table(run.dir, "deployment_windows")["start"].tolist() == STARTS.tolist()
    in_window = splits.fixed_windows(frame["te"], "6h") - splits.fixed_windows(frame["te"], "6h")[role == 3].min()
    for w in windows.itertuples():
        mine = (role == 3) & (in_window == w.window)
        assert w.flows == int((mine & kept).sum()) and w.benign == int((mine & (family == "benign")).sum())
        for f in [*FAMILIES, "blacklist", "anomaly_udpscan", "anomaly_sshscan"]:
            assert getattr(w, f) == int((mine & (family == f)).sum())
        labelled = int((mine & kept & (family != "benign")).sum())
        assert np.isclose(w.natural_prevalence, labelled / w.flows)
        left = int((mine & ~kept).sum())
        assert np.isclose(w.prevalence_with_left_out, (labelled + left) / (w.flows + left))
    assert windows["has_attacks"].tolist() == [True] * 6 + [False] * 2
    assert windows["in_claim"].tolist() == [False, True, False, True, True, True, True, True]
    assert summary["windows_without_attacks"] == [6, 7] and summary["empty_windows"] == 0
    assert np.isnan(windows["auc"].iloc[6]) and (windows["auc"][:6] > 0.9).all()

    # scores of the selection, calibration and deployment roles in table order, without the blacklist, the udp scan
    # and the ssh scan, with the role and the deployment window of every flow
    used = np.flatnonzero((role >= 1) & kept)
    scores = runs.load_arrays(run.dir, "scores")
    assert scores["row"].tolist() == used.tolist() and scores["score"].dtype == np.float32
    assert np.array_equal(scores["role"], role[used]) and scores["roles"].tolist() == module.ROLES
    assert np.array_equal(scores["window"], np.where(role[used] == 3, in_window[used], -1))
    assert scores["families"][scores["family"]].tolist() == family[used].tolist()
    for w in windows.itertuples():
        mine = scores["window"] == w.window
        assert np.isclose(w.auc, auc(scores["score"][mine], scores["family"][mine] > 0), equal_nan=True)
    scorer = runs.load_json(run.dir, "scorer")
    assert scorer["features"] == cfg["scorer"]["features"] and "sp_range" in scorer["encoder"]["numeric"]
    assert scorer["train_flows"] == int(((role == 0) & kept).sum())
    assert (run.results / "models" / "scorer.txt").stat().st_size > 0

    # the calibration part at prevalence 0.1, every family capped at one count, so scan11 keeps all its calibration
    # attacks and is pooled with dos, since its expected count is below the floor of the certificate
    families = hist["families"].tolist()
    assert families == FAMILIES
    n_benign, n_attack = hist["calibration_sizes"][0], hist["calibration_sizes"][1:]
    assert n_attack.tolist() == roles.loc[2, FAMILIES].tolist() and n_benign == roles.loc[2, "benign"]
    budget = PI_CAL / (1 - PI_CAL) * n_benign
    shares_cal = module.calibration_shares(n_attack, budget, "balanced")
    assert np.allclose(hist["shares"], shares_cal)
    whole = np.isclose(shares_cal * budget, n_attack)
    assert whole.tolist() == [False, True, True, True, False]
    assert np.allclose(shares_cal[~whole] * budget, (shares_cal * budget).max())
    n_cal = summary["calibration_sessions"]
    assert n_cal == round(n_benign / (1 - PI_CAL))
    assert np.allclose(list(summary["calibration_expected_attacks"].values()), n_cal * PI_CAL * shares_cal)
    assert hist["groups"].tolist() == [0, 0, 1, 2, 3]

    # the baselines tuned on the selection part read it at the calibration mix
    b_sel, m_sel, _ = cdfs(hist["selection_benign_below"], hist["selection_family_below"], hist["selection_sizes"])
    fixed = int(fixed_sequence(for_from_rates(PI_CAL, shares_cal @ m_sel, b_sel) <= EPS, 0))
    assert fixed >= 0 and summary["fixed_index"] == fixed
    assert np.isclose(summary["beta"], (shares_cal @ m_sel)[fixed])

    # the calibration sets and every certificate follow from the saved counts and the documented streams, one for the
    # calibration sets, one for each window, prevalence and mix of each arm, then one for each natural window
    b_cal, m_cal, _ = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"], hist["calibration_sizes"])
    cells = len(windows) * len(prevalences) * len(mixes)
    stream_seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(ARMS) * cells + len(windows))
    cal = draw_counts(np.random.default_rng(stream_seeds[0]), n_cal, PI_CAL, shares_cal, b_cal, m_cal, DRAWS)
    assert np.array_equal(saved["counts"], np.column_stack([cal.n_benign, cal.n_family]))
    grouped = group_rows(cal.family_below, cal.n_family, hist["groups"])
    pooled_below, pooled_n = cal.family_below.sum(1), cal.n_family.sum(1)
    args = (EPS, c["delta"], PI_MAX, c["delta_benign"])
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    simplex = certify(cal.benign_below, cal.n_benign, *grouped, *args, **start)
    pooled = certify(cal.benign_below, cal.n_benign, pooled_below[:, None], pooled_n[:, None], *args, **start)
    assert np.array_equal(saved["forge_simplex_index"], simplex["index"])
    assert np.array_equal(saved["forge_pooled_index"], pooled["index"])
    assert c["ltt_guarded_start"]
    ltt = ltt_iid(cal.benign_below, pooled_below, EPS, c["delta"], n=cal.n_benign + cal.n_family.sum(1), **start)
    assert np.array_equal(saved["ltt_iid_index"], ltt["index"])
    assert np.array_equal(saved["conformal_fnr_index"], conformal_fnr(pooled_below, pooled_n, summary["beta"]))
    assert (saved["fixed_sel_index"] == fixed).all()
    assert (pooled["index"] >= simplex["index"]).all() and (pooled["index"] > simplex["index"]).any()

    def population(i, arm):
        b_w, m_w, own = cdfs(hist["window_benign_below"][i], hist["window_family_below"][i], hist["window_sizes"][i])
        return {"reference": (b_cal, m_cal), "s2": (b_w, m_cal), "real": (b_w, m_w)}[arm], own

    # a real window at the calibration mix needs attacks of every family, which the first window alone holds, a
    # window without attacks has reference and S2 windows at the calibration mix only
    skipped = [[i, pi, "calibration"] for i in range(1, 6) for pi in prevalences]
    assert summary["skipped"] == skipped
    without = claims[claims["window"].isin([6, 7])]
    assert set(without["arm"]) == {"reference", "s2"} and set(without["mix"]) == {"calibration"}
    # the two windows without attacks lack the real arm at both mixes and the window mix in the two other arms
    absent = 2 * len(prevalences) * len(mixes) + 2 * len(prevalences) * (len(ARMS) - 1)
    assert len(claims) == len(methods) * (len(ARMS) * cells - absent - len(skipped))
    assert (claims["start"] == STARTS[claims["window"]].to_numpy()).all()
    certified = {}
    for row in claims.itertuples():
        i, j, k = row.window, prevalences.index(row.prevalence), mixes.index(row.mix)
        (b, m_family), own = population(i, row.arm)
        shares = own if row.mix == "window" else shares_cal
        m = shares @ m_family
        sizes = hist["window_sizes"][i]
        full = round(sizes[0] / (1 - row.prevalence))
        attacks = sizes[1:] if row.arm == "real" else n_attack
        # a real window short of attacks keeps them all and has its benign flows subsampled
        if row.arm == "real" and (attacks < full * row.prevalence * shares).any():
            need = shares > 0
            assert row.benign_subsampled
            assert row.window_sessions == int(np.floor((attacks[need] / (row.prevalence * shares[need])).min()))
        else:
            assert not row.benign_subsampled and row.window_sessions == full
        assert row.in_claim == module.in_claim(shares, shares_cal, hist["groups"])
        assert row.few_attacks == bool((attacks[shares > 0] < MIN_ATTACKS).any())
        rate = for_from_rates(row.prevalence, m, b)
        share = row.prevalence * m + (1 - row.prevalence) * b
        if row.method in module.PER_WINDOW:
            index = saved[f"{module.PER_WINDOW[row.method]}_index"][ARMS.index(row.arm), i, j, k]
        else:
            index = saved[f"{row.method}_index"]
        hit = index >= 0
        certified.setdefault((row.arm, row.method, row.prevalence, row.mix), {})[i] = rate[index[hit]]
        assert np.isclose(row.rate, np.mean(np.where(hit, rate[np.maximum(index, 0)], 0.0) > EPS))
        assert np.isclose(row.dismissed_mean, np.where(hit, share[np.maximum(index, 0)], 0.0).mean())
        assert np.isclose(row.refused, np.mean(~hit))
        for name, at in [("best_at_prevalence", row.prevalence), ("best_up_to_pi_max", PI_MAX)]:
            safe = np.flatnonzero(for_from_rates(at, m, b) <= EPS)
            best = safe[-1] if len(safe) else -1
            assert getattr(row, f"{name}_index") == best
            assert np.isclose(getattr(row, f"{name}_dismissed"), share[best] if best >= 0 else 0.0)
        if row.arm == "real":
            assert row.best_every_mix_index == -1 and np.isnan(row.best_every_mix_dismissed)
        else:
            groups = hist["groups"]
            m_groups = [shares_cal[groups == g] @ m_family[groups == g] / shares_cal[groups == g].sum()
                        for g in range(groups.max() + 1)]
            safe = np.flatnonzero(np.all([for_from_rates(PI_MAX, mg, b) <= EPS for mg in m_groups], axis=0))
            every = safe[-1] if len(safe) else -1
            assert row.best_every_mix_index == every
            assert np.isclose(row.best_every_mix_dismissed, share[every] if every >= 0 else 0.0)
            assert not row.in_claim or every <= row.best_up_to_pi_max_index
    # the spam surge at prevalence 0.4 is short of attacks, so its benign flows are subsampled
    surge = claims[(claims["arm"] == "real") & (claims["window"] == 4) & (claims["prevalence"] == 0.4)]
    assert surge["benign_subsampled"].all() and not claims.loc[claims["arm"] != "real", "benign_subsampled"].any()
    assert claims.loc[claims["mix"] == "calibration", "in_claim"].all()
    assert not claims.loc[(claims["mix"] == "window") & claims["window"].isin([0, 2]), "in_claim"].any()

    # the online certificates of some windows of each arm, a subsampled real window among them, drawn again
    for arm, i, j, k in [("reference", 0, 1, 0), ("reference", 7, 2, 1), ("s2", 2, 0, 0), ("s2", 6, 1, 1),
                         ("real", 1, 2, 0), ("real", 4, 2, 0), ("real", 0, 1, 1)]:
        (b, m_family), own = population(i, arm)
        row = claims[(claims["arm"] == arm) & (claims["window"] == i) & (claims["prevalence"] == prevalences[j])
                     & (claims["mix"] == mixes[k])].iloc[0]
        stream = 1 + ARMS.index(arm) * cells + (i * len(prevalences) + j) * len(mixes) + k
        window = draw_counts(np.random.default_rng(stream_seeds[stream]), row["window_sessions"], prevalences[j],
                             own if mixes[k] == "window" else shares_cal, b, m_family, DRAWS)
        online = certify_online(window.benign_below + window.family_below.sum(1), row["window_sessions"], *grouped,
                                *args, **start)
        assert np.array_equal(online["index"], saved["online_index"][ARMS.index(arm), i, j, k])
    # a window without an arm or a mix keeps the empty value
    assert (saved["online_index"][2, 6:] == -1).all() and (saved["online_index"][:, 6:, :, 0] == -1).all()
    assert (saved["online_index"][2, 1:6, :, 1] == -1).all()

    # each real window at its own prevalence, all its flows, the surge above pi_max
    assert natural["window"].unique().tolist() == list(range(6))
    assert (natural["start"] == STARTS[natural["window"]].to_numpy()).all()
    for row in natural.itertuples():
        i = row.window
        sizes = hist["window_sizes"][i]
        b_w, m_w, own = cdfs(hist["window_benign_below"][i], hist["window_family_below"][i], sizes)
        pi = sizes[1:].sum() / sizes.sum()
        assert np.isclose(row.prevalence, pi) and row.window_sessions == sizes.sum()
        assert row.up_to_pi_max == (pi <= PI_MAX) and row.in_claim == windows.loc[i, "in_claim"]
        m = own @ m_w
        rate, share = for_from_rates(pi, m, b_w), pi * m + (1 - pi) * b_w
        if row.method == "forge_online":
            window = draw_counts(np.random.default_rng(stream_seeds[1 + len(ARMS) * cells + i]), int(sizes.sum()),
                                 pi, own, b_w, m_w, DRAWS)
            again = certify_online(window.benign_below + window.family_below.sum(1), int(sizes.sum()), *grouped,
                                   *args, **start)
            assert np.array_equal(again["index"], saved["natural_online_index"][i])
            index = saved["natural_online_index"][i]
        elif row.method in module.PER_WINDOW:
            index = saved[f"natural_{module.PER_WINDOW[row.method]}_index"][i]
        else:
            index = saved[f"{row.method}_index"]
        hit = index >= 0
        assert np.isclose(row.rate, np.mean(np.where(hit, rate[np.maximum(index, 0)], 0.0) > EPS))
        assert np.isclose(row.dismissed_mean, np.where(hit, share[np.maximum(index, 0)], 0.0).mean())
    assert natural.loc[natural["window"] == 4, "up_to_pi_max"].eq(False).all()
    assert natural.loc[natural["window"] != 4, "up_to_pi_max"].all()
    assert (saved["natural_online_index"][6:] == -1).all()

    # rates at the offline FORGE thresholds average the calibration draws that certify one
    chosen = saved["forge_simplex_index"][saved["forge_simplex_index"] >= 0]
    for w in windows.itertuples():
        sizes = hist["window_sizes"][w.window]
        b, m, _ = cdfs(hist["window_benign_below"][w.window], hist["window_family_below"][w.window], sizes)
        assert np.isclose(w.forge_benign_dismissed, b[chosen].mean())
        for k, f in enumerate(families):
            assert getattr(w, f) == sizes[k + 1]
            expected = m[k][chosen].mean() if sizes[k + 1] else np.nan
            assert np.isclose(getattr(w, f"forge_miss_{f}"), expected, equal_nan=True)
    forge_cal = summary["forge_calibration"]
    assert np.isclose(forge_cal["benign_dismissed"], b_cal[chosen].mean())
    assert np.allclose([forge_cal["miss"][f] for f in families], m_cal[:, chosen].mean(1))

    # pooled over all windows and over those inside the claim, the windows of a key in the same order as its rows
    by_prevalence = runs.load_table(run.dir, "by_prevalence")
    keys = ["arm", "method", "prevalence", "mix"]
    for subset, source in [("all", claims), ("in_claim", claims[claims["in_claim"]])]:
        table = by_prevalence[by_prevalence["subset"] == subset].set_index(keys).sort_index()
        assert len(table) == source.groupby(keys).ngroups
        for key, g in source.groupby(keys):
            assert table.loc[key, "windows"] == len(g) and table.loc[key, "pairs"] == len(g) * DRAWS
            assert np.isclose(table.loc[key, "violation_rate"], g["rate"].mean())
            assert table.loc[key, "violation_window_max"] == g["rate"].max()
            assert np.isclose(table.loc[key, "dismissed_mean"], g["dismissed_mean"].mean())
            assert np.isclose(table.loc[key, "best_every_mix_dismissed"], g["best_every_mix_dismissed"].mean(),
                              equal_nan=True)
            rates = np.concatenate([certified[key][i] for i in g["window"]])
            assert np.isclose(table.loc[key, "for_median"], np.median(rates), equal_nan=True) or not len(rates)
    covered = claims[(claims["prevalence"] <= PI_MAX) & claims["in_claim"]]
    for (arm, method, mix), g in covered.groupby(["arm", "method", "mix"]):
        top = summary["largest_violation_up_to_pi_max"][arm][method][mix]
        assert top["rate"] == g["rate"].max() and top["prevalence"] <= PI_MAX
        assert top["start"] == f"{STARTS[top['window']]:%Y-%m-%d %H:%M}"
    # the baselines tuned at the calibration prevalence fail above it, FORGE stays within its claim
    main = claims[claims["prevalence"] == dep["main_prevalence"]]
    assert (main.loc[main["method"] == "ltt_iid", "rate"] > 0.5).mean() > 0.5
    assert covered.loc[covered["method"] == "forge_simplex", "rate"].max() <= 0.1
    assert (claims.loc[claims["refused"] == 1, "violations"] == 0).all()
    without_floor = ~claims["method"].str.startswith("forge") & ~claims["method"].isin(module.ESTIMATED)
    assert (claims.loc[without_floor, "floor_violations"] == 0).all()
    # the estimates per window, the natural windows among them, and the saved arrays of the two estimated methods
    estimates = runs.load_table(run.dir, "estimates")
    assert len(estimates[~estimates["natural"]]) == len(claims) // len(methods)
    assert len(estimates[estimates["natural"]]) == len(natural) // len(methods)
    assert (estimates["upper_median"] >= estimates["plugin_median"]).all()
    assert saved["plugin_index"].shape == saved["online_index"].shape and "natural_upper_prevalence" in saved
    assert summary["estimate"]["t_star"] == int(np.argmax(b_sel - shares_cal @ m_sel))

    # the families outside every claim, the share of their scored flows dismissed at the thresholds of every method,
    # the per window methods over the six real windows at their own prevalence
    for f, name, flows in [("anomaly_udpscan", "unseen_udpscan", 200), ("anomaly_sshscan", "unseen_sshscan", 1)]:
        table = runs.load_table(run.dir, name)
        assert table["method"].tolist() == methods and (table["flows"] == flows).all()
        assert summary["unseen"][f]["table"] == name and summary["unseen"][f]["flows"] == flows
        below = hist[f"{name}_below"]
        assert below[-1] <= flows and (np.diff(below) >= 0).all()
        for row in table.itertuples():
            if row.method in module.PER_WINDOW:
                index = (saved["natural_online_index"] if row.method == "forge_online"
                         else saved[f"natural_{module.PER_WINDOW[row.method]}_index"])[:6].ravel()
                assert row.natural_windows == 6
            else:
                index = saved[f"{row.method}_index"]
                assert row.natural_windows == 0
            hit = index >= 0
            share = np.where(hit, below[np.maximum(index, 0)] / flows, 0.0)
            assert row.thresholds == len(index) and np.isclose(row.refused, np.mean(~hit))
            assert np.isclose(row.dismissed_mean, share.mean())
            assert np.isclose(row.dismissed_median, np.median(share[hit]))
            assert np.isclose(summary["unseen"][f]["dismissed_mean"][row.method], share.mean())
    assert summary["unseen"]["anomaly_udpscan"]["flows_by_role"] == {"selection": 200}
    # the udp scan sits between the benign flows and the attacks, so the loose baselines dismiss all of it and FORGE
    # part of it
    udp = runs.load_table(run.dir, "unseen_udpscan").set_index("method")
    assert udp.loc["fixed_sel", "dismissed_mean"] > 0.9 > udp.loc["forge_simplex", "dismissed_mean"] > 0.1
    assert scores["row"].tolist() == used.tolist()

    log = (run.dir / "log.txt").read_text()
    assert ("selection, 2016-07-31 12:00 to 2016-08-01 00:00, 2 windows of 6h, flows from 2016-07-31 12:00:00 to "
            "2016-07-31 23:59:48, kept 9155, benign 8200, attacks dos 270, scan11 40, scan44 135, nerisbotnet 120, "
            "anomaly_spam 390, left out anomaly_sshscan 0, anomaly_udpscan 200, blacklist 35") in log
    assert ("families with calibration attacks dos, scan11, scan44, nerisbotnet, anomaly_spam, left out "
            "anomaly_sshscan no calibration attack, anomaly_udpscan no calibration attack, blacklist left out, 8 "
            "deployment windows hold kept flows") in log
    assert "2 windows hold no attack and have reference and S2 windows at the calibration mix only" in log
    assert f"{len(skipped)} real windows, prevalences and mixes left out" in log
    for arm in ARMS:
        for method in methods:
            assert f"{method}, {module.ARM_LABELS[arm]} in the claim, window mix, FOR above epsilon in " in log
            assert f"{method}, {module.ARM_LABELS[arm]} in the claim, calibration mix, FOR above epsilon in " in log
    best = by_prevalence[(by_prevalence["subset"] == "in_claim") & (by_prevalence["arm"] == "reference")
                         & (by_prevalence["mix"] == "window") & (by_prevalence["method"] == methods[0])
                         & (by_prevalence["prevalence"] == dep["main_prevalence"])].iloc[0]
    assert (f"best thresholds, reference windows in the claim, window mix, mean share of window flows dismissed by the "
            f"loosest threshold with FOR at or below epsilon up to pi_max {best['best_up_to_pi_max_dismissed']:.4f} at "
            f"prevalence {dep['main_prevalence']}, by the loosest safe for every mix the claim covers "
            f"{best['best_every_mix_dismissed']:.4f} at prevalence {dep['main_prevalence']}") in log
    g = main[(main["window"] == 0) & (main["arm"] == "real") & (main["mix"] == "window")]
    assert (f"window 2016-08-01 12:00, real window at prevalence {dep['main_prevalence']}, FOR above epsilon in "
            + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples()) + ", outside the FORGE claim") in log
    g = natural[natural["window"] == 4]
    assert (f"window 2016-08-02 12:00, real window at its own prevalence {g['prevalence'].iloc[0]:.5f}, FOR above "
            "epsilon in " + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples())) in log
    assert "above pi_max" in log.split("window 2016-08-02 12:00, real window at its own prevalence")[1].split("\n")[0]
    assert "window 2016-08-03 00:00, flows 3200, benign 3200, attacks none, left out anomaly_sshscan 0, " \
           "anomaly_udpscan 0, blacklist 13, prevalence 0.00000, AUC nan" in log
    assert "anomaly_udpscan, 200 flows outside every claim, selection 200, share dismissed at the thresholds of " in log

    def drawn(ax, label):
        [line] = [line for line in ax.get_lines() if line.get_label() == label]
        return line.get_ydata()

    # the figures over prevalences pool the windows inside the claim of their arm
    inside = by_prevalence[(by_prevalence["subset"] == "in_claim") & (by_prevalence["mix"] == "window")]
    for name, arm in [("for_by_prevalence", "reference"), ("for_by_prevalence_s2", "s2"),
                      ("for_by_prevalence_real", "real")]:
        fig = module.FIGURES[name](run.dir)
        for method in methods:
            g = inside[(inside["arm"] == arm) & (inside["method"] == method)].sort_values("prevalence")
            assert np.array_equal(drawn(fig.axes[2], module.LABELS[method]), g["dismissed_mean"].to_numpy())
        assert fig.axes[2].get_ylabel() == "window flows dismissed"
        labels = [line.get_label() for line in fig.axes[2].get_lines()]
        assert (module.EVERY_LINE[2] in labels) == (arm != "real")
        assert "fixed threshold, selection part" in [t.get_text() for t in fig.legends[0].get_texts()]
        plt.close(fig)
    # the figure over windows, a band behind each window outside the claim and the dismissed shares of the reference
    # windows in the last row, on a time axis
    fig = module.FIGURES["violation_by_window"](run.dir)
    assert len(fig.axes[0].patches) == 2
    reference = main[(main["arm"] == "reference") & (main["mix"] == "window")]
    for method in methods:
        g = reference[reference["method"] == method].sort_values("window")
        assert np.array_equal(drawn(fig.axes[-1], module.LABELS[method]), g["dismissed_mean"].to_numpy())
    assert np.array_equal(drawn(fig.axes[-1], r"best threshold up to $\pi_{\max}$"),
                          g["best_up_to_pi_max_dismissed"].to_numpy())
    assert fig.axes[-1].get_xlabel() == "start of the window of 6h, 2016"
    plt.close(fig)
    # the drift figure keeps the miss rates of the families with enough attacks, the spam of two windows
    fig = module.FIGURES["drift_by_window"](run.dir)
    spam = drawn(fig.axes[1], "spam")
    enough = windows["anomaly_spam"] >= MIN_ATTACKS
    assert enough.sum() == 2 and np.isfinite(spam).sum() == 2
    assert np.allclose(spam[enough], windows.loc[enough, "forge_miss_anomaly_spam"] / forge_cal["miss"]["anomaly_spam"])
    assert np.isnan(drawn(fig.axes[1], "DoS")).all()
    plt.close(fig)
    names = {p.name for p in run.figures.iterdir()}
    for name in module.FIGURES:
        assert {f"{name}.png", f"{name}.pdf", f"{name}_a4.pdf"} <= names

    # the run pools over seeds like e12, with the start of every window beside its index
    tables = seeds.pool_experiment("e13_ugr16_prevalence", tmp_path / "runs", out=tmp_path / "pooled",
                                   log=lambda s: None)
    assert {"by_prevalence", "claims", "natural", "windows", "roles", "unseen_udpscan", "unseen_sshscan",
            "e1_for_by_prevalence", "e2_by_family", "e2_natural"} <= set(tables)
    assert tables["unseen_udpscan"]["seeds"].tolist() == [1] * len(tables["unseen_udpscan"])
    assert "start" in tables["e2_by_family"] and len(tables["e2_by_family"]) == len(main[main["mix"] == "window"])
    assert len(tables["e1_for_by_prevalence"]) == len(inside[inside["arm"] == "reference"]) + len(
        by_prevalence[(by_prevalence["subset"] == "in_claim") & (by_prevalence["mix"] == "calibration")
                      & (by_prevalence["arm"] == "reference")])

    # the same config and seed give the same results
    again, _ = run_e13(cfg, tmp_path / "again")
    assert runs.load_table(again.dir, "claims").equals(claims)
    assert np.array_equal(runs.load_arrays(again.dir, "scores")["score"], scores["score"])


def test_e13_checks_its_config(ugr16_signal, tmp_path):
    # one folder per case, since runs started in the same second share a run id
    cases = [("deployment.main_prevalence=0.3", "main prevalence 0.3"),
             ("deployment.mixes=[window, dos]", "mixes are"),
             ("simulation.methods=[forge_simplex, oracle]", "unknown methods oracle"),
             ('split.calibration_start="2016-08-01 03:00"', "not the start of a window"),
             ('split.deployment_start="2016-08-01 00:00"', "must come after"),
             ('split.deployment_start="2016-08-04 00:00"', "no kept flow falls in the deployment"),
             ("dataset.time_column=ts", "time_column is ts")]
    for n, (override, message) in enumerate(cases):
        cfg = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml", OVERRIDES + [override])
        with pytest.raises(ValueError, match=message):
            run_e13(cfg, tmp_path / f"runs{n}")


# the udp scan in the selection part, the calibration part and the deployment, with one empty window in the
# deployment span
SEEN = [dict(w) for w in UGR16_SIGNAL_WINDOWS]
SEEN[4] = {**SEEN[4], "anomaly_udpscan": 50}
SEEN[6] = {**SEEN[6], "anomaly_udpscan": 200}
SEEN[9] = {**SEEN[9], "anomaly_udpscan": 150}
SEEN[11] = None


@pytest.mark.parametrize("ugr16_signal", [SEEN], indirect=True)
def test_e13_unseen_family_becomes_a_family_with_calibration_attacks_and_an_empty_window_is_left_out(ugr16_signal,
                                                                                                    tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e13_ugr16_prevalence.yaml",
                      OVERRIDES + ["simulation.methods=[forge_simplex, forge_online]"])
    run, _ = run_e13(cfg, tmp_path / "runs")
    summary = runs.load_json(run.dir, "summary")
    windows = runs.load_table(run.dir, "windows")
    hist = runs.load_arrays(run.dir, "histograms")
    assert hist["families"].tolist() == [*FAMILIES, "anomaly_udpscan"]
    assert summary["families_left_out"] == {"anomaly_sshscan": "no calibration attack", "blacklist": "left out"}
    assert list(summary["unseen"]) == ["anomaly_sshscan"] and not (run.results / "unseen_udpscan.parquet").exists()
    # the window of 2016-08-02 06:00 holds no flow, so seven windows remain, numbered without it
    assert summary["empty_windows"] == 1 and summary["windows"] == 7
    assert windows["start"].tolist() == [s for s in STARTS if s != pd.Timestamp("2016-08-02 06:00")]
    assert windows["anomaly_udpscan"].tolist() == [0, 150, 0, 0, 0, 0, 0]
    scores = runs.load_arrays(run.dir, "scores")
    assert set(scores["window"].tolist()) == set(range(-1, 7))
    assert "1 windows of the deployment span hold none and are left out" in (run.dir / "log.txt").read_text()
