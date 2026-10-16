import matplotlib.pyplot as plt
import numpy as np
import pytest

from forge import experiments, runs
from forge.certify.certificate import (certify, certify_online, conformal_fnr, fixed_sequence, for_from_rates,
                                       group_rows, ltt_iid)
from forge.config import load_config
from forge.data import litnet2020, splits
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge
from forge.eval.metrics import auc
from forge.paths import configs_dir
from forge.runs import Run

from conftest import LITNET_SIGNAL_CAPTURES

# a looser epsilon and pi_max that a few thousand flows can certify, a short tail of the grid, and a floor on the test
# attacks of a family that some windows of the fixture reach. the last prevalence lies above pi_max
OVERRIDES = [
    "scorer.rounds=60", "scorer.early_stopping=10", "scorer.params.min_data_in_leaf=20", "certify.epsilon=0.05",
    "certify.pi_max=0.3", "certify.grid_size=50", "certify.grid_tail=[0.995, 0.999]", "calibration.prevalence=0.1",
    "deployment.prevalences=[0.05, 0.2, 0.4]", "deployment.main_prevalence=0.2", "deployment.min_family_attacks=300",
    "simulation.draws=200",
]
EPS, PI_MAX, PI_CAL, DRAWS, MIN_ATTACKS = 0.05, 0.3, 0.1, 200, 300
ARMS = ["reference", "s2", "real"]
SEGMENTS = ["UDP flood", "Reaper worm", "SYN flood", "spam bot", "Code Red worm", "Smurf", "ICMP flood"]


def run_e12(cfg, root):
    run = Run(cfg, root=root)
    module = experiments.get("litnet_prevalence")
    module.run(cfg, run)
    run.finish()
    return run, module


def cdfs(benign_below, family_below, sizes):
    # cdfs of the benign class and of each family at the grid, and the mix of the families. a window of a filtered
    # capture holds no benign flow
    with np.errstate(divide="ignore", invalid="ignore"):
        benign = benign_below / sizes[0]
    family = np.divide(family_below, sizes[1:, None], out=np.zeros(family_below.shape), where=sizes[1:, None] > 0)
    return benign, family, sizes[1:] / sizes[1:].sum()


def test_attack_rank_parts_give_every_part_its_share_of_the_attacks():
    rng = np.random.default_rng(0)
    attack = rng.random(500) < 0.1
    shares = [0.35, 0.1, 0.2, 0.35]
    part = splits.attack_rank_parts(attack, shares)
    ranks = np.ceil(np.round(np.cumsum(shares) * attack.sum(), 6)).astype(int)
    assert np.bincount(part[attack], minlength=4).tolist() == np.diff(np.r_[0, ranks]).tolist()
    assert (np.diff(part) >= 0).all() and part[0] == 0 and part[-1] == 3
    # the rows after the last attack of a part go with the next part
    last = np.flatnonzero(attack)[ranks[0] - 1]
    assert part[last] == 0 and (part[last + 1:np.flatnonzero(attack)[ranks[0]] + 1] == 1).all()
    # 0.1 + 0.2 is a hair above 0.3 in floating point, which must not move the cut past the third of ten attacks
    ten = np.zeros(30, dtype=bool)
    ten[::3] = True
    assert np.bincount(splits.attack_rank_parts(ten, [0.1, 0.2, 0.7])[ten]).tolist() == [1, 2, 7]
    with pytest.raises(ValueError, match="sum to 1"):
        splits.attack_rank_parts(attack, [0.5, 0.4])
    with pytest.raises(ValueError, match="at least one"):
        splits.attack_rank_parts(np.array([True, True, False]), shares)


def test_capture_parts_cut_families_that_follow_each_other_and_stop_on_skewed_ones():
    # capture a holds family x from second 10 to 58 and y from 120 to 178, capture b both families interleaved
    second = np.r_[np.arange(200), np.arange(200)]
    capture = np.array(["a"] * 200 + ["b"] * 200, dtype=object)
    label = np.zeros(400, dtype=np.int8)
    family = np.full(400, "benign", dtype=object)
    for rows, name in [(np.arange(10, 60, 2), "x"), (np.arange(120, 180, 2), "y"), (np.arange(200, 400, 8), "x"),
                       (np.arange(204, 400, 8), "y")]:
        label[rows], family[rows] = 1, name
    shares = [0.35, 0.1, 0.2, 0.35]
    segment, part = litnet2020.capture_parts(capture, label, family, second, shares)
    # a is cut at the first attack of y, the benign flows between the two families stay with x
    assert (segment[:120] == 0).all() and (segment[120:200] == 1).all() and (segment[200:] == 2).all()
    for g in range(3):
        for name in ["x", "y"]:
            mine = (segment == g) & (family == name)
            assert not mine.any() or (np.bincount(part[mine], minlength=4) > 0).all()
    # in capture c the last x attack and the first y attack share second 50, so it is split as one and x has no test
    # attack
    r = np.arange(200)
    c_family = np.where((r % 2 == 0) & (r <= 100), "x", np.where((r % 2 == 1) & (r >= 101), "y", "benign"))
    c_family = c_family.astype(object)
    c_label = (c_family != "benign").astype(np.int8)
    c_capture = np.array(["c"] * 200, dtype=object)
    assert (litnet2020.capture_segments(c_capture, c_label, c_family, r // 2) == 0).all()
    with pytest.raises(ValueError, match="attacks of x in its parts"):
        litnet2020.capture_parts(c_capture, c_label, c_family, r // 2, shares)
    # a capture out of time order or in two runs stops the split too
    with pytest.raises(ValueError, match="not in time order"):
        litnet2020.capture_parts(capture, label, family, second[::-1], shares)
    with pytest.raises(ValueError, match="not consecutive"):
        litnet2020.capture_parts(np.r_[capture[:100], capture[200:300], capture[100:200], capture[300:]], label,
                                 family, second, shares)


def test_calibration_mix_caps_the_largest_families_and_the_claim_covers_merged_ones_at_their_mix():
    module = experiments.get("litnet_prevalence")
    n = np.array([10, 2000, 50, 1000])
    # the cap is (560 - 10 - 50) / 2 = 250, so the two small families keep all their attacks
    assert np.allclose(module.calibration_shares(n, 560, "balanced") * 560, [10, 250, 50, 250])
    assert np.allclose(module.calibration_shares(n, n.sum(), "balanced"), n / n.sum())
    assert np.allclose(module.calibration_shares(n, 560, "natural"), n / n.sum())
    with pytest.raises(ValueError, match="fewer than"):
        module.calibration_shares(n, 5000, "balanced")
    with pytest.raises(ValueError, match="natural or balanced"):
        module.calibration_shares(n, 560, "flat")
    groups, calibration = np.array([0, 1, 1]), np.array([0.5, 0.3, 0.2])
    assert module.in_claim(np.array([1.0, 0.0, 0.0]), calibration, groups)
    assert not module.in_claim(np.array([0.0, 1.0, 0.0]), calibration, groups)
    assert module.in_claim(calibration, calibration, groups)
    assert module.in_claim(np.array([0.2, 0.48, 0.32]), calibration, groups)
    assert not module.in_claim(np.array([0.2, 0.4, 0.4]), calibration, groups)
    assert litnet2020.port_range([0, 1, 1023, 1024, 49151, 49152, 65535]).tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_every_mix_threshold_covers_each_family_alone_and_merged_groups_at_their_mix():
    # family 0 stands alone, families 1 and 2 form a group at proportions 1 to 3. family 1 alone is unsafe from the
    # third threshold on, but in the group at its mix only from the fourth, while family 0 is unsafe at the last
    m = np.array([[0.0, 0.0, 0.0, 0.0, 0.05], [0.0, 0.0, 0.02, 0.04, 0.04], [0.0, 0.0, 0.0, 0.0, 0.0]])
    b = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    groups, shares = np.array([0, 1, 1]), np.array([0.5, 0.125, 0.375])
    kappa = 1e-3 * 0.9 / (0.999 * 0.1)
    assert 0.02 / 0.6 / 4 < kappa < 0.04 / 0.8 / 4 and 0.02 / 0.6 > kappa
    assert judge.every_mix_threshold(0.1, m, b, groups, shares, 1e-3) == 2
    # with every family alone the second family stops the sequence one threshold earlier
    assert judge.every_mix_threshold(0.1, m, b, np.arange(3), shares, 1e-3) == 1
    assert judge.every_mix_threshold(0.1, np.ones((3, 5)), b, groups, shares, 1e-3) == -1


def test_e12_tables_figures_and_window_rates(litnet_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES)
    run, module = run_e12(cfg, tmp_path / "runs")
    frame = litnet_signal.frame
    dep, c, methods = cfg["deployment"], cfg["certify"], cfg["simulation"]["methods"]
    prevalences, mixes, shares_split = dep["prevalences"], dep["mixes"], cfg["split"]["shares"]
    summary = runs.load_json(run.dir, "summary")
    hist = runs.load_arrays(run.dir, "histograms")
    saved = runs.load_arrays(run.dir, "draws")
    windows = runs.load_table(run.dir, "windows")
    claims = runs.load_table(run.dir, "claims")
    natural = runs.load_table(run.dir, "natural")
    cuts = runs.load_table(run.dir, "cuts")

    # the icmp flood capture is cut at its first ICMP flood attack, since its two families follow each other, which
    # makes seven windows in time order
    assert summary["segments"] == SEGMENTS and windows["segment"].tolist() == SEGMENTS
    first_flood = frame.loc[frame["family"] == "icmp_flood", "te"].min()
    assert frame.loc[frame["family"] == "smurf", "te"].max() < first_flood
    icmp = cuts[cuts["capture"] == "icmp_flood"]
    assert (icmp.loc[icmp["label"] == "Smurf", "last_te"] < first_flood).all()
    assert icmp.loc[icmp["label"] == "ICMP flood", "first_te"].min() == first_flood
    # every part of a window holds its share of the window's attacks, and the parts follow each other in time
    for _, g in cuts.groupby("segment"):
        ranks = np.ceil(np.round(np.cumsum(shares_split) * g["attacks"].sum(), 6)).astype(int)
        assert g["attacks"].tolist() == np.diff(np.r_[0, ranks]).tolist() and g["part"].tolist() == module.PARTS
        assert (g["first_te"].to_numpy()[1:] >= g["last_te"].to_numpy()[:-1]).all()
    assert cuts["flows"].sum() == len(frame) and cuts["filtered"].tolist() == [True] * 4 + [False] * 24

    # scores of the selection, calibration and test parts in table order, without the benign flows of the udp flood
    # capture, whose attacks stay
    segment, part = litnet2020.capture_parts(np.asarray(frame["capture"]), frame["label"].to_numpy(),
                                             np.asarray(frame["family"]), frame["te"].to_numpy(), shares_split)
    kept = (frame["label"] == 1).to_numpy() | (frame["capture"] != "udp_flood").to_numpy()
    used = np.flatnonzero((part >= 1) & kept)
    scores = runs.load_arrays(run.dir, "scores")
    assert scores["row"].tolist() == used.tolist() and scores["score"].dtype == np.float32
    assert np.array_equal(scores["part"], part[used]) and np.array_equal(scores["segment"], segment[used])
    assert scores["families"][scores["family"]].tolist() == frame["family"].astype(str).to_numpy()[used].tolist()
    for w in windows.itertuples():
        test = (scores["segment"] == w.window) & (scores["part"] == 3)
        assert np.isclose(w.auc, auc(scores["score"][test], scores["family"][test] > 0), equal_nan=True)
    assert np.isnan(windows.loc[0, "auc"]) and (windows["auc"][1:] > 0.9).all()
    scorer = runs.load_json(run.dir, "scorer")
    assert scorer["features"] == cfg["scorer"]["features"] and "sp_range" in scorer["encoder"]["numeric"]
    assert (run.results / "models" / "scorer.txt").stat().st_size > 0

    # the calibration parts at prevalence 0.1, every family capped at one count, so the smaller families keep all their
    # calibration attacks, and the families that cannot stand alone pooled into one group
    families = hist["families"].tolist()
    assert families == ["smurf", "icmp_flood", "udp_flood", "syn_flood", "code_red_worm", "spam_bot", "reaper_worm"]
    n_benign, n_attack = hist["calibration_sizes"][0], hist["calibration_sizes"][1:]
    budget = PI_CAL / (1 - PI_CAL) * n_benign
    shares_cal = module.calibration_shares(n_attack, budget, "balanced")
    assert np.allclose(hist["shares"], shares_cal)
    whole = np.isclose(shares_cal * budget, n_attack)
    assert whole.sum() == 3 and np.allclose(shares_cal[~whole] * budget, (shares_cal * budget).max())
    assert summary["families"] == [["smurf"], ["icmp_flood", "spam_bot", "reaper_worm"], ["udp_flood"], ["syn_flood"],
                                   ["code_red_worm"]]
    n_cal = summary["calibration_sessions"]
    assert n_cal == round(n_benign / (1 - PI_CAL))
    assert np.allclose(list(summary["calibration_expected_attacks"].values()), n_cal * PI_CAL * shares_cal)

    # the baselines tuned on the selection parts read them at the calibration mix
    b_sel, m_sel, _ = cdfs(hist["selection_benign_below"], hist["selection_family_below"], hist["selection_sizes"])
    fixed = int(fixed_sequence(for_from_rates(PI_CAL, shares_cal @ m_sel, b_sel) <= EPS, 0))
    assert fixed >= 0 and summary["fixed_index"] == fixed
    assert np.isclose(summary["beta"], (shares_cal @ m_sel)[fixed])

    # the calibration sets and every certificate follow from the saved counts and the documented streams, one for the
    # calibration sets, one for each window, prevalence and mix of each arm, then one for each natural window
    b_cal, m_cal, _ = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"], hist["calibration_sizes"])
    cells = len(windows) * len(prevalences) * len(mixes)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(ARMS) * cells + len(windows))
    cal = draw_counts(np.random.default_rng(seeds[0]), n_cal, PI_CAL, shares_cal, b_cal, m_cal, DRAWS)
    assert np.array_equal(saved["counts"], np.column_stack([cal.n_benign, cal.n_family]))
    grouped = group_rows(cal.family_below, cal.n_family, hist["groups"])
    pooled_below, pooled_n = cal.family_below.sum(1), cal.n_family.sum(1)
    args = (EPS, c["delta"], PI_MAX, c["delta_benign"])
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    simplex = certify(cal.benign_below, cal.n_benign, *grouped, *args, **start)
    pooled = certify(cal.benign_below, cal.n_benign, pooled_below[:, None], pooled_n[:, None], *args, **start)
    assert np.array_equal(saved["forge_simplex_index"], simplex["index"])
    assert np.array_equal(saved["forge_pooled_index"], pooled["index"])
    # LTT starts by the rule of FORGE, the config sets ltt_guarded_start
    assert c["ltt_guarded_start"]
    ltt = ltt_iid(cal.benign_below, pooled_below, EPS, c["delta"], n=cal.n_benign + cal.n_family.sum(1), **start)
    assert np.array_equal(saved["ltt_iid_index"], ltt["index"])
    assert np.array_equal(saved["conformal_fnr_index"], conformal_fnr(pooled_below, pooled_n, summary["beta"]))
    assert (saved["fixed_sel_index"] == fixed).all()
    # the merged group makes the simplex claim refuse in some draws, while the pooled one, which covers the calibration
    # mix only, certifies looser thresholds
    assert 0 < np.mean(simplex["index"] < 0) < 0.6
    assert (pooled["index"] >= simplex["index"]).mean() > 0.9 and (pooled["index"] > simplex["index"]).any()

    filtered = windows["filtered"].to_numpy()
    reference_benign = round(hist["window_sizes"][~filtered, 0].mean())
    assert filtered.tolist() == [True] + [False] * 6 and summary["reference_benign"] == reference_benign

    def population(i, arm):
        b_w, m_w, own = cdfs(hist["window_benign_below"][i], hist["window_family_below"][i], hist["window_sizes"][i])
        return {"reference": (b_cal, m_cal), "s2": (b_w, m_cal), "real": (b_w, m_w)}[arm], own

    # the filtered capture has reference windows only, and the real windows at the calibration mix are left out, since a
    # test part holds the attacks of its own family only
    skipped = [[w.segment, pi, "calibration"] for w in windows.itertuples() if not w.filtered for pi in prevalences]
    assert summary["skipped"] == skipped
    assert set(claims.loc[claims["window"] == 0, "arm"]) == {"reference"}
    assert len(claims) == len(methods) * (len(ARMS) * cells - 2 * len(prevalences) * len(mixes) - len(skipped))
    certified = {}
    for row in claims.itertuples():
        i, j, k = row.window, prevalences.index(row.prevalence), mixes.index(row.mix)
        (b, m_family), own = population(i, row.arm)
        shares = own if row.mix == "window" else shares_cal
        m = shares @ m_family
        sizes = hist["window_sizes"][i]
        full = round((reference_benign if filtered[i] else sizes[0]) / (1 - row.prevalence))
        attacks = sizes[1:] if row.arm == "real" else n_attack
        # a real window short of test attacks keeps them all and has its benign flows subsampled
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
        # the loosest threshold safe at pi_max for each family alone and each merged group at its calibration mix, on
        # the distributions of the arm, which a real window lacks for the families of other captures
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
            # a threshold safe for every covered mix is safe for the mix of the window when the claim covers it
            assert not row.in_claim or every <= row.best_up_to_pi_max_index
    assert claims["benign_subsampled"].any() and not claims.loc[claims["arm"] != "real", "benign_subsampled"].any()
    # windows of a pooled family lie outside the simplex claim at their own mix, every window lies inside at the
    # calibration mix
    assert windows.loc[~windows["in_claim"], "segment"].tolist() == ["Reaper worm", "spam bot", "ICMP flood"]
    assert claims.loc[claims["mix"] == "calibration", "in_claim"].all()

    # the online certificates of some windows of each arm, a subsampled real window among them, drawn again
    online = {}
    for arm, i, j, k in [("reference", 0, 1, 0), ("reference", 3, 2, 1), ("s2", 2, 0, 0), ("s2", 5, 1, 1),
                         ("real", 1, 2, 0), ("real", 6, 0, 0)]:
        (b, m_family), own = population(i, arm)
        row = claims[(claims["arm"] == arm) & (claims["window"] == i) & (claims["prevalence"] == prevalences[j])
                     & (claims["mix"] == mixes[k])].iloc[0]
        stream = 1 + ARMS.index(arm) * cells + (i * len(prevalences) + j) * len(mixes) + k
        window = draw_counts(np.random.default_rng(seeds[stream]), row["window_sessions"], prevalences[j],
                             own if mixes[k] == "window" else shares_cal, b, m_family, DRAWS)
        online[(arm, i)] = certify_online(window.benign_below + window.family_below.sum(1), row["window_sessions"],
                                          *grouped, *args, **start)
        assert np.array_equal(online[(arm, i)]["index"], saved["online_index"][ARMS.index(arm), i, j, k])
    assert claims.loc[(claims["arm"] == "real") & (claims["window"] == 1) & (claims["prevalence"] == 0.4),
                      "benign_subsampled"].all()
    # a window without an arm keeps the empty value
    assert (saved["online_index"][1:, 0] == -1).all() and (saved["online_index"][2, :, :, 1] == -1).all()

    # each real window at its own prevalence, all flows of its test part
    assert natural["window"].unique().tolist() == list(range(1, len(windows)))
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
            window = draw_counts(np.random.default_rng(seeds[1 + len(ARMS) * cells + i]), int(sizes.sum()), pi, own,
                                 b_w, m_w, DRAWS)
            again = certify_online(window.benign_below + window.family_below.sum(1), int(sizes.sum()), *grouped, *args,
                                   **start)
            assert np.array_equal(again["index"], saved["natural_online_index"][i])
            index = saved["natural_online_index"][i]
        elif row.method in module.PER_WINDOW:
            index = saved[f"natural_{module.PER_WINDOW[row.method]}_index"][i]
        else:
            index = saved[f"{row.method}_index"]
        hit = index >= 0
        assert np.isclose(row.rate, np.mean(np.where(hit, rate[np.maximum(index, 0)], 0.0) > EPS))
        assert np.isclose(row.dismissed_mean, np.where(hit, share[np.maximum(index, 0)], 0.0).mean())
    assert (saved["natural_online_index"][0] == -1).all()

    # rates at the offline FORGE thresholds average the calibration draws that certify one
    chosen = saved["forge_simplex_index"][saved["forge_simplex_index"] >= 0]
    for w in windows.itertuples():
        sizes = hist["window_sizes"][w.window]
        b, m, _ = cdfs(hist["window_benign_below"][w.window], hist["window_family_below"][w.window], sizes)
        assert np.isclose(w.forge_benign_dismissed, np.nan if w.filtered else b[chosen].mean(), equal_nan=True)
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
            assert np.isclose(table.loc[key, "best_up_to_pi_max_dismissed"], g["best_up_to_pi_max_dismissed"].mean())
            assert np.isclose(table.loc[key, "best_every_mix_dismissed"], g["best_every_mix_dismissed"].mean(),
                              equal_nan=True)
            rates = np.concatenate([certified[key][i] for i in g["window"]])
            assert np.isclose(table.loc[key, "for_median"], np.median(rates), equal_nan=True) or not len(rates)
    covered = claims[(claims["prevalence"] <= PI_MAX) & claims["in_claim"]]
    for (arm, method, mix), g in covered.groupby(["arm", "method", "mix"]):
        top = summary["largest_violation_up_to_pi_max"][arm][method][mix]
        assert top["rate"] == g["rate"].max() and top["prevalence"] <= PI_MAX
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
    # a reference window of one family at the calibration mix keeps the calibration distributions, so the upper
    # bound covers its prevalence in most draws, while a window of one family alone moves the estimate
    at_mix = estimates[(estimates["arm"] == "reference") & (estimates["mix"] == "calibration")]
    assert (at_mix["upper_under_rate"] <= 0.1).all()
    assert summary["estimate"]["t_star"] == int(np.argmax(b_sel - shares_cal @ m_sel))

    log = (run.dir / "log.txt").read_text()
    for arm in ARMS:
        for method in methods:
            assert f"{method}, {module.ARM_LABELS[arm]} in the claim, window mix, FOR above epsilon in " in log
            assert (f"{method}, {module.ARM_LABELS[arm]} in the claim, calibration mix, FOR above epsilon in "
                    in log) == (arm != "real")
    best = by_prevalence[(by_prevalence["subset"] == "in_claim") & (by_prevalence["arm"] == "reference")
                         & (by_prevalence["mix"] == "window") & (by_prevalence["method"] == methods[0])
                         & (by_prevalence["prevalence"] == dep["main_prevalence"])].iloc[0]
    assert (f"best thresholds, reference windows in the claim, window mix, mean share of window flows dismissed by the "
            f"loosest threshold with FOR at or below epsilon up to pi_max {best['best_up_to_pi_max_dismissed']:.4f} at "
            f"prevalence {dep['main_prevalence']}, by the loosest safe for every mix the claim covers "
            f"{best['best_every_mix_dismissed']:.4f} at prevalence {dep['main_prevalence']}, by the loosest at the "
            "window's prevalence ") in log
    assert "real windows in the claim, window mix, mean share of window flows dismissed by the loosest threshold " \
           "with FOR at or below epsilon up to pi_max" in log and "covers" not in log.split("best thresholds, real")[1]
    g = main[(main["window"] == 1) & (main["arm"] == "real") & (main["mix"] == "window")]
    assert (f"Reaper worm, real window at prevalence {dep['main_prevalence']}, FOR above epsilon in "
            + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples()) + ", outside the FORGE claim") in log
    g = natural[natural["window"] == 2]
    assert (f"SYN flood, real window at its own prevalence {g['prevalence'].iloc[0]:.5f}, FOR above epsilon in "
            + ", ".join(f"{r.method} {r.rate:.4f}" for r in g.itertuples())) in log
    assert "UDP flood, benign flows left out, at the offline FORGE thresholds miss rate udp_flood " in log
    assert "Smurf, capture icmp_flood, flows " in log and "ICMP flood, capture icmp_flood, flows " in log

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
        # the yardstick for every covered mix where the arm has the distributions of every family
        labels = [line.get_label() for line in fig.axes[2].get_lines()]
        assert (module.EVERY_LINE[2] in labels) == (arm != "real")
        if arm != "real":
            assert np.array_equal(drawn(fig.axes[2], module.EVERY_LINE[2]), g["best_every_mix_dismissed"].to_numpy())
        plt.close(fig)
    # the figure over windows, a band behind each window outside the claim and the dismissed shares of the reference
    # windows in the last row
    fig = module.FIGURES["violation_by_window"](run.dir)
    assert len(fig.axes[0].patches) == 3
    reference = main[(main["arm"] == "reference") & (main["mix"] == "window")]
    for method in methods:
        g = reference[reference["method"] == method].sort_values("window")
        assert np.array_equal(drawn(fig.axes[-1], module.LABELS[method]), g["dismissed_mean"].to_numpy())
    assert np.array_equal(drawn(fig.axes[-1], r"best threshold up to $\pi_{\max}$"),
                          g["best_up_to_pi_max_dismissed"].to_numpy())
    assert [t.get_text() for t in fig.axes[-1].get_xticklabels()] == SEGMENTS
    plt.close(fig)
    # the drift figure keeps the miss rates of the families with enough test attacks
    fig = module.FIGURES["drift_by_window"](run.dir)
    ratios = [getattr(w, f"forge_miss_{f}") / forge_cal["miss"][f] for f in families for w in windows.itertuples()
              if getattr(w, f) >= MIN_ATTACKS]
    assert len(ratios) == 3 and np.allclose(fig.axes[1].get_lines()[0].get_ydata(), ratios)
    plt.close(fig)
    names = {p.name for p in run.figures.iterdir()}
    for name in module.FIGURES:
        assert {f"{name}.png", f"{name}.pdf", f"{name}_a4.pdf"} <= names

    # the same config and seed give the same results
    again, _ = run_e12(cfg, tmp_path / "again")
    assert runs.load_table(again.dir, "claims").equals(claims)
    assert np.array_equal(runs.load_arrays(again.dir, "scores")["score"], scores["score"])


def test_e12_natural_mix_and_no_benign_subsampling(litnet_signal, tmp_path):
    # thinned in their natural mix the calibration attacks follow their counts, and without subsampling a real window
    # short of test attacks is left out as in e11
    cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml",
                      OVERRIDES + ["calibration.mix=natural", "deployment.subsample_benign=false",
                                   "simulation.methods=[forge_simplex, ltt_iid]"])
    run, _ = run_e12(cfg, tmp_path / "runs")
    hist = runs.load_arrays(run.dir, "histograms")
    sizes = hist["calibration_sizes"][1:]
    assert np.allclose(hist["shares"], sizes / sizes.sum())
    claims = runs.load_table(run.dir, "claims")
    summary = runs.load_json(run.dir, "summary")
    assert not claims["benign_subsampled"].any()
    # Reaper worm holds 28 test attacks, too few for a window of its benign flows at prevalence 0.05
    assert ["Reaper worm", 0.05, "window"] in summary["skipped"]
    real = claims[claims["arm"] == "real"]
    for row in real.itertuples():
        test = hist["window_sizes"][row.window]
        assert (test[1:] >= row.window_sessions * row.prevalence * test[1:] / test[1:].sum()).all()


def test_e12_checks_its_config(litnet_signal, tmp_path):
    # one folder per case, since runs started in the same second share a run id
    for n, (override, message) in enumerate([("deployment.main_prevalence=0.3", "main prevalence 0.3"),
                                             ("deployment.mixes=[window, known]", "mixes are"),
                                             ("simulation.methods=[forge_simplex, oracle]", "unknown methods oracle"),
                                             ("split.shares=[0.5, 0.5]", "one for each of")]):
        cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES + [override])
        with pytest.raises(ValueError, match=message):
            run_e12(cfg, tmp_path / f"runs{n}")


# the icmp flood capture with Smurf and ICMP flood attacking through the same minutes
OVERLAPPING = [(name, start, benign, {"smurf": (700, 0, 360), "icmp_flood": (450, 0, 360)})
               if name == "icmp_flood" else (name, start, benign, families)
               for name, start, benign, families in LITNET_SIGNAL_CAPTURES]


@pytest.mark.parametrize("litnet_signal", [OVERLAPPING], indirect=True)
def test_e12_families_that_overlap_in_time_share_one_window(litnet_signal, tmp_path):
    # the capture stays one window holding both families, and its real window above its own prevalence keeps the test
    # attacks of both
    cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml",
                      OVERRIDES + ["simulation.methods=[forge_simplex, forge_online]"])
    run, _ = run_e12(cfg, tmp_path / "runs")
    windows = runs.load_table(run.dir, "windows")
    assert windows["segment"].tolist() == SEGMENTS[:5] + ["Smurf and ICMP flood"]
    cuts = runs.load_table(run.dir, "cuts")
    assert (cuts.loc[cuts["capture"] == "icmp_flood", "families"] == "smurf+icmp_flood").all()
    hist = runs.load_arrays(run.dir, "histograms")
    families = hist["families"].tolist()
    sizes = hist["window_sizes"][5]
    both = [families.index("smurf"), families.index("icmp_flood")]
    assert (sizes[1:][both] > 0).all() and sizes[1:].sum() == sizes[1:][both].sum()
    own = sizes[1:] / sizes[1:].sum()
    real = runs.load_table(run.dir, "claims").query("arm == 'real' and window == 5")
    assert real["mix"].unique().tolist() == ["window"] and real["benign_subsampled"].any()
    for row in real.itertuples():
        full = round(sizes[0] / (1 - row.prevalence))
        short = (sizes[1:] < full * row.prevalence * own).any()
        need = own > 0
        n = int(np.floor((sizes[1:][need] / (row.prevalence * own[need])).min())) if short else full
        assert row.benign_subsampled == short and row.window_sessions == n
    # ICMP flood sits in a merged group, so a window of both icmp families lies outside the claim
    merged = runs.load_json(run.dir, "summary")["families"]
    assert any("icmp_flood" in g and len(g) > 1 for g in merged) and not windows.loc[5, "in_claim"]
    natural = runs.load_table(run.dir, "natural")
    assert np.isclose(natural.loc[natural["window"] == 5, "prevalence"].iloc[0], sizes[1:].sum() / sizes.sum())


@pytest.mark.parametrize("litnet_signal", [[(name, start, 0 if name == "spam" else benign, families)
                                            for name, start, benign, families in LITNET_SIGNAL_CAPTURES]],
                         indirect=True)
def test_e12_stops_on_a_window_without_benign_flows(litnet_signal, tmp_path):
    # the spam capture holds attacks only, and it is no filtered capture, so its window would have no benign population
    cfg = load_config(configs_dir() / "experiments" / "e12_litnet_prevalence.yaml", OVERRIDES)
    with pytest.raises(ValueError, match="spam bot hold no benign flow"):
        run_e12(cfg, tmp_path / "runs")
