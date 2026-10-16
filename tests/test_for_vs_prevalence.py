import matplotlib.pyplot as plt
import numpy as np
import pytest

from forge import experiments, runs
from forge.certify.certificate import (certify, certify_online, estimate_prevalence, estimated_certifier,
                                       for_from_rates, group_rows, ks_threshold, plug_in)
from forge.config import load_config
from forge.data.synthetic import draw_counts
from forge.eval.metrics import auc
from forge.paths import configs_dir
from forge.runs import Run

from conftest import ANOSHIFT_FAMILY_YEARS, ANOSHIFT_SIGNAL_YEARS

# the six years of the fixture in the four roles, two blocks per deployment year, and a looser epsilon and pi_max that
# a few thousand sessions can certify. the last prevalence lies above pi_max. every kind of mix is used once
OVERRIDES = [
    "split.train=[2006, 2007]", "split.selection=[2008, 2008]", "split.calibration=[2009, 2009]",
    "split.deployment=[2010, 2011]", "dataset.blocks_per_year=2", "scorer.rounds=60", "scorer.early_stopping=10",
    "scorer.params.min_data_in_leaf=20", "certify.epsilon=0.05", "certify.pi_max=0.3", "certify.grid_size=50",
    "calibration.prevalence=0.1", "deployment.prevalences=[0.05, 0.2, 0.4]", "deployment.main_prevalence=0.2",
    "deployment.mixes=[window, known, calibration]", "simulation.draws=200",
]
OFFLINE = ["forge_simplex", "ltt_iid", "conformal_fnr", "fixed_sel"]
EPS, DRAWS, PI_CAL = 0.05, 200, 0.1


def run_e11(cfg, root):
    run = Run(cfg, root=root)
    module = experiments.get("for_vs_prevalence")
    module.run(cfg, run)
    run.finish()
    return run, module


def cdfs(benign_below, family_below, sizes):
    # cdfs of the benign class and of each attack type at the grid, and the mix of the types
    family = np.divide(family_below, sizes[1:, None], out=np.zeros(family_below.shape), where=sizes[1:, None] > 0)
    return benign_below / sizes[0], family, sizes[1:] / sizes[1:].sum()


def mix_shares(mix, own, calibration):
    return {"window": own, "calibration": calibration, "known": np.array([1.0, 0.0])}[mix]


def recompute_certificates(cfg, run, cells, reference_cells=()):
    # the calibration sets and the windows of the given cells, a window, prevalence and mix each, drawn again from the
    # saved counts, give the same offline and online certificates. one stream serves the calibration sets, one each
    # window, prevalence and mix, then one each reference window in the same layout. a reference window takes the
    # benign count and mix of its block and the score distributions of the calibration year. the online certificate
    # sees every session of the window whatever its label
    c, dep, draws = cfg["certify"], cfg["deployment"], cfg["simulation"]["draws"]
    prevalences, mixes = dep["prevalences"], dep["mixes"]
    hist, saved = runs.load_arrays(run.dir, "histograms"), runs.load_arrays(run.dir, "draws")
    summary = runs.load_json(run.dir, "summary")
    args = (c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"])
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    cells_per_arm = len(hist["window_sizes"]) * len(prevalences) * len(mixes)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + 2 * cells_per_arm)
    b_cal, m_cal, shares = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"],
                                hist["calibration_sizes"])
    cal = draw_counts(np.random.default_rng(seeds[0]), summary["calibration_sessions"],
                      cfg["calibration"]["prevalence"], shares, b_cal, m_cal, draws)
    grouped = group_rows(cal.family_below, cal.n_family, hist["groups"])
    offline = certify(cal.benign_below, cal.n_benign, *grouped, *args, **start)
    assert np.array_equal(offline["index"], saved["forge_simplex_index"])
    windows = runs.load_table(run.dir, "windows")
    tested = [(arm, cell) for arm, group in [("real", cells), ("reference", reference_cells)] for cell in group]
    for arm, (i, j, k) in tested:
        b, m, own = cdfs(hist["window_benign_below"][i], hist["window_family_below"][i], hist["window_sizes"][i])
        if arm == "real":
            assert [*windows.loc[i, ["year", "block"]].tolist(), prevalences[j], mixes[k]] not in summary["skipped"]
        else:
            assert windows.loc[i, "period"] == summary["main_period"]
            b, m = b_cal, m_cal
        n = round(hist["window_sizes"][i][0] / (1 - prevalences[j]))
        stream = 1 + (cells_per_arm if arm == "reference" else 0) + (i * len(prevalences) + j) * len(mixes) + k
        window = draw_counts(np.random.default_rng(seeds[stream]), n, prevalences[j],
                             mix_shares(mixes[k], own, shares), b, m, draws)
        online = certify_online(window.benign_below + window.family_below.sum(1), n, *grouped, *args, **start)
        key = "online_index" if arm == "real" else "reference_online_index"
        assert np.array_equal(online["index"], saved[key][i, j, k])
    return offline, saved


def test_e11_tables_figures_and_window_rates(anoshift_signal, tmp_path):
    cfg = load_config(configs_dir() / "experiments" / "e11_anoshift_prevalence.yaml", OVERRIDES)
    run, module = run_e11(cfg, tmp_path / "runs")
    dep = cfg["deployment"]
    methods, prevalences, mixes = cfg["simulation"]["methods"], dep["prevalences"], dep["mixes"]
    frame = anoshift_signal.frame

    windows = runs.load_table(run.dir, "windows")
    assert windows[["year", "block"]].values.tolist() == [[2010, 0], [2010, 1], [2011, 0], [2011, 1]]
    assert windows["sessions"].sum() == frame["year"].isin([2010, 2011]).sum()
    assert windows.groupby("year")["benign"].sum().tolist() == [ANOSHIFT_SIGNAL_YEARS[y][2] for y in (2010, 2011)]
    summary = runs.load_json(run.dir, "summary")
    hist = runs.load_arrays(run.dir, "histograms")
    # the blocks of 2011 hold no unknown type, so they have no window in the calibration mix, while the blocks of 2010
    # hold enough of both types for every prevalence
    assert summary["skipped"] == [[2011, b, pi, "calibration"] for b in (0, 1) for pi in prevalences]
    claims = runs.load_table(run.dir, "claims")
    # every block of 2010, the main period, also has a reference window at every prevalence and mix, the calibration
    # mix of the unknown type included, since a reference window takes no attacks of its block
    assert summary["arms"] == ["reference", "real"]
    reference = claims[claims["arm"] == "reference"]
    assert set(reference["year"]) == {2010} and set(reference["period"]) == {"iid"}
    cells = len(prevalences) * len(mixes)
    assert len(reference) == len(methods) * 2 * cells
    assert len(claims) == len(reference) + len(methods) * (len(windows) * cells - len(summary["skipped"]))
    # a thinned window keeps every benign session of its block and takes no more attacks of a type than the block holds,
    # and a reference window has the benign count of its block
    _, _, cal_shares = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"],
                            hist["calibration_sizes"])
    for row in claims.itertuples():
        sizes = hist["window_sizes"][row.window]
        assert row.window_sessions == round(sizes[0] / (1 - row.prevalence))
        shares = mix_shares(row.mix, sizes[1:] / sizes[1:].sum(), cal_shares)
        assert row.arm == "reference" or (sizes[1:] >= row.window_sessions * row.prevalence * shares).all()

    known, unknown, normal = ANOSHIFT_SIGNAL_YEARS[2009]
    assert hist["calibration_sizes"].tolist() == [normal, known, unknown]
    assert hist["window_sizes"].sum(0).tolist() == [sum(ANOSHIFT_SIGNAL_YEARS[y][k] for y in (2010, 2011))
                                                    for k in (2, 0, 1)]
    # the unknown type has far fewer expected calibration attacks than a family needs, so it joins the known one
    assert summary["families"] == [["known", "unknown"]]
    assert summary["calibration_sessions"] == round(normal / (1 - PI_CAL))
    assert summary["grid_thresholds"] == len(hist["grid"]) <= 50

    # the baselines tuned on the selection year follow from its counts at the calibration prevalence
    sel_b, sel_f, sel_n = hist["selection_benign_below"], hist["selection_family_below"], hist["selection_sizes"]
    fixed = int(plug_in(sel_b, sel_n[0], sel_f.sum(0), sel_n[1:].sum(), EPS, PI_CAL))
    assert fixed >= 0 and summary["fixed_index"] == fixed
    assert np.isclose(summary["beta"], sel_f.sum(0)[fixed] / sel_n[1:].sum())

    # an offline threshold is fixed per draw and an online one per draw and window, so the true rate and dismissed
    # share in a window follow from the counts of the whole block, or of the calibration year for a reference window,
    # with the attack types weighted by the mix of the window
    draws = runs.load_arrays(run.dir, "draws")
    b_cal, m_cal, _ = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"],
                           hist["calibration_sizes"])
    certified = {}
    for row in claims.itertuples():
        b, m, own = cdfs(hist["window_benign_below"][row.window], hist["window_family_below"][row.window],
                         hist["window_sizes"][row.window])
        if row.arm == "reference":
            b, m = b_cal, m_cal
        miss = mix_shares(row.mix, own, cal_shares) @ m
        rate = for_from_rates(row.prevalence, miss, b)
        share = row.prevalence * miss + (1 - row.prevalence) * b
        if row.method in module.PER_WINDOW:
            key = ("" if row.arm == "real" else "reference_") + module.PER_WINDOW[row.method] + "_index"
            index = draws[key][row.window, prevalences.index(row.prevalence), mixes.index(row.mix)]
        else:
            index = draws[f"{row.method}_index"]
        hit = index >= 0
        certified.setdefault((row.arm, row.method, row.prevalence, row.mix), []).append(rate[index[hit]])
        assert np.isclose(row.rate, np.mean(np.where(hit, rate[np.maximum(index, 0)], 0.0) > EPS))
        assert np.isclose(row.dismissed_mean, np.where(hit, share[np.maximum(index, 0)], 0.0).mean())
        assert np.isclose(row.refused, np.mean(~hit))
        # the best thresholds are the loosest whose true FOR stays at or below epsilon at the window's prevalence and
        # at pi_max, and the sessions they dismiss
        for name, at in [("best_at_prevalence", row.prevalence), ("best_up_to_pi_max", 0.3)]:
            safe = np.flatnonzero(for_from_rates(at, miss, b) <= EPS)
            best = safe[-1] if len(safe) else -1
            assert getattr(row, f"{name}_index") == best
            assert np.isclose(getattr(row, f"{name}_dismissed"), share[best] if best >= 0 else 0.0)
    # a threshold safe up to pi_max is safe at every lower prevalence, and one safe above pi_max is safe at pi_max
    below, above = claims[claims["prevalence"] < 0.3], claims[claims["prevalence"] > 0.3]
    assert (below["best_up_to_pi_max_index"] <= below["best_at_prevalence_index"]).all()
    assert (above["best_up_to_pi_max_index"] >= above["best_at_prevalence_index"]).all()
    assert (claims["best_at_prevalence_index"] >= 0).all()
    assert (draws["fixed_sel_index"] == fixed).all()
    for key in ["online_index", "reference_online_index"]:
        assert draws[key].shape == (len(windows), len(prevalences), len(mixes), DRAWS)
        assert ((draws[key] >= -1) & (draws[key] < len(hist["grid"]))).all()
    # only the blocks of the main period have reference windows, the others keep the empty value
    assert (draws["reference_online_index"][2:] == -1).all() and (draws["reference_online_floor"][2:] == 0).all()
    assert (draws["reference_online_index"][:2] >= 0).any()

    recompute_certificates(cfg, run, [(0, 1, 0), (1, 2, 1), (3, 0, 1), (0, 0, 2)],
                           [(0, 0, 0), (1, 2, 2), (1, 1, 1)])

    # the rates at the offline FORGE thresholds average the calibration draws that certify one, in each block and in
    # the calibration year, and a type absent from a block, the unknown one in 2011, has no miss rate there
    chosen = draws["forge_simplex_index"][draws["forge_simplex_index"] >= 0]
    assert len(chosen) > 0
    for w in windows.itertuples():
        sizes = hist["window_sizes"][w.window]
        b, m, _ = cdfs(hist["window_benign_below"][w.window], hist["window_family_below"][w.window], sizes)
        assert np.isclose(w.forge_benign_dismissed, b[chosen].mean())
        for k, f in enumerate(["known", "unknown"]):
            expected = m[k][chosen].mean() if sizes[k + 1] else np.nan
            assert np.isclose(getattr(w, f"forge_miss_{f}"), expected, equal_nan=True)
    assert windows.loc[windows["year"] == 2011, "forge_miss_unknown"].isna().all()
    b, m, _ = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"], hist["calibration_sizes"])
    forge_cal = summary["forge_calibration"]
    assert np.isclose(forge_cal["benign_dismissed"], b[chosen].mean())
    assert np.allclose([forge_cal["miss"][f] for f in ("known", "unknown")], m[:, chosen].mean(1))

    refused = claims.groupby("method")["refused"].mean()
    assert refused["forge_simplex"] < 1 and refused["forge_online"] < 1
    # a refusal dismisses nothing, so a method that refuses everywhere has no violation either
    assert (claims.loc[claims["refused"] == 1, "violations"] == 0).all()
    # the years of the fixture share their score distributions, so the floors hold where the claims cover the window,
    # and only FORGE has a floor
    covered = claims[claims["prevalence"] <= 0.3]
    assert (covered.loc[covered["method"] == "forge_simplex", "floor_violations"] <= 0.1 * DRAWS).all()
    # the largest window share up to pi_max is kept for each arm and period apart, the claim is read on the reference
    # windows of the main period
    for (arm, method, period, mix), g in covered.groupby(["arm", "method", "period", "mix"]):
        top = summary["largest_violation_up_to_pi_max"][arm][method][period][mix]
        assert top["rate"] == g["rate"].max() and top["prevalence"] <= 0.3
    assert set(summary["largest_violation_up_to_pi_max"]["reference"]["forge_simplex"]) == {"iid"}
    assert summary["deployment_periods"] == ["iid", "near"] and summary["main_period"] == "iid"
    log = (run.dir / "log.txt").read_text()
    for method in methods:
        assert f"{method}, iid reference windows, known mix, FOR above epsilon in " in log
        for period in summary["deployment_periods"]:
            assert f"{method}, {period} windows, known mix, FOR above epsilon in " in log
    by_prevalence = runs.load_table(run.dir, "by_prevalence")
    for arm, period, name in [("reference", "iid", "iid reference windows"), ("real", "iid", "iid windows"),
                              ("real", "near", "near windows")]:
        g = by_prevalence[(by_prevalence["arm"] == arm) & (by_prevalence["period"] == period)
                          & (by_prevalence["mix"] == "known") & (by_prevalence["method"] == methods[0])]
        g = g.sort_values("prevalence")
        best = g.loc[g["prevalence"] == dep["main_prevalence"], "best_up_to_pi_max_dismissed"].iloc[0]
        assert (f"best thresholds, {name}, known mix, mean share of window sessions dismissed by the loosest threshold "
                f"with FOR at or below epsilon up to pi_max {best:.4f} at prevalence {dep['main_prevalence']}, by the "
                f"loosest at the window's prevalence "
                + ", ".join(f"{s:.4f} at {pi}" for pi, s in zip(g["prevalence"], g["best_at_prevalence_dismissed"]))
                ) in log
    assert "near reference windows" not in log
    # each block of the main period has its own line, with the sessions its miss rates rest on
    for w in windows[windows["period"] == "iid"].itertuples():
        assert (f"{w.year} block {w.block}, AUC {w.auc:.4f}, attack share {w.attack_share:.4f}, at the offline FORGE "
                f"thresholds benign share dismissed {w.forge_benign_dismissed:.4f}, miss rate known "
                f"{w.forge_miss_known:.4f} over {w.known} sessions, unknown {w.forge_miss_unknown:.4f} over "
                f"{w.unknown} sessions") in log
    assert "2011 block 0, AUC" not in log
    assert (claims.loc[claims["method"] == "forge_online", "floor_violations"] <= 0.1 * DRAWS).all()
    without_floor = ~claims["method"].str.startswith("forge") & ~claims["method"].isin(module.ESTIMATED)
    assert (claims.loc[without_floor, "floor_violations"] == 0).all()

    # the methods at an estimated prevalence, t_star where the selection shares at the calibration mix differ most,
    # the estimates per window and draw saved with their thresholds, and the estimates table over draws
    est = summary["estimate"]
    cal_shares = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"],
                      hist["calibration_sizes"])[2]
    assert est["threshold"] == "ks" and est["t_star"] == ks_threshold(sel_b / sel_n[0], cal_shares @ (
        sel_f / sel_n[1:, None]))
    assert np.isclose(est["delta_family"], 0.05 - 0.01 - 0.01 - 0.005)
    estimates = runs.load_table(run.dir, "estimates")
    assert len(estimates) == len(claims) // len(methods)
    assert (estimates["upper_median"] >= estimates["plugin_median"]).all()
    for key in ["plugin_index", "upper_index", "reference_plugin_index", "reference_upper_index"]:
        assert draws[key].shape == (len(windows), len(prevalences), len(mixes), DRAWS)
    assert (draws["upper_prevalence"] >= draws["plugin_prevalence"]).all()
    # the reference windows keep the calibration distributions, so the upper bound covers the prevalence in most
    # draws and the plug in sits near it, while the real windows of the fixture drift little
    reference = estimates[estimates["arm"] == "reference"]
    assert (reference["upper_under_rate"] <= 0.1).all()
    assert (np.abs(reference["plugin_median"] - reference["prevalence"]) <= 0.5 * reference["prevalence"] + 0.02).all()
    # one cell recomputed, the window drawn again from its seed gives the saved estimates and thresholds
    cells_per_arm = len(windows) * len(prevalences) * len(mixes)
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + 2 * cells_per_arm)
    cal = draw_counts(np.random.default_rng(seeds[0]), summary["calibration_sessions"], PI_CAL, cal_shares, b_cal,
                      m_cal, DRAWS)
    pooled_below, pooled_n = cal.family_below.sum(1), cal.n_family.sum(1)
    i, j, k = 1, 1, 0
    n = round(hist["window_sizes"][i][0] / (1 - prevalences[j]))
    window = draw_counts(np.random.default_rng(seeds[1 + cells_per_arm + (i * len(prevalences) + j) * len(mixes) + k]),
                         n, prevalences[j], mix_shares(mixes[k], None, cal_shares) if mixes[k] != "window" else
                         cdfs(*[hist[f"window_{x}"][i] for x in ("benign_below", "family_below", "sizes")])[2],
                         b_cal, m_cal, DRAWS)
    below = window.benign_below + window.family_below.sum(1)
    t = est["t_star"]
    out = estimate_prevalence(below[:, t], n, cal.benign_below[:, t], cal.n_benign, pooled_below[:, t], pooled_n,
                              0.01, 0.01, 0.005)
    assert np.allclose(out["plugin"], draws["reference_plugin_prevalence"][i, j, k], atol=1e-6)
    assert np.allclose(out["upper"], draws["reference_upper_prevalence"][i, j, k], atol=1e-6)
    at = estimated_certifier(cal.benign_below, cal.n_benign, pooled_below, pooled_n, EPS, 0.01, est["delta_family"],
                             start_misses=cfg["certify"]["start_misses"], start_slack=cfg["certify"]["start_slack"])
    assert np.array_equal(at(out["plugin"])["index"], draws["reference_plugin_index"][i, j, k])
    assert np.array_equal(at(out["upper"])["index"], draws["reference_upper_index"][i, j, k])
    assert "LTT at the estimated prevalence, t_star at grid index" in log
    assert "estimated prevalence, reference windows, window mix, at prevalence 0.2, plugin median" in log
    # the realized share of attacks among the dismissed sessions of a drawn window follows the true rate, LTT has a
    # true rate far above epsilon at prevalence 0.4 and far below it at 0.05
    ltt = claims[(claims["method"] == "ltt_iid") & (claims["mix"] == "window")].groupby("prevalence")
    assert ltt["rate"].min()[0.4] == 1 and ltt["rate"].max()[0.05] == 0
    assert ltt["realized_rate"].min()[0.4] > 0.8 and ltt["realized_rate"].max()[0.05] < 0.2

    keys = ["arm", "method", "prevalence", "mix"]
    pooled = by_prevalence[by_prevalence["period"] == "all"].set_index(keys).sort_index()
    for key, g in claims.groupby(keys):
        assert np.isclose(pooled.loc[key, "violation_rate"], g["rate"].mean())
        assert pooled.loc[key, "violation_window_max"] == g["rate"].max()
        assert pooled.loc[key, "windows"] == len(g)
        assert np.isclose(pooled.loc[key, "dismissed_mean"], g["dismissed_mean"].mean())
        for name in ["best_at_prevalence_dismissed", "best_up_to_pi_max_dismissed"]:
            assert np.isclose(pooled.loc[key, name], g[name].mean())
        rates = np.concatenate(certified[key])
        if len(rates):
            assert np.isclose(pooled.loc[key, "for_median"], np.median(rates))
    assert set(by_prevalence["period"]) == {"all", "iid", "near"}
    assert set(by_prevalence.loc[by_prevalence["arm"] == "reference", "period"]) == {"all", "iid"}

    # the figure of RQ1 shows the reference windows of the main period, the second figure over prevalences its real
    # windows, and the figure over windows the real window of every block
    def drawn(ax, label):
        [line] = [line for line in ax.get_lines() if line.get_label() == label]
        return line.get_ydata()

    main = by_prevalence[(by_prevalence["period"] == "iid") & (by_prevalence["mix"] == "window")]
    assert {column: label for column, _, label in module.BEST_LINES} == {
        "best_up_to_pi_max": r"best threshold up to $\pi_{\max}$",
        "best_at_prevalence": r"best threshold at the window's $\pi$"}
    for name, arm in [("for_by_prevalence", "reference"), ("for_by_prevalence_real", "real")]:
        fig = module.FIGURES[name](run.dir)
        for method in methods:
            g = main[(main["arm"] == arm) & (main["method"] == method)].sort_values("prevalence")
            assert np.array_equal(drawn(fig.axes[2], module.LABELS[method]), g["dismissed_mean"].to_numpy())
        for column, _, label in module.BEST_LINES:
            assert np.array_equal(drawn(fig.axes[2], label), g[f"{column}_dismissed"].to_numpy())
        plt.close(fig)
    fig = module.FIGURES["violation_by_window"](run.dir)
    for method in methods:
        g = claims[(claims["arm"] == "real") & (claims["method"] == method) & (claims["mix"] == "window")
                   & (claims["prevalence"] == dep["main_prevalence"])].sort_values("position")
        assert np.array_equal(drawn(fig.axes[-1], module.LABELS[method]), g["dismissed_mean"].to_numpy())
    column, _, label = module.BEST_LINES[0]
    assert np.array_equal(drawn(fig.axes[-1], label), g[f"{column}_dismissed"].to_numpy())
    plt.close(fig)

    # per session scores of the selection, calibration and deployment years, in the order of the processed table
    scores = runs.load_arrays(run.dir, "scores")
    used = frame[frame["year"] >= 2008]
    assert len(scores["score"]) == len(used) and scores["score"].dtype == np.float32
    assert scores["year"].tolist() == used["year"].tolist() and scores["order"].tolist() == used["order"].tolist()
    assert scores["label"].tolist() == used["label"].tolist()
    assert scores["attack_types"][scores["attack_type"]].tolist() == used["attack_type"].astype(str).tolist()
    # the AUC of a window comes from the scores of its block, the year cut into two equal counts in saved order
    for w in windows.itertuples():
        year = scores["year"] == w.year
        block = np.arange(year.sum()) * 2 // year.sum() == w.block
        assert np.isclose(w.auc, auc(scores["score"][year][block], scores["label"][year][block]))
    scorer = runs.load_json(run.dir, "scorer")
    assert scorer["selection_auc"] > 0.8 and scorer["calibration_auc"] > 0.8
    # the service seen only in the last year is no category of the encoder
    assert "ftp" not in scorer["encoder"]["categories"]["service"]
    assert (run.results / "models" / "scorer.txt").stat().st_size > 0

    names = {p.name for p in run.figures.iterdir()}
    for name in module.FIGURES:
        assert {f"{name}.png", f"{name}.pdf", f"{name}_a4.pdf"} <= names

    # the same config and seed give the same results
    again, _ = run_e11(cfg, tmp_path / "again")
    assert runs.load_table(again.dir, "claims").equals(claims)
    assert np.array_equal(runs.load_arrays(again.dir, "scores")["score"], scores["score"])


def test_best_thresholds_take_the_loosest_safe_threshold_past_an_unsafe_one():
    # the FOR of a drifted window need not rise with the threshold. here it exceeds epsilon at the second threshold only
    # at the window's prevalence, and from the second on at pi_max, so the best threshold at the prevalence is the last
    # one, past the unsafe second, and the best one up to pi_max the first
    module = experiments.get("for_vs_prevalence")
    m, b, eps = np.array([0.001, 0.06, 0.06, 0.07]), np.array([0.5, 0.55, 0.9, 0.95]), 0.001
    assert list(for_from_rates(0.01, m, b) <= eps) == [True, False, True, True]
    assert list(for_from_rates(0.1, m, b) <= eps) == [True, False, False, False]
    out = module.best_thresholds(0.01, 0.1, m, b, eps)
    assert out["best_at_prevalence_index"] == 3 and out["best_up_to_pi_max_index"] == 0
    assert np.isclose(out["best_at_prevalence_dismissed"], 0.01 * 0.07 + 0.99 * 0.95)
    assert np.isclose(out["best_up_to_pi_max_dismissed"], 0.01 * 0.001 + 0.99 * 0.5)
    # a window whose FOR exceeds epsilon at every threshold has no best threshold and dismisses nothing
    none = module.best_thresholds(0.01, 0.1, np.full(4, 0.5), np.full(4, 0.01), eps)
    assert none["best_at_prevalence_index"] == -1 and none["best_at_prevalence_dismissed"] == 0.0
    # a FOR equal to epsilon counts as safe, as in the certificates, here no attack dismissed at epsilon 0
    edge = module.best_thresholds(0.01, 0.1, np.array([0.0, 0.0, 0.1]), np.array([0.1, 0.2, 0.3]), 0.0)
    assert edge["best_at_prevalence_index"] == 1 and edge["best_up_to_pi_max_index"] == 1


def test_e11_main_period_must_be_among_the_deployment_periods(anoshift_signal, tmp_path):
    # the fixture deploys in 2010 and 2011, IID and NEAR, so it has no FAR window to read RQ1 on
    cfg = load_config(configs_dir() / "experiments" / "e11_anoshift_prevalence.yaml",
                      OVERRIDES + ["deployment.main_period=far"])
    with pytest.raises(ValueError, match="main period far"):
        run_e11(cfg, tmp_path / "runs")


@pytest.mark.parametrize("anoshift_signal", [ANOSHIFT_FAMILY_YEARS], indirect=True)
def test_e11_certificates_with_each_attack_type_its_own_family(anoshift_signal, tmp_path):
    # the calibration year holds 1,500 unknown attacks, so the unknown type stands alone and every bound takes the
    # largest of two family bounds, and delta splits unevenly, so the benign and the attack side of a bound cannot
    # trade places unnoticed. pi_max 0.1 lets FORGE certify a threshold in most draws on so few sessions
    cfg = load_config(configs_dir() / "experiments" / "e11_anoshift_prevalence.yaml",
                      OVERRIDES + ["certify.pi_max=0.1", "certify.delta_benign=0.02"])
    run, _ = run_e11(cfg, tmp_path / "runs")
    summary = runs.load_json(run.dir, "summary")
    assert summary["families"] == [["known"], ["unknown"]]
    assert runs.load_arrays(run.dir, "histograms")["groups"].tolist() == [0, 1]
    # at the calibration mix a window of 2010 takes more unknown attacks than its block holds from prevalence 0.2 on,
    # which leaves out the real window but not the reference one, whose attacks come from the calibration year
    assert [2010, 0, 0.2, "calibration"] in summary["skipped"]
    reference = runs.load_table(run.dir, "claims").query("arm == 'reference'")
    dep = cfg["deployment"]
    assert len(reference) == len(cfg["simulation"]["methods"]) * 2 * len(dep["prevalences"]) * len(dep["mixes"])
    offline, saved = recompute_certificates(cfg, run, [(0, 0, 0), (1, 1, 1), (2, 0, 1), (3, 2, 0)],
                                            [(0, 1, 0), (1, 0, 1)])
    # both certificates refuse in some draws and certify in most, so the comparison covers both outcomes
    assert 0 < np.mean(offline["index"] < 0) < 0.5
    tested = saved["online_index"][[0, 1, 2, 3], [0, 1, 0, 2], [0, 1, 1, 0]]
    assert 0 < np.mean(tested < 0) < 0.5
