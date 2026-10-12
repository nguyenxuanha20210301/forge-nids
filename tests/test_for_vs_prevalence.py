import numpy as np
import pytest

from forge import experiments, runs
from forge.certify.certificate import certify, certify_online, for_from_rates, group_rows, plug_in
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


def recompute_certificates(cfg, run, cells):
    # the calibration sets and the windows of the given cells, a window, prevalence and mix each, drawn again from the
    # saved counts, with one stream for the calibration sets and one per window, prevalence and mix, give the same
    # offline and online certificates. the online one sees every session of the window whatever its label
    c, dep, draws = cfg["certify"], cfg["deployment"], cfg["simulation"]["draws"]
    prevalences, mixes = dep["prevalences"], dep["mixes"]
    hist, saved = runs.load_arrays(run.dir, "histograms"), runs.load_arrays(run.dir, "draws")
    summary = runs.load_json(run.dir, "summary")
    args = (c["epsilon"], c["delta"], c["pi_max"], c["delta_benign"])
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(1 + len(hist["window_sizes"]) * len(prevalences) * len(mixes))
    b, m, shares = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"], hist["calibration_sizes"])
    cal = draw_counts(np.random.default_rng(seeds[0]), summary["calibration_sessions"],
                      cfg["calibration"]["prevalence"], shares, b, m, draws)
    grouped = group_rows(cal.family_below, cal.n_family, hist["groups"])
    offline = certify(cal.benign_below, cal.n_benign, *grouped, *args, **start)
    assert np.array_equal(offline["index"], saved["forge_simplex_index"])
    windows = runs.load_table(run.dir, "windows")
    for i, j, k in cells:
        assert [*windows.loc[i, ["year", "block"]].tolist(), prevalences[j], mixes[k]] not in summary["skipped"]
        b, m, own = cdfs(hist["window_benign_below"][i], hist["window_family_below"][i], hist["window_sizes"][i])
        n = round(hist["window_sizes"][i][0] / (1 - prevalences[j]))
        rng = np.random.default_rng(seeds[1 + (i * len(prevalences) + j) * len(mixes) + k])
        window = draw_counts(rng, n, prevalences[j], mix_shares(mixes[k], own, shares), b, m, draws)
        online = certify_online(window.benign_below + window.family_below.sum(1), n, *grouped, *args, **start)
        assert np.array_equal(online["index"], saved["online_index"][i, j, k])
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
    assert len(claims) == len(methods) * (len(windows) * len(prevalences) * len(mixes) - len(summary["skipped"]))
    # a thinned window keeps every benign session of its block and takes no more attacks of a type than the block holds
    _, _, cal_shares = cdfs(hist["calibration_benign_below"], hist["calibration_family_below"],
                            hist["calibration_sizes"])
    for row in claims.itertuples():
        sizes = hist["window_sizes"][row.window]
        assert row.window_sessions == round(sizes[0] / (1 - row.prevalence))
        shares = mix_shares(row.mix, sizes[1:] / sizes[1:].sum(), cal_shares)
        assert (sizes[1:] >= row.window_sessions * row.prevalence * shares).all()

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

    # an offline threshold is fixed per draw, so its true rate in a window follows from the counts of the whole block,
    # with the attack types weighted by the mix of the window
    draws = runs.load_arrays(run.dir, "draws")
    for row in claims[claims["method"].isin(OFFLINE)].itertuples():
        b, m, own = cdfs(hist["window_benign_below"][row.window], hist["window_family_below"][row.window],
                         hist["window_sizes"][row.window])
        rate = for_from_rates(row.prevalence, mix_shares(row.mix, own, cal_shares) @ m, b)
        index = draws[f"{row.method}_index"]
        assert np.isclose(row.rate, np.mean(np.where(index >= 0, rate[np.maximum(index, 0)], 0.0) > EPS))
    assert (draws["fixed_sel_index"] == fixed).all()
    assert draws["online_index"].shape == (len(windows), len(prevalences), len(mixes), DRAWS)
    assert ((draws["online_index"] >= -1) & (draws["online_index"] < len(hist["grid"]))).all()

    recompute_certificates(cfg, run, [(0, 1, 0), (1, 2, 1), (3, 0, 1), (0, 0, 2)])

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
    # the largest window share up to pi_max is kept for each period apart, the claim is read on the main one
    for (method, period, mix), g in covered.groupby(["method", "period", "mix"]):
        top = summary["largest_violation_up_to_pi_max"][method][period][mix]
        assert top["rate"] == g["rate"].max() and top["prevalence"] <= 0.3
    assert summary["deployment_periods"] == ["iid", "near"] and summary["main_period"] == "iid"
    log = (run.dir / "log.txt").read_text()
    for method in methods:
        for period in summary["deployment_periods"]:
            assert f"{method}, {period} windows, known mix, FOR above epsilon in " in log
    assert (claims.loc[claims["method"] == "forge_online", "floor_violations"] <= 0.1 * DRAWS).all()
    assert (claims.loc[~claims["method"].str.startswith("forge"), "floor_violations"] == 0).all()
    # the realized share of attacks among the dismissed sessions of a drawn window follows the true rate, LTT has a
    # true rate far above epsilon at prevalence 0.4 and far below it at 0.05
    ltt = claims[(claims["method"] == "ltt_iid") & (claims["mix"] == "window")].groupby("prevalence")
    assert ltt["rate"].min()[0.4] == 1 and ltt["rate"].max()[0.05] == 0
    assert ltt["realized_rate"].min()[0.4] > 0.8 and ltt["realized_rate"].max()[0.05] < 0.2

    by_prevalence = runs.load_table(run.dir, "by_prevalence")
    pooled = by_prevalence[by_prevalence["period"] == "all"].set_index(["method", "prevalence", "mix"]).sort_index()
    for key, g in claims.groupby(["method", "prevalence", "mix"]):
        assert np.isclose(pooled.loc[key, "violation_rate"], g["rate"].mean())
        assert pooled.loc[key, "violation_window_max"] == g["rate"].max()
        assert pooled.loc[key, "windows"] == len(g)
    assert set(by_prevalence["period"]) == {"all", "iid", "near"}

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
    assert runs.load_json(run.dir, "summary")["families"] == [["known"], ["unknown"]]
    assert runs.load_arrays(run.dir, "histograms")["groups"].tolist() == [0, 1]
    offline, saved = recompute_certificates(cfg, run, [(0, 0, 0), (1, 1, 1), (2, 0, 1), (3, 2, 0)])
    # both certificates refuse in some draws and certify in most, so the comparison covers both outcomes
    assert 0 < np.mean(offline["index"] < 0) < 0.5
    tested = saved["online_index"][[0, 1, 2, 3], [0, 1, 0, 2], [0, 1, 1, 0]]
    assert 0 < np.mean(tested < 0) < 0.5
