import numpy as np
import pandas as pd
import pytest

from forge import experiments, runs
from forge.config import load_config
from forge.eval.metrics import auc
from forge.models import combiners, scorers
from forge.models.encoding import Encoder
from forge.paths import configs_dir
from forge.runs import Run

from test_for_vs_prevalence import OVERRIDES as E11_OVERRIDES
from test_litnet_prevalence import OVERRIDES as E12_OVERRIDES

# small models and few steps, which a few thousand flows train in seconds, and two values of epsilon
SMALL = [
    "scorers.xgboost.rounds=30", "scorers.xgboost.early_stopping=5", "scorers.xgboost.params.min_child_weight=5",
    "scorers.extra_trees.n_estimators=5", "scorers.extra_trees.min_samples_leaf=5", "scorers.mlp.max_iter=5",
    "scorers.mlp.batch_size=256", "scorers.logistic.max_iter=50", "scorers.isolation_forest.n_estimators=20",
    "scorers.gate.steps=100", "scorers.gate.batch=512", "scorers.tail.steps=100", "scorers.tail.batch_benign=512",
    "scorers.tail.batch_family=64", "comparison.epsilons=[0.05, 0.1]", "comparison.main_epsilon=0.05",
]
BASE = ["lightgbm", "xgboost", "extra_trees", "mlp", "logistic", "isolation_forest"]
COMBINERS = ["s_best", "s_mean", "s_stack", "s_gate", "s_tail", "s_2d"]


def labeled(n, seed):
    # attacks have a higher rate and use service a more often, the count is heavy tailed and the bin has -1 for
    # missing
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    service = np.where((y == 1) & (rng.random(n) < 0.5), "a", rng.choice(["a", "b", "c"], n))
    return pd.DataFrame({
        "rate": np.clip(np.where(y == 1, rng.normal(0.65, 0.15, n), rng.normal(0.35, 0.15, n)), 0, 1).astype(
            np.float32),
        "count": np.round(np.exp(rng.normal(3 + y, 1, n))).astype(np.int64),
        "bin": rng.integers(-1, 20, n).astype(np.int16),
        "service": pd.Categorical(service),
    }), y


def run_comparison(config, overrides, root):
    cfg = load_config(configs_dir() / "experiments" / config, overrides)
    run = Run(cfg, root=root)
    module = experiments.get("scorer_comparison")
    module.run(cfg, run)
    run.finish()
    return run, cfg


def test_dense_features_and_every_base_scorer_orders_attacks_above_benign_flows():
    train, y = labeled(3000, 0)
    later, y_later = labeled(1000, 1)
    enc = Encoder(["rate", "count", "bin"], ["service"], {"bin": -1}).fit(train)
    x, x_later = enc.transform(train), enc.transform(later)
    dense = scorers.Dense(enc.categorical_index).fit(x)
    out = dense.transform(x_later)
    # three numeric columns at or above 0 on a log scale and standardized, then one indicator per service, and a
    # missing bin becomes the mean, 0. a column with negative training values keeps its scale
    assert out.shape == (1000, 3 + 3) and out.dtype == np.float32
    assert dense.log.tolist() == [True, True, True] and dense.codes == [3]
    assert np.allclose(out[:, 3:].sum(1), 1)
    assert np.all(out[np.isnan(x_later[:, 2]), 2] == 0)
    assert np.isfinite(out).all()
    signed = scorers.Dense([]).fit(np.array([[-1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert signed.log.tolist() == [False, True] and signed.width == 2
    for name in BASE:
        cfg = {"lightgbm": {"params": {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 20},
                            "rounds": 30, "early_stopping": 5},
               "xgboost": {"params": {"objective": "binary:logistic", "tree_method": "hist", "verbosity": 0},
                           "rounds": 30, "early_stopping": 5},
               "extra_trees": {"n_estimators": 10, "min_samples_leaf": 5},
               "mlp": {"hidden": [16], "batch_size": 128, "max_iter": 10, "learning_rate": 0.01},
               "logistic": {"c": 1.0, "max_iter": 100},
               "isolation_forest": {"n_estimators": 20, "max_samples": 128}}[name]
        fitted = scorers.fit(name, x, y, x_later, y_later, cfg, 0, enc.columns, enc.categorical_index)
        score = scorers.score(name, fitted, x_later, enc.categorical_index)
        assert score.dtype == np.float32 and score.shape == (1000,) and np.isfinite(score).all()
        # the novelty signal sees benign rows only and still ranks the attacks, whose rate and count sit apart, above
        # most benign rows, while a supervised scorer separates them well
        assert auc(score, y_later) > (0.6 if name == "isolation_forest" else 0.85), name
        rounds = scorers.rounds_used(name, fitted)
        assert rounds is None or 1 <= rounds <= 30
    with pytest.raises(ValueError):
        scorers.fit("svm", x, y, x_later, y_later, {}, 0, enc.columns, enc.categorical_index)
    with pytest.raises(ValueError):
        scorers.Dense([3]).transform(x)


def numerical_gradient(loss, params, key, eps=1e-6):
    grad = np.zeros_like(params[key], dtype=np.float64)
    flat = grad.reshape(-1)
    base = params[key].reshape(-1)
    for i in range(len(flat)):
        base[i] += eps
        up = loss()
        base[i] -= 2 * eps
        down = loss()
        base[i] += eps
        flat[i] = (up - down) / (2 * eps)
    return grad


def test_gate_and_tail_losses_match_their_numerical_gradients():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(60, 3))
    family = np.r_[np.zeros(40, dtype=int), np.ones(12, dtype=int), np.full(8, 2)]
    y = (family > 0).astype(float)
    params = {"W": rng.normal(size=(3, 3)) * 0.3, "c": rng.normal(size=3) * 0.3, "b": np.array(0.2)}
    _, grads = combiners.gate_loss(params, z, y)
    for key in params:
        numeric = numerical_gradient(lambda: combiners.gate_loss(params, z, y)[0], params, key)
        assert np.allclose(grads[key], numeric, atol=1e-6), key
    tail = {"w": rng.normal(size=3), "b": np.array(0.1)}
    args = (z, y, family, 0.5, 0.7, 0.2, 0.3)
    _, grads, parts = combiners.tail_loss(tail, *args)
    assert parts["tail"] > 0 and parts["bce"] > 0
    # the tail part is shift invariant, so the bias moves only through the cross entropy
    for key in tail:
        numeric = numerical_gradient(lambda: combiners.tail_loss(tail, *args)[0], tail, key)
        assert np.allclose(grads[key], numeric, atol=1e-5), key


def test_tail_combiner_weights_the_clean_low_tail_more_than_the_stack_does():
    # scorer a separates the rare family from the benign flows in the low tail but is noisy on the common family,
    # scorer b has the higher AUC and scores a tenth of the rare family below most benign flows
    rng = np.random.default_rng(5)
    n_benign, n_common, n_rare = 6000, 1500, 300
    family = np.r_[np.zeros(n_benign, dtype=int), np.ones(n_common, dtype=int), np.full(n_rare, 2)]
    y = (family > 0).astype(float)
    a = np.r_[rng.normal(0, 1, n_benign), rng.normal(1.2, 1.5, n_common), rng.normal(3, 0.5, n_rare)]
    b = np.r_[rng.normal(0, 1, n_benign), rng.normal(3, 1, n_common),
              np.where(rng.random(n_rare) < 0.1, rng.normal(-2, 0.3, n_rare), rng.normal(3, 1, n_rare))]
    z = combiners.Standardizer().fit(np.column_stack([a, b])).transform(np.column_stack([a, b]))
    assert auc(b, y) > auc(a, y)
    stack = combiners.fit_stack(z, y, {"c": 1.0, "max_iter": 200})
    tail = combiners.fit_tail(z, y, family, {"steps": 400, "batch_benign": 1024, "batch_family": 128,
                                             "temperature": 0.5, "family_temperature": 0.1, "weight": 0.1,
                                             "learning_rate": 0.02}, 0, 0.5)
    share_of_a = lambda p: p["w"][0] / np.abs(p["w"]).sum()
    assert share_of_a(tail) > share_of_a(stack)
    assert tail["trace"][-1] < tail["trace"][0]
    # at the benign median the rare family loses fewer attacks under the tail combiner
    def misses(p):
        s = combiners.score_linear(p, z)
        t = np.quantile(s[family == 0], 0.5)
        return np.mean(s[family == 2] <= t)
    assert misses(tail) < misses(stack)
    gate = combiners.fit_gate(z, y, {"steps": 200, "batch": 512, "learning_rate": 0.02}, 0)
    assert gate["trace"][-1] < gate["trace"][0]
    assert combiners.score_gate(gate, z).shape == (len(z),)
    # S-mean is the logit of the mean probability, and S-2D the larger of two benign ranks
    mean = combiners.mean_logit(np.column_stack([a, b]))
    assert np.allclose(combiners.sigmoid(mean), (combiners.sigmoid(a) + combiners.sigmoid(b)) / 2, atol=1e-5)
    ranks = combiners.benign_rank(np.sort(a[family == 0]), np.array([a[family == 0].min() - 1, a[family == 0].max()]))
    assert ranks[0] == 0 and ranks[1] == 1
    assert np.array_equal(combiners.two_d(np.array([0.2, 0.9]), np.array([0.5, 0.1])), np.array([0.5, 0.9]))


def test_first_share_and_the_calibration_error():
    module = experiments.get("scorer_comparison")
    fit = module.first_share(np.array([0, 0, 0, 0, 1, 1, 1]), 0.5)
    assert fit.tolist() == [True, True, False, False, True, True, False]
    p = np.array([0.1, 0.1, 0.9, 0.9])
    assert module.expected_calibration_error(p, np.array([0, 0, 1, 1]), 10) == pytest.approx(0.1)
    assert module.expected_calibration_error(p, np.array([0, 1, 1, 0]), 10) == pytest.approx(0.4)
    assert module.benign_at_miss(np.array([0.0, 0.005, 0.02]), np.array([0.3, 0.5, 0.7]), 0.01) == 0.5
    assert module.benign_at_miss(np.array([0.02, 0.05]), np.array([0.3, 0.5]), 0.01) == 0.0


def test_e32_compares_every_scorer_on_the_e12_split(litnet_signal, tmp_path):
    run, cfg = run_comparison("e32_litnet_scorers.yaml", E12_OVERRIDES + SMALL, tmp_path / "runs")
    assert cfg["experiment"] == {"name": "e32_litnet_scorers", "module": "scorer_comparison"}
    assert cfg["split"]["shares"] == [0.35, 0.10, 0.20, 0.35] and cfg["scorer"]["features"][-1] == "sp_range"
    table = runs.load_table(run.dir, "scorers")
    summary = runs.load_json(run.dir, "summary")
    assert table["scorer"].tolist() == BASE + COMBINERS
    assert table["kind"].tolist() == ["base"] * 6 + ["combiner"] * 6
    base = table[table["kind"] == "base"].set_index("scorer")
    assert (base.loc[["extra_trees", "mlp", "logistic", "isolation_forest"], "train_rows"]
            == summary["capped_rows"]).all()
    assert (base["selection_auc"].drop("isolation_forest") > 0.8).all()
    # S-best is the supervised base scorer whose every mix threshold dismisses most of the second selection part
    supervised = base.drop("isolation_forest")
    assert summary["best"] == supervised["selection_yardstick"].idxmax()
    assert table.set_index("scorer")["best"].sum() == 1 and table.set_index("scorer").loc[summary["best"], "best"]
    assert table["ece"].notna().tolist() == [True] * 5 + [False] + [True] * 5 + [False]
    assert (table["benign_at_miss_hardest"] <= table["benign_at_miss_pooled"]).all()
    info = runs.load_json(run.dir, "combiners")
    assert info["best"] == summary["best"] and info["s_best"]["chosen"] == summary["best"]
    assert set(info["s_stack"]["weights"]) == set(BASE) and set(info["s_tail"]["weights"]) == set(BASE)
    assert 0 < info["s_tail"]["benign_share"] <= 1 and info["s_2d"]["novelty"] == "isolation_forest"
    by_eps = runs.load_table(run.dir, "by_epsilon")
    reference = by_eps[(by_eps["arm"] == "reference") & (by_eps["subset"] == "all")]
    assert len(reference) == 12 * 2 and (reference["windows"] == 7).all()
    for scorer, g in reference.groupby("scorer"):
        g = g.sort_values("epsilon")
        # the loosest threshold safe for every mix only loosens with epsilon
        assert g["best_every_mix_dismissed"].is_monotonic_increasing, scorer
        assert (g["dismissed_mean"] <= 1).all() and (g["refused"] <= 1).all()
    # the claim holds up to noise on the windows inside it, fewer than all once a family is pooled
    claim = by_eps[(by_eps["arm"] == "reference") & (by_eps["subset"] == "in_claim")]
    assert len(claim) == 12 * 2 and (claim["windows"] < 7).all() and (claim["violation_rate"] <= 0.2).all()
    s_best = reference[reference["scorer"] == "s_best"].set_index("epsilon")
    chosen = reference[reference["scorer"] == summary["best"]].set_index("epsilon")
    assert np.allclose(s_best["dismissed_mean"], chosen["dismissed_mean"])
    assert summary["winner"] in table["scorer"].tolist() and len(summary["certified"]) == 12
    assert summary["certified"][summary["winner"]] == max(summary["certified"].values())
    # the real arm exists for the windows with benign flows only
    claims = runs.load_table(run.dir, "claims")
    windows = runs.load_table(run.dir, "windows")
    assert len(windows) == 12 * 7 and windows["auc"].isna().sum() == 12 * 1
    assert set(claims.loc[claims["arm"] == "real", "segment"]) == set(windows.loc[~windows["filtered"], "segment"])
    assert (claims["prevalence"] == 0.3).all()
    # S-mean from the saved logits gives the AUC of its row
    saved = runs.load_arrays(run.dir, "scores")
    mean = combiners.mean_logit(np.column_stack([saved[f"{n}_calibration"] for n in BASE if n != "isolation_forest"]))
    assert auc(mean, saved["calibration_label"]) == pytest.approx(
        table.set_index("scorer").loc["s_mean", "calibration_auc"])
    assert saved["fit"].sum() == summary["fit_rows"] and (~saved["fit"]).sum() == summary["selection_rows"]
    for name in ("dismissed_by_epsilon", "ece_vs_dismissed", "dismissed_by_scorer"):
        for suffix in (".png", ".pdf", "_a4.pdf"):
            assert (run.figures / f"{name}{suffix}").is_file()


def test_e31_compares_every_scorer_on_the_e11_split(anoshift_signal, tmp_path):
    overrides = [o for o in E11_OVERRIDES if not o.startswith("split.deployment")] + ["split.deployment=[2010, 2010]"]
    run, cfg = run_comparison("e31_anoshift_scorers.yaml", overrides + SMALL, tmp_path / "runs")
    assert cfg["experiment"] == {"name": "e31_anoshift_scorers", "module": "scorer_comparison"}
    summary = runs.load_json(run.dir, "summary")
    assert summary["windows"] == 2 and summary["families"] == ["known", "unknown"] and summary["unit"] == "sessions"
    assert summary["main_period"] == "iid" and summary["years"]["deployment"] == [2010]
    table = runs.load_table(run.dir, "scorers")
    assert table["scorer"].tolist() == BASE + COMBINERS
    by_eps = runs.load_table(run.dir, "by_epsilon")
    assert set(by_eps["arm"]) == {"reference", "real"}
    reference = by_eps[(by_eps["arm"] == "reference") & (by_eps["subset"] == "all")]
    assert len(reference) == 12 * 2 and (reference["windows"] == 2).all()
    assert (reference["dismissed_mean"] <= 1).all()
    windows = runs.load_table(run.dir, "windows")
    assert windows["segment"].tolist()[:2] == ["2010 block 0", "2010 block 1"] and windows["auc"].notna().all()
    with pytest.raises(ValueError):
        run_comparison("e31_anoshift_scorers.yaml", overrides + SMALL + ["comparison.main_epsilon=0.02"],
                       tmp_path / "runs2")
