"""e31 and e32, RQ3 and E5, the certified dismissed share of every scorer and combiner of proposal 6.2 by epsilon on
the reference windows of e11 and e12, and what of it holds on the real windows.

the certificate reads a scorer only through the counts of each class at the grid thresholds, so what decides its
dismissed share is where the attacks of the hardest family sit among the benign scores in the low tail, Proposition
4. the comparison takes the split, the calibration sets and the windows of the E1 experiment of the dataset and
changes the scorer alone.

the base scorers train on the training parts, LightGBM and XGBoost on all of them and stopped early on the selection
parts, the slower ones on at most cap rows, every attack kept and the benign rows subsampled with the seed, and
IsolationForest on the benign rows of the same subsample. the selection parts are cut in two in time, within each
capture or year. the first part fits the combiners on the standardized base logits, with the novelty score among them,
and fixes the benign quantile that the low tail objective aims at, the benign share dismissed by the loosest threshold
safe for every family on that part under the best base scorer. the second part gives every scorer its grid, picks
S-best among the supervised base scorers by the benign share its every mix threshold dismisses there, and sets the
fixed threshold and beta where the E1 experiments do. nothing of the calibration parts enters any scorer or grid, so
Proposition 6 covers the comparison and each scorer is certified once per epsilon on the calibration draws, the same
draws for every scorer and epsilon through one seed.

every window comes in two arms at pi_max with the family mix of the window. the reference arm takes the score
distributions of the calibration parts for both classes, so label shift holds and RQ3 is read there, and the real arm
the window's own flows. the readings take every window with its own family mix, so every scorer and every epsilon
face the same windows, and the windows inside the simplex claim, which a window of one pooled family lies outside and
which shrink as epsilon falls and more families pool, are kept beside them. the tables hold the certified dismissed
share, the refusals and the violation share of FORGE under each scorer and epsilon, the two
yardsticks of the windows, the benign share dismissed in the calibration parts, the benign share dismissed at a miss
rate of 1% of the pooled calibration attacks and of the hardest family, and the expected calibration error of the
scorers that give a probability.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forge import runs
from forge.certify.certificate import certify, family_groups, grid_from, group_rows, min_family_size
from forge.data import anoshift, litnet2020, splits
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge
from forge.eval.metrics import auc
from forge.experiments import for_vs_prevalence as e11
from forge.experiments import litnet_prevalence as e12
from forge.models import combiners, scorers
from forge.models.encoding import Encoder
from forge.viz import layout, style
from forge.viz.export import save_figure

PARTS = ["train", "selection", "calibration", "test"]
ARMS = ["reference", "real"]
ARM_LABELS = {"reference": "reference windows", "real": "real windows"}
LABELS = {**scorers.LABELS, **combiners.LABELS}
MARKERS = {"lightgbm": "o", "xgboost": "s", "extra_trees": "^", "mlp": "v", "logistic": "D",
           "isolation_forest": "p", "s_mean": "X", "s_stack": "*", "s_gate": "P", "s_tail": "h", "s_2d": "<"}
# rows encoded at a time, which bounds the copy of the frame that the encoder reads
CHUNK = 2_000_000


def encode(enc: Encoder, flows: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    out = np.empty((len(rows), len(enc.columns)), dtype=np.float32)
    for lo in range(0, len(rows), CHUNK):
        out[lo:lo + CHUNK] = enc.transform(flows.iloc[rows[lo:lo + CHUNK]])
    return out


def first_share(segment: np.ndarray, share: float) -> np.ndarray:
    # the first share of the rows of every segment, which are in time order within it
    fit = np.zeros(len(segment), dtype=bool)
    for g in np.unique(segment):
        rows = np.flatnonzero(segment == g)
        fit[rows[:int(np.ceil(share * len(rows)))]] = True
    return fit


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    # the gap between the mean probability and the attack share within equal width bins, weighted by the bin counts
    edges = np.linspace(0, 1, bins + 1)
    at = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    count = np.bincount(at, minlength=bins)
    gap = np.abs(np.bincount(at, weights=y, minlength=bins) - np.bincount(at, weights=p, minlength=bins))
    return float(gap.sum() / count.sum())


def benign_at_miss(m: np.ndarray, b: np.ndarray, miss_rate: float) -> float:
    # the benign share dismissed by the loosest threshold whose miss rate stays at or below miss_rate
    safe = np.flatnonzero(m <= miss_rate)
    return float(b[safe[-1]]) if len(safe) else 0.0


def load_litnet(cfg: dict, run: runs.Run) -> dict:
    d, sc, cc = cfg["dataset"], cfg["scorer"], cfg["calibration"]
    pi_cal = cc["prevalence"]
    filtered = set(d.get("filtered_captures", []))
    manifest = litnet2020.read_manifest(cfg)
    flows = e12.load_flows(cfg, sc["features"])
    label = flows["label"].to_numpy()
    run.log(f"{d['name']} {d['processed_version']}, fingerprint {manifest['fingerprint']}, {len(flows)} flows, "
            f"{int(label.sum())} attacks, {flows['capture'].nunique()} captures")
    split = e12.split_flows(flows, cfg["split"]["shares"], filtered)
    segment, part, kept, families, code = (split[k] for k in ("segment", "part", "kept", "families", "code"))
    cuts = e12.segment_table(flows, segment, part, filtered)
    run.save_table(cuts, "cuts")
    segments = cuts.drop_duplicates("segment").reset_index(drop=True)
    features = sc["features"]
    categorical = [f for f in features if f in sc["categorical"]]
    rows = {name: np.flatnonzero((part == p) & kept) for p, name in enumerate(PARTS)}
    enc = Encoder([f for f in features if f not in categorical], categorical).fit(flows.iloc[rows["train"]])
    x = {name: encode(enc, flows, r) for name, r in rows.items()}
    test_segment = segment[rows["test"]]
    windows = [{"label": g.label, "rows": np.flatnonzero(test_segment == g.segment), "filtered": bool(g.filtered)}
               for g in segments.itertuples()]
    return {
        "manifest": manifest, "families": families, "enc": enc, "x": x,
        "label": {name: label[r] for name, r in rows.items()}, "code": {name: code[r] for name, r in rows.items()},
        "selection_segment": segment[rows["selection"]], "windows": windows, "unit": "flows",
        "shares_cal": lambda n_attack, n_benign: e12.calibration_shares(n_attack, pi_cal / (1 - pi_cal) * n_benign,
                                                                        cc["mix"]),
        "summary": {"split_shares": cfg["split"]["shares"], "filtered_captures": sorted(filtered),
                    "calibration_mix": cc["mix"], "segments": segments["label"].tolist()},
    }


def load_anoshift(cfg: dict, run: runs.Run) -> dict:
    d, dep = cfg["dataset"], cfg["deployment"]
    years = e11.role_years(cfg["split"])
    main = [y for y in years["deployment"] if anoshift.period(y) == dep["main_period"]]
    if not main:
        raise ValueError(f"no deployment year lies in the main period {dep['main_period']}")
    manifest = anoshift.read_manifest(cfg)
    run.log(f"{d['name']} {d['processed_version']}, fingerprint {manifest['fingerprint']}, years "
            + ", ".join(f"{role} {years[role][0]} to {years[role][-1]}" for role in e11.ROLES)
            + f", windows from {', '.join(str(y) for y in main)}")
    frames = {"train": e11.load_years(cfg, years["train"]), "selection": e11.load_years(cfg, years["selection"]),
              "calibration": e11.load_years(cfg, years["calibration"]), "test": e11.load_years(cfg, main)}
    enc = e11.encoder().fit(frames["train"])
    x, label, code = {}, {}, {}
    for name, frame in frames.items():
        x[name] = enc.transform(frame)
        label[name] = frame["label"].to_numpy(np.int8)
        code[name] = frame["attack_type"].cat.codes.to_numpy(np.int8)
    test = frames["test"]
    windows = []
    for year in main:
        rows = np.flatnonzero(test["year"].to_numpy() == year)
        block = splits.order_blocks(np.zeros(len(rows), dtype=np.int8), d["blocks_per_year"])
        for b in range(d["blocks_per_year"]):
            inside = rows[block == b]
            if len(inside):
                windows.append({"label": f"{year} block {b}", "rows": inside, "filtered": False})
    selection_segment = frames["selection"]["year"].to_numpy()
    return {
        "manifest": manifest, "families": e11.FAMILIES, "enc": enc, "x": x, "label": label, "code": code,
        "selection_segment": selection_segment, "windows": windows, "unit": "sessions",
        "shares_cal": lambda n_attack, n_benign: np.asarray(n_attack, dtype=float) / np.sum(n_attack),
        "summary": {"years": years, "main_period": dep["main_period"], "blocks_per_year": d["blocks_per_year"]},
    }


LOADERS = {"litnet2020": load_litnet, "anoshift": load_anoshift}


def run(cfg: dict, run: runs.Run) -> None:
    d, sc, c, cc, sim, cmp, scs = (cfg["dataset"], cfg["scorer"], cfg["certify"], cfg["calibration"],
                                   cfg["simulation"], cfg["comparison"], cfg["scorers"])
    delta, pi_max, part_b = c["delta"], c["pi_max"], c["delta_benign"]
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    pi_cal, draws, seed = cc["prevalence"], sim["draws"], cfg["seed"]
    epsilons, eps_main = [float(e) for e in cmp["epsilons"]], float(cmp["main_epsilon"])
    if eps_main not in epsilons:
        raise ValueError(f"main epsilon {eps_main} is not among the epsilons {epsilons}")
    base, combos = list(scs["base"]), list(scs["combiners"])
    unknown = (set(base) - set(scorers.BASE)) | (set(combos) - set(combiners.COMBINERS))
    if unknown:
        raise ValueError(f"unknown scorers {', '.join(sorted(unknown))}")
    supervised = [name for name in base if name in scorers.SUPERVISED]
    if combos and not supervised:
        raise ValueError("the combiners need at least one supervised base scorer")
    if "s_2d" in combos and "isolation_forest" not in base:
        raise ValueError("s_2d needs isolation_forest among the base scorers")
    if d["name"] not in LOADERS:
        raise ValueError(f"dataset {d['name']} has no E1 split to compare scorers on")
    data = LOADERS[d["name"]](cfg, run)
    families, enc, x, label, code, unit = (data[k] for k in ("families", "enc", "x", "label", "code", "unit"))
    windows = data["windows"]
    n_families = len(families)
    seeds = np.random.SeedSequence(seed).spawn(4)

    # the training rows of the capped scorers, every attack and a benign subsample
    y_train = label["train"]
    attacks_train = np.flatnonzero(y_train == 1)
    benign_train = np.flatnonzero(y_train == 0)
    cap = int(scs["cap"])
    if len(y_train) > cap:
        keep = max(cap - len(attacks_train), 0)
        rng = np.random.default_rng(seeds[0])
        capped_rows = np.sort(np.r_[attacks_train, rng.choice(benign_train, keep, replace=False)])
    else:
        capped_rows = np.arange(len(y_train))
    capped = set(scs["capped"])
    run.log(f"training parts of {len(y_train)} {unit} with {len(attacks_train)} attacks, the capped scorers take "
            f"{len(capped_rows)} of them with every attack")

    # the base scorers, their logits on the selection, calibration and test parts
    fit_rows = first_share(data["selection_segment"], cmp["fit_share"])
    sel_rows = ~fit_rows
    logits, scorer_rows = {}, []
    for name in base:
        rows = capped_rows if name in capped else np.arange(len(y_train))
        model_cfg = sc if name == "lightgbm" else scs[name]
        fitted, fit_seconds = scorers.timed(scorers.fit, name, x["train"][rows], y_train[rows], x["selection"],
                                            label["selection"], model_cfg, seed, enc.columns, enc.categorical_index)
        scored, score_seconds = scorers.timed(
            lambda: {part: scorers.score(name, fitted, x[part], enc.categorical_index)
                     for part in ("selection", "calibration", "test")})
        logits[name] = scored
        sel_auc = auc(scored["selection"][sel_rows], label["selection"][sel_rows])
        cal_auc = auc(scored["calibration"], label["calibration"])
        rounds = scorers.rounds_used(name, fitted)
        scorer_rows.append({"scorer": name, "kind": "base", "label": LABELS[name], "train_rows": int(len(rows)),
                            "train_attacks": int(y_train[rows].sum()), "fit_seconds": fit_seconds,
                            "score_seconds": score_seconds, "rounds": rounds})
        run.log(f"{LABELS[name]} trained on {len(rows)} {unit} with {int(y_train[rows].sum())} attacks in "
                f"{fit_seconds:.0f} s" + (f", {rounds} rounds" if rounds is not None else "")
                + f", scored the selection, calibration and test parts in {score_seconds:.0f} s, selection AUC "
                f"{sel_auc:.5f}, calibration AUC {cal_auc:.5f}")
    del x

    # the calibration sets and the family groups of every epsilon follow from the parts alone, so they are the same
    # for every scorer
    n_benign_cal = int((code["calibration"] == 0).sum())
    n_attack_cal = np.bincount(code["calibration"], minlength=n_families + 1)[1:]
    shares_cal = data["shares_cal"](n_attack_cal, n_benign_cal)
    n_cal = int(round(n_benign_cal / (1 - pi_cal)))
    expected = n_cal * pi_cal * shares_cal
    groups_of, minimum_of = {}, {}
    for eps in epsilons:
        minimum_of[eps] = min_family_size(eps, pi_max, delta - part_b, c["family_floor"])
        groups_of[eps] = family_groups(expected, minimum_of[eps])
        merged = [[f for f, g in zip(families, groups_of[eps]) if g == k] for k in range(groups_of[eps].max() + 1)]
        run.log(f"epsilon {eps}, a family needs {minimum_of[eps]} calibration attacks to stand alone, families "
                + ", ".join("+".join(f) for f in merged))
    run.log(f"calibration sets of {n_cal} {unit} at prevalence {pi_cal}, expected attacks per family "
            + ", ".join(f"{f} {e:.0f}" for f, e in zip(families, expected)))

    def every_mix_share(score_sel, code_sel, eps, rows):
        # the benign share of the given selection rows dismissed by the loosest threshold safe for every family at
        # pi_max, on a grid of those rows
        grid = grid_from(score_sel[rows & (code_sel == 0)], c["grid_size"], c.get("grid_tail", []))
        counts = judge.class_counts(score_sel[rows], code_sel[rows], grid, n_families)
        b, m_family, _, _ = judge.rates(counts)
        j = judge.every_mix_threshold(pi_max, m_family, b, groups_of[eps], shares_cal, eps)
        return float(b[j]) if j >= 0 else 0.0

    # S-best by the second part of the selection parts, the low tail target by the first
    yardstick = {name: every_mix_share(logits[name]["selection"], code["selection"], eps_main, sel_rows)
                 for name in base}
    best = max(supervised, key=lambda name: yardstick[name])
    run.log("benign share of the second part of the selection parts dismissed by the loosest threshold safe for "
            f"every family at epsilon {eps_main}, " + ", ".join(f"{LABELS[n]} {v:.4f}" for n, v in yardstick.items())
            + f", S-best is {LABELS[best]}")
    scores = {name: logits[name] for name in base}
    for name in base:
        scorer_rows[base.index(name)]["selection_yardstick"] = yardstick[name]
    if combos:
        std = combiners.Standardizer().fit(np.column_stack([logits[n]["selection"][fit_rows] for n in base]))
        z = {part: std.transform(np.column_stack([logits[n][part] for n in base]))
             for part in ("selection", "calibration", "test")}
        z_fit, y_fit, family_fit = z["selection"][fit_rows], label["selection"][fit_rows], code["selection"][fit_rows]
        combiner_info = {"base": base, "standardizer": std.to_dict(), "best": best}
        sup = [base.index(n) for n in supervised]
        for name in combos:
            t0 = pd.Timestamp.now()
            if name == "s_best":
                scores[name] = scores[best]
                combiner_info[name] = {"chosen": best}
            elif name == "s_mean":
                scores[name] = {part: combiners.mean_logit(np.column_stack([logits[n][part] for n in supervised]))
                                for part in z}
                combiner_info[name] = {"over": supervised}
            elif name == "s_stack":
                params = combiners.fit_stack(z_fit, y_fit, scs["stack"])
                scores[name] = {part: combiners.score_linear(params, z[part]) for part in z}
                combiner_info[name] = {"weights": dict(zip(base, params["w"].tolist())), "bias": params["b"]}
            elif name == "s_gate":
                params = combiners.fit_gate(z_fit, y_fit, scs["gate"], seed)
                scores[name] = {part: combiners.score_gate(params, z[part]) for part in z}
                gate_mean = combiners.gate_forward(params, z_fit.astype(np.float64))[0].mean(0)
                combiner_info[name] = {"mean_gate": dict(zip(base, gate_mean.tolist())),
                                       "loss_first": params["trace"][0], "loss_last": params["trace"][-1]}
            elif name == "s_tail":
                share = scs["tail"].get("benign_share")
                if share is None:
                    share = every_mix_share(logits[best]["selection"], code["selection"], eps_main, fit_rows)
                params = combiners.fit_tail(z_fit, y_fit, family_fit, scs["tail"], seed, share)
                scores[name] = {part: combiners.score_linear(params, z[part]) for part in z}
                combiner_info[name] = {"weights": dict(zip(base, params["w"].tolist())), "bias": float(params["b"]),
                                       "benign_share": share, "tail_loss_first": params["trace"][0],
                                       "tail_loss_last": params["trace"][-1]}
            elif name == "s_2d":
                sorted_best = np.sort(logits[best]["selection"][fit_rows & (code["selection"] == 0)])
                sorted_novelty = np.sort(logits["isolation_forest"]["selection"][fit_rows & (code["selection"] == 0)])
                scores[name] = {part: combiners.two_d(combiners.benign_rank(sorted_best, logits[best][part]),
                                                      combiners.benign_rank(sorted_novelty,
                                                                            logits["isolation_forest"][part]))
                                for part in z}
                combiner_info[name] = {"supervised": best, "novelty": "isolation_forest"}
            seconds = (pd.Timestamp.now() - t0).total_seconds()
            scorer_rows.append({"scorer": name, "kind": "combiner", "label": LABELS[name], "fit_seconds": seconds,
                                "selection_yardstick": every_mix_share(scores[name]["selection"], code["selection"],
                                                                       eps_main, sel_rows)})
            details = []
            for k, v in combiner_info[name].items():
                if isinstance(v, dict):
                    details.append(f"{k} " + ", ".join(f"{n} {w:.3f}" for n, w in v.items()))
                elif isinstance(v, list):
                    details.append(f"{k} " + ", ".join(str(n) for n in v))
                elif isinstance(v, float):
                    details.append(f"{k} {v:.4f}")
                else:
                    details.append(f"{k} {v}")
            run.log(f"{LABELS[name]} fit on the first part of the selection parts in {seconds:.0f} s, "
                    + ", ".join(details))
        run.save_json(combiner_info, "combiners")
        del z, z_fit
    if cmp.get("save_scores", True):
        test_window = np.full(len(label["test"]), -1, dtype=np.int16)
        for i, w in enumerate(windows):
            test_window[w["rows"]] = i
        run.save_arrays("scores", base=np.array(base), fit=fit_rows,
                        **{f"{part}_{key}": label[part] if key == "label" else code[part]
                           for part in ("selection", "calibration", "test") for key in ("label", "code")},
                        test_window=test_window,
                        **{f"{name}_{part}": logits[name][part] for name in base
                           for part in ("selection", "calibration", "test")})

    # every scorer certified at every epsilon, the same calibration draws and window draws for all of them
    code_sel, code_cal, code_test = code["selection"], code["calibration"], code["test"]
    claims, pooled_rows, window_rows = [], [], []
    for row in scorer_rows:
        name = row["scorer"]
        s = scores[name]
        grid = grid_from(s["selection"][sel_rows & (code_sel == 0)], c["grid_size"], c.get("grid_tail", []))
        sel_counts = judge.class_counts(s["selection"][sel_rows], code_sel[sel_rows], grid, n_families)
        cal_counts = judge.class_counts(s["calibration"], code_cal, grid, n_families)
        win_counts = [judge.class_counts(s["test"][w["rows"]], code_test[w["rows"]], grid, n_families)
                      for w in windows]
        b_sel, m_sel_family, _, _ = judge.rates(sel_counts)
        b_cal, m_cal_family, _, _ = judge.rates(cal_counts)
        m_cal = shares_cal @ m_cal_family
        with np.errstate(invalid="ignore"):
            hardest = {f: benign_at_miss(m, b_cal, cmp["miss_rate"]) for f, m, sh in zip(families, m_cal_family,
                                                                                         shares_cal) if sh > 0}
        hardest_family = min(hardest, key=hardest.get)
        probability = name in scorers.PROBABILITY or name in combiners.PROBABILITY
        ece = (expected_calibration_error(combiners.sigmoid(s["calibration"]), label["calibration"], cmp["ece_bins"])
               if probability else np.nan)
        row.update({
            "grid_thresholds": len(grid), "selection_auc": auc(s["selection"][sel_rows], label["selection"][sel_rows]),
            "calibration_auc": auc(s["calibration"], label["calibration"]),
            "benign_at_miss_pooled": benign_at_miss(m_cal, b_cal, cmp["miss_rate"]),
            "benign_at_miss_hardest": hardest[hardest_family], "hardest_family": hardest_family, "ece": ece,
            "best": name == best,
        })
        filtered = np.array([w["filtered"] for w in windows])
        benign_test = np.array([wc["sizes"][0] for wc in win_counts])
        reference_benign = int(round(benign_test[~filtered].mean())) if (~filtered).any() else 0
        for i, (w, wc) in enumerate(zip(windows, win_counts)):
            with np.errstate(divide="ignore", invalid="ignore"):
                b_w, m_w_family, _, own = judge.rates(wc)
            window_rows.append({"scorer": name, "window": i, "segment": w["label"], "filtered": w["filtered"],
                                "test_rows": int(wc["sizes"].sum()), "benign": int(wc["sizes"][0]),
                                **{f: int(n) for f, n in zip(families, wc["sizes"][1:])},
                                "auc": auc(s["test"][w["rows"]], label["test"][w["rows"]]) if not w["filtered"]
                                else np.nan})
        for eps in epsilons:
            groups = groups_of[eps]
            cal = draw_counts(np.random.default_rng(seeds[1]), n_cal, pi_cal, shares_cal, b_cal, m_cal_family, draws)
            grouped_below, grouped_n = group_rows(cal.family_below, cal.n_family, groups)
            offline = {"forge_simplex": certify(cal.benign_below, cal.n_benign, grouped_below, grouped_n, eps, delta,
                                                pi_max, part_b, **start)}
            index = offline["forge_simplex"]["index"]
            hit = index >= 0
            benign_dismissed = float(np.where(hit, b_cal[np.maximum(index, 0)], 0.0).mean())
            rng = np.random.default_rng(seeds[2])
            values, rows = {}, []
            for i, (w, wc) in enumerate(zip(windows, win_counts)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    b_w, m_w_family, _, own = judge.rates(wc)
                test_attacks = wc["sizes"][1:]
                populations = {"reference": (b_cal, m_cal_family, n_attack_cal)}
                if not w["filtered"]:
                    populations["real"] = (b_w, m_w_family, test_attacks)
                benign_w = reference_benign if w["filtered"] else int(wc["sizes"][0])
                full = int(round(benign_w / (1 - pi_max)))
                claim = e12.in_claim(own, shares_cal, groups)
                j_every = judge.every_mix_threshold(pi_max, m_cal_family, b_cal, groups, shares_cal, eps)
                for arm, (b, m_family, attacks) in populations.items():
                    n_w, subsampled = full, False
                    if arm == "real" and np.any(attacks < full * pi_max * own):
                        need = own > 0
                        if (attacks[need] == 0).any():
                            continue
                        n_w, subsampled = int(np.floor((attacks[need] / (pi_max * own[need])).min())), True
                    drawn = draw_counts(rng, n_w, pi_max, own, b, m_family, draws)
                    attack_below = drawn.family_below.sum(1)
                    window_below = drawn.benign_below + attack_below
                    m = own @ m_family
                    every_share = (float(pi_max * m[j_every] + (1 - pi_max) * b[j_every]) if j_every >= 0 else 0.0) \
                        if arm == "reference" else np.nan
                    rows += judge.evaluate(["forge_simplex"], offline, None, pi_max, m, b, attack_below, window_below,
                                           eps, values,
                                           {"scorer": name, "epsilon": eps, "arm": arm, "window": i,
                                            "segment": w["label"], "prevalence": pi_max, "mix": "window",
                                            "window_sessions": n_w, "benign_subsampled": subsampled,
                                            "in_claim": claim, "draws": draws,
                                            **judge.best_thresholds(pi_max, pi_max, m, b, eps),
                                            "best_every_mix_index": j_every,
                                            "best_every_mix_dismissed": every_share})
            cell = pd.DataFrame(rows)
            claims.append(cell)
            for arm, g_arm in cell.groupby("arm", sort=False):
                for subset, g in [("all", g_arm), ("in_claim", g_arm[g_arm["in_claim"]])]:
                    if g.empty:
                        continue
                    certified = np.concatenate([values[(arm, "forge_simplex", pi_max, "window", w)]
                                                for w in g["window"]])
                    pooled_rows.append({"scorer": name, "kind": row["kind"], "epsilon": eps, "arm": arm,
                                        "subset": subset, **judge.pooled(g, certified, draws),
                                        "best_every_mix_dismissed": g["best_every_mix_dismissed"].mean(),
                                        "benign_dismissed_calibration": benign_dismissed,
                                        "family_minimum": minimum_of[eps], "groups": int(groups.max() + 1)})
        shown = [r for r in pooled_rows if r["scorer"] == name and r["subset"] == "all"]
        for arm in ARMS:
            lines = [r for r in shown if r["arm"] == arm]
            if lines:
                run.log(f"{LABELS[name]}, {ARM_LABELS[arm]} at prevalence {pi_max}, certified share of the window "
                        + unit + " dismissed "
                        + ", ".join(f"{r['dismissed_mean']:.4f} at {r['epsilon']}" for r in lines)
                        + ", refused " + ", ".join(f"{r['refused']:.3f}" for r in lines)
                        + ", FOR above epsilon in " + ", ".join(f"{r['violation_rate']:.4f}" for r in lines)
                        + " of draw and window pairs"
                        + (", by the loosest threshold safe for every mix "
                           + ", ".join(f"{r['best_every_mix_dismissed']:.4f}" for r in lines)
                           if arm == "reference" else
                           ", by the loosest threshold safe up to pi_max "
                           + ", ".join(f"{r['best_up_to_pi_max_dismissed']:.4f}" for r in lines)))
        run.log(f"{LABELS[name]}, selection AUC {row['selection_auc']:.5f}, calibration AUC "
                f"{row['calibration_auc']:.5f}, benign share dismissed at a miss rate of {cmp['miss_rate']} of the "
                f"pooled calibration attacks {row['benign_at_miss_pooled']:.4f} and of the hardest family "
                f"{row['benign_at_miss_hardest']:.4f}, {hardest_family}"
                + (f", expected calibration error {ece:.4f}" if probability else ""))

    scorer_table = pd.DataFrame(scorer_rows)
    run.save_table(scorer_table, "scorers")
    claims_table = pd.concat(claims, ignore_index=True)
    run.save_table(claims_table, "claims")
    by_epsilon = pd.DataFrame(pooled_rows)
    run.save_table(by_epsilon, "by_epsilon")
    run.save_table(pd.DataFrame(window_rows), "windows")

    # the winner at the main epsilon on the reference windows in the claim, and whether calibration goes with the
    # certified share across the scorers that give a probability
    main = by_epsilon[(by_epsilon["epsilon"] == eps_main) & (by_epsilon["arm"] == "reference")
                      & (by_epsilon["subset"] == "all")].set_index("scorer")
    ranked = main["dismissed_mean"].sort_values(ascending=False)
    winner = str(ranked.index[0])
    with_ece = scorer_table[scorer_table["ece"].notna() & scorer_table["scorer"].isin(main.index)]
    correlation = (float(spearmanr(with_ece["ece"], main.loc[with_ece["scorer"], "dismissed_mean"]).statistic)
                   if len(with_ece) >= 3 else np.nan)
    run.log(f"at epsilon {eps_main} on the reference windows the certified share is "
            + ", ".join(f"{LABELS[n]} {v:.4f}" for n, v in ranked.items()) + f", the winner is {LABELS[winner]}"
            + (f", rank correlation of the expected calibration error with the certified share {correlation:.3f} "
               f"over {len(with_ece)} scorers" if np.isfinite(correlation) else ""))
    run.save_json({
        "dataset": d["name"], "processed_version": d["processed_version"], "fingerprint": data["manifest"]["fingerprint"],
        **data["summary"], "epsilons": epsilons, "main_epsilon": eps_main, "delta": delta, "delta_benign": part_b,
        "pi_max": pi_max, **start, "grid_size": c["grid_size"], "grid_tail": c.get("grid_tail", []),
        "family_floor": c["family_floor"], "family_minimum": {str(e): int(m) for e, m in minimum_of.items()},
        "families": families,
        "groups": {str(e): [[f for f, g in zip(families, gr) if g == k] for k in range(gr.max() + 1)]
                   for e, gr in groups_of.items()},
        "calibration_prevalence": pi_cal, "calibration_sessions": n_cal,
        "calibration_attacks": dict(zip(families, n_attack_cal.tolist())),
        "calibration_expected_attacks": dict(zip(families, expected.tolist())), "draws": draws,
        "windows": len(windows), "base": base, "combiners": combos, "cap": cap, "capped": sorted(capped),
        "capped_rows": int(len(capped_rows)), "fit_share": cmp["fit_share"], "fit_rows": int(fit_rows.sum()),
        "selection_rows": int(sel_rows.sum()), "best": best, "winner": winner, "unit": unit,
        "selection_yardstick": {r["scorer"]: r["selection_yardstick"] for r in scorer_rows},
        "certified": {n: float(v) for n, v in ranked.items()},
        "refused": {n: float(v) for n, v in main["refused"].items()},
        "every_mix": {n: float(v) for n, v in main["best_every_mix_dismissed"].items()},
        "ece_correlation": correlation, "miss_rate": cmp["miss_rate"], "ece_bins": cmp["ece_bins"],
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _line(scorer: str, kind: str) -> dict:
    # base scorers in solid lines, combiners dashed, every scorer with its own marker and a color from the fixed list
    names = list(scorers.BASE) if kind == "base" else [c for c in combiners.COMBINERS if c != "s_best"]
    return {"color": style.COLORS[names.index(scorer) % len(style.COLORS)], "marker": MARKERS.get(scorer, "o"),
            "linestyle": "-" if kind == "base" else "--", "label": LABELS.get(scorer, scorer)}


def _shown(table: pd.DataFrame) -> pd.DataFrame:
    # S-best repeats a base scorer, so the figures leave it out
    return table[table["scorer"] != "s_best"]


def fig_dismissed_by_epsilon(run_dir) -> plt.Figure:
    style.use()
    summary = runs.load_json(run_dir, "summary")
    table = _shown(runs.load_table(run_dir, "by_epsilon"))
    table = table[(table["arm"] == "reference") & (table["subset"] == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_DOUBLE, 2.6), sharex=True)
    # left, the certified share of the reference windows dismissed under each scorer by epsilon. right, the share the
    # loosest threshold safe for every mix dismisses, the ceiling of each scorer under the certificate
    for scorer, g in table.groupby("scorer", sort=False):
        g = g.sort_values("epsilon")
        line = _line(scorer, g["kind"].iloc[0])
        axes[0].plot(g["epsilon"], g["dismissed_mean"], markersize=3.5, linewidth=1.2, **line)
        axes[1].plot(g["epsilon"], g["best_every_mix_dismissed"], markersize=3.5, linewidth=1.2, **line)
    for ax, letter, name in zip(axes, "ab", ["certified share dismissed", "share safe for every mix"]):
        ax.set_xscale("log")
        ax.axvline(summary["main_epsilon"], color=style.GRAY, linestyle="-.", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(name)
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[0], [], ncol=4)
    return fig


def fig_ece_vs_dismissed(run_dir) -> plt.Figure:
    style.use()
    summary = runs.load_json(run_dir, "summary")
    scorers_table = _shown(runs.load_table(run_dir, "scorers"))
    table = runs.load_table(run_dir, "by_epsilon")
    main = table[(table["epsilon"] == summary["main_epsilon"]) & (table["arm"] == "reference")
                 & (table["subset"] == "all")].set_index("scorer")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.6))
    # every scorer with a probability, its expected calibration error against its certified share at the main epsilon
    for r in scorers_table.itertuples():
        if not np.isfinite(r.ece) or r.scorer not in main.index:
            continue
        line = {k: v for k, v in _line(r.scorer, r.kind).items() if k != "linestyle"}
        ax.plot(r.ece, main.loc[r.scorer, "dismissed_mean"], linestyle="none", markersize=4.5, **line)
    ax.set_xlabel("expected calibration error on the calibration parts")
    ax.set_ylabel("certified share dismissed")
    ax.set_ylim(0, 1)
    layout.legend_below(fig, ax, [], ncol=3)
    return fig


def fig_dismissed_by_scorer(run_dir) -> plt.Figure:
    style.use()
    summary = runs.load_json(run_dir, "summary")
    table = _shown(runs.load_table(run_dir, "by_epsilon"))
    main = table[(table["epsilon"] == summary["main_epsilon"]) & (table["subset"] == "all")]
    names = list(dict.fromkeys(main["scorer"]))
    fig, ax = plt.subplots(figsize=(style.WIDTH_DOUBLE, 2.6))
    # the certified share dismissed under each scorer at the main epsilon on the reference windows, filled, and on
    # the real windows, hollow, with the loosest threshold safe for every mix of the reference windows as a tick
    width = 0.38
    x = np.arange(len(names))
    for k, (arm, fill) in enumerate([("reference", True), ("real", False)]):
        g = main[main["arm"] == arm].set_index("scorer")
        heights = [g.loc[n, "dismissed_mean"] if n in g.index else 0.0 for n in names]
        colors = [_line(n, main[main["scorer"] == n]["kind"].iloc[0])["color"] for n in names]
        ax.bar(x + (k - 0.5) * width, heights, width, color=colors if fill else "none",
               edgecolor=colors, linewidth=0.8, label=ARM_LABELS[arm])
    g = main[main["arm"] == "reference"].set_index("scorer")
    ticks = [g.loc[n, "best_every_mix_dismissed"] if n in g.index else np.nan for n in names]
    ax.plot(x - 0.5 * width, ticks, linestyle="none", marker="_", markersize=9, color=style.INK,
            label="safe for every mix")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share dismissed")
    filled = plt.Rectangle((0, 0), 1, 1, color=style.INK)
    hollow = plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=style.INK)
    tick = plt.Line2D([], [], linestyle="none", marker="_", markersize=9, color=style.INK)
    fig.tight_layout()
    fig.legend([filled, hollow, tick], [ARM_LABELS["reference"], ARM_LABELS["real"], "safe for every mix"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    return fig


FIGURES = {"dismissed_by_epsilon": fig_dismissed_by_epsilon, "ece_vs_dismissed": fig_ece_vs_dismissed,
           "dismissed_by_scorer": fig_dismissed_by_scorer}
