"""checks on an e12 run from its saved results only. why LTT refuses, where the low tail of LAND sits, which windows
break the floor of the FORGE certificates, and the loosest grid threshold that is safe for every family at once, the
yardstick between the best threshold of one window and the certified one.

    python analysis/e12_checks.py drive/runs/e12_litnet_prevalence/<run_id>
"""

import sys

import numpy as np
import yaml

from forge import runs
from forge.certify.bounds import clopper_pearson_upper
from forge.certify.certificate import certify, for_from_rates, group_rows, ltt_iid
from forge.data.synthetic import draw_counts
from forge.eval import windows as judge

run_dir = sys.argv[1]
summary = runs.load_json(run_dir, "summary")
with open(f"{run_dir}/config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
hist = runs.load_arrays(run_dir, "histograms")
saved = runs.load_arrays(run_dir, "draws")
claims = runs.load_table(run_dir, "claims")
cuts = runs.load_table(run_dir, "cuts")
families = hist["families"].tolist()
eps, delta, pi_max, part_b = summary["epsilon"], summary["delta"], summary["pi_max"], summary["delta_benign"]
pi_cal, draws, main = summary["calibration_prevalence"], summary["draws"], summary["main_prevalence"]
b_cal, m_cal, _, _ = judge.rates({"benign_below": hist["calibration_benign_below"],
                                  "family_below": hist["calibration_family_below"],
                                  "sizes": hist["calibration_sizes"]})
shares, groups, n_cal, grid = hist["shares"], hist["groups"], summary["calibration_sessions"], hist["grid"]
kappa = eps * (1 - pi_max) / ((1 - eps) * pi_max)

print("thresholds per draw")
for method in ["forge_simplex", "forge_pooled", "ltt_iid", "conformal_fnr", "fixed_sel"]:
    values, counts = np.unique(saved[f"{method}_index"], return_counts=True)
    print(f"  {method}, " + ", ".join(f"index {v} in {c}" for v, c in zip(values, counts)))

print("\n1. LTT refusals")
# the calibration sets drawn again from the saved counts and the first of the documented streams. the class sizes and
# every FORGE threshold come back exactly, while the counts in the lowest bins differ in about one draw in a hundred,
# likely from another numpy version than the one of the run, so the saved LTT thresholds are used
seed = np.random.SeedSequence(config["seed"]).spawn(1)[0]
cal = draw_counts(np.random.default_rng(seed), n_cal, pi_cal, shares, b_cal, m_cal, draws)
assert np.array_equal(saved["counts"], np.column_stack([cal.n_benign, cal.n_family]))
pooled_below = cal.family_below.sum(1)
grouped = group_rows(cal.family_below, cal.n_family, groups)
again = certify(cal.benign_below, cal.n_benign, *grouped, eps, delta, pi_max, part_b,
                start_misses=config["certify"]["start_misses"], start_slack=config["certify"]["start_slack"])
ltt = ltt_iid(cal.benign_below, pooled_below, eps, delta)["index"]
print(f"  thresholds drawn again equal to the saved ones in {np.mean(again['index'] == saved['forge_simplex_index']):.3f}"
      f" of the sets for FORGE, {np.mean(ltt == saved['ltt_iid_index']):.3f} for LTT")
below = hist["calibration_family_below"][:, 0]
print("  calibration attacks at or below the first grid threshold, "
      + ", ".join(f"{f} {n} of {s}" for f, n, s in zip(families, below, hist["calibration_sizes"][1:]) if n))
expected = (n_cal * pi_cal * shares * below / hist["calibration_sizes"][1:]).sum()
dismissed = np.median(cal.benign_below[:, 0] + pooled_below[:, 0])
print(f"  per calibration set, flows dismissed at the first threshold {dismissed:.0f} median, attacks there {expected:.3f}"
      " expected, upper bound of LTT with 0, 1, 2 of them "
      + ", ".join(f"{clopper_pearson_upper(k, dismissed, delta):.5f}" for k in range(3)))
refused, hit = saved["ltt_iid_index"] < 0, pooled_below[:, 0] >= 1
print(f"  LTT refused in {refused.mean():.3f} of the sets, sets drawn again with an attack at or below the first "
      f"threshold {hit.mean():.3f}, agreeing in {np.mean(refused == hit):.3f}, Poisson {1 - np.exp(-expected):.3f}")

print("\n2. low tail of LAND")
scores = runs.load_arrays(run_dir, "scores")
mine = scores["family"] == list(scores["families"]).index("land")
order = np.argsort(scores["row"][mine], kind="stable")
score, part = scores["score"][mine][order], scores["part"][mine][order]
land = cuts[cuts["families"] == "land"].set_index("part")
# the scores hold the selection, calibration and test parts, so the rank of an attack in time order counts the
# attacks of the training part first
rank = land.loc["train", "attacks"] + np.arange(len(score))
k = families.index("land")
print(f"  calibration miss rate at grid indices 638 to 643 " + ", ".join(f"{v:.5f}" for v in m_cal[k, 638:644])
      + ", benign share " + ", ".join(f"{v:.4f}" for v in b_cal[638:644]))
for method in ["forge_simplex", "forge_pooled", "ltt_iid"]:
    j = int(np.median(saved[f"{method}_index"][saved[f"{method}_index"] >= 0]))
    low = score <= grid[j]
    print(f"  at the {method} threshold, index {j}, score {grid[j]:.2f}, benign share {b_cal[j]:.4f}, LAND attacks at "
          f"or below it, selection {(low & (part == 1)).sum()} of {(part == 1).sum()}, calibration "
          f"{(low & (part == 2)).sum()} of {(part == 2).sum()}, test {(low & (part == 3)).sum()} of {(part == 3).sum()}")
low = score <= grid[int(np.median(saved["ltt_iid_index"][saved["ltt_iid_index"] >= 0]))]
first, last = rank[low].min(), rank[low].max()
burst = (rank >= first) & (rank <= last)
print(f"  ranks in time order of the low LAND attacks among the {land['attacks'].sum()} of the capture, {first} to {last},"
      f" {(low & burst).sum()} of the {burst.sum()} attacks there and {(low & ~burst).sum()} elsewhere, the calibration "
      f"part holds ranks {rank[part == 2].min()} to {rank[part == 2].max()}, te {land.loc['calibration', 'first_te']} to "
      f"{land.loc['calibration', 'last_te']}")
edges = np.arange(first - first % 250, last + 251, 250)
low_per, _ = np.histogram(rank[low], edges)
print("  low attacks per 250 ranks from " + str(edges[0]) + ", " + ", ".join(str(n) for n in low_per))
print("  score quantiles 0.1, 0.5, 0.9 of the calibration attacks of LAND inside those ranks "
      + ", ".join(f"{v:.2f}" for v in np.quantile(score[burst], [0.1, 0.5, 0.9])) + ", outside them "
      + ", ".join(f"{v:.2f}" for v in np.quantile(score[(part == 2) & ~burst], [0.1, 0.5, 0.9])))

print(f"\n3. floor of the FORGE certificates at prevalence {main}, share of draws below it, own mix")
at = claims[(claims["prevalence"] == main) & (claims["mix"] == "window")]
for method in ["forge_simplex", "forge_pooled", "forge_online"]:
    for arm in summary["arms"]:
        g = at[(at["method"] == method) & (at["arm"] == arm)].sort_values("window")
        print(f"  {method}, {arm}, " + ", ".join(f"{r.segment} {r.floor_violations / r.draws:.3f}"
                                                 for r in g.itertuples()))

print("\n4. loosest grid threshold whose FOR at pi_max stays at or below epsilon")
own = hist["window_sizes"][:, 1:] / hist["window_sizes"][:, 1:].sum(1, keepdims=True)


def reference_share(j: int, mix: str) -> float:
    # the mean share of a reference window's flows dismissed at the main prevalence, over all windows
    return float(np.mean([main * (judge.mix_shares(mix, o, shares, families) @ m_cal[:, j]) + (1 - main) * b_cal[j]
                          for o in own]))


safe_family = for_from_rates(pi_max, m_cal, b_cal) <= eps
for name, safe in [("for every family", safe_family.all(0)),
                   ("for every family but LAND", np.delete(safe_family, k, 0).all(0)),
                   ("at the calibration mix", for_from_rates(pi_max, shares @ m_cal, b_cal) <= eps)]:
    j = int(np.flatnonzero(safe)[-1])
    print(f"  {name}, index {j}, benign share {b_cal[j]:.4f}, mean share of a reference window dismissed at prevalence "
          f"{main} {reference_share(j, 'window'):.4f} own mix, {reference_share(j, 'calibration'):.4f} calibration mix")
j = int(np.flatnonzero(safe_family.all(0))[-1])
print(f"  past index {j} the first to fail is " + ", ".join(f for f, ok in zip(families, safe_family[:, j + 1]) if not ok)
      + f", whose miss rate there may be at most {kappa * b_cal[j + 1]:.5f}")

print(f"\n5. offline and online FORGE on the reference windows at prevalence {main}")
online = saved["online_index"][summary["arms"].index("reference"), :, summary["prevalences"].index(main)]
print(f"  online index equal to the offline one in {np.mean(online == saved['forge_simplex_index']):.4f} of the window, "
      "mix and draw cells")
