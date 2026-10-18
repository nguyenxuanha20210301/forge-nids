"""e24, RQ2 on UGR16, FORGE-Monitor on the dismissed streams of the thresholds of e13 through the 86 windows of six
hours of August 2016 in order, against the baselines that read the same audits, on a simulated stream and on the
replayed one, with the attack prevalence moving as it did.

the streams come from an e13 run as those of e21 come from an e11 run, and this module shares the stream generator,
the judges and the arms of monitor_streams. what differs is the source and the clock.

    windows      the deployment windows of e13 in time order, each thinned to its own mix of families at the
                 prevalence of a composition. a window short of attacks keeps all of them and has its benign flows
                 subsampled, the rule of the real windows of e13, so the thinned window of a draw is the one e13
                 certified and its online thresholds are checked against the saved ones. a window without attacks
                 has no mix of its own and goes through at prevalence 0 with all its benign flows, its online
                 threshold drawn afresh
    natural      the composition that keeps every window as it was, all its flows at its own prevalence, which only
                 a dataset with a clock and a prevalence that moves can give, so the stream sees the launched attacks
                 stop, the spam surge and fade and the daily cycle as they happened. its online thresholds are those
                 e13 saved for the natural windows, and its replay keeps every row, so the coins are the only
                 simulation left
    replay       the per flow scores e13 saved with the deployment window of every flow, in te order, so a window is
                 one stretch of the saved arrays, checked against the window histograms of the run

arms, all on the same audits. monitor, one sequence for the whole deployment. monitor_week, a sequence per clock
week from the first window, the operating rule of e21 with the week in place of the year, a last week with fewer
windows than the config's minimum joined to the one before, so the two windows of August 29 run with the third
week. monitor_window, a sequence per window at delta over the windows. any_start, the rule of e22, a sequence started
at every window at delta over the windows with the first alarm of any of them. none, window_audit and naive as in
e21.

the tables keep the columns of e21, where a block is a window here, so the pooled E3 table reads both experiments.
claims per stream, policy, composition and arm. by_window per stream, policy, composition and window, with the
window's start, its segment, its prevalence under the composition and whether its benign flows were subsampled.
windows, the table of the source run with the segment of each window. draws.npz and trace.npz as e21 saves them.
figures bounds_over_time, for_by_window on the clock, alarm_by_policy and leakage_by_policy.
"""

import string
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from forge import runs
from forge.certify.certificate import certify, for_from_rates, group_rows, online_certifier
from forge.data.synthetic import draw_counts
from forge.eval.metrics import share_with_interval
from forge.eval.windows import rates
from forge.experiments.monitor_streams import HIGH, MAIN_STREAM, OFFLINE, POLICIES, STATE_CODES, STREAM_LABELS, STREAMS
from forge.experiments.monitor_streams import POLICY_LABELS as E21_POLICY_LABELS
from forge.experiments.monitor_streams import _policy_line as _e21_policy_line
from forge.experiments.monitor_streams import (_quantiles, _stream_word, _threshold_rate, _trace_streams, _with_prefix,
                                               audited_stream, block_counts, judge_stream, load_source, of_stream,
                                               recalibrate)
from forge.experiments.monitor_sweep import any_start_monitor
from forge.monitor import rule
from forge.monitor.replay import (assemble_stream, block_rows, check_block, dismissed_stream, thin, thin_rates,
                                  window_counts)
from forge.monitor.sequence import KINDS, Sequence, Tables
from forge.viz import layout, style
from forge.viz.export import save_figure

# the fixed threshold of e13 is tuned on the selection part, not on a selection year
POLICY_LABELS = {**E21_POLICY_LABELS, "fixed_sel": "fixed threshold, selection part"}
ARMS = ["monitor", "monitor_week", "monitor_window", "any_start", "none", "window_audit", "naive"]
MONITORS = ["monitor", "monitor_week", "monitor_window", "any_start"]
ARM_LABELS = {"monitor": "FORGE-Monitor", "monitor_week": "FORGE-Monitor, a sequence per week",
              "monitor_window": "FORGE-Monitor, a sequence per window",
              "any_start": "FORGE-Monitor, a sequence from every window", "none": "no monitoring",
              "window_audit": "per window audit, Clopper-Pearson", "naive": "running estimate, no bound"}
ARM_SHORT = {"monitor": "FORGE-Monitor", "monitor_week": "FORGE-Monitor per week",
             "monitor_window": "FORGE-Monitor per window", "any_start": "FORGE-Monitor from every window",
             "window_audit": "per window audit", "naive": "running estimate"}
ARM_MARKERS = {"monitor": "o", "monitor_week": "D", "monitor_window": "v", "any_start": "P", "window_audit": "s",
               "naive": "^", "none": "x"}
# the four monitors share the blue of FORGE in four shades, the baselines ink and gray as in e21
ARM_COLORS = {"monitor": style.COLORS[0], "monitor_week": style.SEQUENTIAL[8], "monitor_window": style.SEQUENTIAL[1],
              "any_start": style.SEQUENTIAL[6], "window_audit": style.INK, "naive": style.GRAY, "none": style.GRAY}


def composition_word(co: dict) -> str:
    # the prevalence of a composition for a label, or the window's own one
    pi = co.get("prevalence")
    return "the window's own prevalence" if pi is None else f"attack prevalence {style.percent(pi)}"


def segments_of(windows: pd.DataFrame, length: str, minimum: int) -> tuple[np.ndarray, int]:
    """the segment of every window for the sequence per week, the windows cut into stretches of length from the
    start of the first one, numbered from 0 without gaps, and a last stretch with fewer windows than minimum joined
    to the one before. returns the segments and the number of windows joined."""
    if minimum < 1:
        raise ValueError(f"minimum_windows is {minimum}, expected 1 or more")
    starts = pd.to_datetime(windows["start"])
    if (starts.diff().dropna() <= pd.Timedelta(0)).any():
        raise ValueError("the windows are not in time order")
    stretch = ((starts - starts.iloc[0]) // pd.Timedelta(length)).to_numpy().astype(np.int64)
    segments = np.unique(stretch, return_inverse=True)[1].astype(np.int64).reshape(-1)
    last = int(segments.max())
    joined = 0
    if last > 0 and int((segments == last).sum()) < minimum:
        joined = int((segments == last).sum())
        segments[segments == last] = last - 1
    return segments, joined


def source_seeds(source: dict) -> list:
    # the seed streams of the source run, one for the calibration sets, one per window, prevalence and mix of each
    # arm, and one per window at its own prevalence
    dep, ssum = source["cfg"]["deployment"], source["summary"]
    cells = int(ssum["windows"]) * len(dep["prevalences"]) * len(dep["mixes"])
    return np.random.SeedSequence(source["cfg"]["seed"]).spawn(1 + len(ssum["arms"]) * cells + int(ssum["windows"]))


def window_stream(source: dict, i: int, pi: float) -> int:
    # the index of the seed stream e13 used for the real window i at this prevalence and the window mix
    dep, ssum = source["cfg"]["deployment"], source["summary"]
    prevalences, mixes = dep["prevalences"], dep["mixes"]
    if pi not in prevalences or "window" not in mixes:
        raise ValueError(f"prevalence {pi} with the window mix has no window in the source run, which has prevalences "
                         f"{prevalences} and mixes {mixes}")
    cells = int(ssum["windows"]) * len(prevalences) * len(mixes)
    return (1 + ssum["arms"].index("real") * cells + (i * len(prevalences) + prevalences.index(pi)) * len(mixes)
            + mixes.index("window"))


def natural_stream(source: dict, i: int) -> int:
    # the index of the seed stream e13 used for window i at its own prevalence
    dep, ssum = source["cfg"]["deployment"], source["summary"]
    cells = int(ssum["windows"]) * len(dep["prevalences"]) * len(dep["mixes"])
    return 1 + len(ssum["arms"]) * cells + i


def window_plan(counts: dict, co: dict) -> dict:
    """the window of a composition by the rule of the real windows of e13, its prevalence, its flows, whether its
    benign flows are subsampled, the shares of its families and the cdfs of its classes at the grid.

    a window keeps all its benign flows and has its attacks thinned to the prevalence, one short of attacks keeps all
    of them and has its benign flows subsampled to the count that gives the prevalence, and one without attacks goes
    through at prevalence 0. the composition without a prevalence keeps the window as it is.
    """
    sizes = np.asarray(counts["sizes"])
    n_b, attacks = int(sizes[0]), sizes[1:]
    n_a = int(attacks.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        b_w, m_w, _, own = rates(counts)
    own = np.where(np.isfinite(own), own, 0.0)
    plan = {"has_attacks": n_a > 0, "shares": own, "benign": b_w, "miss": m_w, "subsampled": False}
    pi = co.get("prevalence")
    if pi is None:
        return {**plan, "prevalence": n_a / int(sizes.sum()), "sessions": int(sizes.sum())}
    if n_a == 0:
        return {**plan, "prevalence": 0.0, "sessions": n_b}
    full = int(round(n_b / (1 - pi)))
    need = own > 0
    if np.any(attacks[need] < full * pi * own[need]):
        return {**plan, "prevalence": pi, "sessions": int(np.floor((attacks[need] / (pi * own[need])).min())),
                "subsampled": True}
    return {**plan, "prevalence": pi, "sessions": full}


def load_replay(source: dict, windows: pd.DataFrame, log) -> list:
    """the rows of every window in their real order from the per flow scores the source run saved, split by class and
    binned at the grid, checked against the window histograms of the run. the scores sit in te order with the
    deployment window of every flow, so a window is one stretch of the saved arrays."""
    path = Path(source["dir"]) / "results" / "scores.npz"
    if not path.is_file():
        raise FileNotFoundError(f"the replay needs {path}, the per flow scores of the source run, which the export "
                                "leaves out of the mirror, so the run must come from the runs folder of Drive")
    hist = source["hist"]
    with np.load(path) as z:
        window = z["window"]
        score = z["score"]
        code = z["family"]
    if (np.diff(window) < 0).any():
        raise ValueError("the saved scores are not in window order")
    edges = np.searchsorted(window, np.arange(len(windows) + 1))
    blocks = []
    for i in range(len(windows)):
        lo, hi = edges[i], edges[i + 1]
        block = block_rows(score[lo:hi], code[lo:hi], hist["grid"])
        if not check_block(block, hist["grid"], hist["window_benign_below"][i], hist["window_family_below"][i],
                           hist["window_sizes"][i], code[lo:hi]):
            raise ValueError(f"the saved scores of window {i} do not give the window histogram of the source run")
        blocks.append(block)
    log(f"replay, per flow scores of the source run read, {int(edges[-1] - edges[0])} flows in {len(blocks)} windows, "
        "the window histograms of the source run reproduced from the saved scores in every window")
    return blocks


def simulated_composition(co: dict, source: dict, policies, windows, thresholds, certify_window, fixed_index: int,
                          pi_cal: float, eps: float, q: float, stream_seeds, log) -> dict:
    """the thresholds, the dismissed flows, the population FOR and the audited stream of every policy on the
    simulated stream of one composition, the windows drawn again with the seeds of the source run."""
    ssum, hist, sdraws = source["summary"], source["hist"], source["draws"]
    draws, w_count = int(ssum["draws"]), len(windows)
    seeds = source_seeds(source)
    pi = co.get("prevalence")
    natural = pi is None
    if not natural:
        dep = source["cfg"]["deployment"]
        j, k, real = dep["prevalences"].index(pi), dep["mixes"].index("window"), ssum["arms"].index("real")
    index = {p: np.full((w_count, draws), -1, dtype=np.int64) for p in policies}
    dismissed = {p: np.zeros((w_count, draws), dtype=np.int64) for p in policies}
    rate = {p: np.zeros((w_count, draws)) for p in policies}
    n_window = np.zeros(w_count, dtype=np.int64)
    prevalence = np.zeros(w_count)
    subsampled = np.zeros(w_count, dtype=bool)
    fresh = []
    recal = np.full(draws, fixed_index, dtype=np.int64)
    for i in range(w_count):
        plan = window_plan(block_counts(hist, i), co)
        n_w, pi_w = plan["sessions"], plan["prevalence"]
        n_window[i], prevalence[i], subsampled[i] = n_w, pi_w, plan["subsampled"]
        rng = np.random.default_rng(seeds[natural_stream(source, i) if natural else window_stream(source, i, pi)])
        drawn = draw_counts(rng, n_w, pi_w, plan["shares"], plan["benign"], plan["miss"], draws)
        attack_below = drawn.family_below.sum(1)
        window_below = drawn.benign_below + attack_below
        curve = for_from_rates(pi_w, plan["shares"] @ plan["miss"], plan["benign"])
        for p in policies:
            if p == "forge_online":
                t = np.asarray(certify_window(window_below, np.full(draws, n_w))["index"], dtype=np.int64)
                if plan["has_attacks"]:
                    saved = sdraws["natural_online_index"][i] if natural else sdraws["online_index"][real, i, j, k]
                    if not np.array_equal(t, saved.astype(np.int64)):
                        raise ValueError(f"the online thresholds of window {i} at {composition_word(co)} could not "
                                         "be reproduced")
                else:
                    fresh.append(i)
            elif p == "recalibration":
                t = recal
            else:
                t = thresholds[p]
            hit = t >= 0
            index[p][i] = t
            dismissed[p][i] = np.where(hit, np.take_along_axis(window_below, np.maximum(t, 0)[:, None], 1)[:, 0], 0)
            rate[p][i] = _threshold_rate(t, curve)
        if "recalibration" in policies:
            recal = recalibrate(window_below, attack_below, n_w, drawn.n_family.sum(1), recal, pi_cal, eps)
    log(f"composition {co['name']}, {composition_word(co)}, windows of {n_window.min()} to {n_window.max()} flows, "
        f"{int(subsampled.sum())} short of attacks with their benign flows subsampled, {len(fresh)} without attacks "
        "at prevalence 0, online thresholds of the source run reproduced in every window with attacks")
    out = {"n_window": n_window, "prevalence": prevalence, "subsampled": subsampled}
    for pj, p in enumerate(policies):
        rng = np.random.default_rng(stream_seeds[pj])
        out[p] = {"index": index[p], "dismissed": dismissed[p], "rate": rate[p],
                  "stream": audited_stream(rng, dismissed[p], rate[p], q), "burst": None}
    return out


def replayed_composition(co: dict, source: dict, blocks: list, policies, windows, thresholds, certifier_of,
                         fixed_index: int, pi_cal: float, eps: float, q: float, chunk: int, thin_seeds, coin_seeds,
                         log) -> dict:
    """the same on the replayed stream, the flows of every window in their real order thinned to the composition, a
    chunk of draws at a time. the natural composition and a window without attacks keep every row, a window short of
    attacks keeps all of them and thins its benign rows, the rest thin their attack rows. the online certificate and
    the recalibration refit read the replayed window's own counts at the grid, and certifier_of gives the online
    certifier of a slice of the calibration draws, since the window of a draw is certified against its own
    calibration set."""
    ssum, hist = source["summary"], source["hist"]
    draws, w_count, g = int(ssum["draws"]), len(windows), len(hist["grid"])
    natural = co.get("prevalence") is None
    floor = rule.alarm_floor(eps)
    index = {p: np.full((w_count, draws), -1, dtype=np.int64) for p in policies}
    counted = {name: {p: np.zeros((w_count, draws), dtype=np.int64) for p in policies}
               for name in ("dismissed", "attacks", "audited_attacks", "audited_benign", "burst")}
    rate = {p: np.zeros((w_count, draws)) for p in policies}
    parts = {p: [] for p in policies}
    offset = {p: np.zeros(draws, dtype=np.int64) for p in policies}
    before = {p: np.zeros(draws, dtype=np.int64) for p in policies}
    n_window = np.zeros(w_count, dtype=np.int64)
    prevalence = np.zeros(w_count)
    subsampled = np.zeros(w_count, dtype=bool)
    kept_attack, kept_benign = np.ones(w_count), np.ones(w_count)
    recal = np.full(draws, fixed_index, dtype=np.int64)
    thin_rngs = [np.random.default_rng(seed) for seed in thin_seeds]
    coin_rngs = {p: np.random.default_rng(seed) for p, seed in zip(policies, coin_seeds)}
    for i in range(w_count):
        counts = block_counts(hist, i)
        plan = window_plan(counts, co)
        n_b, n_a = int(counts["sizes"][0]), int(counts["sizes"][1:].sum())
        n_w, pi_w = plan["sessions"], plan["prevalence"]
        n_window[i], prevalence[i], subsampled[i] = n_w, pi_w, plan["subsampled"]
        if not natural and plan["has_attacks"]:
            kept_attack[i], kept_benign[i] = thin_rates(n_b, n_a, n_w, pi_w, plan["subsampled"])
        curve = for_from_rates(pi_w, plan["shares"] @ plan["miss"], plan["benign"])
        block = blocks[i]
        for lo in range(0, draws, chunk):
            d = slice(lo, min(lo + chunk, draws))
            th = thin(thin_rngs[i], block, kept_attack[i], kept_benign[i], d.stop - d.start)
            attack_below, benign_below = window_counts(block, th, g)
            window_below = benign_below + attack_below
            n_w_d = th.n_benign + th.n_attacks
            online = None
            if "forge_online" in policies:
                online = np.asarray(certifier_of(d)(window_below, n_w_d)["index"], dtype=np.int64)
            for p in policies:
                if p == "forge_online":
                    t = online
                elif p == "recalibration":
                    t = recal[d]
                else:
                    t = thresholds[p][d]
                out = dismissed_stream(coin_rngs[p], block, th, t, q, floor)
                index[p][i, d] = t
                rate[p][i, d] = _threshold_rate(t, curve)
                for name in counted:
                    counted[name][p][i, d] = out[name]
                rows = out["ev_row"] + lo
                parts[p].append((rows, out["ev_pos"] + offset[p][rows], out["ev_count"] + before[p][rows], i))
                offset[p][d] += out["dismissed"]
                before[p][d] += out["attacks"]
            if "recalibration" in policies:
                recal[d] = recalibrate(window_below, attack_below, n_w_d, th.n_attacks, recal[d], pi_cal, eps)
    benign_word = (f", with their benign rows kept with probability {kept_benign[subsampled].min():.4f} to "
                   f"{kept_benign[subsampled].max():.4f}" if subsampled.any() else "")
    log(f"replay, composition {co['name']}, {composition_word(co)}, every attack row of a window kept with "
        f"probability {kept_attack.min():.4f} to {kept_attack.max():.4f}, {int(subsampled.sum())} windows short of "
        f"attacks keep all of them{benign_word}, windows of {n_window.min()} to {n_window.max()} flows")
    out = {"n_window": n_window, "prevalence": prevalence, "subsampled": subsampled}
    for p in policies:
        stream = assemble_stream(draws, parts[p], counted["dismissed"][p], counted["attacks"][p],
                                 counted["audited_attacks"][p], counted["audited_benign"][p])
        out[p] = {"index": index[p], "dismissed": counted["dismissed"][p], "rate": rate[p], "stream": stream,
                  "burst": counted["burst"][p]}
    return out


def run(cfg: dict, run: runs.Run) -> None:
    m, au, sim, sg = cfg["monitor"], cfg["audits"], cfg["simulation"], cfg["segments"]
    eps, delta, q = m["epsilon"], m["delta"], au["rate"]
    delta_lower = m["delta_lower"]
    if not 0 < delta_lower < delta:
        raise ValueError(f"delta_lower is {delta_lower}, expected a value between 0 and delta {delta}")
    if m["kind"] not in KINDS:
        raise ValueError(f"unknown kind {m['kind']}")
    if not 0 < q <= 1:
        raise ValueError(f"the audit rate is {q}, expected a probability above 0")
    policies, arms, compositions = list(cfg["policies"]), list(cfg["arms"]), list(cfg["compositions"])
    streams = list(cfg.get("streams", ["simulated"]))
    unknown = (set(policies) - set(POLICIES)) | (set(arms) - set(ARMS)) | (set(streams) - set(STREAMS))
    if unknown:
        raise ValueError(f"unknown policies, arms or streams {', '.join(sorted(unknown))}")
    if not streams or len(set(streams)) != len(streams):
        raise ValueError(f"streams are {streams}, expected some of {', '.join(STREAMS)}, each once")
    names = [co["name"] for co in compositions]
    if len(set(names)) != len(names):
        raise ValueError("every composition needs its own name")
    trace = sim["trace"]
    if trace["policy"] not in policies or trace["composition"] not in names:
        raise ValueError("the trace names a policy or composition that is not run")
    chunk = int(sim.get("chunk", 100))
    if chunk < 1:
        raise ValueError(f"chunk is {chunk}, expected 1 or more draws")

    source = load_source(cfg)
    scfg, ssum, hist, sdraws = (source[key] for key in ("cfg", "summary", "hist", "draws"))
    windows = source["windows"].sort_values("window").reset_index(drop=True)
    c, dep = scfg["certify"], scfg["deployment"]
    if c["epsilon"] != eps:
        raise ValueError(f"the monitor's epsilon {eps} differs from the certificate's {c['epsilon']}")
    if not dep["subsample_benign"]:
        raise ValueError("the source run leaves a real window short of attacks out and the streams need every "
                         "window, so the source must subsample the benign flows of such a window")
    missing = set(policies) - set(ssum["methods"]) - {"recalibration"}
    if missing:
        raise ValueError(f"the source run has no thresholds of {', '.join(sorted(missing))}")
    for co in compositions:
        if co.get("prevalence") is not None and co["prevalence"] not in dep["prevalences"]:
            raise ValueError(f"composition {co['name']} at prevalence {co['prevalence']} has no window in the source "
                             f"run, which has prevalences {dep['prevalences']}")
    draws, w_count = int(ssum["draws"]), len(windows)
    if w_count != int(ssum["windows"]) or windows["window"].tolist() != list(range(w_count)):
        raise ValueError("the windows table of the source run does not number its windows from 0 in order")
    pi_cal = scfg["calibration"]["prevalence"]
    segments, joined = segments_of(windows, sg["length"], int(sg["minimum_windows"]))
    n_seg = int(segments.max()) + 1
    windows["segment"] = segments
    without_attacks = [int(i) for i in ssum["windows_without_attacks"]]
    run.log(f"source {cfg['source']['experiment']} run {cfg['source']['run']}, {ssum['dataset']} "
            f"{ssum['processed_version']}, fingerprint {ssum['fingerprint']}, {w_count} windows of "
            f"{ssum['window_length']} from {pd.Timestamp(windows['start'].iloc[0]):%Y-%m-%d %H:%M} to "
            f"{pd.Timestamp(windows['end'].iloc[-1]):%Y-%m-%d %H:%M}, {len(without_attacks)} without attacks, "
            f"{draws} draws, {len(hist['grid'])} thresholds, families "
            + ", ".join("+".join(f) for f in ssum["families"]) + ", streams " + ", ".join(streams)
            + f", {n_seg} segments of {sg['length']} for the sequence per week"
            + (f", the last {joined} windows joined to the segment before" if joined else ""))

    # the calibration draws of e13 again, at the family shares of its calibration mix, which fix the online
    # certifier and are checked against the saved offline thresholds
    b_cal, m_cal, _, _ = rates({"benign_below": hist["calibration_benign_below"],
                                "family_below": hist["calibration_family_below"],
                                "sizes": hist["calibration_sizes"]})
    seeds = source_seeds(source)
    cal = draw_counts(np.random.default_rng(seeds[0]), int(ssum["calibration_sessions"]), pi_cal, hist["shares"],
                      b_cal, m_cal, draws)
    grouped_below, grouped_n = group_rows(cal.family_below, cal.n_family, hist["groups"])
    args = (eps, c["delta"], c["pi_max"], c["delta_benign"])
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    offline = certify(cal.benign_below, cal.n_benign, grouped_below, grouped_n, *args, **start)
    if not np.array_equal(offline["index"], sdraws["forge_simplex_index"]):
        raise ValueError("the calibration draws of the source run could not be reproduced")
    certify_window = online_certifier(grouped_below, grouped_n, *args, **start)
    certifiers = {}

    def certifier_of(d: slice):
        # the online certifier of a chunk of calibration draws, built once per chunk
        if (d.start, d.stop) not in certifiers:
            certifiers[(d.start, d.stop)] = online_certifier(grouped_below[d], grouped_n[d], *args, **start)
        return certifiers[(d.start, d.stop)]

    thresholds = {p: sdraws[f"{p}_index"].astype(np.int64) for p in OFFLINE if p in policies}
    for p, index in thresholds.items():
        run.log(f"{p}, refused in {np.mean(index < 0):.3f} of the calibration sets of the source run")

    seq = Sequence(m["kind"], m["lower_ratios"], m["upper_ratios"], q, q, delta_lower, delta - delta_lower)
    tables = Tables(seq, q, q)
    # a sequence per week keeps the level of one sequence, a sequence per window and the any start rule split it
    # over the windows
    per_window = Sequence(m["kind"], m["lower_ratios"], m["upper_ratios"], q, q, delta_lower / w_count,
                          (delta - delta_lower) / w_count)
    window_tables = Tables(per_window, q, q)
    restarts = {name: value for name, value in
                {"monitor_week": (tables, segments), "monitor_window": (window_tables, np.arange(w_count))}.items()
                if name in arms}
    # the coins of the simulated stream take the first children of the seed and the replay spawns its own after
    # them, one per composition and window for the thinning and one per composition and policy for the coins
    seed_sequence = np.random.SeedSequence(cfg["seed"])
    stream_seeds = seed_sequence.spawn(len(compositions) * len(policies))
    thin_seeds = seed_sequence.spawn(len(compositions) * w_count) if "replay" in streams else []
    coin_seeds = seed_sequence.spawn(len(compositions) * len(policies)) if "replay" in streams else []
    blocks = load_replay(source, windows, run.log) if "replay" in streams else None
    fixed_index = int(ssum["fixed_index"])
    rows, window_rows, saved, traces, subsampled_by = [], [], {}, {}, {}
    for ci, co in enumerate(compositions):
        results = {}
        if "simulated" in streams:
            results["simulated"] = simulated_composition(
                co, source, policies, windows, thresholds, certify_window, fixed_index, pi_cal, eps, q,
                stream_seeds[ci * len(policies):(ci + 1) * len(policies)], run.log)
        if "replay" in streams:
            results["replay"] = replayed_composition(
                co, source, blocks, policies, windows, thresholds, certifier_of, fixed_index, pi_cal, eps, q,
                chunk, thin_seeds[ci * w_count:(ci + 1) * w_count],
                coin_seeds[ci * len(policies):(ci + 1) * len(policies)], run.log)
        subsampled_by[co["name"]] = [int(i) for i in np.flatnonzero(results[streams[0]]["subsampled"])]
        for stream in streams:
            res = results[stream]
            n_window, prevalence, subsampled = res["n_window"], res["prevalence"], res["subsampled"]
            total_windows = n_window.sum()
            for p in policies:
                r_p = res[p]
                index, dismissed, rate, burst = r_p["index"], r_p["dismissed"], r_p["rate"], r_p["burst"]
                out = judge_stream(tables, r_p["stream"], rate, eps, delta, q, restarts)
                if "any_start" in arms:
                    out["any_start"] = any_start_monitor(window_tables, r_p["stream"], eps)
                n_end = r_p["stream"]["offsets"][-1]
                held = dismissed > 0
                for w in range(w_count):
                    win = windows.iloc[w]
                    window_rows.append({
                        "stream": stream, "policy": p, "composition": co["name"], "window": w,
                        "start": win["start"], "segment": int(segments[w]), "prevalence": float(prevalence[w]),
                        "subsampled": bool(subsampled[w]), "window_sessions": int(n_window[w]),
                        "refused_rate": float(np.mean(~held[w])),
                        "threshold_mean": float(index[w][held[w]].mean()) if held[w].any() else np.nan,
                        "dismissed_share_mean": float((dismissed[w] / n_window[w]).mean()),
                        "for_true_mean": float(rate[w][held[w]].mean()) if held[w].any() else np.nan,
                        "for_true_median": float(np.median(rate[w][held[w]])) if held[w].any() else np.nan,
                        "violated_rate": float(np.mean(held[w] & (rate[w] > eps))),
                        "attacks_mean": float(r_p["stream"]["attacks"][w].mean()),
                        "realized_for_mean": float(np.mean(r_p["stream"]["attacks"][w][held[w]]
                                                           / dismissed[w][held[w]])) if held[w].any() else np.nan,
                        "burst_max_median": float(np.median(burst[w])) if burst is not None else np.nan,
                        "burst_max_high": float(np.quantile(burst[w], HIGH)) if burst is not None else np.nan,
                        **{f"state_{name}": float(np.mean(out["monitor"]["state"][w] == code))
                           for name, code in STATE_CODES.items()},
                        "window_audit_alarm_rate": float(out["window_audit"]["block_alarm"][w].mean()),
                        "window_audit_certified_rate": float(out["window_audit"]["block_certified"][w].mean()),
                    })
                onset = {"onset_true_rate": float(np.mean(out["onset_true"] >= 0)),
                         **_with_prefix("onset_true_block", _quantiles(out["onset_true"] + 1, out["onset_true"] >= 0)),
                         "onset_realized_rate": float(np.mean(out["onset_realized"] >= 0)),
                         **_with_prefix("onset_realized_block",
                                        _quantiles(out["onset_realized"] + 1, out["onset_realized"] >= 0))}
                for arm in arms:
                    r = out[arm]
                    alarmed = r["alarmed"]
                    row = {
                        "stream": stream, "policy": p, "composition": co["name"],
                        "prevalence": np.nan if co.get("prevalence") is None else float(co["prevalence"]),
                        "arm": arm, "draws": draws, "refused_rate": float(np.mean(n_end == 0)),
                        "stream_sessions_mean": float(n_end.mean()),
                        "alarm_rate": float(alarmed.mean()), "false_alarm_rate": float(r["false_alarm"].mean()),
                        **_with_prefix("alarm_block", _quantiles(r["alarm_block"] + 1, alarmed)),
                        **_with_prefix("alarm_session", _quantiles(r["alarm_pos"], alarmed)),
                        **onset,
                        **_with_prefix("leak", _quantiles(r["leak"], np.ones(draws, dtype=bool))),
                        "leak_mean": float(r["leak"].mean()),
                        **_with_prefix("excess", _quantiles(r["leak"] - eps * r["sessions"],
                                                            np.ones(draws, dtype=bool))),
                        **_with_prefix("sessions", _quantiles(r["sessions"], np.ones(draws, dtype=bool))),
                        "dismissed_share_mean": float((r["sessions"] / total_windows).mean()),
                        **_with_prefix("realized_for", _quantiles(np.divide(r["leak"], r["sessions"],
                                                                            out=np.zeros(draws),
                                                                            where=r["sessions"] > 0),
                                                                  r["sessions"] > 0)),
                        "audits_mean": float(out["audits"].mean()),
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
                            **_with_prefix("bound", _quantiles(r["bound"], alarmed)),
                            "over_bound_rate": float(r["over_bound"].sum() / max(alarmed.sum(), 1)),
                            **{f"blocks_{name}_mean": float((r["state"] == code).sum(0).mean())
                               for name, code in STATE_CODES.items()},
                        })
                    if arm == "window_audit":
                        row.update({"block_false_alarm_rate": float(r["block_false_alarm"].mean()),
                                    "blocks_certified_mean": float(r["block_certified"].sum(0).mean())})
                    rows.append(row)
                key = f"{stream}|{p}|{co['name']}|"
                saved[key + "attacks"] = r_p["stream"]["attacks"]
                saved[key + "dismissed"] = dismissed
                saved[key + "for_true"] = rate.astype(np.float32)
                if burst is not None:
                    saved[key + "burst"] = burst
                for arm in arms:
                    saved[key + f"{arm}_alarm_pos"] = out[arm]["alarm_pos"]
                    saved[key + f"{arm}_leak"] = out[arm]["leak"]
                    if arm in MONITORS:
                        saved[key + f"{arm}_state"] = out[arm]["state"]
                        saved[key + f"{arm}_bound"] = out[arm]["bound"]
                if p == trace["policy"] and co["name"] == trace["composition"]:
                    traces.update({f"{stream}|{k}": v for k, v in
                                   {**out["trace"], "window_sessions": n_window, "segments": segments}.items()})
                by_arm = {x["arm"]: x for x in rows if x["stream"] == stream and x["policy"] == p
                          and x["composition"] == co["name"]}
                line = (f"{stream}, {p}, composition {co['name']}, population FOR above epsilon in a first window for "
                        f"{onset['onset_true_rate']:.3f} of draws at window {onset['onset_true_block_median']:.0f}, "
                        f"realized count above epsilon times the flows at a window end for "
                        f"{onset['onset_realized_rate']:.3f} at window {onset['onset_realized_block_median']:.0f}")
                if "none" in by_arm:
                    line += f", {by_arm['none']['leak_median']:.0f} attacks dismissed without monitoring"
                for arm in arms:
                    if arm == "none":
                        continue
                    x = by_arm[arm]
                    when = (f"at window {x['alarm_block_median']:.0f} after {x['alarm_session_median']:.0f} flows"
                            if x["alarm_rate"] > 0 else "never")
                    line += (f", {ARM_SHORT[arm]} alarm in {x['alarm_rate']:.4f} of draws, false in "
                             f"{x['false_alarm_rate']:.4f}, {when}, leakage median {x['leak_median']:.0f}")
                    if arm in MONITORS:
                        line += (f", bounds left in {x['miscover_rate']:.4f}, above the bound in "
                                 f"{x['over_bound_rate']:.4f} of alarms, certified windows "
                                 f"{x['blocks_certified_mean']:.1f}")
                    if arm != "monitor" and arm in MONITORS:
                        line += f", {x['sequences']} sequences, one left in {x['miscover_sequence_rate']:.4f}"
                run.log(line)
        del results

    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    run.save_table(pd.DataFrame(window_rows), "by_window")
    run.save_table(windows, "windows")
    run.save_arrays("draws", **saved)
    run.save_arrays("trace", **traces)
    exceeds = {}
    for stream in streams:
        for arm in [a for a in MONITORS if a in arms]:
            g = claims[(claims["stream"] == stream) & (claims["arm"] == arm)]
            if arm == "monitor_week":
                # a sequence per week holds its level per sequence, so the check reads the share of failing sequences
                worst = g.loc[g["miscover_sequence_rate"].idxmax()]
                exceeds[f"{stream}|{arm}"] = bool((g["miscover_sequence_rate"] > delta).any())
                run.log(f"{stream}, {ARM_SHORT[arm]}, largest share of sequences leaving the bounds "
                        f"{worst['miscover_sequence_rate']:.4f} on {worst['policy']} at composition "
                        f"{worst['composition']}, over {int(worst['sequences'])} sequences per draw, delta {delta}, "
                        f"above delta {exceeds[f'{stream}|{arm}']}")
                continue
            worst = g.loc[g["miscover_rate"].idxmax()]
            exceeds[f"{stream}|{arm}"] = bool((g["miscover_low"] > delta).any())
            run.log(f"{stream}, {ARM_SHORT[arm]}, largest share of draws leaving the bounds "
                    f"{worst['miscover_rate']:.4f} on {worst['policy']} at composition {worst['composition']}, 95% "
                    f"interval {worst['miscover_low']:.4f} to {worst['miscover_high']:.4f}, delta {delta}, above "
                    f"delta beyond simulation noise {exceeds[f'{stream}|{arm}']}")
    first_of = [int(np.flatnonzero(segments == s)[0]) for s in range(n_seg)]
    run.save_json({
        "source": {**cfg["source"], "fingerprint": ssum["fingerprint"], "dataset": ssum["dataset"],
                   "processed_version": ssum["processed_version"], "draws": draws, "windows": w_count,
                   "window_length": ssum["window_length"], "deployment": ssum["roles"]["deployment"],
                   "families": ssum["families"], "grid_thresholds": int(len(hist["grid"])),
                   "windows_without_attacks": without_attacks},
        "epsilon": eps, "delta": delta, "delta_lower": delta_lower, "kind": m["kind"],
        "lower_ratios": m["lower_ratios"], "upper_ratios": m["upper_ratios"], "audit_rate": q,
        "alarm_floor": rule.alarm_floor(eps),
        "calibration_prevalence": pi_cal, "policies": policies, "arms": arms, "compositions": compositions,
        "streams": streams, "main_stream": MAIN_STREAM if MAIN_STREAM in streams else streams[0],
        "chunk": chunk,
        "segments": {"length": sg["length"], "minimum_windows": int(sg["minimum_windows"]), "count": n_seg,
                     "joined": joined, "first_window": first_of,
                     "windows": [int((segments == s).sum()) for s in range(n_seg)]},
        "restart_sequences": {"monitor_week": n_seg, "monitor_window": w_count, "any_start": w_count},
        "windows_subsampled": subsampled_by, "windows_at_prevalence_zero": without_attacks,
        "trace": trace, "miscoverage_exceeds_delta": exceeds,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _policy_line(policy: str) -> dict:
    return {**_e21_policy_line(policy), "label": POLICY_LABELS[policy]}


def _arm_line(arm: str) -> dict:
    return {"color": ARM_COLORS[arm], "marker": ARM_MARKERS[arm], "label": ARM_LABELS[arm]}


def _dates(values) -> np.ndarray:
    return mdates.date2num(pd.to_datetime(values).to_numpy())


def _time_axis(ax, windows: pd.DataFrame) -> None:
    # time on the x axis, a labelled tick on every monday and a bare one on every other day, the window starts padded
    # by one window on each side
    starts, ends = _dates(windows["start"]), _dates(windows["end"])
    pad = ends[0] - starts[0]
    ax.set_xlim(starts.min() - pad, ends.max() + pad)
    weekly = ends.max() - starts.min() >= 7
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO) if weekly else mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def _segment_lines(ax, windows: pd.DataFrame) -> None:
    # the start of every segment after the first, the weeks of the restarted monitor
    starts = _dates(windows["start"])
    segments = windows["segment"].to_numpy()
    for w in np.flatnonzero(np.r_[False, segments[1:] != segments[:-1]]):
        ax.axvline(starts[w], color=style.GRAY, linestyle=":", linewidth=1)


def fig_bounds_over_time(run_dir) -> plt.Figure:
    style.use()
    traces = _trace_streams(runs.load_arrays(run_dir, "trace"))
    summary = runs.load_json(run_dir, "summary")
    eps = summary["epsilon"]
    scale = 1e6
    color = style.color_for("forge_simplex")
    fig, axes = plt.subplots(1, len(traces), figsize=(style.WIDTH_SINGLE if len(traces) == 1 else style.WIDTH_DOUBLE,
                                                      2.6), squeeze=False)
    # the count is known at the audited attacks and at the window ends, the bounds move at the audited attacks only,
    # one panel per stream, the first window of every segment marked
    for ax, (stream, t), letter in zip(axes[0], traces, string.ascii_lowercase):
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
        segments = t["segments"]
        for w in np.flatnonzero(np.r_[False, segments[1:] != segments[:-1]]):
            ax.axvline(t["ends"][w - 1] / scale, color=style.GRAY, linestyle=":", linewidth=0.6)
        if t["alarm_pos"] >= 0:
            ax.axvline(t["alarm_pos"] / scale, color=color, linestyle="-.", linewidth=0.8)
        ax.set_xlim(0, n_end)
        ax.set_ylim(0, 1.1 * max(float(t["ucb"].max()), float(y.max()), 1.0))
        ax.set_xlabel("millions of dismissed flows" + _stream_word(stream, traces).replace("\n", ", "))
        if len(traces) > 1:
            layout.panel_letter(ax, letter)
    axes[0, 0].set_ylabel("attacks")
    layout.legend_below(fig, axes[0, 0], ["attacks among the dismissed flows", "lower bound", "upper bound"],
                        ncol=2)
    return fig


def fig_for_by_window(run_dir) -> plt.Figure:
    style.use()
    table = runs.load_table(run_dir, "by_window")
    windows = runs.load_table(run_dir, "windows").sort_values("window")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    streams = summary.get("streams", [None])
    fig, axes = plt.subplots(len(streams), len(compositions), figsize=(style.WIDTH_DOUBLE, 2.6 * len(streams)),
                             sharex=True, sharey=True, squeeze=False)
    # the population FOR of each policy's threshold in every window on the clock, one row per stream, a dismissed
    # share of zero has no rate
    floor = summary["epsilon"] / 100
    starts = _dates(windows["start"])
    letters = iter(string.ascii_lowercase)
    for row, stream in zip(axes, streams):
        data = of_stream(table, stream) if stream is not None else table
        for ax, co in zip(row, compositions):
            g = data[data["composition"] == co["name"]]
            for p in policies:
                h = g[g["policy"] == p].sort_values("window")
                ax.plot(starts[h["window"].to_numpy()], h["for_true_mean"].clip(lower=floor), markersize=2.5,
                        linewidth=1, **_policy_line(p))
            ax.axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
            _segment_lines(ax, windows)
            ax.set_yscale("log")
            layout.panel_letter(ax, next(letters))
        row[0].set_ylabel("FOR of the window at the threshold" + _stream_word(stream, streams))
    year = pd.Timestamp(windows["start"].iloc[0]).year
    for ax, co in zip(axes[-1], compositions):
        _time_axis(ax, windows)
        ax.set_xlabel(f"start of the window of {summary['source']['window_length']}, {year},\n{composition_word(co)}")
    layout.legend_below(fig, axes[0, 0], [POLICY_LABELS[p] for p in policies])
    return fig


def fig_alarm_by_policy(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    arms = [a for a in [*MONITORS, "window_audit", "naive"] if a in summary["arms"]]
    compositions = summary["compositions"]
    streams = summary.get("streams", [None])
    fig, axes = plt.subplots(2 * len(streams), len(compositions), figsize=(style.WIDTH_DOUBLE, 4.4 * len(streams)),
                             sharex=True, squeeze=False)
    x = np.arange(len(policies))
    # per stream two rows. top, the share of draws with an alarm, filled, and with a false alarm, hollow. bottom,
    # the window of the alarm with its 5% to 95% range, and the first window whose population FOR exceeds epsilon
    letters = iter(string.ascii_lowercase)
    for si, stream in enumerate(streams):
        data = of_stream(claims, stream) if stream is not None else claims
        top, bottom = axes[2 * si], axes[2 * si + 1]
        for col, co in enumerate(compositions):
            g = data[data["composition"] == co["name"]]
            for k, arm in enumerate(arms):
                h = g[g["arm"] == arm].set_index("policy").reindex(policies)
                xs = layout.dodge(x, k, len(arms), 0.12)
                line = _arm_line(arm)
                top[col].plot(xs, h["alarm_rate"], linestyle="none", markersize=4, **line)
                top[col].plot(xs, h["false_alarm_rate"], linestyle="none", markersize=4, markerfacecolor="white",
                              color=line["color"], marker=line["marker"])
                layout.points(bottom[col], xs, h["alarm_block_median"], h["alarm_block_low"], h["alarm_block_high"],
                              linestyle="none", **line)
            h = g[g["arm"] == arms[0]].set_index("policy").reindex(policies)
            bottom[col].plot(x, h["onset_true_block_median"], linestyle="none", marker="_", markersize=9,
                             color=style.INK, label="first window with FOR above epsilon")
            top[col].set_ylim(-0.02, 1.02)
            bottom[col].set_ylim(0, summary["source"]["windows"] + 1)
            bottom[col].set_xticks(x)
            bottom[col].set_xticklabels([POLICY_LABELS[p].split(",")[0] for p in policies], rotation=30, ha="right")
        top[0].set_ylabel("share of draws with an alarm" + _stream_word(stream, streams))
        bottom[0].set_ylabel("window of the alarm" + _stream_word(stream, streams))
        for ax in [*top, *bottom]:
            layout.panel_letter(ax, next(letters))
    for ax, co in zip(axes[-1], compositions):
        ax.set_xlabel(f"threshold policy,\n{composition_word(co)}")
    hollow = Line2D([], [], color=style.GRAY, marker="o", markerfacecolor="white", linestyle="none")
    layout.legend_below(fig, axes[0, 0], [ARM_LABELS[a] for a in arms], [(hollow, "false alarms")])
    return fig


def fig_leakage_by_policy(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    arms = [a for a in [*MONITORS, "window_audit", "naive", "none"] if a in summary["arms"]]
    compositions = summary["compositions"]
    streams = summary.get("streams", [None])
    fig, axes = plt.subplots(len(streams), len(compositions), figsize=(style.WIDTH_DOUBLE, 2.8 * len(streams)),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(policies))
    # the attacks dismissed until the switch off per policy and arm, one row per stream, and the bound of Theorem 3
    # at the alarm of FORGE-Monitor
    letters = iter(string.ascii_lowercase)
    for row, stream in zip(axes, streams):
        data = of_stream(claims, stream) if stream is not None else claims
        for ax, co in zip(row, compositions):
            g = data[data["composition"] == co["name"]]
            for k, arm in enumerate(arms):
                h = g[g["arm"] == arm].set_index("policy").reindex(policies)
                xs = layout.dodge(x, k, len(arms), 0.11)
                layout.points(ax, xs, h["leak_median"].clip(lower=1), h["leak_low"].clip(lower=1),
                              h["leak_high"].clip(lower=1), linestyle="none", **_arm_line(arm))
                if arm == "monitor":
                    ax.plot(xs, h["bound_median"].clip(lower=1), linestyle="none", marker="s", markersize=4,
                            markerfacecolor="white", color=ARM_COLORS[arm], label="bound of Theorem 3")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_LABELS[p].split(",")[0] for p in policies], rotation=30, ha="right")
            layout.panel_letter(ax, next(letters))
        row[0].set_ylabel("attacks dismissed until the switch off" + _stream_word(stream, streams))
    for ax, co in zip(axes[-1], compositions):
        ax.set_xlabel(f"threshold policy,\n{composition_word(co)}")
    layout.legend_below(fig, axes[0, 0], [ARM_LABELS[a] for a in arms])
    return fig


FIGURES = {
    "bounds_over_time": fig_bounds_over_time,
    "for_by_window": fig_for_by_window,
    "alarm_by_policy": fig_alarm_by_policy,
    "leakage_by_policy": fig_leakage_by_policy,
}
