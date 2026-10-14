"""e21, RQ2 on AnoShift, FORGE-Monitor on the dismissed streams of the thresholds of e11 through the 72 blocks of 2010
to 2015 in order, against the baselines that read the same audits.

the streams come from an e11 run. its grid, block histograms, calibration draws and thresholds are read from the run
folder, the calibration sets are drawn again from the same stream of seeds, which the run checks against the saved
offline thresholds, and the window of each block is drawn again from the block's score distributions with the seed
e11 used, which the run checks against the saved online thresholds. a stream is the dismissed sessions of one
threshold policy through the windows in order, each window thinned to a composition, the block's own mix of attack
types at a prevalence. a window keeps all benign sessions of its block, and the threshold of a draw decides how many
of its sessions are dismissed. every dismissed session is an attack with the FOR of the block's population at the
threshold, independently of the rest, and is audited with probability q, so the audited attacks of a window come at
geometric gaps and the attacks between two of them, which the monitor never sees, are binomial in the gap. the count
of attacks among the dismissed sessions of the drawn stream is the ground truth the bounds are judged on, and the FOR
of the block's population at the threshold is the ground truth of the drift.

threshold policies, the streams.
    forge_simplex    the offline certificate of Theorem 1 from the calibration draw, one threshold per draw
    forge_online     Theorem 1b, a threshold per window from the window's own unlabeled scores
    ltt_iid          the i.i.d. baseline, one threshold per draw
    conformal_fnr    the conformal threshold of Şahin and Mert, one per draw
    fixed_sel        the threshold of common practice, tuned on the selection year
    recalibration    the fixed threshold refit after every block on the block's escalated labels, the dismissed
                     sessions counted as benign, the common practice with one sided labels, which never tightens

monitors, the arms on every stream, all reading the same audits.
    monitor          FORGE-Monitor, the audit martingale with uniform audits, the alarm read from 1 / epsilon sessions
                     on, the dismissal switched off at the alarm, one sequence for the whole deployment
    monitor_year     the same with a new sequence at the start of every year, each at delta, the sequences counted,
                     so the bounds speak of the count since the year began
    monitor_block    the same with a new sequence at the start of every block, each at delta over the number of
                     blocks, the counterpart of the per window audit
    none             the same stream without an alarm, so its leakage is the whole stream's
    window_audit     the uniform per window audit in the style of Manoharan, a Clopper-Pearson bound on the FOR of
                     each block from its audited sessions at delta over the number of blocks, read at the block's end
    naive            the running Horvitz Thompson estimate against epsilon times the dismissed count, no bound

counted per policy, composition and arm. the share of draws with an alarm and with a false alarm, one whose count at
the alarm is at or below epsilon times the dismissed count, both since the start of the sequence, the block and the
session of the alarm, the first block
whose population FOR at the policy's threshold exceeds epsilon and the first block end at which the realized count
does, the attacks dismissed until the switch off against the whole stream's, the excess over epsilon times the
sessions dismissed, the bound of Theorem 3 at the alarm, the share of draws in which the count leaves the bounds, the
state at every block end, and the audits. per block the population FOR, the share of the window dismissed and the
realized FOR of each policy.

figures. bounds_over_time, one draw of one policy, the count, both bounds and epsilon times the sessions through the
stream. for_by_block, the population FOR of each policy's threshold in every block. alarm_by_policy, the share of
draws with an alarm, the false alarms and the block of the alarm per policy and arm. leakage_by_policy, the attacks
dismissed until the switch off per policy and arm against the bound of Theorem 3.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from forge import paths, runs
from forge.certify.bounds import clopper_pearson_lower, clopper_pearson_upper
from forge.certify.certificate import certify, fixed_sequence, for_from_rates, group_rows, online_certifier
from forge.config import load_config
from forge.data.synthetic import draw_counts
from forge.eval.metrics import share_with_interval
from forge.eval.windows import rates
from forge.monitor import rule
from forge.monitor.sequence import KINDS, Sequence, Tables, leakage_bound
from forge.viz import layout, style
from forge.viz.export import save_figure

POLICIES = ["forge_simplex", "forge_online", "ltt_iid", "conformal_fnr", "fixed_sel", "recalibration"]
OFFLINE = ["forge_simplex", "ltt_iid", "conformal_fnr", "fixed_sel"]
POLICY_LABELS = {
    "forge_simplex": "FORGE offline, Theorem 1",
    "forge_online": "FORGE online, Theorem 1b",
    "ltt_iid": "LTT, i.i.d.",
    "conformal_fnr": "conformal FNR, Şahin and Mert",
    "fixed_sel": "fixed threshold, selection year",
    "recalibration": "recalibration on escalated labels",
}
POLICY_MARKERS = {"forge_simplex": "o", "forge_online": "v", "ltt_iid": "^", "conformal_fnr": "p", "fixed_sel": "X",
                  "recalibration": "D"}
ARMS = ["monitor", "monitor_year", "monitor_block", "none", "window_audit", "naive"]
MONITORS = ["monitor", "monitor_year", "monitor_block"]
ARM_LABELS = {"monitor": "FORGE-Monitor", "monitor_year": "FORGE-Monitor, a sequence per year",
              "monitor_block": "FORGE-Monitor, a sequence per block", "none": "no monitoring",
              "window_audit": "per window audit, Clopper-Pearson", "naive": "running estimate, no bound"}
ARM_MARKERS = {"monitor": "o", "monitor_year": "D", "monitor_block": "v", "window_audit": "s", "naive": "^",
               "none": "x"}
ARM_SHORT = {"monitor": "FORGE-Monitor", "monitor_year": "FORGE-Monitor per year",
             "monitor_block": "FORGE-Monitor per block", "window_audit": "per window audit",
             "naive": "running estimate"}
STATE_CODES = {name: code for code, name in enumerate(rule.STATES)}
LOW, HIGH = 0.05, 0.95


def load_source(cfg: dict) -> dict:
    # the e11 run the streams come from, in the runs folder of the storage root
    src = cfg["source"]
    run_dir = paths.runs_dir() / src["experiment"] / src["run"]
    if not (run_dir / "results").is_dir():
        raise FileNotFoundError(f"no results in {run_dir}, the source run must be in the runs folder")
    return {"dir": run_dir, "cfg": load_config(run_dir / "config.yaml"), "summary": runs.load_json(run_dir, "summary"),
            "hist": runs.load_arrays(run_dir, "histograms"), "draws": runs.load_arrays(run_dir, "draws"),
            "windows": runs.load_table(run_dir, "windows")}


def window_stream(source: dict, i: int, pi: float, mix: str = "window") -> int:
    # the index of the seed stream e11 used for the real window of block i at this prevalence and mix
    dep = source["cfg"]["deployment"]
    prevalences, mixes = dep["prevalences"], dep["mixes"]
    if pi not in prevalences or mix not in mixes:
        raise ValueError(f"prevalence {pi} with mix {mix} has no window in the source run, which has prevalences "
                         f"{prevalences} and mixes {mixes}")
    return 1 + (i * len(prevalences) + prevalences.index(pi)) * len(mixes) + mixes.index(mix)


def block_counts(hist: dict, i: int) -> dict:
    return {"benign_below": hist["window_benign_below"][i], "family_below": hist["window_family_below"][i],
            "sizes": hist["window_sizes"][i]}


def recalibrate(window_below, attack_below, n_window, n_attacks, index, pi: float, eps: float) -> np.ndarray:
    """the fixed threshold refit on the escalated labels of one window, the common practice with one sided labels.

    the sessions at or below the window's threshold were dismissed and got no label, so they count as benign, and the
    sessions above it got their labels. the plug in rule then takes the loosest threshold whose FOR at prevalence pi,
    estimated on those labels, stays at or below eps, searched from the strictest one. nothing at or below the
    threshold can fail, so the threshold never tightens. an index of -1 dismissed nothing, so every label was seen.
    """
    window_below, attack_below = np.asarray(window_below), np.asarray(attack_below)
    index = np.asarray(index)
    at = np.where(index >= 0, np.take_along_axis(attack_below, np.maximum(index, 0)[:, None], 1)[:, 0], 0)
    seen = np.maximum(attack_below - at[:, None], 0)
    escalated = np.asarray(n_attacks) - at
    labelled_benign = np.asarray(n_window) - escalated
    m = np.divide(seen, escalated[:, None], out=np.zeros(window_below.shape), where=escalated[:, None] > 0)
    b = np.divide(window_below - seen, labelled_benign[:, None], out=np.zeros(window_below.shape),
                  where=labelled_benign[:, None] > 0)
    return fixed_sequence(for_from_rates(pi, m, b) <= eps, 0)


def audited_stream(rng, dismissed, rate, q: float) -> dict:
    """the audited attacks of the streams of independent draws through windows in order.

    dismissed and rate have shape (W, draws), the sessions dismissed in each window and the FOR of its population at
    the threshold. a dismissed session is an attack with that probability and audited with probability q,
    independently, so the audited attacks come at geometric gaps with probability q times the FOR, the attacks between
    two of them are binomial in the gap, and the audited benign sessions are binomial in the rest. returns the events
    of each draw padded on the right after a virtual first event at position 0, with the session index of each event
    in the whole stream, the attack count at it and its window, the attacks, audited attacks and audited benign
    sessions of each window, and the session index at which each window ends.
    """
    dismissed = np.asarray(dismissed, dtype=np.int64)
    rate = np.asarray(rate, dtype=float)
    w_count, draws = dismissed.shape
    offsets = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(dismissed, axis=0)])
    attacks = np.zeros((w_count, draws), dtype=np.int64)
    audited_attacks = np.zeros((w_count, draws), dtype=np.int64)
    audited_benign = np.zeros((w_count, draws), dtype=np.int64)
    parts = []
    before = np.zeros(draws, dtype=np.int64)
    for w in range(w_count):
        flows, f = dismissed[w], rate[w]
        p = q * f
        live = (flows > 0) & (p > 0)
        # an unaudited session is an attack with the probability that remains once the audited attacks are known
        rest = np.where(live, f * (1 - q) / (1 - p), f)
        if live.any():
            expected = (flows * p).max()
            m = int(np.ceil(expected + 10 * np.sqrt(expected) + 20))
            p_live = np.where(live, p, 1.0)[:, None]
            pos = np.cumsum(rng.geometric(p_live, size=(draws, m)), axis=1)
            # a row whose gaps stop before the end of the window would lose audited attacks, so it gets more gaps
            while (live & (pos[:, -1] <= flows)).any():
                more = np.cumsum(rng.geometric(p_live, size=(draws, m)), axis=1)
                pos = np.concatenate([pos, pos[:, -1:] + more], axis=1)
            keep = live[:, None] & (pos <= flows[:, None])
            k = keep.sum(1)
            previous = np.concatenate([np.zeros((draws, 1), dtype=np.int64), pos[:, :-1]], axis=1)
            between = rng.binomial(np.where(keep, pos - previous - 1, 0), rest[:, None])
            last = np.where(k > 0, np.take_along_axis(pos, np.maximum(k - 1, 0)[:, None], 1)[:, 0], 0)
        else:
            pos = np.zeros((draws, 0), dtype=np.int64)
            keep = np.zeros((draws, 0), dtype=bool)
            between = np.zeros((draws, 0), dtype=np.int64)
            k = np.zeros(draws, dtype=np.int64)
            last = np.zeros(draws, dtype=np.int64)
        tail = rng.binomial(np.maximum(flows - last, 0), rest)
        count = before[:, None] + np.cumsum(between + keep, axis=1)
        attacks[w] = between.sum(1) + k + tail
        audited_attacks[w] = k
        audited_benign[w] = rng.binomial(flows - attacks[w], q)
        before = before + attacks[w]
        parts.append((np.where(keep, pos + offsets[w][:, None], -1), np.where(keep, count, 0), keep))
    pos = np.concatenate([np.zeros((draws, 1), dtype=np.int64)] + [p for p, _, _ in parts], axis=1)
    count = np.concatenate([np.zeros((draws, 1), dtype=np.int64)] + [c for _, c, _ in parts], axis=1)
    window = np.concatenate([np.full((draws, 1), -1)] + [np.full(p.shape, w) for w, (p, _, _) in enumerate(parts)],
                            axis=1)
    valid = np.concatenate([np.ones((draws, 1), dtype=bool)] + [v for _, _, v in parts], axis=1)
    # the events of a row in the order of the stream, the padding after them
    order = np.argsort(np.where(valid, pos, np.iinfo(np.int64).max), axis=1, kind="stable")
    width = int(valid.sum(1).max())

    def take(a):
        return np.take_along_axis(a, order, axis=1)[:, :width]

    return {"ev_pos": take(pos), "ev_count": take(count), "ev_window": take(window), "valid": take(valid),
            "attacks": attacks, "audited_attacks": audited_attacks, "audited_benign": audited_benign,
            "offsets": offsets}


def _first(mask):
    # the first true along the first axis, -1 when there is none
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask.any(0), mask.argmax(0), -1)


def restarted_monitor(tables: Tables, stream: dict, eps: float, segments) -> dict:
    """FORGE-Monitor with a new sequence at the start of every segment of windows, per draw.

    segments gives the segment of each window, nondecreasing from 0. a sequence reads the audited attacks, the
    sessions and the count since its segment began, so its bounds and its alarm speak of that segment, the alarm
    switches the dismissal off for good, and the attacks dismissed until then are counted from the start of the
    deployment. the bound of Theorem 3 on them adds the bound of every finished segment, read at its end, to the
    bound of the segment of the alarm. the bounds hold with the level of one sequence per segment and with the level
    times the segments over the deployment, and both failure shares are returned.
    """
    ev_pos, ev_count, ev_win, valid = stream["ev_pos"], stream["ev_count"], stream["ev_window"], stream["valid"]
    offsets, attacks = stream["offsets"], stream["attacks"]
    draws = ev_pos.shape[0]
    w_count = attacks.shape[0]
    rows = np.arange(draws)
    segments = np.asarray(segments)
    if segments.shape != (w_count,) or segments[0] != 0 or (np.diff(segments) < 0).any() or \
            (np.diff(segments) > 1).any():
        raise ValueError("segments must number the windows from 0 on without gaps, in order")
    n_seg = int(segments[-1]) + 1
    first = np.array([np.flatnonzero(segments == s)[0] for s in range(n_seg)])
    ends = offsets[1:]
    cum_attacks = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(attacks, axis=0)])
    cum_audited = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(stream["audited_attacks"], axis=0)])
    seg_start, seg_end = offsets[first], offsets[np.r_[first[1:], w_count]]
    before = cum_attacks[first]
    seg_total = cum_attacks[np.r_[first[1:], w_count]] - before
    # the real events and a virtual event at the start of every segment, in the order of the stream. the real events
    # come first in the concatenation, so an audited attack on the last session of a segment sorts before the virtual
    # start of the next one and stays in its own segment
    real = valid.copy()
    real[:, 0] = False
    pos = np.concatenate([np.where(real, ev_pos, np.iinfo(np.int64).max), seg_start.T], axis=1)
    count = np.concatenate([ev_count, before.T], axis=1)
    seg = np.concatenate([segments[np.maximum(ev_win, 0)], np.broadcast_to(np.arange(n_seg), (draws, n_seg))],
                         axis=1)
    win = np.concatenate([ev_win, np.broadcast_to(first, (draws, n_seg))], axis=1)
    virtual = np.concatenate([np.zeros(ev_pos.shape, dtype=bool), np.ones((draws, n_seg), dtype=bool)], axis=1)
    live = np.concatenate([real, np.ones((draws, n_seg), dtype=bool)], axis=1)
    order = np.argsort(pos, axis=1, kind="stable")
    width = n_seg + int(real.sum(1).max())
    pos, count, seg, win, virtual, live = (np.take_along_axis(a, order, axis=1)[:, :width]
                                           for a in (pos, count, seg, win, virtual, live))
    pos = np.where(live, pos, 0)
    rank = np.cumsum(live & ~virtual, axis=1)
    rank_at_start = rank[virtual].reshape(draws, n_seg)
    k = np.where(live, rank - rank_at_start[rows[:, None], seg], 0)
    n_in = np.where(live, pos - seg_start.T[rows[:, None], seg], 0)
    c_in = np.where(live, count - before.T[rows[:, None], seg], 0)
    length = (seg_end - seg_start).T
    lcb_t, ucb_t = tables.up_to(int(k.max()), int(length.max()), eps)
    lcb, ucb = lcb_t[k], ucb_t[k]
    floor = rule.alarm_floor(eps)
    hit = rule.first_alarm(lcb, n_in, live, eps, floor)
    alarmed = hit >= 0
    at = np.maximum(hit, 0)
    seg_at = seg[rows, at]
    p_in = np.where(alarmed, np.maximum(n_in[rows, at], floor), -1)
    p = np.where(alarmed, seg_start.T[rows, seg_at] + p_in, -1)
    g = np.where(alarmed, count[rows, at], -1)
    g_in = np.where(alarmed, c_in[rows, at], -1)
    total = cum_attacks[-1]
    n_end = offsets[-1]
    # the bound of the segment of the alarm, then those of the segments before it, each read at its end
    at_event = alarmed & (n_in[rows, at] > floor)
    bound = np.full(draws, -1, dtype=np.int64)
    if at_event.any():
        bound[at_event] = leakage_bound(p_in[at_event] - 1, lcb_t, ucb_t, eps) + 1
    bound[alarmed & ~at_event] = floor
    finished = np.where(length > 0, leakage_bound(np.maximum(length, 1), lcb_t, ucb_t, eps), 0)
    earlier = (np.arange(n_seg)[None, :] < seg_at[:, None]) & alarmed[:, None]
    bound = np.where(alarmed, bound + (finished * earlier).sum(1), -1)
    # the count before the next event of the same segment, or the segment's whole count at its end
    same = np.zeros(pos.shape, dtype=bool)
    same[:, :-1] = live[:, 1:] & (seg[:, 1:] == seg[:, :-1])
    nxt = np.where(same, np.concatenate([c_in[:, 1:], np.zeros((draws, 1), dtype=np.int64)], axis=1),
                   seg_total.T[rows[:, None], seg] + 1)
    bad_lower = live & ~virtual & (c_in < lcb)
    bad_upper = live & (nxt - 1 > ucb)
    failed = np.zeros((draws, n_seg), dtype=bool)
    np.logical_or.at(failed, (rows[:, None].repeat(width, 1)[bad_lower | bad_upper], seg[bad_lower | bad_upper]),
                     True)
    # the state at every block end, read in the block's own segment
    k_block = cum_audited[1:] - cum_audited[first[segments]]
    n_block = ends - seg_start[segments]
    certified = (ucb_t[k_block] <= eps * n_block) & (n_block > 0)
    violation = alarmed & (p[None, :] <= ends)
    state = np.where(violation, STATE_CODES["violation"],
                     np.where(certified, STATE_CODES["certified"], STATE_CODES["running"])).astype(np.int8)
    return {"alarmed": alarmed, "alarm_pos": p, "leak": np.where(alarmed, g, total),
            "sessions": np.where(alarmed, p, n_end), "alarm_block": np.where(alarmed, win[rows, at], -1),
            "false_alarm": alarmed & (g_in <= eps * p_in), "at_event": at_event, "bound": bound,
            "over_bound": alarmed & (np.where(alarmed, g, 0) > bound), "miss_lower": bad_lower.any(1),
            "miss_upper": bad_upper.any(1), "failed_sequences": failed, "sequences": n_seg, "state": state}


def judge_stream(tables: Tables, stream: dict, rate, eps: float, delta: float, q: float,
                 restarts: dict | None = None) -> dict:
    """the arms on the events of one stream, per draw. restarts maps the name of a restarted monitor to its tables
    and the segment of each window."""
    ev_pos, ev_count, ev_win, valid = stream["ev_pos"], stream["ev_count"], stream["ev_window"], stream["valid"]
    offsets, attacks = stream["offsets"], stream["attacks"]
    draws, width = ev_pos.shape
    w_count = attacks.shape[0]
    rows = np.arange(draws)
    n_end = offsets[-1]
    ends = offsets[1:]
    total = attacks.sum(0)
    k = np.broadcast_to(np.arange(width), ev_pos.shape)
    lcb_t, ucb_t = tables.up_to(width - 1, int(n_end.max()), eps)
    lcb, ucb = lcb_t[k], ucb_t[k]
    floor = rule.alarm_floor(eps)
    # the count rises between two events by the attacks the monitor never sees, so just before the next event it is
    # one less than the count there, and at the end of the stream it is the whole count
    nxt = np.empty_like(ev_count)
    nxt[:, :-1] = np.where(valid[:, 1:], ev_count[:, 1:], total[:, None] + 1)
    nxt[:, -1] = total + 1
    miss_lower = (valid & (ev_count < lcb)).any(1)
    miss_upper = (valid & (nxt - 1 > ucb)).any(1)
    cum_attacks = np.cumsum(attacks, axis=0)
    cum_audited = np.cumsum(stream["audited_attacks"], axis=0)

    def alarm_at(hit):
        # the alarm fires at the hit event or at the floor, the count and the audited attacks are read there
        alarmed = hit >= 0
        at = np.maximum(hit, 0)
        p = np.where(alarmed, np.maximum(ev_pos[rows, at], floor), -1)
        g = np.where(alarmed, ev_count[rows, at], -1)
        block = np.where(alarmed, (ends < p[None, :]).sum(0), -1)
        return {"alarmed": alarmed, "alarm_pos": p, "leak": np.where(alarmed, g, total),
                "sessions": np.where(alarmed, p, n_end), "alarm_block": block,
                "false_alarm": alarmed & (g <= eps * p), "at_event": alarmed & (ev_pos[rows, at] > floor)}

    out = {}
    monitor = alarm_at(rule.first_alarm(lcb, ev_pos, valid, eps, floor))
    bound = np.full(draws, -1, dtype=np.int64)
    if monitor["at_event"].any():
        bound[monitor["at_event"]] = leakage_bound(monitor["alarm_pos"][monitor["at_event"]] - 1, lcb_t, ucb_t,
                                                   eps) + 1
    bound[monitor["alarmed"] & ~monitor["at_event"]] = floor
    monitor["bound"] = bound
    monitor["over_bound"] = monitor["alarmed"] & (monitor["leak"] > bound)
    monitor["miss_lower"], monitor["miss_upper"] = miss_lower, miss_upper
    # the state at every block end, violation once the alarm has fired, certified while the upper bound stays at or
    # below epsilon times the sessions, running otherwise
    certified = (ucb_t[cum_audited] <= eps * ends) & (ends > 0)
    violation = monitor["alarmed"] & (monitor["alarm_pos"][None, :] <= ends)
    monitor["state"] = np.where(violation, STATE_CODES["violation"],
                                np.where(certified, STATE_CODES["certified"], STATE_CODES["running"])).astype(np.int8)
    out["monitor"] = monitor
    for name, (restart_tables, segments) in (restarts or {}).items():
        out[name] = restarted_monitor(restart_tables, stream, eps, segments)

    naive_hit = valid & (k / q > eps * np.maximum(ev_pos, floor))
    out["naive"] = alarm_at(np.where(naive_hit.any(1), naive_hit.argmax(1), -1))

    # the per window audit reads the labels of a block at its end and tests that block alone, with delta split over
    # the blocks
    audited = stream["audited_attacks"] + stream["audited_benign"]
    lower = np.asarray(clopper_pearson_lower(stream["audited_attacks"], audited, delta / w_count))
    upper = np.asarray(clopper_pearson_upper(stream["audited_attacks"], audited, delta / w_count))
    block_alarm = (lower > eps) & (audited > 0)
    first = _first(block_alarm)
    alarmed = first >= 0
    at = np.maximum(first, 0)
    p = np.where(alarmed, ends[at, rows], -1)
    g = np.where(alarmed, cum_attacks[at, rows], -1)
    out["window_audit"] = {"alarmed": alarmed, "alarm_pos": p, "leak": np.where(alarmed, g, total),
                           "sessions": np.where(alarmed, p, n_end), "alarm_block": first,
                           "false_alarm": alarmed & (g <= eps * p),
                           "block_false_alarm": alarmed & (np.asarray(rate)[at, rows] <= eps),
                           "block_alarm": block_alarm, "block_certified": (upper <= eps) & (audited > 0)}
    out["none"] = {"alarmed": np.zeros(draws, dtype=bool), "alarm_pos": np.full(draws, -1), "leak": total,
                   "sessions": n_end, "alarm_block": np.full(draws, -1), "false_alarm": np.zeros(draws, dtype=bool)}
    # the onsets, the first block whose population FOR at the threshold exceeds epsilon, and the first block end at
    # which the realized count does
    out["onset_true"] = _first((np.asarray(rate) > eps) & (stream["offsets"][1:] > stream["offsets"][:-1]))
    out["onset_realized"] = _first((cum_attacks > eps * ends) & (ends > 0))
    out["audits"] = audited.sum(0)
    out["trace"] = {"ev_pos": ev_pos[0][valid[0]], "ev_count": ev_count[0][valid[0]], "lcb": lcb[0][valid[0]],
                    "ucb": ucb[0][valid[0]], "ends": ends[:, 0], "cum_attacks": cum_attacks[:, 0],
                    "alarm_pos": int(monitor["alarm_pos"][0]), "n_end": int(n_end[0])}
    return out


def _quantiles(values, mask) -> dict:
    v = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    if len(v) == 0:
        return {"median": np.nan, "low": np.nan, "high": np.nan}
    return {"median": float(np.median(v)), "low": float(np.quantile(v, LOW)), "high": float(np.quantile(v, HIGH))}


def _with_prefix(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in d.items()}


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
    if trace["policy"] not in policies or trace["composition"] not in names:
        raise ValueError("the trace names a policy or composition that is not run")

    source = load_source(cfg)
    scfg, ssum, hist, sdraws, windows = (source[key] for key in ("cfg", "summary", "hist", "draws", "windows"))
    c = scfg["certify"]
    if c["epsilon"] != eps:
        raise ValueError(f"the monitor's epsilon {eps} differs from the certificate's {c['epsilon']}")
    missing = set(policies) - set(ssum["methods"]) - {"recalibration"}
    if missing:
        raise ValueError(f"the source run has no thresholds of {', '.join(sorted(missing))}")
    draws, w_count = int(ssum["draws"]), len(windows)
    pi_cal = scfg["calibration"]["prevalence"]
    run.log(f"source {cfg['source']['experiment']} run {cfg['source']['run']}, {ssum['dataset']} "
            f"{ssum['processed_version']}, fingerprint {ssum['fingerprint']}, {w_count} windows of "
            f"{ssum['years']['deployment'][0]} to {ssum['years']['deployment'][-1]}, {draws} draws, "
            f"{len(hist['grid'])} thresholds, families " + ", ".join("+".join(f) for f in ssum["families"]))

    # the calibration draws of e11 again, which fix the online certifier and are checked against the saved offline
    # thresholds
    b_cal, m_cal, _, shares_cal = rates({"benign_below": hist["calibration_benign_below"],
                                         "family_below": hist["calibration_family_below"],
                                         "sizes": hist["calibration_sizes"]})
    dep = scfg["deployment"]
    cells = w_count * len(dep["prevalences"]) * len(dep["mixes"])
    seeds = np.random.SeedSequence(scfg["seed"]).spawn(1 + 2 * cells)
    cal = draw_counts(np.random.default_rng(seeds[0]), int(ssum["calibration_sessions"]), pi_cal, shares_cal, b_cal,
                      m_cal, draws)
    grouped_below, grouped_n = group_rows(cal.family_below, cal.n_family, hist["groups"])
    args = (eps, c["delta"], c["pi_max"], c["delta_benign"])
    start = {"start_misses": c["start_misses"], "start_slack": c["start_slack"]}
    offline = certify(cal.benign_below, cal.n_benign, grouped_below, grouped_n, *args, **start)
    if not np.array_equal(offline["index"], sdraws["forge_simplex_index"]):
        raise ValueError("the calibration draws of the source run could not be reproduced")
    certify_window = online_certifier(grouped_below, grouped_n, *args, **start)
    thresholds = {p: sdraws[f"{p}_index"].astype(np.int64) for p in OFFLINE if p in policies}
    for p, index in thresholds.items():
        run.log(f"{p}, refused in {np.mean(index < 0):.3f} of the calibration sets of the source run")

    seq = Sequence(m["kind"], m["lower_ratios"], m["upper_ratios"], q, q, delta_lower, delta - delta_lower)
    tables = Tables(seq, q, q)
    # a sequence per year keeps the level of one sequence, a sequence per block splits it over the blocks
    years = np.unique(windows["year"].to_numpy(), return_inverse=True)[1]
    per_block = Sequence(m["kind"], m["lower_ratios"], m["upper_ratios"], q, q, delta_lower / w_count,
                         (delta - delta_lower) / w_count)
    restarts = {"monitor_year": (tables, years), "monitor_block": (Tables(per_block, q, q), np.arange(w_count))}
    stream_seeds = np.random.SeedSequence(cfg["seed"]).spawn(len(compositions) * len(policies))
    rows, block_rows, saved, kept_trace = [], [], {}, None
    k_online = dep["mixes"].index("window")
    for ci, co in enumerate(compositions):
        pi = co["prevalence"]
        j = dep["prevalences"].index(pi)
        index = {p: np.full((w_count, draws), -1, dtype=np.int64) for p in policies}
        dismissed = {p: np.zeros((w_count, draws), dtype=np.int64) for p in policies}
        rate = {p: np.zeros((w_count, draws)) for p in policies}
        n_window = np.zeros(w_count, dtype=np.int64)
        recal = np.full(draws, int(ssum["fixed_index"]), dtype=np.int64)
        for i, win in enumerate(windows.itertuples()):
            counts = block_counts(hist, i)
            b_w, m_w, _, shares_w = rates(counts)
            n_w = int(round(counts["sizes"][0] / (1 - pi)))
            n_window[i] = n_w
            rng = np.random.default_rng(seeds[window_stream(source, i, pi)])
            drawn = draw_counts(rng, n_w, pi, shares_w, b_w, m_w, draws)
            attack_below = drawn.family_below.sum(1)
            window_below = drawn.benign_below + attack_below
            curve = for_from_rates(pi, shares_w @ m_w, b_w)
            for p in policies:
                if p == "forge_online":
                    if [int(win.year), int(win.block), pi, "window"] in ssum["skipped"]:
                        t = np.full(draws, -1, dtype=np.int64)
                    else:
                        t = sdraws["online_index"][i, j, k_online].astype(np.int64)
                        online = certify_window(window_below, np.full(draws, n_w))
                        if not np.array_equal(online["index"], t):
                            raise ValueError(f"the online thresholds of window {i} at prevalence {pi} could not be "
                                             f"reproduced")
                elif p == "recalibration":
                    t = recal
                else:
                    t = thresholds[p]
                hit = t >= 0
                index[p][i] = t
                dismissed[p][i] = np.where(hit, np.take_along_axis(window_below, np.maximum(t, 0)[:, None], 1)[:, 0],
                                           0)
                rate[p][i] = np.where(hit, curve[np.maximum(t, 0)], 0.0)
            if "recalibration" in policies:
                recal = recalibrate(window_below, attack_below, n_w, drawn.n_family.sum(1), recal, pi_cal, eps)
        run.log(f"composition {co['name']}, prevalence {pi}, windows of {n_window.min()} to {n_window.max()} "
                f"sessions, online thresholds of the source run reproduced in every window")

        for pj, p in enumerate(policies):
            rng = np.random.default_rng(stream_seeds[ci * len(policies) + pj])
            stream = audited_stream(rng, dismissed[p], rate[p], q)
            out = judge_stream(tables, stream, rate[p], eps, delta, q, restarts)
            n_end = stream["offsets"][-1]
            total_windows = n_window.sum()
            held = dismissed[p] > 0
            for w in range(w_count):
                win = windows.iloc[w]
                block_rows.append({
                    "policy": p, "composition": co["name"], "prevalence": pi, "window": w, "year": int(win["year"]),
                    "block": int(win["block"]), "period": win["period"], "position": float(win["position"]),
                    "window_sessions": int(n_window[w]), "refused_rate": float(np.mean(~held[w])),
                    "threshold_mean": float(index[p][w][held[w]].mean()) if held[w].any() else np.nan,
                    "dismissed_share_mean": float((dismissed[p][w] / n_window[w]).mean()),
                    "for_true_mean": float(rate[p][w][held[w]].mean()) if held[w].any() else np.nan,
                    "for_true_median": float(np.median(rate[p][w][held[w]])) if held[w].any() else np.nan,
                    "violated_rate": float(np.mean(held[w] & (rate[p][w] > eps))),
                    "attacks_mean": float(stream["attacks"][w].mean()),
                    "realized_for_mean": float(np.mean(stream["attacks"][w][held[w]] / dismissed[p][w][held[w]]))
                    if held[w].any() else np.nan,
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
                    "policy": p, "composition": co["name"], "prevalence": pi, "arm": arm, "draws": draws,
                    "refused_rate": float(np.mean(n_end == 0)), "stream_sessions_mean": float(n_end.mean()),
                    "alarm_rate": float(alarmed.mean()), "false_alarm_rate": float(r["false_alarm"].mean()),
                    **_with_prefix("alarm_block", _quantiles(r["alarm_block"] + 1, alarmed)),
                    **_with_prefix("alarm_session", _quantiles(r["alarm_pos"], alarmed)),
                    **onset,
                    **_with_prefix("leak", _quantiles(r["leak"], np.ones(draws, dtype=bool))),
                    "leak_mean": float(r["leak"].mean()),
                    **_with_prefix("excess", _quantiles(r["leak"] - eps * r["sessions"], np.ones(draws, dtype=bool))),
                    **_with_prefix("sessions", _quantiles(r["sessions"], np.ones(draws, dtype=bool))),
                    "dismissed_share_mean": float((r["sessions"] / total_windows).mean()),
                    **_with_prefix("realized_for", _quantiles(np.divide(r["leak"], r["sessions"],
                                                                        out=np.zeros(draws), where=r["sessions"] > 0),
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
            key = f"{p}|{co['name']}|"
            saved[key + "attacks"] = stream["attacks"]
            saved[key + "dismissed"] = dismissed[p]
            saved[key + "for_true"] = rate[p].astype(np.float32)
            for arm in arms:
                saved[key + f"{arm}_alarm_pos"] = out[arm]["alarm_pos"]
                saved[key + f"{arm}_leak"] = out[arm]["leak"]
                if arm in MONITORS:
                    saved[key + f"{arm}_state"] = out[arm]["state"]
                    saved[key + f"{arm}_bound"] = out[arm]["bound"]
            if p == trace["policy"] and co["name"] == trace["composition"]:
                kept_trace = {**out["trace"], "window_sessions": n_window, "years": windows["year"].to_numpy()}
            by_arm = {x["arm"]: x for x in rows if x["policy"] == p and x["composition"] == co["name"]}
            line = (f"{p}, composition {co['name']}, population FOR above epsilon in a first block for "
                    f"{onset['onset_true_rate']:.3f} of draws at block {onset['onset_true_block_median']:.0f}, "
                    f"realized count above epsilon times the sessions at a block end for "
                    f"{onset['onset_realized_rate']:.3f} at block {onset['onset_realized_block_median']:.0f}")
            if "none" in by_arm:
                line += f", {by_arm['none']['leak_median']:.0f} attacks dismissed without monitoring"
            for arm in arms:
                if arm == "none":
                    continue
                x = by_arm[arm]
                when = (f"at block {x['alarm_block_median']:.0f} after {x['alarm_session_median']:.0f} sessions"
                        if x["alarm_rate"] > 0 else "never")
                line += (f", {ARM_SHORT[arm]} alarm in {x['alarm_rate']:.4f} of draws, false in "
                         f"{x['false_alarm_rate']:.4f}, {when}, leakage median {x['leak_median']:.0f}")
                if arm in MONITORS:
                    line += (f", bounds left in {x['miscover_rate']:.4f}, above the bound in {x['over_bound_rate']:.4f}"
                             f" of alarms, certified blocks {x['blocks_certified_mean']:.1f}")
                if arm != "monitor" and arm in MONITORS:
                    line += f", {x['sequences']} sequences, one left in {x['miscover_sequence_rate']:.4f}"
            run.log(line)

    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    run.save_table(pd.DataFrame(block_rows), "blocks")
    run.save_table(windows, "windows")
    run.save_arrays("draws", **saved)
    run.save_arrays("trace", **kept_trace)
    exceeds = {}
    for arm in [a for a in MONITORS if a in arms]:
        g = claims[claims["arm"] == arm]
        if arm == "monitor_year":
            # a sequence per year holds its level per sequence, so the check reads the share of failing sequences
            worst = g.loc[g["miscover_sequence_rate"].idxmax()]
            exceeds[arm] = bool((g["miscover_sequence_rate"] > delta).any())
            run.log(f"{ARM_SHORT[arm]}, largest share of sequences leaving the bounds "
                    f"{worst['miscover_sequence_rate']:.4f} on {worst['policy']} at composition "
                    f"{worst['composition']}, over {int(worst['sequences'])} sequences per draw, delta {delta}, "
                    f"above delta {exceeds[arm]}")
            continue
        worst = g.loc[g["miscover_rate"].idxmax()]
        exceeds[arm] = bool((g["miscover_low"] > delta).any())
        run.log(f"{ARM_SHORT[arm]}, largest share of draws leaving the bounds {worst['miscover_rate']:.4f} on "
                f"{worst['policy']} at composition {worst['composition']}, 95% interval {worst['miscover_low']:.4f} "
                f"to {worst['miscover_high']:.4f}, delta {delta}, above delta beyond simulation noise {exceeds[arm]}")
    run.save_json({
        "source": {**cfg["source"], "fingerprint": ssum["fingerprint"], "dataset": ssum["dataset"],
                   "processed_version": ssum["processed_version"], "draws": draws, "windows": w_count,
                   "years": ssum["years"], "periods": ssum["periods"], "blocks_per_year": ssum["blocks_per_year"],
                   "families": ssum["families"], "grid_thresholds": int(len(hist["grid"]))},
        "epsilon": eps, "delta": delta, "delta_lower": delta_lower, "kind": m["kind"],
        "lower_ratios": m["lower_ratios"], "upper_ratios": m["upper_ratios"], "audit_rate": q,
        "alarm_floor": rule.alarm_floor(eps),
        "calibration_prevalence": pi_cal, "policies": policies, "arms": arms, "compositions": compositions,
        "trace": trace, "miscoverage_exceeds_delta": exceeds,
        "restart_sequences": {"monitor_year": int(years.max()) + 1, "monitor_block": w_count},
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _policy_line(policy: str) -> dict:
    # the sixth policy has no slot in the method order, so it takes the ink color with its own marker
    color = style.INK if policy == "recalibration" else style.color_for(policy)
    return {"color": color, "marker": POLICY_MARKERS[policy], "label": POLICY_LABELS[policy]}


ARM_COLORS = {"monitor": style.COLORS[0], "monitor_year": style.SEQUENTIAL[8], "monitor_block": style.SEQUENTIAL[1],
              "window_audit": style.INK, "naive": style.GRAY, "none": style.GRAY}


def _arm_line(arm: str) -> dict:
    return {"color": ARM_COLORS[arm], "marker": ARM_MARKERS[arm], "label": ARM_LABELS[arm]}


def _period_lines(ax, summary: dict) -> None:
    deployment = summary["source"]["years"]["deployment"]
    for first, _ in summary["source"]["periods"].values():
        if deployment[0] < first <= deployment[-1]:
            ax.axvline(first, color=style.GRAY, linestyle=":", linewidth=1)


def fig_bounds_over_time(run_dir) -> plt.Figure:
    style.use()
    t = runs.load_arrays(run_dir, "trace")
    summary = runs.load_json(run_dir, "summary")
    eps = summary["epsilon"]
    scale = 1e6
    color = style.color_for("forge_simplex")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.6))
    # the count is known at the audited attacks and at the block ends, the bounds move at the audited attacks only
    x = np.concatenate([t["ev_pos"], t["ends"]]) / scale
    y = np.concatenate([t["ev_count"], t["cum_attacks"]])
    order = np.argsort(x, kind="stable")
    ax.plot(x[order], y[order], color=style.INK, linewidth=1, label="attacks among the dismissed sessions")
    pos = t["ev_pos"] / scale
    n_end = t["n_end"] / scale
    ax.step(np.r_[pos, n_end], np.r_[t["lcb"], t["lcb"][-1]], where="post", color=color, linewidth=1.2,
            label="lower bound")
    ax.step(np.r_[pos, n_end], np.r_[t["ucb"], t["ucb"][-1]], where="post", color=color, linestyle="--",
            linewidth=1.2, label="upper bound")
    ax.plot([0, n_end], [0, eps * t["n_end"]], color=style.GRAY, linestyle=":", linewidth=1,
            label=r"$\varepsilon$ times the dismissed sessions")
    # the first block of every year
    years = t["years"]
    for w in np.flatnonzero(np.r_[False, years[1:] != years[:-1]]):
        ax.axvline(t["ends"][w - 1] / scale, color=style.GRAY, linestyle=":", linewidth=0.6)
    if t["alarm_pos"] >= 0:
        ax.axvline(t["alarm_pos"] / scale, color=color, linestyle="-.", linewidth=0.8)
    ax.set_xlim(0, n_end)
    ax.set_ylim(0, 1.1 * max(float(t["ucb"].max()), float(y.max()), 1.0))
    ax.set_xlabel("millions of dismissed sessions")
    ax.set_ylabel("attacks")
    layout.legend_below(fig, ax, ["attacks among the dismissed sessions", "lower bound", "upper bound"], ncol=2)
    return fig


def fig_for_by_block(run_dir) -> plt.Figure:
    style.use()
    blocks = runs.load_table(run_dir, "blocks")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    fig, axes = plt.subplots(1, len(compositions), figsize=(style.WIDTH_DOUBLE, 2.6), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]
    # the population FOR of each policy's threshold in every block, a dismissed share of zero has no rate
    floor = summary["epsilon"] / 100
    deployment = summary["source"]["years"]["deployment"]
    for ax, co, letter in zip(axes, compositions, "abcdef"):
        g = blocks[blocks["composition"] == co["name"]]
        for p in policies:
            h = g[g["policy"] == p].sort_values("position")
            ax.plot(h["position"], h["for_true_mean"].clip(lower=floor), markersize=2.5, linewidth=1,
                    **_policy_line(p))
        ax.axhline(summary["epsilon"], color=style.GRAY, linestyle="--", linewidth=1, label=r"$\varepsilon$")
        _period_lines(ax, summary)
        ax.set_yscale("log")
        ax.set_xlim(deployment[0], deployment[-1] + 1)
        ax.set_xticks(np.arange(deployment[0], deployment[-1] + 2))
        ax.set_xlabel(f"deployment year, attack prevalence {style.percent(co['prevalence'])}")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("FOR of the block at the threshold")
    layout.legend_below(fig, axes[0], [POLICY_LABELS[p] for p in policies])
    return fig


def fig_alarm_by_policy(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    arms = [a for a in [*MONITORS, "window_audit", "naive"] if a in summary["arms"]]
    compositions = summary["compositions"]
    fig, axes = plt.subplots(2, len(compositions), figsize=(style.WIDTH_DOUBLE, 4.4), sharex=True, squeeze=False)
    x = np.arange(len(policies))
    # top, the share of draws with an alarm, filled, and with a false alarm, hollow. bottom, the block of the alarm
    # with its 5% to 95% range, and the first block whose population FOR exceeds epsilon
    for col, co in enumerate(compositions):
        g = claims[claims["composition"] == co["name"]]
        for k, arm in enumerate(arms):
            h = g[g["arm"] == arm].set_index("policy").reindex(policies)
            xs = layout.dodge(x, k, len(arms), 0.14)
            line = _arm_line(arm)
            axes[0, col].plot(xs, h["alarm_rate"], linestyle="none", markersize=4, **line)
            axes[0, col].plot(xs, h["false_alarm_rate"], linestyle="none", markersize=4, markerfacecolor="white",
                              color=line["color"], marker=line["marker"])
            layout.points(axes[1, col], xs, h["alarm_block_median"], h["alarm_block_low"], h["alarm_block_high"],
                          linestyle="none", **line)
        h = g[g["arm"] == arms[0]].set_index("policy").reindex(policies)
        axes[1, col].plot(x, h["onset_true_block_median"], linestyle="none", marker="_", markersize=9,
                          color=style.INK, label="first block with FOR above epsilon")
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_ylim(0, summary["source"]["windows"] + 1)
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels([POLICY_LABELS[p].split(",")[0] for p in policies], rotation=30, ha="right")
        axes[1, col].set_xlabel(f"threshold policy, attack prevalence {style.percent(co['prevalence'])}")
    axes[0, 0].set_ylabel("share of draws with an alarm")
    axes[1, 0].set_ylabel("block of the alarm")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
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
    fig, axes = plt.subplots(1, len(compositions), figsize=(style.WIDTH_DOUBLE, 2.8), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(policies))
    # the attacks dismissed until the switch off per policy and arm, and the bound of Theorem 3 at the alarm of
    # FORGE-Monitor
    for ax, co, letter in zip(axes, compositions, "abcdef"):
        g = claims[claims["composition"] == co["name"]]
        for k, arm in enumerate(arms):
            h = g[g["arm"] == arm].set_index("policy").reindex(policies)
            xs = layout.dodge(x, k, len(arms), 0.13)
            layout.points(ax, xs, h["leak_median"].clip(lower=1), h["leak_low"].clip(lower=1),
                          h["leak_high"].clip(lower=1), linestyle="none", **_arm_line(arm))
            if arm == "monitor":
                ax.plot(xs, h["bound_median"].clip(lower=1), linestyle="none", marker="s", markersize=4,
                        markerfacecolor="white", color=ARM_COLORS[arm], label="bound of Theorem 3")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[p].split(",")[0] for p in policies], rotation=30, ha="right")
        ax.set_xlabel(f"threshold policy, attack prevalence {style.percent(co['prevalence'])}")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("attacks dismissed until the switch off")
    layout.legend_below(fig, axes[0], [ARM_LABELS[a] for a in arms])
    return fig


FIGURES = {
    "bounds_over_time": fig_bounds_over_time,
    "for_by_block": fig_for_by_block,
    "alarm_by_policy": fig_alarm_by_policy,
    "leakage_by_policy": fig_leakage_by_policy,
}
