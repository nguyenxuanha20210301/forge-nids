"""e22, RQ2 on AnoShift, the monitor of e21 swept around the rule of the paper, a sequence per year with uniform
audits at 0.5%, on the same dismissed streams of the thresholds of e11 through the 72 blocks of 2010 to 2015.

the streams, the policies and the compositions are those of e21, read from the same e11 run and drawn again with
its seeds. every dismissed session falls into a score bin, the bins being the grid cut at the benign quantiles of the
selection year, and carries the attack and benign shares of its bin in the block's population at the threshold, so a
score dependent audit policy can audit it with the probability of its bin. the audited attacks of a window come at
geometric gaps with the mean audited attack rate over the bins, each takes its bin with the probability of an audited
attack falling there, and the attacks between two of them are binomial in the gap, as in e21.

the cells, each one step from the main cell, uniform audits at 0.5% with the floor 0.05%, the thinning kind, a
sequence per year, no label delay and the count as the target.
    rate         uniform audits at 0.1%, 1% and 2%
    allocation   q proportional to the square root of the attack share of the bin, Proposition 5, and to the share
                 itself, each with the floors 0.05% and 0.25%, the shares read from the selection year at the
                 calibration prevalence, at a mean of 0.5% over the sessions of the selection year
    kind         the Freedman bound of the earlier proposal in place of the audit martingale
    restart      one sequence for the deployment, one per block at delta over the blocks, and a sequence started at
                 every block at delta over the blocks with the first alarm of any of them, the any start rule, the
                 simplest e-detector
    delay        the labels of every audit arrive 25,000 or 75,000 sessions later, about one and three days of
                 sessions in the test files, so the sequence is read that far back and the alarm fires that much later
    target       the per step conditional FOR in the form of Timans et al. with the predictable bet that is growth
                 optimal against the running estimate, one bet instead of the mixture, which bounds no count

counted per policy, composition and cell as in e21, with the realized audits per million sessions of the stream and,
with a delay, the attacks dismissed between the reading and the alarm. figures leakage_by_rate, leakage_by_allocation,
leakage_by_rule and leakage_by_delay, each with the share of draws alarmed above the attacks dismissed until the
switch off.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forge import runs
from forge.certify.certificate import for_from_rates
from forge.data.synthetic import draw_counts
from forge.eval.metrics import share_with_interval
from forge.eval.windows import rates
from forge.experiments.monitor_streams import (OFFLINE, POLICIES, POLICY_LABELS, POLICY_MARKERS, STATE_CODES,
                                               _policy_line, _quantiles, _with_prefix, block_counts, load_source,
                                               recalibrate, window_stream)
from forge.monitor import rule
from forge.monitor.audit import allocate
from forge.monitor.sequence import KINDS, Sequence, Tables, gains, leakage_bound
from forge.viz import layout, style
from forge.viz.export import save_figure

RESTARTS = ("deployment", "year", "block", "any")
TARGETS = ("count", "step")
ALLOCATION_POWER = {"uniform": 0.0, "sqrt": 0.5, "proportional": 1.0}
AXES = ("rate", "allocation", "kind", "restart", "delay", "target")
RESTART_LABELS = {"deployment": "one sequence", "year": "a sequence per year", "block": "a sequence per block",
                  "any": "a sequence from every block"}
LOW, HIGH = 0.05, 0.95


def score_bins(hist: dict, pi_cal: float, bins: int) -> dict:
    """the grid cut into bins at the benign quantiles of the selection year, with the attack share and the session
    share of each bin at the calibration prevalence, as a policy would estimate them on the selection set."""
    benign = hist["selection_benign_below"] / hist["selection_sizes"][0]
    attack = hist["selection_family_below"].sum(0) / hist["selection_sizes"][1:].sum()
    g = len(benign)
    last = np.searchsorted(benign, np.arange(1, bins) / bins, side="left") - 1
    last = np.unique(np.r_[last[last >= 0], g - 1])
    lo = np.r_[-1, last[:-1]]
    with_zero = np.r_[0.0, benign], np.r_[0.0, attack]
    d_benign = with_zero[0][last + 1] - with_zero[0][lo + 1]
    d_attack = with_zero[1][last + 1] - with_zero[1][lo + 1]
    mass = pi_cal * d_attack + (1 - pi_cal) * d_benign
    r = np.divide(pi_cal * d_attack, mass, out=np.zeros(mass.shape), where=mass > 0)
    return {"last": last, "r": r, "mass": mass / mass.sum()}


def dismissed_shares(b_w, m_w, pi: float, last, t):
    """the attack and the benign share of the dismissed sessions in every bin, shape (draws, B), for the thresholds t
    of the draws, from the cdfs of the block's population at the grid. a threshold of -1 dismisses nothing."""
    t = np.asarray(t)
    lo = np.r_[-1, last[:-1]]
    hi = np.minimum(last[None, :], t[:, None])
    inside = hi > lo[None, :]
    cb, cm = np.r_[0.0, b_w], np.r_[0.0, m_w]
    attack = pi * np.where(inside, cm[hi + 1] - cm[lo + 1][None, :], 0.0)
    benign = (1 - pi) * np.where(inside, cb[hi + 1] - cb[lo + 1][None, :], 0.0)
    d = attack.sum(1) + benign.sum(1)
    safe = np.where(d > 0, d, 1.0)[:, None]
    return np.where(d[:, None] > 0, attack / safe, 0.0), np.where(d[:, None] > 0, benign / safe, 0.0)


def audited_stream(rng, dismissed, attack_share, benign_share, q_by_bin) -> dict:
    """the audited attacks of the streams of independent draws through windows in order, with score bins.

    dismissed has shape (W, draws), attack_share and benign_share (W, draws, B) the share of the dismissed sessions
    of each window that are attacks and benign sessions of each bin, q_by_bin (B,) the audit probability of each bin.
    every dismissed session is an attack of bin b with its share and audited with the probability of its bin, so the
    audited attacks come at geometric gaps with the mean audited attack rate, each falls into a bin with the share of
    the audited attacks there, and the attacks between two of them are binomial in the gap. returns the events as in
    e21 with the audit probability of each event, the attacks, audited attacks and audited benign sessions of each
    window, and the session index at which each window ends.
    """
    dismissed = np.asarray(dismissed, dtype=np.int64)
    attack_share = np.asarray(attack_share, dtype=float)
    benign_share = np.asarray(benign_share, dtype=float)
    q_by_bin = np.asarray(q_by_bin, dtype=float)
    w_count, draws = dismissed.shape
    offsets = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(dismissed, axis=0)])
    attacks = np.zeros((w_count, draws), dtype=np.int64)
    audited_attacks = np.zeros((w_count, draws), dtype=np.int64)
    audited_benign = np.zeros((w_count, draws), dtype=np.int64)
    parts = []
    before = np.zeros(draws, dtype=np.int64)
    for w in range(w_count):
        flows = dismissed[w]
        weighted = attack_share[w] * q_by_bin[None, :]
        f, p = attack_share[w].sum(1), weighted.sum(1)
        live = (flows > 0) & (p > 0)
        # the shares of a window are normalized, so when every dismissed session is an attack their sum can sit a
        # few ulp past 1 and the ratio with it, the clip keeps the binomial probabilities inside the unit interval
        rest = np.clip(np.where(live, (f - p) / (1 - p), f), 0.0, 1.0)
        if live.any():
            expected = (flows * p).max()
            m = int(np.ceil(expected + 10 * np.sqrt(expected) + 20))
            p_live = np.where(live, p, 1.0)[:, None]
            pos = np.cumsum(rng.geometric(p_live, size=(draws, m)), axis=1)
            while (live & (pos[:, -1] <= flows)).any():
                more = np.cumsum(rng.geometric(p_live, size=(draws, m)), axis=1)
                pos = np.concatenate([pos, pos[:, -1:] + more], axis=1)
            keep = live[:, None] & (pos <= flows[:, None])
            k = keep.sum(1)
            # the bin of an audited attack, with the share of the audited attacks of the window that fall there
            cdf = np.cumsum(weighted, axis=1) / np.where(live, p, 1.0)[:, None]
            u = rng.random(pos.shape)
            bin_of = np.minimum((u[:, :, None] > cdf[:, None, :]).sum(2), len(q_by_bin) - 1)
            previous = np.concatenate([np.zeros((draws, 1), dtype=np.int64), pos[:, :-1]], axis=1)
            between = rng.binomial(np.where(keep, pos - previous - 1, 0), rest[:, None])
            last = np.where(k > 0, np.take_along_axis(pos, np.maximum(k - 1, 0)[:, None], 1)[:, 0], 0)
        else:
            pos = np.zeros((draws, 0), dtype=np.int64)
            keep = np.zeros((draws, 0), dtype=bool)
            bin_of = np.zeros((draws, 0), dtype=np.int64)
            between = np.zeros((draws, 0), dtype=np.int64)
            k = np.zeros(draws, dtype=np.int64)
            last = np.zeros(draws, dtype=np.int64)
        tail = rng.binomial(np.maximum(flows - last, 0), rest)
        count = before[:, None] + np.cumsum(between + keep, axis=1)
        attacks[w] = between.sum(1) + k + tail
        audited_attacks[w] = k
        benign_sessions = np.rint(benign_share[w] * flows[:, None]).astype(np.int64)
        audited_benign[w] = rng.binomial(benign_sessions, q_by_bin[None, :]).sum(1)
        before = before + attacks[w]
        parts.append((np.where(keep, pos + offsets[w][:, None], -1), np.where(keep, count, 0),
                      np.where(keep, q_by_bin[bin_of], 1.0), keep))
    pos = np.concatenate([np.zeros((draws, 1), dtype=np.int64)] + [p for p, _, _, _ in parts], axis=1)
    count = np.concatenate([np.zeros((draws, 1), dtype=np.int64)] + [c for _, c, _, _ in parts], axis=1)
    q = np.concatenate([np.ones((draws, 1))] + [x for _, _, x, _ in parts], axis=1)
    window = np.concatenate([np.full((draws, 1), -1)] + [np.full(p.shape, w) for w, (p, _, _, _) in enumerate(parts)],
                            axis=1)
    valid = np.concatenate([np.ones((draws, 1), dtype=bool)] + [v for _, _, _, v in parts], axis=1)
    order = np.argsort(np.where(valid, pos, np.iinfo(np.int64).max), axis=1, kind="stable")
    width = int(valid.sum(1).max())

    def take(a):
        return np.take_along_axis(a, order, axis=1)[:, :width]

    return {"ev_pos": take(pos), "ev_count": take(count), "ev_window": take(window), "ev_q": take(q),
            "valid": take(valid), "attacks": attacks, "audited_attacks": audited_attacks,
            "audited_benign": audited_benign, "offsets": offsets}


def count_at(stream: dict, p) -> np.ndarray:
    # the attacks among the first p sessions of each draw, exact at the block ends and interpolated inside a block
    # with the block's rate, since only the audited attacks of a block have positions
    p = np.asarray(p, dtype=np.int64)
    offsets, attacks = stream["offsets"], stream["attacks"]
    draws = attacks.shape[1]
    rows = np.arange(draws)
    w = np.minimum((offsets[1:] < p[None, :]).sum(0), attacks.shape[0] - 1)
    size = offsets[1:][w, rows] - offsets[:-1][w, rows]
    cum = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(attacks, axis=0)])
    inside = np.clip(p - offsets[:-1][w, rows], 0, size)
    share = np.divide(inside, size, out=np.zeros(draws), where=size > 0)
    return cum[w, rows] + np.rint(share * attacks[w, rows]).astype(np.int64)


def segment_events(stream: dict, segments) -> dict:
    """the events of every draw with a virtual event at the start of every segment of windows, in the order of the
    stream, and for each event its segment, its window, the audited attacks, sessions and count since the segment
    began, and its audit probability, one for a virtual event."""
    ev_pos, ev_count, ev_win, valid = stream["ev_pos"], stream["ev_count"], stream["ev_window"], stream["valid"]
    ev_q = stream.get("ev_q", np.ones(ev_pos.shape))
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
    cum_attacks = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(attacks, axis=0)])
    seg_start, seg_end = offsets[first], offsets[np.r_[first[1:], w_count]]
    before = cum_attacks[first]
    seg_total = cum_attacks[np.r_[first[1:], w_count]] - before
    # the real events come first in the concatenation, so an audited attack on the last session of a segment sorts
    # before the virtual start of the next one and stays in its own segment
    real = valid.copy()
    real[:, 0] = False
    pos = np.concatenate([np.where(real, ev_pos, np.iinfo(np.int64).max), seg_start.T], axis=1)
    count = np.concatenate([ev_count, before.T], axis=1)
    q = np.concatenate([ev_q, np.ones((draws, n_seg))], axis=1)
    seg = np.concatenate([segments[np.maximum(ev_win, 0)], np.broadcast_to(np.arange(n_seg), (draws, n_seg))],
                         axis=1)
    win = np.concatenate([ev_win, np.broadcast_to(first, (draws, n_seg))], axis=1)
    virtual = np.concatenate([np.zeros(ev_pos.shape, dtype=bool), np.ones((draws, n_seg), dtype=bool)], axis=1)
    live = np.concatenate([real, np.ones((draws, n_seg), dtype=bool)], axis=1)
    order = np.argsort(pos, axis=1, kind="stable")
    width = n_seg + int(real.sum(1).max())
    pos, count, q, seg, win, virtual, live = (np.take_along_axis(a, order, axis=1)[:, :width]
                                              for a in (pos, count, q, seg, win, virtual, live))
    pos = np.where(live, pos, 0)
    rank = np.cumsum(live & ~virtual, axis=1)
    rank_at_start = rank[virtual].reshape(draws, n_seg)
    k = np.where(live, rank - rank_at_start[rows[:, None], seg], 0)
    n_in = np.where(live, pos - seg_start.T[rows[:, None], seg], 0)
    c_in = np.where(live, count - before.T[rows[:, None], seg], 0)
    same = np.zeros(pos.shape, dtype=bool)
    same[:, :-1] = live[:, 1:] & (seg[:, 1:] == seg[:, :-1])
    nxt = np.where(same, np.concatenate([c_in[:, 1:], np.zeros((draws, 1), dtype=np.int64)], axis=1),
                   seg_total.T[rows[:, None], seg] + 1)
    return {"pos": pos, "count": count, "q": q, "seg": seg, "win": win, "virtual": virtual, "live": live, "k": k,
            "n_in": n_in, "c_in": c_in, "nxt": nxt, "first": first, "seg_start": seg_start, "seg_end": seg_end,
            "before": before, "seg_total": seg_total, "length": (seg_end - seg_start).T, "sequences": n_seg,
            "cum_attacks": cum_attacks}


def _alarm_outcome(stream: dict, ev: dict, hit, floor: int, bound_in, eps: float, delay: int, ucb_t, k_block) -> dict:
    """what follows from the first hit of a restarted monitor, with the bound of Theorem 3 of the segment of the
    alarm plus those of the segments before it, and the reading moved by the delay."""
    offsets, attacks = stream["offsets"], stream["attacks"]
    draws = attacks.shape[1]
    rows = np.arange(draws)
    n_end = offsets[-1]
    total = ev["cum_attacks"][-1]
    alarmed = hit >= 0
    at = np.maximum(hit, 0)
    seg_at = ev["seg"][rows, at]
    p_in = np.where(alarmed, np.maximum(ev["n_in"][rows, at], floor), -1)
    p = np.where(alarmed, ev["seg_start"].T[rows, seg_at] + p_in, -1)
    g = np.where(alarmed, ev["count"][rows, at], -1)
    g_in = np.where(alarmed, ev["c_in"][rows, at], -1)
    finished = np.where(ev["length"] > 0, bound_in(np.maximum(ev["length"], 1)), 0)
    earlier = (np.arange(ev["sequences"])[None, :] < seg_at[:, None]) & alarmed[:, None]
    at_event = alarmed & (ev["n_in"][rows, at] > floor)
    bound = np.full(draws, -1, dtype=np.int64)
    if at_event.any():
        bound[at_event] = bound_in(p_in[at_event] - 1) + 1
    bound[alarmed & ~at_event] = floor
    bound = np.where(alarmed, bound + (finished * earlier).sum(1), -1)
    # with a delay the alarm fires that many sessions after the reading, and the attacks of the meantime leak too
    fired = np.where(alarmed, np.minimum(p + delay, n_end), -1)
    leak = np.where(alarmed, count_at(stream, np.maximum(fired, 0)) if delay > 0 else g, total)
    ends = offsets[1:]
    violation = alarmed & (fired[None, :] <= ends)
    n_block = ends - ev["seg_start"][ev["segments_of_blocks"]]
    certified = (ucb_t[k_block] <= eps * n_block) & (n_block > 0)
    state = np.where(violation, STATE_CODES["violation"],
                     np.where(certified, STATE_CODES["certified"], STATE_CODES["running"])).astype(np.int8)
    return {"alarmed": alarmed, "alarm_pos": fired, "reading_pos": p, "leak": leak,
            "leak_in_delay": np.where(alarmed, leak - g, 0), "sessions": np.where(alarmed, fired, n_end),
            "alarm_block": np.where(alarmed, (ends < fired[None, :]).sum(0), -1),
            "false_alarm": alarmed & (g_in <= eps * p_in), "bound": bound,
            "over_bound": alarmed & (np.where(alarmed, g, 0) > bound), "state": state}


def _segment_statistic(seq: Sequence, side: str, ev: dict):
    # the statistic of the sequence cumulated since the start of each event's segment. a bet of the upper side dies
    # at an audited attack at the floor, where its log factor is minus infinity, so the dead bets are counted apart
    # and a bet is dead in a segment once it died there, while the finite factors cumulate and are taken back to the
    # segment start
    live, virtual, q, seg = ev["live"], ev["virtual"], ev["q"], ev["seg"]
    draws = live.shape[0]
    rows = np.arange(draws)
    active = live & ~virtual
    if seq.kind == "thinning":
        step = gains(side, seq.bet[side], q)
        dead = active[..., None] & ~np.isfinite(step)
        finite = np.where(active[..., None] & ~dead, step, 0.0)
    else:
        dead = np.zeros(live.shape + (1,), dtype=bool)
        finite = np.where(active, 1 / q, 0.0)
    whole = np.cumsum(finite, axis=1)
    died = np.cumsum(dead, axis=1)
    at_start = whole[virtual].reshape(draws, ev["sequences"], *whole.shape[2:])
    died_at_start = died[virtual].reshape(draws, ev["sequences"], *died.shape[2:])
    stat = whole - at_start[rows[:, None], seg]
    dead_here = died - died_at_start[rows[:, None], seg] > 0
    if seq.kind == "thinning":
        return np.where(dead_here, -np.inf, stat)
    return stat


def monitor_cell(seq: Sequence, tables: Tables, stream: dict, eps: float, segments, delay: int = 0,
                 uniform: bool = True) -> dict:
    """FORGE-Monitor with a sequence per segment, uniform or score dependent audits, and a label delay, per draw.

    with uniform audits both bounds read from the tables by the audited attacks, otherwise from the audit
    probabilities of the events through the statistic of the sequence, with the bound of Theorem 3 read from the
    tables at the policy's least favorable probabilities in both cases.
    """
    ev = segment_events(stream, segments)
    draws = ev["pos"].shape[0]
    rows = np.arange(draws)
    k, live, virtual, seg = ev["k"], ev["live"], ev["virtual"], ev["seg"]
    lcb_t, ucb_t = tables.up_to(int(k.max()), int(ev["length"].max()), eps)
    if uniform:
        lcb, ucb = lcb_t[k], ucb_t[k]
    else:
        stat = {side: _segment_statistic(seq, side, ev) for side in ("lower", "upper")}
        lcb = seq.lower(k, stat["lower"])
        ucb = seq.upper(k, stat["upper"], int(ev["length"].max()) + 1)
    floor = rule.alarm_floor(eps)
    hit = rule.first_alarm(lcb, ev["n_in"], live, eps, floor)
    bad_lower = live & ~virtual & (ev["c_in"] < lcb)
    bad_upper = live & (ev["nxt"] - 1 > ucb)
    failed = np.zeros((draws, ev["sequences"]), dtype=bool)
    bad = bad_lower | bad_upper
    np.logical_or.at(failed, (rows[:, None].repeat(ev["pos"].shape[1], 1)[bad], seg[bad]), True)
    segments = np.asarray(segments)
    ev["segments_of_blocks"] = segments
    cum_audited = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(stream["audited_attacks"], axis=0)])
    k_block = cum_audited[1:] - cum_audited[ev["first"][segments]]

    def bound_in(n):
        return leakage_bound(n, lcb_t, ucb_t, eps)

    out = _alarm_outcome(stream, ev, hit, floor, bound_in, eps, delay, ucb_t, k_block)
    out.update({"miss_lower": bad_lower.any(1), "miss_upper": bad_upper.any(1), "failed_sequences": failed,
                "sequences": ev["sequences"]})
    return out


def any_start_monitor(tables: Tables, stream: dict, eps: float) -> dict:
    """FORGE-Monitor with a sequence started at every block and the first alarm of any of them, uniform audits, each
    sequence at the level of the tables, per draw. the sequence started at the first block certifies, and the bound of
    Theorem 3 at the alarm adds its bound on the attacks before the alarming sequence began."""
    ev_pos, ev_count, valid = stream["ev_pos"], stream["ev_count"], stream["valid"]
    offsets, attacks, audited = stream["offsets"], stream["attacks"], stream["audited_attacks"]
    w_count, draws = attacks.shape
    rows = np.arange(draws)
    n_end = offsets[-1]
    cum_attacks = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(attacks, axis=0)])
    cum_audited = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(audited, axis=0)])
    total = cum_attacks[-1]
    k_all = np.cumsum(valid, axis=1) - 1
    lcb_t, ucb_t = tables.up_to(int(k_all.max()), int(n_end.max()), eps)
    floor = rule.alarm_floor(eps)
    nxt_all = np.empty_like(ev_count)
    nxt_all[:, :-1] = np.where(valid[:, 1:], ev_count[:, 1:], total[:, None] + 1)
    nxt_all[:, -1] = total + 1
    best = np.full(draws, np.iinfo(np.int64).max)
    best_start, best_hit = np.full(draws, -1), np.full(draws, -1)
    miss_lower = np.zeros(draws, dtype=bool)
    miss_upper = np.zeros(draws, dtype=bool)
    for s in range(w_count):
        after = valid & (ev_pos > offsets[s][:, None])
        k = np.where(after, k_all - cum_audited[s][:, None], 0)
        n_in = np.where(after, ev_pos - offsets[s][:, None], 0)
        c_in = np.where(after, ev_count - cum_attacks[s][:, None], 0)
        lcb, ucb = lcb_t[k], ucb_t[k]
        hit = rule.first_alarm(lcb, n_in, after, eps, floor)
        alarmed = hit >= 0
        p = np.where(alarmed, offsets[s] + np.maximum(n_in[rows, np.maximum(hit, 0)], floor), np.iinfo(np.int64).max)
        better = p < best
        best, best_start, best_hit = np.where(better, p, best), np.where(better, s, best_start), \
            np.where(better, hit, best_hit)
        miss_lower |= (after & (c_in < lcb)).any(1)
        # the stretch from the start to the first event after it is read at zero audited attacks
        first_after = np.where(after.any(1), after.argmax(1), 0)
        before_first = np.where(after.any(1), c_in[rows, first_after] - 1, total - cum_attacks[s])
        miss_upper |= (before_first > ucb_t[0]) | (after & (nxt_all - cum_attacks[s][:, None] - 1 > ucb)).any(1)
    alarmed = best_start >= 0
    p = np.where(alarmed, best, -1)
    at = np.maximum(best_hit, 0)
    g = np.where(alarmed, ev_count[rows, at], -1)
    start = np.maximum(best_start, 0)
    g_in = np.where(alarmed, g - cum_attacks[start, rows], -1)
    p_in = np.where(alarmed, p - offsets[start, rows], -1)
    at_event = alarmed & (ev_pos[rows, at] - offsets[start, rows] > floor)
    bound = np.full(draws, -1, dtype=np.int64)
    if at_event.any():
        bound[at_event] = leakage_bound(p_in[at_event] - 1, lcb_t, ucb_t, eps) + 1
    bound[alarmed & ~at_event] = floor
    earlier = np.where(offsets[start, rows] > 0, leakage_bound(np.maximum(offsets[start, rows], 1), lcb_t, ucb_t, eps),
                       0)
    bound = np.where(alarmed, bound + earlier, -1)
    ends = offsets[1:]
    certified = (ucb_t[cum_audited[1:]] <= eps * ends) & (ends > 0)
    violation = alarmed & (p[None, :] <= ends)
    state = np.where(violation, STATE_CODES["violation"],
                     np.where(certified, STATE_CODES["certified"], STATE_CODES["running"])).astype(np.int8)
    return {"alarmed": alarmed, "alarm_pos": p, "reading_pos": p, "leak": np.where(alarmed, g, total),
            "leak_in_delay": np.zeros(draws, dtype=np.int64), "sessions": np.where(alarmed, p, n_end),
            "alarm_block": np.where(alarmed, (ends < p[None, :]).sum(0), -1),
            "false_alarm": alarmed & (g_in <= eps * p_in),
            "bound": bound, "over_bound": alarmed & (np.where(alarmed, g, 0) > bound), "state": state,
            "miss_lower": miss_lower, "miss_upper": miss_upper,
            "failed_sequences": (miss_lower | miss_upper)[:, None], "sequences": w_count}


def step_monitor(stream: dict, eps: float, q: float, level: float, segments) -> dict:
    """the per step target in the form of Timans et al., per draw, with uniform audits and a sequence per segment.

    the wealth multiplies by 1 + lam (z - eps) at every dismissed session, z being 1 / q at an audited attack and 0
    elsewhere, with the bet lam set before each audited attack as the growth optimal one against the running
    estimate of the FOR since the segment began, capped at half of 1 / eps. the alarm comes at the first audited
    attack from the floor on at which the wealth reaches 1 / delta. it tests that the conditional FOR stayed at or
    below eps at every step and bounds no count, so the leakage at the alarm is reported without a bound.
    """
    ev = segment_events(stream, segments)
    draws, width = ev["pos"].shape
    rows = np.arange(draws)
    live, virtual = ev["live"], ev["virtual"]
    k, n_in = ev["k"], ev["n_in"]
    prev_k = np.concatenate([np.zeros((draws, 1), dtype=np.int64), k[:, :-1]], axis=1)
    prev_n = np.concatenate([np.zeros((draws, 1), dtype=np.int64), n_in[:, :-1]], axis=1)
    estimate = np.divide(prev_k, q * prev_n, out=np.zeros(k.shape), where=prev_n > 0)
    lam = np.clip(q * (estimate - eps) / (eps * (1 - q * eps)), 0, 0.5 / eps)
    gap = np.where(live & ~virtual, n_in - prev_n - 1, 0)
    step = np.where(live & ~virtual, gap * np.log1p(-lam * eps) + np.log1p(lam * (1 / q - eps)), 0.0)
    whole = np.cumsum(step, axis=1)
    at_start = whole[virtual].reshape(draws, ev["sequences"])
    wealth = whole - at_start[rows[:, None], ev["seg"]]
    hit_mask = live & ~virtual & (wealth >= np.log(1 / level)) & (n_in >= rule.alarm_floor(eps))
    hit = np.where(hit_mask.any(1), hit_mask.argmax(1), -1)
    offsets, attacks = stream["offsets"], stream["attacks"]
    n_end = offsets[-1]
    total = ev["cum_attacks"][-1]
    alarmed = hit >= 0
    at = np.maximum(hit, 0)
    seg_at = ev["seg"][rows, at]
    p = np.where(alarmed, ev["seg_start"].T[rows, seg_at] + n_in[rows, at], -1)
    g = np.where(alarmed, ev["count"][rows, at], -1)
    ends = offsets[1:]
    return {"alarmed": alarmed, "alarm_pos": p, "reading_pos": p, "leak": np.where(alarmed, g, total),
            "leak_in_delay": np.zeros(draws, dtype=np.int64), "sessions": np.where(alarmed, p, n_end),
            "alarm_block": np.where(alarmed, (ends < p[None, :]).sum(0), -1),
            "false_alarm": alarmed & (ev["c_in"][rows, at] <= eps * n_in[rows, at]),
            "bound": np.full(draws, -1, dtype=np.int64), "over_bound": np.zeros(draws, dtype=bool),
            "state": np.where(alarmed & (p[None, :] <= ends), STATE_CODES["violation"],
                              STATE_CODES["running"]).astype(np.int8),
            "miss_lower": np.zeros(draws, dtype=bool), "miss_upper": np.zeros(draws, dtype=bool),
            "failed_sequences": np.zeros((draws, ev["sequences"]), dtype=bool), "sequences": ev["sequences"]}


def cells_of(cfg: dict) -> list[dict]:
    # the main cell and one step from it along every axis
    au, sw = cfg["audits"], cfg["sweep"]
    main = {"rate": au["rate"], "allocation": au["allocation"], "floor": au["floor"], "kind": cfg["monitor"]["kind"],
            "restart": au["restart"], "delay": 0, "target": "count"}
    cells = [{"name": "main", "axis": "main", **main}]
    for rate in sw.get("rates", []):
        cells.append({"name": f"rate_{rate}", "axis": "rate", **{**main, "rate": rate}})
    for a in sw.get("allocations", []):
        cells.append({"name": f"{a['allocation']}_floor_{a['floor']}", "axis": "allocation",
                      **{**main, "allocation": a["allocation"], "floor": a["floor"]}})
    for kind in sw.get("kinds", []):
        cells.append({"name": f"kind_{kind}", "axis": "kind", **{**main, "kind": kind}})
    for restart in sw.get("restarts", []):
        cells.append({"name": f"restart_{restart}", "axis": "restart", **{**main, "restart": restart}})
    for delay in sw.get("delays", []):
        cells.append({"name": f"delay_{delay}", "axis": "delay", **{**main, "delay": int(delay)}})
    for target in sw.get("targets", []):
        cells.append({"name": f"target_{target}", "axis": "target", **{**main, "target": target}})
    for cell in cells:
        if cell["allocation"] not in ALLOCATION_POWER or cell["kind"] not in KINDS or cell["restart"] not in RESTARTS \
                or cell["target"] not in TARGETS or cell["delay"] < 0:
            raise ValueError(f"cell {cell['name']} names an unknown allocation, kind, restart, target or delay")
        if cell["target"] == "step" and (cell["allocation"] != "uniform" or cell["restart"] == "any"):
            raise ValueError("the per step target runs with uniform audits and a sequence per segment")
        if cell["restart"] == "any" and cell["allocation"] != "uniform":
            raise ValueError("the any start rule runs with uniform audits")
    names = [c["name"] for c in cells]
    if len(set(names)) != len(names):
        raise ValueError("two cells share a name")
    return cells


def run(cfg: dict, run: runs.Run) -> None:
    m, au = cfg["monitor"], cfg["audits"]
    eps, delta, delta_lower = m["epsilon"], m["delta"], m["delta_lower"]
    if not 0 < delta_lower < delta:
        raise ValueError(f"delta_lower is {delta_lower}, expected a value between 0 and delta {delta}")
    policies, compositions = list(cfg["policies"]), list(cfg["compositions"])
    if set(policies) - set(POLICIES):
        raise ValueError(f"unknown policies {', '.join(sorted(set(policies) - set(POLICIES)))}")
    cells = cells_of(cfg)
    source = load_source(cfg)
    scfg, ssum, hist, sdraws, windows = (source[key] for key in ("cfg", "summary", "hist", "draws", "windows"))
    c = scfg["certify"]
    if c["epsilon"] != eps:
        raise ValueError(f"the monitor's epsilon {eps} differs from the certificate's {c['epsilon']}")
    draws, w_count = int(ssum["draws"]), len(windows)
    pi_cal = scfg["calibration"]["prevalence"]
    years = np.unique(windows["year"].to_numpy(), return_inverse=True)[1]
    segments_of = {"deployment": np.zeros(w_count, dtype=int), "year": years, "block": np.arange(w_count)}
    bins = score_bins(hist, pi_cal, int(cfg["bins"]))
    run.log(f"source {cfg['source']['experiment']} run {cfg['source']['run']}, {w_count} windows, {draws} draws, "
            f"{len(bins['last'])} score bins at the benign quantiles of the selection year, attack share per bin at "
            f"prevalence {pi_cal} " + ", ".join(f"{r:.5f}" for r in bins["r"]))

    # one audit policy per distinct rate, allocation and floor, the probability of each bin
    audit_policies = {}
    for cell in cells:
        key = (cell["rate"], cell["allocation"], cell["floor"])
        if key not in audit_policies:
            audit_policies[key] = allocate(bins["r"], bins["mass"], cell["rate"], cell["floor"],
                                           ALLOCATION_POWER[cell["allocation"]])
            run.log(f"audit policy rate {cell['rate']}, {cell['allocation']}, floor {cell['floor']}, probability by "
                    f"bin " + ", ".join(f"{q:.5f}" for q in audit_policies[key]))
    run.save_table(pd.DataFrame([{"rate": r, "allocation": a, "floor": f, "bin": b, "q": float(q)}
                                 for (r, a, f), qs in audit_policies.items() for b, q in enumerate(qs)]), "policies")
    run.save_table(pd.DataFrame(cells), "cells")

    # the thresholds of e11, the online ones read from the source run and the recalibration refit as in e21
    thresholds = {p: sdraws[f"{p}_index"].astype(np.int64) for p in OFFLINE if p in policies}
    dep = scfg["deployment"]
    k_online = dep["mixes"].index("window")
    cells_e11 = w_count * len(dep["prevalences"]) * len(dep["mixes"])
    seeds = np.random.SeedSequence(scfg["seed"]).spawn(1 + 2 * cells_e11)
    stream_seeds = np.random.SeedSequence(cfg["seed"]).spawn(len(compositions) * len(policies) * len(audit_policies))
    rows, saved = [], {}
    for ci, co in enumerate(compositions):
        pi = co["prevalence"]
        j = dep["prevalences"].index(pi)
        dismissed = {p: np.zeros((w_count, draws), dtype=np.int64) for p in policies}
        attack_share = {p: np.zeros((w_count, draws, len(bins["last"]))) for p in policies}
        benign_share = {p: np.zeros((w_count, draws, len(bins["last"]))) for p in policies}
        rate = {p: np.zeros((w_count, draws)) for p in policies}
        recal = np.full(draws, int(ssum["fixed_index"]), dtype=np.int64)
        for i, win in enumerate(windows.itertuples()):
            counts = block_counts(hist, i)
            b_w, m_w, _, shares_w = rates(counts)
            n_w = int(round(counts["sizes"][0] / (1 - pi)))
            rng = np.random.default_rng(seeds[window_stream(source, i, pi)])
            drawn = draw_counts(rng, n_w, pi, shares_w, b_w, m_w, draws)
            attack_below = drawn.family_below.sum(1)
            window_below = drawn.benign_below + attack_below
            mix = shares_w @ m_w
            curve = for_from_rates(pi, mix, b_w)
            for p in policies:
                if p == "forge_online":
                    skipped = [int(win.year), int(win.block), pi, "window"] in ssum["skipped"]
                    t = np.full(draws, -1, dtype=np.int64) if skipped else sdraws["online_index"][i, j, k_online]
                    t = t.astype(np.int64)
                elif p == "recalibration":
                    t = recal
                else:
                    t = thresholds[p]
                hit = t >= 0
                dismissed[p][i] = np.where(hit, np.take_along_axis(window_below, np.maximum(t, 0)[:, None], 1)[:, 0],
                                           0)
                rate[p][i] = np.where(hit, curve[np.maximum(t, 0)], 0.0)
                attack_share[p][i], benign_share[p][i] = dismissed_shares(b_w, mix, pi, bins["last"], t)
            if "recalibration" in policies:
                recal = recalibrate(window_below, attack_below, n_w, drawn.n_family.sum(1), recal, pi_cal, eps)
        run.log(f"composition {co['name']}, prevalence {pi}, windows drawn again with the seeds of the source run")

        for pj, p in enumerate(policies):
            total_windows = sum(int(round(block_counts(hist, i)["sizes"][0] / (1 - pi))) for i in range(w_count))
            for aj, (key, q_by_bin) in enumerate(audit_policies.items()):
                rng = np.random.default_rng(stream_seeds[(ci * len(policies) + pj) * len(audit_policies) + aj])
                stream = audited_stream(rng, dismissed[p], attack_share[p], benign_share[p], q_by_bin)
                n_end = stream["offsets"][-1]
                audits = (stream["audited_attacks"] + stream["audited_benign"]).sum(0)
                per_million = float(audits.sum() / max(n_end.sum(), 1) * 1e6)
                q_low, q_high = float(q_by_bin.min()), float(q_by_bin.max())
                uniform = np.allclose(q_by_bin, q_by_bin[0])
                for cell in [x for x in cells if (x["rate"], x["allocation"], x["floor"]) == key]:
                    split = w_count if cell["restart"] in ("block", "any") else 1
                    seq = Sequence(cell["kind"], m["lower_ratios"], m["upper_ratios"], cell["rate"], q_low,
                                   delta_lower / split, (delta - delta_lower) / split)
                    tables = Tables(seq, q_high, q_low)
                    if cell["target"] == "step":
                        out = step_monitor(stream, eps, cell["rate"], delta_lower / split, segments_of[cell["restart"]])
                    elif cell["restart"] == "any":
                        out = any_start_monitor(tables, stream, eps)
                    else:
                        out = monitor_cell(seq, tables, stream, eps, segments_of[cell["restart"]], cell["delay"],
                                           uniform)
                    alarmed = out["alarmed"]
                    either = out["miss_lower"] | out["miss_upper"]
                    row = {
                        "policy": p, "composition": co["name"], "prevalence": pi, **cell, "draws": draws,
                        "stream_sessions_mean": float(n_end.mean()), "audits_per_million": per_million,
                        "alarm_rate": float(alarmed.mean()), "false_alarm_rate": float(out["false_alarm"].mean()),
                        **_with_prefix("alarm_block", _quantiles(out["alarm_block"] + 1, alarmed)),
                        **_with_prefix("alarm_session", _quantiles(out["alarm_pos"], alarmed)),
                        **_with_prefix("leak", _quantiles(out["leak"], np.ones(draws, dtype=bool))),
                        "leak_mean": float(out["leak"].mean()),
                        **_with_prefix("leak_in_delay", _quantiles(out["leak_in_delay"], alarmed)),
                        **_with_prefix("excess", _quantiles(out["leak"] - eps * out["sessions"],
                                                            np.ones(draws, dtype=bool))),
                        **_with_prefix("sessions", _quantiles(out["sessions"], np.ones(draws, dtype=bool))),
                        "dismissed_share_mean": float((out["sessions"] / total_windows).mean()),
                        "miscover": int(either.sum()), **share_with_interval(int(either.sum()), draws, "miscover_"),
                        "miscover_lower_rate": float(out["miss_lower"].mean()),
                        "miscover_upper_rate": float(out["miss_upper"].mean()),
                        "sequences": int(out["sequences"]),
                        "miscover_sequence_rate": float(out["failed_sequences"].mean()),
                        **_with_prefix("bound", _quantiles(out["bound"], alarmed & (out["bound"] >= 0))),
                        "over_bound_rate": float(out["over_bound"].sum() / max(alarmed.sum(), 1)),
                        **{f"blocks_{name}_mean": float((out["state"] == code).sum(0).mean())
                           for name, code in STATE_CODES.items()},
                    }
                    rows.append(row)
                    key_name = f"{p}|{co['name']}|{cell['name']}|"
                    saved[key_name + "alarm_pos"] = out["alarm_pos"]
                    saved[key_name + "leak"] = out["leak"]
                    saved[key_name + "bound"] = out["bound"]
            lines = [x for x in rows if x["policy"] == p and x["composition"] == co["name"]]
            main = next(x for x in lines if x["name"] == "main")
            run.log(f"{p}, composition {co['name']}, main cell alarm in {main['alarm_rate']:.4f} of draws, false in "
                    f"{main['false_alarm_rate']:.4f}, at block {main['alarm_block_median']:.0f}, leakage median "
                    f"{main['leak_median']:.0f}, {main['audits_per_million']:.0f} audits per million sessions, "
                    + ", ".join(f"{x['name']} alarm {x['alarm_rate']:.3f} false {x['false_alarm_rate']:.3f} leakage "
                                f"{x['leak_median']:.0f} audits {x['audits_per_million']:.0f}"
                                for x in lines if x["name"] != "main"))
    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    run.save_arrays("draws", **saved)
    # the level of a cell is delta per sequence, so the check reads the share of failing sequences, except for the
    # per step target, which bounds no count
    counted = claims[claims["target"] == "count"]
    worst = counted.loc[counted["miscover_sequence_rate"].idxmax()]
    exceeds = bool((counted["miscover_sequence_rate"] > delta).any())
    run.log(f"largest share of sequences leaving the bounds {worst['miscover_sequence_rate']:.4f} in cell "
            f"{worst['name']} on {worst['policy']} at composition {worst['composition']}, delta {delta}, above delta "
            f"{exceeds}, largest share of alarms above the bound of Theorem 3 {counted['over_bound_rate'].max():.4f}")
    run.save_json({
        "source": {**cfg["source"], "fingerprint": ssum["fingerprint"], "draws": draws, "windows": w_count},
        "epsilon": eps, "delta": delta, "delta_lower": delta_lower, "lower_ratios": m["lower_ratios"],
        "upper_ratios": m["upper_ratios"], "alarm_floor": rule.alarm_floor(eps), "bins": int(cfg["bins"]),
        "bin_last": bins["last"].tolist(), "bin_attack_share": bins["r"].tolist(), "bin_mass": bins["mass"].tolist(),
        "calibration_prevalence": pi_cal, "policies": policies, "compositions": compositions, "cells": cells,
        "main": {k: v for k, v in cells[0].items() if k in AXES or k == "floor"},
        "miscoverage_exceeds_delta": exceeds,
    }, "summary")
    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _axis_figure(run_dir, axis: str, order, labels, xlabel: str, log_x: bool = False) -> plt.Figure:
    # the share of draws alarmed above the attacks dismissed until the switch off, one series per policy, the cells
    # of one axis in the given order with the main cell among them, one column per composition. a numeric axis joins
    # the points of a policy, a categorical one sets them side by side
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    data = claims[(claims["axis"] == axis) | (claims["name"] == "main")]
    fig, axes = plt.subplots(2, len(compositions), figsize=(style.WIDTH_DOUBLE, 4.4), sharex=True, squeeze=False)
    x = np.arange(len(order)) if not log_x else np.asarray(order, dtype=float)
    for col, co in enumerate(compositions):
        g = data[data["composition"] == co["name"]]
        for k, p in enumerate(policies):
            h = g[g["policy"] == p].set_index("name").reindex([labels[o] for o in order])
            xs = x if log_x else layout.dodge(x, k, len(policies), 0.1)
            line = _policy_line(p)
            joined = {} if log_x else {"linestyle": "none"}
            axes[0, col].plot(xs, h["alarm_rate"], markersize=3.5, linewidth=1, **joined, **line)
            layout.points(axes[1, col], xs, h["leak_median"].clip(lower=1), h["leak_low"].clip(lower=1),
                          h["leak_high"].clip(lower=1), **joined, **line)
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_yscale("log")
        if log_x:
            axes[1, col].set_xscale("log")
            axes[1, col].set_xticks(x)
            axes[1, col].set_xticklabels([style.percent(v) for v in order])
        else:
            axes[1, col].set_xticks(x)
            axes[1, col].set_xticklabels([str(o) for o in order], rotation=30, ha="right")
        axes[1, col].set_xlabel(f"{xlabel}, attack prevalence {style.percent(co['prevalence'])}")
    axes[0, 0].set_ylabel("share of draws with an alarm")
    axes[1, 0].set_ylabel("attacks dismissed until the switch off")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[0, 0], [POLICY_LABELS[p] for p in policies])
    return fig


def _cell_names(run_dir, axis: str) -> dict:
    # the cell names of one axis keyed by the value on that axis, the main cell included
    summary = runs.load_json(run_dir, "summary")
    main = summary["main"]
    names = {}
    for cell in summary["cells"]:
        if cell["axis"] == axis or cell["name"] == "main":
            value = cell[axis]
            if axis == "allocation":
                value = f"{cell['allocation']}, floor {style.percent(cell['floor'])}"
            names[value] = cell["name"]
    return names, main


def fig_leakage_by_rate(run_dir) -> plt.Figure:
    names, _ = _cell_names(run_dir, "rate")
    order = sorted(names)
    return _axis_figure(run_dir, "rate", order, names, "uniform audit rate", log_x=True)


def fig_leakage_by_allocation(run_dir) -> plt.Figure:
    names, main = _cell_names(run_dir, "allocation")
    order = sorted(names, key=lambda v: (v.split(",")[0] != main["allocation"], v))
    return _axis_figure(run_dir, "allocation", order, names, "audit allocation and floor")


def fig_leakage_by_rule(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    # the restart rules, the kind and the target in one row, the main cell first
    cells = [c for c in summary["cells"] if c["axis"] in ("restart", "kind", "target") or c["name"] == "main"]
    labels = {}
    for c in cells:
        if c["axis"] == "kind":
            labels[c["name"]] = f"{c['kind']} kind"
        elif c["axis"] == "target":
            labels[c["name"]] = "per step target"
        else:
            labels[c["name"]] = RESTART_LABELS[c["restart"]]
    order = [c["name"] for c in cells]
    names = {labels[n]: n for n in order}
    policies = [p for p in POLICIES if p in summary["policies"]]
    compositions = summary["compositions"]
    data = claims[claims["name"].isin(order)]
    fig, axes = plt.subplots(2, len(compositions), figsize=(style.WIDTH_DOUBLE, 4.4), sharex=True, squeeze=False)
    x = np.arange(len(order))
    for col, co in enumerate(compositions):
        g = data[data["composition"] == co["name"]]
        for k, p in enumerate(policies):
            h = g[g["policy"] == p].set_index("name").reindex(order)
            xs = layout.dodge(x, k, len(policies), 0.1)
            line = _policy_line(p)
            axes[0, col].plot(xs, h["alarm_rate"], linestyle="none", markersize=3.5, **line)
            layout.points(axes[1, col], xs, h["leak_median"].clip(lower=1), h["leak_low"].clip(lower=1),
                          h["leak_high"].clip(lower=1), linestyle="none", **line)
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels([labels[n] for n in order], rotation=30, ha="right")
        axes[1, col].set_xlabel(f"monitor rule, attack prevalence {style.percent(co['prevalence'])}")
    axes[0, 0].set_ylabel("share of draws with an alarm")
    axes[1, 0].set_ylabel("attacks dismissed until the switch off")
    for ax, letter in zip(axes.flat, "abcdef"):
        layout.panel_letter(ax, letter)
    layout.legend_below(fig, axes[0, 0], [POLICY_LABELS[p] for p in policies])
    return fig


def fig_leakage_by_delay(run_dir) -> plt.Figure:
    names, _ = _cell_names(run_dir, "delay")
    order = sorted(names)
    return _axis_figure(run_dir, "delay", order, names, "label delay in sessions")


FIGURES = {
    "leakage_by_rate": fig_leakage_by_rate,
    "leakage_by_allocation": fig_leakage_by_allocation,
    "leakage_by_rule": fig_leakage_by_rule,
    "leakage_by_delay": fig_leakage_by_delay,
}
