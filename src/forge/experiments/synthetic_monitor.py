"""e20, a check of Theorems 2 and 3 and Proposition 5 on synthetic streams of dismissed flows.

a stream is the sequence of dismissed flows of a deployment, each an attack with a probability given by the scenario
in units of epsilon, which stays put, rises, jumps or bursts, see the config. every dismissed flow is audited with a
known probability, the same for all flows or depending on the score bin of the flow, and only the audited flows show
their label. the sequence of Theorem 2 bounds the number of attacks among the dismissed flows at every moment at once
from the audited attacks alone. a draw is one stream with its coins, or, for the scenarios listed under fixed, one set
of coins on a stream drawn once, so that only the coins vary, which is the event the theorem speaks about.

what is counted for each scenario, audit policy, label delay and kind of sequence.
    miscoverage      the share of draws in which the true count leaves the bounds at some moment, against delta
    alarm and delay  the share of draws in which the lower bound exceeds epsilon times the dismissed count, the moment
                     it does, counted from the start of the violation, and whether the count was really above epsilon
                     times the dismissed count then
    leakage          the attacks among the dismissed flows when the switch off takes effect, its excess over epsilon
                     times the dismissed count, and the bound of Theorem 3 on it
    certification    when the upper bound first falls to epsilon times the dismissed count, the state in which the
                     guarantee rests on the audits alone, and the share of the stream spent there
    audits           the audited attacks at the alarm and the expected audits of benign flows up to it

the bounds move only at audited attacks and the alarm can only fire there, so a draw reduces to the positions of its
audited attacks and the attack count at each, which keeps 10^4 draws per cell cheap. a second simulation walks flow by
flow with an adversary who reads the past coins and the bounds before placing each attack, and with labels that arrive
later for attacks than for benign flows, see adaptive in the config.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from forge import runs
from forge.data.synthetic import attack_bins, attack_positions
from forge.eval.metrics import share_with_interval
from forge.monitor import rule
from forge.monitor.audit import allocate, bin_profile
from forge.monitor.sequence import KINDS, Sequence, leakage_bound
from forge.viz import layout, style
from forge.viz.export import save_figure

KIND_LABELS = {"thinning": "audit martingale", "freedman": "Freedman bound"}
KIND_DASHES = {"thinning": "-", "freedman": "--"}
ALLOCATION_DASHES = {"uniform": "-", "sqrt": "--", "proportional": ":"}
# bits of the per draw flags
FLAGS = {"miscover_lower": 1, "miscover_upper": 2, "alarm": 4, "false_alarm": 8}
LOW, HIGH = 0.05, 0.95


def segments_of(scenario: dict, eps: float) -> list[tuple[int, float]]:
    # the (flows, probability) segments of a scenario, a ramp becomes equal steps from start to end
    if "ramp" in scenario:
        r = scenario["ramp"]
        steps = int(r["steps"])
        flows = [r["flows"] // steps] * steps
        flows[-1] += r["flows"] - sum(flows)
        return [(int(f), float(x) * eps) for f, x in zip(flows, np.linspace(r["start"], r["end"], steps))]
    return [(int(s["flows"]), float(s["rate"]) * eps) for s in scenario["segments"]]


def violation_start(segments, eps: float) -> int:
    # the first flow whose attack probability exceeds eps, one based, 0 when none does
    n = 0
    for flows, p in segments:
        if p > eps:
            return n + 1
        n += flows
    return 0


def audited_events(coins, positions, bins, q_by_bin, n_end: int):
    """the audited attacks of each draw as events padded on the right, after a virtual first event at position 0.

    coins holds one uniform draw per attack, and an attack is audited when its coin falls below the probability of its
    bin, so the policies of a chunk share their coins and differ in their probabilities only. returns the position of
    each event, the attack count at it, the audit probability of the audited attack and the mask of the events that
    exist. the attacks of a row are sorted, so the count at an attack is its column plus one.
    """
    valid_attack = positions <= n_end
    q_attack = q_by_bin[bins]
    audited = valid_attack & (coins < q_attack)
    rank = np.cumsum(audited, axis=1)
    width = int(rank[:, -1].max()) + 1 if rank.size else 1
    draws = positions.shape[0]
    ev_pos = np.zeros((draws, width), dtype=np.int64)
    ev_g = np.zeros((draws, width), dtype=np.int64)
    ev_q = np.ones((draws, width))
    valid = np.zeros((draws, width), dtype=bool)
    valid[:, 0] = True
    row, col = np.nonzero(audited)
    at = rank[row, col]
    ev_pos[row, at] = positions[row, col]
    ev_g[row, at] = col + 1
    ev_q[row, at] = q_attack[row, col]
    valid[row, at] = True
    return ev_pos, ev_g, ev_q, valid


class Tables:
    # both bounds by the number of audited attacks at the least favorable audit probabilities of a policy, grown on
    # demand, and the lower bound of a uniform policy reads from them as well
    def __init__(self, seq: Sequence, q_high: float, q_low: float):
        self.seq, self.q_high, self.q_low = seq, q_high, q_low
        self.lcb = self.ucb = np.zeros(0)

    def up_to(self, k_max: int, n_end: int, eps: float):
        # the lower bound must pass eps n_end so that the bound of Theorem 3 can be read at any dismissed count
        while len(self.lcb) <= k_max or self.lcb[-1] <= eps * n_end:
            size = max(k_max, 2 * (len(self.lcb) - 1), 64)
            self.lcb = self.seq.tables(self.q_high, size)[0]
            self.ucb = self.seq.tables(self.q_low, size)[1]
        return self.lcb, self.ucb


def judge(seq, tables, events, positions, counts, n_end, eps, delays, q_by_bin, profile, mass):
    """per draw outcomes of one sequence on the events of one policy, see the module docstring."""
    ev_pos, ev_g, ev_q, valid = events
    draws, width = ev_pos.shape
    k = np.broadcast_to(np.arange(width), ev_pos.shape)
    lcb_high, ucb_low = tables.up_to(width - 1, n_end, eps)
    if np.allclose(q_by_bin, q_by_bin[0]):
        lcb, ucb = lcb_high[k], ucb_low[k]
    else:
        lcb = seq.lower(k, seq.statistic("lower", ev_q, valid))
        ucb = seq.upper(k, seq.statistic("upper", ev_q, valid), n_end)
    nxt = np.full_like(ev_pos, n_end + 1)
    nxt[:, :-1] = np.where(valid[:, 1:], ev_pos[:, 1:], n_end + 1)
    next_g = np.empty_like(ev_g)
    next_g[:, :-1] = np.where(valid[:, 1:], ev_g[:, 1:], counts[:, None] + 1)
    next_g[:, -1] = counts + 1
    # the count at an event is the attack count there, and it reaches the next count less one before the next event
    miss_lower = (valid & (ev_g < lcb)).any(1)
    miss_upper = (valid & (next_g - 1 > ucb)).any(1)
    r_alarm = rule.first_alarm(lcb, ev_pos, valid, eps)
    alarmed = r_alarm >= 0
    rows = np.arange(draws)
    at = np.maximum(r_alarm, 0)
    p_alarm = np.where(alarmed, ev_pos[rows, at], -1)
    g_alarm = np.where(alarmed, ev_g[rows, at], -1)
    false_alarm = alarmed & (g_alarm <= eps * p_alarm)
    bound = np.full(draws, -1, dtype=np.int64)
    if alarmed.any():
        # the attack at the alarm itself comes on top of the bound on the flows before it
        bound[alarmed] = leakage_bound(p_alarm[alarmed] - 1, lcb_high, ucb_low, eps) + 1
    stop = np.where(alarmed, p_alarm, n_end + 1)
    start, length = rule.certified_stretches(ucb, ev_pos, nxt, valid, eps, stop)
    certified = (length > 0).any(1)
    certified_time = np.where(certified, np.where(length > 0, start, np.iinfo(np.int64).max).min(1), -1)
    until = np.minimum(stop, n_end)
    certified_share = length.sum(1) / until
    leak = {}
    for d in delays:
        tau = np.where(alarmed, np.minimum(p_alarm + d, n_end), n_end)
        leak[d] = (positions <= tau[:, None]).sum(1)
    # the audits up to the switch off, the audited attacks and the expected audits of benign flows by bin
    benign = np.maximum(mass[None, :] * until[:, None] - leak[0][:, None] * profile[None, :], 0)
    audits = np.where(alarmed, r_alarm, valid.sum(1) - 1) + (benign * q_by_bin[None, :]).sum(1)
    flags = (miss_lower * FLAGS["miscover_lower"] + miss_upper * FLAGS["miscover_upper"] + alarmed * FLAGS["alarm"]
             + false_alarm * FLAGS["false_alarm"]).astype(np.uint8)
    return {"flags": flags, "alarm_time": p_alarm.astype(np.int64), "alarm_audited": r_alarm.astype(np.int64),
            "leak_at_alarm": g_alarm.astype(np.int64), "bound": bound, "certified_time": certified_time,
            "certified_share": certified_share.astype(np.float32), "audits": audits.astype(np.float32),
            "audits_per_million": (audits / until * 1e6).astype(np.float32),
            "leak": {d: v.astype(np.int64) for d, v in leak.items()},
            "trace": {"ev_pos": ev_pos[0], "ev_g": ev_g[0], "ev_q": ev_q[0], "valid": valid[0], "lcb": lcb[0],
                      "ucb": ucb[0], "alarm_time": p_alarm[0]}}


def simulate_adaptive(rng, cell: dict, eps: float, q: float, lcb_t, ucb_t, draws: int, flows: int) -> dict:
    """flow by flow, an adversary who reads the past coins and the bounds the monitor has read before placing each
    attack, and labels that arrive later for audited attacks than for audited benign flows.

    lcb_t and ucb_t are the bounds by the number of audited attacks for uniform audits at probability q. returns the
    per draw outcomes of the monitor on the stream of flows dismissed flows.
    """
    f = cell["rate"] * eps
    evade, skip = cell.get("evade_below"), bool(cell.get("skip_after_audit", False))
    d_b, d_a = int(cell["benign_delay"]), int(cell["attack_delay"])
    if len(lcb_t) < flows + 1:
        raise ValueError("the tables must reach the number of flows, every flow could be an audited attack")
    rows = np.arange(draws)
    count = np.zeros(draws, dtype=np.int64)
    k = np.zeros(draws, dtype=np.int64)
    audited_at = np.full((draws, flows + 1), flows + 1, dtype=np.int64)
    pointer = np.zeros(draws, dtype=np.int64)
    prev_audited = np.zeros(draws, dtype=bool)
    miss_lower = np.zeros(draws, dtype=bool)
    miss_upper = np.zeros(draws, dtype=bool)
    alarm_time = np.full(draws, -1, dtype=np.int64)
    leak_at_alarm = np.full(draws, -1, dtype=np.int64)
    leak_at_reading = np.full(draws, -1, dtype=np.int64)
    bound = np.full(draws, -1, dtype=np.int64)
    attacked = np.zeros(draws, dtype=np.int64)
    count_at = np.zeros((draws, flows + 1), dtype=np.int64)
    for n in range(1, flows + 1):
        # what the monitor had read at the end of the flow before, the adversary reads the same
        pending = audited_at[rows, pointer]
        n_read = rule.reading(n - 1, d_b, d_a, pending)
        quiet = np.ones(draws, dtype=bool) if evade is None else lcb_t[pointer] <= evade * eps * n_read
        allowed = quiet & ~(prev_audited if skip else np.zeros(draws, dtype=bool))
        attack = allowed & (rng.random(draws) < f)
        attacked += attack
        count += attack
        count_at[:, n] = count
        coin = rng.random(draws) < q
        hit = attack & coin
        audited_at[rows[hit], k[hit]] = n
        k += hit
        prev_audited = coin
        miss_lower |= count < lcb_t[k]
        miss_upper |= count > ucb_t[k]
        # the label of an audited attack at position p arrives d_a flows later, so by the end of flow n the pointer
        # passes the audits at or before n - d_a, at most one per flow since positions are distinct
        pointer += (pointer < k) & (audited_at[rows, pointer] <= n - d_a)
        pending = audited_at[rows, pointer]
        n_read = rule.reading(n, d_b, d_a, pending)
        fresh = (alarm_time < 0) & (lcb_t[pointer] > eps * n_read)
        if fresh.any():
            alarm_time[fresh] = n
            leak_at_alarm[fresh] = count[fresh]
            leak_at_reading[fresh] = count_at[rows[fresh], n_read[fresh]]
            bound[fresh] = leakage_bound(n_read[fresh] - 1, lcb_t, ucb_t, eps) + 1
    alarmed = alarm_time >= 0
    n_read_end = rule.reading(flows, d_b, d_a, audited_at[rows, pointer])
    false_alarm = alarmed & (leak_at_alarm <= eps * alarm_time)
    return {"miss_lower": miss_lower, "miss_upper": miss_upper, "alarmed": alarmed, "false_alarm": false_alarm,
            "alarm_time": alarm_time, "leak_at_alarm": leak_at_alarm, "leak_at_reading": leak_at_reading,
            "bound": bound, "count": count, "attacked": attacked, "audited": k, "lag_at_end": flows - n_read_end}


def _quantiles(values, mask) -> dict:
    v = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    if len(v) == 0:
        return {"median": np.nan, "low": np.nan, "high": np.nan}
    return {"median": float(np.median(v)), "low": float(np.quantile(v, LOW)), "high": float(np.quantile(v, HIGH))}


def _with_prefix(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in d.items()}


def run(cfg: dict, run: runs.Run) -> None:
    m, au, st, ad, sim = cfg["monitor"], cfg["audits"], cfg["stream"], cfg["adaptive"], cfg["simulation"]
    eps, delta, q_min = m["epsilon"], m["delta"], m["q_min"]
    delta_lower = m["delta_lower"]
    if not 0 < delta_lower < delta:
        raise ValueError(f"delta_lower is {delta_lower}, expected a value between 0 and delta {delta}")
    delta_upper = delta - delta_lower
    kinds, rates, delays = list(m["kinds"]), list(au["rates"]), [int(d) for d in au["delays"]]
    allocations = dict(au["allocations"])
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kinds {', '.join(sorted(unknown))}")
    if "thinning" not in kinds or allocations.get("uniform") != 0 or 0 not in delays:
        raise ValueError("the main cell needs the thinning kind, the uniform allocation at power 0 and the delay 0")
    scenarios = list(st["scenarios"])
    names = [sc["name"] for sc in scenarios]
    if st["trace"] not in names:
        raise ValueError(f"the trace scenario {st['trace']} is not among the scenarios")
    bins = int(st["bins"])
    profile = bin_profile(bins, float(st["bin_ratio"]))
    mass = np.full(bins, 1 / bins)
    main_rate = rates[0]

    # the audit probability per score bin of every policy, from the attack shares of the bins as if estimated on the
    # selection set at a FOR of eps / 2
    policies = [(rate, name, allocate(profile * bins * eps / 2, mass, rate, q_min, power))
                for rate in rates for name, power in allocations.items()]
    run.save_table(pd.DataFrame([{"rate": rate, "allocation": name, "bin": b, "q": float(q)}
                                 for rate, name, q_by_bin in policies for b, q in enumerate(q_by_bin)]), "policies")
    sequences, tables, bet_rows = {}, {}, []
    for rate, name, q_by_bin in policies:
        q_low, q_high = float(q_by_bin.min()), float(q_by_bin.max())
        for kind in kinds:
            seq = Sequence(kind, m["lower_ratios"], m["upper_ratios"], rate, q_low, delta_lower, delta_upper)
            sequences[(rate, name, kind)] = seq
            tables[(rate, name, kind)] = Tables(seq, q_high, q_low)
            for side, ratios in (("lower", m["lower_ratios"]), ("upper", m["upper_ratios"])):
                bet_rows += [{"rate": rate, "allocation": name, "kind": kind, "side": side, "ratio": r, "bet": float(b)}
                             for r, b in zip(ratios, seq.bet[side])]
    run.save_table(pd.DataFrame(bet_rows), "bets")

    segment_rows, starts, lengths = [], {}, {}
    for sc in scenarios:
        segments = segments_of(sc, eps)
        starts[sc["name"]] = violation_start(segments, eps)
        lengths[sc["name"]] = sum(flows for flows, _ in segments)
        segment_rows += [{"scenario": sc["name"], "segment": i, "flows": flows, "rate": p / eps,
                          "violation_start": starts[sc["name"]], "fixed": sc["name"] in st.get("fixed", [])}
                         for i, (flows, p) in enumerate(segments)]
    run.save_table(pd.DataFrame(segment_rows), "scenarios")

    draws, chunk = int(sim["draws"]), int(sim["draws_per_chunk"])
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(len(scenarios) + len(ad["cells"]))
    records = {}
    trace = None
    for i, sc in enumerate(scenarios):
        rng = np.random.default_rng(seeds[i])
        segments = segments_of(sc, eps)
        n_end = lengths[sc["name"]]
        fixed = sc["name"] in st.get("fixed", [])
        if fixed:
            one_pos, one_cnt = attack_positions(rng, segments, 1)
            one_bins = attack_bins(rng, one_pos.shape, profile)
        done = 0
        while done < draws:
            d = min(chunk, draws - done)
            if fixed:
                pos = np.broadcast_to(one_pos, (d, one_pos.shape[1]))
                cnt = np.broadcast_to(one_cnt, (d,))
                bn = np.broadcast_to(one_bins, pos.shape)
            else:
                pos, cnt = attack_positions(rng, segments, d)
                bn = attack_bins(rng, pos.shape, profile)
            coins = rng.random(pos.shape)
            for rate, name, q_by_bin in policies:
                events = audited_events(coins, pos, bn, q_by_bin, n_end)
                for kind in kinds:
                    key = (sc["name"], rate, name, kind)
                    out = judge(sequences[(rate, name, kind)], tables[(rate, name, kind)], events, pos, cnt, n_end,
                                eps, delays, q_by_bin, profile, mass)
                    if done == 0 and sc["name"] == st["trace"] and key[1:] == (main_rate, "uniform", "thinning"):
                        trace = {**out["trace"], "attacks": pos[0][pos[0] <= n_end], "n_end": n_end,
                                 "violation_start": starts[sc["name"]], "rate": main_rate}
                    rec = records.setdefault(key, {})
                    for field, value in out.items():
                        if field == "trace":
                            continue
                        if field == "leak":
                            for dl, v in value.items():
                                rec.setdefault(f"leak_{dl}", []).append(v)
                        else:
                            rec.setdefault(field, []).append(value)
            done += d
        run.log(f"scenario {sc['name']}, {n_end} dismissed flows, violation from flow {starts[sc['name']]}, "
                f"{'one stream' if fixed else 'a stream per draw'}, {draws} draws done")

    rows, saved = [], {}
    for (scenario, rate, name, kind), rec in records.items():
        rec = {field: np.concatenate(v) for field, v in rec.items()}
        flags = rec["flags"]
        for field in ("flags", "alarm_time", "bound", "certified_time", "audits_per_million"):
            saved[f"{scenario}|{rate}|{name}|{kind}|{field}"] = rec[field]
        for d in delays:
            saved[f"{scenario}|{rate}|{name}|{kind}|leak_{d}"] = rec[f"leak_{d}"]
        lower = (flags & FLAGS["miscover_lower"]) > 0
        upper = (flags & FLAGS["miscover_upper"]) > 0
        alarmed = (flags & FLAGS["alarm"]) > 0
        false_alarm = (flags & FLAGS["false_alarm"]) > 0
        start = starts[scenario]
        n_end = lengths[scenario]
        base = {
            "scenario": scenario, "rate": rate, "allocation": name, "kind": kind, "draws": draws,
            "miscover_lower": int(lower.sum()), **_with_prefix("miscover_lower", share_with_interval(int(lower.sum()), draws)),
            "miscover_upper": int(upper.sum()), **_with_prefix("miscover_upper", share_with_interval(int(upper.sum()), draws)),
            "miscover": int((lower | upper).sum()),
            **_with_prefix("miscover", share_with_interval(int((lower | upper).sum()), draws)),
            "alarm_rate": float(alarmed.mean()), "false_alarm_rate": float(false_alarm.mean()),
            "violation_start": start,
            **_with_prefix("delay", _quantiles(rec["alarm_time"] - start, alarmed & (start > 0))),
            **_with_prefix("alarm_audited", _quantiles(rec["alarm_audited"], alarmed)),
            **_with_prefix("leak_at_alarm", _quantiles(rec["leak_at_alarm"], alarmed)),
            **_with_prefix("excess_at_alarm", _quantiles(rec["leak_at_alarm"] - eps * rec["alarm_time"], alarmed)),
            **_with_prefix("bound", _quantiles(rec["bound"], alarmed)),
            "over_bound": int((alarmed & (rec["leak_at_alarm"] > rec["bound"])).sum()),
            "over_bound_rate": float((alarmed & (rec["leak_at_alarm"] > rec["bound"])).sum() / max(alarmed.sum(), 1)),
            "certified_rate": float((rec["certified_time"] >= 0).mean()),
            **_with_prefix("certified_time", _quantiles(rec["certified_time"], rec["certified_time"] >= 0)),
            "certified_share_mean": float(rec["certified_share"].mean()),
            "audits_per_million_mean": float(rec["audits_per_million"].mean()),
        }
        for d in delays:
            leak = rec[f"leak_{d}"]
            tau = np.where(alarmed, np.minimum(rec["alarm_time"] + d, n_end), n_end)
            rows.append({**base, "delay": d, "switched_off_rate": float((alarmed & (tau < n_end)).mean()),
                         **_with_prefix("leak", _quantiles(leak, np.ones(draws, dtype=bool))),
                         **_with_prefix("excess", _quantiles(leak - eps * tau, np.ones(draws, dtype=bool))),
                         "realized_for_mean": float(np.mean(leak / tau))})
    claims = pd.DataFrame(rows)
    run.save_table(claims, "claims")
    run.save_arrays("draws", **saved)
    run.save_arrays("trace", **trace)

    # a kind fails the check only when even the lower end of the interval lies above delta in some cell
    largest, exceeds = {}, {}
    for kind in kinds:
        g = claims[(claims["kind"] == kind) & (claims["delay"] == 0)]
        top = g.loc[g["miscover_rate"].idxmax()]
        largest[kind] = {"scenario": top["scenario"], "rate": float(top["rate"]), "allocation": top["allocation"],
                         "rate_value": float(top["miscover_rate"]), "low": float(top["miscover_low"]),
                         "high": float(top["miscover_high"])}
        exceeds[kind] = bool((g["miscover_low"] > delta).any())
        run.log(f"{kind}, largest share of draws leaving the bounds {top['miscover_rate']:.4f} in scenario "
                f"{top['scenario']} at audit rate {top['rate']}, allocation {top['allocation']}, 95% interval "
                f"{top['miscover_low']:.4f} to {top['miscover_high']:.4f}, delta {delta}, above delta beyond "
                f"simulation noise {exceeds[kind]}")
        for scenario in names:
            r = g[(g["scenario"] == scenario) & (g["rate"] == main_rate) & (g["allocation"] == "uniform")].iloc[0]
            run.log(f"{kind}, scenario {scenario}, main policy, bounds left in {r['miscover_rate']:.4f} of draws, "
                    f"alarm in {r['alarm_rate']:.4f}, false alarm in {r['false_alarm_rate']:.4f}, median delay "
                    f"{r['delay_median']:.0f} flows and {r['alarm_audited_median']:.0f} audited attacks, leakage "
                    f"excess at the alarm median {r['excess_at_alarm_median']:.0f} against the bound "
                    f"{r['bound_median']:.0f}, above the bound in {r['over_bound_rate']:.4f} of alarms, certified "
                    f"by audit in {r['certified_rate']:.4f} of draws after a median of {r['certified_time_median']:.0f} "
                    f"flows, {r['audits_per_million_mean']:.0f} audits per million flows")

    adaptive_rows = []
    main_seq = sequences[(main_rate, "uniform", "thinning")]
    lcb_t, ucb_t = main_seq.tables(main_rate, int(ad["flows"]) + 1)
    for j, cell in enumerate(ad["cells"]):
        rng = np.random.default_rng(seeds[len(scenarios) + j])
        out = simulate_adaptive(rng, cell, eps, main_rate, lcb_t, ucb_t, int(ad["draws"]), int(ad["flows"]))
        n_draws = int(ad["draws"])
        either = out["miss_lower"] | out["miss_upper"]
        alarmed = out["alarmed"]
        row = {
            "cell": cell["name"], "rate": cell["rate"], "evade_below": cell.get("evade_below"),
            "skip_after_audit": bool(cell.get("skip_after_audit", False)), "benign_delay": cell["benign_delay"],
            "attack_delay": cell["attack_delay"], "draws": n_draws, "flows": int(ad["flows"]),
            "miscover": int(either.sum()), **_with_prefix("miscover", share_with_interval(int(either.sum()), n_draws)),
            "miscover_lower_rate": float(out["miss_lower"].mean()),
            "miscover_upper_rate": float(out["miss_upper"].mean()),
            "alarm_rate": float(alarmed.mean()), "false_alarm_rate": float(out["false_alarm"].mean()),
            **_with_prefix("alarm_time", _quantiles(out["alarm_time"], alarmed)),
            **_with_prefix("leak_at_alarm", _quantiles(out["leak_at_alarm"], alarmed)),
            **_with_prefix("leak_in_lag", _quantiles(out["leak_at_alarm"] - out["leak_at_reading"], alarmed)),
            **_with_prefix("bound", _quantiles(out["bound"], alarmed)),
            "over_bound_rate": float((alarmed & (out["leak_at_reading"] > out["bound"])).sum() / max(alarmed.sum(), 1)),
            "attacks_mean": float(out["attacked"].mean()), "audited_attacks_mean": float(out["audited"].mean()),
            "realized_for_mean": float(np.mean(out["count"] / int(ad["flows"]))),
            "lag_at_end_mean": float(out["lag_at_end"].mean()),
        }
        adaptive_rows.append(row)
        run.log(f"adaptive cell {cell['name']}, bounds left in {row['miscover_rate']:.4f} of draws, 95% interval "
                f"{row['miscover_low']:.4f} to {row['miscover_high']:.4f}, alarm in {row['alarm_rate']:.4f} after a "
                f"median of {row['alarm_time_median']:.0f} flows, leakage at the alarm median "
                f"{row['leak_at_alarm_median']:.0f} of which {row['leak_in_lag_median']:.0f} since the reading, "
                f"realized FOR {row['realized_for_mean']:.5f}, mean attacks {row['attacks_mean']:.0f}")
    adaptive = pd.DataFrame(adaptive_rows)
    run.save_table(adaptive, "adaptive")
    for kind in ["thinning"]:
        exceeds[f"adaptive_{kind}"] = bool((adaptive["miscover_low"] > delta).any()) if len(adaptive) else False

    run.save_json({
        "epsilon": eps, "delta": delta, "delta_lower": delta_lower, "q_min": q_min,
        "lower_ratios": m["lower_ratios"], "upper_ratios": m["upper_ratios"], "kinds": kinds, "rates": rates,
        "main_rate": main_rate, "allocations": allocations, "delays": delays, "bins": bins,
        "bin_ratio": st["bin_ratio"], "scenarios": names, "fixed": st.get("fixed", []), "trace": st["trace"],
        "violation_start": starts, "stream_length": lengths, "draws": draws, "adaptive": ad,
        "largest_miscoverage": largest, "miscoverage_exceeds_delta": exceeds,
    }, "summary")

    for name, draw in FIGURES.items():
        save_figure(draw(run.dir), run.figures, name)
        run.log(f"saved figure {name}")


def _rate_colors(rates) -> dict:
    # one shade per audit rate, darker for more audits
    order = sorted(rates)
    shades = np.linspace(0, len(style.SEQUENTIAL) - 1, len(order)).round().astype(int)
    return {rate: style.SEQUENTIAL[s] for rate, s in zip(order, shades)}


def _constant_scenarios(scenarios: pd.DataFrame) -> list[str]:
    # scenarios of one segment above epsilon, from the smallest violation to the largest
    g = scenarios.groupby("scenario", sort=False)
    one = [(name, float(t["rate"].iloc[0])) for name, t in g if len(t) == 1 and t["rate"].iloc[0] > 1]
    return [name for name, _ in sorted(one, key=lambda x: x[1])]


def _percent_rate(v, _pos=None) -> str:
    return f"{v * 100:g}%"


def fig_coverage_by_scenario(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    delta, names = summary["delta"], summary["scenarios"]
    data = claims[(claims["delay"] == 0) & (claims["allocation"] == "uniform")]
    colors = _rate_colors(summary["rates"])
    fig, axes = plt.subplots(1, len(summary["kinds"]), figsize=(style.WIDTH_DOUBLE, 2.6), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(names))
    for ax, kind, letter in zip(axes, summary["kinds"], "abcdef"):
        g = data[data["kind"] == kind]
        for k, rate in enumerate(sorted(summary["rates"])):
            h = g[g["rate"] == rate].set_index("scenario").reindex(names)
            layout.points(ax, layout.dodge(x, k, len(summary["rates"]), 0.16), h["miscover_rate"], h["miscover_low"],
                          h["miscover_high"], color=colors[rate], marker="o", label=f"audit rate {rate * 100:g}%")
        ax.axhline(delta, color=style.GRAY, linestyle="--", linewidth=1, label=r"$\delta$")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_xlabel("scenario")
        layout.panel_letter(ax, letter)
    axes[0].set_ylim(0, max(2 * delta, 1.15 * data["miscover_high"].max()))
    axes[0].set_ylabel("draws leaving the bounds")
    layout.legend_below(fig, axes[0], [f"audit rate {r * 100:g}%" for r in sorted(summary["rates"])])
    return fig


def fig_delay_by_audit_rate(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    scenarios = runs.load_table(run_dir, "scenarios")
    summary = runs.load_json(run_dir, "summary")
    names = _constant_scenarios(scenarios)
    sizes = {n: float(scenarios.loc[scenarios["scenario"] == n, "rate"].iloc[0]) for n in names}
    shades = np.linspace(2, len(style.SEQUENTIAL) - 1, max(len(names), 1)).round().astype(int)
    colors = {n: style.SEQUENTIAL[s] for n, s in zip(names, shades)}
    data = claims[claims["delay"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_DOUBLE, 2.6), sharey=True)
    # a, uniform audits, the two kinds of sequence. b, the thinning kind under the three allocations
    for name in names:
        label = f"FOR {sizes[name]:g}" + r"$\varepsilon$"
        for kind in summary["kinds"]:
            g = data[(data["scenario"] == name) & (data["kind"] == kind) & (data["allocation"] == "uniform")]
            g = g.sort_values("rate")
            axes[0].plot(g["rate"], g["delay_median"], color=colors[name], linestyle=KIND_DASHES[kind], marker="o",
                         markersize=3, label=label if kind == "thinning" else None)
        for allocation in summary["allocations"]:
            g = data[(data["scenario"] == name) & (data["kind"] == "thinning") & (data["allocation"] == allocation)]
            g = g.sort_values("rate")
            axes[1].plot(g["rate"], g["delay_median"], color=colors[name],
                         linestyle=ALLOCATION_DASHES.get(allocation, "-"), marker="o", markersize=3,
                         label=label if allocation == "uniform" else None)
    for ax, letter in zip(axes, "ab"):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(_percent_rate)
        ax.set_xlabel("audit rate")
        layout.panel_letter(ax, letter)
    axes[0].set_ylabel("median dismissed flows to the alarm")
    extra = [(Line2D([], [], color=style.INK, linestyle=KIND_DASHES[k], linewidth=1), KIND_LABELS[k])
             for k in summary["kinds"]]
    extra += [(Line2D([], [], color=style.INK, linestyle=ALLOCATION_DASHES.get(a, "-"), linewidth=1),
               f"{a} audits") for a in summary["allocations"] if a != "uniform"]
    layout.legend_below(fig, axes[0], [f"FOR {sizes[n]:g}" + r"$\varepsilon$" for n in names], extra)
    return fig


def fig_leakage_vs_bound(run_dir) -> plt.Figure:
    style.use()
    claims = runs.load_table(run_dir, "claims")
    summary = runs.load_json(run_dir, "summary")
    names = [n for n in summary["scenarios"] if summary["violation_start"][n] > 0]
    data = claims[(claims["delay"] == 0) & (claims["allocation"] == "uniform") & (claims["rate"] == summary["main_rate"])]
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.6))
    x = np.arange(len(names))
    colors = {"thinning": style.color_for("forge_simplex"), "freedman": style.GRAY}
    kinds = summary["kinds"]
    for k, kind in enumerate(kinds):
        g = data[data["kind"] == kind].set_index("scenario").reindex(names)
        xs = layout.dodge(x, k, len(kinds), 0.2)
        layout.points(ax, xs, g["excess_at_alarm_median"].clip(lower=1), g["excess_at_alarm_low"].clip(lower=1),
                      g["excess_at_alarm_high"].clip(lower=1), color=colors[kind], marker="o",
                      label=f"excess at the alarm, {KIND_LABELS[kind]}")
        ax.plot(xs, g["bound_median"].clip(lower=1), color=colors[kind], marker="s", markerfacecolor="white",
                markersize=4, linestyle="none", label=f"bound of Theorem 3, {KIND_LABELS[kind]}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("scenario")
    ax.set_ylabel(r"attacks above $\varepsilon N$ at the switch off")
    layout.legend_below(fig, ax, [f"excess at the alarm, {KIND_LABELS[k]}" for k in kinds], ncol=2)
    return fig


def fig_bounds_over_time(run_dir) -> plt.Figure:
    style.use()
    t = runs.load_arrays(run_dir, "trace")
    summary = runs.load_json(run_dir, "summary")
    eps = summary["epsilon"]
    valid = t["valid"]
    pos, g, lcb, ucb = t["ev_pos"][valid], t["ev_g"][valid], t["lcb"][valid], t["ucb"][valid]
    n_end = int(t["n_end"])
    attacks = t["attacks"]
    color = style.color_for("forge_simplex")
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.6))
    # the true count rises by one at every attack, the bounds and the estimate move at the audited attacks only
    ax.step(np.r_[0, attacks, n_end], np.r_[0, np.arange(1, len(attacks) + 1), len(attacks)], where="post",
            color=style.INK, linewidth=1, label="attacks among the dismissed flows")
    ax.step(np.r_[pos, n_end], np.r_[np.arange(len(pos)) / float(t["rate"]), (len(pos) - 1) / float(t["rate"])],
            where="post", color=style.GRAY, linestyle="--", linewidth=1, label="estimate from the audits")
    ax.step(np.r_[pos, n_end], np.r_[lcb, lcb[-1]], where="post", color=color, linewidth=1.2, label="lower bound")
    ax.step(np.r_[pos, n_end], np.r_[ucb, ucb[-1]], where="post", color=color, linestyle="--", linewidth=1.2,
            label="upper bound")
    ax.plot([0, n_end], [0, eps * n_end], color=style.GRAY, linestyle=":", linewidth=1,
            label=r"$\varepsilon$ times the dismissed flows")
    if int(t["violation_start"]) > 0:
        ax.axvline(int(t["violation_start"]), color=style.GRAY, linestyle="-.", linewidth=0.8)
    if int(t["alarm_time"]) >= 0:
        ax.axvline(int(t["alarm_time"]), color=color, linestyle="-.", linewidth=0.8)
    ax.set_xlim(0, n_end)
    ax.set_ylim(0, 1.1 * max(float(ucb.max()), float(len(attacks))))
    ax.set_xlabel("dismissed flows")
    ax.set_ylabel("attacks")
    layout.legend_below(fig, ax, ["attacks among the dismissed flows", "estimate from the audits", "lower bound",
                                  "upper bound"], ncol=2)
    return fig


FIGURES = {
    "coverage_by_scenario": fig_coverage_by_scenario,
    "delay_by_audit_rate": fig_delay_by_audit_rate,
    "leakage_vs_bound": fig_leakage_vs_bound,
    "bounds_over_time": fig_bounds_over_time,
}
