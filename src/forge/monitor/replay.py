"""the replayed stream of a deployment window, the real sessions of a block in their real order thinned to a
prevalence, dismissed at a threshold with their real labels and audited with a known probability.

a window of prevalence pi drawn from a block keeps every benign row of the block in row order and every attack row
with probability n_b pi / ((1 - pi) n_a) independently, so the kept attacks are binomial in number as the attacks of
a simulated window are, carry the block's own mix of attack types in expectation, and keep the real density of the
attacks along the block, a burst in the block a burst in the window. a window short of attacks keeps all of them and
keeps each benign row with the probability that gives the prevalence instead, the rule of the real windows of e12.
the kept rows with a score at or below the threshold of a draw are the dismissed stream of the window in row order,
the audit coins are drawn over its attacks, and the audited attacks are the events the monitor reads, each with the
count of dismissed attacks up to it. so the coin and the thinning are the only simulations left, against the
simulated stream of e21, where every dismissed session is an attack with the block's population FOR independently.

the thinning is drawn with geometric gaps over the rows, so its cost is the number of kept rows and not the size of
the block, and the draws are thinned a chunk at a time, the gaps of a chunk drawn together, so the chunk is fixed by
the config and not by the memory of the machine.
"""

from typing import NamedTuple

import numpy as np


class Block(NamedTuple):
    # the rows of one block or test part in their real order, split by class. a_pos is the row position of each
    # attack row in the block, a_bin and b_bin the index of the first grid threshold at or above the score, the
    # length of the grid where the score lies above every threshold
    a_score: np.ndarray
    a_pos: np.ndarray
    a_bin: np.ndarray
    b_score: np.ndarray
    b_bin: np.ndarray


class Thinned(NamedTuple):
    # the kept rows of a chunk of draws, indices into the attack rows and into the benign rows of the block padded on
    # the right, with the masks of the entries that exist. b_idx is none when every benign row is kept
    a_idx: np.ndarray
    a_valid: np.ndarray
    b_idx: np.ndarray | None
    b_valid: np.ndarray | None
    n_attacks: np.ndarray
    n_benign: np.ndarray


def block_rows(score, code, grid) -> Block:
    # code 0 is a benign row, 1 and up an attack of a family. a row with a score at or below grid[t] has a bin at
    # most t, so the counts at the grid follow from the bins
    score = np.asarray(score, dtype=np.float32)
    attack = np.asarray(code) > 0
    bins = np.searchsorted(np.asarray(grid), score, side="left").astype(np.int32)
    return Block(score[attack], np.flatnonzero(attack).astype(np.int64), bins[attack], score[~attack], bins[~attack])


def thin_rates(n_benign: int, n_attack: int, n_window: int, pi: float, subsampled: bool) -> tuple[float, float]:
    # the probability that keeps an attack row and the one that keeps a benign row, for a window of n_window
    # sessions at prevalence pi. a window short of attacks keeps all of them and thins its benign rows instead
    if subsampled:
        return 1.0, min(1.0, n_window * (1 - pi) / n_benign) if n_benign > 0 else 1.0
    return (min(1.0, n_window * pi / n_attack) if n_attack > 0 else 1.0), 1.0


def kept_rows(rng, n, p: float, draws: int) -> tuple[np.ndarray, np.ndarray]:
    """indices of the rows a Bernoulli thinning with probability p keeps out of n, for independent draws, from
    geometric gaps, shape (draws, m) padded on the right, with the mask of the entries that exist. n may be one
    count or one per draw, and p of 1 keeps every row."""
    n = np.broadcast_to(np.asarray(n, dtype=np.int64), (draws,))
    top = int(n.max()) if draws else 0
    if top == 0 or p <= 0:
        return np.zeros((draws, 0), dtype=np.int64), np.zeros((draws, 0), dtype=bool)
    if p >= 1:
        idx = np.broadcast_to(np.arange(top, dtype=np.int64), (draws, top)).copy()
        return idx, idx < n[:, None]
    expected = top * p
    m = int(np.ceil(expected + 10 * np.sqrt(expected) + 20))
    pos = np.cumsum(rng.geometric(p, size=(draws, m)), axis=1)
    # a row whose gaps stop before its last row would lose kept rows, so it gets more gaps
    while (pos[:, -1] <= n).any():
        more = np.cumsum(rng.geometric(p, size=(draws, m)), axis=1)
        pos = np.concatenate([pos, pos[:, -1:] + more], axis=1)
    return pos - 1, pos <= n[:, None]


def thin(rng, block: Block, p_attack: float, p_benign: float, draws: int) -> Thinned:
    # the attack rows are thinned first, then the benign rows when they are thinned at all
    a_idx, a_valid = kept_rows(rng, len(block.a_score), p_attack, draws)
    if p_benign >= 1:
        b_idx, b_valid, n_benign = None, None, np.full(draws, len(block.b_score), dtype=np.int64)
    else:
        b_idx, b_valid = kept_rows(rng, len(block.b_score), p_benign, draws)
        n_benign = b_valid.sum(1)
    return Thinned(a_idx, a_valid, b_idx, b_valid, a_valid.sum(1), n_benign)


def counts_at_grid(bins, valid, g: int) -> np.ndarray:
    # counts at or below each of the g grid thresholds of the rows kept in each draw, from their bins
    d, m = bins.shape
    flat = (np.asarray(bins, dtype=np.int64) + np.arange(d)[:, None] * (g + 1))[valid]
    counts = np.bincount(flat, minlength=d * (g + 1)).reshape(d, g + 1)
    return np.cumsum(counts[:, :g], axis=1)


def window_counts(block: Block, thinned: Thinned, g: int) -> tuple[np.ndarray, np.ndarray]:
    # counts at the grid of the kept attacks and of the kept benign rows of every draw, shape (draws, g) each
    a_safe = np.minimum(thinned.a_idx, max(len(block.a_score) - 1, 0))
    attack = counts_at_grid(block.a_bin[a_safe], thinned.a_valid, g)
    if thinned.b_idx is None:
        benign = np.broadcast_to(np.cumsum(np.bincount(block.b_bin, minlength=g + 1)[:g]), attack.shape)
    else:
        b_safe = np.minimum(thinned.b_idx, max(len(block.b_score) - 1, 0))
        benign = counts_at_grid(block.b_bin[b_safe], thinned.b_valid, g)
    return attack, benign


def _benign_before(block: Block, thinned: Thinned, index, row, j):
    # the dismissed benign rows before each dismissed attack, and the dismissed benign rows of every draw. the
    # benign rows before the attack row of index j number a_pos[j] - j, and those kept and dismissed among them are
    # counted from a cumulative sum, over every benign row per threshold when every benign row is kept, over the
    # kept benign rows of each draw otherwise
    draws = len(index)
    before = np.asarray(block.a_pos[j] - j, dtype=np.int64)
    out = np.zeros(len(j), dtype=np.int64)
    if thinned.b_idx is None:
        n_dismissed = np.zeros(draws, dtype=np.int64)
        for t in np.unique(index):
            cumulative = np.concatenate([[0], np.cumsum(block.b_bin <= t)])
            n_dismissed[index == t] = cumulative[-1]
            sel = index[row] == t
            out[sel] = cumulative[before[sel]]
        return out, n_dismissed
    d, kb = thinned.b_idx.shape
    b_safe = np.minimum(thinned.b_idx, max(len(block.b_score) - 1, 0))
    dismissed = thinned.b_valid & (block.b_bin[b_safe] <= index[:, None])
    cumulative = np.concatenate([np.zeros((d, 1), dtype=np.int64), np.cumsum(dismissed, axis=1)], axis=1)
    # the kept benign rows before the attack, found in the kept indices of its draw, which rise along the row
    big = len(block.b_score) + 1
    keys = np.where(thinned.b_valid, b_safe, big) + np.arange(d)[:, None] * (big + 1)
    kept_before = np.searchsorted(keys.ravel(), before + row * (big + 1), side="left") - row * kb
    out = cumulative[row, kept_before]
    return out, cumulative[:, -1]


def dismissed_stream(rng, block: Block, thinned: Thinned, index, q: float, span: int) -> dict:
    """the dismissed stream of a window under the thresholds of a chunk of draws, with its audits.

    index holds the grid index of the threshold of each draw, -1 dismissing nothing, and a kept row is dismissed when
    its bin is at most that index. every dismissed attack is audited with probability q, drawn as geometric gaps over
    the dismissed attacks in row order, and the audited benign sessions are binomial in the rest. returns per draw the
    dismissed sessions, the attacks among them, the audited attacks and the audited benign sessions, the largest
    number of attacks among any span consecutive dismissed sessions, and the events, the draw of each audited attack,
    its one based position in the window's dismissed stream and the count of dismissed attacks up to it.
    """
    index = np.asarray(index, dtype=np.int64)
    draws, ka = thinned.a_idx.shape
    a_safe = np.minimum(thinned.a_idx, max(len(block.a_score) - 1, 0))
    dismissed = thinned.a_valid & (block.a_bin[a_safe] <= index[:, None])
    flat = np.flatnonzero(dismissed.ravel())
    row = flat // max(ka, 1)
    j = a_safe.ravel()[flat]
    n_attacks = np.bincount(row, minlength=draws)
    start = np.concatenate([[0], np.cumsum(n_attacks)[:-1]])
    rank = np.arange(len(flat), dtype=np.int64) - start[row] + 1
    benign_before, n_benign = _benign_before(block, thinned, index, row, j)
    pos = benign_before + rank
    # the audits over the dismissed attacks of each draw, the rank of each audited one
    k_rank, k_valid = kept_rows(rng, n_attacks, q, draws)
    ev_row = np.nonzero(k_valid)[0]
    ev_rank = (k_rank + 1)[k_valid]
    ev_pos = pos[start[ev_row] + ev_rank - 1]
    audited_benign = rng.binomial(n_benign, q)
    # the attacks among the span sessions from each dismissed attack on, the positions rising within a draw
    burst = np.zeros(draws, dtype=np.int64)
    if len(flat):
        big = int(pos.max()) + span + 1
        keys = pos + row * big
        count = np.searchsorted(keys, keys + span, side="left") - np.arange(len(keys))
        live = n_attacks > 0
        burst[live] = np.maximum.reduceat(count, start[live])
    return {"dismissed": n_benign + n_attacks, "attacks": n_attacks, "audited_attacks": k_valid.sum(1),
            "audited_benign": audited_benign, "burst": burst, "ev_row": ev_row, "ev_pos": ev_pos,
            "ev_count": ev_rank}


def assemble_stream(draws: int, parts: list, dismissed, attacks, audited_attacks, audited_benign) -> dict:
    """the events of a whole stream in the layout of monitor_streams.audited_stream, from the events of its windows.

    parts holds one entry per window and chunk, the draw of each event, its position in the whole stream and the
    count of dismissed attacks since the start of the stream, with the index of the window. dismissed, attacks,
    audited_attacks and audited_benign have shape (windows, draws). the events of a draw come in the order of the
    stream after a virtual first event at position 0, padded on the right.
    """
    dismissed = np.asarray(dismissed, dtype=np.int64)
    offsets = np.concatenate([np.zeros((1, draws), dtype=np.int64), np.cumsum(dismissed, axis=0)])
    rows = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    pos = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    count = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    window = np.concatenate([np.full(len(p[0]), p[3], dtype=np.int64) for p in parts]) if parts else \
        np.zeros(0, dtype=np.int64)
    order = np.lexsort((pos, rows))
    rows, pos, count, window = rows[order], pos[order], count[order], window[order]
    per_row = np.bincount(rows, minlength=draws)
    width = 1 + int(per_row.max()) if draws else 1
    ev_pos = np.full((draws, width), -1, dtype=np.int64)
    ev_count = np.zeros((draws, width), dtype=np.int64)
    ev_window = np.full((draws, width), -1, dtype=np.int64)
    valid = np.zeros((draws, width), dtype=bool)
    ev_pos[:, 0], valid[:, 0] = 0, True
    first = np.concatenate([[0], np.cumsum(per_row)[:-1]])
    col = 1 + np.arange(len(rows)) - first[rows]
    ev_pos[rows, col], ev_count[rows, col], ev_window[rows, col], valid[rows, col] = pos, count, window, True
    return {"ev_pos": ev_pos, "ev_count": ev_count, "ev_window": ev_window, "valid": valid,
            "attacks": np.asarray(attacks, dtype=np.int64),
            "audited_attacks": np.asarray(audited_attacks, dtype=np.int64),
            "audited_benign": np.asarray(audited_benign, dtype=np.int64), "offsets": offsets}


def check_block(block: Block, grid, benign_below, family_below, sizes, code) -> bool:
    # whether the counts of the block's rows at the grid equal the saved histograms of its window, class by class
    g = len(grid)
    benign = np.cumsum(np.bincount(block.b_bin, minlength=g + 1)[:g])
    if len(block.b_score) != sizes[0] or not np.array_equal(benign, benign_below):
        return False
    a_code = np.asarray(code)[block.a_pos]
    for k in range(1, len(sizes)):
        mine = a_code == k
        family = np.cumsum(np.bincount(block.a_bin[mine], minlength=g + 1)[:g])
        if mine.sum() != sizes[k] or not np.array_equal(family, family_below[k - 1]):
            return False
    return True
