"""the operating rule of FORGE-Monitor on the bounds of a sequence.

the lower bound moves only when an audited attack arrives, so the monitor reads the bounds at those moments and in
between counts the dismissed flows. the state is certified while the upper bound stays at or below epsilon times the
dismissed count, violation once the lower bound exceeds it, and running on C1 otherwise. at a violation the dismissal
is switched off, so nothing more leaks, and a new certificate starts a new sequence. the alarm is read from 1 / epsilon
dismissed flows on, since before that one audited attack alone puts the lower bound above epsilon times the count. a
label that arrives late holds the reading back, the sequence is read at the last flow before the earliest audit still
waiting for its label, which keeps every bound valid under any delay at the price of what leaks until the reading
catches up.

the functions below take the events of many draws at once, one row per draw, the events padded on the right, and
return one value per draw.
"""

import numpy as np

STATES = ("certified", "running", "violation")


def alarm_floor(eps: float) -> int:
    # the dismissed count from which the rule reads the alarm, 1 / eps rounded up
    return int(np.ceil(1 / eps))


def first_alarm(lcb, positions, valid, eps, start: int = 0):
    """index of the first event from which the lower bound exceeds eps times the dismissed count, -1 when there is none.

    the alarm is read from start flows on, so an event before start fires at start if its bound exceeds eps times
    start, and an event from start on fires at its own position.
    """
    hit = valid & (lcb > eps * np.maximum(positions, start))
    return np.where(hit.any(-1), hit.argmax(-1), -1)


def certified_stretches(ucb, positions, next_positions, valid, eps, stop=None):
    """for each event, the first flow from which the state is certified until the next event, and the length of that
    stretch, both 0 where the upper bound does not fall to eps times the dismissed count before the next event.

    the bound is constant between two events while the dismissed count grows, so the state turns certified at the
    flow where eps times the count reaches the bound, if that comes before the next event. stop ends every stretch at
    the switch off of each draw.
    """
    start = np.maximum(positions, np.ceil(ucb / eps)).astype(np.int64)
    end = next_positions if stop is None else np.minimum(next_positions, np.asarray(stop)[..., None])
    length = np.where(valid, np.maximum(end - start, 0), 0)
    return np.where(length > 0, start, 0), length


def state(lcb, ucb, n, eps, start: int = 0):
    # the state read from both bounds at n dismissed flows, the alarm read from start flows on
    n = np.asarray(n)
    out = np.full(np.shape(n), "running", dtype=object)
    out[np.asarray(ucb) <= eps * n] = "certified"
    out[(np.asarray(lcb) > eps * n) & (n >= start)] = "violation"
    return out


def reading(n, benign_delay, attack_delay, pending):
    """the flow up to which the sequence can be read at time n.

    pending is the position of the earliest audited attack whose label has not arrived, or n plus one when there is
    none. the labels of audited benign flows arrive benign_delay flows after the audit, so the sequence is read at the
    last flow before the earliest pending audit of either kind, and attack_delay may not be shorter than benign_delay
    for the count of audited attacks at the reading to be the count whose labels arrived.
    """
    if attack_delay < benign_delay:
        raise ValueError("the attack delay may not be shorter than the benign one")
    return np.maximum(np.minimum(np.asarray(n) - benign_delay, np.asarray(pending) - 1), 0)
