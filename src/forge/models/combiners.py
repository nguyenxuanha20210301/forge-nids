"""the combiners of the scorer comparison, each a scorer built from the logits of the base scorers on rows the base
scorers never trained on.

S-mean averages the probabilities, S-stack is a logistic regression on the standardized logits, S-gate weights the
logits of every row with a softmax gate read off the same logits, S-tail is a linear combiner trained on the low tail
objective of proposal 6.2, which pushes the attacks of the hardest family above the benign quantile the certificate
would dismiss, and S-2D takes the larger of two benign cdf ranks, of the best supervised score and of the novelty
score, so a flow is dismissed only when both are low. S-best is the base scorer itself and needs no code here.

the gate and the tail combiner are small enough to train with plain numpy and Adam, on batches drawn with the config
seed, so they run alike on Colab and in the tests.
"""

import numpy as np

COMBINERS = ("s_best", "s_mean", "s_stack", "s_gate", "s_tail", "s_2d")
LABELS = {"s_best": "S-best", "s_mean": "S-mean", "s_stack": "S-stack", "s_gate": "S-gate", "s_tail": "S-tail",
          "s_2d": "S-2D"}
# combiners whose score is a logit of an attack probability
PROBABILITY = ("s_best", "s_mean", "s_stack", "s_gate", "s_tail")


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def softplus(z):
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))


class Standardizer:
    # the base logits centered and scaled on the fitting rows, so every combiner starts from comparable inputs

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, z: np.ndarray) -> "Standardizer":
        self.mean = z.mean(0, dtype=np.float64)
        scale = z.std(0, dtype=np.float64)
        self.scale = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        return ((np.asarray(z, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def mean_logit(z: np.ndarray) -> np.ndarray:
    # the logit of the mean attack probability of the base scorers
    p = sigmoid(z).mean(1)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def fit_stack(z: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=cfg["c"], max_iter=cfg["max_iter"]).fit(z, y)
    return {"w": model.coef_[0].astype(np.float64), "b": float(model.intercept_[0])}


def score_linear(params: dict, z: np.ndarray) -> np.ndarray:
    return (np.asarray(z, dtype=np.float64) @ params["w"] + params["b"]).astype(np.float32)


class Adam:
    # the optimizer of the gate and the tail combiner, one state per parameter array

    def __init__(self, params: dict, learning_rate: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.m = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g * g
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] = self.params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def gate_forward(params: dict, z: np.ndarray):
    # the softmax gate over the base scorers, read off the standardized logits of the row, and the gated logit
    a = z @ params["W"] + params["c"]
    a = a - a.max(1, keepdims=True)
    g = np.exp(a)
    g /= g.sum(1, keepdims=True)
    return g, (g * z).sum(1) + params["b"]


def score_gate(params: dict, z: np.ndarray) -> np.ndarray:
    out = np.empty(len(z), dtype=np.float32)
    for lo in range(0, len(z), 1_000_000):
        out[lo:lo + 1_000_000] = gate_forward(params, np.asarray(z[lo:lo + 1_000_000], dtype=np.float64))[1]
    return out


def gate_loss(params: dict, z: np.ndarray, y: np.ndarray):
    # the binary cross entropy of the gated logit on a batch and its gradient
    g, s = gate_forward(params, z)
    p = sigmoid(s)
    loss = float(np.mean(softplus(s) - y * s))
    ds = (p - y) / len(y)
    dg = ds[:, None] * z
    da = g * (dg - (g * dg).sum(1, keepdims=True))
    return loss, {"W": z.T @ da, "c": da.sum(0), "b": float(ds.sum())}


def fit_gate(z: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> dict:
    # starts at the uniform gate, which is the mean of the standardized logits
    k = z.shape[1]
    params = {"W": np.zeros((k, k)), "c": np.zeros(k), "b": np.array(0.0)}
    rng = np.random.default_rng(seed)
    optimizer = Adam(params, cfg["learning_rate"])
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    trace = []
    for step in range(cfg["steps"]):
        rows = rng.integers(0, len(z), cfg["batch"])
        loss, grads = gate_loss(params, z[rows], y[rows])
        optimizer.step(grads)
        trace.append(loss)
    return {**params, "trace": trace}


def tail_loss(params: dict, z: np.ndarray, y: np.ndarray, family: np.ndarray, benign_share: float,
              temperature: float, family_temperature: float, weight: float):
    """the low tail objective on one batch and its gradient.

    the threshold is the score of the benign row at the benign_share quantile of the batch's benign scores, the
    one the certificate would dismiss up to. every attack scored at or below it is a miss, counted softly as
    softplus((t - s) / temperature) and averaged within each family, and the families are combined by a soft maximum
    at family_temperature, so the hardest family in the batch gets most of the gradient. weight times the binary cross
    entropy keeps the combiner a scorer of the attack label as well.
    """
    w, b = params["w"], params["b"]
    s = z @ w + b
    benign = np.flatnonzero(family == 0)
    order = benign[np.argsort(s[benign], kind="stable")]
    pivot = order[min(int(np.ceil(benign_share * len(benign))) - 1, len(benign) - 1)] if len(benign) else None
    ds = np.zeros(len(s))
    tail = 0.0
    if pivot is not None:
        t = s[pivot]
        codes = [k for k in np.unique(family) if k != 0]
        losses, grads = [], []
        for k in codes:
            rows = np.flatnonzero(family == k)
            u = (t - s[rows]) / temperature
            losses.append(float(np.mean(softplus(u))))
            grads.append((rows, sigmoid(u) / (temperature * len(rows))))
        if codes:
            losses = np.array(losses)
            scaled = losses / family_temperature
            alpha = np.exp(scaled - scaled.max())
            alpha /= alpha.sum()
            tail = float(family_temperature * (np.log(np.exp(scaled - scaled.max()).sum()) + scaled.max()))
            for a, (rows, g) in zip(alpha, grads):
                ds[rows] -= a * g
                ds[pivot] += a * g.sum()
    p = sigmoid(s)
    bce = float(np.mean(softplus(s) - y * s))
    ds_bce = weight * (p - y) / len(y)
    ds_total = ds + ds_bce
    return tail + weight * bce, {"w": z.T @ ds_total, "b": float(ds_bce.sum())}, {"tail": tail, "bce": bce}


def fit_tail(z: np.ndarray, y: np.ndarray, family: np.ndarray, cfg: dict, seed: int, benign_share: float) -> dict:
    # starts at the mean of the standardized logits. every batch holds benign rows and rows of every family, drawn
    # with replacement where a family is small, so the quantile and every family's tail are in each batch
    k = z.shape[1]
    params = {"w": np.full(k, 1.0 / k), "b": np.array(0.0)}
    rng = np.random.default_rng(seed)
    optimizer = Adam(params, cfg["learning_rate"])
    z, y, family = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(family)
    pools = {k: np.flatnonzero(family == k) for k in np.unique(family)}
    trace = []
    for step in range(cfg["steps"]):
        rows = np.concatenate([rng.choice(pool, cfg["batch_benign"] if k == 0 else cfg["batch_family"],
                                          replace=len(pool) < (cfg["batch_benign"] if k == 0 else cfg["batch_family"]))
                               for k, pool in pools.items()])
        loss, grads, parts = tail_loss(params, z[rows], y[rows], family[rows], benign_share, cfg["temperature"],
                                       cfg["family_temperature"], cfg["weight"])
        optimizer.step(grads)
        trace.append(parts["tail"])
    return {**params, "trace": trace}


def benign_rank(sorted_benign: np.ndarray, x: np.ndarray) -> np.ndarray:
    # the empirical cdf of the benign fitting scores at x, the share of them at or below x
    return (np.searchsorted(sorted_benign, x, side="right") / len(sorted_benign)).astype(np.float32)


def two_d(best_rank: np.ndarray, novelty_rank: np.ndarray) -> np.ndarray:
    # low only when the supervised score and the novelty score are both low
    return np.maximum(best_rank, novelty_rank)
