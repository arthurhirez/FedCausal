"""Stage 4 -- quantise the trajectory, and read the chain rather than the histogram.

Idea 2, with the honest part kept in front: for an ergodic chain estimated
from one trajectory the stationary distribution IS the empirical occupancy,
so computing `pi` by simulation is a histogram with extra steps and it
discards exactly the dynamics quantisation was for. The dynamics live in the
transition matrix; the cross-client statement lives in the joint chain; and
the fitted chains are worth keeping mainly because they make a much better
NULL than a circular shift.

Torch-free and package-free, like `transfer`: it takes symbol tables.

The codebook
------------
States must be comparable across clients or no cross-client statement is
defined, so the codebook is FIT ON THE INIT PHASE AND FROZEN -- the same
frozen-ruler logic as the encoder. `space="shape"` drops the level features,
which is mandatory under a per-client scaler (`minmax_ref`), where the level
is in each client's own units and clustering on it clusters CLIENTS rather
than regimes; `space="all"` is legitimate only under a shared scaler.

    s_c[b] = argmin_j || phi_c[b] - mu_j ||

Estimation budget, stated with every number
-------------------------------------------
Plug-in mutual information over N samples at alphabet size k is biased
upward by roughly df / 2N nats, with df = (k-1)^2 for I(s_c ; s_d) and
df = k (k-1)^2 for the transfer entropy

    T_{c->d} = I(s_d[b] ; s_c[b-1] | s_d[b-1]).

`mi` and `transfer_entropy` return that bias next to the estimate and apply
the Miller-Madow correction; nothing below the bias should be read as a
finding. Pooling the calendar classes of a block multiplies N by the number
of classes and fixes ESTIMATION -- it does not fix INFERENCE, because
classes inside a month are not independent. That is what
`month_block_null` is for: it permutes whole months with all their classes
attached, where `chain_null` resamples from each client's own fitted chain.
Running both and reporting the gap is the honest version.

Tests that can fail informatively
---------------------------------
``markov_order_test``
    Likelihood ratio, order 0 vs 1 and 1 vs 2. If order 0 is not rejected
    there are no dynamics to read and the chain adds nothing over the
    occupancy histogram -- which is this idea's real failure mode, so it is
    tested before anything is built on it.

``homogeneity_test`` / ``stationarity_split``
    The same LR statistic applied to two halves of a stretch. Init should
    PASS (one regime, one chain) and transition should FAIL (the regime is
    moving). That single contrast carries most of the drift claim, and it
    needs no reconstruction quality at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["fit_codebook", "codebook_stability", "assign", "symbol_table",
           "sequences",
           "transition_matrix", "stationary", "entropy_rate", "occupancy",
           "js_divergence", "occupancy_matrix", "markov_order_test",
           "homogeneity_test", "stationarity_split", "mi", "transfer_entropy",
           "pair_table", "chain_null", "month_block_null", "MI_DF", "TE_DF"]


def MI_DF(k: int) -> int:
    return (int(k) - 1) ** 2


def TE_DF(k: int) -> int:
    return int(k) * (int(k) - 1) ** 2


# --------------------------------------------------------------------------
# codebook
# --------------------------------------------------------------------------
def _space_cols(phi: pd.DataFrame, space: str) -> list[str]:
    skip = {"client", "bid", "phase", "source", "cycle_pos", "gid", "month"}
    cols = [c for c in phi.columns
            if c not in skip and pd.api.types.is_numeric_dtype(phi[c])]
    if space == "shape":
        cols = [c for c in cols if not c.startswith("lvl")]
    return cols


def fit_codebook(phi: pd.DataFrame, k: int = 4, space: str = "shape",
                 fit_phase: str = "init", seed: int = 0) -> dict:
    """K-means on `fit_phase` rows only; the centres are then frozen."""
    from sklearn.cluster import KMeans
    cols = _space_cols(phi, space)
    fit = phi[phi["phase"] == fit_phase] if "phase" in phi else phi
    if len(fit) < k:
        fit = phi
    X = fit[cols].to_numpy(float)
    X = X[np.isfinite(X).all(axis=1)]
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed)).fit(X)
    order = np.argsort(km.cluster_centers_[:, 0])       # stable state naming
    centres = km.cluster_centers_[order]
    inertia = float(km.inertia_)
    tot = float(((X - X.mean(axis=0)) ** 2).sum())
    return {"centres": centres, "cols": cols, "k": int(k), "space": space,
            "fit_phase": fit_phase, "seed": int(seed),
            "distortion": 1 - inertia / tot if tot > 0 else np.nan,
            "n_fit": int(len(X))}


def codebook_stability(phi: pd.DataFrame, k: int = 4, space: str = "shape",
                       fit_phase: str = "init", n_rep: int = 10,
                       frac: float = 0.5, seed: int = 0) -> dict:
    """Refit the codebook on random halves; ARI of the labellings against the
    full fit. A low ARI means `k` is too large or the space is not clustered,
    and everything downstream is codebook noise rather than dynamics."""
    from sklearn.metrics import adjusted_rand_score
    full = fit_codebook(phi, k, space, fit_phase, seed)
    ref = assign(phi, full)
    rng = np.random.default_rng(seed)
    fit_rows = phi[phi["phase"] == fit_phase] if "phase" in phi else phi
    scores = []
    for i in range(int(n_rep)):
        take = rng.choice(len(fit_rows), max(k + 1, int(frac * len(fit_rows))),
                          replace=False)
        cb = fit_codebook(fit_rows.iloc[take], k, space, fit_phase, seed + i)
        scores.append(adjusted_rand_score(ref, assign(phi, cb)))
    v = np.asarray(scores, float)
    return {"k": int(k), "ari_mean": float(v.mean()), "ari_min": float(v.min()),
            "distortion": full["distortion"], "n_rep": int(n_rep),
            "stable": bool(v.mean() >= 0.7)}


def assign(phi: pd.DataFrame, cb: dict) -> np.ndarray:
    X = phi[cb["cols"]].to_numpy(float)
    d = ((X[:, None, :] - cb["centres"][None, :, :]) ** 2).sum(axis=2)
    s = np.argmin(d, axis=1).astype(int)
    s[~np.isfinite(X).all(axis=1)] = -1
    return s


def symbol_table(phi: pd.DataFrame, cb: dict, seq_key: str | None = None
                 ) -> pd.DataFrame:
    """phi + `s`. `seq_key` (e.g. `cycle_pos`) names the replicate axis."""
    out = phi.copy()
    out["s"] = assign(out, cb)
    out["seq"] = 0 if seq_key is None or seq_key not in out.columns \
        else out[seq_key].astype(int)
    keep = ["client", "seq", "bid", "s"] + \
        (["phase"] if "phase" in out.columns else [])
    return out[keep].sort_values(["client", "seq", "bid"]).reset_index(drop=True)


def sequences(sym: pd.DataFrame, client: str | None = None,
              phase: str | None = None) -> list[np.ndarray]:
    """One array per replicate sequence, ordered by block, invalid dropped."""
    d = sym
    if client is not None:
        d = d[d["client"] == client]
    if phase is not None and "phase" in d:
        d = d[d["phase"] == phase]
    out = []
    for _, g in d.sort_values("bid").groupby("seq"):
        a = g["s"].to_numpy(int)
        if (a >= 0).all() and len(a) > 1:
            out.append(a)
    return out


# --------------------------------------------------------------------------
# chain objects
# --------------------------------------------------------------------------
def transition_matrix(seqs, k: int, alpha: float = 0.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(P-hat, counts). `alpha` is a Laplace prior per cell (0 = plug-in)."""
    C = np.zeros((k, k), float)
    for a in seqs:
        a = np.asarray(a, int)
        np.add.at(C, (a[:-1], a[1:]), 1.0)
    P = C + float(alpha)
    rs = P.sum(axis=1, keepdims=True)
    P = np.divide(P, np.where(rs > 0, rs, 1.0))
    P[rs[:, 0] == 0] = 1.0 / k                     # unvisited row: uninformative
    return P, C


def stationary(P: np.ndarray) -> np.ndarray:
    """Leading left eigenvector of P (closed form -- no sampling needed)."""
    w, V = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(w - 1.0)))
    v = np.real(V[:, i])
    v = np.clip(v, 0, None)
    s = v.sum()
    return v / s if s > 0 else np.full(len(P), 1.0 / len(P))


def occupancy(seqs, k: int) -> np.ndarray:
    a = np.concatenate([np.asarray(s, int) for s in seqs]) if seqs \
        else np.zeros(0, int)
    c = np.bincount(a, minlength=k).astype(float)
    return c / c.sum() if c.sum() else np.full(k, 1.0 / k)


def entropy_rate(P: np.ndarray, pi: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(P > 0, np.log(P), 0.0)
    return float(-(pi[:, None] * P * lg).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log(a[ok] / np.where(b[ok] > 0,
                                                            b[ok], 1e-300))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def occupancy_matrix(sym: pd.DataFrame, k: int, phase: str | None = None
                     ) -> pd.DataFrame:
    """Pairwise similarity from occupancy: `1 - JS(pi_c, pi_d)/log 2`.

    Unlike a cosine this does not divide out the norm, so on an ordinal
    regime ladder the magnitude survives.
    """
    clients = sorted(sym["client"].unique())
    pis = {c: occupancy(sequences(sym, c, phase), k) for c in clients}
    rows = []
    for i, a in enumerate(clients):
        for b in clients[i + 1:]:
            js = js_divergence(pis[a], pis[b])
            rows.append({"client_a": a, "client_b": b, "phase": phase,
                         "js": js, "sim": 1 - js / np.log(2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# entropies with Miller-Madow
# --------------------------------------------------------------------------
def _H(counts: np.ndarray) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    H = float(-(p * np.log(p)).sum())
    return H + (np.count_nonzero(counts) - 1) / (2.0 * n)   # Miller-Madow


def _joint(*cols, k: int) -> np.ndarray:
    idx = np.zeros(len(cols[0]), int)
    for c in cols:
        idx = idx * k + np.asarray(c, int)
    return np.bincount(idx, minlength=k ** len(cols)).astype(float)


def mi(x, y, k: int) -> dict:
    """I(x ; y) with Miller-Madow, and the plug-in bias for reference."""
    x, y = np.asarray(x, int), np.asarray(y, int)
    n = len(x)
    if n < 2:
        return {"value": np.nan, "bias": np.nan, "n": n}
    v = _H(_joint(x, k=k)) + _H(_joint(y, k=k)) - _H(_joint(x, y, k=k))
    return {"value": float(v), "bias": MI_DF(k) / (2.0 * n), "n": int(n)}


def transfer_entropy(x_prev, y_prev, y_next, k: int) -> dict:
    """T = I(y_next ; x_prev | y_prev), Miller-Madow on each joint."""
    xp, yp, yn = (np.asarray(a, int) for a in (x_prev, y_prev, y_next))
    n = len(xp)
    if n < 2:
        return {"value": np.nan, "bias": np.nan, "n": n}
    v = (_H(_joint(yn, yp, k=k)) - _H(_joint(yp, k=k))
         - _H(_joint(yn, yp, xp, k=k)) + _H(_joint(yp, xp, k=k)))
    return {"value": float(v), "bias": TE_DF(k) / (2.0 * n), "n": int(n)}


def _aligned_lags(sym: pd.DataFrame, src: str, tgt: str):
    """(x_prev, y_prev, y_next, x_now, y_now) pooled over replicate sequences."""
    a = sym[sym["client"] == src].set_index(["seq", "bid"])["s"]
    b = sym[sym["client"] == tgt].set_index(["seq", "bid"])["s"]
    xp, yp, yn, xn, yn0 = [], [], [], [], []
    for seq in sorted(set(a.index.get_level_values(0))
                      & set(b.index.get_level_values(0))):
        ai, bi = a.loc[seq].sort_index(), b.loc[seq].sort_index()
        bids = sorted(set(ai.index) & set(bi.index))
        for t in range(1, len(bids)):
            b0, b1 = bids[t - 1], bids[t]
            if int(b1) - int(b0) != 1:
                continue                      # no lag across a gap
            if min(ai[b0], bi[b0], bi[b1], ai[b1]) < 0:
                continue
            xp.append(ai[b0]); yp.append(bi[b0]); yn.append(bi[b1])
            xn.append(ai[b1]); yn0.append(bi[b1])
    return (np.array(xp, int), np.array(yp, int), np.array(yn, int),
            np.array(xn, int), np.array(yn0, int))


def pair_table(sym: pd.DataFrame, k: int, phase: str | None = None
               ) -> pd.DataFrame:
    """MI and both directions of TE for every pair, with bias attached."""
    d = sym if phase is None or "phase" not in sym else sym[sym["phase"] == phase]
    clients = sorted(d["client"].unique())
    rows = []
    for i, a in enumerate(clients):
        for b in clients[i + 1:]:
            xp, yp, yn, xn, yn0 = _aligned_lags(d, a, b)
            if not len(xp):
                continue
            m = mi(xn, yn0, k)
            t_ab = transfer_entropy(xp, yp, yn, k)
            yp2, xp2, xn2, _, _ = _aligned_lags(d, b, a)
            t_ba = transfer_entropy(yp2, xp2, xn2, k)
            rows.append({"client_a": a, "client_b": b, "phase": phase,
                         "n": m["n"], "mi": m["value"], "mi_bias": m["bias"],
                         "te_a_to_b": t_ab["value"], "te_b_to_a": t_ba["value"],
                         "te_bias": t_ab["bias"],
                         "te_asym": t_ab["value"] - t_ba["value"],
                         "mi_above_bias": m["value"] > m["bias"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# structure tests
# --------------------------------------------------------------------------
def _lr(counts: np.ndarray, P_null: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        P = counts / np.where(counts.sum(axis=-1, keepdims=True) > 0,
                              counts.sum(axis=-1, keepdims=True), 1.0)
        term = np.where((counts > 0) & (P_null > 0),
                        counts * np.log(np.where(P > 0, P, 1.0)
                                        / np.where(P_null > 0, P_null, 1.0)),
                        0.0)
    return float(2.0 * term.sum())


def markov_order_test(seqs, k: int) -> pd.DataFrame:
    """Order 0 vs 1 and order 1 vs 2, likelihood ratio against chi2."""
    from scipy import stats
    P1, C1 = transition_matrix(seqs, k)
    pi = occupancy([s[1:] for s in seqs if len(s) > 1], k)
    g01 = _lr(C1, np.tile(pi, (k, 1)))
    df01 = k * (k - 1)
    C2 = np.zeros((k, k, k))
    for a in seqs:
        a = np.asarray(a, int)
        if len(a) > 2:
            np.add.at(C2, (a[:-2], a[1:-1], a[2:]), 1.0)
    g12 = _lr(C2, np.broadcast_to(P1, (k, k, k)))
    df12 = k * k * (k - 1) - k * (k - 1)
    return pd.DataFrame([
        {"test": "order 0 vs 1", "G2": g01, "df": df01,
         "p": float(stats.chi2.sf(g01, df01)),
         "reads": "reject = there ARE dynamics; not rejected = the chain "
                  "adds nothing over the occupancy histogram"},
        {"test": "order 1 vs 2", "G2": g12, "df": df12,
         "p": float(stats.chi2.sf(g12, df12)),
         "reads": "reject = order 1 is too short a memory"}])


def homogeneity_test(groups: dict, k: int) -> dict:
    """Are these stretches the same chain? `groups` = {name: seqs}."""
    from scipy import stats
    mats = {g: transition_matrix(s, k)[1] for g, s in groups.items()}
    pooled = sum(mats.values())
    rs = pooled.sum(axis=1, keepdims=True)
    P0 = np.divide(pooled, np.where(rs > 0, rs, 1.0))
    G2 = float(sum(_lr(C, P0) for C in mats.values()))
    df = (len(mats) - 1) * k * (k - 1)
    return {"G2": G2, "df": df, "p": float(stats.chi2.sf(G2, df)),
            "n_groups": len(mats),
            "n_transitions": {g: float(C.sum()) for g, C in mats.items()}}


def stationarity_split(sym: pd.DataFrame, k: int, client: str, phase: str
                       ) -> dict:
    """Split a phase in half and test the two halves for the same chain.

    Init is expected to PASS (one regime) and transition to FAIL (the regime
    is moving) -- the contrast is the claim, not either p-value alone.
    """
    seqs = sequences(sym, client, phase)
    first, second = [], []
    for a in seqs:
        h = len(a) // 2
        if h > 1:
            first.append(a[:h])
            second.append(a[h:])
    if not first:
        return {"client": client, "phase": phase, "G2": np.nan, "df": 0,
                "p": np.nan, "n_groups": 0, "n_transitions": {}}
    out = homogeneity_test({"first": first, "second": second}, k)
    out.update(client=client, phase=phase)
    return out


# --------------------------------------------------------------------------
# nulls
# --------------------------------------------------------------------------
def _simulate(P: np.ndarray, starts, lengths, rng) -> list[np.ndarray]:
    """One sequence per (start, length); each starts at its OBSERVED first
    state, never at a draw from the stationary distribution."""
    out = []
    k = len(P)
    for s0, n in zip(starts, lengths):
        s = np.empty(int(n), int)
        s[0] = int(s0)
        for t in range(1, int(n)):
            s[t] = rng.choice(k, p=P[s[t - 1]])
        out.append(s)
    return out


def _seq_meta(d: pd.DataFrame, client: str) -> list[tuple[int, list, np.ndarray]]:
    """(seq, bids, symbols) for each valid sequence of `client` -- the same
    filter as `sequences`, keeping the block ids so simulations land on the
    right blocks."""
    out = []
    for seq, g in d[d["client"] == client].sort_values("bid").groupby("seq"):
        a = g["s"].to_numpy(int)
        if (a >= 0).all() and len(a) > 1:
            out.append((seq, g["bid"].tolist(), a))
    return out


def _null_summary(value: float, v: np.ndarray, label: str, a: str, b: str,
                  stat: str) -> dict:
    """A null whose draws are all identical cannot rank the observation."""
    degenerate = len(v) > 0 and float(np.std(v)) < 1e-12
    return {"client_a": a, "client_b": b, "stat": stat, "value": value,
            "null_mean": float(v.mean()) if len(v) else np.nan,
            "null_q95": float(np.quantile(v, 0.95)) if len(v) else np.nan,
            "p_emp": (np.nan if (degenerate or not len(v)) else
                      float((1 + np.sum(v >= value)) / (1 + len(v)))),
            "n_rep": int(len(v)),
            "null": label + (" (degenerate: all draws equal)" if degenerate else "")}


def chain_null(sym: pd.DataFrame, k: int, a: str, b: str,
               phase: str | None = None, n_rep: int = 200, seed: int = 0,
               stat: str = "mi") -> dict:
    """Resample both clients from their OWN fitted chains, independently.

    Preserves each client's marginal dynamics and destroys the cross-client
    alignment -- strictly stronger than a circular shift, which leaves the
    shared trend in place.

    Each simulated sequence starts at the client's OBSERVED first state. On a
    one-way drift the fitted chain is close to absorbing, and starting from
    its stationary distribution would place every draw inside the absorbing
    state: the null collapses to a constant (MI = 0 on every draw) and any
    observed MI looks significant. A null whose draws are all equal is
    reported as degenerate with `p_emp = NaN` rather than ranked against.
    """
    rng = np.random.default_rng(seed)
    d = sym if phase is None or "phase" not in sym else sym[sym["phase"] == phase]
    obs = pair_table(d, k, None)
    obs = obs[(obs["client_a"] == a) & (obs["client_b"] == b)]
    if not len(obs):
        return {"client_a": a, "client_b": b, "stat": stat, "value": np.nan}
    key = {"mi": "mi", "te": "te_a_to_b"}[stat]
    value = float(obs[key].iloc[0])

    meta = {c: _seq_meta(d, c) for c in (a, b)}
    fitted = {c: transition_matrix([m[2] for m in meta[c]], k)[0] for c in (a, b)}
    vals = []
    for _ in range(int(n_rep)):
        rows = []
        for c in (a, b):
            sims = _simulate(fitted[c], [m[2][0] for m in meta[c]],
                             [len(m[2]) for m in meta[c]], rng)
            for (seq, bids, _), arr in zip(meta[c], sims):
                rows.extend({"client": c, "seq": seq, "bid": bid, "s": int(x)}
                            for bid, x in zip(bids, arr))
        pt = pair_table(pd.DataFrame(rows), k, None)
        pt = pt[(pt["client_a"] == a) & (pt["client_b"] == b)]
        if len(pt):
            vals.append(float(pt[key].iloc[0]))
    return _null_summary(value, np.asarray(vals, float), "fitted chains",
                         a, b, stat)


def month_block_null(sym: pd.DataFrame, k: int, a: str, b: str,
                     phase: str | None = None, n_rep: int = 200,
                     seed: int = 0, stat: str = "mi") -> dict:
    """Permute WHOLE blocks of client `a`, classes attached.

    The inference-correct null for a class-pooled estimate: shuffling class
    sequences independently would treat the classes of one month as
    independent replicates, which they are not.
    """
    rng = np.random.default_rng(seed)
    d = (sym if phase is None or "phase" not in sym
         else sym[sym["phase"] == phase]).copy()
    key = {"mi": "mi", "te": "te_a_to_b"}[stat]
    obs = pair_table(d, k, None)
    obs = obs[(obs["client_a"] == a) & (obs["client_b"] == b)]
    if not len(obs):
        return {"client_a": a, "client_b": b, "stat": stat, "value": np.nan}
    value = float(obs[key].iloc[0])

    bids = np.array(sorted(d.loc[d["client"] == a, "bid"].unique()))
    vals = []
    for _ in range(int(n_rep)):
        perm = dict(zip(bids, rng.permutation(bids)))
        e = d.copy()
        m = e["client"] == a
        e.loc[m, "bid"] = e.loc[m, "bid"].map(perm)
        pt = pair_table(e, k, None)
        pt = pt[(pt["client_a"] == a) & (pt["client_b"] == b)]
        if len(pt):
            vals.append(float(pt[key].iloc[0]))
    return _null_summary(value, np.asarray(vals, float), "month blocks",
                         a, b, stat)
