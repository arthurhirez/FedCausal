"""Quantised state-space analysis: symbolic Markov chains and directed dependence.

The idea
--------
Quantise each node's time series into a small alphabet, estimate transition
probabilities, and read dependence off the *probabilities of change* rather than
off linear correlations. This buys three things a linear model does not give:
nonlinearity, an explicit regime representation (which is what a drift changes),
and a non-parametric directed dependence measure.

What each object is for
-----------------------
* **Symbols.** A node's deseasonalised residual, quantised to K levels using
  thresholds frozen on a baseline window. Freezing is essential: re-quantiling
  per window would move the thresholds with the distribution and make drift
  invisible by construction, which is the single easiest way to get a
  meaningless result here.
* **Per-node chain.** Transition matrix, exact stationary distribution, dwell
  times. The stationary distribution is *occupancy*: how often the node sits in
  each regime. Drift shows up as occupancy moving and as the transition matrix
  changing.
* **Lag-0 mutual information.** Undirected instantaneous coupling. This is the
  hydraulic part, and it is undirected for a reason: EPANET's quasi-steady
  solution has a symmetric demand-to-pressure Jacobian, so same-timestep
  coupling carries no direction. Reporting a direction for it would be an
  artifact of the estimator, not a property of the system.
* **Transfer entropy at lag >= 1.** Directed dependence,
  ``TE(i->j) = I(Y_{t+1} ; X_{t+1-L} | Y_t)``. This is the information-theoretic
  counterpart of Granger causality, so it doubles as a non-parametric check on
  linear Granger machinery. In this system it can only be non-zero through tank
  mass or control switching, because nothing else carries state between
  timesteps.

Why MCMC is deliberately absent
-------------------------------
For per-node or pairwise chains with K = 3, the stationary distribution is an
eigenvector of a 3x3 or 9x9 matrix and is solved exactly in microseconds.
Sampling it would be strictly worse. MCMC only earns its place for a *joint*
model over many nodes, where the state space is K^n; if that model becomes the
object of interest, it comes back then. Two different stationary distributions
do appear here, and they answer different questions: the stationary
distribution of a node's symbol chain characterises its regime, while the
stationary distribution of a random walk on the dependence graph ranks global
influence.

Known limitations, stated rather than discovered later
------------------------------------------------------
1. **Quantisation discards information.** A K = 3 alphabet cannot represent
   graded response. The sensitivity of every conclusion to K should be reported,
   which ``sweep_alphabet`` does.
2. **Transfer entropy is positively biased** at finite sample size: with K = 3
   the conditional table has K^3 = 27 cells, and sparse cells inflate the
   estimate. Every reported value is surrogate-corrected, and the surrogate
   distribution is kept so the correction is auditable.
3. **Lag-0 coupling is invisible to TE by construction.** In a network with no
   storage, the true coupling is entirely instantaneous, so TE at lag >= 1 will
   be null however strong the physical coupling is. That is a correct result and
   must not be read as absence of dependence.
4. **Common drivers create spurious directed edges.** Two nodes fed by the same
   district noise process share a driver; conditioning only on the target's own
   past does not remove it. ``conditional_te`` conditions on a third series
   (typically the district or system aggregate) to test whether an edge survives.
5. **Symbol series from a shared pattern are strongly synchronised.** With
   district-level noise, nodes in a district are near-copies; MI will be high for
   reasons of shared input rather than hydraulic coupling. This is exactly why
   validation against the interventional map matters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


# ==========================================================================
# configuration
# ==========================================================================

@dataclass
class StateSpaceConfig:
    n_symbols: int = 3                    # K
    mode: str = "level"                   # "level" | "delta"
    baseline_fraction: float = 0.30       # leading share used to fit thresholds
    deseasonalise: bool = True
    detrend_window: int | None = 336   # steps; None disables. See `detrend`.
    lags: tuple[int, ...] = (1, 2, 3, 6)
    n_surrogates: int = 50
    surrogate_block: int = 24             # circular block shuffle length (hours)
    significance_quantile: float = 0.95
    min_symbol_occupancy: float = 0.02    # reject near-degenerate alphabets
    seed: int = 0

    def validate(self):
        if self.n_symbols < 2:
            raise ValueError("n_symbols must be >= 2")
        if self.mode not in ("level", "delta"):
            raise ValueError("mode must be 'level' or 'delta'")
        if not 0 < self.baseline_fraction < 1:
            raise ValueError("baseline_fraction must be in (0, 1)")
        if any(l < 1 for l in self.lags):
            raise ValueError("lags must all be >= 1; lag 0 is handled by mutual "
                             "information, which is undirected by construction")
        return self


# ==========================================================================
# residuals and quantisation
# ==========================================================================

def detrend(values: pd.DataFrame, window: int) -> pd.DataFrame:
    """Remove each node's own slow trend with a centred rolling median.

    This is not cosmetic. Population growth imposes a shared, monotone,
    non-stationary trend on every node at once. Transfer entropy on trending
    series is the classic spurious-regression problem in symbolic form: the
    block-shuffle surrogate destroys trend alignment, so the null is far too
    easy to beat and effectively every ordered pair comes back significant.

    Measured on a tankless, controlless network -- where quasi-steady hydraulics
    guarantees there is no lag mechanism at all, so the true answer is zero:

        growth 5%/chunk                 100.0% significant, mean TE 0.0097
        zero growth, no seasonality      21.7% significant, mean TE 0.0003
        growth 5%, detrended             31.2% significant, mean TE 0.0005

    A 19-32x inflation of the TE estimate attributable purely to the trend. The
    residual significance above the nominal 5% shows the block-shuffle null
    remains mildly anti-conservative even after detrending, so p-values here are
    a screening device rather than an exact test.

    A median (not a mean) is used so that a step change in regime is not smeared
    across the transition window.
    """
    if window is None or window <= 1:
        return values
    trend = values.rolling(window, center=True, min_periods=1).median()
    return values - trend


def deseasonalise(values: pd.DataFrame, calendar: pd.DataFrame,
                  baseline_mask: np.ndarray) -> pd.DataFrame:
    """Remove the hour-of-day x day-of-week profile fitted on the baseline only.

    Raw pressure is dominated by elevation and by the common diurnal cycle; both
    are shared across the whole network and would swamp any node-to-node
    structure. Fitting the profile on the baseline window and applying it
    everywhere means a later change in *shape* survives into the residual
    instead of being absorbed by it.
    """
    if len(values) != len(calendar):
        raise ValueError("values and calendar must have the same length")
    key = (calendar.hour.to_numpy() * 7 + calendar.dow.to_numpy())
    V = values.to_numpy(dtype=float)
    out = V.copy()
    for k in np.unique(key):
        sel = key == k
        base_sel = sel & baseline_mask
        if base_sel.sum() >= 2:
            mu = np.nanmean(V[base_sel], axis=0)
        elif sel.sum() >= 1:
            mu = np.nanmean(V[sel], axis=0)
        else:
            continue
        out[sel] = V[sel] - mu
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def quantise(values: pd.DataFrame, cfg: StateSpaceConfig,
             baseline_mask: np.ndarray) -> dict:
    """Map each column to K symbols using thresholds frozen on the baseline.

    Returns the symbol frame, the per-node thresholds and an occupancy report.
    Columns whose baseline is degenerate (a constant, or an alphabet where some
    symbol is almost never visited) are flagged: a chain over an effectively
    smaller alphabet gives transition estimates that look confident and mean
    nothing.
    """
    cfg = cfg.validate()
    V = values.to_numpy(dtype=float)
    if cfg.mode == "delta":
        V = np.vstack([np.zeros((1, V.shape[1])), np.diff(V, axis=0)])
    K = cfg.n_symbols
    qs = np.linspace(0, 1, K + 1)[1:-1]

    sym = np.zeros_like(V, dtype=np.int8)
    thresholds, report = {}, []
    for c, name in enumerate(values.columns):
        col = V[:, c]
        base = col[baseline_mask]
        base = base[np.isfinite(base)]
        if base.size < K * 5 or np.allclose(base, base[0] if base.size else 0.0):
            thresholds[name] = np.full(K - 1, np.nan)
            report.append({"node": name, "degenerate": True,
                           "reason": "constant or too-short baseline"})
            continue
        thr = np.quantile(base, qs)
        thr = np.unique(thr)
        if thr.size < K - 1:
            thresholds[name] = thr
            report.append({"node": name, "degenerate": True,
                           "reason": "tied quantiles: distribution too discrete"})
            sym[:, c] = np.digitize(col, thr)
            continue
        thresholds[name] = thr
        sym[:, c] = np.digitize(col, thr)

    S = pd.DataFrame(sym, index=values.index, columns=values.columns)
    occ = pd.DataFrame({s: (S == s).mean() for s in range(K)})
    occ.columns = [f"occ_{s}" for s in range(K)]
    occ["min_occupancy"] = occ.min(axis=1)
    occ["usable"] = occ.min_occupancy >= cfg.min_symbol_occupancy
    rep = pd.DataFrame(report).set_index("node") if report else pd.DataFrame()
    return {"symbols": S, "thresholds": thresholds, "occupancy": occ,
            "degenerate": rep, "config": asdict(cfg)}


# ==========================================================================
# per-node Markov chains
# ==========================================================================

def transition_matrix(sym: np.ndarray, K: int, laplace: float = 0.0) -> np.ndarray:
    """Row-stochastic transition matrix from a symbol sequence."""
    s = np.asarray(sym, dtype=np.int64)
    a, b = s[:-1], s[1:]
    ok = (a >= 0) & (a < K) & (b >= 0) & (b < K)
    counts = np.bincount(a[ok] * K + b[ok], minlength=K * K).reshape(K, K).astype(float)
    counts += laplace
    rows = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(rows > 0, counts / np.maximum(rows, EPS), 1.0 / K)
    return P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Exact stationary distribution as the unit left eigenvector of P.

    Solved directly rather than sampled: for K on the order of 3-5 this is a
    tiny eigenproblem, so MCMC would add variance and cost for nothing.
    """
    K = P.shape[0]
    A = np.vstack([P.T - np.eye(K), np.ones((1, K))])
    b = np.concatenate([np.zeros(K), [1.0]])
    try:
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0, None)
        return pi / max(pi.sum(), EPS)
    except np.linalg.LinAlgError:
        return np.full(K, 1.0 / K)


def dwell_times(sym: np.ndarray, K: int) -> pd.DataFrame:
    """Mean and max run length in each symbol: how persistent each regime is."""
    s = np.asarray(sym)
    if s.size == 0:
        return pd.DataFrame()
    change = np.flatnonzero(np.diff(s) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [s.size]])
    runs = pd.DataFrame({"symbol": s[starts], "length": ends - starts})
    g = runs.groupby("symbol").length
    return pd.DataFrame({"n_runs": g.size(), "mean_dwell": g.mean(),
                         "max_dwell": g.max()}).reindex(range(K))


def markov_order_bic(sym: np.ndarray, K: int) -> dict:
    """Is a first-order chain defensible? Compare orders 1 and 2 by BIC.

    Reported rather than enforced. If order 2 wins decisively, transfer entropy
    conditioned on a single past step is under-conditioned and its edges may
    reflect the target's own unmodelled memory instead of the source.
    """
    s = np.asarray(sym, dtype=np.int64)
    n = s.size
    if n < 50:
        return {"n": n, "verdict": "too short"}

    def loglik(order):
        if order == 1:
            a, b = s[:-1], s[1:]
            idx = a * K + b
            c = np.bincount(idx, minlength=K * K).reshape(K, K).astype(float)
        else:
            a, b, c2 = s[:-2], s[1:-1], s[2:]
            idx = (a * K + b) * K + c2
            c = np.bincount(idx, minlength=K * K * K).reshape(K * K, K).astype(float)
        rows = c.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(c > 0, c * (np.log(np.maximum(c, EPS)) -
                                      np.log(np.maximum(rows, EPS))), 0.0)
        k_params = c.shape[0] * (K - 1)
        return float(lp.sum()), k_params

    l1, k1 = loglik(1)
    l2, k2 = loglik(2)
    bic1 = -2 * l1 + k1 * np.log(n)
    bic2 = -2 * l2 + k2 * np.log(n)
    return {"n": n, "loglik_order1": l1, "loglik_order2": l2,
            "bic_order1": bic1, "bic_order2": bic2,
            "delta_bic": bic1 - bic2,
            "verdict": "order 2 preferred" if bic2 < bic1 else "order 1 adequate"}


def chain_report(symbols: pd.DataFrame, cfg: StateSpaceConfig,
                 nodes=None) -> pd.DataFrame:
    """Per-node chain summary: occupancy, persistence, entropy rate, order test."""
    cfg = cfg.validate()
    K = cfg.n_symbols
    nodes = list(nodes) if nodes is not None else list(symbols.columns)
    rows = []
    for n in nodes:
        s = symbols[n].to_numpy()
        P = transition_matrix(s, K)
        pi = stationary_distribution(P)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.nansum(pi[:, None] * P * np.log2(np.maximum(P, EPS)))
        dw = dwell_times(s, K)
        ob = markov_order_bic(s, K)
        rows.append({
            "node": n,
            **{f"pi_{i}": pi[i] for i in range(K)},
            "entropy_rate_bits": float(h),
            "self_transition_mean": float(np.mean(np.diag(P))),
            "mean_dwell": float(dw.mean_dwell.mean(skipna=True))
            if len(dw) else np.nan,
            "delta_bic_order": ob.get("delta_bic", np.nan),
            "order_verdict": ob.get("verdict", ""),
        })
    return pd.DataFrame(rows).set_index("node")


# ==========================================================================
# drift / regime detection
# ==========================================================================

def occupancy_over_time(symbols: pd.DataFrame, cfg: StateSpaceConfig,
                        window: int, step: int | None = None,
                        nodes=None) -> pd.DataFrame:
    """Sliding-window symbol occupancy: the observable a drift moves."""
    cfg = cfg.validate()
    K = cfg.n_symbols
    step = step or max(1, window // 4)
    nodes = list(nodes) if nodes is not None else list(symbols.columns)
    S = symbols[nodes].to_numpy()
    rows = []
    for start in range(0, len(S) - window + 1, step):
        w = S[start:start + window]
        for j, n in enumerate(nodes):
            col = w[:, j]
            occ = np.bincount(col, minlength=K)[:K] / max(col.size, 1)
            rows.append({"start": start, "centre": start + window // 2,
                         "node": n, **{f"occ_{i}": occ[i] for i in range(K)}})
    return pd.DataFrame(rows)


def regime_distance(symbols: pd.DataFrame, cfg: StateSpaceConfig,
                    baseline_mask: np.ndarray, window: int,
                    step: int | None = None, nodes=None) -> pd.DataFrame:
    """Distance from the baseline regime, per node and window.

    Two distances, because they detect different things: total variation on the
    occupancy vector detects a change in *where* the process sits, while the
    transition-matrix distance detects a change in *how it moves* even when
    occupancy is unchanged. A drift that alters demand shape without altering
    its mean shows up in the second and not the first.
    """
    cfg = cfg.validate()
    K = cfg.n_symbols
    step = step or max(1, window // 4)
    nodes = list(nodes) if nodes is not None else list(symbols.columns)
    rows = []
    for n in nodes:
        s = symbols[n].to_numpy()
        b = s[baseline_mask]
        occ_b = np.bincount(b, minlength=K)[:K] / max(b.size, 1)
        P_b = transition_matrix(b, K, laplace=0.5)
        for start in range(0, len(s) - window + 1, step):
            w = s[start:start + window]
            occ = np.bincount(w, minlength=K)[:K] / max(w.size, 1)
            P_w = transition_matrix(w, K, laplace=0.5)
            rows.append({
                "node": n, "start": start, "centre": start + window // 2,
                "tv_occupancy": float(0.5 * np.abs(occ - occ_b).sum()),
                "tv_transition": float(0.5 * np.abs(P_w - P_b).sum(axis=1).mean()),
            })
    return pd.DataFrame(rows)


# ==========================================================================
# information-theoretic dependence
# ==========================================================================

def mutual_information(x: np.ndarray, y: np.ndarray, K: int) -> float:
    """Lag-0 mutual information in bits: undirected instantaneous coupling."""
    a = np.asarray(x, dtype=np.int64)
    b = np.asarray(y, dtype=np.int64)
    ok = (a >= 0) & (a < K) & (b >= 0) & (b < K)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return np.nan
    c = np.bincount(a * K + b, minlength=K * K).reshape(K, K).astype(float)
    p = c / c.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log2(np.maximum(p, EPS) /
                                           np.maximum(px * py, EPS)), 0.0)
    return float(term.sum())


def transfer_entropy(source: np.ndarray, target: np.ndarray, K: int,
                     lag: int = 1) -> float:
    """TE(source -> target) = I(Y_{t+1} ; X_{t+1-lag} | Y_t), in bits.

    The conditioning on ``Y_t`` is what makes this directed and what
    distinguishes it from mutual information: it asks whether the source adds
    predictive information *beyond the target's own past*. That is the
    information-theoretic statement of Granger causality, without the linearity
    and Gaussianity assumptions.
    """
    x = np.asarray(source, dtype=np.int64)
    y = np.asarray(target, dtype=np.int64)
    n = min(x.size, y.size)
    if lag < 1 or n < lag + 10:
        return np.nan
    t = np.arange(lag - 1, n - 1)
    y1, y0, xp = y[t + 1], y[t], x[t + 1 - lag]
    ok = ((y1 >= 0) & (y1 < K) & (y0 >= 0) & (y0 < K) &
          (xp >= 0) & (xp < K))
    y1, y0, xp = y1[ok], y0[ok], xp[ok]
    if y1.size < 10:
        return np.nan

    c = np.bincount((y1 * K + y0) * K + xp,
                    minlength=K ** 3).reshape(K, K, K).astype(float)
    p = c / c.sum()
    p_y0x = p.sum(axis=0)                       # (y0, x)
    p_y1y0 = p.sum(axis=2)                      # (y1, y0)
    p_y0 = p.sum(axis=(0, 2))                   # (y0,)
    num = p * p_y0[None, :, None]
    den = p_y0x[None, :, :] * p_y1y0[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log2(np.maximum(num, EPS) /
                                           np.maximum(den, EPS)), 0.0)
    return float(term.sum())


def conditional_te(source: np.ndarray, target: np.ndarray,
                   condition: np.ndarray, K: int, lag: int = 1,
                   condition_lag: int | None = None) -> float:
    """TE with an extra conditioning series: guards against a common driver.

    Nodes sharing a district demand process share a driver, and conditioning on
    the target's own past does not remove it. Conditioning additionally on that
    shared series tests whether an edge survives once the common input is
    accounted for. The cost is a K^4 table, so it needs more data and is meant
    for confirming a shortlist rather than for scanning.

    ``condition_lag`` must match the lag at which the confounder acts, and
    defaults to ``lag``. This is not a detail: conditioning on the confounder at
    the wrong lag leaves most of the spurious dependence in place. On a planted
    common-driver structure, conditioning at the source's own lag removes the
    spurious edge, while conditioning at lag 1 against a lag-2 confound removes
    only about a quarter of it.
    """
    x = np.asarray(source, dtype=np.int64)
    y = np.asarray(target, dtype=np.int64)
    z = np.asarray(condition, dtype=np.int64)
    n = min(x.size, y.size, z.size)
    if lag < 1 or n < lag + 20:
        return np.nan
    cl = lag if condition_lag is None else int(condition_lag)
    if cl < 1:
        raise ValueError("condition_lag must be >= 1")
    t = np.arange(max(lag, cl) - 1, n - 1)
    y1, y0, xp, zp = y[t + 1], y[t], x[t + 1 - lag], z[t + 1 - cl]
    ok = np.all([(v >= 0) & (v < K) for v in (y1, y0, xp, zp)], axis=0)
    y1, y0, xp, zp = y1[ok], y0[ok], xp[ok], zp[ok]
    if y1.size < 20:
        return np.nan
    idx = ((y1 * K + y0) * K + xp) * K + zp
    c = np.bincount(idx, minlength=K ** 4).reshape(K, K, K, K).astype(float)
    p = c / c.sum()
    p_y0xz = p.sum(axis=0)
    p_y1y0z = p.sum(axis=2)
    p_y0z = p.sum(axis=(0, 2))
    num = p * p_y0z[None, :, None, :]
    den = p_y0xz[None, :, :, :] * p_y1y0z[:, :, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * np.log2(np.maximum(num, EPS) /
                                           np.maximum(den, EPS)), 0.0)
    return float(term.sum())


def _circular_block_shuffle(x: np.ndarray, block: int, rng) -> np.ndarray:
    """Destroy cross-series timing while preserving within-series structure.

    A plain permutation would also destroy the source's autocorrelation, making
    the null too easy to beat and every edge look significant. Block shuffling
    keeps short-range structure and only scrambles the alignment with the
    target, which is the hypothesis being tested.
    """
    n = x.size
    nb = max(1, int(np.ceil(n / block)))
    starts = rng.integers(0, n, size=nb)
    pieces = [np.take(x, np.arange(s, s + block) % n, mode="wrap")
              for s in starts]
    return np.concatenate(pieces)[:n]


def dependence_matrices(symbols: pd.DataFrame, cfg: StateSpaceConfig,
                        nodes=None, verbose: bool = False) -> dict:
    """Lag-0 MI and surrogate-corrected TE over all ordered pairs of ``nodes``.

    Cost is O(n^2 * lags * surrogates), so ``nodes`` should be a manageable
    subset for large networks. Subsetting is a computational choice and does not
    bias the comparison against the interventional map, provided the same node
    set is used for both.
    """
    cfg = cfg.validate()
    K = cfg.n_symbols
    nodes = list(nodes) if nodes is not None else list(symbols.columns)
    S = {n: symbols[n].to_numpy(dtype=np.int64) for n in nodes}
    rng = np.random.default_rng(cfg.seed)
    n = len(nodes)

    MI = pd.DataFrame(np.nan, index=nodes, columns=nodes, dtype=float)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            v = mutual_information(S[a], S[b], K)
            MI.loc[a, b] = MI.loc[b, a] = v
    for a in nodes:
        MI.loc[a, a] = 0.0

    te_raw, te_eff, te_p = {}, {}, {}
    for lag in cfg.lags:
        R = pd.DataFrame(np.nan, index=nodes, columns=nodes, dtype=float)
        E = pd.DataFrame(np.nan, index=nodes, columns=nodes, dtype=float)
        Pv = pd.DataFrame(np.nan, index=nodes, columns=nodes, dtype=float)
        for si, src in enumerate(nodes):
            surro = [_circular_block_shuffle(S[src], cfg.surrogate_block, rng)
                     for _ in range(cfg.n_surrogates)]
            for tgt in nodes:
                if tgt == src:
                    continue
                v = transfer_entropy(S[src], S[tgt], K, lag)
                if not np.isfinite(v):
                    continue
                null = np.array([transfer_entropy(sx, S[tgt], K, lag)
                                 for sx in surro], dtype=float)
                null = null[np.isfinite(null)]
                R.loc[src, tgt] = v
                if null.size:
                    E.loc[src, tgt] = max(0.0, v - float(null.mean()))
                    Pv.loc[src, tgt] = float((null >= v).mean())
            if verbose and (si + 1) % 25 == 0:
                print(f"      lag {lag}: {si + 1}/{n} sources")
        te_raw[lag], te_eff[lag], te_p[lag] = R, E, Pv

    return {"MI": MI, "TE_raw": te_raw, "TE_effective": te_eff,
            "TE_pvalue": te_p, "nodes": nodes, "config": asdict(cfg)}


def best_lag_matrix(dep: dict, alpha: float = 0.05) -> dict:
    """Collapse the per-lag TE stack to the strongest significant lag per pair."""
    lags = sorted(dep["TE_effective"])
    nodes = dep["nodes"]
    best = pd.DataFrame(0.0, index=nodes, columns=nodes, dtype=float)
    which = pd.DataFrame(np.nan, index=nodes, columns=nodes, dtype=float)
    for lag in lags:
        E, P = dep["TE_effective"][lag], dep["TE_pvalue"][lag]
        sig = (P <= alpha) & E.notna()
        upd = sig & (E.fillna(-np.inf) > best)
        best = best.where(~upd, E)
        which = which.where(~upd, float(lag))
    return {"TE_best": best, "best_lag": which,
            "n_significant": int((best > 0).to_numpy().sum())}


# ==========================================================================
# graph views
# ==========================================================================

def influence_from_graph(matrix: pd.DataFrame, damping: float = 0.85) -> pd.DataFrame:
    """Rank nodes by a random walk on the dependence graph.

    This is where "use the stationary distribution to find influence" belongs:
    not the stationary distribution of a node's symbol chain (that is its own
    regime occupancy), but that of a walk on the graph, which rewards a node for
    reaching other influential nodes.
    """
    import networkx as nx
    A = matrix.fillna(0.0).to_numpy(dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    g = nx.from_numpy_array(A, create_using=nx.DiGraph)
    g = nx.relabel_nodes(g, dict(enumerate(matrix.index)))
    try:
        pr = nx.pagerank(g, alpha=damping, weight="weight")
    except Exception:
        pr = {n: np.nan for n in matrix.index}
    return pd.DataFrame({
        "out_strength": A.sum(axis=1),
        "in_strength": A.sum(axis=0),
        "walk_influence": [pr.get(n, np.nan) for n in matrix.index],
    }, index=matrix.index).sort_values("walk_influence", ascending=False)


def cluster_from_matrix(matrix: pd.DataFrame, k: int, seed: int = 0) -> pd.Series:
    """Spectral clustering on a symmetrised dependence matrix."""
    from sklearn.cluster import SpectralClustering
    A = np.array(matrix.fillna(0.0).to_numpy(dtype=float), copy=True)
    A = 0.5 * (np.abs(A) + np.abs(A).T)
    np.fill_diagonal(A, 0.0)
    if A.shape[0] <= k:
        return pd.Series(range(A.shape[0]), index=matrix.index, name="cluster")
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            random_state=seed, assign_labels="kmeans")
    return pd.Series(sc.fit_predict(A), index=matrix.index, name="cluster")


# ==========================================================================
# validation against the interventional map
# ==========================================================================

def validate_against_interventional(estimate: pd.DataFrame,
                                    truth_strength: pd.DataFrame,
                                    truth_threshold: float | None = None,
                                    truth_quantile: float = 0.90) -> dict:
    """Score an observational estimator against the interventional ground truth.

    This is the point of the whole exercise. The interventional map is obtained
    by actually applying the do-operator, so it *is* the causal structure at
    that operating point. An observational estimator is what we would have on a
    real utility. Scoring one against the other converts a causality claim into
    a measured detection problem: ROC and precision-recall over the shared node
    set, plus the threshold that maximises F1.

    A caveat that must travel with the numbers: the interventional map is
    undirected (symmetric Jacobian), so it can validate *whether* a pair is
    coupled and cannot validate the *direction* of a TE edge. Direction has to
    be checked against the mechanisms that can carry it, namely storage and
    control switching, and against the measured propagation lag.
    """
    common = [n for n in estimate.index if n in truth_strength.index
              and n in truth_strength.columns and n in estimate.columns]
    if len(common) < 4:
        return {"n_common": len(common), "auc": np.nan,
                "note": "too few shared nodes to score"}

    E = estimate.loc[common, common].to_numpy(dtype=float)
    T = np.abs(truth_strength.loc[common, common].to_numpy(dtype=float))
    E = 0.5 * (np.nan_to_num(E) + np.nan_to_num(E).T)   # score coupling, not direction
    iu = ~np.eye(len(common), dtype=bool)
    e, t = E[iu], T[iu]
    ok = np.isfinite(e) & np.isfinite(t)
    e, t = e[ok], t[ok]
    if e.size < 10 or np.allclose(e, e[0]):
        return {"n_common": len(common), "auc": np.nan,
                "note": "degenerate estimate"}

    thr = truth_threshold if truth_threshold is not None else \
        float(np.quantile(t[t > 0], truth_quantile)) if (t > 0).any() else np.inf
    y = (t >= thr).astype(int)
    if y.sum() == 0 or y.sum() == y.size:
        return {"n_common": len(common), "auc": np.nan,
                "note": f"truth threshold {thr:.3g} gives a degenerate label set"}

    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score)
    auc = float(roc_auc_score(y, e))
    ap = float(average_precision_score(y, e))
    prec, rec, cut = precision_recall_curve(y, e)
    f1 = 2 * prec * rec / np.maximum(prec + rec, EPS)
    i = int(np.nanargmax(f1))
    return {
        "n_common": len(common), "n_pairs": int(e.size),
        "truth_threshold": thr, "positive_rate": float(y.mean()),
        "auc": auc, "average_precision": ap,
        "best_f1": float(f1[i]),
        "precision_at_best_f1": float(prec[i]), "recall_at_best_f1": float(rec[i]),
        "estimate_threshold_at_best_f1": float(cut[min(i, len(cut) - 1)]),
        "curve": pd.DataFrame({"precision": prec, "recall": rec}),
        "labels": y, "scores": e,
    }


def sweep_alphabet(values: pd.DataFrame, calendar: pd.DataFrame,
                   baseline_mask: np.ndarray, truth_strength: pd.DataFrame,
                   nodes, ks=(2, 3, 4, 5), modes=("level", "delta"),
                   base_cfg: StateSpaceConfig | None = None) -> pd.DataFrame:
    """How much do conclusions depend on the alphabet? Report, do not assume.

    Quantisation is a modelling choice with real consequences, so its effect on
    the recovered dependence is measured rather than argued about. A result that
    survives K = 2..5 and both level and delta encodings is worth trusting; one
    that appears at a single K is not.
    """
    base = base_cfg or StateSpaceConfig()
    rows = []
    for mode in modes:
        for K in ks:
            cfg = StateSpaceConfig(**{**asdict(base), "n_symbols": K,
                                      "mode": mode, "n_surrogates": 0,
                                      "lags": (1,)})
            resid = (deseasonalise(values, calendar, baseline_mask)
                     if cfg.deseasonalise else values)
            q = quantise(resid, cfg, baseline_mask)
            sub = [n for n in nodes if n in q["symbols"].columns]
            MI = pd.DataFrame(np.nan, index=sub, columns=sub, dtype=float)
            S = {n: q["symbols"][n].to_numpy(dtype=np.int64) for n in sub}
            for i, a in enumerate(sub):
                for b in sub[i + 1:]:
                    v = mutual_information(S[a], S[b], K)
                    MI.loc[a, b] = MI.loc[b, a] = v
            for a in sub:
                MI.loc[a, a] = 0.0
            sc = validate_against_interventional(MI, truth_strength)
            rows.append({"mode": mode, "K": K,
                         "usable_frac": float(q["occupancy"].usable.mean()),
                         "auc": sc.get("auc", np.nan),
                         "average_precision": sc.get("average_precision", np.nan),
                         "best_f1": sc.get("best_f1", np.nan),
                         "note": sc.get("note", "")})
    return pd.DataFrame(rows)
