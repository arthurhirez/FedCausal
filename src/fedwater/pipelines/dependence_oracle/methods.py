"""Statistical methods for the dependence battery — pure functions, no I/O.

Conventions
-----------
* Every method takes numpy arrays and returns a scalar statistic (plus
  auxiliaries where noted). Higher |statistic| = stronger dependence.
* Directional methods follow ``f(a, b)`` = "a acts on b". Surrogate
  significance always circularly shifts ``a`` (the putative source), which
  destroys cross-dependence while preserving each series' autocorrelation —
  the correct null for dependent-in-time data.
* p-values use the add-one estimator p = (1 + #{|null| >= |obs|}) / (n + 1).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------
# surrogate machinery
# --------------------------------------------------------------------------
def circular_shift_pvalue(stat_fn, a: np.ndarray, b: np.ndarray,
                          n_surrogates: int, rng: np.random.Generator,
                          min_shift_frac: float = 0.1):
    """Observed statistic + circular-shift surrogate p-value."""
    observed = stat_fn(a, b)
    obs_val = observed[0] if isinstance(observed, tuple) else observed
    n = len(a)
    lo = max(1, int(min_shift_frac * n))
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shift = int(rng.integers(lo, n - lo))
        s = stat_fn(np.roll(a, shift), b)
        null[i] = s[0] if isinstance(s, tuple) else s
    p = (1.0 + np.sum(np.abs(null) >= abs(obs_val))) / (n_surrogates + 1.0)
    return observed, float(p)


def contiguous(a: np.ndarray, n: int, offset_frac: float = 0.25) -> np.ndarray:
    """Central contiguous chunk — for lag-structure methods (Granger, CCM)."""
    if len(a) <= n:
        return a
    start = int(offset_frac * (len(a) - n))
    return a[start:start + n]


def strided(a: np.ndarray, n: int) -> np.ndarray:
    """Stride subsample — for distributional methods (MI, dCor)."""
    step = max(1, len(a) // n)
    return a[::step][:n]


# --------------------------------------------------------------------------
# Tier 1 — linear
# --------------------------------------------------------------------------
def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    return float(sps.spearmanr(a, b).statistic)


def max_lagged_xcorr(a, b, max_lag: int):
    """(signed r at best |r| lag, lead_lag). lead_lag > 0: a leads b."""
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.corrcoef(a[:len(a) - lag or None], b[lag:])[0, 1]
        else:
            r = np.corrcoef(a[-lag:], b[:lag])[0, 1]
        if abs(r) > abs(best_r):
            best_r, best_lag = float(r), lag
    return best_r, best_lag


def precision_partial_corr(X: np.ndarray) -> np.ndarray:
    """Partial correlation of each pair given ALL other columns.

    Gaussian-graphical-model view via the precision matrix: suppresses
    dependence mediated by third districts, keeping only direct coupling.
    Joint by construction, so significance is reported as NaN rather than
    with a pairwise surrogate (which would be inconsistent with the null).
    """
    X = (X - X.mean(0)) / X.std(0)
    prec = np.linalg.pinv(np.cov(X, rowvar=False))
    d = np.sqrt(np.diag(prec))
    partial = -prec / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    return partial


# --------------------------------------------------------------------------
# Tier 2 — nonlinear
# --------------------------------------------------------------------------
def distance_correlation(a, b) -> float:
    """Szekely-Rizzo distance correlation: 0 iff independent. O(n^2)."""
    def centered(x):
        d = np.abs(x[:, None] - x[None, :])
        return d - d.mean(0)[None, :] - d.mean(1)[:, None] + d.mean()
    A, B = centered(np.asarray(a, float)), centered(np.asarray(b, float))
    dcov2 = (A * B).mean()
    dvar_a, dvar_b = (A * A).mean(), (B * B).mean()
    denom = np.sqrt(dvar_a * dvar_b)
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 0 else 0.0


def mutual_information(a, b, n_neighbors: int = 4) -> float:
    from sklearn.feature_selection import mutual_info_regression
    return float(mutual_info_regression(
        a.reshape(-1, 1), b, n_neighbors=n_neighbors, random_state=0)[0])


# --------------------------------------------------------------------------
# Tier 3 — directional
# --------------------------------------------------------------------------
def granger_f(a, b, lags: int) -> float:
    """F-statistic for 'a Granger-causes b': do a's lags improve the
    autoregression of b? Plain OLS, restricted vs unrestricted RSS."""
    n = len(b) - lags
    y = b[lags:]
    B_lags = np.column_stack([b[lags - k:len(b) - k] for k in range(1, lags + 1)])
    A_lags = np.column_stack([a[lags - k:len(a) - k] for k in range(1, lags + 1)])
    ones = np.ones((n, 1))

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r)

    rss_r = rss(np.hstack([ones, B_lags]))
    rss_f = rss(np.hstack([ones, B_lags, A_lags]))
    dof = n - 2 * lags - 1
    if rss_f <= 0 or dof <= 0:
        return 0.0
    return float(((rss_r - rss_f) / lags) / (rss_f / dof))


def _delay_embed(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    return np.column_stack([x[i * tau:i * tau + n] for i in range(dim)])


def ccm_skill(a, b, embed_dim: int, tau: int, lib_sizes: list[int],
              n_neighbors: int, rng: np.random.Generator):
    """Convergent Cross Mapping: 'a causes b' => a is reconstructable from
    b's shadow manifold. Returns (skill at largest library, convergence =
    skill gain from small to large library). Sugihara et al. (2012).
    """
    from sklearn.neighbors import NearestNeighbors

    Mb = _delay_embed(b, embed_dim, tau)
    target = a[(embed_dim - 1) * tau:]
    skills = []
    for lib in lib_sizes:
        lib = min(lib, len(Mb) - 1)
        idx = rng.choice(len(Mb), size=lib, replace=False)
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, lib)).fit(Mb[idx])
        dist, nbr = nn.kneighbors(Mb)
        w = np.exp(-dist / np.maximum(dist[:, :1], 1e-12))
        w /= w.sum(1, keepdims=True)
        pred = (w * target[idx][nbr]).sum(1)
        skills.append(np.corrcoef(target, pred)[0, 1])
    return float(skills[-1]), float(skills[-1] - skills[0])


# --------------------------------------------------------------------------
# Tier 4 — representation space (feature trajectories)
# --------------------------------------------------------------------------
def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """RV coefficient: multivariate generalization of r^2 between two
    (n x p) configuration matrices. 0 = unrelated, 1 = identical shape."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Sxy = X.T @ Y
    Sxx = X.T @ X
    Syy = Y.T @ Y
    num = np.trace(Sxy @ Sxy.T)
    den = np.sqrt(np.trace(Sxx @ Sxx) * np.trace(Syy @ Syy))
    return float(num / den) if den > 0 else 0.0


def trajectory_dcor(X: np.ndarray, Y: np.ndarray, max_n: int = 1000) -> float:
    """Distance correlation between multivariate trajectories (row-subsampled)."""
    step = max(1, len(X) // max_n)
    Xs, Ys = X[::step][:max_n], Y[::step][:max_n]

    def centered(M):
        d = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
        return d - d.mean(0)[None, :] - d.mean(1)[:, None] + d.mean()
    A, B = centered(Xs), centered(Ys)
    dcov2 = (A * B).mean()
    den = np.sqrt((A * A).mean() * (B * B).mean())
    return float(np.sqrt(max(dcov2, 0.0) / den)) if den > 0 else 0.0
