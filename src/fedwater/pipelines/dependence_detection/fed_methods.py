"""Federated dependence methods — pure functions.

Reuses the oracle battery's estimators (RV, trajectory dCor, lagged xcorr)
and adds the two pieces specific to this pipeline:

* ``partial_rv`` — RV coefficient between two trajectory matrices after
  regressing out ALL other clients' trajectories: separates direct coupling
  from coupling mediated by third clients (the latent-space analogue of the
  Tier-1 precision-matrix logic).
* ``bh_fdr`` — Benjamini-Hochberg q-values across the pair family.

Surrogate convention: circularly shift the rows (window order) of X only —
preserves each trajectory's autocorrelation, destroys every cross-relation
involving X, which is the null being tested.
"""
from __future__ import annotations

import numpy as np

from fedwater.pipelines.dependence_oracle import methods as M


def residual_projector(Z: np.ndarray) -> np.ndarray:
    """pinv(Z) for residualizing on confounder matrix Z (n x q), centered."""
    Zc = Z - Z.mean(0)
    return np.linalg.pinv(Zc)


def regress_out(X: np.ndarray, Zc_pinv: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Zc = Z - Z.mean(0)
    Xc = X - X.mean(0)
    return Xc - Zc @ (Zc_pinv @ Xc)


def partial_rv(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """RV(X, Y | Z): shared co-inertia not explained by the other clients."""
    pinvZ = residual_projector(Z)
    return M.rv_coefficient(regress_out(X, pinvZ, Z), regress_out(Y, pinvZ, Z))


def roll_pvalue(stat_fn, X, Y, n_surrogates: int, rng: np.random.Generator):
    """Observed statistic + row-circular-shift p-value (shifts X).

    ``stat_fn`` may return a scalar or a (statistic, extra) tuple; the
    p-value is computed on the statistic, the full observed value returned.
    """
    observed = stat_fn(X, Y)
    obs_val = observed[0] if isinstance(observed, tuple) else observed
    lo = max(1, len(X) // 10)
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shift = int(rng.integers(lo, len(X) - lo))
        s = stat_fn(np.roll(X, shift, axis=0), Y)
        null[i] = s[0] if isinstance(s, tuple) else s
    p = (1.0 + np.sum(np.abs(null) >= abs(obs_val))) / (n_surrogates + 1.0)
    return observed, float(p)


def first_pc(X: np.ndarray) -> np.ndarray:
    """Leading principal-component score series of a trajectory (client-local)."""
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[0]


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q
