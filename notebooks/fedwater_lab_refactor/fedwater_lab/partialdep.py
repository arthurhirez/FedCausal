"""Stage 5 -- what survives conditioning on the others.

The confound this project exists to fight is that most observed cross-client
similarity is shared hydraulics and shared season, not demand dependence. A
partial correlation states that directly: if the similarity between c and d
vanishes once the other clients are held fixed, what was being measured was
the shared path.

For one scalar feature (typically `amp`, which survives a per-client scaler)
arranged as `X` of shape (blocks x clients), with `R` the correlation matrix
and `Theta = (R + lam I)^-1`:

    pcorr(c, d) = -Theta_cd / sqrt(Theta_cc Theta_dd)

`lam` is a ridge on the correlation matrix, reported rather than tuned away:
with four clients and a few dozen blocks the matrix is small but the sample
is short, and a partial correlation whose SIGN flips inside a sensible
penalty range is not a finding. `shrinkage_path` is that check.

`condition_report` prints the two numbers that decide whether any of this is
estimable at all -- rows per parameter and the condition number of the
correlation matrix -- because the failure mode here is silent: a nearly
singular R produces large, confident, meaningless partial correlations. On
the stand-in, a panel whose shared factor dominates (condition number ~250)
INVERTS the known structure, while the same code recovers it cleanly once
the condition number is ~20. The threshold is therefore part of the method:
`estimable` requires condition number <= 30, and above 100 the partial
correlations should not be read at all.

`structural_contrast` scores marginal and partial similarity against the
network's own boundary structure (adjacent district pairs versus the rest).
That adjacency is GROUND TRUTH and is used for scoring only; it never enters
an estimator. The prediction that makes it informative: marginal similarity
should not separate adjacency much (everything is coupled through the supply
path), while partial similarity should, because conditioning removes the
shared path and leaves the direct one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["client_matrix", "condition_report", "partial_corr",
           "marginal_vs_partial", "shrinkage_path", "structural_contrast"]


def client_matrix(phi: pd.DataFrame, feature: str = "amp",
                  phase: str | None = None, seq_key: str | None = None
                  ) -> tuple[np.ndarray, list[str]]:
    """(rows x clients) for one feature, on the blocks every client has.

    With `seq_key` the calendar classes stay as separate rows instead of
    being averaged into their block, which multiplies the row count. As
    everywhere else, that buys estimation and not inference.
    """
    d = phi if phase is None or "phase" not in phi else phi[phi["phase"] == phase]
    index = ["bid"] + ([seq_key] if seq_key and seq_key in d.columns else [])
    wide = d.pivot_table(index=index, columns="client", values=feature)
    wide = wide.dropna(axis=0, how="any")
    return wide.to_numpy(float), list(wide.columns)


def condition_report(X: np.ndarray, clients: list[str]) -> pd.Series:
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    ev = np.linalg.eigvalsh(R)
    cond = float(ev.max() / max(ev.min(), 1e-12))
    verdict = ("ok" if cond <= 30 else
               "collinear -- read as direction only" if cond <= 100 else
               "near rank-1 -- partial correlations are not interpretable")
    return pd.Series({
        "n_rows": int(n), "n_clients": int(p),
        "rows_per_parameter": float(n / max(p, 1)),
        "cond_number": cond, "min_eigenvalue": float(ev.min()),
        "verdict": verdict,
        "estimable": bool(n >= 4 * p and ev.min() > 1e-6 and cond <= 30)})


def partial_corr(X: np.ndarray, lam: float = 0.1) -> np.ndarray:
    R = np.corrcoef(X, rowvar=False)
    T = np.linalg.inv(R + float(lam) * np.eye(len(R)))
    d = np.sqrt(np.outer(np.diag(T), np.diag(T)))
    P = -T / np.where(d > 0, d, 1.0)
    np.fill_diagonal(P, 1.0)
    return P


def marginal_vs_partial(X: np.ndarray, clients: list[str], lam: float = 0.1
                        ) -> pd.DataFrame:
    R = np.corrcoef(X, rowvar=False)
    P = partial_corr(X, lam)
    rows = []
    for i, a in enumerate(clients):
        for j, b in enumerate(clients):
            if j <= i:
                continue
            rows.append({"client_a": a, "client_b": b, "lam": float(lam),
                         "marginal_r": float(R[i, j]),
                         "partial_r": float(P[i, j]),
                         "drop": float(abs(R[i, j]) - abs(P[i, j]))})
    return pd.DataFrame(rows)


def shrinkage_path(X: np.ndarray, clients: list[str],
                   lams=(0.01, 0.05, 0.1, 0.2, 0.5)) -> pd.DataFrame:
    """Partial correlations across the penalty range, with sign stability."""
    frames = [marginal_vs_partial(X, clients, l) for l in lams]
    out = pd.concat(frames, ignore_index=True)
    st = (out.groupby(["client_a", "client_b"])["partial_r"]
          .agg(sign_stable=lambda s: bool(np.all(np.sign(s) == np.sign(s.iloc[0]))),
               min_partial="min", max_partial="max").reset_index())
    return out.merge(st, on=["client_a", "client_b"])


def structural_contrast(tab: pd.DataFrame, adjacency: pd.DataFrame
                        ) -> pd.DataFrame:
    """Do adjacent pairs score higher? One row per similarity column.

    `auc` is P(adjacent pair ranks above a non-adjacent one), ties at 0.5 --
    with a handful of pairs this is a direction, never an estimate, and the
    pair counts are returned next to it so it cannot be read without them.
    """
    if not len(adjacency):
        return pd.DataFrame()
    j = tab.merge(adjacency, on=["client_a", "client_b"], how="inner")
    rows = []
    for col in ("marginal_r", "partial_r"):
        if col not in j:
            continue
        a = j.loc[j["adjacent"], col].abs().to_numpy(float)
        b = j.loc[~j["adjacent"], col].abs().to_numpy(float)
        if not len(a) or not len(b):
            continue
        wins = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
        rows.append({"signal": col, "n_adjacent": len(a), "n_other": len(b),
                     "mean_adjacent": float(a.mean()),
                     "mean_other": float(b.mean()), "auc": float(wins)})
    return pd.DataFrame(rows)
