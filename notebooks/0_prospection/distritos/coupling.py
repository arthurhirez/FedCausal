"""Optional: a measured hydraulic coupling matrix as a districting factor.

Strictly optional, and deliberately isolated in its own module. A coupling
matrix ``S[i, j] = d p_i / d d_j`` comes from an interventional sweep -- one
EPANET solve per probed node -- and most networks will not have one. Everything
here therefore degrades to ``None`` or ``NaN`` rather than raising, and nothing
in the rest of the package imports this module at load time.

The reason it exists is honesty about the cheap proxy. The structural coupling
ratio is a graph-only guess at separability; the true ratio computed from a
measured ``S`` is the thing it is guessing at. Where both are available they
should be reported together, because they can disagree badly -- on Graeme the
proxy reads ~0.05 ("looks separable") while the true ratio is ~0.78 ("is not
separable"), and the ceiling from clustering ``S`` directly is no better, which
says the network genuinely has no separable structure rather than that the
method needs tuning.

Two caveats that decide whether the factor does anything at all:

* **Probe coverage.** ``S`` is a probe-by-responder matrix, usually over a
  subset of nodes. An edge can use the coupling factor only when *both* its
  endpoints were probed, so on a lightly sampled map the factor is silently
  inert over most of the graph. :func:`edge_coverage` measures that.
* **Symmetry.** Instantaneous hydraulic coupling is undirected for a network of
  pipes (``S`` inverts a symmetric matrix), so measured asymmetry is numerical
  error or a genuinely directional element -- a PRV, a check valve, a pump.
  Symmetrising is therefore the better estimate of one physical coupling, not a
  loss of information.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def load_coupling(network: str,
                  gt_dir: str | Path = "data/08_reporting/dependency/ground_truth"
                  ) -> pd.DataFrame | None:
    """Load a precomputed coupling matrix for ``network``, or ``None``.

    Returning ``None`` rather than raising is deliberate: the coupling factor is
    optional everywhere it is used, and most networks will not have a map.
    """
    gt_dir = Path(gt_dir)
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = gt_dir / ("%s__coupling%s" % (network, suffix))
        if p.exists():
            return reader(p) if suffix == ".parquet" else reader(p, index_col=0)
    return None


def symmetrize(S: pd.DataFrame) -> pd.DataFrame:
    """``|S|`` restricted to probed-and-responding nodes, symmetrised, zero diagonal."""
    common = [n for n in S.columns if n in S.index]
    M = np.asarray(S.loc[common, common].to_numpy(), dtype=float)
    W = 0.5 * (np.abs(M) + np.abs(M).T)
    np.fill_diagonal(W, 0.0)
    return pd.DataFrame(W, index=common, columns=common)


def edge_coverage(g: nx.Graph, coupling: pd.DataFrame) -> dict:
    """Fraction of graph edges whose *both* endpoints were probed.

    Measured, not assumed: a low number is the signal to probe more nodes
    upstream before trusting ``w_coupling`` at all.
    """
    idx = set(coupling.index) & set(coupling.columns)
    n_edges = g.number_of_edges()
    n_covered = sum(1 for a, b in g.edges() if a in idx and b in idx)
    return {"n_edges": n_edges, "n_covered": n_covered,
            "coverage": n_covered / n_edges if n_edges else np.nan}


def district_coupling(S: pd.DataFrame, labels: pd.Series) -> dict:
    """Inter-district dependence: the federated coupling dial, measured.

    ``coupling_matrix[A, B]`` is the mean absolute pressure sensitivity of
    district A's nodes to demand in district B. ``coupling_ratio`` is
    off-diagonal mass over total mass: 0 means fully separable clients (a
    federated problem with no cross-client information), 1 means district labels
    carry no hydraulic meaning at all.
    """
    W = symmetrize(S)
    lab = labels.reindex(W.index).dropna()
    W = W.loc[lab.index, lab.index]
    groups = sorted(lab.unique())
    C = pd.DataFrame(0.0, index=groups, columns=groups, dtype=float)
    for a in groups:
        ia = lab[lab == a].index
        for b in groups:
            ib = lab[lab == b].index
            block = W.loc[ia, ib].to_numpy()
            if a == b and block.size > 1:
                m = block[~np.eye(len(ia), dtype=bool)]
                C.loc[a, b] = float(m.mean()) if m.size else 0.0
            else:
                C.loc[a, b] = float(block.mean()) if block.size else 0.0
    tot = float(C.to_numpy().sum())
    off = tot - float(np.trace(C.to_numpy()))
    return {"coupling_matrix": C, "coupling_ratio": off / tot if tot > 0 else np.nan,
            "sizes": lab.value_counts().sort_index()}


def coupling_ceiling(S: pd.DataFrame, k: int, seed: int = 0,
                     normalize: str = "log") -> dict:
    """The best any method could do: cluster the measured ``S`` directly.

    The honest ceiling for a coupling-blended partition. If a graph partition
    and this ceiling land on a similarly high (bad) ratio, the network has no
    separable structure to find and no amount of weight tuning will produce any.
    If the ceiling is much better, the graph factors are leaving real
    separability unexploited and ``w_coupling`` should be raised, coverage
    permitting.

    Influence spans orders of magnitude, so a log transform is applied by
    default; without it a handful of near-source nodes dominate the spectrum and
    the partition degenerates into one big cluster plus singletons.
    """
    from sklearn.cluster import SpectralClustering

    W = symmetrize(S)
    A = W.to_numpy().copy()
    if normalize == "log":
        pos = A[A > 0]
        A = np.log10(1.0 + A / (pos.min() if pos.size else 1.0))
    elif normalize == "rank":
        order = A.flatten().argsort().argsort().astype(float)
        A = (order / max(order.max(), 1)).reshape(A.shape)
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)

    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            random_state=seed, assign_labels="kmeans")
    part = pd.Series(sc.fit_predict(A), index=W.index, name="district")
    dc = district_coupling(S, part)
    return {"partition": part, "coupling_ratio": dc["coupling_ratio"],
            "coupling_matrix": dc["coupling_matrix"]}


def true_coupling_ratio(S: pd.DataFrame, labels: pd.Series) -> dict:
    """The measured inter-district coupling of a partition, restricted to probes."""
    common = [n for n in S.index if n in S.columns]
    lab = labels.reindex(common).dropna()
    if lab.nunique() < 2:
        return {"coupling_ratio": np.nan, "n_probed_nodes": int(len(lab))}
    dc = district_coupling(S, lab)
    return {"coupling_ratio": dc["coupling_ratio"], "n_probed_nodes": int(len(lab)),
            "coupling_matrix": dc["coupling_matrix"]}
