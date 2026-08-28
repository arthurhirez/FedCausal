"""Federated similarity -> personalization clusters.

The synthesis the whole project builds toward: combine *drift* (how each
client's domain moves) with *dependence* (how clients are hydraulically
coupled) into a client grouping for personalized federated models.

The scientific hazard, stated first because the design exists to avoid it:
dependence-induced co-movement is NOT domain similarity. Two clients on the
same trunk main look alike because water couples them, not because their
demand regimes are alike; clustering on raw prototype similarity would group
by plumbing and personalize the wrong thing. So similarity is built in two
explicitly separated channels (mirroring the Design-A/B "L1 detect ->
L2 deconfound" split):

* ``S_drift``      — cosine similarity of clients' drift SIGNATURES (their
  delta_first trajectories over months). Two clients are domain-similar if
  their domains move the same way over time.
* ``S_domain``    — the deconfounded similarity: prototype similarity with
  the shared common mode removed (C2 median residual, the same operator the
  corrector ladder validated), so hydraulic spillover no longer inflates it.

These are FUSED into domain similarity ``S = w * S_domain + (1-w) * S_drift``.
Dependence enters as a SEPARATE role — a quantified coupling weight, not a
similarity term:

* ``D_couple``    — partial-RV magnitude per pair (direct coupling, mediated
  paths already removed), q-gated. High coupling means the pair's apparent
  similarity is the LEAST trustworthy as domain evidence, so it *discounts*
  S rather than adding to it: ``S_eff = S * (1 - beta * D_couple)``. This is
  the one-pass deconfounding; the Design-B loop would iterate it.

Clustering runs on ``S_eff`` (FINCH if available — same routine the FPL
server already uses on prototypes — else a correlation-distance
agglomeration fallback). Output: cluster assignments = the personalization
groups, plus the full similarity decomposition for audit and the
``quantified dependence`` that gated it.

Everything here is server-side over artifacts the pipeline already produces;
no new client communication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cosine_sim_matrix(rows: dict[str, np.ndarray]) -> pd.DataFrame:
    clients = sorted(rows)
    M = np.vstack([rows[c] for c in clients])
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    S = (M / norm) @ (M / norm).T
    return pd.DataFrame(S, index=clients, columns=clients)


def _drift_signatures(drift_signals: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per client: delta_first trajectory over months (the domain-motion
    fingerprint), mean-centered so overall drift *magnitude* doesn't
    dominate over drift *shape*."""
    piv = drift_signals.pivot_table(index="client", columns="month",
                                    values="delta_first").fillna(0.0)
    arr = piv.to_numpy()
    arr = arr - arr.mean(axis=1, keepdims=True)
    return {c: arr[i] for i, c in enumerate(piv.index)}


def _deconfounded_prototypes(prototype_history: pd.DataFrame):
    """Final-round prototypes with the per-month MEDIAN (common mode)
    removed, then flattened per client. Same robust operator the corrector
    ladder validated: the median ignores the drifted minority, so what
    remains is each client's idiosyncratic domain, not the shared spillover.
    """
    fcols = [c for c in prototype_history.columns if c.startswith("f")]
    final = prototype_history[prototype_history["round"]
                              == prototype_history["round"].max()]
    protos = {c: g.sort_values("month")[fcols].to_numpy()
              for c, g in final.groupby("client")}
    clients = sorted(protos)
    stack = np.stack([protos[c] for c in clients])       # (K, M, D)
    common = np.median(stack, axis=0)                    # (M, D)
    return {c: (protos[c] - common).ravel() for c in clients}


def _coupling_matrix(client_dependence: pd.DataFrame, clients: list[str],
                     q_threshold: float) -> pd.DataFrame:
    """Quantified DIRECT coupling per pair: partial-RV magnitude (Level T),
    q-gated, normalized to [0, 1] by its own max. Mediated paths are already
    partialled out, so this is the coupling that genuinely threatens the
    domain-similarity reading."""
    D = pd.DataFrame(0.0, index=clients, columns=clients)
    prv = client_dependence[(client_dependence["level"] == "T")
                            & (client_dependence["method"] == "partial_rv")]
    vals = prv["statistic"].abs()
    scale = vals.max() if len(vals) and vals.max() > 0 else 1.0
    for r in prv.itertuples():
        if r.district_a not in D.index or r.district_b not in D.columns:
            continue
        w = abs(r.statistic) / scale if r.q_value < q_threshold else 0.0
        D.loc[r.district_a, r.district_b] = w
        D.loc[r.district_b, r.district_a] = w
    return D


def _cluster(S_eff: pd.DataFrame, seed: int) -> dict[str, int]:
    """Cluster on the effective-similarity matrix. FINCH on the similarity-
    derived features if available, else correlation-distance agglomeration."""
    clients = list(S_eff.index)
    if len(clients) <= 2:
        return {c: 0 for c in clients}
    try:
        from fedwater.pipelines.fl_training.finch import FINCH
        c, _, _ = FINCH(S_eff.to_numpy(), distance="cosine",
                        ensure_early_exit=False, verbose=False)
        labels = c[:, 0]
    except Exception:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        dist = 1.0 - S_eff.to_numpy()
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2
        Z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(Z, t=max(2, len(clients) // 2), criterion="maxclust")
    return {cl: int(lab) for cl, lab in zip(clients, labels)}


def build_personalization_clusters(prototype_history: pd.DataFrame,
                                   drift_signals: pd.DataFrame,
                                   client_dependence: pd.DataFrame,
                                   fl: dict, seed: int):
    """Fuse drift + dependence into personalization clusters.

    Returns
    -------
    cluster_assignments : DataFrame(client, cluster, mean_coupling)
    similarity_matrix   : tidy DataFrame(client_a, client_b, s_drift,
                          s_domain, s_fused, coupling, s_effective) — the
                          full audit trail of how each pair's similarity was
                          built and discounted.
    """
    cfg = fl["personalization"]

    S_drift = _cosine_sim_matrix(_drift_signatures(drift_signals))
    S_domain = _cosine_sim_matrix(_deconfounded_prototypes(prototype_history))
    clients = sorted(set(S_drift.index) & set(S_domain.index))
    S_drift = S_drift.loc[clients, clients]
    S_domain = S_domain.loc[clients, clients]

    w = cfg["domain_weight"]
    S_fused = w * S_domain + (1 - w) * S_drift

    D = _coupling_matrix(client_dependence, clients, cfg["coupling_q_threshold"])
    # coupling DISCOUNTS similarity as domain evidence (one-pass deconfound)
    S_eff = S_fused * (1.0 - cfg["coupling_penalty"] * D)
    np.fill_diagonal(S_eff.values, 1.0)

    assign = _cluster(S_eff, seed)
    rows = []
    for i, a in enumerate(clients):
        for b in clients[i + 1:]:
            rows.append(dict(client_a=a, client_b=b,
                             s_drift=float(S_drift.loc[a, b]),
                             s_domain=float(S_domain.loc[a, b]),
                             s_fused=float(S_fused.loc[a, b]),
                             coupling=float(D.loc[a, b]),
                             s_effective=float(S_eff.loc[a, b])))
    similarity = pd.DataFrame(rows)

    mean_coupling = D.sum(axis=1) / max(len(clients) - 1, 1)
    assignments = pd.DataFrame({
        "client": clients,
        "cluster": [assign[c] for c in clients],
        "mean_coupling": [float(mean_coupling[c]) for c in clients],
    })
    # sanity: partition is total and non-degenerate-by-construction is NOT
    # required (all-similar clients legitimately form one cluster), but every
    # client must be assigned.
    if assignments["cluster"].isna().any():
        raise AssertionError("Unassigned client in personalization clustering.")
    return assignments, similarity
