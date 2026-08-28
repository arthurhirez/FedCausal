"""Personalization clustering: the two channels do their jobs, coupling
discounts rather than groups."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedwater.pipelines.personalization.nodes import (
    build_personalization_clusters,
)

FL = {"personalization": {"domain_weight": 0.5, "coupling_penalty": 0.5,
                          "coupling_q_threshold": 0.05}}


def _proto_hist(domain_of, n_months=12, dim=8, common_amp=0.0, seed=0):
    """Clients with a domain vector (drives their prototypes); optional
    shared common mode = hydraulic spillover."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, common_amp, (n_months, dim)), axis=0)
    rows = []
    dirs = {d: rng.normal(size=dim) for d in sorted(set(domain_of.values()))}
    for c, dom in domain_of.items():
        base = rng.normal(size=dim)
        for m in range(n_months):
            p = base + common[m] + (m / n_months) * dirs[dom] * 3 \
                + 0.05 * rng.normal(size=dim)
            rows.append(dict(round=0, client=c, month=m,
                             **{f"f{i}": v for i, v in enumerate(p)}))
    return pd.DataFrame(rows)


def _drift_signals(domain_of, n_months=12, seed=1):
    """delta_first trajectories that share SHAPE within a domain group."""
    rng = np.random.default_rng(seed)
    shapes = {d: np.sort(rng.uniform(0, 1, n_months))
              for d in sorted(set(domain_of.values()))}
    rows = []
    for c, dom in domain_of.items():
        traj = shapes[dom] + 0.02 * rng.normal(size=n_months)
        for m in range(n_months):
            rows.append(dict(client=c, month=m, delta_first=traj[m],
                             delta_roll=np.nan))
    return pd.DataFrame(rows)


def _dep(pairs_coupling, clients):
    rows = []
    for i, a in enumerate(clients):
        for b in clients[i + 1:]:
            w = pairs_coupling.get((a, b), 0.0)
            rows.append(dict(level="T", kind="aer_latent", district_a=a,
                             district_b=b, method="partial_rv", statistic=w,
                             p_value=0.001, q_value=0.001 if w > 0 else 0.5,
                             comm_floats_per_client=1, extra=np.nan))
    return pd.DataFrame(rows)


def test_domain_groups_recovered():
    """Two domain groups -> two clusters, on deconfounded + drift channels."""
    domain = {"A": "x", "B": "x", "C": "y", "D": "y"}
    clients = list(domain)
    assign, sim = build_personalization_clusters(
        _proto_hist(domain), _drift_signals(domain),
        _dep({}, clients), FL, seed=0)
    lab = assign.set_index("client")["cluster"]
    assert lab["A"] == lab["B"] and lab["C"] == lab["D"]
    assert lab["A"] != lab["C"]
    # same-domain pairs are more similar than cross-domain
    s = sim.set_index(["client_a", "client_b"])["s_fused"]
    assert s[("A", "B")] > s[("A", "C")]


def test_coupling_discounts_similarity():
    """A pure hydraulic-coupling pair (no domain similarity) must have its
    effective similarity discounted below its raw fused value."""
    domain = {"A": "x", "B": "y", "C": "z"}      # all different domains
    clients = list(domain)
    assign, sim = build_personalization_clusters(
        _proto_hist(domain), _drift_signals(domain),
        _dep({("A", "B"): 0.9}, clients), FL, seed=0)
    row = sim.set_index(["client_a", "client_b"]).loc[("A", "B")]
    assert row["s_effective"] < row["s_fused"]    # coupling discounted it
    assert row["coupling"] > 0.5


def test_spillover_does_not_create_false_grouping():
    """Independent domains + STRONG common mode: the deconfounded channel
    must not fuse everyone into one cluster purely from spillover."""
    domain = {"A": "w", "B": "x", "C": "y", "D": "z"}   # all distinct
    clients = list(domain)
    # strong common mode in prototypes, but coupling is quantified & penalized
    protos = _proto_hist(domain, common_amp=1.5)
    coup = {(a, b): 0.8 for i, a in enumerate(clients) for b in clients[i+1:]}
    assign, sim = build_personalization_clusters(
        protos, _drift_signals(domain), _dep(coup, clients), FL, seed=0)
    # not all identical: more than one cluster survives the spillover
    assert assign["cluster"].nunique() >= 2


def test_every_client_assigned_and_audit_complete():
    domain = {"A": "x", "B": "x", "C": "y"}
    clients = list(domain)
    assign, sim = build_personalization_clusters(
        _proto_hist(domain), _drift_signals(domain), _dep({}, clients),
        FL, seed=0)
    assert set(assign["client"]) == set(clients)
    assert not assign["cluster"].isna().any()
    assert set(sim.columns) >= {"s_drift", "s_domain", "s_fused",
                                "coupling", "s_effective"}
