"""How good is a partition -- measured the same way whoever produced it.

Everything here takes ``(graph, labels)`` and nothing about the method, which is
what makes a spectral partition and a Girvan-Newman partition comparable at all.

The metrics fall into three groups:

**Post-conditions** -- contiguity, above all. It is *measured on every run*,
whether or not repair was enabled, because a district that arrives in three
pieces is not a client.

**Quality without a solver** -- balance (node count and demand), elevation
separation, and the structural coupling ratio. All cheap, all available on any
network, none of them requiring hydraulics.

**External validation** -- agreement against a real published partition. This is
the strongest check available but it exists for almost no network: it needs a
partition a human engineer actually chose. D-Town ships one in ``[TAGS]``.
Read :func:`agreement` before quoting an AMI anywhere.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


# ==========================================================================
# post-conditions
# ==========================================================================

def contiguity_report(g: nx.Graph, labels: pd.Series) -> pd.DataFrame:
    """Per-district connected-component count. Every row should read 1.

    ``labels`` must be the FULL node set, not just consumers: contiguity is a
    property of the district's physical infrastructure. The subgraph induced on
    consumers alone can look fragmented while the district is one connected
    piece, whenever the connecting path runs through an excluded structural node
    (a valve dummy, a tank).
    """
    rows = []
    for lab in sorted(labels.unique()):
        members = labels.index[labels == lab]
        n_comp = nx.number_connected_components(g.subgraph(members))
        rows.append({"district": lab, "n_nodes": len(members),
                     "n_components": n_comp, "contiguous": n_comp <= 1})
    return pd.DataFrame(rows)


# ==========================================================================
# quality, no solver required
# ==========================================================================

def size_balance(labels: pd.Series, sizes: pd.Series | None = None) -> dict:
    """Node-count and (if given) demand balance across districts."""
    counts = labels.value_counts()
    out = {"min_count": int(counts.min()), "max_count": int(counts.max()),
           "count_imbalance": float(counts.max() / max(counts.min(), 1))}
    if sizes is not None:
        s = sizes.groupby(labels).sum()
        out |= {"min_demand_lps": float(s.min()), "max_demand_lps": float(s.max()),
                "demand_imbalance": float(s.max() / max(s.min(), 1e-9))}
    return out


def structural_coupling_ratio(g: nx.Graph, labels: pd.Series,
                              weight: str = "affinity") -> dict:
    """Cut mass over total mass -- separability without a solver.

    The graph-only analogue of the hydraulic coupling ratio: 0 means districts
    share almost no edge weight (well separated), 1 means the partition ignores
    the graph's own structure entirely.

    Edges carry an ``affinity`` only when the spectral method built one, so a
    missing weight falls back to 1.0 and the measure degrades gracefully into an
    unweighted cut-edge fraction. That is the honest comparison for community
    methods, which never saw an affinity in the first place -- but it means the
    number is only comparable across methods when they share a weighting, so
    ``weight_used`` is reported alongside it.
    """
    weighted = any(weight in d for _, _, d in g.edges(data=True))
    total = cut = 0.0
    cut_by_type = {"pipe": 0.0, "valve": 0.0, "pump": 0.0, "closed": 0.0}
    n_cut = 0
    for a, b, d in g.edges(data=True):
        if a not in labels.index or b not in labels.index:
            continue
        wt = float(d.get(weight, 1.0)) if weighted else 1.0
        total += wt
        if labels[a] != labels[b]:
            cut += wt
            n_cut += 1
            key = ("valve" if d.get("is_valve") else "pump" if d.get("is_pump")
                   else "closed" if d.get("is_closed") else "pipe")
            cut_by_type[key] += wt
    return {"structural_coupling_ratio": cut / total if total > 0 else np.nan,
            "cut_edge_weight_by_type": cut_by_type, "n_cut_edges": n_cut,
            "weight_used": weight if weighted else "unweighted"}


def elevation_separation(g: nx.Graph, labels: pd.Series) -> dict:
    """Between- over within-district elevation variance (an F-ratio).

    High values mean districts correspond to real elevation bands -- expected in
    a gravity system, and expected near 1 (no separation) on a flat network.
    """
    el = pd.Series({n: g.nodes[n]["elevation"] for n in labels.index
                    if n in g and np.isfinite(g.nodes[n].get("elevation", np.nan))})
    lab = labels.reindex(el.index)
    if el.empty or lab.nunique() < 2:
        return {"f_ratio": np.nan, "between_var": np.nan, "within_var": np.nan}
    grand = el.mean()
    between = sum(len(grp) * (grp.mean() - grand) ** 2
                  for _, grp in el.groupby(lab)) / max(lab.nunique() - 1, 1)
    within = sum(((grp - grp.mean()) ** 2).sum()
                 for _, grp in el.groupby(lab)) / max(len(el) - lab.nunique(), 1)
    return {"f_ratio": float(between / within) if within > 0 else np.inf,
            "between_var": float(between), "within_var": float(within)}


def modularity(g: nx.Graph, labels: pd.Series) -> float:
    """Newman modularity of the partition on the bare topology.

    The community methods optimise (a hierarchy over) this, so it is their home
    ground and the spectral method should not be expected to win on it. Reported
    because it is the comparison the paper makes, not because it is the
    objective this project cares about.
    """
    common = [n for n in g.nodes() if n in labels.index]
    sub = g.subgraph(common)
    comms = [set(labels.index[labels == d]) & set(common) for d in sorted(labels.unique())]
    comms = [c for c in comms if c]
    if len(comms) < 2 or sub.number_of_edges() == 0:
        return np.nan
    return float(nx.community.modularity(sub, comms))


# ==========================================================================
# external validation
# ==========================================================================

def agreement(labels: pd.Series, reference) -> dict:
    """AMI and ARI against a reference partition.

    ``reference`` may be ``{node: district}`` or ``{district: [nodes]}``; both
    shapes existed in the code this replaces and both are accepted here.

    **This metric is only meaningful where the reference is a real partition a
    human engineer chose.** In this corpus that is D-Town, which ships its
    5-DMA utility decision in ``[TAGS]``. Scoring against a partition that some
    earlier run of this same code produced measures self-consistency, not
    correctness, and an AMI quoted from such a comparison means nothing. The
    returned dict carries ``n_common`` precisely so a comparison resting on a
    handful of shared nodes is visible rather than averaged away.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    if reference and isinstance(next(iter(reference.values())), (list, tuple, set)):
        reference = {n: d for d, members in reference.items() for n in members}

    common = [n for n in labels.index if n in reference]
    if len(common) < 4:
        return {"n_common": len(common), "ami": np.nan, "ari": np.nan}
    y_true = [reference[n] for n in common]
    y_pred = [labels[n] for n in common]
    return {"n_common": len(common),
            "ami": round(float(adjusted_mutual_info_score(y_true, y_pred)), 4),
            "ari": round(float(adjusted_rand_score(y_true, y_pred)), 4)}


def published_dma_tags(wn) -> dict[str, str]:
    """A network's own published DMA partition from ``[TAGS]``, if it has one.

    The one external, non-synthetic ground truth in this corpus. Returns an
    empty dict when the file carries no tags, so the caller can skip external
    validation rather than compare against nothing.
    """
    out = {}
    for n in wn.junction_name_list:
        tag = getattr(wn.get_node(n), "tag", None)
        if tag:
            out[n] = str(tag)
    return out


# ==========================================================================
# one call
# ==========================================================================

def full_report(g: nx.Graph, labels: pd.Series,
                consumer_labels: pd.Series | None = None,
                reference: dict | None = None) -> dict:
    """Every metric above in one call.

    ``labels`` is the full node set (contiguity is about infrastructure);
    balance is measured on ``consumer_labels`` when given, since FL client data
    volume is about consumers, not scaffolding.
    """
    bal = consumer_labels if consumer_labels is not None else labels
    demand = pd.Series({n: g.nodes[n].get("base_demand_lps", 0.0)
                        for n in bal.index if n in g})
    out = {"contiguity": contiguity_report(g, labels),
           "balance": size_balance(bal, demand),
           "structural": structural_coupling_ratio(g, labels),
           "elevation": elevation_separation(g, labels),
           "modularity": modularity(g, labels)}
    if reference:
        out["agreement"] = agreement(labels, reference)
    return out


def summary_row(network: str, result) -> dict:
    """One comparable row per (network, method), whatever the method was.

    Columns split into three blocks: what the partition *is* (districts, sizes,
    imbalance), what it *costs* (boundary links, meters, gate valves, open
    length), and how it *scores* (contiguity, modularity, coupling ratio,
    agreement). Fields that need an optimisation run come back NaN when it was
    skipped, never silently as zero -- "no meters" and "never counted the
    meters" are different statements.
    """
    rep, opt, best = result.report, result.optimization or {}, result.best
    counts = result.types.n_junctions if result.types is not None else pd.Series(dtype=float)
    n_boundary = len(result.boundary) if result.boundary is not None else np.nan

    row = {"network": network, "method": result.method,
           "k_requested": result.k_requested, "n_districts": result.k,
           "min_junctions": int(counts.min()) if len(counts) else np.nan,
           "max_junctions": int(counts.max()) if len(counts) else np.nan,
           "size_imbalance": round(rep["balance"]["count_imbalance"], 2),
           "demand_imbalance": round(rep["balance"].get("demand_imbalance", np.nan), 2),
           "all_contiguous": bool(rep["contiguity"].contiguous.all()),
           "n_fragments_repaired": result.n_fragments_repaired,
           "rebalance_moves": (result.balance_trace or {}).get("n_moves", np.nan),
           "rebalance_reached_tol": (result.balance_trace or {}).get("reached_tol", np.nan),
           "modularity": round(rep["modularity"], 4) if np.isfinite(rep["modularity"]) else np.nan,
           "structural_coupling_ratio": round(rep["structural"]["structural_coupling_ratio"], 4),
           "elevation_f_ratio": round(rep["elevation"]["f_ratio"], 2),
           "n_boundary_links": n_boundary,
           "n_forced_open": len(result.forced_open) if result.forced_open is not None else np.nan,
           "n_flow_meters": best["NO"] if best else np.nan,
           "n_gate_valves": (n_boundary - best["NO"]) if best else np.nan,
           "open_length_m": round(best["LO"], 1) if best else np.nan,
           "min_pressure_m": round(best["min_pressure_m"], 2) if best else np.nan,
           "max_velocity_ms": round(best["max_velocity_ms"], 2) if best else np.nan,
           "baseline_min_pressure_m": (round(result.baseline["min_pressure_m"], 2)
                                       if result.baseline else np.nan),
           "baseline_feasible": result.baseline_feasible,
           "evaluations": opt.get("evaluations", np.nan),
           "exact_front": opt.get("complete", np.nan),
           "backend": opt.get("backend", "none")}
    if "agreement" in rep:
        row |= {"ami_vs_reference": rep["agreement"]["ami"],
                "ari_vs_reference": rep["agreement"]["ari"]}
    return row
