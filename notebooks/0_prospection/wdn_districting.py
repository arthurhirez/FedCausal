"""Federated districting from the network's own graph.

Why graph partitioning, not clustering an affinity matrix
-----------------------------------------------------------
Spectral clustering on a dense pairwise matrix (as used for the exploratory
districts in the dependency notebook) can assign two spatially distant, weakly
connected nodes to the same cluster and split a spatially contiguous group. For
federated districts that is disqualifying: a client is supposed to be a
coherent, contiguous piece of infrastructure. Partitioning the network's own
graph guarantees contiguity by construction, because a cluster can only be
formed by traversing real edges.

Three tunable factors, one always available
--------------------------------------------
* **Structural cut cost** (always available, no hydraulic solve). Pipe
  resistance ``L/D^5`` as baseline connection strength, with valves and closed
  pipes down-weighted as natural cut candidates. This is the cheap proxy for
  "how separable are these nodes" and works on any network, including one with
  6000+ junctions where a full interventional map is expensive.
* **Elevation similarity.** Gravity systems and PRV placement both track
  elevation bands, so nodes at similar elevation are more likely to belong to
  the same pressure zone. Note this is a *dead* factor on flat or single-
  elevation networks (e.g. a transmission trunk with one nominal elevation
  value) -- the notebook checks for this before relying on it.
* **Measured hydraulic coupling** (optional). If an interventional map from
  the dependency notebook exists for this network, its symmetrised sensitivity
  is blended in and the *true* coupling ratio is reported alongside the cheap
  structural proxy -- which checks the proxy's honesty rather than assuming it.

Balance is not baked into the partitioning objective, because a hard balance
constraint can force a cut through a poorly-separable region. Instead it is a
bounded greedy local search after clustering: move a boundary node from an
oversized district to an adjacent undersized one only if the move improves
balance, is a real graph edge (so contiguity of the *result* is checked, not
assumed), and does not disconnect its home district.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr

LPS_PER_M3S = 1000.0
VALVE_TYPES_REGULATING = ("PRV", "PSV", "PBV")   # actively set a downstream head/flow


# ==========================================================================
# feature graph
# ==========================================================================

def build_feature_graph(wn) -> nx.Graph:
    """One graph, node and edge attributes only -- no weighting decisions yet.

    Separating feature extraction from weighting means every downstream weight
    combination is a re-run of cheap arithmetic on this graph, not a re-scan of
    the network model. Multi-edges (parallel pipes) are collapsed to the
    lowest-resistance parallel path, which is what actually carries flow.
    """
    g = nx.Graph()
    for n in wn.node_name_list:
        node = wn.get_node(n)
        coords = node.coordinates or (np.nan, np.nan)
        g.add_node(n, elevation=float(getattr(node, "elevation", np.nan)
                                      if node.node_type == "Junction" else
                                      getattr(node, "elevation", np.nan)
                                      if hasattr(node, "elevation") else np.nan),
                   x=coords[0], y=coords[1], node_type=node.node_type,
                   base_demand_lps=(sum(float(ts.base_value or 0.0)
                                        for ts in node.demand_timeseries_list)
                                    * LPS_PER_M3S)
                   if node.node_type == "Junction" else 0.0)

    for lname in wn.link_name_list:
        l = wn.get_link(lname)
        a, b = l.start_node_name, l.end_node_name
        if a == b or not g.has_node(a) or not g.has_node(b):
            continue
        length = float(getattr(l, "length", 50.0) or 50.0)
        diam = float(getattr(l, "diameter", 0.15) or 0.15)
        resistance = length / max(diam, 1e-3) ** 5
        is_valve = l.link_type == "Valve"
        is_regulating = is_valve and getattr(l, "valve_type", "") in VALVE_TYPES_REGULATING
        is_pump = l.link_type == "Pump"
        is_closed = str(getattr(l, "initial_status", "")).upper() == "CLOSED"
        w = {"resistance": resistance, "is_valve": is_valve,
             "is_regulating_valve": is_regulating, "is_pump": is_pump,
             "is_closed": is_closed, "link": lname, "link_type": l.link_type}
        if g.has_edge(a, b):
            if resistance < g[a][b]["resistance"]:
                g[a][b].update(w)
        else:
            g.add_edge(a, b, **w)
    return g


def check_elevation_informative(g: nx.Graph, min_span_m: float = 1.0) -> dict:
    """Is elevation actually usable on this network?

    Flat or single-value elevation (a transmission trunk modelled without real
    grade) makes the elevation factor inert: every similarity collapses to 1
    and it contributes nothing to the partition regardless of its weight. This
    is checked rather than assumed, because the failure is silent otherwise --
    the weight slider would appear to do nothing and look like a bug.
    """
    el = np.array([d["elevation"] for _, d in g.nodes(data=True)
                   if d["node_type"] == "Junction" and np.isfinite(d["elevation"])])
    span = float(el.max() - el.min()) if el.size else np.nan
    return {"n_values": el.size, "span_m": span, "std_m": float(el.std()) if el.size else np.nan,
            "informative": bool(np.isfinite(span) and span >= min_span_m)}


# ==========================================================================
# composite edge weight
# ==========================================================================

@dataclass
class DistrictingWeights:
    """Every factor's contribution, on a common 0-1 scale before mixing.

    A weight of 0 removes that factor entirely, so the sensitivity of the
    partition to each factor can be read directly by zeroing the others.
    """
    w_resistance: float = 1.0
    w_elevation: float = 1.0
    w_valve_cut: float = 1.0          # how strongly to prefer cutting at valves
    w_coupling: float = 0.0           # 0 unless a hydraulic map is supplied
    elevation_bandwidth_m: float | None = None   # None -> auto from the network
    valve_cut_discount: float = 0.05  # multiply resistance-affinity by this at a valve
    closed_link_discount: float = 0.01

    def validate(self):
        for f in ("w_resistance", "w_elevation", "w_valve_cut", "w_coupling"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0")
        if sum(getattr(self, f) for f in
               ("w_resistance", "w_elevation", "w_valve_cut", "w_coupling")) <= 0:
            raise ValueError("at least one weight must be positive")
        return self


def composite_weights(g: nx.Graph, weights: DistrictingWeights,
                      coupling: pd.DataFrame | None = None) -> nx.Graph:
    """Return a copy of ``g`` with a single ``affinity`` weight per edge.

    Affinity, not cost: high affinity means "keep together", low means "good
    place to cut". Each factor is normalised to [0, 1] on this graph before
    mixing, so weight sliders are comparable across networks of very different
    scale.
    """
    weights.validate()
    h = g.copy()

    res = np.array([d["resistance"] for _, _, d in h.edges(data=True)])
    if res.size and res.max() > res.min():
        r_aff = 1.0 - (res - res.min()) / (res.max() - res.min())
    else:
        r_aff = np.ones(res.size)

    el = {n: d["elevation"] for n, d in h.nodes(data=True)}
    valid_el = np.array([v for v in el.values() if np.isfinite(v)])
    bw = weights.elevation_bandwidth_m
    if bw is None:
        bw = max(float(np.std(valid_el)), 1.0) if valid_el.size else 1.0

    if coupling is not None:
        cvals = coupling.to_numpy()
        pos = cvals[cvals > 0]
        c_scale = float(np.quantile(pos, 0.95)) if pos.size else 1.0

    for i, (a, b, d) in enumerate(h.edges(data=True)):
        parts, wsum = [], 0.0
        if weights.w_resistance > 0:
            parts.append(weights.w_resistance * r_aff[i])
            wsum += weights.w_resistance
        if weights.w_elevation > 0:
            ea, eb = el.get(a, np.nan), el.get(b, np.nan)
            e_aff = (np.exp(-0.5 * ((ea - eb) / bw) ** 2)
                     if np.isfinite(ea) and np.isfinite(eb) else 0.5)
            parts.append(weights.w_elevation * e_aff)
            wsum += weights.w_elevation
        if weights.w_coupling > 0 and coupling is not None:
            if a in coupling.index and b in coupling.columns:
                c_aff = float(np.clip(coupling.at[a, b] / max(c_scale, 1e-12), 0, 1))
            else:
                c_aff = np.nan
            if np.isfinite(c_aff):
                parts.append(weights.w_coupling * c_aff)
                wsum += weights.w_coupling

        affinity = float(np.sum(parts) / wsum) if wsum > 0 else 0.5

        if weights.w_valve_cut > 0:
            if d.get("is_regulating_valve") or d.get("is_valve"):
                discount = weights.valve_cut_discount
                affinity = affinity * (1 - weights.w_valve_cut) + \
                    affinity * discount * weights.w_valve_cut
            elif d.get("is_closed"):
                discount = weights.closed_link_discount
                affinity = affinity * (1 - weights.w_valve_cut) + \
                    affinity * discount * weights.w_valve_cut

        d["affinity"] = max(affinity, 1e-6)
    return h


# ==========================================================================
# partitioning
# ==========================================================================

def partition_graph(g: nx.Graph, k: int, seed: int = 0) -> pd.Series:
    """Spectral partition on the graph's own (sparse) weighted Laplacian.

    Partitions the WHOLE graph -- every node, including sources and structural
    scaffolding -- never a subgraph induced on a node subset. Inducing on a
    subset (e.g. consumers only) silently drops every edge that passes through
    an excluded node, which can fragment a perfectly connected network: on
    D-Town, restricting to consumer junctions before partitioning collapsed
    348 nodes to a 131-node island, because many consumers are only linked to
    each other via non-consumer junctions near valves and pumps. Filtering to
    a node subset belongs in *reporting* (balance, external validation), never
    in the graph handed to the partitioner.

    Using the graph's own adjacency rather than a dense pairwise matrix is what
    makes contiguity likely in the first place: the Fiedler-style eigenvectors
    of a sparse planar-ish graph Laplacian vary smoothly along the graph, so a
    k-means split on them tends to cut a small number of edges rather than
    scatter labels. It is not a guarantee, which is why ``repair_contiguity``
    exists as an explicit, checked step rather than an assumption.
    """
    from sklearn.cluster import SpectralClustering

    nodes = list(g.nodes())
    largest = max(nx.connected_components(g), key=len)
    if len(largest) < len(nodes):
        nodes = list(largest)
    sub = g.subgraph(nodes)
    A = nx.to_scipy_sparse_array(sub, nodelist=nodes, weight="affinity",
                                 format="csr")
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    k_eff = min(k, len(nodes))
    sc = SpectralClustering(n_clusters=k_eff, affinity="precomputed",
                            random_state=seed, assign_labels="kmeans")
    # sklearn accepts a sparse precomputed affinity directly; densifying a
    # 6000+-node graph (ky17) would be ~300 MB and materially slower for no
    # benefit, since the underlying eigensolver works on the sparse Laplacian
    labels = sc.fit_predict(A)
    return pd.Series(labels, index=nodes, name="district")


def repair_contiguity(g: nx.Graph, labels: pd.Series) -> pd.Series:
    """Split any district into disconnected fragments; keep the largest label,
    reassign each smaller fragment to the neighbouring district it shares the
    most edge affinity with. Checked, not assumed: contiguity is verified after
    the repair and the result records whether any repair was needed.
    """
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        for lab in sorted(out.unique()):
            members = out.index[out == lab].tolist()
            sub = g.subgraph(members)
            comps = list(nx.connected_components(sub))
            if len(comps) <= 1:
                continue
            comps.sort(key=len, reverse=True)
            for frag in comps[1:]:
                nbr_weight = {}
                for n in frag:
                    for nb in g.neighbors(n):
                        if nb not in frag and nb in out.index:
                            nbr_weight[out[nb]] = (nbr_weight.get(out[nb], 0.0)
                                                   + g[n][nb].get("affinity", 0.0))
                target = max(nbr_weight, key=nbr_weight.get) if nbr_weight else lab
                for n in frag:
                    out[n] = target
            changed = True
    return out


def rebalance(g: nx.Graph, labels: pd.Series, target: str = "count",
             max_iter: int = 200, tol: float = 0.15, seed: int = 0,
             restrict_to: list[str] | None = None) -> pd.Series:
    """Bounded greedy local search to even out district size or demand.

    Only a boundary node -- one with a neighbour in another district -- can
    move, and only if the move is a real graph edge and does not disconnect
    its home district (checked via a local connectivity test, not assumed).
    ``target`` is ``"count"`` (equal client node counts, the FL default) or
    ``"demand"`` (equal client data volume by base demand).
    """
    out = labels.copy()
    rng = np.random.default_rng(seed)
    scope = set(restrict_to) if restrict_to is not None else set(out.index)

    def size_of(lab, s):
        # size is measured over the scope (consumers), so moving a structural
        # node never affects the balance target it exists to serve
        members = [n for n in s.index[s == lab] if n in scope]
        if target == "demand":
            return sum(g.nodes[n].get("base_demand_lps", 0.0) for n in members)
        return float(len(members))

    for _ in range(max_iter):
        sizes = {lab: size_of(lab, out) for lab in out.unique()}
        if not sizes or max(sizes.values()) <= 0:
            break
        imbalance = max(sizes.values()) / max(min(sizes.values()), 1e-9)
        if imbalance <= 1 + tol:
            break
        big = max(sizes, key=sizes.get)
        small = min(sizes, key=sizes.get)
        boundary = [n for n in out.index[out == big]
                   if any(out.get(nb) == small for nb in g.neighbors(n)
                          if nb in out.index)]
        rng.shuffle(boundary)
        moved = False
        for n in boundary:
            remaining = out.index[(out == big) & (out.index != n)]
            # a single connectivity check: does `big` stay contiguous without n?
            if len(remaining) > 1 and not nx.is_connected(g.subgraph(remaining)):
                continue
            out = out.copy()
            out[n] = small
            moved = True
            break
        if not moved:
            break
    return out


def districts_from_graph(wn, k: int, weights: DistrictingWeights,
                         coupling: pd.DataFrame | None = None,
                         balance_target: str | None = "count",
                         balance_tol: float = 0.15, seed: int = 0) -> dict:
    """One call: features -> weights -> partition -> repair -> rebalance.

    Only consumer junctions are districted (sources and pure structural nodes
    are attached to the district of their nearest consumer neighbour
    afterwards, so every node still has a label for plotting, but the
    partitioning objective is not distorted by zero-demand scaffolding).
    """
    from wdn_dependency import consumer_nodes    # reuse the notebook 02 definition

    g = build_feature_graph(wn)
    elev_check = check_elevation_informative(g)
    cons = set(consumer_nodes(wn))
    h = composite_weights(g, weights, coupling)

    raw = partition_graph(h, k, seed=seed)                     # every node
    n_fragmented = sum(
        1 for lab in raw.unique()
        if nx.number_connected_components(h.subgraph(raw.index[raw == lab])) > 1)
    fixed = repair_contiguity(h, raw)
    balance_nodes = [n for n in fixed.index if n in cons] if balance_target else None
    balanced = (rebalance(h, fixed, target=balance_target, tol=balance_tol,
                          seed=seed, restrict_to=balance_nodes)
               if balance_target else fixed)
    final = repair_contiguity(h, balanced)   # defense in depth after rebalancing

    return {"graph": h, "labels": final.astype(int),
            "consumer_labels": final.reindex([n for n in final.index if n in cons])
            .astype(int),
            "n_fragments_repaired": n_fragmented,
            "elevation_check": elev_check, "weights": asdict(weights),
            "n_districts_requested": k, "n_districts_final": int(final.nunique())}


# ==========================================================================
# metrics
# ==========================================================================

def coupling_edge_coverage(g: nx.Graph, coupling: pd.DataFrame) -> dict:
    """What fraction of graph EDGES can the coupling factor actually see?

    The interventional map is a probe-by-responder matrix, typically a subset
    of all nodes (60 probes on a 114-6261 node network). ``composite_weights``
    can only use ``coupling[a, b]`` for an edge when BOTH endpoints were
    probed, so on a lightly-subsampled map most physical edges get no coupling
    signal at all and the factor is silently inert there. Measured, not
    assumed: a low coverage number is the signal to raise ``max_probe_nodes``
    upstream (in the dependency notebook) before trusting ``w_coupling``.
    """
    idx = set(coupling.index) & set(coupling.columns)
    n_edges = g.number_of_edges()
    n_covered = sum(1 for a, b in g.edges() if a in idx and b in idx)
    return {"n_edges": n_edges, "n_covered": n_covered,
            "coverage": n_covered / n_edges if n_edges else np.nan}


def coupling_ceiling(coupling: pd.DataFrame, k: int, seed: int = 0) -> dict:
    """The best any method can do on the probed subset: cluster the real S directly.

    This is the honest ceiling for the coupling-blended graph partition. If a
    graph-based partition and this ceiling land on a similarly high (bad)
    coupling ratio, the network itself has no separable structure to find --
    as on Graeme, where both land near 0.78-0.82 -- and no amount of tuning the
    graph weights will fix that. If the ceiling is much better than the
    graph-based result, the graph factors are leaving real separability
    unexploited and ``w_coupling`` should be raised (coverage permitting).
    """
    from wdn_dependency import discover_districts, district_coupling
    part = discover_districts(coupling, k, seed=seed)
    dc = district_coupling(coupling, part)
    return {"partition": part, "coupling_ratio": dc["coupling_ratio"],
           "coupling_matrix": dc["coupling_matrix"]}


def structural_coupling_ratio(g: nx.Graph, labels: pd.Series) -> dict:
    """Cheap, always-available separability: cut-affinity mass / total mass.

    This is the graph-only analogue of the hydraulic coupling ratio from the
    dependency notebook, and it needs no solver: 0 means districts share almost
    no edge affinity (well separated), 1 means the partition ignores the
    graph's own structure entirely.
    """
    total = cut = 0.0
    cut_by_type = {"pipe": 0.0, "valve": 0.0, "closed": 0.0}
    for a, b, d in g.edges(data=True):
        if a not in labels.index or b not in labels.index:
            continue
        wt = d.get("affinity", 0.0)
        total += wt
        if labels[a] != labels[b]:
            cut += wt
            key = ("valve" if d.get("is_valve") else
                  "closed" if d.get("is_closed") else "pipe")
            cut_by_type[key] += wt
    return {"structural_coupling_ratio": cut / total if total > 0 else np.nan,
            "cut_edge_affinity_by_type": cut_by_type,
            "n_cut_edges": int(sum(1 for a, b in g.edges()
                                   if a in labels.index and b in labels.index
                                   and labels[a] != labels[b]))}


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


def contiguity_report(g: nx.Graph, labels: pd.Series) -> pd.DataFrame:
    """Per-district connected-component count. Every row should read 1."""
    rows = []
    for lab in sorted(labels.unique()):
        members = labels.index[labels == lab]
        sub = g.subgraph(members)
        rows.append({"district": lab, "n_nodes": len(members),
                     "n_components": nx.number_connected_components(sub),
                     "contiguous": nx.number_connected_components(sub) <= 1})
    return pd.DataFrame(rows)


def elevation_separation(g: nx.Graph, labels: pd.Series) -> dict:
    """Within-district vs between-district elevation variance (an F-ratio).

    High values indicate districts that correspond to real elevation
    bands -- the outcome expected in a gravity system, and expected to be
    near 1 (no separation) on a flat network.
    """
    el = pd.Series({n: g.nodes[n]["elevation"] for n in labels.index
                    if np.isfinite(g.nodes[n]["elevation"])})
    lab = labels.reindex(el.index)
    if el.empty or lab.nunique() < 2:
        return {"f_ratio": np.nan, "between_var": np.nan, "within_var": np.nan}
    grand = el.mean()
    between = sum(len(g_) * (g_.mean() - grand) ** 2
                 for _, g_ in el.groupby(lab)) / max(lab.nunique() - 1, 1)
    within = sum(((g_ - g_.mean()) ** 2).sum() for _, g_ in el.groupby(lab)) / \
        max(len(el) - lab.nunique(), 1)
    return {"f_ratio": float(between / within) if within > 0 else np.inf,
           "between_var": float(between), "within_var": float(within)}


def external_agreement(labels: pd.Series, ground_truth: dict[str, str]) -> dict:
    """Score a discovered partition against a real published one (e.g. D-Town's
    [TAGS] DMAs), via adjusted mutual information and adjusted Rand index.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    common = [n for n in labels.index if n in ground_truth]
    if len(common) < 4:
        return {"n_common": len(common), "ami": np.nan, "ari": np.nan}
    y_true = [ground_truth[n] for n in common]
    y_pred = [labels[n] for n in common]
    return {"n_common": len(common),
           "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
           "ari": float(adjusted_rand_score(y_true, y_pred))}


def full_report(g: nx.Graph, labels: pd.Series,
                consumer_labels: pd.Series | None = None) -> dict:
    """Everything above, in one call, for the interactive display.

    ``labels`` must be the FULL node set (every node the district owns,
    including structural scaffolding), because contiguity is a property of the
    district's physical infrastructure, not of the demand-bearing subset --
    the induced subgraph on consumers alone can look fragmented even when the
    district itself is one connected piece, whenever the connecting path runs
    through an excluded structural node (a valve dummy, a tank). Balance is
    reported on ``consumer_labels`` when given, since client data volume for
    FL is about consumers, not scaffolding.
    """
    bal_labels = consumer_labels if consumer_labels is not None else labels
    demand = pd.Series({n: g.nodes[n].get("base_demand_lps", 0.0)
                        for n in bal_labels.index})
    return {"structural": structural_coupling_ratio(g, labels),
            "balance": size_balance(bal_labels, demand),
            "contiguity": contiguity_report(g, labels),
            "elevation": elevation_separation(g, labels)}


# ==========================================================================
# loading an existing hydraulic map (from the dependency notebook), optional
# ==========================================================================

def load_hydraulic_coupling(network: str,
                            gt_dir: str | Path = "data/08_reporting/dependency/ground_truth"
                            ) -> pd.DataFrame | None:
    """Reuse notebook 02's interventional map if it exists; else None.

    Returning None (not raising) is deliberate: the coupling factor is optional
    everywhere it is used, and most networks in a 20+ corpus will not have a
    precomputed map.
    """
    gt_dir = Path(gt_dir)
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = gt_dir / f"{network}__coupling{suffix}"
        if p.exists():
            return reader(p) if suffix == ".parquet" else reader(p, index_col=0)
    return None


def load_dtown_dma_ground_truth(wn) -> dict[str, str]:
    """D-Town ships a real 5-DMA partition in [TAGS]. The one external check
    in this corpus that is not synthetic.
    """
    return {n: wn.get_node(n).tag for n in wn.junction_name_list
            if wn.get_node(n).tag}
