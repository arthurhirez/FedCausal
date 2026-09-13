"""Every way of cutting the network into districts, behind one signature.

    labels = partition_labels(wn, g, method, k, ...)   ->  pd.Series over ALL nodes

Four methods, two families, one contract
----------------------------------------
* ``spectral`` -- eigen-partition of the network's own weighted Laplacian under
  the composite affinity from :mod:`districting.weights`. The factors are dials.
* ``girvan_newman`` (edge betweenness), ``fast_greedy`` (modularity),
  ``walktrap`` (short random walks) -- community detection on bare topology,
  after Shekofteh et al. (2023). All three are hierarchical, so cutting the
  dendrogram at ``k`` is exactly the paper's "external stopping condition":
  stop at the DMA count the project wants rather than at the algorithm's own
  modularity optimum.
* ``external`` -- adopt an existing mapping (a ``districts.yml``, or a
  network's published DMA tags).

Every method labels the WHOLE node set, never a subgraph induced on consumers.
Inducing on a subset silently drops every edge that passes through an excluded
node, which fragments a perfectly connected network: on D-Town, restricting to
consumer junctions collapses 348 nodes to a 131-node island, because many
consumers reach each other only via zero-demand junctions near valves and
pumps. Filtering to a subset belongs in *reporting* (balance, agreement), never
in the graph handed to the partitioner.

Post-processing, and who gets it
--------------------------------
``repair_contiguity`` and ``rebalance`` are separate, explicit steps rather than
part of any method. Contiguity is *always measured* (see
:func:`districting.metrics.contiguity_report`); whether it is repaired is a
choice, and the default differs by family on purpose: on for ``spectral``,
where contiguous clients are the whole point, and off for the community methods,
so their published behaviour is reproduced rather than quietly improved. Both
defaults are overridable per run.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .graph import require_connected
from .weights import DistrictingWeights, composite_weights

COMMUNITY_METHODS = ("girvan_newman", "fast_greedy", "walktrap")
METHODS = ("spectral",) + COMMUNITY_METHODS + ("external",)

#: Post-processing defaults, per family. Contiguity repair and size rebalancing
#: are both ON for ``spectral`` -- contiguous, comparably-sized clients are what
#: that method exists to produce -- and OFF for the community methods and for an
#: adopted partition, which are reproduced as published rather than quietly
#: improved. Both are always overridable per run, and contiguity is *measured*
#: either way.
DEFAULT_REPAIR = {m: (m == "spectral") for m in METHODS}
DEFAULT_BALANCE = {m: ("count" if m == "spectral" else None) for m in METHODS}


# ==========================================================================
# the methods
# ==========================================================================

def partition_labels(wn, g: nx.Graph, method: str, k: int, *,
                     weights: DistrictingWeights | None = None,
                     coupling: pd.DataFrame | None = None,
                     mapping: dict[str, list[str]] | None = None,
                     seed: int = 0,
                     on_disconnected: str = "raise") -> dict:
    """Dispatch to one method and return labels plus what the method did.

    Returns ``{"labels", "graph", "components", "method", "k_requested"}``.
    ``graph`` is the affinity-weighted copy for ``spectral`` and the plain
    feature graph otherwise, so the caller always scores against the graph the
    partition was actually made on.
    """
    if method not in METHODS:
        raise ValueError("method must be one of %s, got %r" % (METHODS, method))

    work, comps = require_connected(g, on_disconnected)

    if method == "spectral":
        h = composite_weights(work, weights or DistrictingWeights(), coupling)
        raw = _spectral(h, k, seed=seed)
    elif method in COMMUNITY_METHODS:
        h = work
        raw = _community(work, k, method)
    else:
        if mapping is None:
            raise ValueError("method='external' needs a mapping={district: [nodes]}")
        h = work
        raw = _from_mapping(work, mapping)

    labels = name_districts(raw, wn)
    if comps.get("excluded"):
        # never silently dropped: excluded nodes are labelled and reported
        labels = pd.concat([labels, pd.Series("District_UNASSIGNED",
                                              index=comps["excluded"], name="district")])
    return {"labels": labels, "graph": h, "components": comps,
            "method": method, "k_requested": int(k)}


def _spectral(h: nx.Graph, k: int, seed: int = 0) -> pd.Series:
    """Spectral partition of the graph's own sparse weighted Laplacian.

    Using the graph's own adjacency rather than a dense pairwise matrix is what
    makes contiguity likely in the first place: Fiedler-style eigenvectors of a
    sparse, near-planar graph Laplacian vary smoothly *along the graph*, so a
    k-means split on them cuts few edges rather than scattering labels. Likely,
    not guaranteed -- which is why :func:`repair_contiguity` is an explicit,
    verified step and not an assumption.

    The affinity is handed to sklearn sparse. Densifying a 6000-node graph is
    ~300 MB for no benefit: the eigensolver works on the sparse Laplacian.
    """
    from sklearn.cluster import SpectralClustering

    nodes = list(h.nodes())
    A = nx.to_scipy_sparse_array(h, nodelist=nodes, weight="affinity", format="csr")
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    sc = SpectralClustering(n_clusters=min(k, len(nodes)), affinity="precomputed",
                            random_state=seed, assign_labels="kmeans")
    return pd.Series(sc.fit_predict(A), index=nodes, name="district")


def _community(g: nx.Graph, k: int, method: str) -> pd.Series:
    """Cut the dendrogram at exactly ``k`` communities (paper Sec. 2.1-2.2).

    Topology only -- no resistance, no elevation, no hydraulics. That is the
    paper's stated novelty: the DMA layout is decoupled from the hydraulic
    criteria, so the layout survives a change of criteria unchanged.

    Vertex and edge order is the network's own -- nodes in ``.inp`` declaration
    order, edges in link declaration order -- and that is load-bearing, not
    incidental. ``fast_greedy`` is a greedy agglomeration, so near-ties in the
    merge sequence are resolved by vertex index and route it to a different
    local optimum. On D-Town, re-indexing this same graph lexicographically
    yields a partition agreeing with this one at only AMI 0.78 while scoring
    essentially the same modularity (0.7714 vs 0.7729, a gap of 0.0016) -- so
    the two are equally good answers that disagree about a fifth of the network.
    Worse for anyone reading the comparison table, the lexical order collapses
    the gap between ``fast_greedy`` and
    ``girvan_newman`` from AMI 0.76 to 0.94, making two distinct methods look
    like one. ``girvan_newman`` and ``walktrap`` are unaffected: edge
    betweenness and random-walk distances leave far fewer near-ties to break.

    Pinning the order to the network's reproduces the reference implementation
    bit for bit. It is deterministic for a given ``.inp`` but NOT invariant to
    reordering that file, so where :func:`order_sensitivity` flags a method the
    partition should be read as one of several comparably modular answers rather
    than as the method's unique output.
    """
    import igraph as ig

    nodes = list(g.nodes())                      # nx preserves insertion order
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[a], idx[b]) for a, b in g.edges()]
    graph = ig.Graph(n=len(nodes), edges=edges)

    dendro = {"girvan_newman": lambda: graph.community_edge_betweenness(directed=False),
              "fast_greedy": lambda: graph.community_fastgreedy(),
              "walktrap": lambda: graph.community_walktrap(steps=4)}[method]()
    return pd.Series(dendro.as_clustering(n=k).membership, index=nodes, name="district")


def _from_mapping(g: nx.Graph, mapping: dict[str, list[str]]) -> pd.Series:
    """Adopt an external partition, attaching whatever it leaves out.

    ``districts.yml`` deliberately keeps sources and structural scaffolding out
    of the junction partition, but boundary-pipe accounting needs a label on
    *every* node to decide which links cross a boundary. Unlabelled nodes are
    attached to the nearest labelled node by hop count, ties broken by name.
    """
    lab = {n: d for d, members in mapping.items() for n in members if n in g}
    if not lab:
        raise ValueError("mapping labels none of this network's nodes")
    return _attach_unlabelled(g, pd.Series(lab, name="district"))


def _attach_unlabelled(g: nx.Graph, labels: pd.Series) -> pd.Series:
    out = dict(labels)
    for n in g.nodes():
        if n in out:
            continue
        for layer in nx.bfs_layers(g, [n]):
            hit = sorted(m for m in layer if m in labels.index)
            if hit:
                out[n] = labels[hit[0]]
                break
    return pd.Series({n: out[n] for n in g.nodes() if n in out}, name="district")


# ==========================================================================
# naming
# ==========================================================================

def name_districts(raw: pd.Series, wn) -> pd.Series:
    """Relabel to ``District_A``, ``District_B``, ... deterministically.

    Ordered by descending junction count, ties broken by the lexicographically
    smallest member name. One naming convention for every method, because
    ``districts.yml`` names districts and the integer labels the spectral
    method used to emit could not be written to it or compared against it
    without an ad-hoc translation at every call site.

    Already-named partitions (from ``external``) pass through unchanged, so
    adopting a ``districts.yml`` never silently renames the user's districts.
    """
    if raw.map(lambda v: isinstance(v, str) and v.startswith("District_")).all():
        return raw.rename("district")
    juncs = set(wn.junction_name_list)
    order = sorted(raw.unique(),
                   key=lambda c: (-sum(1 for n in raw.index[raw == c] if n in juncs),
                                  str(min(raw.index[raw == c]))))
    letters = [chr(ord("A") + i) if i < 26 else str(i + 1) for i in range(len(order))]
    return raw.map(dict(zip(order, letters))).map("District_{}".format).rename("district")


# ==========================================================================
# post-processing
# ==========================================================================

def order_sensitivity(g: nx.Graph, wn, k: int, method: str) -> dict:
    """Is this community partition a unique answer, or one of several tied ones?

    Re-indexes the same graph lexicographically instead of in declaration order
    and re-runs the method. The graph is identical, so every difference comes
    from how near-ties were broken. ``modularity_gap`` says how much that cost:
    a gap near zero means the two answers are genuinely equivalent and the
    method simply has no basis for preferring either.

    Reported rather than hidden because the consequence is easy to misread. On
    D-Town, ``fast_greedy`` under the two orders agrees with itself at only
    AMI ~0.78, and one of those orders makes it look nearly identical to
    ``girvan_newman`` while the other keeps them clearly distinct. Comparing
    methods on an order-sensitive instance without knowing that invites reading
    a tie-break as a finding about the methods.
    """
    import igraph as ig

    from .metrics import agreement, modularity as _modularity

    def run_order(nodes, sort_edges=False):
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[a], idx[b]) for a, b in g.edges()]
        if sort_edges:
            edges = sorted((min(u, v), max(u, v)) for u, v in edges)
        graph = ig.Graph(n=len(nodes), edges=edges)
        dendro = {"girvan_newman": lambda: graph.community_edge_betweenness(directed=False),
                  "fast_greedy": lambda: graph.community_fastgreedy(),
                  "walktrap": lambda: graph.community_walktrap(steps=4)}[method]()
        return pd.Series(dendro.as_clustering(n=k).membership, index=nodes, name="district")

    declared = run_order(list(g.nodes()))
    lexical = run_order(sorted(g.nodes()), sort_edges=True)
    ag = agreement(name_districts(declared, wn),
                   {n: "D%d" % v for n, v in lexical.items()})
    q_dec, q_lex = _modularity(g, declared), _modularity(g, lexical)
    return {"method": method, "ami_between_orders": ag["ami"],
            "modularity_declared": round(q_dec, 4), "modularity_lexical": round(q_lex, 4),
            "modularity_gap": round(abs(q_dec - q_lex), 5),
            "order_sensitive": bool(ag["ami"] < 0.999)}


def repair_contiguity(g: nx.Graph, labels: pd.Series) -> tuple[pd.Series, int]:
    """Reassign disconnected fragments; return the fixed labels and a count.

    A district that comes back in several pieces keeps its largest piece; every
    smaller fragment joins whichever neighbouring district it shares the most
    edge affinity with. Iterated to a fixed point, because absorbing a fragment
    can itself fragment the absorber.
    """
    out = labels.copy()
    n_repaired, changed = 0, True
    while changed:
        changed = False
        for lab in sorted(out.unique()):
            members = out.index[out == lab].tolist()
            comps = list(nx.connected_components(g.subgraph(members)))
            if len(comps) <= 1:
                continue
            comps.sort(key=len, reverse=True)
            for frag in comps[1:]:
                nbr = {}
                for n in frag:
                    for m in g.neighbors(n):
                        if m not in frag and m in out.index:
                            nbr[out[m]] = nbr.get(out[m], 0.0) + g[n][m].get("affinity", 1.0)
                target = max(nbr, key=nbr.get) if nbr else lab
                for n in frag:
                    out[n] = target
                n_repaired += 1
            changed = True
    return out, n_repaired


def rebalance(g: nx.Graph, labels: pd.Series, target: str = "count",
              max_iter: int = 200, tol: float = 0.15, seed: int = 0,
              scope: list[str] | None = None) -> tuple[pd.Series, dict]:
    """Bounded greedy local search to even out district size or demand.

    Balance is not baked into the partitioning objective on purpose: a hard
    balance constraint can force a cut straight through a poorly separable
    region, which is a worse partition that merely looks tidier. It is a
    bounded repair afterwards instead -- and only a boundary node may move, only
    along a real graph edge, and only if its home district stays connected
    (checked, not assumed).

    ``scope`` is the node set balance is *measured* over -- consumers, since FL
    client volume is about consumers, not scaffolding. Candidate moves are
    restricted to the same set: moving a structural node cannot change a size
    measured over consumers, so allowing it only burned iterations on no-ops
    and let the search stall before reaching the tolerance.

    Returns the labels and a small trace: ``{"imbalance_before", "imbalance_after",
    "n_moves", "reached_tol"}``. The trace matters because this search can
    legitimately stall -- every movable boundary node may be an articulation
    point of its own district -- and a stall that leaves the partition well
    outside ``tol`` must be visible in the report rather than mistaken for a
    tolerance that was met.
    """
    out = labels.copy()
    rng = np.random.default_rng(seed)
    in_scope = set(scope) if scope is not None else set(out.index)
    n_moves = 0

    def size_of(lab, s):
        members = [n for n in s.index[s == lab] if n in in_scope]
        if target == "demand":
            return float(sum(g.nodes[n].get("base_demand_lps", 0.0) for n in members))
        return float(len(members))

    def imbalance(s):
        sizes = {lab: size_of(lab, s) for lab in s.unique()}
        if not sizes or max(sizes.values()) <= 0:
            return np.nan, sizes
        return max(sizes.values()) / max(min(sizes.values()), 1e-9), sizes

    before, _ = imbalance(out)

    for _ in range(max_iter):
        imb, sizes = imbalance(out)
        if not np.isfinite(imb) or imb <= 1 + tol:
            break

        # Candidate donor/receiver pairs, largest gap first. Trying only
        # (largest, smallest) stalls the whole search whenever those two happen
        # not to touch -- which is common, since the biggest and smallest
        # districts are often at opposite ends of the network. The loop then
        # exited on the first non-adjacent pair and returned a partition still
        # far outside `tol`, reporting nothing about it. Walking the pairs by
        # descending gap keeps the move bounded and greedy but lets the search
        # continue past a pair that simply shares no edge.
        pairs = sorted(((sizes[a] - sizes[b], a, b)
                        for a in sizes for b in sizes if sizes[a] > sizes[b]),
                       key=lambda t: -t[0])

        # A node may leave its district only if the district stays connected
        # without it -- i.e. iff it is not an articulation point of the
        # district's induced subgraph. Testing that per candidate meant a fresh
        # graph traversal per candidate; one articulation-point pass per donor
        # district answers it for every candidate at once, for the same cost as
        # a single old test, and is the identical condition rather than an
        # approximation of it.
        cut_vertices: dict[str, set] = {}
        n_touching = n_touching_in_scope = 0
        moved = False
        for _gap, big, small in pairs:
            members = out.index[out == big]
            if len(members) <= 1:
                continue                       # never empty a district
            if big not in cut_vertices:
                cut_vertices[big] = set(nx.articulation_points(g.subgraph(members)))
            touching = [n for n in members
                        if any(out.get(m) == small
                               for m in g.neighbors(n) if m in out.index)]
            n_touching += len(touching)
            n_touching_in_scope += sum(1 for n in touching if n in in_scope)
            candidates = [n for n in touching
                          if n in in_scope and n not in cut_vertices[big]]
            if not candidates:
                continue
            rng.shuffle(candidates)
            out[candidates[0]] = small
            moved = True
            n_moves += 1
            break
        if not moved:
            break                       # genuinely stuck: no legal move anywhere

    after, _ = imbalance(out)
    # Why it stopped, in the terms that make it actionable. A cut that runs
    # through valve scaffolding puts NO in-scope node on any boundary, so the
    # search has nothing legal to move and no amount of iterations will help --
    # a different outcome from "every candidate is an articulation point", and
    # from "the tolerance was simply met".
    if after <= 1 + tol:
        reason = "reached tolerance"
    elif n_touching == 0:
        reason = "no node of either district touches the other"
    elif n_touching_in_scope == 0:
        reason = ("every boundary node is out of scope (the cut runs through "
                  "structural scaffolding, not consumers)")
    elif n_moves == 0:
        reason = "every movable boundary node is an articulation point of its own district"
    else:
        reason = "no further legal move after %d move(s)" % n_moves
    return out, {"imbalance_before": float(before), "imbalance_after": float(after),
                 "n_moves": n_moves, "reached_tol": bool(after <= 1 + tol),
                 "stall_reason": reason}
