"""DMA partitioning after Shekofteh, Yousefi-Khoshqalb & Piratla (2023).

Water Resources Management 37:5007-5022, "An Efficient Approach for Partitioning
Water Distribution Networks Using Multi-Objective Optimization and Graph Theory".

The paper's three phases, kept in that order and named after it:

1. **Graph characterisation + partitioning** (Sec. 2.1-2.2). The WDN becomes a
   simple graph -- every node, every link, no hydraulics -- and one of three
   community detection algorithms cuts it into DMAs: Girvan-Newman (edge
   betweenness), Fast Greedy (modularity), Short Random Walks (walktrap). All
   three are hierarchical, so the paper's "external stopping condition" (stop at
   the desired DMA count rather than at the algorithm's own optimum) is exactly a
   dendrogram cut at ``k``.
2. **DMA configuration** (Sec. 2.3). Type 1 DMAs hold a water source, Type 2 are
   adjacent to one, Type 3 are far. Single-link Type 2 and every Type 3 DMA must
   keep their shortest path to a source open; those pipes are pinned open and
   leave the decision set.
3. **Optimisation** (Sec. 2.4-2.5). Over the remaining boundary pipes, minimise
   the number of open pipes (NO, Eq. 1) and their total length (LO, Eq. 2)
   subject to minimum pressure and maximum velocity (Eq. 3). Open pipes get flow
   meters, closed ones get gate valves.

Exhaustive search instead of NSGA-II, and why it is exact
---------------------------------------------------------
Both objectives strictly increase with every pipe added to the open set, so for
open sets ``S`` subset ``T``, ``S`` dominates ``T``. Any feasible set holding a
feasible proper subset is therefore dominated, and the Pareto front is contained
in the *minimal* feasible sets. Enumerating open sets by ascending cardinality
and skipping every superset of an already-found feasible set visits all of them,
so the returned front is exact -- not an approximation of one. This needs only
that feasibility is a deterministic function of the open set (true of a
single-period EPANET solve) and that lengths are positive. It does NOT assume
feasibility is monotone: a superset is skipped because it is dominated, not
because it is presumed feasible.

The cost is ``sum_c C(n, c)`` up to the smallest feasible cardinality, which is
cheap when few pipes must stay open and expensive when many must. ``budget``
bounds it and the result records whether the search ran to completion, so a
truncated search is never mistaken for an exact front. ``optimize_nsga2`` keeps
the paper's own optimiser available; on any instance small enough to enumerate,
the exhaustive front is the oracle it should be checked against.
"""
from __future__ import annotations

import copy
import itertools
import os
import tempfile
import warnings
from dataclasses import dataclass
from math import comb
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import wntr

PARTITION_METHODS = ("girvan_newman", "fast_greedy", "walktrap")


# ==========================================================================
# 0. loading -- the corpus is not uniformly modern EPANET
# ==========================================================================

def modernize_inp(text: str) -> str:
    """Rewrite an EPANET-1 era ``.inp`` into the dialect wntr parses.

    Wolf-Cordera Ranch ships in the older format and wntr rejects it outright.
    Four incompatibilities, all mechanical: an unsupported ``SEGMENTS`` option; a
    ``[TANKS]`` section whose two-field rows (id, head) are what EPANET 2 calls
    reservoirs; ``[PUMPS]`` given as inline multi-point head curves rather than a
    ``HEAD`` reference; and ``[PIPES]`` rows that omit the minor-loss column when
    a status word follows. ``[REPORT]`` is dropped as cosmetic. Nothing here
    changes the hydraulics -- it is a transcription, and the node/link counts are
    asserted against the paper in the notebook.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out, sec, res_rows, pump_rows = [], None, [], []

    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            sec = s.upper()
            if sec in ("[TANKS]", "[PUMPS]", "[REPORT]", "[END]"):
                continue
            out.append(ln)
            continue
        if sec == "[REPORT]":
            continue
        if sec in ("[TANKS]", "[PUMPS]"):
            if s and not s.startswith(";"):
                (res_rows if sec == "[TANKS]" else pump_rows).append(s.split())
            continue
        if sec == "[OPTIONS]" and s.upper().startswith("SEGMENTS"):
            continue
        if sec == "[PIPES]" and s and not s.startswith(";"):
            f = s.split()
            if len(f) == 6:
                f += ["0.0", "Open"]
            elif len(f) == 7:
                try:
                    float(f[6])          # 7th field is a numeric minor loss
                    f += ["Open"]
                except ValueError:       # 7th field is a status word
                    f = f[:6] + ["0.0", f[6]]
            out.append("  " + "  ".join(f))
            continue
        out.append(ln)

    # two-field [TANKS] rows are fixed-head sources, i.e. reservoirs
    res = ["[RESERVOIRS]"] + ["  %s  %s" % (r[0], r[1]) for r in res_rows if len(r) == 2]
    tanks = ["[TANKS]"] + ["  " + "  ".join(r) for r in res_rows if len(r) != 2]
    # [PUMPS] old form: id n1 n2 H0 H1 Q1 H2 Q2 [H3 Q3] -> HEAD curve + [CURVES]
    pumps, curves = ["[PUMPS]"], ["[CURVES]"]
    for p in pump_rows:
        nums = [float(x) for x in p[3:]]
        cid = "C_%s" % p[0]
        pumps.append("  %s  %s  %s  HEAD  %s" % (p[0], p[1], p[2], cid))
        curves.append(" %s  0.0  %s" % (cid, nums[0]))
        for i in range(1, len(nums) - 1, 2):
            curves.append(" %s  %s  %s" % (cid, nums[i + 1], nums[i]))

    return "\n".join(out + [""] + res + tanks + pumps + [""] + curves + ["", "[END]", ""])


def load_network(path: str | Path) -> wntr.network.WaterNetworkModel:
    """Load an ``.inp``, falling back to :func:`modernize_inp` if wntr refuses it.

    The fallback is attempted only after a straight read has failed, so a modern
    file is never rewritten and the conversion can never silently alter a network
    that did not need it.
    """
    path = Path(path)
    try:
        return wntr.network.WaterNetworkModel(str(path))
    except Exception:
        pass
    converted = modernize_inp(path.read_text(encoding="utf-8", errors="replace"))
    tmp = Path(tempfile.gettempdir()) / ("modern_" + path.name)
    tmp.write_text(converted)
    return wntr.network.WaterNetworkModel(str(tmp))


# ==========================================================================
# 1. graph characterisation and partitioning (paper Sec. 2.1-2.2)
# ==========================================================================

def simple_graph(wn) -> nx.Graph:
    """The paper's Phase 1 graph: every node, every link, no hydraulic terms.

    "vertices represent consumption nodes, tanks, and reservoirs, while edges
    represent pipes, pumps, and valves" (Sec. 2.1). Taken literally -- no
    filtering to consumers, no resistance weighting, no elevation. The community
    detection acts on topology alone, which is what makes the DMA layout
    reusable when hydraulic criteria later change.
    """
    g = nx.Graph()
    g.add_nodes_from(wn.node_name_list)
    for name in wn.link_name_list:
        link = wn.get_link(name)
        if link.start_node_name != link.end_node_name:
            g.add_edge(link.start_node_name, link.end_node_name)
    return g


def partition(wn, k: int, method: str = "girvan_newman") -> pd.Series:
    """Cut the network into exactly ``k`` DMAs with one of the paper's algorithms.

    All three are agglomerative or divisive hierarchies, so igraph returns a
    dendrogram and ``as_clustering(n=k)`` is precisely the paper's external
    stopping condition: interrupt at the desired DMA count rather than at the
    algorithm's own modularity optimum, "allowing the comparison of these layouts
    to choose the best one based on the project requirements and budget".
    """
    if method not in PARTITION_METHODS:
        raise ValueError("method must be one of %s" % (PARTITION_METHODS,))
    g = simple_graph(wn)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    if len(comps) > 1:
        raise ValueError("graph is disconnected (%d components, sizes %s); "
                         "community detection assumes one network"
                         % (len(comps), [len(c) for c in comps[:5]]))

    nodes = list(wn.node_name_list)          # fixed order => deterministic ties
    idx = {n: i for i, n in enumerate(nodes)}
    graph = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in g.edges()])
    dendro = {"girvan_newman": lambda: graph.community_edge_betweenness(directed=False),
              "fast_greedy": lambda: graph.community_fastgreedy(),
              "walktrap": lambda: graph.community_walktrap(steps=4)}[method]()
    membership = dendro.as_clustering(n=k).membership
    return _name_dmas(pd.Series(membership, index=nodes, name="dma"), wn)


def partition_from_mapping(wn, mapping: dict[str, list[str]]) -> pd.Series:
    """Adopt an externally supplied partition (a ``districts.yml``) as DMAs.

    Lets an existing districting be pushed through the paper's Phase 2 and 3
    unchanged, so its boundary pipes, flow meters and gate valves are computed on
    exactly the same footing as the ones the paper's algorithms produce. Nodes
    absent from the mapping -- sources and structural scaffolding, which
    ``districts.yml`` deliberately keeps out of the junction partition -- are
    attached to the DMA of their nearest labelled node, since Phase 2 needs a
    label on every node to decide which links cross a boundary.
    """
    known = set(wn.node_name_list)
    lab = {n: d for d, members in mapping.items() for n in members if n in known}
    g = simple_graph(wn)
    missing = [n for n in wn.node_name_list if n not in lab]
    for n in missing:                         # nearest labelled node by hop count
        for layer in nx.bfs_layers(g, [n]):
            hit = sorted(m for m in layer if m in lab)
            if hit:
                lab[n] = lab[hit[0]]
                break
    return pd.Series({n: lab[n] for n in wn.node_name_list}, name="dma")


def _name_dmas(raw: pd.Series, wn) -> pd.Series:
    """Relabel to ``District_A``.. by descending junction count, deterministically."""
    juncs = set(wn.junction_name_list)
    order = sorted(raw.unique(),
                   key=lambda c: (-sum(1 for n in raw.index[raw == c] if n in juncs),
                                  str(min(raw.index[raw == c]))))
    letters = [chr(ord("A") + i) if i < 26 else str(i + 1) for i in range(len(order))]
    return raw.map(dict(zip(order, letters))).map("District_{}".format)


# ==========================================================================
# 2. DMA configuration (paper Sec. 2.3)
# ==========================================================================

def source_nodes(wn) -> list[str]:
    """Reservoirs and tanks -- the paper's "water sources" for Type 1 DMAs."""
    return list(wn.reservoir_name_list) + list(wn.tank_name_list)


def boundary_table(wn, labels: pd.Series) -> pd.DataFrame:
    """Every link whose endpoints fall in different DMAs.

    ``closable`` marks pipes only: a pump or a control valve on a boundary cannot
    be swapped for a gate valve without changing what the network is, so
    non-pipe boundary links are held open and simply counted as needing a flow
    meter. ``length_m`` is wntr's SI length and feeds Eq. (2) directly.
    """
    rows = []
    for name in wn.link_name_list:
        link = wn.get_link(name)
        a, b = link.start_node_name, link.end_node_name
        if a == b or labels[a] == labels[b]:
            continue
        rows.append({"link": name, "link_type": link.link_type,
                     "dma_a": min(labels[a], labels[b]),
                     "dma_b": max(labels[a], labels[b]),
                     "length_m": float(getattr(link, "length", 0.0) or 0.0),
                     "closable": link.link_type == "Pipe"})
    return pd.DataFrame(rows, columns=["link", "link_type", "dma_a", "dma_b",
                                       "length_m", "closable"])


def dma_types(wn, labels: pd.Series, boundary: pd.DataFrame) -> pd.DataFrame:
    """Classify each DMA as Type 1 / 2 / 3 exactly as Sec. 2.3 defines them.

    Type 1 holds a water source and supplies the network. Type 2 holds none but
    touches a Type 1; the paper splits it further by whether *one* pipe or
    several make that connection, because a single-link Type 2 DMA loses its
    entire supply if that one pipe is closed. Type 3 ("far") DMAs belong to
    neither. Type 3 and single-link Type 2 are the ones whose supply path must be
    pinned open.
    """
    src_dmas = {labels[s] for s in source_nodes(wn) if s in labels.index}
    adj = {d: {} for d in labels.unique()}
    for _, r in boundary.iterrows():
        adj[r.dma_a][r.dma_b] = adj[r.dma_a].get(r.dma_b, 0) + 1
        adj[r.dma_b][r.dma_a] = adj[r.dma_b].get(r.dma_a, 0) + 1

    juncs = set(wn.junction_name_list)
    rows = []
    for d in sorted(labels.unique()):
        links_to_src = sum(n for nb, n in adj[d].items() if nb in src_dmas)
        kind = (1 if d in src_dmas else
                2 if links_to_src > 0 else 3)
        rows.append({"dma": d,
                     "n_junctions": sum(1 for n in labels.index[labels == d] if n in juncs),
                     "type": kind,
                     "links_to_source_dma": links_to_src,
                     "vital": bool(kind == 3 or (kind == 2 and links_to_src == 1))})
    return pd.DataFrame(rows)


def forced_open_links(wn, labels: pd.Series, boundary: pd.DataFrame,
                      types: pd.DataFrame) -> list[str]:
    """Pipes that must stay open: supply paths for vital DMAs, plus non-pipes.

    The paper works on the DMA-level schematic (its Fig. 1), not the node graph:
    "the shortest path between them and the nearest water resource must be found,
    and the path must be kept open". So the search runs over a graph whose
    vertices are DMAs and whose edges are boundary pipes, shortest in hops to any
    Type 1 DMA. Where several pipes join the same DMA pair, the shortest is the
    one pinned open, which is the other half of the paper's rule that closure
    "prioritize[s] ... longer pipes over shorter ones" on the grounds that longer
    pipes are likelier to leak or break.
    """
    src_dmas = set(types.dma[types.type == 1])
    vital = list(types.dma[types.vital])
    if not src_dmas or not vital:
        return sorted(boundary.link[~boundary.closable])

    dma_g = nx.Graph()
    dma_g.add_nodes_from(labels.unique())
    for (a, b), grp in boundary.groupby(["dma_a", "dma_b"]):
        pick = grp.sort_values(["length_m", "link"]).iloc[0]   # shortest pipe wins
        dma_g.add_edge(a, b, link=pick.link)

    pinned = set(boundary.link[~boundary.closable])            # pumps/valves: not closable
    for d in vital:
        best = None
        for s in src_dmas:
            if nx.has_path(dma_g, d, s):
                path = nx.shortest_path(dma_g, d, s)
                if best is None or len(path) < len(best):
                    best = path
        if best:
            pinned |= {dma_g[u][v]["link"] for u, v in zip(best, best[1:])}
    return sorted(pinned)


# ==========================================================================
# 3. hydraulic criteria (paper Sec. 2.4)
# ==========================================================================

@dataclass
class HydraulicCriteria:
    """Eq. (3), plus an escape hatch for networks that fail it before any cut.

    ``mode="paper"`` is the constraint as written: every junction at or above
    ``min_pressure_m``, every pipe at or below ``max_velocity_ms``.

    ``mode="baseline"`` lowers the pressure floor (and raises the velocity cap) to
    whatever the intact network already achieves, whenever the intact network
    itself misses the stated threshold. It exists because Wolf-Cordera with every
    boundary pipe open reaches 10.10 m against the paper's own 12 m requirement,
    so the literal constraint admits no solution on that file while the paper
    still reports a layout for it. The requirement becomes "sectorisation may not
    make the network worse than it already is", and it is identical to the
    literal constraint whenever the intact network satisfies it.

    The relaxation is deliberately a single global floor rather than a per-node
    one. Per-node would pin the seven already-failing junctions to their exact
    current pressure, so a millimetre of drop anywhere would reject a layout --
    strictly conservative, but brittle enough that nothing is feasible below full
    cardinality and the enumeration loses every chance to prune.
    """
    min_pressure_m: float
    max_velocity_ms: float
    mode: str = "paper"                     # "paper" | "baseline"

    def thresholds(self, baseline: dict | None) -> tuple[float, float]:
        """The pressure floor and velocity cap actually enforced."""
        if self.mode != "baseline" or baseline is None:
            return self.min_pressure_m, self.max_velocity_ms
        return (min(self.min_pressure_m, float(baseline["pressure"].min())),
                max(self.max_velocity_ms, float(baseline["velocity"].max())))


def peak_demand_model(wn) -> wntr.network.WaterNetworkModel:
    """A single-period copy at each node's peak multiplier.

    "the design of DMAs should be based on critical scenarios, with the maximum
    demand selected from demand patterns" (Sec. 2.4), and for the already-
    sectorised benchmarks "a single-period analysis was conducted based on the
    maximum available demand pattern" (Sec. 3). Folding the peak multiplier into
    the base demand and clearing the pattern gives exactly that, and keeps every
    candidate layout to one solve.
    """
    out = copy.deepcopy(wn)
    out.options.time.duration = 0
    for j in out.junction_name_list:
        for ts in out.get_node(j).demand_timeseries_list:
            mult = (max(out.get_pattern(ts.pattern_name).multipliers)
                    if ts.pattern_name else 1.0)
            ts.base_value = float(ts.base_value or 0.0) * float(mult)
            ts.pattern_name = None
    return out


def _solve(wn) -> dict:
    """Single EPANET solve into a scratch directory, by absolute path.

    Deliberately does not ``chdir``. Changing the working directory is how the
    rest of the project drives EPANET and it is fine in a script, but in a
    notebook it is fragile: interrupt a long cell between the ``chdir`` and its
    restore and the kernel is left sitting in a temp directory that the context
    manager has already deleted, after which every later solve fails with EPANET
    error 302 (cannot open input file) for reasons that have nothing to do with
    the network. Passing an absolute ``file_prefix`` removes the dependency on
    the working directory entirely.
    """
    with tempfile.TemporaryDirectory() as scratch:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = wntr.sim.EpanetSimulator(wn).run_sim(
                file_prefix=str(Path(scratch) / "en"))
    p = res.node["pressure"].iloc[0][wn.junction_name_list]
    v = res.link["velocity"].iloc[0][wn.pipe_name_list]
    return {"pressure": p, "velocity": v}


def _connected_to_source(wn, closed: set[str]) -> bool:
    """Does every demand-bearing junction still reach a source? Checked first.

    A closure that islands a DMA does not make EPANET fail; it makes EPANET
    return nonsense for the isolated nodes, which a pressure test may or may not
    catch. This is a graph traversal costing microseconds against a ~50 ms solve,
    so it is the first gate rather than a post-hoc sanity check.
    """
    g = nx.Graph()
    g.add_nodes_from(wn.node_name_list)
    for name in wn.link_name_list:
        if name in closed:
            continue
        link = wn.get_link(name)
        if str(link.initial_status).upper() == "CLOSED":
            continue
        g.add_edge(link.start_node_name, link.end_node_name)
    reach = set()
    for s in source_nodes(wn):
        if s in g:
            reach |= nx.node_connected_component(g, s)
    demanders = [j for j in wn.junction_name_list
                 if sum(float(ts.base_value or 0.0)
                        for ts in wn.get_node(j).demand_timeseries_list) > 0]
    return all(j in reach for j in demanders)


def evaluate(wn, closed, criteria: HydraulicCriteria, baseline: dict | None = None) -> dict:
    """Feasibility of one boundary-pipe configuration, per Eq. (3)."""
    closed = set(closed)
    if not _connected_to_source(wn, closed):
        return {"feasible": False, "min_pressure_m": np.nan,
                "max_velocity_ms": np.nan, "reason": "islands a demand node"}

    saved = {}
    for name in closed:
        link = wn.get_link(name)
        saved[name] = link.initial_status
        link.initial_status = wntr.network.LinkStatus.Closed
    try:
        r = _solve(wn)
    except Exception as exc:
        # EPANET error 110 and friends: a closure left the network hydraulically
        # unbalanced. That is a verdict on the layout, not a bug to propagate --
        # an unsolvable configuration is exactly an infeasible one.
        return {"feasible": False, "min_pressure_m": np.nan,
                "max_velocity_ms": np.nan, "reason": "solver: %s" % exc}
    finally:
        for name, st in saved.items():
            wn.get_link(name).initial_status = st

    p, v = r["pressure"], r["velocity"]
    p_min, v_max = criteria.thresholds(baseline)
    ok_p, ok_v = bool((p >= p_min).all()), bool((v <= v_max).all())
    return {"feasible": ok_p and ok_v, "min_pressure_m": float(p.min()),
            "max_velocity_ms": float(v.max()),
            "reason": "" if ok_p and ok_v else
                      ("pressure" if not ok_p else "") + ("velocity" if not ok_v else "")}


# ==========================================================================
# 4. optimisation (paper Sec. 2.5)
# ==========================================================================

def _objectives(open_links, lengths) -> tuple[int, float]:
    """Eq. (1) NO and Eq. (2) LO over the open boundary pipes."""
    return len(open_links), float(sum(lengths[l] for l in open_links))


def _pareto(rows: list[dict]) -> list[dict]:
    keep = []
    for r in rows:
        if not any(o is not r and o["NO"] <= r["NO"] and o["LO"] <= r["LO"]
                   and (o["NO"] < r["NO"] or o["LO"] < r["LO"]) for o in rows):
            keep.append(r)
    return sorted(keep, key=lambda r: (r["NO"], r["LO"]))


def optimize_exhaustive(wn, boundary: pd.DataFrame, forced: list[str],
                        criteria: HydraulicCriteria, baseline: dict | None = None,
                        budget: int = 20000) -> dict:
    """Exact Pareto front by ascending-cardinality enumeration with superset pruning.

    See the module docstring for why this is exact rather than approximate. The
    only thing that can make it inexact is exhausting ``budget``, and that is
    reported as ``complete=False`` rather than absorbed silently.
    """
    lengths = dict(zip(boundary.link, boundary.length_m))
    forced = [l for l in forced if l in lengths]
    free = [l for l in boundary.link if l not in forced]
    n = len(free)

    feasible, evals, complete = [], 0, True
    for card in range(n + 1):
        if not complete:
            break
        for combo in itertools.combinations(range(n), card):
            s = frozenset(combo)
            if any(f <= s for f in (r["_set"] for r in feasible)):
                continue                                   # dominated by a subset
            if evals >= budget:
                complete = False
                break
            closed = [free[i] for i in range(n) if i not in s]
            res = evaluate(wn, closed, criteria, baseline)
            evals += 1
            if res["feasible"]:
                open_links = forced + [free[i] for i in sorted(s)]
                no, lo = _objectives(open_links, lengths)
                feasible.append({"_set": s, "NO": no, "LO": lo,
                                 "open_links": open_links,
                                 "closed_links": sorted(closed),
                                 "min_pressure_m": res["min_pressure_m"],
                                 "max_velocity_ms": res["max_velocity_ms"]})

    front = _pareto(feasible)
    for r in front:
        r.pop("_set", None)
    return {"front": front, "n_free": n, "n_forced": len(forced),
            "search_space": 2 ** n, "evaluations": evals, "complete": complete,
            "backend": "exhaustive"}


def optimize_nsga2(wn, boundary: pd.DataFrame, forced: list[str],
                   criteria: HydraulicCriteria, baseline: dict | None = None,
                   pop_size: int = 100, generations: int = 40,
                   p_crossover: float = 0.9, p_mutation: float = 0.2,
                   seed: int = 0) -> dict:
    """The paper's own optimiser: binary-coded NSGA-II with Pareto elitism.

    Kept so the paper's method can be run as stated and, on any instance small
    enough to enumerate, checked against the exact front rather than trusted.
    The paper's own settings (population 400, 2000 generations) are exposed but
    not the default: at ~0.08 s per EPANET solve that is roughly 17 hours for a
    17-bit search space, which the exhaustive backend settles exactly in minutes.
    Infeasible individuals are ranked behind every feasible one, which is the
    standard constrained-domination handling for Eq. (3).
    """
    rng = np.random.default_rng(seed)
    lengths = dict(zip(boundary.link, boundary.length_m))
    forced = [l for l in forced if l in lengths]
    free = [l for l in boundary.link if l not in forced]
    n = len(free)
    if n == 0:
        return {"front": [], "n_free": 0, "n_forced": len(forced),
                "evaluations": 0, "complete": True, "backend": "nsga2"}

    memo: dict[tuple, dict] = {}

    def score(gene) -> dict:
        key = tuple(int(b) for b in gene)
        if key not in memo:
            open_links = forced + [free[i] for i in range(n) if key[i]]
            closed = [free[i] for i in range(n) if not key[i]]
            res = evaluate(wn, closed, criteria, baseline)
            no, lo = _objectives(open_links, lengths)
            memo[key] = {"NO": no, "LO": lo, "feasible": res["feasible"],
                         "open_links": open_links, "closed_links": sorted(closed),
                         "min_pressure_m": res["min_pressure_m"],
                         "max_velocity_ms": res["max_velocity_ms"]}
        return memo[key]

    def dominates(a, b):
        if a["feasible"] != b["feasible"]:
            return a["feasible"]
        return (a["NO"] <= b["NO"] and a["LO"] <= b["LO"]
                and (a["NO"] < b["NO"] or a["LO"] < b["LO"]))

    def rank_and_crowd(pop):
        sc = [score(g) for g in pop]
        ranks = [sum(1 for o in sc if dominates(o, s)) for s in sc]
        crowd = np.zeros(len(pop))
        for obj in ("NO", "LO"):
            order = np.argsort([s[obj] for s in sc])
            crowd[order[0]] = crowd[order[-1]] = np.inf
            lo, hi = sc[order[0]][obj], sc[order[-1]][obj]
            span = max(hi - lo, 1e-9)
            for a, b, c in zip(order, order[1:], order[2:]):
                crowd[b] += (sc[c][obj] - sc[a][obj]) / span
        return sc, ranks, crowd

    pop = [rng.integers(0, 2, n) for _ in range(pop_size)]
    for _ in range(generations):
        sc, ranks, crowd = rank_and_crowd(pop)

        def tournament():
            i, j = rng.integers(0, len(pop), 2)
            better = (ranks[i] < ranks[j] or
                      (ranks[i] == ranks[j] and crowd[i] > crowd[j]))
            return pop[i if better else j]

        children = []
        while len(children) < pop_size:
            a, b = tournament(), tournament()
            if rng.random() < p_crossover and n > 1:
                cut = rng.integers(1, n)
                a, b = np.concatenate([a[:cut], b[cut:]]), np.concatenate([b[:cut], a[cut:]])
            for child in (a.copy(), b.copy()):
                flip = rng.random(n) < p_mutation / max(n, 1)
                child[flip] ^= 1
                children.append(child)
        merged = pop + children[:pop_size]
        sc, ranks, crowd = rank_and_crowd(merged)
        order = sorted(range(len(merged)), key=lambda i: (ranks[i], -crowd[i]))
        pop = [merged[i] for i in order[:pop_size]]

    rows = [dict(v) for v in memo.values() if v["feasible"]]
    for r in rows:
        r.pop("feasible", None)
    return {"front": _pareto(rows), "n_free": n, "n_forced": len(forced),
            "search_space": 2 ** n, "evaluations": len(memo), "complete": True,
            "backend": "nsga2"}


def optimize(wn, boundary, forced, criteria, baseline=None,
             backend: str = "auto", budget: int = 20000, **kw) -> dict:
    """Exhaustive when the search space fits the budget, NSGA-II otherwise."""
    n_free = sum(1 for l in boundary.link if l not in set(forced))
    if backend == "auto":
        need = sum(comb(n_free, c) for c in range(n_free + 1))
        backend = "exhaustive" if need <= budget else "nsga2"
    if backend == "exhaustive":
        return optimize_exhaustive(wn, boundary, forced, criteria, baseline, budget)
    return optimize_nsga2(wn, boundary, forced, criteria, baseline, **kw)


# ==========================================================================
# 5. one call, and the outputs
# ==========================================================================

def sectorize(wn, k: int, method: str = "girvan_newman",
              criteria: HydraulicCriteria | None = None,
              mapping: dict | None = None, backend: str = "auto",
              budget: int = 20000, force: bool = False, **kw) -> dict:
    """Phases 1-3 end to end for one (network, method, k).

    A pre-flight check on the intact network comes first, because the exhaustive
    search has one pathological case and this is it: with no feasible set
    anywhere, nothing prunes and the enumeration runs the full ``2^n_free``. On
    Wolf-Cordera under the paper's literal 12 m requirement that is 8192 solves
    to conclude what the single baseline solve already showed -- the intact
    network reaches 10.10 m, so sectorisation cannot possibly satisfy Eq. (3).
    The gate reports that in one solve instead of thousands.

    It is a default, not a theorem: closing a pipe can raise pressure somewhere
    by rerouting flow, so an infeasible baseline does not strictly prove every
    configuration infeasible. ``force=True`` runs the search regardless, and
    ``mode="baseline"`` is the constructive alternative.
    """
    labels = (partition_from_mapping(wn, mapping) if method == "external"
              else partition(wn, k, method))
    boundary = boundary_table(wn, labels)
    types = dma_types(wn, labels, boundary)
    forced = forced_open_links(wn, labels, boundary, types)

    peak = peak_demand_model(wn)
    base = _solve(peak)
    base_p, base_v = float(base["pressure"].min()), float(base["velocity"].max())
    result = {"labels": labels, "boundary": boundary, "types": types,
              "forced_open": forced, "method": method, "k": int(labels.nunique()),
              "baseline": {"min_pressure_m": base_p, "max_velocity_ms": base_v}}
    if criteria is None:
        return result
    result["criteria"] = criteria

    base_ok = base_p >= criteria.min_pressure_m and base_v <= criteria.max_velocity_ms
    result["baseline_feasible"] = bool(base_ok)
    if not base_ok and criteria.mode == "paper" and not force:
        result["optimization"] = {"front": [], "n_free": len(boundary) - len(forced),
                                  "n_forced": len(forced), "evaluations": 1,
                                  "complete": True, "backend": "skipped",
                                  "note": "intact network already violates Eq. (3): "
                                          "min pressure %.2f m vs %.1f m required"
                                          % (base_p, criteria.min_pressure_m)}
        result["status"] = _status_table(boundary, list(boundary.link))
        result["best"] = None
        return result

    opt = optimize(peak, boundary, forced, criteria, base, backend, budget, **kw)
    result["optimization"] = opt
    if opt["front"]:
        best = opt["front"][0]                       # fewest open pipes, Eq. (1) first
        result["status"] = _status_table(boundary, best["open_links"])
        result["best"] = best
    else:
        result["status"] = _status_table(boundary, list(boundary.link))
        result["best"] = None
    return result


def _status_table(boundary: pd.DataFrame, open_links) -> pd.DataFrame:
    """Boundary pipes with their device: open pipes take flow meters, closed ones
    gate valves (Sec. 2.3, and the paper's Fig. 2 legend)."""
    open_set = set(open_links)
    out = boundary.copy()
    out["status"] = np.where(out.link.isin(open_set), "open", "closed")
    out["device"] = np.where(out.link.isin(open_set), "flow meter", "gate valve")
    return out.sort_values(["dma_a", "dma_b", "link"]).reset_index(drop=True)


def summary_row(network: str, res: dict) -> dict:
    """One row in the paper's Table 2 shape, plus the balance numbers it omits."""
    counts = res["types"].n_junctions
    opt = res.get("optimization", {})
    best = res.get("best")
    crit = res.get("criteria")
    row = {"network": network, "method": res["method"], "n_DMAs": res["k"],
           "mode": getattr(crit, "mode", ""),
           "n_boundary": len(res["boundary"]),
           "n_forced_open": len(res["forced_open"]),
           "n_flow_meters": best["NO"] if best else np.nan,
           "n_gate_valves": (len(res["boundary"]) - best["NO"]) if best else np.nan,
           "open_length_m": round(best["LO"], 1) if best else np.nan,
           "min_junctions": int(counts.min()), "max_junctions": int(counts.max()),
           "size_imbalance": round(counts.max() / max(counts.min(), 1), 2),
           "min_pressure_m": round(best["min_pressure_m"], 2) if best else np.nan,
           "max_velocity_ms": round(best["max_velocity_ms"], 2) if best else np.nan,
           "baseline_min_pressure_m": round(res["baseline"]["min_pressure_m"], 2),
           "baseline_feasible": res.get("baseline_feasible", np.nan),
           "evaluations": opt.get("evaluations", np.nan),
           "exact": opt.get("complete", np.nan), "backend": opt.get("backend", "")}
    return row


def write_districts_yml(path, wn, labels: pd.Series, header: str = "") -> Path:
    """Write the project's ``districts.yml`` schema: junctions partitioned under
    ``districts``, tanks and reservoirs recorded under ``assets``.

    The split matters and is not cosmetic: ``districts`` is the simulation domain
    and must partition the junction set exactly, while a tank or reservoir listed
    there would break portfolio building and would make every tank feed read as
    an inter-district boundary pipe.
    """
    path = Path(path)
    juncs, assets = {}, {}
    for node, dma in labels.items():
        (juncs if node in set(wn.junction_name_list) else assets).setdefault(dma, []).append(node)
    src = set(source_nodes(wn))
    assets = {d: sorted(n for n in v if n in src) for d, v in assets.items()}

    lines = [l if l.startswith("#") else "# " + l for l in header.splitlines()] + [""]
    lines.append("districts:")
    for d in sorted(juncs):
        lines.append("  %s: [%s]" % (d, ", ".join("'%s'" % n for n in juncs[d])))
    lines += ["", "assets:"]
    for d in sorted(assets):
        if assets[d]:
            lines.append("  %s: [%s]" % (d, ", ".join("'%s'" % n for n in assets[d])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def agreement(labels: pd.Series, reference: dict[str, list[str]]) -> dict:
    """AMI / ARI of a discovered partition against an existing ``districts.yml``."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    ref = {n: d for d, members in reference.items() for n in members}
    common = [n for n in labels.index if n in ref]
    if len(common) < 4:
        return {"n_common": len(common), "ami": np.nan, "ari": np.nan}
    return {"n_common": len(common),
            "ami": round(float(adjusted_mutual_info_score(
                [ref[n] for n in common], [labels[n] for n in common])), 4),
            "ari": round(float(adjusted_rand_score(
                [ref[n] for n in common], [labels[n] for n in common])), 4)}