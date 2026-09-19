"""What the partition costs in hardware: boundary pipes, meters and valves.

This is Phases 2 and 3 of Shekofteh et al. (2023), and it is deliberately
*method-agnostic*. Any labelling -- spectral, community-detected, or adopted
from an existing ``districts.yml`` -- goes through exactly the same boundary
accounting and the same optional optimisation, so the hardware cost of two
partitions is a comparison and not an apples-to-oranges guess.

Phase 2 (Sec. 2.3): classify each district. Type 1 holds a water source. Type 2
holds none but touches a Type 1 -- and a *single-link* Type 2 loses its entire
supply if that one pipe closes. Type 3 is far from any source. Type 3 and
single-link Type 2 are "vital": their shortest path to a source is pinned open
and leaves the decision set.

Phase 3 (Sec. 2.4-2.5): over the remaining boundary pipes, minimise the number
left open (NO, Eq. 1) and their total length (LO, Eq. 2), subject to minimum
junction pressure and maximum pipe velocity (Eq. 3). Open pipes get flow
meters; closed ones get gate valves.

Why exhaustive enumeration is exact, not an approximation
---------------------------------------------------------
Both objectives strictly increase with every pipe added to the open set, so for
open sets ``S`` subset ``T``, ``S`` dominates ``T``. Any feasible set containing
a feasible proper subset is therefore dominated, and the Pareto front lies
entirely among the *minimal* feasible sets. Enumerating by ascending cardinality
and skipping every superset of an already-found feasible set visits all of them,
so the returned front is exact. This needs only that feasibility is a
deterministic function of the open set (true of a single-period EPANET solve)
and that lengths are positive. It does NOT assume feasibility is monotone: a
superset is skipped because it is dominated, not because it is presumed feasible.

The cost is ``sum_c C(n, c)`` up to the smallest feasible cardinality -- cheap
when few pipes must stay open, expensive when many must. ``budget`` bounds it
and the result records ``complete``, so a truncated search is never mistaken for
an exact front. ``optimize_nsga2`` is the paper's own optimiser, kept available
and, on any instance small enough to enumerate, checkable against the exact
front rather than trusted.
"""
from __future__ import annotations

import copy
import itertools
import tempfile
import warnings
from dataclasses import dataclass
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr

from .graph import source_nodes

OPTIMIZER_BACKENDS = ("none", "auto", "exhaustive", "nsga2")


# ==========================================================================
# Phase 2 -- boundary pipes and district typing
# ==========================================================================

def boundary_table(wn, labels: pd.Series) -> pd.DataFrame:
    """Every link whose endpoints fall in different districts.

    ``closable`` marks pipes only: a pump or a control valve on a boundary
    cannot be swapped for a gate valve without changing what the network is, so
    non-pipe boundary links are held open and simply counted as needing a meter.
    ``length_m`` is wntr's SI length and feeds Eq. (2) directly.
    """
    rows = []
    for name in wn.link_name_list:
        link = wn.get_link(name)
        a, b = link.start_node_name, link.end_node_name
        if a == b or a not in labels.index or b not in labels.index:
            continue
        if labels[a] == labels[b]:
            continue
        rows.append({"link": name, "link_type": link.link_type,
                     "district_a": min(labels[a], labels[b]),
                     "district_b": max(labels[a], labels[b]),
                     "length_m": float(getattr(link, "length", 0.0) or 0.0),
                     "closable": link.link_type == "Pipe"})
    return pd.DataFrame(rows, columns=["link", "link_type", "district_a", "district_b",
                                       "length_m", "closable"])


def district_types(wn, labels: pd.Series, boundary: pd.DataFrame) -> pd.DataFrame:
    """Type 1 / 2 / 3 classification exactly as Sec. 2.3 defines it."""
    src_districts = {labels[s] for s in source_nodes(wn) if s in labels.index}
    adj = {d: {} for d in labels.unique()}
    for _, r in boundary.iterrows():
        adj[r.district_a][r.district_b] = adj[r.district_a].get(r.district_b, 0) + 1
        adj[r.district_b][r.district_a] = adj[r.district_b].get(r.district_a, 0) + 1

    juncs = set(wn.junction_name_list)
    rows = []
    for d in sorted(labels.unique()):
        links_to_src = sum(n for nb, n in adj[d].items() if nb in src_districts)
        kind = 1 if d in src_districts else (2 if links_to_src > 0 else 3)
        rows.append({"district": d,
                     "n_junctions": sum(1 for n in labels.index[labels == d] if n in juncs),
                     "type": kind, "links_to_source_district": links_to_src,
                     "vital": bool(kind == 3 or (kind == 2 and links_to_src == 1))})
    return pd.DataFrame(rows)


def forced_open_links(wn, labels: pd.Series, boundary: pd.DataFrame,
                      types: pd.DataFrame) -> list[str]:
    """Pipes that must stay open: vital districts' supply paths, plus non-pipes.

    The search runs on the *district-level* schematic (the paper's Fig. 1), not
    the node graph: vertices are districts, edges are boundary pipes, shortest
    in hops to any Type 1 district. Where several pipes join the same district
    pair, the shortest is the one pinned open -- the other half of the paper's
    rule that closure prioritises longer pipes, on the grounds that longer pipes
    are likelier to leak or break.
    """
    src = set(types.district[types.type == 1])
    vital = list(types.district[types.vital])
    pinned = set(boundary.link[~boundary.closable])          # pumps/valves: not closable
    if not src or not vital:
        return sorted(pinned)

    dg = nx.Graph()
    dg.add_nodes_from(labels.unique())
    for (a, b), grp in boundary.groupby(["district_a", "district_b"]):
        pick = grp.sort_values(["length_m", "link"]).iloc[0]
        dg.add_edge(a, b, link=pick.link)

    for d in vital:
        best = None
        for s in src:
            if dg.has_node(d) and dg.has_node(s) and nx.has_path(dg, d, s):
                path = nx.shortest_path(dg, d, s)
                if best is None or len(path) < len(best):
                    best = path
        if best:
            pinned |= {dg[u][v]["link"] for u, v in zip(best, best[1:])}
    return sorted(pinned)


# ==========================================================================
# Phase 3 -- criteria and evaluation
# ==========================================================================

@dataclass
class HydraulicCriteria:
    """Eq. (3), plus an escape hatch for a network that fails it before any cut.

    ``mode="paper"`` is the constraint as written: every junction at or above
    ``min_pressure_m``, every pipe at or below ``max_velocity_ms``.

    ``mode="baseline"`` lowers the pressure floor (and raises the velocity cap)
    to whatever the intact network already achieves, whenever the intact network
    itself misses the stated threshold. The requirement becomes "districting may
    not make the network worse than it already is", and it is identical to the
    literal constraint whenever the intact network satisfies it. The relaxation
    is a single global floor rather than a per-node one on purpose: per-node
    would pin every already-failing junction to its exact current pressure, so a
    millimetre of drop anywhere would reject a layout -- strictly conservative,
    but brittle enough that nothing is feasible below full cardinality and the
    enumeration loses every chance to prune.
    """
    min_pressure_m: float
    max_velocity_ms: float
    mode: str = "paper"                       # "paper" | "baseline"

    def thresholds(self, baseline: dict | None) -> tuple[float, float]:
        if self.mode != "baseline" or baseline is None:
            return self.min_pressure_m, self.max_velocity_ms
        return (min(self.min_pressure_m, float(baseline["pressure"].min())),
                max(self.max_velocity_ms, float(baseline["velocity"].max())))


def peak_demand_model(wn) -> wntr.network.WaterNetworkModel:
    """A single-period copy at each node's peak multiplier.

    "the design of DMAs should be based on critical scenarios, with the maximum
    demand selected from demand patterns" (Sec. 2.4). Folding the peak
    multiplier into the base demand and clearing the pattern gives exactly that,
    and keeps every candidate layout to one solve.
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


def solve(wn) -> dict:
    """One EPANET solve into a scratch directory, addressed absolutely.

    Deliberately does not ``chdir``. Changing the working directory is fine in a
    script but fragile in a notebook: interrupt a long cell between the ``chdir``
    and its restore and the kernel is left sitting in a temp directory the
    context manager has already deleted, after which every later solve fails
    with EPANET error 302 for reasons having nothing to do with the network.
    An absolute ``file_prefix`` removes the dependency on cwd entirely.
    """
    with tempfile.TemporaryDirectory() as scratch:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = wntr.sim.EpanetSimulator(wn).run_sim(
                file_prefix=str(Path(scratch) / "en"))
    return {"pressure": res.node["pressure"].iloc[0][wn.junction_name_list],
            "velocity": res.link["velocity"].iloc[0][wn.pipe_name_list]}


def connected_to_source(wn, closed: set[str]) -> bool:
    """Does every demand-bearing junction still reach a source?

    Checked first, because a closure that islands a district does not make
    EPANET fail -- it makes EPANET return nonsense for the isolated nodes, which
    a pressure test may or may not catch. A graph traversal costs microseconds
    against a ~50 ms solve, so it is the gate rather than a post-hoc check.
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


def evaluate(wn, closed, criteria: HydraulicCriteria,
             baseline: dict | None = None) -> dict:
    """Feasibility of one boundary-pipe configuration, per Eq. (3)."""
    closed = set(closed)
    if not connected_to_source(wn, closed):
        return {"feasible": False, "min_pressure_m": np.nan,
                "max_velocity_ms": np.nan, "reason": "islands a demand node"}

    saved = {}
    for name in closed:
        link = wn.get_link(name)
        saved[name] = link.initial_status
        link.initial_status = wntr.network.LinkStatus.Closed
    try:
        r = solve(wn)
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
# Phase 3 -- optimisation
# ==========================================================================

def _objectives(open_links, lengths) -> tuple[int, float]:
    """Eq. (1) NO and Eq. (2) LO over the open boundary pipes."""
    return len(open_links), float(sum(lengths[l] for l in open_links))


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated (NO, LO) pairs, ordered by NO then LO."""
    keep = [r for r in rows
            if not any(o is not r and o["NO"] <= r["NO"] and o["LO"] <= r["LO"]
                       and (o["NO"] < r["NO"] or o["LO"] < r["LO"]) for o in rows)]
    return sorted(keep, key=lambda r: (r["NO"], r["LO"]))


def optimize_exhaustive(wn, boundary: pd.DataFrame, forced: list[str],
                        criteria: HydraulicCriteria, baseline: dict | None = None,
                        budget: int = 20000) -> dict:
    """Exact Pareto front by ascending-cardinality enumeration with superset pruning.

    See the module docstring for why this is exact. The only thing that can make
    it inexact is exhausting ``budget``, reported as ``complete=False`` rather
    than absorbed silently.
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

    front = pareto_front(feasible)
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

    The paper's settings (population 400, 2000 generations) are exposed but not
    the default: at ~0.08 s per EPANET solve that is roughly 17 hours for a
    17-bit search space the exhaustive backend settles exactly in minutes.
    Infeasible individuals rank behind every feasible one, which is the standard
    constrained-domination handling for Eq. (3). Evaluations are memoised on the
    genome, so a converged population stops paying for solves.
    """
    rng = np.random.default_rng(seed)
    lengths = dict(zip(boundary.link, boundary.length_m))
    forced = [l for l in forced if l in lengths]
    free = [l for l in boundary.link if l not in forced]
    n = len(free)
    if n == 0:
        return {"front": [], "n_free": 0, "n_forced": len(forced), "search_space": 1,
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
            better = (ranks[i] < ranks[j]
                      or (ranks[i] == ranks[j] and crowd[i] > crowd[j]))
            return pop[i if better else j]

        children = []
        while len(children) < pop_size:
            a, b = tournament(), tournament()
            if rng.random() < p_crossover and n > 1:
                cut = rng.integers(1, n)
                a, b = (np.concatenate([a[:cut], b[cut:]]),
                        np.concatenate([b[:cut], a[cut:]]))
            for child in (a.copy(), b.copy()):
                flip = rng.random(n) < p_mutation / max(n, 1)
                child[flip] ^= 1
                children.append(child)
        merged = pop + children[:pop_size]
        sc, ranks, crowd = rank_and_crowd(merged)
        order = sorted(range(len(merged)), key=lambda i: (ranks[i], -crowd[i]))
        pop = [merged[i] for i in order[:pop_size]]

    rows = [{k: v for k, v in r.items() if k != "feasible"}
            for r in memo.values() if r["feasible"]]
    return {"front": pareto_front(rows), "n_free": n, "n_forced": len(forced),
            "search_space": 2 ** n, "evaluations": len(memo), "complete": True,
            "backend": "nsga2"}


def optimize(wn, boundary: pd.DataFrame, forced: list[str],
             criteria: HydraulicCriteria, baseline: dict | None = None,
             backend: str = "auto", budget: int = 20000, **kw) -> dict:
    """Exhaustive when the search space fits the budget, NSGA-II otherwise."""
    if backend not in OPTIMIZER_BACKENDS:
        raise ValueError("backend must be one of %s" % (OPTIMIZER_BACKENDS,))
    n_free = sum(1 for l in boundary.link if l not in set(forced))
    if backend == "auto":
        need = sum(comb(n_free, c) for c in range(n_free + 1))
        backend = "exhaustive" if need <= budget else "nsga2"
    if backend == "exhaustive":
        return optimize_exhaustive(wn, boundary, forced, criteria, baseline, budget)
    return optimize_nsga2(wn, boundary, forced, criteria, baseline, **kw)


def status_table(boundary: pd.DataFrame, open_links) -> pd.DataFrame:
    """Boundary pipes with the device each one needs.

    Open pipes take flow meters, closed ones gate valves (Sec. 2.3). This table
    is the hand-off to the sensor-placement stage: it is where the meters are.
    """
    open_set = set(open_links)
    out = boundary.copy()
    out["status"] = np.where(out.link.isin(open_set), "open", "closed")
    out["device"] = np.where(out.link.isin(open_set), "flow meter", "gate valve")
    return out.sort_values(["district_a", "district_b", "link"]).reset_index(drop=True)
