"""One call per method, one result object, one sanity suite.

    cfg = DistrictingConfig(k=5, method="spectral")
    res = run(wn, cfg)
    res.write_yml("out/dtown__spectral.yml", wn)

Every stage is a plain function taking explicit inputs and returning explicit
outputs, with no I/O and no globals, so each maps to a Kedro node without
rewriting: :func:`run` is the node body, :class:`DistrictingConfig` is a
``parameters.yml`` block (``DistrictingConfig.from_dict`` takes the nested dict
straight), and :class:`DistrictingResult` carries only DataFrames, Series and
plain dicts -- catalogue-serialisable, apart from the graph, which is kept for
plotting and is not meant to be persisted.

The stages, in order: partition -> repair -> rebalance -> report -> boundary ->
optional Phase-3 optimisation. Stages after the partition are identical for
every method, which is exactly what makes the comparison in :func:`compare`
legitimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import metrics, yaml_io
from .graph import build_graph, check_elevation_informative, consumer_nodes
from .hydraulics import (
    HydraulicCriteria,
    boundary_table,
    district_types,
    forced_open_links,
    optimize,
    peak_demand_model,
    solve,
    status_table,
)
from .partition import (
    DEFAULT_BALANCE,
    DEFAULT_REPAIR,
    METHODS,
    partition_labels,
    rebalance,
    repair_contiguity,
)
from .weights import DistrictingWeights


# ==========================================================================
# configuration
# ==========================================================================

@dataclass
class DistrictingConfig:
    """Everything one run needs, flat enough to live in ``parameters.yml``."""
    k: int = 5
    method: str = "spectral"
    seed: int = 0
    weights: DistrictingWeights = field(default_factory=DistrictingWeights)

    # post-processing
    repair: bool | None = None            # None -> DEFAULT_REPAIR for this method
    balance_target: str | None = "default"   # "count" | "demand" | None | "default"
    balance_tol: float = 0.15

    # connectivity policy
    on_disconnected: str = "raise"        # "raise" | "largest"

    # Phase 3 (optional)
    criteria: HydraulicCriteria | None = None
    optimizer: str = "none"               # "none" | "auto" | "exhaustive" | "nsga2"
    budget: int = 20000
    nsga2: dict = field(default_factory=dict)
    force_optimize: bool = False          # run even when the intact network is infeasible

    @classmethod
    def from_dict(cls, d: dict) -> "DistrictingConfig":
        d = dict(d)
        if isinstance(d.get("weights"), dict):
            d["weights"] = DistrictingWeights(**d["weights"])
        if isinstance(d.get("criteria"), dict):
            d["criteria"] = HydraulicCriteria(**d["criteria"])
        return cls(**d)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DistrictingResult:
    """Everything one run produced. Serialisable apart from ``graph``."""
    network: str
    method: str
    k_requested: int
    k: int
    labels: pd.Series                      # every node
    consumer_labels: pd.Series             # consumers only, for balance
    graph: nx.Graph
    report: dict
    components: dict
    elevation_check: dict
    n_fragments_repaired: int = 0
    balance_trace: dict | None = None
    boundary: pd.DataFrame | None = None
    types: pd.DataFrame | None = None
    forced_open: list[str] | None = None
    baseline: dict | None = None
    baseline_feasible: bool | float = np.nan
    optimization: dict | None = None
    best: dict | None = None
    status: pd.DataFrame | None = None
    config: DistrictingConfig | None = None

    # -- outputs ---------------------------------------------------------
    def districts_mapping(self, wn) -> dict:
        return yaml_io.districts_mapping(wn, self.labels)

    def write_yml(self, path, wn, header: str = "") -> Path:
        return yaml_io.write_districts_yml(path, wn, self.labels, header or self.default_header())

    def default_header(self) -> str:
        return ("District partition of %s by method '%s', k=%d.\n"
                "districts MUST partition the junction set exactly -- it is the "
                "simulation domain.\nassets records tanks and reservoirs per district."
                % (self.network, self.method, self.k))

    def summary_row(self) -> dict:
        return metrics.summary_row(self.network, self)


# ==========================================================================
# the run
# ==========================================================================

def run(wn, cfg: DistrictingConfig, network: str = "network",
        coupling: pd.DataFrame | None = None,
        mapping: dict | None = None,
        reference: dict | None = None,
        graph: nx.Graph | None = None) -> DistrictingResult:
    """Partition -> repair -> rebalance -> report -> boundary -> optimise.

    ``graph`` may be passed in to reuse one build across several methods, which
    is what :func:`run_all` does: building it once also guarantees every method
    is scored on a byte-identical graph.
    """
    if cfg.method not in METHODS:
        raise ValueError("method must be one of %s, got %r" % (METHODS, cfg.method))
    g = graph if graph is not None else build_graph(wn)

    out = partition_labels(wn, g, cfg.method, cfg.k, weights=cfg.weights,
                           coupling=coupling, mapping=mapping, seed=cfg.seed,
                           on_disconnected=cfg.on_disconnected)
    labels, h = out["labels"], out["graph"]

    # --- post-processing. Contiguity is always MEASURED below; repairing it is
    #     a choice, defaulted per family (on for spectral, off for community).
    repair = DEFAULT_REPAIR[cfg.method] if cfg.repair is None else bool(cfg.repair)
    n_repaired = 0
    if repair:
        labels, n_repaired = repair_contiguity(h, labels)

    consumers = [n for n in labels.index if n in set(consumer_nodes(wn))]
    balance = (DEFAULT_BALANCE[cfg.method] if cfg.balance_target == "default"
               else cfg.balance_target)
    balance_trace = None
    if balance:
        labels, balance_trace = rebalance(h, labels, target=balance, tol=cfg.balance_tol,
                                          seed=cfg.seed, scope=consumers)
        if repair:
            labels, extra = repair_contiguity(h, labels)   # rebalancing can refragment
            n_repaired += extra

    consumer_labels = labels.reindex(consumers)
    report = metrics.full_report(h, labels, consumer_labels, reference)

    res = DistrictingResult(
        network=network, method=cfg.method, k_requested=int(cfg.k),
        k=int(labels.nunique()), labels=labels, consumer_labels=consumer_labels,
        graph=h, report=report, components=out["components"],
        elevation_check=check_elevation_informative(h),
        n_fragments_repaired=n_repaired, balance_trace=balance_trace, config=cfg,
    )

    # --- Phase 2: boundary accounting. Always computed -- it is the hardware
    #     cost of the partition and the hand-off to sensor placement.
    res.boundary = boundary_table(wn, labels)
    res.types = district_types(wn, labels, res.boundary)
    res.forced_open = forced_open_links(wn, labels, res.boundary, res.types)

    if cfg.criteria is None or cfg.optimizer == "none":
        res.status = status_table(res.boundary, list(res.boundary.link))
        return res

    # --- Phase 3: optimisation, behind a pre-flight gate.
    peak = peak_demand_model(wn)
    base = solve(peak)
    base_p, base_v = float(base["pressure"].min()), float(base["velocity"].max())
    res.baseline = {"min_pressure_m": base_p, "max_velocity_ms": base_v}
    base_ok = (base_p >= cfg.criteria.min_pressure_m
               and base_v <= cfg.criteria.max_velocity_ms)
    res.baseline_feasible = bool(base_ok)

    # The pre-flight gate exists for one pathological case: with no feasible set
    # anywhere, nothing prunes and the enumeration runs the full 2^n_free to
    # conclude what the single baseline solve already showed. It is a default,
    # not a theorem -- closing a pipe can raise pressure elsewhere by rerouting
    # flow -- so force_optimize=True runs the search regardless, and
    # criteria.mode="baseline" is the constructive alternative.
    if not base_ok and cfg.criteria.mode == "paper" and not cfg.force_optimize:
        res.optimization = {
            "front": [], "n_free": len(res.boundary) - len(res.forced_open),
            "n_forced": len(res.forced_open), "evaluations": 1, "complete": True,
            "backend": "skipped",
            "note": "intact network already violates Eq. (3): min pressure %.2f m "
                    "vs %.1f m required" % (base_p, cfg.criteria.min_pressure_m)}
        res.status = status_table(res.boundary, list(res.boundary.link))
        return res

    res.optimization = optimize(peak, res.boundary, res.forced_open, cfg.criteria,
                                base, cfg.optimizer, cfg.budget, **cfg.nsga2)
    if res.optimization["front"]:
        res.best = res.optimization["front"][0]          # fewest meters, Eq. (1) first
        res.status = status_table(res.boundary, res.best["open_links"])
    else:
        res.status = status_table(res.boundary, list(res.boundary.link))
    return res


def run_all(wn, cfg: DistrictingConfig, methods=METHODS, network: str = "network",
            coupling: pd.DataFrame | None = None, mapping: dict | None = None,
            reference: dict | None = None, verbose: bool = True
            ) -> dict[str, DistrictingResult]:
    """Run several methods against one network on one shared graph."""
    from dataclasses import replace

    g = build_graph(wn)
    out = {}
    for m in methods:
        if m == "external" and mapping is None:
            continue
        out[m] = run(wn, replace(cfg, method=m), network=network, coupling=coupling,
                     mapping=mapping, reference=reference, graph=g)
        if verbose:
            r = out[m]
            print("%-14s k=%d  contiguous=%-5s  imbalance=%.2f  boundary=%d"
                  % (m, r.k, bool(r.report["contiguity"].contiguous.all()),
                     r.report["balance"]["count_imbalance"], len(r.boundary)))
    return out


def compare(results: dict[str, DistrictingResult]) -> pd.DataFrame:
    """One row per method, all-NaN columns dropped."""
    df = pd.DataFrame([r.summary_row() for r in results.values()])
    return df.dropna(axis=1, how="all")


# ==========================================================================
# sanity suite
# ==========================================================================

def sanity_check(wn, result: DistrictingResult, *, check_determinism: bool = True,
                 coupling: pd.DataFrame | None = None, mapping: dict | None = None
                 ) -> pd.DataFrame:
    """Assertions a partition must survive before anything downstream reads it.

    Structural first (does the output satisfy the ``districts.yml`` contract),
    then partition-level (contiguity, non-empty, k achieved), then
    reproducibility (same config, same labels), then hydraulic where an
    optimisation ran. Every check reports a ``detail`` naming the offending
    nodes, because "contiguity failed" without saying where is not actionable.
    """
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    labels = result.labels
    doc = result.districts_mapping(wn)
    for _, row in yaml_io.validate_districts(wn, doc).iterrows():
        add("schema: " + row["check"], row["ok"], row["detail"])

    # --- partition-level
    con = result.report["contiguity"]
    bad = con.loc[~con.contiguous, "district"].tolist()
    add("every district is contiguous", not bad,
        "ok" if not bad else "fragmented: %s" % bad)

    add("k achieved equals k requested", result.k == result.k_requested,
        "requested %d, got %d" % (result.k_requested, result.k))

    empty = [d for d in labels.unique() if (labels == d).sum() == 0]
    add("no empty district", not empty, "ok" if not empty else str(empty))

    unassigned = [n for n in wn.node_name_list if n not in labels.index]
    add("every network node carries a label", not unassigned,
        "ok" if not unassigned else "%d unlabelled: %s" % (len(unassigned), unassigned[:8]))

    add("graph is one connected component", result.components.get("connected", False),
        "components: %s" % result.components.get("sizes"))

    n_cons = int(result.consumer_labels.notna().sum())
    covered = sorted(result.consumer_labels.dropna().unique())
    add("every district holds at least one consumer",
        len(covered) == result.k,
        "%d consumers across %d of %d districts" % (n_cons, len(covered), result.k))

    # --- reproducibility
    if check_determinism and result.config is not None:
        # labels only -- re-running Phase 3 would pay for a second full EPANET
        # enumeration to re-check something the optimiser does not influence
        from dataclasses import replace as _replace

        again = run(wn, _replace(result.config, criteria=None, optimizer="none"),
                    network=result.network, coupling=coupling,
                    mapping=mapping, graph=result.graph)
        same = again.labels.reindex(labels.index).equals(labels)
        add("same config reproduces the same labels", same,
            "ok" if same else "labels differ on re-run -- an unseeded RNG is leaking")

    # --- informational, not failures in themselves
    ec = result.elevation_check
    add("elevation factor is informative on this network", ec["informative"],
        "span %.1f m -- the elevation dial is inert below ~1 m" % ec["span_m"])

    if result.balance_trace:
        bt = result.balance_trace
        add("rebalancing reached its tolerance", bt["reached_tol"],
            "imbalance %.2f -> %.2f in %d move(s) — %s"
            % (bt["imbalance_before"], bt["imbalance_after"], bt["n_moves"],
               bt["stall_reason"]))

    # --- hydraulic, only where Phase 3 ran
    if result.baseline is not None:
        add("intact network meets the stated criteria",
            bool(result.baseline_feasible),
            "baseline min pressure %.2f m, max velocity %.2f m/s"
            % (result.baseline["min_pressure_m"], result.baseline["max_velocity_ms"]))
    if result.optimization is not None and result.optimization["front"]:
        add("Pareto front is exact (not budget-truncated)",
            result.optimization.get("complete", False),
            "%d evaluations, backend %s"
            % (result.optimization.get("evaluations", -1),
               result.optimization.get("backend", "?")))
        add("forced-open pipes are all open in the chosen layout",
            set(result.forced_open) <= set(result.best["open_links"]),
            "forced %d, open %d" % (len(result.forced_open), result.best["NO"]))

    if result.boundary is not None and len(result.boundary):
        crossing = all(labels[wn.get_link(l).start_node_name]
                       != labels[wn.get_link(l).end_node_name]
                       for l in result.boundary.link)
        add("every boundary link really crosses a district edge", crossing)

    return pd.DataFrame(checks)
