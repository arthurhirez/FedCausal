"""Network preparation: hydraulic options, district partition, coupling variants.

Units note (wntr): the model stores demands in SI (m3/s) regardless of the
``.inp`` unit system, so a GPM file and an LPS file arrive identically;
pressures are meters of water column (mca). Option pinning is delegated to
``fedwater.networks.options`` so the assessment pipeline certifies the same
model this builds.
"""
from __future__ import annotations

import copy

import pandas as pd
import wntr

from fedwater.networks import options as opt
from fedwater.networks.partition import (
    district_nodes,
    validate_partition,
)


def configure_network(wn, hydraulics: dict, time: dict):
    """Set *explicit* hydraulic and time options.

    Rationale: Graeme's .inp carries a hidden global ``Demand Multiplier = 0.2``.
    Every option that affects physics is pinned here, from parameters, so the
    simulation never depends on silent defaults baked into the input file.

    The multiplier is PINNED, never folded into the base demands. That is a
    deliberate asymmetry with the assessment pipeline, which folds it so that
    lambda is its only knob: ``build_portfolios`` derives every node's demand
    anchor from the RAW .inp base demand, so folding here would rescale the
    entire scenario by the file's own multiplier (5x on Graeme). The assessment
    reports base demand under both conventions for exactly this reason.

    Junction demand patterns are left alone: ``run_hydraulics`` replaces all of
    them with the synthesized series. Reservoir-head and pump-speed patterns are
    also untouched, which matters for networks that have them (D-Town ships 140
    patterns and 11 pumps) and is next-stage work.
    """
    wn = copy.deepcopy(wn)
    opt.check_headloss(wn, hydraulics.get("expected_headloss", "H-W"))
    opt.pin_inpfile_units(wn, hydraulics.get("inpfile_units", "LPS"))
    opt.pin_demand_multiplier(wn, float(hydraulics["demand_multiplier"]))
    opt.pin_demand_model(wn, hydraulics["demand_model"])  # 'DD' or 'PDD'

    horizon_h = float(time["n_months"] * time["days_per_month"] * 24)
    opt.pin_time(wn, horizon_h, int(time["resolution_h"] * 3600))
    return wn


def _boundary_pipes(wn, districts: dict) -> pd.DataFrame:
    """All pipes whose endpoints belong to two different districts."""
    node_to_district = {
        n: d for d, nodes in district_nodes(districts).items() for n in nodes
    }
    rows = []
    for name in wn.pipe_name_list:
        pipe = wn.get_link(name)
        da = node_to_district.get(pipe.start_node_name)
        db = node_to_district.get(pipe.end_node_name)
        if da and db and da != db:
            rows.append({"pipe": name, "district_a": min(da, db), "district_b": max(da, db)})
    return pd.DataFrame(rows)


def apply_coupling(wn, districts: dict, coupling: dict, seed: int):
    """The dependence dial. Returns (network variant, ground-truth boundary table).

    Variants
    --------
    baseline : Graeme as-is — single reservoir, open boundaries (max coupling).
    partial  : close a fraction ``close_fraction`` of inter-district pipes.
    isolated : close *all* inter-district pipes and give each district its own
               reservoir (same head as the original source) — min coupling.

    The returned boundary table records which pipes exist between districts and
    which were closed: this is dependence ground truth, not a side effect.
    """
    import numpy as np

    wn = copy.deepcopy(wn)
    boundaries = _boundary_pipes(wn, districts)
    variant = coupling["variant"]

    if variant == "baseline":
        boundaries["closed"] = False
        return wn, boundaries

    rng = np.random.default_rng(seed)
    if variant == "partial":
        # Close boundaries CONNECTIVITY-PRESERVINGLY: candidates in random
        # order, a closure is kept only if every junction still reaches a
        # source — as a real DMA valve plan would. From a single source not
        # every boundary can close; the actual closures are recorded in the
        # returned table (full isolation is what the 'isolated' variant is
        # for). This is what V3 taught us: random closure severs service.
        import networkx as nx

        G = nx.MultiGraph()
        for name in wn.pipe_name_list:
            pipe = wn.get_link(name)
            G.add_edge(pipe.start_node_name, pipe.end_node_name, key=name)
        sources = set(wn.reservoir_name_list)
        junctions = set(wn.junction_name_list)

        def all_served(graph):
            reached = set()
            for s in sources:
                reached |= nx.node_connected_component(graph, s)
            return junctions <= reached

        k = int(round(coupling["close_fraction"] * len(boundaries)))
        to_close: set[str] = set()
        for idx in rng.permutation(len(boundaries)):
            if len(to_close) == k:
                break
            row = boundaries.iloc[idx]
            pipe = wn.get_link(row["pipe"])
            u, v = pipe.start_node_name, pipe.end_node_name
            G.remove_edge(u, v, key=row["pipe"])
            if all_served(G):
                to_close.add(row["pipe"])
            else:
                G.add_edge(u, v, key=row["pipe"])
    elif variant == "isolated":
        to_close = set(boundaries["pipe"])
    else:
        raise ValueError(f"Unknown coupling variant: {variant!r}")

    for pipe in to_close:
        wn.get_link(pipe).initial_status = wntr.network.LinkStatus.Closed
    boundaries["closed"] = boundaries["pipe"].isin(to_close)

    if variant == "isolated":
        src_head = wn.get_node(wn.reservoir_name_list[0]).base_head
        for i, (district, nodes) in enumerate(district_nodes(districts).items()):
            res_name, pipe_name = f"R_{district}", f"PR_{district}"
            wn.add_reservoir(res_name, base_head=src_head)
            # Feed each district at its first node through a short, wide pipe.
            wn.add_pipe(pipe_name, res_name, nodes[0], length=10, diameter=0.5,
                        roughness=130)
    return wn, boundaries
