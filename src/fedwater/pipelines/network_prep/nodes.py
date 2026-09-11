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
from fedwater.networks import profile as nprofile
from fedwater.networks.partition import (
    district_nodes,
    validate_partition,
)


def resolve_network_parameters(network_profile: dict, hydraulics: dict,
                               scenario: dict, validation: dict):
    """Fill the network-scoped parameters that ``parameters.yml`` left null.

    ``conf/base/parameters.yml`` is shared by every network, but a handful of
    its keys are only meaningful against one: ``anchor_scale`` (0.05 is right
    for Graeme's 5557 L/s of base demand and absurd for KY7's 67),
    ``income_landuse_mapping`` (POSITIONAL over districts, so a five-entry
    list is a DIFFERENT SCENARIO on a four-district network, not merely a
    wrong one), ``drift.tgt_district``, ``pressure_band_mca``. Each was a
    silent wrong answer after a network switch.

    Those keys are ``null`` in parameters and filled from the bundle's
    ``profile.yml``. The resolver only ever fills nulls, so the experiments
    engine -- which applies the same function in ``spec.resolve_world`` and
    writes the result into ``conf/local/parameters.yml`` -- always wins, and
    running it twice changes nothing. See ``networks/profile.py``.
    """
    params = {"hydraulics": hydraulics, "scenario": scenario,
              "validation": validation}
    resolved, report = nprofile.resolve_params(params, network_profile)
    return (resolved["hydraulics"], resolved["scenario"],
            resolved["validation"], report)


def configure_network(wn, network_profile: dict, hydraulics: dict, time: dict):
    """Set *explicit* hydraulic and time options, and normalise the model.

    Rationale: Graeme's .inp carries a hidden global ``Demand Multiplier = 0.2``.
    Every option that affects physics is pinned here, from parameters, so the
    simulation never depends on silent defaults baked into the input file.

    The multiplier is PINNED, never folded into the base demands. That is a
    deliberate asymmetry with the assessment pipeline, which folds it so that
    lambda is its only knob: ``build_portfolios`` derives every node's demand
    anchor from the RAW .inp base demand, so folding here would rescale the
    entire scenario by the file's own multiplier (5x on Graeme). The assessment
    reports base demand under both conventions for exactly this reason.

    Units: wntr converts an ``.inp`` to SI on load regardless of its ``UNITS``
    line, so KY7's GPM file and Graeme's LPS file arrive identically -- demands
    in m3/s, elevations in m, pressures in mca. Nothing downstream needs to
    know which the file used. ``inpfile_units`` is pinned so that a model
    written back out is never a GPM/SI hybrid, and the declared unit system is
    recorded in the prep report.

    Two normalisations, both no-ops on Graeme:

    * **demand slots.** Every junction is given exactly one demand timeseries
      entry, so ``[0]`` downstream is the whole of a node's demand rather than
      the first of several ``[DEMANDS]`` categories.
    * **pattern lengths.** Junction demand patterns are replaced wholesale by
      ``run_hydraulics``, but reservoir-head, pump-speed and energy patterns
      are not, and EPANET WRAPS a pattern when it runs out. KY7 ships a
      23-step ``ENRG1``, which would walk its cycle backwards an hour a day.
      Exact multiples of 24 are left alone -- truncating D-Town's 168-step
      weekly patterns would delete the weekday/weekend structure the land-use
      model exists to carry.
    """
    wn = copy.deepcopy(wn)
    declared = opt.describe_units(wn)
    opt.check_headloss(wn, hydraulics["expected_headloss"])
    opt.pin_inpfile_units(wn, hydraulics.get("inpfile_units", "LPS"))
    opt.pin_demand_multiplier(wn, float(hydraulics["demand_multiplier"]))
    opt.pin_demand_model(wn, hydraulics["demand_model"])  # 'DD' or 'PDD'

    slots = opt.normalize_demand_slots(wn)
    fixed = opt.normalize_pattern_lengths(
        wn, int(hydraulics.get("normalize_pattern_length", 24)))

    horizon_h = float(time["n_months"] * time["days_per_month"] * 24)
    opt.pin_time(wn, horizon_h, int(time["resolution_h"] * 3600))

    census = opt.components(wn)
    report = pd.DataFrame([{
        "network": network_profile.get("name", "?"),
        **declared,
        "profile_declares_units": network_profile.get("source", {}).get(
            "inp_units", ""),
        **census,
        "horizon_h": horizon_h,
        "timestep_s": int(time["resolution_h"] * 3600),
        "demand_model": hydraulics["demand_model"],
        "anchor_scale": float(hydraulics["anchor_scale"]),
        "demand_slots_added": len(slots["added_empty_slot"]),
        "demand_slots_merged": len(slots["merged_extra_categories"]),
        "patterns_normalized": "; ".join(fixed),
    }])
    if census["has_storage"] or census["has_pumps"] or census["has_valves"]:
        print(f"configure_network: {census['n_tanks']} tank(s), "
              f"{census['n_pumps']} pump(s) [{census['pump_types'] or '-'}], "
              f"{census['n_valves']} valve(s), {census['n_controls']} "
              "control(s) are active in this model.")
    return wn, report


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
