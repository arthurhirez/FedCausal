"""excite.py -- the excitation experiment that produces the mixture labels.

A gauge's district mixture is identified by stacking K worlds that share the
target world's operating point (network, partition, initial consumption map,
beta, anchor, coupling) and in each of which exactly ONE district changes land
use. This module builds those worlds.

What decides each probe's change
--------------------------------
``sensor_placement.probe.transitions`` maps a district's INITIAL land use to
the land use its probe converts it to::

    residential -> industrial      mixed      -> industrial
    commercial  -> industrial      industrial -> residential

Every class must be mapped and none may map to itself (a no-op excitation
gives ``||dz_d(k)|| = 0`` and an undefined gain). The district keeps its own
income: the probe measures a land-use signature change. Because
``mixture_probe.world_response`` divides by each excitation's own size, columns
built from different transitions are commensurable as long as the network
responds roughly linearly -- ``estimator_checks`` and the shape-vs-native
elasticities stay in the report as the check.

How a probe world is simulated
------------------------------
Through the project's OWN simulation pipeline (``network_prep`` ->
``urban_scenario`` -> ``demand_synthesis`` -> ``hydraulics`` ->
``sim_validation``), executed in-process node by node in the pipeline's
topological order. The POC's ``landuse_world.build`` copied that node order by
hand; this reads it from the ``Pipeline`` objects, so a change upstream cannot
leave the probe chain behind. Hard V-checks raise here exactly as they do in a
``kedro run``: a probe world that fails physics fails the stack.

Parameters are the target world's RESOLVED blocks (``hydraulics_cfg`` et al.),
so ``anchor_scale`` reaches the probes from the one place it is ever resolved
-- the bundle profile, or a study's explicit value. There is no network table
of anchors anywhere in this package.
"""
from __future__ import annotations

import copy
import math

import networkx as nx
import pandas as pd

from fedwater.networks import profile as nprofile
from fedwater.networks.partition import auto_seed_node, district_nodes

__all__ = ["validate_transitions", "excitation", "horizon_plan",
           "probe_params", "core_pipeline", "simulate_probe", "PROBE_TARGETS"]

# What a probe world must hand back to the analysis.
PROBE_TARGETS = ("pressures", "flows", "demand_series", "gt_drift_schedule",
                 "wn_variant", "validation_report", "scenario_cfg",
                 "hydraulics_cfg")


# ==========================================================================
# transitions
# ==========================================================================
def validate_transitions(transitions: dict, land_use: dict) -> dict:
    classes = set((land_use or {}).get("mix") or {})
    t = {str(k): str(v) for k, v in (transitions or {}).items()}
    missing = classes - set(t)
    unknown = (set(t) | set(t.values())) - classes
    selfmap = sorted(k for k, v in t.items() if k == v)
    if missing or unknown or selfmap:
        raise ValueError(
            "sensor_placement.probe.transitions must map every land-use class "
            f"in land_use.mix {sorted(classes)} to a DIFFERENT class. "
            f"missing={sorted(missing)} unknown={sorted(unknown)} "
            f"self-mapped={selfmap}")
    return t


def excitation(regime, transitions: dict) -> dict:
    """``(income, land_use)`` -> the probe's drift target for that district."""
    income, land_use = regime
    return {"to_income": str(income), "to_land_use": transitions[str(land_use)]}


# ==========================================================================
# horizon
# ==========================================================================
def _seed_reach(wn, nodes: list, seed: str) -> tuple[int, float]:
    """(hops from seed to the furthest node it reaches, share reached) inside
    the district -- the same undirected subgraph ``build_drift_schedule``
    diffuses on."""
    G = wn.to_graph().to_undirected().subgraph(nodes)
    if seed not in G:
        return 0, 0.0
    dist = nx.single_source_shortest_path_length(G, seed)
    return int(max(dist.values())), len(dist) / max(len(nodes), 1)


def horizon_plan(wn, districts: dict, seed_source: dict, probe: dict) -> dict:
    """Size the probe horizon and diffusion front against the partition.

    Ported from the POC's ``district_sweep.plan``. Under an uneven partition a
    single ``max_neighbors_per_month`` converts the small districts in two
    months and the large one in thirty, and the large district's world then
    has no settled tail. So the front width is derived from the LARGEST
    district, and the conversion time is bounded below by the eccentricity of
    each district's seed -- the front advances at most one hop per month
    however wide it is.

    Seeds follow the project rule: the partition's seed source (the profile's
    ``drift_seed_nodes`` for ``manual``, none otherwise), then the
    largest-base-demand junction.
    """
    parts = district_nodes(districts)
    rows = []
    for d, nodes in parts.items():
        seed = nprofile.resolve_seed_node(
            None, seed_source, d, lambda d=d: auto_seed_node(wn, districts, d))
        ecc, cov = _seed_reach(wn, nodes, seed)
        rows.append({"district": d, "seed_node": seed, "n_nodes": len(nodes),
                     "eccentricity": ecc, "coverage": round(cov, 4)})
    seeds = pd.DataFrame(rows)

    largest = int(seeds["n_nodes"].max())
    max_ecc = int(seeds["eccentricity"].max())
    convert = int(probe["convert_months"])
    width = probe.get("max_neighbors_per_month", "auto")
    auto = width in (None, "auto")
    front = max(1, math.ceil(largest / max(1, convert))) if auto else int(width)
    conv = max(math.ceil(largest / front), max_ecc)
    if auto and conv > 0:
        # when eccentricity binds, a wide front only converts whole annuli in
        # jumps; narrow it to the width that finishes in the same months
        front = max(1, math.ceil(largest / conv))
    ramp_months = math.ceil(int(probe["drift_ramp_days"])
                            / int(probe["days_per_month"]))
    need = (int(probe["warmup_months"]) + conv + ramp_months + 1
            + int(probe["settled_months"]))
    n = probe.get("n_months", "auto")
    months = need if n in (None, "auto") else int(n)
    if months < need:
        print(f"horizon_plan: n_months={months} is below the {need} this "
              "partition needs; settled_window will raise on the large "
              "districts")
    return {"n_months": int(months), "need_n_months": int(need),
            "max_neighbors_per_month": int(front), "convert_months": int(conv),
            "largest_district": largest, "max_eccentricity": max_ecc,
            "seeds": seeds}


# ==========================================================================
# parameters of one probe world
# ==========================================================================
def probe_params(world: dict, probe: dict, plan: dict, district: str,
                 regime, transitions: dict, coupling: dict,
                 seed: int) -> dict:
    """The parameter blocks one probe world runs with.

    ``world`` holds the target world's RESOLVED blocks: ``hydraulics``,
    ``scenario``, ``validation`` (from ``*_cfg``) and the base ``time``,
    ``land_use``, ``buildings``, ``patterns``. Only the probe's own horizon,
    drift and dials are changed; the operating point is inherited.
    """
    time = {**copy.deepcopy(world["time"]),
            "n_months": int(plan["n_months"]),
            "days_per_month": int(probe["days_per_month"])}
    scenario = copy.deepcopy(world["scenario"])
    scenario["n_months"] = int(plan["n_months"])
    scenario["drift"] = {
        **copy.deepcopy(scenario.get("drift") or {}),
        "tgt_district": district,
        # the world's own seed node belongs to the WORLD's target district;
        # a probe resolves its seed from the partition, then auto
        "seed_node": None,
        "warmup_months": int(probe["warmup_months"]),
        "growth_chance": float(probe["growth_chance"]),
        "max_neighbors_per_month": int(plan["max_neighbors_per_month"]),
        **excitation(regime, transitions),
    }
    patterns = {**copy.deepcopy(world["patterns"]),
                "drift_ramp_days": int(probe["drift_ramp_days"]),
                "seasonality_scale": float(probe["seasonality_scale"])}
    return {"time": time, "scenario": scenario, "patterns": patterns,
            "hydraulics": copy.deepcopy(world["hydraulics"]),
            "validation": copy.deepcopy(world["validation"]),
            "land_use": copy.deepcopy(world["land_use"]),
            "buildings": copy.deepcopy(world["buildings"]),
            "coupling": copy.deepcopy(coupling), "seed": int(seed)}


# ==========================================================================
# simulation, through the project's own pipelines
# ==========================================================================
def core_pipeline(targets=PROBE_TARGETS):
    """The simulation chain up to (and including) the V-checks."""
    from fedwater.pipelines import (demand_synthesis, hydraulics,
                                    network_prep, sim_validation,
                                    urban_scenario)
    full = (network_prep.create_pipeline() + urban_scenario.create_pipeline()
            + demand_synthesis.create_pipeline() + hydraulics.create_pipeline()
            + sim_validation.create_pipeline())
    return full.to_outputs(*targets)


def _run_inprocess(pipeline, data: dict) -> dict:
    data = dict(data)
    for n in pipeline.nodes:                      # topologically sorted
        missing = [i for i in n.inputs if i not in data]
        if missing:
            raise KeyError(f"probe world: node {n.name} needs {missing}")
        data.update(n.run({i: data[i] for i in n.inputs}))
    return data


def simulate_probe(inputs: dict, params: dict, label: str = "probe") -> dict:
    """Run one probe world. Returns the dict ``signal_probe.Probe.from_world``
    reads.

    ``inputs`` carries the catalog objects the chain starts from:
    ``network_inp`` (the raw model), ``network_profile``, ``districts`` and
    ``partition_manifest``.
    """
    data = {**inputs}
    for key in ("hydraulics", "scenario", "validation", "time", "coupling",
                "seed", "land_use", "buildings", "patterns"):
        data[f"params:{key}"] = params[key]
    out = _run_inprocess(core_pipeline(), data)
    resolved = {**params, "hydraulics": out["hydraulics_cfg"],
                "scenario": out["scenario_cfg"]}
    return {"params": resolved, "districts": inputs["districts"],
            "demand": out["demand_series"], "pressures": out["pressures"],
            "flows": out["flows"], "schedule": out["gt_drift_schedule"],
            "wn": out["wn_variant"], "validation": out["validation_report"],
            "label": label}
