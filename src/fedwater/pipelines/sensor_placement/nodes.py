"""Sensor placement nodes: probe stacks -> slot selection -> verification.

``ensure_probe_stacks`` is the one node in the project that does its own
I/O: the stacks are content-addressed and shared across worlds, so their path
is only known once the spec has been hashed at run time. Its output is a small
pointer (``placement_stacks``, persisted as ``stacks.yml``) and everything
downstream reads the stacks through it. Two stacks per world, at most:

* **selection** -- at ``sensor_placement.selection_coupling`` (baseline by
  default), so every coupling arm of a study gets the SAME sensors;
* **label**     -- at the world's REALISED coupling (its actual closure set,
  read from ``gt_boundaries``), so the mixture written for each sensor
  describes the physics this world actually has.

When both specs hash the same (a world at the reference coupling) they are
one stack, built once.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from fedwater.hashing import canonical_hash
from fedwater.placement import classification as cl
from fedwater.placement import excite, store
from fedwater.placement import select as sel
from fedwater.placement.verify import verify_placement

SOURCES = ("dynamic", "manual")


def _reference_coupling(coupling: dict) -> dict:
    """The selection stack's coupling (a configured reference, no draw)."""
    variant = coupling["variant"]
    if variant not in ("baseline", "isolated"):
        raise ValueError(
            "sensor_placement.selection_coupling.variant must be baseline or "
            f"isolated (a reference must not depend on a draw), got {variant!r}")
    return {"variant": variant}


def _realised_coupling(coupling: dict, gt_boundaries: pd.DataFrame) -> dict:
    """The label stack's coupling: what the world's network ACTUALLY is.

    Keyed on the realised closure set rather than on the dial that produced
    it. A ``partial`` world whose draw closed nothing (KY7's manual partition:
    every boundary is a bridge to the reservoir) is physically a baseline
    world and shares the baseline stack; one that closed pipes is rebuilt in
    the probes with exactly those pipes closed, independent of the sim seed.
    """
    if coupling["variant"] == "isolated":
        return {"variant": "isolated"}
    closed = sorted(str(p) for p in
                    gt_boundaries.loc[gt_boundaries["closed"].astype(bool),
                                      "pipe"])
    if not closed:
        return {"variant": "baseline"}
    return {"variant": "explicit", "closed": closed}


def _source(cfg: dict) -> str:
    source = cfg.get("source", "dynamic")
    if source not in SOURCES:
        raise ValueError(f"sensor_placement.source must be one of {SOURCES}, "
                         f"got {source!r}")
    return source


# --------------------------------------------------------------------------
# 1. stacks
# --------------------------------------------------------------------------
def ensure_probe_stacks(wn, network_profile: dict, districts: dict,
                        partition_manifest: dict, partition_meta: dict,
                        hydraulics: dict, scenario: dict, validation: dict,
                        time: dict, coupling: dict, gt_boundaries: pd.DataFrame,
                        land_use: dict, buildings: dict, patterns: dict,
                        sensor_placement: dict) -> dict:
    """Load or build the selection and label stacks; return the pointer."""
    pointer = {"source": _source(sensor_placement),
               "network": partition_meta["name"],
               "districting": partition_meta["method"],
               "partition_id": partition_meta["partition_id"]}
    if pointer["source"] == "manual":
        return pointer

    probe = sensor_placement["probe"]
    classes = sensor_placement["classes"]
    transitions = excite.validate_transitions(probe["transitions"], land_use)
    world = {"hydraulics": hydraulics, "scenario": scenario,
             "validation": validation, "time": time, "land_use": land_use,
             "buildings": buildings, "patterns": patterns}
    plan = excite.horizon_plan(wn, districts, partition_meta, probe, world)
    inputs = {"network_inp": wn, "network_profile": network_profile,
              "districts": districts, "partition_manifest": partition_manifest}
    root = Path(sensor_placement["store"]).expanduser().resolve()

    reference = _reference_coupling(sensor_placement["selection_coupling"])
    actual = _realised_coupling(coupling, gt_boundaries)
    s = int(probe["seed"])
    for role, c in (("selection", reference), ("label", actual)):
        spec = store.stack_spec(network=partition_meta["name"],
                                partition=partition_meta, world=world,
                                probe=probe, classes=classes, plan=plan,
                                coupling=c, seed=s)
        h = canonical_hash(spec)
        if role == "label" and h == pointer["selection"]["stack_hash"]:
            pointer["label"] = copy.deepcopy(pointer["selection"])
            continue
        st = store.ensure_stack(store_root=root, spec=spec, inputs=inputs,
                                world=world, probe=probe, classes=classes,
                                plan=plan, transitions=transitions,
                                coupling=c, seed=s)
        pointer[role] = {"stack_hash": st.stack_hash, "path": str(st.path),
                         "coupling": c, "seed": s,
                         "src_drift": bool(st.src_drift)}
    pointer["world_coupling"] = dict(coupling)
    pointer["same_stack"] = (pointer["selection"]["stack_hash"]
                             == pointer["label"]["stack_hash"])
    pointer["plan"] = {k: plan[k] for k in (
        "n_months", "need_n_months", "days_per_month", "warmup_months",
        "drift_ramp_days", "seasonality_scale", "season_aligned",
        "growth_chance", "max_neighbors_per_month", "convert_months",
        "largest_district", "max_eccentricity")} | {"modes": plan["modes"]}
    return pointer


# --------------------------------------------------------------------------
# 2. selection
# --------------------------------------------------------------------------
def select_sensors(placement_stacks: dict, districts: dict,
                   network_profile: dict, partition_meta: dict,
                   sensor_placement: dict):
    """-> sensor_placement, coverage, channels, report-only arms."""
    names = list(districts["districts"])
    meta = {"network": partition_meta["name"],
            "districting": partition_meta["method"],
            "partition_id": partition_meta["partition_id"]}
    if placement_stacks["source"] == "manual":
        placement = sel.manual_placement(
            network_profile, {**meta, "selection_stack": "", "label_stack": ""})
        empty = pd.DataFrame({"note": ["sensor_placement.source is manual"]})
        return placement, empty, empty, empty

    s_ptr, l_ptr = placement_stacks["selection"], placement_stacks["label"]
    s_stack = store.load_stack(s_ptr["path"])
    l_stack = (s_stack if placement_stacks["same_stack"]
               else store.load_stack(l_ptr["path"]))
    s_table = cl.classification(s_stack)
    l_table = s_table if l_stack is s_stack else cl.classification(l_stack)

    selection, coverage = sel.select_slots(
        s_table, names, sensor_placement["slots"],
        sensor_placement["classes"],
        float(sensor_placement["dedupe_min_distance"]))
    placement = sel.placement_frame(
        selection, l_table, names, l_stack.channels,
        {**meta, "selection_stack": s_ptr["stack_hash"],
         "label_stack": l_ptr["stack_hash"],
         "label_coupling": l_ptr["coupling"]["variant"],
         "label_closed": ";".join(l_ptr["coupling"].get("closed", []))})

    channels = [s_stack.channels.assign(stack="selection",
                                        stack_hash=s_ptr["stack_hash"])]
    if l_stack is not s_stack:
        channels.append(l_stack.channels.assign(
            stack="label", stack_hash=l_ptr["stack_hash"]))
    channels = pd.concat(channels, ignore_index=True)

    n_max = max(int(v.get("pure", 0)) + int(v.get("mixed", 0))
                for v in sensor_placement["slots"].values())
    arms = cl.selection_table(
        l_table, names, l_stack.config_id, arms=("control_low", "unusable"),
        n_max=n_max, channels=l_stack.channels, enforce_channel=False,
        min_distance=float(sensor_placement["dedupe_min_distance"]))
    if not len(arms):
        arms = pd.DataFrame({"note": ["no control_low / unusable gauges"]})
    return placement, coverage.assign(**{k: v for k, v in meta.items()}), \
        channels, arms


# --------------------------------------------------------------------------
# 3. verification
# --------------------------------------------------------------------------
def verify_sensors(sensor_placement_df: pd.DataFrame, pressures: pd.DataFrame,
                   flows: pd.DataFrame, demand_series: pd.DataFrame,
                   gt_drift_schedule: pd.DataFrame, districts: dict,
                   time: dict, patterns: dict,
                   sensor_placement: dict) -> pd.DataFrame:
    """The chosen set against this world's own drift (report only)."""
    settle = sensor_placement["probe"]["settle"]
    return verify_placement(
        sensor_placement_df, pressures, flows, demand_series,
        gt_drift_schedule, districts, time, patterns, settle,
        float(sensor_placement["classes"]["null_factor"]))
