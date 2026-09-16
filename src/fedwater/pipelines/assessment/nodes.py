"""Assessment: what is this network, and can we simulate on it?

A pre-flight, run once per network before any world is built. It answers three
questions and writes each answer down:

1. **What is in the file.** Component inventory, declared units, headloss
   formula, global demand multiplier, pattern lengths. D-Town ships 7 tanks,
   11 pumps, 5 valves, 44 curves and 24 controls -- none of which the
   simulation handles yet, and all of which are invisible until counted.
2. **Do the districts hold.** The junction partition must be exact, and links
   crossing between districts are inventoried BY TYPE, because the coupling
   variants only ever close pipes.
3. **How much demand it can carry.** ``lambda_star`` and ``max_demand_lps``
   from the constant-demand steady-state sweep -- the number that replaces the
   hand-tuned ``hydraulics.anchor_scale`` in the next stage.

Nothing here feeds the simulation yet. Everything here is a report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedwater.networks import capacity as cap
from fedwater.networks import conditioning as nc
from fedwater.networks import options as opt
from fedwater.networks import partition as part

SEC_PER_DAY = 86400.0


def _cfg_conditioning(assessment: dict) -> nc.ConditioningConfig:
    c = assessment["conditioning"]
    return nc.ConditioningConfig(
        fold_demand_multiplier=True,
        flatten_patterns=False,
        normalize_pattern_length=int(c["normalize_pattern_length"]),
        duration_h=float(c["duration_h"]),
        timestep_s=int(c["timestep_s"]),
        demand_model="DD",
        required_pressure_m=float(assessment["pressure_floor_m"]),
        inpfile_units=c["inpfile_units"],
        expected_headloss=c["expected_headloss"],
    )


def _cfg_capacity(assessment: dict) -> cap.CapacityConfig:
    c = assessment["capacity"]
    return cap.CapacityConfig(
        pressure_floor_m=float(assessment["pressure_floor_m"]),
        lambda_grid=tuple(float(x) for x in c["lambda_grid"]),
        tank_levels=tuple(c["tank_levels"]),
        modes=tuple(c["modes"]),
        bisect_tol=float(c["bisect_tol"]),
        expected_headloss=assessment["conditioning"]["expected_headloss"],
    )


# --------------------------------------------------------------------------
# 1. what is in the file
# --------------------------------------------------------------------------
def inventory_network(wn, network_profile: dict, districts: dict) -> pd.DataFrame:
    """One row: components, units, and the two base-demand conventions.

    ``base_demand_lps_raw`` is what ``build_portfolios`` anchors on (the
    simulation pins the global multiplier without folding it).
    ``base_demand_lps_folded`` is what the capacity sweep measures against.
    On Graeme they differ by 5x. Both are reported so stage 2 cannot wire the
    wrong one into ``anchor_scale``.
    """
    units = opt.describe_units(wn)
    raw = opt.total_base_demand_lps(wn)
    lengths = opt.pattern_lengths(wn)
    consumers = nc.consumer_junctions(wn)

    row = {
        "network": network_profile["name"],
        "n_junctions": len(wn.junction_name_list),
        "n_consumers": len(consumers),
        "n_reservoirs": len(wn.reservoir_name_list),
        "n_tanks": len(wn.tank_name_list),
        "n_pipes": len(wn.pipe_name_list),
        "n_pumps": len(wn.pump_name_list),
        "n_valves": len(wn.valve_name_list),
        "n_curves": len(wn.curve_name_list),
        "n_controls": len(wn.control_name_list),
        "n_patterns": len(wn.pattern_name_list),
        "n_districts": len(part.district_nodes(districts)),
        "n_district_assets": sum(
            len(v) for v in part.district_assets(districts).values()),
        **units,
        "base_demand_lps_raw": round(raw, 4),
        "base_demand_lps_folded": round(
            raw * units["inp_demand_multiplier"], 4),
        "pattern_len_min": min(lengths.values()) if lengths else 0,
        "pattern_len_max": max(lengths.values()) if lengths else 0,
        # Ragged = not a whole number of 24-step cycles, so EPANET's wrap
        # shifts the diurnal phase. Multi-day = an exact multiple (D-Town's
        # 168-step weekly patterns), which is fine and must NOT be truncated.
        "pattern_lens_ragged": ", ".join(sorted(
            {str(v) for v in lengths.values() if v > 1 and v % 24})) or "",
        "pattern_lens_multiday": ", ".join(sorted(
            {str(v) for v in lengths.values() if v > 24 and not v % 24})) or "",
        # Everything the simulation does not handle yet, in one number.
        "unhandled_components": int(
            len(wn.tank_name_list) + len(wn.pump_name_list)
            + len(wn.valve_name_list)),
    }
    return pd.DataFrame([row])


# --------------------------------------------------------------------------
# 2. do the districts hold
# --------------------------------------------------------------------------
def assess_partition(wn, districts: dict,
                     network_profile: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the junction partition; inventory inter-district links by type.

    ``validate_partition`` raises on a bad partition -- that is the point of
    running this first. The link table is the warning the coupling variants
    cannot give you: ``apply_coupling`` walks pipes only, so a district pair
    joined through a pump or a valve stays connected in an `isolated` world
    and is absent from ``gt_boundaries``.
    """
    net = network_profile["name"]
    report = part.validate_partition(wn, districts)
    report.insert(0, "network", net)

    links = part.district_links(wn, districts)
    if links.empty:
        crossing = pd.DataFrame(columns=["network", "district_a", "district_b",
                                         "Pipe", "Pump", "Valve", "n_links",
                                         "pipes_only"])
    else:
        crossing = (links.groupby(["district_a", "district_b", "link_type"])
                    .size().unstack(fill_value=0).reset_index())
        for col in ("Pipe", "Pump", "Valve"):
            if col not in crossing:
                crossing[col] = 0
        crossing["n_links"] = crossing[["Pipe", "Pump", "Valve"]].sum(axis=1)
        # False = apply_coupling cannot fully separate this pair.
        crossing["pipes_only"] = crossing["n_links"] == crossing["Pipe"]
        crossing.insert(0, "network", net)

    # Which junction the drift auto-picker would choose per district, so the
    # choice is inspectable before a world is built rather than after.
    seeds = []
    for d in part.district_nodes(districts):
        try:
            node, why = part.auto_seed_node(wn, districts, d), "auto"
        except ValueError as exc:
            node, why = None, str(exc)
        override = (network_profile.get("drift_seed_nodes") or {}).get(d)
        seeds.append({"network": net, "district": d,
                      "auto_seed_node": node, "profile_override": override,
                      "note": why})
    report = report.merge(pd.DataFrame(seeds), on=["network", "district"],
                          how="left")
    return report, crossing


# --------------------------------------------------------------------------
# 3. can it be conditioned, and how much can it carry
# --------------------------------------------------------------------------
def condition_and_tier(wn, network_profile: dict,
                       assessment: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                                  pd.DataFrame]:
    """Storage-policy ladder: keep the least intrusive policy that works.

    Returns (tier, attempts, recipe). Every rejected policy and its reason is
    in ``attempts`` -- a network that only runs with frozen tanks is a
    different object from one that runs as shipped, and the tier says which.
    """
    net = network_profile["name"]
    cfg = _cfg_conditioning(assessment)
    floor = float(assessment["pressure_floor_m"])
    ladder = tuple(assessment["conditioning"]["storage_ladder"])

    conditioned, recipe, attempts = nc.auto_condition(
        wn, net, cfg, ladder, floor,
        source=network_profile.get("source", {}).get("file", ""))
    chosen = [a for a in recipe.actions if a["step"] == "auto_condition"][0]
    v = nc.validate_model(conditioned, floor)

    tier = pd.DataFrame([{
        "network": net,
        "tier": chosen["params"]["tier"],
        "storage_policy": chosen["params"]["policy"],
        "n_attempts": len(attempts),
        "n_tanks": len(conditioned.tank_name_list),
        "n_controls": len(conditioned.control_name_list),
        "usable": v["usable"],
        "meets_floor": v["meets_floor"],
        "pressure_floor_m": floor,
        "pressure_min_m": v["pressure_min_m"],
        "pressure_max_m": v.get("pressure_max_m", np.nan),
        "frac_negative": v["frac_negative"],
        "overpressure": v.get("overpressure", False),
        "worst_nodes": v["worst_nodes"],
    }])
    if attempts.empty:
        attempts = pd.DataFrame(columns=["network", "policy", "usable", "reason"])
    return tier, attempts, recipe.to_frame()


def assess_capacity(wn, network_profile: dict,
                    assessment: dict) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame, pd.DataFrame]:
    """Constant-demand steady-state capacity. Returns (summary, curve,
    defects, binding).

    ``max_demand_lps`` on the ``(initial, DD)`` row is the alpha that replaces
    the hand-tuned ``hydraulics.anchor_scale`` next stage. It is measured in
    the FOLDED demand convention -- see ``inventory_network``.
    """
    net = network_profile["name"]
    out = cap.assess_capacity(wn, net, _cfg_capacity(assessment))

    summary = out["summary"]
    if not summary.empty:
        summary = summary.copy()
        summary["is_headline"] = (
            (summary.get("tank_level") == assessment["capacity"]["tank_levels"][0])
            & (summary.get("mode") == "DD"))
    empty = pd.DataFrame()
    return (summary,
            out["curve"] if len(out["curve"]) else empty,
            out["defects"] if len(out["defects"]) else empty,
            out["binding"] if len(out["binding"]) else empty)


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------
def compile_assessment_report(inventory: pd.DataFrame, partition: pd.DataFrame,
                              crossing: pd.DataFrame, tier: pd.DataFrame,
                              capacity_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per check, with a verdict. This is the file to read.

    Verdicts are ``ok`` / ``warn`` / ``fail``. Nothing raises here: a network
    that cannot be simulated should produce a readable report, not a traceback
    two minutes into EPANET.
    """
    # Every input has been through a CSVDataset round-trip, so an empty string
    # comes back as NaN and a bare `if value:` would read as True. Normalize.
    def txt(value) -> str:
        return "" if pd.isna(value) else str(value)

    inv = inventory.iloc[0]
    net = inv["network"]
    rows = []

    def add(check, verdict, value, note):
        rows.append({"network": net, "check": check, "verdict": verdict,
                     "value": value, "note": note})

    add("units", "ok", inv["inp_units"],
        "wntr converts to SI on load; inpfile_units pinned to LPS on output")
    add("headloss", "ok", inv["headloss"],
        "asserted against parameters, never coerced")

    m = float(inv["inp_demand_multiplier"])
    add("demand_multiplier", "ok" if m == 1.0 else "warn", m,
        "pinned to 1.0 by the simulation (raw base demands are the anchor); "
        "folded by the capacity sweep. Compare like with like."
        if m != 1.0 else "no hidden scale in the file")

    ragged = txt(inv["pattern_lens_ragged"])
    multi = txt(inv["pattern_lens_multiday"])
    add("pattern_lengths", "warn" if ragged else "ok",
        ragged or multi or f"{inv['pattern_len_min']}-{inv['pattern_len_max']}",
        f"lengths {ragged} are not whole 24-step cycles: EPANET wraps them, so "
        "the diurnal cycle drifts a little every day. Conditioning trims or "
        "pads them."
        if ragged else
        (f"multi-day patterns ({multi} steps); whole cycles, left intact"
         if multi else "all patterns 24-step or scalar"))

    add("partition", "ok", f"{len(partition)} districts, "
        f"{int(partition['n_nodes'].sum())} junctions",
        "exact junction partition verified")

    n_assets = int(inv["n_district_assets"])
    add("district_assets", "ok" if n_assets == 0 else "warn", n_assets,
        "tanks/reservoirs recorded under `assets`; nothing reads them yet"
        if n_assets else "no storage assigned to a district")

    if len(crossing):
        leaky = crossing[~crossing["pipes_only"]]
        add("inter_district_links", "ok" if leaky.empty else "warn",
            f"{int(crossing['n_links'].sum())} links, "
            f"{int(crossing['Pipe'].sum())} pipes",
            "apply_coupling closes pipes only, so these pairs stay connected "
            f"in an isolated world: {sorted(zip(leaky.district_a, leaky.district_b))}"
            if not leaky.empty else "every crossing is a pipe; coupling can separate them")

    unhandled = int(inv["unhandled_components"])
    add("unhandled_components", "ok" if unhandled == 0 else "warn", unhandled,
        f"{int(inv['n_tanks'])} tanks, {int(inv['n_pumps'])} pumps, "
        f"{int(inv['n_valves'])} valves — the simulation does not model these yet"
        if unhandled else "gravity network, nothing deferred")

    t = tier.iloc[0]
    add("conditioning_tier", "ok" if t["tier"].startswith(("T0", "T1")) else "fail",
        f"{t['tier']} via {t['storage_policy']}",
        f"min pressure {t['pressure_min_m']:.2f} m, "
        f"max {t['pressure_max_m']:.2f} m")

    head = capacity_summary[capacity_summary.get("is_headline", False)] \
        if "is_headline" in capacity_summary else capacity_summary
    if len(head):
        h = head.iloc[0]
        lam = h.get("lambda_star", np.nan)
        add("capacity", "fail" if pd.isna(lam) else
            ("warn" if float(lam) < 2.0 else "ok"),
            "n/a" if pd.isna(lam) else round(float(lam), 4),
            h.get("status", "") if pd.isna(lam) else
            f"max_demand_lps={h.get('max_demand_lps')} "
            f"(folded convention; raw base demand is "
            f"{inv['base_demand_lps_raw']} L/s). "
            "lambda*<1 means overloaded as shipped; <2 means little headroom.")

    return pd.DataFrame(rows)
