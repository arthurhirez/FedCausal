"""Hydraulic options, pinned from config -- the single source of truth.

Both the simulation (``network_prep.configure_network``) and the assessment
(``networks.conditioning``) go through these functions. Before, each set
duration, timestep and demand model independently, so an assessment could
certify a model the simulation never actually ran.

Units
-----
wntr converts an ``.inp`` to SI on load regardless of its ``UNITS`` line, so
GPM and LPS files arrive with demands in m3/s, elevations in m and pressures
in mca. Nothing downstream needs to know which the file used. What is *not*
automatic, and is handled here:

* ``inpfile_units`` -- decides the units of any ``.inp`` written back out.
  Pinned to LPS so a conditioned file is never a GPM/SI hybrid.
* the headloss formula -- H-W and D-W give different friction for the same
  pipe. Asserted, not coerced: silently switching a network's formula would
  change its physics without a papertrail.
* the global demand multiplier -- see ``fold_demand_multiplier`` below.
"""
from __future__ import annotations

import numpy as np

LPS_PER_M3S = 1000.0


# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------
def describe_units(wn) -> dict:
    """What the source file declared, for the record."""
    return {
        "inp_units": str(wn.options.hydraulic.inpfile_units),
        "headloss": str(wn.options.hydraulic.headloss),
        "inp_demand_multiplier": float(
            wn.options.hydraulic.demand_multiplier or 1.0),
        "viscosity": float(wn.options.hydraulic.viscosity or 1.0),
        "specific_gravity": float(wn.options.hydraulic.specific_gravity or 1.0),
    }


def total_base_demand_lps(wn) -> float:
    """Sum of junction base demands, in L/s. Excludes the global multiplier."""
    return LPS_PER_M3S * sum(
        sum(float(ts.base_value or 0.0)
            for ts in wn.get_node(j).demand_timeseries_list)
        for j in wn.junction_name_list)


def pattern_lengths(wn) -> dict[str, int]:
    return {p: int(np.asarray(wn.get_pattern(p).multipliers).size)
            for p in wn.pattern_name_list}


def components(wn) -> dict:
    """Component census, in the shape the simulation cares about.

    ``has_storage`` is the dispatch flag, not ``n_tanks``: the presence of
    storage is what turns a network from a sequence of independent
    steady-state solves into a dynamic system, where continuity holds over the
    horizon rather than at each step and a district's supply can be borrowed
    from a tank filled hours earlier.
    """
    pump_kinds: dict[str, int] = {}
    for name in wn.pump_name_list:
        kind = str(getattr(wn.get_link(name), "pump_type", "?")).upper()
        pump_kinds[kind] = pump_kinds.get(kind, 0) + 1
    valve_kinds: dict[str, int] = {}
    for name in wn.valve_name_list:
        kind = str(getattr(wn.get_link(name), "valve_type", "?")).upper()
        valve_kinds[kind] = valve_kinds.get(kind, 0) + 1
    return {
        "n_junctions": len(wn.junction_name_list),
        "n_reservoirs": len(wn.reservoir_name_list),
        "n_tanks": len(wn.tank_name_list),
        "n_pipes": len(wn.pipe_name_list),
        "n_pumps": len(wn.pump_name_list),
        "n_valves": len(wn.valve_name_list),
        "n_curves": len(wn.curve_name_list),
        "n_controls": len(wn.control_name_list),
        "n_patterns": len(wn.pattern_name_list),
        "has_storage": bool(wn.tank_name_list),
        "has_pumps": bool(wn.pump_name_list),
        "has_valves": bool(wn.valve_name_list),
        "pump_types": ",".join(f"{k}:{v}" for k, v in sorted(pump_kinds.items())),
        "valve_types": ",".join(f"{k}:{v}" for k, v in sorted(valve_kinds.items())),
    }


def normalize_demand_slots(wn) -> dict:
    """Give every junction exactly one demand timeseries entry.

    ``run_hydraulics`` writes the synthesized series into
    ``demand_timeseries_list[0]`` and assumes that entry is the whole of the
    node's demand. Two ways an ``.inp`` breaks that assumption:

    * a junction with NO entry (nothing to write into -- an ``IndexError``);
    * a junction with SEVERAL, from a ``[DEMANDS]`` block carrying extra
      demand categories. Slot 0 would be overwritten and the remaining
      categories would keep drawing their original ``.inp`` demand on top of
      the synthesized series -- unsynthesized volume, in the model, invisible
      to V2 because V2 compares only what it wrote.

    Extra entries are folded into slot 0's base value before being dropped, so
    the node's raw base demand -- which ``build_portfolios`` anchors on -- is
    preserved exactly. Returns what it touched.
    """
    added, merged = [], []
    for name in wn.junction_name_list:
        node = wn.get_node(name)
        dtl = node.demand_timeseries_list
        if len(dtl) == 0:
            node.add_demand(base=0.0, pattern_name=None)
            added.append(name)
        elif len(dtl) > 1:
            extra = sum(float(ts.base_value or 0.0) for ts in list(dtl)[1:])
            while len(dtl) > 1:
                dtl.pop()
            dtl[0].base_value = float(dtl[0].base_value or 0.0) + extra
            merged.append(name)
    return {"added_empty_slot": added, "merged_extra_categories": merged}


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def check_headloss(wn, expected: str = "H-W") -> str:
    """Assert the friction formula. Raises rather than coercing."""
    actual = str(wn.options.hydraulic.headloss).upper().replace("_", "-")
    if actual != expected.upper():
        raise ValueError(
            f"Network declares headloss={actual!r}, expected {expected!r}. "
            "Coercing it would change every pipe's friction without a "
            "papertrail; set the expected formula in parameters if this "
            "network really is different.")
    return actual


def pin_inpfile_units(wn, units: str = "LPS") -> str:
    """Units used when this model is written back out as an ``.inp``."""
    wn.options.hydraulic.inpfile_units = units
    return units


def pin_demand_multiplier(wn, value: float = 1.0) -> float:
    """Pin EPANET's global multiplier WITHOUT touching base demands.

    This is what the simulation wants. ``build_portfolios`` derives every
    node's demand anchor from the raw ``.inp`` base demand, so folding the
    multiplier in (see ``fold_demand_multiplier``) would rescale the entire
    scenario by it -- 5x on Graeme, which ships 0.2.
    """
    wn.options.hydraulic.demand_multiplier = float(value)
    return float(value)


def fold_demand_multiplier(wn) -> dict:
    """Move the global multiplier INTO the base demands; pin the option to 1.

    Physically a no-op, and the opposite convention to
    ``pin_demand_multiplier``. The assessment wants it because it frees
    ``demand_multiplier`` as the capacity-scaling knob lambda; the simulation
    must NOT have it, because its demand anchor is read from the raw base
    demands. Every table the assessment emits therefore reports base demand
    both ways, so the two conventions can never be compared by accident.
    """
    m = float(wn.options.hydraulic.demand_multiplier or 1.0)
    if abs(m - 1.0) < 1e-12:
        return {"multiplier": m, "folded": False, "n_entries": 0}
    n = 0
    for jname in wn.junction_name_list:
        for ts in wn.get_node(jname).demand_timeseries_list:
            if ts.base_value:
                ts.base_value = float(ts.base_value) * m
                n += 1
    wn.options.hydraulic.demand_multiplier = 1.0
    return {"multiplier": m, "folded": True, "n_entries": n}


def normalize_pattern_lengths(wn, target: int = 24) -> list[str]:
    """Make every pattern a whole number of ``target``-step cycles.

    EPANET wraps a pattern when it runs out, so a pattern whose length is not a
    multiple of the daily cycle walks that cycle backwards a little every day.
    The KY files ship 23-step "daily" patterns, which lose an hour per day and
    would quietly destroy any hour-of-day residualization downstream.

    A pattern that is ALREADY an exact multiple is left alone. D-Town ships
    168-step (7 x 24) weekly patterns; truncating those to 24 would collapse
    the week into its first day and delete the weekday/weekend structure --
    which is precisely the signal the land-use model is built to carry.

    Otherwise the pattern is truncated down to the nearest whole multiple, or
    padded cyclically up to one cycle if it is shorter than ``target``.

    NOT applied by the simulation: ``run_hydraulics`` replaces every junction
    demand pattern with the synthesized series. Reservoir-head and pump-speed
    patterns are left as the file defines them -- next-stage work.
    """
    fixed = []
    for pname in wn.pattern_name_list:
        pat = wn.get_pattern(pname)
        mult = np.asarray(pat.multipliers, dtype=float)
        n = mult.size
        if n <= 1 or n % target == 0:
            continue
        if n < target:
            reps = int(np.ceil(target / n))
            pat.multipliers = np.tile(mult, reps)[:target]
            new_n, how = target, "padded cyclically"
        else:
            new_n = (n // target) * target
            pat.multipliers = mult[:new_n]
            how = "truncated to whole cycles"
        fixed.append(f"{pname}:{n}->{new_n} ({how})")
    return fixed


def pin_time(wn, duration_h: float, timestep_s: int) -> dict:
    """Duration and every timestep, from config, never from file defaults."""
    wn.options.time.duration = int(round(duration_h * 3600))
    wn.options.time.hydraulic_timestep = int(timestep_s)
    wn.options.time.pattern_timestep = int(timestep_s)
    wn.options.time.report_timestep = int(timestep_s)
    return {"duration_h": duration_h, "timestep_s": int(timestep_s)}


def pin_demand_model(wn, model: str = "DD", required_pressure_m: float = 15.0,
                     minimum_pressure_m: float = 0.0) -> str:
    """``DD`` (demand-driven) or ``PDD`` (pressure-driven)."""
    wn.options.hydraulic.demand_model = model
    if model.upper().startswith("P"):
        wn.options.hydraulic.required_pressure = float(required_pressure_m)
        wn.options.hydraulic.minimum_pressure = float(minimum_pressure_m)
    return model
