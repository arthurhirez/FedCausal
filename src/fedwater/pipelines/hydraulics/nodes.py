"""Hydraulics: inject demand series into the network and run EPANET via wntr.

Convention that makes volumes un-breakable: every junction gets
``base_value = 0.001 m3/s`` (exactly 1 L/s) and its pattern multipliers ARE
the demand series in L/s. EPANET's demand = base x multiplier then reproduces
the synthesized L/s series identically — there is no second bookkeeping of
scales to get wrong (the 1/24 bug class is structurally impossible).
"""
from __future__ import annotations

import copy

import pandas as pd
import wntr

UNIT_BASE_SI = 0.001  # 1 L/s in m3/s


def run_hydraulics(wn, demand_series: pd.DataFrame):
    """Assign patterns, run EpanetSimulator, return (pressures, flows).

    Pressures: meters of water column, per node. Flows: L/s, per link.
    """
    wn = copy.deepcopy(wn)
    node_cols = [c for c in demand_series.columns if c != "month"]

    for node in node_cols:
        junction = wn.get_node(node)
        pattern_name = f"P{node}"
        wn.add_pattern(pattern_name, demand_series[node].to_list())
        junction.demand_timeseries_list[0].base_value = UNIT_BASE_SI
        junction.demand_timeseries_list[0].pattern_name = pattern_name

    results = wntr.sim.EpanetSimulator(wn).run_sim()

    pressures = results.node["pressure"]
    flows = results.link["flowrate"] * 1000.0  # m3/s -> L/s
    demands = results.node["demand"] * 1000.0  # m3/s -> L/s (for mass balance)

    # EPANET reports one extra step at t=duration; align to the pattern length.
    n = len(demand_series)
    pressures, flows, demands = pressures.iloc[:n], flows.iloc[:n], demands.iloc[:n]
    for df in (pressures, flows, demands):
        df.index = pd.RangeIndex(len(df), name="step")

    month = demand_series["month"].reset_index(drop=True)
    pressures.insert(0, "month", month)
    flows.insert(0, "month", month)
    demands.insert(0, "month", month)
    return pressures, flows, demands
