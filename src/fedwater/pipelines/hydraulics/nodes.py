"""Hydraulics: inject demand series into the network and run EPANET via wntr.

Convention that makes volumes un-breakable: every junction gets
``base_value = 0.001 m3/s`` (exactly 1 L/s) and its pattern multipliers ARE
the demand series in L/s. EPANET's demand = base x multiplier then reproduces
the synthesized L/s series identically — there is no second bookkeeping of
scales to get wrong (the 1/24 bug class is structurally impossible).
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pandas as pd
import wntr

UNIT_BASE_SI = 0.001  # 1 L/s in m3/s


def run_hydraulics(wn, demand_series: pd.DataFrame):
    """Assign patterns, run EpanetSimulator, return (pressures, flows).

    Pressures: meters of water column, per node. Flows: L/s, per link.

    SCRATCH FILES. ``EpanetSimulator.run_sim()`` defaults to ``file_prefix=
    "temp"``, writing ``temp.inp``/``.rpt``/``.bin`` (and ``.hyd`` if asked)
    into the CURRENT WORKING DIRECTORY, and never deletes them. For a
    24-month, 1 h-step run that ``.bin`` is not incidental: it is EPANET's
    binary results file, one record per node and link PER TIMESTEP -- 117 MB
    on Graeme's 277 elements, 447 MB on KY7's ~1090. The results are already
    fully read into ``results`` by the time this function returns, so the
    file has no further use.

    This matters more than a stray file in ``/home/claude`` would suggest,
    because of WHERE it lands under the experiments engine: `kedro run`'s cwd
    is a world's ``clone/`` directory, which is the CACHE ITSELF -- a `kedro
    run` that never runs again for that ``sim_hash``. Left unhandled, a
    447 MB scratch file becomes 447 MB of PERMANENT dead weight per cached
    KY7 world, for every world a study ever simulates, forever, since nothing
    in the engine's lifecycle touches a world's clone after it is built.

    ``file_prefix`` is pointed at a :func:`tempfile.TemporaryDirectory`
    instead, so the scratch files are born outside the project tree and the
    directory (and everything EPANET wrote into it) is removed the moment
    this function returns -- on the success path and on any exception.
    ``networks.conditioning.solve`` already runs its own EPANET calls this
    way (via ``os.chdir`` into a scratch dir); this takes the same guarantee
    without the process-wide ``chdir``, which is the safer of the two once
    anything in the project runs nodes concurrently within one process rather
    than across the engine's separate `kedro run` subprocesses.
    """
    wn = copy.deepcopy(wn)
    node_cols = [c for c in demand_series.columns if c != "month"]
    synthesized = set(node_cols)

    for node in node_cols:
        junction = wn.get_node(node)
        pattern_name = f"P{node}"
        wn.add_pattern(pattern_name, demand_series[node].to_list())
        junction.demand_timeseries_list[0].base_value = UNIT_BASE_SI
        junction.demand_timeseries_list[0].pattern_name = pattern_name

    silenced = [j for j in wn.junction_name_list if j not in synthesized]
    for node in silenced:
        slot = wn.get_node(node).demand_timeseries_list[0]
        slot.base_value = 0.0
        slot.pattern_name = None
    if silenced:
        print(f"run_hydraulics: {len(silenced)} junction(s) carry no portfolio "
              f"and were pinned to zero demand (e.g. {silenced[:5]}).")

    with tempfile.TemporaryDirectory(prefix="fedwater_epanet_") as scratch:
        results = wntr.sim.EpanetSimulator(wn).run_sim(
            file_prefix=str(Path(scratch) / "run"))

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
