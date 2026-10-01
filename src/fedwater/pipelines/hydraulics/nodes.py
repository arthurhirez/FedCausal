"""Hydraulics: inject demand series into the network and run EPANET via wntr.

Convention that makes volumes un-breakable: every junction gets
``base_value = 0.001 m3/s`` (exactly 1 L/s) and its pattern multipliers ARE
the demand series in L/s. EPANET's demand = base x multiplier then reproduces
the synthesized L/s series identically — there is no second bookkeeping of
scales to get wrong (the 1/24 bug class is structurally impossible).
"""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
import wntr

UNIT_BASE_SI = 0.001  # 1 L/s in m3/s

SCRATCH_PREFIX = "fedwater_epanet_"
# Where EPANET's scratch files go. Defaults to the system temp directory; set
# FEDWATER_SCRATCH to a drive with room (a 70-month hourly KY7 world writes
# ~1.4 GB of .bin alone). Not part of any world's identity.
SCRATCH_ENV = "FEDWATER_SCRATCH"
# A scratch directory untouched this long is debris from a run that died
# before cleaning up (a crash, a killed process, a lock that never cleared).
STALE_HOURS = 24.0
# Margin over the estimate: EPANET also writes its hydraulics scratch file.
SPACE_MARGIN = 1.5


def scratch_root() -> Path:
    root = Path(os.environ.get(SCRATCH_ENV) or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root


def scratch_bytes(wn, n_steps: int) -> int:
    """EPANET's binary results file for ``n_steps`` reported periods:
    4 node and 8 link variables, 4-byte floats, per period (+ the t=0 one).
    Checked against a real KY7 run: 27.3 kB per hourly period."""
    return int((4 * wn.num_nodes + 8 * wn.num_links) * 4 * (n_steps + 1))


def sweep_stale(root: Path, hours: float = STALE_HOURS) -> list[Path]:
    """Delete scratch directories whose newest file is older than ``hours``.

    Only directories carrying this module's prefix, and only stale ones: a
    concurrent run's directory is minutes old and is left alone.
    """
    removed, cutoff = [], time.time() - hours * 3600
    for d in Path(root).glob(f"{SCRATCH_PREFIX}*"):
        if not d.is_dir():
            continue
        try:
            newest = max([f.stat().st_mtime for f in d.rglob("*")]
                         + [d.stat().st_mtime])
        except OSError:
            continue
        if newest < cutoff and _remove(d, attempts=1):
            removed.append(d)
    return removed


def _remove(path: Path, attempts: int = 6, pause: float = 0.5) -> bool:
    """rmtree with retries. On Windows a file EPANET (or an antivirus scan)
    still holds cannot be deleted for a moment; retry, then give up quietly."""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if i + 1 < attempts:
                time.sleep(pause * (i + 1))
    return False


def _check_space(root: Path, need: int) -> None:
    free = shutil.disk_usage(root).free
    if free < need * SPACE_MARGIN:
        raise OSError(
            f"run_hydraulics: EPANET needs ~{need / 1e9:.2f} GB of scratch "
            f"(x{SPACE_MARGIN} margin) in {root}, which has "
            f"{free / 1e9:.2f} GB free. Free space there, or point "
            f"{SCRATCH_ENV} at a drive with room.")


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

    ``file_prefix`` is pointed at a fresh directory under
    :func:`scratch_root` instead, so the scratch files are born outside the
    project tree and the directory is removed the moment this function
    returns -- on the success path and on any exception -- with retries,
    because on Windows a just-written file can stay locked for a moment.
    Before solving, stale scratch left by dead runs is swept and the free
    space is checked against the size of the results file, so a full disk
    fails fast with its own message instead of as EPANET error 308 an hour
    into the solve.
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

    root = scratch_root()
    sweep_stale(root)
    _check_space(root, scratch_bytes(wn, len(demand_series)))
    scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=root))
    try:
        results = wntr.sim.EpanetSimulator(wn).run_sim(
            file_prefix=str(scratch / "run"))
    finally:
        # Cleanup never raises: a scratch directory must not fail a world,
        # and must never MASK the solver's own error (Python's
        # TemporaryDirectory turned a locked .rpt into a NotADirectoryError
        # that hid EPANET's actual failure). What cannot be removed now is
        # swept by a later run once it is stale.
        if not _remove(scratch):
            warnings.warn(f"run_hydraulics: could not remove scratch "
                          f"{scratch} (a file is still locked); it will be "
                          f"swept once older than {STALE_HOURS:.0f} h.")

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
