"""epanet_loop.py -- EPANET 2.2 runs through the toolkit step loop, and the mediator-blocked twin.

Ported from the validated ``epanet_gt`` harness (cells 3/4). Two things the pipeline's
``run_hydraulics`` (EpanetSimulator) does not give:

* the state of every MEDIATOR (pump, valve, control-driven pipe) and every tank at each hour, which
  the blocked twin needs to replay a trajectory;
* run integrity: whether the solve reached the horizon and which EPANET warnings it met. An
  unconverged step under ``UNBALANCED STOP`` halts EPANET silently and the hours after the halt read
  zero, so ``completed`` must be checked by the caller.

The solution is the one EpanetSimulator computes: the same .inp, the same solver, every intermediate
hydraulic step visited. ``snap`` (flows at t = 3600 h) is what the stored ``flows`` hold; ``epanet_gt``
gate G-E5 measured re-simulated worlds against stored flows at ~2e-7 relative (float32 storage).

Demand convention: verbatim :func:`fedwater.pipelines.hydraulics.nodes.run_hydraulics` -- base 1 L/s,
pattern = the series in L/s, junctions without a column pinned to zero.
"""
from __future__ import annotations

import copy
import ctypes
import tempfile
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr
from wntr.epanet import toolkit
from wntr.network.controls import Control, ControlAction
from wntr.network.io import write_inpfile

from fedwater.pipelines.hydraulics.nodes import UNIT_BASE_SI, _remove, scratch_root

__all__ = ["simulate", "blocked_model", "mediators", "head_reference_gaps", "inject_demand",
           "set_horizon", "run_ok"]

EN_HEAD, EN_PRESSURE, EN_TANKVOL = 10, 11, 24          # EPANET 2.2 node codes
EN_FLOW, EN_STATUS, EN_SETTING = 8, 11, 12             # EPANET 2.2 link codes
VALVE_OPEN_SETTING = 1e8                               # a valve fixed open reports no setting (a huge value)
LEVEL_TOL = 1e-3                                       # m: a tank within this of a level limit sits at the limit
SCRATCH_PREFIX = "fedwater_exact_"


def set_horizon(wn, hours: int, hyd_step: int = 3600):
    """Duration only; pattern and report steps stay as ``configure_network`` pinned them (1 h)."""
    wn.options.time.duration = int(hours) * 3600
    wn.options.time.hydraulic_timestep = int(hyd_step)
    return wn


def inject_demand(wn, demand: pd.DataFrame):
    """``run_hydraulics``' convention: base 1 L/s, pattern = the series in L/s; junctions without a
    column pinned to zero."""
    cols = [c for c in demand.columns if c != "month"]
    for node in cols:
        j = wn.get_node(node)
        wn.add_pattern(f"P{node}", demand[node].to_list())
        j.demand_timeseries_list[0].base_value = UNIT_BASE_SI
        j.demand_timeseries_list[0].pattern_name = f"P{node}"
    for node in set(wn.junction_name_list) - set(cols):
        slot = wn.get_node(node).demand_timeseries_list[0]
        slot.base_value, slot.pattern_name = 0.0, None
    return wn


def mediators(wn) -> list[str]:
    """Every pump, and every link a control acts on (pumps, level-controlled valves and pipes)."""
    out = set(wn.pump_name_list)
    for _, c in wn.controls():
        for a in c.actions():
            obj, attr = a.target()
            if attr in ("status", "setting") and getattr(obj, "name", None) in wn.link_name_list:
                out.add(obj.name)
    return sorted(out)


def simulate(wn, demand: pd.DataFrame, hours: int, hyd_step: int = 3600, pmin_nodes=None,
             dtype=np.float32) -> dict:
    """EPANET 2.2 through the toolkit step loop. Per hour h (t = 3600 h):

    snap        link flows at t (what stored ``flows`` hold), L/s, ``dtype``
    med_status / med_setting   mediator links at t
    tank_head, tank_vol        tanks at t
    pmin        min pressure at t over ``pmin_nodes`` (default: junctions that carry demand)
    switches    status changes per pump over all solver steps

    Integrity: ``completed`` (the solve reached the horizon), ``warnings`` (EPANET code -> number of
    solver steps: 1 unbalanced, 2 disconnected, 3/4/5 pumps/valves cannot deliver, 6 negative pressure).
    """
    wn = inject_demand(set_horizon(copy.deepcopy(wn), hours, hyd_step), demand.iloc[:hours])
    links, tanks, pumps = list(wn.link_name_list), list(wn.tank_name_list), list(wn.pump_name_list)
    meds = mediators(wn)
    if pmin_nodes is None:
        dcols = [c for c in demand.columns if c != "month"]
        pmin_nodes = [c for c in dcols if float(np.abs(demand[c].iloc[:hours]).max()) > 0]
    jset = set(wn.junction_name_list)
    juncs = [j for j in pmin_nodes if j in jset] or list(wn.junction_name_list)
    d = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=scratch_root()))
    t0 = time.time()
    try:
        inp = str(d / "run.inp")
        write_inpfile(wn, inp, units=wn.options.hydraulic.inpfile_units, version=2.2)
        en = toolkit.ENepanet(version=2.2)
        en.ENopen(inp, str(d / "run.rpt"), str(d / "run.bin"))
        li = [en.ENgetlinkindex(x) for x in links]
        pi = [en.ENgetlinkindex(x) for x in pumps]
        mi = [en.ENgetlinkindex(x) for x in meds]
        ti = [en.ENgetnodeindex(x) for x in tanks]
        ji = [en.ENgetnodeindex(x) for x in juncs]
        lib, prj, val = en.ENlib, en._project, ctypes.c_double()
        get_l, get_n = lib.EN_getlinkvalue, lib.EN_getnodevalue

        def lv(idx, code):
            out = np.empty(len(idx))
            for j, i in enumerate(idx):
                get_l(prj, i, code, ctypes.byref(val))
                out[j] = val.value
            return out

        def nv(idx, code):
            out = np.empty(len(idx))
            for j, i in enumerate(idx):
                get_n(prj, i, code, ctypes.byref(val))
                out[j] = val.value
            return out

        L, P, M, T = len(links), len(pumps), len(meds), len(tanks)
        snap = np.zeros((hours, L), dtype)
        warn, t_end = Counter(), 0
        run_h, next_h = lib.EN_runH, lib.EN_nextH
        lT, lS = ctypes.c_long(), ctypes.c_long()
        mstat, mset = np.zeros((hours, M)), np.zeros((hours, M))
        thead, tvol, pmin = np.zeros((hours, T)), np.zeros((hours + 1, T)), np.zeros(hours)
        switches, last, n_steps = np.zeros(P), None, 0
        en.ENopenH()
        en.ENinitH(0)
        while True:
            code = run_h(prj, ctypes.byref(lT))
            if code >= 100:
                raise RuntimeError(f"EPANET error {code} at t = {lT.value} s")
            if code:
                warn[int(code)] += 1
            t = int(lT.value)
            t_end = t
            h = t // 3600
            st = lv(pi, EN_STATUS) if P else np.zeros(0)
            if t % 3600 == 0 and h <= hours:
                if T:
                    tvol[h] = nv(ti, EN_TANKVOL)
                if h < hours:
                    snap[h] = lv(li, EN_FLOW)
                    if M:
                        mstat[h], mset[h] = lv(mi, EN_STATUS), lv(mi, EN_SETTING)
                    if T:
                        thead[h] = nv(ti, EN_HEAD)
                    pmin[h] = nv(ji, EN_PRESSURE).min()
            if last is not None:
                switches += st != last
            last = st
            code = next_h(prj, ctypes.byref(lS))
            if code >= 100:
                raise RuntimeError(f"EPANET error {code} after t = {t} s")
            n_steps += 1
            if int(lS.value) <= 0:
                break
        en.ENcloseH()
        en.ENclose()
    finally:
        _remove(d)
    return {"links": links, "pumps": pumps, "meds": meds, "tanks": tanks, "snap": snap,
            "completed": t_end >= hours * 3600, "t_end_h": t_end / 3600.0, "warnings": dict(warn),
            "med_status": mstat, "med_setting": mset, "tank_head": thead, "tank_vol": tvol,
            "pmin": pmin, "switches": switches, "n_steps": n_steps, "seconds": time.time() - t0}


def run_ok(r: dict) -> bool:
    """Reached the horizon with no unconverged step."""
    return bool(r["completed"]) and int(r["warnings"].get(1, 0)) == 0


def blocked_model(wn, demand: pd.DataFrame, traj: dict, hours: int):
    """Mediator-blocked twin (M1) of a configured model.

    All controls are removed; every mediator link follows ``traj`` (its status / setting at each
    hour, as hourly time controls); every tank becomes a reservoir whose head follows traj's tank
    head (fixed-head boundary). Where the tank sat at a level limit, the incident pipes EPANET had
    closed stay closed. Without tanks the twin holds no state: the snapshot at hour h depends only on
    that hour's demand, boundary heads and statuses -- so B(D, traj0) - B(D0, traj0) is the response
    with the hub (pumps, valves, tanks) held on the no-drift trajectory, i.e. demand transit alone.

    Returns (model, demand frame, info: hours without a head reference).
    """
    d = wntr.network.to_dict(wn)
    d["controls"] = []
    dem = demand.iloc[:hours].copy()
    nodes = []
    for nd in d["nodes"]:
        if nd["node_type"] != "Tank":
            nodes.append(nd)
            continue
        t, k = nd["name"], traj["tanks"].index(nd["name"])
        pat = f"HEAD_{t}"
        d["patterns"].append({"name": pat, "multipliers": list(map(float, traj["tank_head"][:hours, k]))})
        nodes.append({"name": t, "coordinates": nd.get("coordinates"), "node_type": "Reservoir",
                      "base_head": 1.0, "head_pattern_name": pat})
    d["nodes"] = nodes
    wb = wntr.network.from_dict(d)
    ix = {m: i for i, m in enumerate(traj["meds"])}
    LS = wntr.network.LinkStatus
    lix = {l: i for i, l in enumerate(traj["links"])}
    for k, t in enumerate(traj["tanks"]):
        tn = wn.get_node(t)
        hd = traj["tank_head"][:hours, k]
        at_lim = (hd >= tn.elevation + tn.max_level - LEVEL_TOL) | (hd <= tn.elevation + tn.min_level + LEVEL_TOL)
        for ln in wn.get_links_for_node(t):
            if ln in ix:
                continue                                    # mediators follow their own trajectory
            shut = at_lim & (np.abs(traj["snap"][:hours, lix[ln]]) < 1e-6)
            if not shut.any():
                continue
            lk = wb.get_link(ln)
            if shut[0]:
                lk.initial_status = LS.Closed
            for h in range(1, hours):
                if shut[h] != shut[h - 1]:
                    act = ControlAction(lk, "status", LS.Closed if shut[h] else LS.Open)
                    wb.add_control(f"lim_{ln}_{h}", Control._time_control(wb, h * 3600, "SIM_TIME", False, act))
    for m in traj["meds"]:
        lk, s, z = wb.get_link(m), traj["med_status"][:hours, ix[m]], traj["med_setting"][:hours, ix[m]]
        kind = "pump" if m in wb.pump_name_list else "valve" if m in wb.valve_name_list else "pipe"

        def state(h, s=s, z=z, kind=kind):
            # 'closed' | 'open' | ('speed', s) for pumps | ('active', setting) for valves
            if s[h] < 0.5:
                return "closed"
            if kind == "pump":
                return ("speed", round(float(z[h]), 6))
            if kind == "valve" and abs(z[h]) < VALVE_OPEN_SETTING:
                return ("active", float(z[h]))
            return "open"

        prev = state(0)
        if prev == "closed":
            lk.initial_status = LS.Closed
        elif prev == "open":
            lk.initial_status = LS.Open
        elif prev[0] == "speed":
            lk.initial_status, lk.initial_setting = LS.Open, prev[1]
        else:
            lk.initial_status, lk.initial_setting = LS.Active, prev[1]
        for h in range(1, hours):
            cur = state(h)
            if cur == prev:
                continue
            if cur == "closed":
                act = ControlAction(lk, "status", LS.Closed)
            elif cur == "open":
                act = ControlAction(lk, "status", LS.Open)
            elif cur[0] == "speed":
                act = ControlAction(lk, "base_speed", cur[1])      # a speed > 0 also opens the pump
            else:
                act = ControlAction(lk, "setting", cur[1])
            wb.add_control(f"blk_{m}_{h}", Control._time_control(wb, h * 3600, "SIM_TIME", False, act))
            prev = cur
    return wb, dem, {"ill_posed_hours": head_reference_gaps(wb, traj, hours)}


def head_reference_gaps(wb, traj: dict, hours: int) -> int:
    """Hours at which some junction of the blocked network has no open path to a fixed-head node.
    Mediators closed at the hour are removed; every other link counts as open (a check valve shut by
    reverse flow is not seen, so gaps can only be under-counted)."""
    ix = {m: i for i, m in enumerate(traj["meds"])}
    fixed, juncs = set(wb.reservoir_name_list), set(wb.junction_name_list)
    edges = [(wb.get_link(l).start_node_name, wb.get_link(l).end_node_name, l) for l in wb.link_name_list]
    always = [(u, v) for u, v, l in edges if l not in ix]
    med_e = [(u, v, ix[l]) for u, v, l in edges if l in ix]
    st = traj["med_status"][:hours] > 0.5
    gaps, cache = 0, {}
    for h in range(hours):
        key = st[h].tobytes()
        if key not in cache:
            G = nx.Graph()
            G.add_nodes_from(wb.node_name_list)
            G.add_edges_from(always)
            G.add_edges_from([(u, v) for u, v, k in med_e if st[h, k]])
            cache[key] = any((c & juncs) and not (c & fixed) for c in map(set, nx.connected_components(G)))
        gaps += int(cache[key])
    return gaps
