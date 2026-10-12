"""A small network with every topological class, a level-controlled pump and a tank.

    R ==PU==> J0 --P0-- J1 --P1-- J2 --P6-- T
                         \\       /
                          P3   P2
                            \\ /
                             J3 --P4-- J4 --P5-- J5

P0 and P6 are source bridges (the reservoir behind the pump on one side, the tank on the other),
P1-P3 a loop, P4 / P5 branches (downstream {J4, J5} / {J5}), PU a pump switched by the tank's level.
Districts: A = {J1, J2, J4}, B = {J3, J5}. Flows come out in L/s (inp units LPS).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import wntr
import yaml
from wntr.network.controls import Control, ControlAction, ValueCondition

from fedwater.placement import epanet_loop as el

DISTRICTS = {"District_A": ["J1", "J2", "J4"], "District_B": ["J3", "J5"]}
BASE = {"J1": 1.0, "J2": 0.8, "J3": 1.2, "J4": 0.5, "J5": 0.7}       # L/s
DAYS_PER_MONTH = 7


def toy_model(parallel: bool = False, isolated: bool = False):
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=10.0)
    for j in ("J0", "J1", "J2", "J3", "J4", "J5"):
        wn.add_junction(j, base_demand=0.0, elevation=0.0)
    wn.add_tank("T", elevation=20.0, init_level=5.0, min_level=0.0, max_level=10.0, diameter=6.0)
    wn.add_curve("C1", "HEAD", [(0.008, 40.0)])
    wn.add_pump("PU", "R", "J0", "HEAD", "C1")
    pipe = dict(length=200.0, diameter=0.15, roughness=100.0)
    for name, u, v in (("P0", "J0", "J1"), ("P1", "J1", "J2"), ("P2", "J2", "J3"), ("P3", "J3", "J1"),
                       ("P4", "J3", "J4"), ("P5", "J4", "J5"), ("P6", "J2", "T")):
        wn.add_pipe(name, u, v, **pipe)
    if parallel:
        wn.add_pipe("P4b", "J3", "J4", **pipe)
    if isolated:
        wn.add_junction("J8", elevation=0.0)
        wn.add_junction("J9", elevation=0.0)
        wn.add_pipe("P9", "J8", "J9", **pipe)
    t, pu = wn.get_node("T"), wn.get_link("PU")
    LS = wntr.network.LinkStatus
    wn.add_control("pump_on", Control(ValueCondition(t, "level", "<", 3.0), ControlAction(pu, "status", LS.Open)))
    wn.add_control("pump_off", Control(ValueCondition(t, "level", ">", 8.0), ControlAction(pu, "status", LS.Closed)))
    o = wn.options
    o.time.hydraulic_timestep = o.time.pattern_timestep = o.time.report_timestep = 3600
    o.hydraulic.inpfile_units = "LPS"
    o.hydraulic.demand_model = "DD"
    return wn


def toy_demand(hours: int, drift: dict | None = None, start_month: int = 3) -> pd.DataFrame:
    """Hourly demand (L/s) with a diurnal shape and a weekend dip; ``drift`` = {district: factor}
    reshapes that district's nodes from ``start_month`` on (morning peak scaled by ``factor``)."""
    h = np.arange(hours)
    hod = h % 24
    day = (h // 24) % 7
    shape = 1.0 + 0.5 * np.sin(2 * np.pi * (hod - 6) / 24) - 0.15 * (day >= 5)
    month = h // (24 * DAYS_PER_MONTH)
    out = {"month": month}
    for n, b in BASE.items():
        s = shape.copy()
        for d, f in (drift or {}).items():
            if n in DISTRICTS[d]:
                s = np.where(month >= start_month, s * (1.0 + f * np.exp(-0.5 * ((hod - 8) / 2.0) ** 2)), s)
        out[n] = b * s
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- a probe stack
N_MONTHS, DRIFT_MONTH = 10, 3


def fake_stack(root):
    """Two probe worlds of the toy city on disk, laid out as ``store.ensure_stack`` writes them."""
    path = root / "toy" / "manual" / "abc123"
    wn = toy_model()
    Hs = N_MONTHS * DAYS_PER_MONTH * 24
    settle = {"ref_months": 3, "slack": 0.01, "pad": 1, "min_months": 3}
    spec = {"network": "toy", "partition": {"method": "manual"}, "probe": {"settle": settle},
            "plan": {"season_aligned": False}, "seed": 42, "analysis": {"gate": "projection"}}
    path.mkdir(parents=True)
    (path / "districts.yml").write_text(yaml.safe_dump({"districts": DISTRICTS, "assets": {}}))
    (path / "manifest.json").write_text(json.dumps({"status": "ok", "stack_hash": "abc123", "network": "toy",
                                                    "method": "manual", "config_id": "manual__abc123",
                                                    "spec": spec}))
    for d in DISTRICTS:
        D = toy_demand(Hs, {d: 0.8}, start_month=DRIFT_MONTH)
        r = el.simulate(wn, D, Hs)
        assert el.run_ok(r)
        w = path / "worlds" / d
        w.mkdir(parents=True)
        D.to_parquet(w / "demand.parquet")
        F = pd.DataFrame(r["snap"].astype(float), columns=r["links"])
        F.insert(0, "month", D["month"])
        F.to_parquet(w / "flows.parquet")
        pd.DataFrame({"node": DISTRICTS[d], "district": d, "drift_month": DRIFT_MONTH}).to_csv(
            w / "schedule.csv", index=False)
        with open(w / "wn.pkl", "wb") as fh:
            pickle.dump(wn, fh)
        (w / "params.yml").write_text(yaml.safe_dump({
            "time": {"n_months": N_MONTHS, "days_per_month": DAYS_PER_MONTH, "resolution_h": 1},
            "scenario": {"income_landuse_mapping": [["low", "residential"]] * 2,
                         "drift": {"tgt_district": d, "to_income": "low", "to_land_use": "industrial"}},
            "patterns": {"drift_ramp_days": 0}}))
    return path
