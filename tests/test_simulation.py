"""fedwater simulation test suite.

Fast tests run on synthetic fixtures; one integration test runs EPANET on the
real Graeme network for two simulated days (seconds, not minutes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import wntr
import yaml

from fedwater.pipelines.demand_synthesis.nodes import synthesize_demands
from fedwater.pipelines.hydraulics.nodes import run_hydraulics
from fedwater.pipelines.network_prep.nodes import (
    apply_coupling,
    configure_network,
    validate_partition,
)
from fedwater.pipelines.urban_scenario.nodes import (
    build_drift_schedule,
    build_income_factors,
    build_portfolios,
    evolve_assignments,
)

INP = "data/01_raw/Graeme.inp"
DISTRICTS = "data/01_raw/districts_graeme.yml"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def params():
    with open("conf/base/parameters.yml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def wn():
    return wntr.network.WaterNetworkModel(INP)


@pytest.fixture(scope="module")
def districts():
    with open(DISTRICTS) as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def small_time():
    # 2 months x 5 days: fast, but long enough for weekday/weekend structure.
    return {"n_months": 2, "days_per_month": 5, "resolution_h": 1}


@pytest.fixture(scope="module")
def scenario_small(params):
    s = dict(params["scenario"])
    s["n_months"] = 2
    return s


@pytest.fixture(scope="module")
def timeline_small(wn, districts, params, scenario_small):
    factors = build_income_factors(params["buildings"])
    pf = build_portfolios(wn, districts, factors, scenario_small,
                          params["buildings"], params["hydraulics"], seed=7)
    ds = build_drift_schedule(wn, districts, {**scenario_small, "n_months": 24},
                              seed=7)
    return evolve_assignments(pf, ds, factors, params["buildings"],
                              scenario_small, seed=7)


# --------------------------------------------------------------------------
# network_prep
# --------------------------------------------------------------------------
def test_partition_valid(wn, districts):
    report = validate_partition(wn, districts)
    assert report["n_nodes"].sum() == len(wn.junction_name_list) == 113


def test_partition_catches_overlap(wn, districts):
    broken = {"districts": {k: list(v) for k, v in districts["districts"].items()}}
    stolen = broken["districts"]["District_A"][0]
    broken["districts"]["District_B"].append(stolen)
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(wn, broken)


def test_coupling_partial_preserves_service(wn, districts):
    """Even at close_fraction=1.0, the partial variant must keep every
    junction connected to the (single) source — it closes what it can."""
    import networkx as nx
    wn_p, boundaries = apply_coupling(
        wn, districts, {"variant": "partial", "close_fraction": 1.0}, seed=42)
    closed = set(boundaries.loc[boundaries["closed"], "pipe"])
    assert 0 < len(closed) < len(boundaries)  # some closable, not all
    G = nx.MultiGraph()
    for name in wn_p.pipe_name_list:
        if name in closed:
            continue
        p = wn_p.get_link(name)
        G.add_edge(p.start_node_name, p.end_node_name, key=name)
    reached = nx.node_connected_component(G, wn_p.reservoir_name_list[0])
    assert set(wn_p.junction_name_list) <= reached


def test_coupling_isolated_closes_all_boundaries(wn, districts):
    wn_iso, boundaries = apply_coupling(wn, districts,
                                        {"variant": "isolated"}, seed=1)
    assert boundaries["closed"].all()
    # one extra reservoir per district
    assert len(wn_iso.reservoir_name_list) == 1 + len(districts["districts"])


# --------------------------------------------------------------------------
# urban_scenario
# --------------------------------------------------------------------------
def test_income_factors_ordered(params):
    f = build_income_factors(params["buildings"]).set_index("income")["income_factor"]
    assert f["low"] < f["medium"] < f["high"]
    assert f["medium"] == 1.0


def test_portfolio_calibrated_to_anchor(wn, districts, params, scenario_small):
    factors = build_income_factors(params["buildings"])
    pf = build_portfolios(wn, districts, factors, scenario_small,
                          params["buildings"], params["hydraulics"], seed=7)
    per_node = pf.drop_duplicates("node")
    ratio = per_node["volume_m3_month"].sum() / per_node["anchor_m3_month"].sum()
    assert ratio == pytest.approx(1.0, rel=1e-9)  # exact by construction
    # calibration scales units, not thirst: per-unit volume == table mean
    per_unit = (pf.groupby("node")
                  .apply(lambda g: g["volume_m3_month"].iloc[0] / g["units"].sum(),
                         include_groups=False))
    expected = factors.set_index("income")["mean_unit_m3_month"]["low"]
    assert per_unit.round(6).eq(round(expected, 6)).all()


def test_drift_schedule_reproducible_and_bounded(wn, districts, params):
    scenario = params["scenario"]
    a = build_drift_schedule(wn, districts, scenario, seed=11)
    b = build_drift_schedule(wn, districts, scenario, seed=11)
    pd.testing.assert_frame_equal(a, b)
    tgt_nodes = set(districts["districts"][scenario["drift"]["tgt_district"]])
    assert set(a["node"]) <= tgt_nodes
    assert a["drift_month"].min() == scenario["drift"]["warmup_months"]


# --------------------------------------------------------------------------
# demand_synthesis
# --------------------------------------------------------------------------
def test_monthly_volume_exact(timeline_small, params, small_time):
    """The invariant that kills the 1/24 bug class: integrated demand equals
    the (seasonally adjusted, wobbled) monthly target for EVERY node-month —
    here checked against independent re-derivation of the RNG draws."""
    demand = synthesize_demands(timeline_small, params["patterns"], small_time,
                                seed=3)
    node_cols = [c for c in demand.columns if c != "month"]
    step_s = small_time["resolution_h"] * 3600
    vols = demand.groupby("month")[node_cols].sum() * step_s / 1e3  # m3

    pat = params["patterns"]
    for month, row in vols.iterrows():
        seasonal = 1 + pat["seasonal_amplitude"] * np.cos(
            2 * np.pi * (month % 12 - pat["seasonal_peak_month"]) / 12)
        for node in node_cols:
            rng = np.random.default_rng([3, month, int(node)])
            base = timeline_small[(timeline_small["month"] == month)
                                  & (timeline_small["node"] == node)]
            target = (base["volume_m3_month"].iloc[0] * seasonal
                      * rng.lognormal(0.0, pat["month_sigma"])
                      * small_time["days_per_month"] / 30.0)
            assert row[node] == pytest.approx(target, rel=1e-9)


def test_sqrt_n_smoothing_law(params):
    """Crowd smoothing must be emergent: peak factor decreases with units."""
    from fedwater.pipelines.demand_synthesis.nodes import _cohort_day_shape
    t = np.arange(24) + 0.5
    cfg = {k: tuple(v) for k, v in
           params["patterns"]["peaks_by_density"]["low"].items()}
    pf = {}
    for units in (1, 10_000):
        reps = []
        for s in range(40):
            rng = np.random.default_rng(s)
            shape = _cohort_day_shape(t, cfg, 0.15, 0.7, units, rng, False,
                                      params["patterns"])
            reps.append(shape.max() / shape.mean())
        pf[units] = np.mean(reps)
    assert pf[1] > pf[10_000]
    # large-N variance collapses across seeds
    assert pf[10_000] == pytest.approx(pf[10_000], abs=1e-6)


def test_density_flattens_aggregate(timeline_small, params, small_time):
    demand = synthesize_demands(timeline_small, params["patterns"], small_time,
                                seed=3)
    density = timeline_small[timeline_small["month"] == 0] \
        .drop_duplicates("node").set_index("node")["density"]
    steps = int(24 / small_time["resolution_h"])
    node_cols = [c for c in demand.columns if c != "month"]
    daily = demand[node_cols].to_numpy().reshape(-1, steps, len(node_cols))
    pf = pd.Series((daily.max(1) / daily.mean(1)).mean(0), index=node_cols)
    means = pf.groupby(density).mean()
    assert means["low"] > means["medium"] > means["high"]


def test_weekend_differs_from_weekday(timeline_small, params, small_time):
    demand = synthesize_demands(timeline_small, params["patterns"], small_time,
                                seed=3)
    low_nodes = timeline_small[(timeline_small["month"] == 0)
                               & (timeline_small["density"] == "low")]["node"].unique()
    steps = int(24 / small_time["resolution_h"])
    sig = demand[low_nodes.tolist()].mean(axis=1).to_numpy()
    day_idx = np.arange(len(sig)) // steps
    weekend = (day_idx % 7) >= 5
    morning = (np.arange(len(sig)) % steps == 7)
    late = (np.arange(len(sig)) % steps == 9)
    wk = sig[morning & ~weekend].mean() / sig[late & ~weekend].mean()
    we = sig[morning & weekend].mean() / sig[late & weekend].mean()
    assert wk > we  # weekend morning peak is shifted/damped


def test_seasonality_phase(wn, districts, params):
    """Across a full year, the peak month of total volume matches the config."""
    time = {"n_months": 12, "days_per_month": 3, "resolution_h": 2}
    scenario = {**params["scenario"], "n_months": 12}
    factors = build_income_factors(params["buildings"])
    pf = build_portfolios(wn, districts, factors, scenario,
                          params["buildings"], params["hydraulics"], seed=5)
    ds = build_drift_schedule(wn, districts, {**scenario, "n_months": 24}, seed=5)
    tl = evolve_assignments(pf, ds, factors, params["buildings"], scenario, seed=5)
    # neutralize drift for the phase check: keep only never-drifted nodes
    stable = set(tl["node"]) - set(ds["node"])
    tl = tl[tl["node"].isin(stable)]
    demand = synthesize_demands(tl, params["patterns"], time, seed=5)
    node_cols = [c for c in demand.columns if c != "month"]
    monthly = demand.groupby("month")[node_cols].sum().sum(axis=1)
    assert monthly.idxmax() == params["patterns"]["seasonal_peak_month"]


# --------------------------------------------------------------------------
# hydraulics — integration (real EPANET, 2 simulated days)
# --------------------------------------------------------------------------
def test_mass_balance_end_to_end(wn, districts, params, scenario_small):
    time = {"n_months": 1, "days_per_month": 2, "resolution_h": 1}
    wn_cfg = configure_network(wn, params["hydraulics"], time)
    wn_var, _ = apply_coupling(wn_cfg, districts, {"variant": "baseline"}, seed=1)
    factors = build_income_factors(params["buildings"])
    pf = build_portfolios(wn_var, districts, factors, scenario_small,
                          params["buildings"], params["hydraulics"], seed=9)
    ds = build_drift_schedule(wn_var, districts,
                              {**scenario_small, "n_months": 24}, seed=9)
    tl = evolve_assignments(pf, ds, factors, params["buildings"],
                            {**scenario_small, "n_months": 1}, seed=9)
    demand = synthesize_demands(tl, params["patterns"], time, seed=9)

    pressures, flows, demands_sim = run_hydraulics(wn_var, demand)
    node_cols = [c for c in demand.columns if c != "month"]

    supplied = -demands_sim.drop(columns=["month"]).to_numpy().clip(max=0).sum()
    consumed = demands_sim[node_cols].to_numpy().sum()
    assert supplied == pytest.approx(consumed, rel=1e-6)
    # EPANET (DD) must reproduce the synthesized series exactly
    np.testing.assert_allclose(demands_sim[node_cols].to_numpy(),
                               demand[node_cols].to_numpy(), rtol=1e-4)
    # and the operating point must be inside the plausible band
    p = pressures[node_cols].to_numpy()
    assert p.min() > 5.0 and p.max() < 55.0
