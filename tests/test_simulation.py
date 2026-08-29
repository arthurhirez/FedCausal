"""fedwater simulation test suite.

Fast tests run on synthetic fixtures; one integration test runs EPANET on the
real Graeme network for two simulated days (seconds, not minutes).

LAND-USE REFACTOR: the density-era tests that asserted crowd smoothing as a
density ordering are gone, replaced by the design invariants the land-use POC
validated as its ``sanity_suite`` (S1-S15). The S-numbers are quoted in the
docstrings so the two can be read against each other.

Two S-checks are deliberately NOT here. S12 (undrifted baseline is feasible)
and the hydraulic half of S10/S11 belong to ``sim_validation``, which raises
in-pipeline; re-asserting them in pytest would only duplicate V1/V2/V3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import wntr
import yaml

from fedwater.pipelines.demand_synthesis.nodes import (
    apply_drift_ramp,
    sector_day_shape,
    synthesize_demands,
)
from fedwater.pipelines.hydraulics.nodes import run_hydraulics
from fedwater.pipelines.network_prep.nodes import (
    apply_coupling,
    configure_network,
    validate_partition,
)
from fedwater.pipelines.urban_scenario.nodes import (
    build_drift_schedule,
    build_income_factors,
    build_landuse_factors,
    build_portfolios,
    evolve_assignments,
    plot_intensity,
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
    # 2 months x 7 days. A whole number of weeks per month makes the weekly
    # sector signatures (commercial's weekend collapse, industrial's mild dip)
    # directly readable; the simulator itself does not require it, because
    # weekends come from a global day counter.
    return {"n_months": 2, "days_per_month": 7, "resolution_h": 1}


@pytest.fixture(scope="module")
def scenario_small(params):
    s = dict(params["scenario"])
    s["n_months"] = 2
    s["drift"] = {**s["drift"], "warmup_months": 1}
    return s


@pytest.fixture(scope="module")
def landuse_factors(params):
    factors = build_income_factors(params["buildings"])
    return build_landuse_factors(factors, params["land_use"], params["scenario"])


@pytest.fixture(scope="module")
def portfolios_small(wn, districts, params, landuse_factors, scenario_small):
    factors = build_income_factors(params["buildings"])
    return build_portfolios(wn, districts, landuse_factors, factors,
                            scenario_small, params["land_use"],
                            params["hydraulics"])


@pytest.fixture(scope="module")
def timeline_small(wn, districts, params, portfolios_small, landuse_factors,
                   scenario_small):
    factors = build_income_factors(params["buildings"])
    schedule = build_drift_schedule(wn, districts,
                                    {**scenario_small, "n_months": 24}, seed=7)
    return evolve_assignments(portfolios_small, schedule, landuse_factors,
                              factors, params["land_use"], scenario_small)


# --------------------------------------------------------------------------
# network_prep (unchanged by the refactor)
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
# urban_scenario — level model
# --------------------------------------------------------------------------
def test_income_factors_ordered(params):
    f = build_income_factors(params["buildings"]).set_index("income")["income_factor"]
    assert f["low"] < f["medium"] < f["high"]
    assert f["medium"] == 1.0


def test_level_factor_ordering(params):
    """S1: at full beta the level ladder is R < M < C < I."""
    factors = build_income_factors(params["buildings"])
    lf = build_landuse_factors(factors, params["land_use"], {"beta": 1.0})
    order = lf.query("income == 'low'").set_index("land_use")["level_factor"]
    assert (order["residential"] < order["mixed"]
            < order["commercial"] < order["industrial"])


def test_beta_zero_is_volume_neutral(params):
    """S2: beta=0 makes the land-use map carry NO level information, so it
    reproduces the pre-refactor hydraulic loading exactly. This is the control
    condition for every shape-vs-level claim."""
    factors = build_income_factors(params["buildings"])
    lf = build_landuse_factors(factors, params["land_use"], {"beta": 0.0})
    np.testing.assert_allclose(lf["level_factor"], 1.0, rtol=0, atol=1e-12)


def test_residential_intensity_comes_from_the_standards_table(params):
    """Income must remain a residential-only attribute: the residential plot
    intensity IS mean_unit_m3_month, not a free parameter."""
    factors = build_income_factors(params["buildings"])
    expected = factors.set_index("income")["mean_unit_m3_month"]
    for income in factors["income"]:
        intensity = plot_intensity(factors, income, params["land_use"])
        assert intensity["residential"] == pytest.approx(expected[income])
    # the non-residential intensities are pinned to MEDIUM income, so changing
    # the income map cannot silently rescale them
    low = plot_intensity(factors, "low", params["land_use"])
    high = plot_intensity(factors, "high", params["land_use"])
    assert low["commercial"] == pytest.approx(high["commercial"])
    assert low["industrial"] == pytest.approx(high["industrial"])


def test_landuse_config_errors_are_caught_at_build_time(params):
    factors = build_income_factors(params["buildings"])
    bad_mix = {**params["land_use"],
               "mix": {**params["land_use"]["mix"],
                       "residential": {"residential": 0.9}}}
    with pytest.raises(ValueError, match="sum to"):
        build_landuse_factors(factors, bad_mix, {"beta": 1.0})

    bad_sector = {**params["land_use"],
                  "mix": {**params["land_use"]["mix"],
                          "mixed": {"residential": 0.5, "agricultural": 0.5}}}
    with pytest.raises(ValueError, match="not defined"):
        build_landuse_factors(factors, bad_sector, {"beta": 1.0})


# --------------------------------------------------------------------------
# urban_scenario — portfolios
# --------------------------------------------------------------------------
def test_portfolio_calibrated_to_anchor(portfolios_small):
    """S8: calibration renormalises month-0 total volume onto the anchor
    total — exact by construction."""
    per_node = portfolios_small.drop_duplicates("node")
    ratio = per_node["volume_m3_month"].sum() / per_node["anchor_m3_month"].sum()
    assert ratio == pytest.approx(1.0, rel=1e-9)


def test_sector_volumes_sum_to_node_volume(portfolios_small):
    """S6: dropping sub-threshold cohorts renormalises the remainder, so no
    volume leaks away with the dropped cohort."""
    per_node = portfolios_small.drop_duplicates("node").set_index("node")
    by_sector = portfolios_small.groupby("node")["sector_volume_m3_month"].sum()
    gap = (by_sector - per_node["volume_m3_month"]).abs().max()
    assert gap / per_node["volume_m3_month"].max() < 1e-9


def test_implied_plot_intensity_survives_calibration(portfolios_small, params):
    """Calibration scales POPULATION, not thirst: it multiplies volume and
    plots together, so per-plot consumption stays at the table value."""
    factors = build_income_factors(params["buildings"])
    per_plot = (portfolios_small["sector_volume_m3_month"]
                / portfolios_small["plots"])
    residential = per_plot[portfolios_small["sector"] == "residential"]
    expected = factors.set_index("income")["mean_unit_m3_month"]["low"]
    assert residential.round(9).eq(round(expected, 9)).all()


def test_zero_demand_junctions_are_dropped(wn, districts, portfolios_small):
    """S9: nodes absent from the portfolio are EXACTLY the zero-base-demand
    trunk junctions. A cohort with no volume would put NaN into the EPANET
    pattern and fail the solver with 'Error 200'."""
    all_junctions = {n for nodes in districts["districts"].values() for n in nodes}
    missing = all_junctions - set(portfolios_small["node"])
    zero = {n for n in all_junctions
            if wn.get_node(n).demand_timeseries_list[0].base_value == 0}
    assert missing == zero
    assert len(zero) > 0, "fixture no longer exercises the guard"


# --------------------------------------------------------------------------
# urban_scenario — drift
# --------------------------------------------------------------------------
def test_drift_schedule_reproducible_and_bounded(wn, districts, params):
    scenario = params["scenario"]
    a = build_drift_schedule(wn, districts, scenario, seed=11)
    b = build_drift_schedule(wn, districts, scenario, seed=11)
    pd.testing.assert_frame_equal(a, b)
    tgt_nodes = set(districts["districts"][scenario["drift"]["tgt_district"]])
    assert set(a["node"]) <= tgt_nodes
    assert a["drift_month"].min() == scenario["drift"]["warmup_months"]
    assert set(a["to_land_use"]) == {scenario["drift"]["to_land_use"]}


def test_drift_schedule_rejects_zero_warmup(wn, districts, params):
    """apply_drift_ramp blends against the month BEFORE the switch."""
    scenario = {**params["scenario"],
                "drift": {**params["scenario"]["drift"], "warmup_months": 0}}
    with pytest.raises(ValueError, match="warmup_months"):
        build_drift_schedule(wn, districts, scenario, seed=11)


def test_evolve_switches_land_use_at_the_drift_month(
        wn, districts, params, portfolios_small, landuse_factors, scenario_small):
    factors = build_income_factors(params["buildings"])
    schedule = build_drift_schedule(wn, districts,
                                    {**scenario_small, "n_months": 24}, seed=7)
    timeline = evolve_assignments(portfolios_small, schedule, landuse_factors,
                                  factors, params["land_use"], scenario_small)
    target = schedule["to_land_use"].iloc[0]
    switched = set(schedule.loc[schedule["drift_month"] == 1, "node"])
    switched &= set(portfolios_small["node"])   # ignore zero-demand junctions
    assert switched

    before = timeline[(timeline["month"] == 0) & timeline["node"].isin(switched)]
    after = timeline[(timeline["month"] == 1) & timeline["node"].isin(switched)]
    assert set(after["land_use"]) == {target}
    assert target not in set(before["land_use"])
    # untouched nodes are byte-identical across the switch
    stable = set(portfolios_small["node"]) - set(schedule["node"])
    for month in (0, 1):
        snap = timeline[(timeline["month"] == month)
                        & timeline["node"].isin(stable)]
        assert set(snap["land_use"]) == set(
            portfolios_small[portfolios_small["node"].isin(stable)]["land_use"])


# --------------------------------------------------------------------------
# demand_synthesis — sector shapes
# --------------------------------------------------------------------------
def _night_day(shape, t):
    return shape[t < 5].mean() / shape[(t >= 9) & (t < 17)].mean()


def test_sector_night_day_ordering(params):
    """S3: commercial < residential < industrial on the night/day ratio.

    This is the minimum-night-flow signature — the classic DMA diagnostic —
    and the two non-residential sectors move it in OPPOSITE directions, which
    is what makes residential->commercial and residential->industrial
    distinguishable rather than merely different.
    """
    t = np.arange(24) + 0.5
    ratios = {}
    for name, cfg in params["land_use"]["sectors"].items():
        shape = sector_day_shape(t, cfg, 200, np.random.default_rng(0), False,
                                 params["patterns"])
        ratios[name] = _night_day(shape, t)
    assert ratios["commercial"] < ratios["residential"] < ratios["industrial"]


def test_commercial_closes_at_weekends(params):
    """S4: the weekly gate is the signature that survives an affine scaler."""
    t = np.arange(24) + 0.5
    sectors = params["land_use"]["sectors"]
    ratios = {}
    for name, cfg in sectors.items():
        weekday = sector_day_shape(t, cfg, 200, np.random.default_rng(0), False,
                                   params["patterns"])
        weekend = sector_day_shape(t, cfg, 200, np.random.default_rng(0), True,
                                   params["patterns"])
        ratios[name] = weekend.mean() / weekday.mean()
    assert ratios["commercial"] < 0.3
    assert ratios["industrial"] > 0.8
    assert ratios["residential"] > ratios["commercial"]


def test_crowd_smoothing_is_derived_not_injected(params):
    """S5: a small cohort is noisier than a large one, from the 1/sqrt(N) law
    alone — no smoothing parameter. Holds for the plateau sectors too, since
    they share the jitter machinery."""
    t = np.arange(24) + 0.5
    for name, cfg in params["land_use"]["sectors"].items():
        cv = {}
        for plots in (10, 4000):
            reps = [sector_day_shape(t, cfg, plots, np.random.default_rng(s),
                                     False, params["patterns"])
                    for s in range(40)]
            peaks = np.array([r.max() / r.mean() for r in reps])
            cv[plots] = peaks.std()
        assert cv[10] > cv[4000], f"{name}: small cohort not noisier"


def test_plateau_is_confined_to_its_window(params):
    """The boxcar is a difference of two erfs, so the plateau sits inside its
    window and decays to the night floor outside it."""
    t = np.arange(24) + 0.5
    cfg = params["land_use"]["sectors"]["commercial"]
    shape = sector_day_shape(t, cfg, 10_000, np.random.default_rng(0), False,
                             params["patterns"])
    a, b = cfg["window"]
    inside = shape[(t > a + 2) & (t < b - 2)].mean()
    outside = shape[(t < a - 2) | (t > b + 2)].mean()
    assert inside > 5 * outside


# --------------------------------------------------------------------------
# demand_synthesis — series
# --------------------------------------------------------------------------
def test_monthly_volume_exact(timeline_small, params, small_time):
    """S7: the invariant that kills the 1/24 bug class. With the month wobble
    and seasonality switched off, integrated demand equals the portfolio's
    sector-volume total for EVERY node-month."""
    patterns = {**params["patterns"], "month_sigma": 0.0,
                "seasonality_scale": 0.0}
    demand = synthesize_demands(timeline_small, params["land_use"], patterns,
                                small_time, seed=3)
    node_cols = [c for c in demand.columns if c != "month"]
    step_s = small_time["resolution_h"] * 3600
    days = small_time["days_per_month"]

    simulated = (demand.groupby("month")[node_cols].sum() * step_s / 1e3
                 * 30.0 / days)                    # back to the 30-day basis
    target = (timeline_small.groupby(["month", "node"])
              ["sector_volume_m3_month"].sum().unstack("node"))
    target = target.reindex(columns=node_cols)
    np.testing.assert_allclose(simulated.to_numpy(), target.to_numpy(),
                               rtol=1e-9)


def test_zero_volume_nodes_never_reach_the_demand_frame(timeline_small, params,
                                                        small_time):
    demand = synthesize_demands(timeline_small, params["land_use"],
                                params["patterns"], small_time, seed=3)
    node_cols = [c for c in demand.columns if c != "month"]
    assert np.isfinite(demand[node_cols].to_numpy()).all()
    assert (demand[node_cols].to_numpy() > 0).all()


def test_demand_series_conventions(timeline_small, params, small_time):
    """Downstream code indexes demand_series positionally and by label."""
    demand = synthesize_demands(timeline_small, params["land_use"],
                                params["patterns"], small_time, seed=3)
    assert demand.columns[0] == "month"
    assert demand.index.name == "step"
    assert list(demand.index) == list(range(len(demand)))


def test_weekend_differs_from_weekday(timeline_small, params, small_time):
    """Residential keeps its V0 weekly behaviour: the morning peak is shifted
    later and damped at weekends."""
    demand = synthesize_demands(timeline_small, params["land_use"],
                                params["patterns"], small_time, seed=3)
    month0 = timeline_small[timeline_small["month"] == 0].drop_duplicates("node")
    residential = month0[month0["land_use"] == "residential"]["node"].tolist()
    assert residential

    steps = int(24 / small_time["resolution_h"])
    sig = demand[residential].mean(axis=1).to_numpy()
    day_idx = np.arange(len(sig)) // steps
    weekend = (day_idx % 7) >= 5
    morning = (np.arange(len(sig)) % steps == 7)
    late = (np.arange(len(sig)) % steps == 9)
    wk = sig[morning & ~weekend].mean() / sig[late & ~weekend].mean()
    we = sig[morning & weekend].mean() / sig[late & weekend].mean()
    assert wk > we


def test_seasonality_phase(wn, districts, params, landuse_factors):
    """Across a full year, the peak month of total volume matches the config.
    Seasonal amplitude is now VOLUME-WEIGHTED across a node's sectors, so this
    also pins that the weighting did not lose the phase."""
    time = {"n_months": 12, "days_per_month": 7, "resolution_h": 2}
    scenario = {**params["scenario"], "n_months": 12}
    factors = build_income_factors(params["buildings"])
    portfolios = build_portfolios(wn, districts, landuse_factors, factors,
                                  scenario, params["land_use"],
                                  params["hydraulics"])
    schedule = build_drift_schedule(wn, districts, {**scenario, "n_months": 24},
                                    seed=5)
    timeline = evolve_assignments(portfolios, schedule, landuse_factors,
                                  factors, params["land_use"], scenario)
    # neutralize drift for the phase check: keep only never-drifted nodes
    stable = set(timeline["node"]) - set(schedule["node"])
    timeline = timeline[timeline["node"].isin(stable)]

    patterns = {**params["patterns"], "month_sigma": 0.0}
    demand = synthesize_demands(timeline, params["land_use"], patterns, time,
                                seed=5)
    node_cols = [c for c in demand.columns if c != "month"]
    monthly = demand.groupby("month")[node_cols].sum().sum(axis=1)
    assert monthly.idxmax() == params["patterns"]["seasonal_peak_month"]


def test_drift_ramp_is_clamped_to_the_horizon(timeline_small, params,
                                              small_time):
    """A ramp longer than the remaining horizon must be clamped rather than
    broadcast past the end of the frame. growth_chance and
    max_neighbors_per_month decide how late the front finishes, so any horizon
    change can silently re-break this."""
    demand = synthesize_demands(timeline_small, params["land_use"],
                                params["patterns"], small_time, seed=3)
    node = [c for c in demand.columns if c != "month"][0]
    schedule = pd.DataFrame([{"node": node, "district": "District_A",
                              "drift_month": small_time["n_months"] - 1,
                              "to_income": "low", "to_land_use": "commercial"}])
    patterns = {**params["patterns"], "drift_ramp_days": 10_000}
    ramped = apply_drift_ramp(demand, schedule, patterns, small_time)
    assert ramped.shape == demand.shape
    assert np.isfinite(ramped[node].to_numpy()).all()
    assert not np.allclose(ramped[node].to_numpy(), demand[node].to_numpy())


# --------------------------------------------------------------------------
# shape / level orthogonality — the claim the model rests on
# --------------------------------------------------------------------------
def _scaled_weekly_profile(demand, nodes, month, time, reference_months=1):
    """Mean weekly profile after per-client MinMax fitted on the reference
    months — exactly what fl_preprocessing does to every client."""
    steps = int(24 / time["resolution_h"])
    days = time["days_per_month"]
    s = demand[nodes].sum(axis=1)
    ref = s[demand["month"] < reference_months]
    s = (s - ref.min()) / (ref.max() - ref.min())
    day = s.to_numpy().reshape(-1, steps)
    month_of = demand["month"].to_numpy()[::steps]
    global_day = month_of * days + np.tile(np.arange(days), time["n_months"])
    sel = month_of == month
    return np.concatenate([day[sel & (global_day % 7 == w)].mean(0)
                           for w in range(7)])


def _shape_shift(demand, nodes, time, before, after):
    a = _scaled_weekly_profile(demand, nodes, before, time)
    b = _scaled_weekly_profile(demand, nodes, after, time)
    return float(1 - np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_shape_shift_survives_the_scaler_and_is_beta_independent(
        wn, districts, params, beta):
    """S15, the load-bearing claim.

    A residential->commercial drift changes the MinMax-scaled weekly profile
    substantially even at beta=0 — i.e. with NO level change whatsoever — and
    the size of that change barely moves with beta. Level is what threatens
    hydraulic feasibility; SHAPE is what carries the regime information, and
    the scaler preserves shape while destroying level.

    First thing to break if the volume-weighted cohort mixing or the
    month-level normalisation in synthesize_demands regresses.
    """
    time = {"n_months": 4, "days_per_month": 7, "resolution_h": 1}
    scenario = {**params["scenario"], "n_months": 4, "beta": beta,
                "drift": {**params["scenario"]["drift"], "warmup_months": 1,
                          "to_income": "low", "to_land_use": "commercial"}}
    factors = build_income_factors(params["buildings"])
    lf = build_landuse_factors(factors, params["land_use"], scenario)
    portfolios = build_portfolios(wn, districts, lf, factors, scenario,
                                  params["land_use"], params["hydraulics"])
    # Schedule over the long horizon so the front is guaranteed to propagate,
    # then keep ONLY the nodes that have actually switched by the month we
    # compare against. Including nodes that drift at months 4-23 would mix an
    # unchanged majority into the aggregate and hide the shift entirely.
    schedule = build_drift_schedule(wn, districts, {**scenario, "n_months": 24},
                                    seed=42)
    timeline = evolve_assignments(portfolios, schedule, lf, factors,
                                  params["land_use"], scenario)
    patterns = {**params["patterns"], "seasonality_scale": 0.0,
                "month_sigma": 0.0}
    demand = synthesize_demands(timeline, params["land_use"], patterns, time,
                                seed=42)

    switched = set(schedule.loc[schedule["drift_month"] <= 3, "node"])
    target = sorted(switched & set(demand.columns), key=int)
    assert len(target) >= 3, "fixture no longer drifts enough nodes to measure"
    shift = _shape_shift(demand, target, time, before=0, after=3)
    # print(f"\nshape_shift beta={beta}: {shift:.3f}")
    assert shift > 0.6, f"beta={beta}: drift is invisible to the scaler"

    # if beta == 0.0:
    #     # and the level genuinely did not move
    #     before = demand.loc[demand["month"] == 0, target].sum(axis=1).mean()
    #     after = demand.loc[demand["month"] == 3, target].sum(axis=1).mean()
    #     assert abs(after / before - 1) < 0.10   # S14

    before = demand.loc[demand["month"] == 0, target].sum(axis=1).mean()
    after = demand.loc[demand["month"] == 3, target].sum(axis=1).mean()
    ratio = after / before
    if beta == 0.0:
        assert abs(ratio - 1) < 0.10
    else:
        assert ratio > 1.3
# --------------------------------------------------------------------------
# hydraulics — integration (real EPANET, a few simulated days)
# --------------------------------------------------------------------------
def test_mass_balance_end_to_end(wn, districts, params, landuse_factors):
    time = {"n_months": 1, "days_per_month": 2, "resolution_h": 1}
    scenario = {**params["scenario"], "n_months": 1}
    wn_cfg = configure_network(wn, params["hydraulics"], time)
    wn_var, _ = apply_coupling(wn_cfg, districts, {"variant": "baseline"}, seed=1)

    factors = build_income_factors(params["buildings"])
    portfolios = build_portfolios(wn_var, districts, landuse_factors, factors,
                                  scenario, params["land_use"],
                                  params["hydraulics"])
    schedule = build_drift_schedule(wn_var, districts,
                                    {**scenario, "n_months": 24}, seed=9)
    timeline = evolve_assignments(portfolios, schedule, landuse_factors,
                                  factors, params["land_use"], scenario)
    demand = synthesize_demands(timeline, params["land_use"],
                                params["patterns"], time, seed=9)

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
