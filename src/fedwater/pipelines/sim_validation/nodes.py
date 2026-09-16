"""Validation: physics and plausibility checks embedded as pipeline nodes.

Philosophy: a dataset that fails physics should be impossible to produce —
hard checks RAISE (the Kedro run dies), soft checks are reported. The report
is a catalog artifact, so every generated dataset ships with the evidence it
was checked.

Hard checks
-----------
V1 mass balance      supplied by sources == consumed at junctions (<1e-6 rel).
V2 volume exactness  simulated volume per node-month == synthesized target.
V3 pressure floor    no pressure below ``hard_pressure_floor`` anywhere.

Soft checks (reported, thresholds in parameters)
-----------------------------------------------
V4 pressure band     share of node-hours inside [p_min, p_max] (ABNT NBR
                     12218 residential band by default).
V5 consumption sanity implied m3/month per RESIDENTIAL plot matches the
                     standards table and sits in a plausible band; the two
                     non-residential intensities are recorded.
V6 peak factors      node daily peak factor within [k_lo, k_hi], reported per
                     land-use class; optional ordering assertion.
V7 calibration       recorded, NEVER a gate (see ``check_consumption_sanity``).

Refactor note: V5 and V6 used to speak the density language. V5 read a
``units`` column that no longer exists, and V6 asserted that the mean peak
factor DECREASES with density — which was the V0 crowd-smoothing artifact and
is meaningless once density is gone. Both are restated below in land-use terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_mass_balance(wn, demands_simulated: pd.DataFrame,
                       demand_series: pd.DataFrame,
                       validation: dict) -> pd.DataFrame:
    """V1 continuity and V2 volume exactness.

    STORAGE, part 1 -- the node set. The previous form of V1 read supply as
    "every negative entry in the demand frame", which is exact on a network
    whose only non-junction is a reservoir, and wrong the moment a tank
    exists: EPANET reports a tank's demand as negative while it FILLS, so a
    filling tank was counted as a source. Supply then exceeded consumption by
    the fill volume and V1 raised on a perfectly balanced model. The statement
    that actually holds, with or without storage, is EPANET's own continuity:
    at every step the demands over all nodes sum to zero, reservoirs negative
    when supplying and tanks signed by whether they drain or fill. So supply
    is read off the SOURCE-AND-STORAGE set explicitly -- identical to the old
    computation on Graeme, correct on KY7.

    STORAGE, part 2 -- the denominator, which matters just as much. EPANET
    writes its output in SINGLE PRECISION. On a storage-free network that is
    invisible: Graeme has one reservoir, no cancellation, and V1 reads exactly
    0.0. On KY7 tank T-3 cycles a GROSS throughput of 5.9e5 L/s-steps -- the
    same order as the reservoir's entire horizon supply -- against a net of
    -9.1e3, so the balance is a difference of large, nearly-cancelling
    float32 numbers. Dividing the residual by CONSUMPTION therefore reports an
    error floor that scales with how hard the tanks cycle, not with how well
    continuity holds: 1.05e-6 on this world, tripping a 1e-6 tolerance for a
    physically exact model.

    The residual is normalised by the gross flux through the source/storage
    set instead, which is what the float32 error actually scales with. On a
    network without storage gross == consumed, so Graeme is unchanged. Here it
    reads 5.2e-7, about 4x float32 eps. The consumption-normalised number is
    still reported, because it is the interpretable one; it is just not the
    one to gate on. Loosening the tolerance would have hidden a real leak just
    as effectively.
    """
    node_cols = [c for c in demand_series.columns if c != "month"]
    frame = demands_simulated.drop(columns=["month"])
    storage = [c for c in frame.columns
               if c in set(wn.reservoir_name_list) | set(wn.tank_name_list)]
    if not storage:
        raise AssertionError(
            "V1 mass balance has no source: the model carries neither a "
            "reservoir nor a tank, so nothing supplies the junctions.")

    supplied_t = -frame[storage].to_numpy().sum(axis=1)
    consumed_t = demands_simulated[node_cols].to_numpy().sum(axis=1)
    gross_t = np.abs(frame[storage].to_numpy()).sum(axis=1)

    residual = abs(supplied_t.sum() - consumed_t.sum())
    consumed, gross = consumed_t.sum(), gross_t.sum()
    rel_err = residual / gross
    if rel_err > validation["mass_balance_rtol"]:
        raise AssertionError(
            f"V1 mass balance violated: rel_err={rel_err:.2e} (residual "
            f"{residual:.3e} over gross source/storage flux {gross:.3e}) "
            f"across {len(storage)} node(s) {storage}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        step_err = np.nanmax(np.abs(supplied_t - consumed_t)
                             / np.where(gross_t > 0, gross_t, np.nan))

    # V2: EPANET reproduced the synthesized series (DD mode delivers demand).
    sim = demands_simulated[node_cols].to_numpy()
    syn = demand_series[node_cols].to_numpy()
    max_rel = np.abs(sim - syn).max() / syn.mean()
    if max_rel > validation["volume_rtol"]:
        raise AssertionError(f"V2 volume exactness violated: max_rel={max_rel:.2e}")
    return pd.DataFrame([
        {"check": "V1_mass_balance", "value": float(rel_err), "hard": True,
         "passed": True},
        {"check": "V1b_residual_vs_consumed", "value": float(residual / consumed),
         "hard": False, "passed": True},      # interpretable, not gated
        {"check": "V1c_worst_step", "value": float(step_err), "hard": False,
         "passed": bool(step_err <= max(validation["mass_balance_rtol"], 1e-4))},
        # How much of the network's supply passes through storage on its way
        # to a customer. 1.0 means the tanks are decoration; KY7 runs ~2.0.
        {"check": "V1d_storage_flux_ratio", "value": float(gross / consumed),
         "hard": False, "passed": True},      # recorded, not gated
        {"check": "V2_volume_exactness", "value": float(max_rel), "hard": True,
         "passed": True},
    ])


def check_storage(pressures: pd.DataFrame, wn, validation: dict) -> pd.DataFrame:
    """V8: are the tanks actually cycling, or pinned against a limit?

    A tank's ``pressure`` in EPANET output IS its water level (head minus
    elevation), so this needs no extra simulator output.

    The obvious test -- did the level fall from start to end -- is worthless:
    once a tank locks into its control cycle it reads essentially zero
    regardless of load, because it returns to the same phase. What IS monotone
    in demand is the SHARE OF TIME spent against a limit, so that is what is
    gated. A tank sitting on its floor is not a low-pressure event and the V3
    pressure floor will not see it; it is a capacity failure of the storage
    system and needs its own check.

    The ceiling share is recorded without a gate: a full tank is a tank with
    spare supply, which is a comfortable place to be, not a defect.
    """
    tanks = [t for t in wn.tank_name_list if t in pressures.columns]
    if not tanks:
        return pd.DataFrame([{"check": "V8_storage", "value": 0.0,
                              "hard": False, "passed": True}])
    tol = float(validation.get("tank_level_tol_m", 0.01))
    max_share = float(validation.get("tank_floor_max_share", 0.05))

    rows = []
    for name in tanks:
        tank = wn.get_node(name)
        level = pressures[name].to_numpy()
        floor = float((level <= tank.min_level + tol).mean())
        ceiling = float((level >= tank.max_level - tol).mean())
        rows.append({"check": f"V8_tank_floor_share[{name}]", "value": floor,
                     "hard": False, "passed": bool(floor <= max_share)})
        rows.append({"check": f"V8_tank_ceiling_share[{name}]", "value": ceiling,
                     "hard": False, "passed": True})   # recorded, not gated
        rows.append({"check": f"V8_tank_level_range_m[{name}]",
                     "value": float(level.max() - level.min()),
                     "hard": False, "passed": True})   # recorded, not gated
    return pd.DataFrame(rows)


def check_pressures(pressures: pd.DataFrame, demand_series: pd.DataFrame,
                    validation: dict) -> pd.DataFrame:
    """V3 hard pressure floor and V4 service band, over CONSUMER junctions.

    Scope, stated because it used to be accidental: the columns of
    ``demand_series`` are exactly the junctions that carry a portfolio, i.e.
    the ones with a non-zero base demand in the ``.inp``. Reservoirs and tanks
    are excluded because their "pressure" is a water level, not a service
    pressure. Zero-demand junctions are excluded because they are trunk and
    connector nodes with no customer behind them -- KY7's ``I-Pump-1`` is the
    case that matters: it is the pump's SUCTION node, at elevation 115.7 m
    drawing from a reservoir whose head is 107.3 m, so it sits at -8.6 mca by
    design. Gating it would fail every KY7 world for a non-defect.

    The count of excluded junctions is reported so the exclusion is visible
    rather than inferred.
    """
    junctions = [c for c in demand_series.columns if c != "month"]
    p = pressures[junctions]
    p_min, p_max = validation["pressure_band_mca"]
    floor = validation["hard_pressure_floor_mca"]

    if p.to_numpy().min() < floor:
        worst = p.min().idxmin()
        raise AssertionError(
            f"V3 pressure floor violated: node {worst} reached "
            f"{p.to_numpy().min():.1f} mca (< {floor}). Demand exceeds network "
            f"capacity — lower anchor_scale, scenario.beta, or drift intensity."
        )
    inside = ((p >= p_min) & (p <= p_max)).to_numpy().mean()
    n_excluded = len([c for c in pressures.columns
                      if c != "month" and c not in junctions])
    return pd.DataFrame([
        {"check": "V3_pressure_floor", "value": float(p.to_numpy().min()),
         "hard": True, "passed": True},
        {"check": "V3b_non_consumer_nodes_excluded", "value": float(n_excluded),
         "hard": False, "passed": True},      # recorded, not gated
        {"check": "V4_pressure_band_share", "value": float(inside), "hard": False,
         "passed": bool(inside >= validation["pressure_band_min_share"])},
    ])


def check_consumption_sanity(assignments_timeline: pd.DataFrame,
                             income_factors: pd.DataFrame,
                             validation: dict) -> pd.DataFrame:
    """V5 (per-plot intensity) and V7 (calibration, recorded only).

    V5 is a cheap plumbing assertion rather than a physical discovery. Cohort
    plot counts are back-solved as ``sector_volume / intensity(sector)``, so
    the implied per-plot volume is identically the intensity the portfolio
    builder was handed. What the check is therefore worth is confirming that
    the residential intensity actually came from ``standards_by_income`` — that
    income is still wired to the residential sector and only to it — and that
    the resulting number is physically plausible. The two non-residential
    intensities are recorded without a band, since the residential band does
    not apply to a shop or a factory.

    V7 records the calibration constant. It is NOT a gate. Calibration
    renormalises the month-0 total volume onto the anchor total, rescaling
    every node by the same constant, so it is hydraulically a no-op; a value
    of 0.4 only means the map's raw factors ran hot before renormalisation.
    V0 raised outside [0.5, 2.0], which short-circuited the feasibility search
    before it could reach the real boundary — the pressure floor.
    """
    lo, hi = validation["unit_m3_month_band"]
    expected = income_factors.set_index("income")["mean_unit_m3_month"]

    per_plot = assignments_timeline.assign(
        m3_per_plot=assignments_timeline["sector_volume_m3_month"]
        / assignments_timeline["plots"])

    rows = []
    residential = per_plot[per_plot["sector"] == "residential"]
    for income, grp in residential.groupby("income"):
        med = float(grp["m3_per_plot"].median())
        in_band = lo <= med <= hi
        matches_table = bool(np.isclose(med, float(expected[income]), rtol=1e-6))
        rows.append({"check": f"V5_residential_m3_month[{income}]", "value": med,
                     "hard": False, "passed": bool(in_band and matches_table)})

    for sector, grp in per_plot[per_plot["sector"] != "residential"].groupby("sector"):
        rows.append({"check": f"V5_plot_m3_month[{sector}]",
                     "value": float(grp["m3_per_plot"].median()),
                     "hard": False, "passed": True})  # recorded, not gated

    calib = float(assignments_timeline["calibration"].iloc[0])
    c_lo, c_hi = validation.get("calibration_band", [-np.inf, np.inf])
    rows.append({"check": "V7_calibration", "value": calib,
                 "hard": False, "passed": True})       # recorded, never a gate
    rows.append({"check": "V7_calibration_in_band", "value": calib,
                 "hard": False, "passed": bool(c_lo <= calib <= c_hi)})
    return pd.DataFrame(rows)


def check_peak_factors(demand_series: pd.DataFrame,
                       assignments_timeline: pd.DataFrame,
                       time: dict, validation: dict) -> pd.DataFrame:
    """V6: daily peak factor plausible per node, reported per land-use class.

    V0 asserted that the mean peak factor decreases with density — the sqrt-N
    crowd-smoothing artifact of a model where density only widened the
    peak-time dispersion. Under the land-use model there is no such single
    monotone story to assert a priori: a land-use CLASS is a sector MIX (the
    ``commercial`` class still holds 35% residential plots) and cohorts are
    mixed by VOLUME, so the node-level ordering is an emergent property of the
    intensity table, not something readable off the sector signatures.

    So the class means are always REPORTED, and the ordering is asserted only
    if ``validation.peak_factor_order`` is set — a list of land-use codes in
    expected DECREASING peak-factor order, to be filled in from a measured run
    rather than guessed. Leave it unset (or null) and this node reports without
    asserting.
    """
    steps_day = int(round(24 / time["resolution_h"]))
    node_cols = [c for c in demand_series.columns if c != "month"]
    land_use0 = (assignments_timeline[assignments_timeline["month"] == 0]
                 .drop_duplicates("node").set_index("node")["land_use"])

    vals = demand_series[node_cols].to_numpy()
    days = vals.shape[0] // steps_day
    daily = vals[: days * steps_day].reshape(days, steps_day, -1)
    peak_factor = (daily.max(axis=1) / daily.mean(axis=1)).mean(axis=0)
    pf = pd.Series(peak_factor, index=node_cols)

    k_lo, k_hi = validation["peak_factor_band"]
    in_band = float(((pf >= k_lo) & (pf <= k_hi)).mean())
    by_land_use = pf.groupby(land_use0.reindex(pf.index)).mean()

    rows = [{"check": "V6_peak_factor_in_band", "value": in_band, "hard": False,
             "passed": bool(in_band >= validation["peak_factor_min_share"])}]
    for land_use, v in by_land_use.items():
        rows.append({"check": f"V6_peak_factor[{land_use}]", "value": float(v),
                     "hard": False, "passed": True})   # recorded, not gated

    order = validation.get("peak_factor_order")
    if order:
        present = [lu for lu in order if lu in by_land_use.index]
        if len(present) >= 2:
            gaps = np.diff([by_land_use[lu] for lu in present])
            # Expected DECREASING, so every successive difference must be < 0;
            # the reported value is the worst (largest) gap.
            rows.append({"check": "V6_peak_factor_ordering",
                         "value": float(gaps.max()), "hard": False,
                         "passed": bool((gaps < 0).all())})
    return pd.DataFrame(rows)


def compile_validation_report(*reports: pd.DataFrame) -> pd.DataFrame:
    report = pd.concat(reports, ignore_index=True)
    failed_soft = report[(~report["passed"]) & (~report["hard"])]
    if len(failed_soft):
        print("WARNING — soft checks failed:\n", failed_soft.to_string(index=False))
    return report
