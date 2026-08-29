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


def check_mass_balance(demands_simulated: pd.DataFrame, demand_series: pd.DataFrame,
                       validation: dict) -> pd.DataFrame:
    node_cols = [c for c in demand_series.columns if c != "month"]
    all_elements = demands_simulated.drop(columns=["month"])
    negative = all_elements.to_numpy().clip(max=0.0)  # reservoirs: negative demand
    supplied = -negative.sum()
    consumed = demands_simulated[node_cols].to_numpy().sum()
    rel_err = abs(supplied - consumed) / consumed
    if rel_err > validation["mass_balance_rtol"]:
        raise AssertionError(f"V1 mass balance violated: rel_err={rel_err:.2e}")

    # V2: EPANET reproduced the synthesized series (DD mode delivers demand).
    sim = demands_simulated[node_cols].to_numpy()
    syn = demand_series[node_cols].to_numpy()
    max_rel = np.abs(sim - syn).max() / syn.mean()
    if max_rel > validation["volume_rtol"]:
        raise AssertionError(f"V2 volume exactness violated: max_rel={max_rel:.2e}")
    return pd.DataFrame([
        {"check": "V1_mass_balance", "value": rel_err, "hard": True, "passed": True},
        {"check": "V2_volume_exactness", "value": max_rel, "hard": True, "passed": True},
    ])


def check_pressures(pressures: pd.DataFrame, demand_series: pd.DataFrame,
                    validation: dict) -> pd.DataFrame:
    # Junctions only: reservoirs report zero pressure by definition.
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
    return pd.DataFrame([
        {"check": "V3_pressure_floor", "value": float(p.to_numpy().min()),
         "hard": True, "passed": True},
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
