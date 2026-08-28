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
V5 consumption sanity implied per-unit m3/month within plausible bounds,
                     per income level.
V6 peak factors      node daily peak factor within [k_lo, k_hi]; mean peak
                     factor must DECREASE with density (crowd smoothing).
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
            f"capacity — lower anchor_scale or drift intensity."
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
    """V5: per-unit consumption by income must sit in a plausible band."""
    lo, hi = validation["unit_m3_month_band"]
    per_node = assignments_timeline.groupby(["month", "node", "income"]).agg(
        volume=("volume_m3_month", "first"), units=("units", "sum")).reset_index()
    per_node["m3_per_unit"] = per_node["volume"] / per_node["units"]

    rows = []
    for income, grp in per_node.groupby("income"):
        med = grp["m3_per_unit"].median()
        rows.append({"check": f"V5_unit_m3_month[{income}]", "value": float(med),
                     "hard": False, "passed": bool(lo <= med <= hi)})
    return pd.DataFrame(rows)


def check_peak_factors(demand_series: pd.DataFrame,
                       assignments_timeline: pd.DataFrame,
                       time: dict, validation: dict) -> pd.DataFrame:
    """V6: daily peak factor plausible per node; decreasing with density."""
    steps_day = int(round(24 / time["resolution_h"]))
    node_cols = [c for c in demand_series.columns if c != "month"]
    density0 = assignments_timeline[assignments_timeline["month"] == 0] \
        .drop_duplicates("node").set_index("node")["density"]

    vals = demand_series[node_cols].to_numpy()
    days = vals.shape[0] // steps_day
    daily = vals[: days * steps_day].reshape(days, steps_day, -1)
    peak_factor = (daily.max(axis=1) / daily.mean(axis=1)).mean(axis=0)
    pf = pd.Series(peak_factor, index=node_cols)

    k_lo, k_hi = validation["peak_factor_band"]
    in_band = float(((pf >= k_lo) & (pf <= k_hi)).mean())
    by_density = pf.groupby(density0).mean()
    ordered = bool(by_density.get("low", np.inf) > by_density.get("high", -np.inf))

    rows = [{"check": "V6_peak_factor_in_band", "value": in_band, "hard": False,
             "passed": bool(in_band >= validation["peak_factor_min_share"])},
            {"check": "V6_smoothing_low_gt_high", "value": float(
                by_density.get("low", np.nan) - by_density.get("high", np.nan)),
             "hard": False, "passed": ordered}]
    for d, v in by_density.items():
        rows.append({"check": f"V6_peak_factor[{d}]", "value": float(v),
                     "hard": False, "passed": True})
    return pd.DataFrame(rows)


def compile_validation_report(*reports: pd.DataFrame) -> pd.DataFrame:
    report = pd.concat(reports, ignore_index=True)
    failed_soft = report[(~report["passed"]) & (~report["hard"])]
    if len(failed_soft):
        print("WARNING — soft checks failed:\n", failed_soft.to_string(index=False))
    return report
