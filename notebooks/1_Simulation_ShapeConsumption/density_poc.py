"""density_poc — PoC engine for a proposed density -> consumption term.

Today's code (``urban_scenario.nodes``): income sets the LEVEL
(``income_factor``), density only reshapes the building-template MIX
(``unit_share_by_density``) — total_units is pinned by income alone. This
tests a proposed extra term: total_units (and hence volume) also scaled by
a ``multiplier`` when a node's density category changes, holding income
fixed. Per fedwater's own module docstring: income->LEVEL, density->SHAPE;
this PoC checks what happens if density also gets a piece of LEVEL.

Tiers, in order:
1. per-unit invariance   -- does the multiplier keep m3/household honest?
2. calibration/plausibility -- how far can multiplier go before the network-
   wide calibration bound or building counts stop being believable?
3. compounding          -- today's code vs proposed, side by side
4. real EPANET pressure -- the actual hydraulic consequence, via the repo's
   own run_hydraulics + V1-V3 checks, fed a level-rescaled demand series.

Everything here reads real catalog artifacts for the "local" world
(``portfolios_t0``, ``income_factors``, ``wn_variant``, ``demand_series``) —
nothing is re-derived from scratch, and no repo file is modified.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def find_repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "conf" / "base" / "parameters.yml").exists() and (cand / "src" / "fedwater").is_dir():
            return cand
    return None


def load_local_artifacts(root: Path) -> dict:
    """Real, already-built artifacts for the "local" world (no recompute)."""
    data = root / "data"
    with open(data / "02_intermediate" / "wn_variant.pkl", "rb") as f:
        wn = pickle.load(f)
    demand_series = pd.read_parquet(data / "02_intermediate" / "demand_series.parquet")
    portfolios_t0 = pd.read_parquet(data / "02_intermediate" / "portfolios_t0.parquet")
    income_factors = pd.read_csv(data / "03_primary" / "income_factors.csv")
    districts = yaml.safe_load((data / "01_raw" / "districts_graeme.yml").read_text())
    params = yaml.safe_load((root / "conf" / "base" / "parameters.yml").read_text())
    return dict(wn=wn, demand_series=demand_series, portfolios_t0=portfolios_t0,
               income_factors=income_factors, districts=districts,
               buildings=params["buildings"], validation=params["validation"],
               time=params["time"], hydraulics=params["hydraulics"],
               patterns=params["patterns"], scenario=params["scenario"])


def mean_unit_lookup(income_factors: pd.DataFrame) -> dict:
    return dict(zip(income_factors["income"], income_factors["mean_unit_m3_month"]))


def node_baseline(portfolios_t0: pd.DataFrame, node: str) -> dict:
    """Real, already-computed values for one node -- no formula re-derivation.
    unit_share_by_density shares sum to 1.0 per density, so summing the
    already-template-split `units` column reconstructs total_units exactly."""
    rows = portfolios_t0[portfolios_t0["node"] == node]
    if rows.empty:
        raise ValueError(f"node {node!r} not found in portfolios_t0")
    r0 = rows.iloc[0]
    return dict(node=node, income=r0["income"], density=r0["density"],
               anchor_m3_month=float(r0["anchor_m3_month"]),
               calibration=float(r0["calibration"]),
               volume_m3_month=float(r0["volume_m3_month"]),
               total_units=float(rows["units"].sum()))


def density_variant(baseline: dict, buildings: dict, mean_unit: dict,
                    to_density: str, multiplier: float, seed: int = 0) -> dict:
    """The proposed formula for one node: total_units *= multiplier, then
    split across templates by unit_share_by_density[to_density] -- the exact
    per-node loop `build_portfolios` runs, with one extra factor. Returns the
    new cohort rows plus a summary for the invariance/plausibility checks."""
    rng = np.random.default_rng(seed)
    income = baseline["income"]
    new_total_units = baseline["total_units"] * multiplier
    new_volume = new_total_units * mean_unit[income]  # by construction: /total_units below == mean_unit[income]

    rows = []
    for tpl, share in buildings["unit_share_by_density"][to_density].items():
        units = float(new_total_units * share)
        if units < 0.5:
            continue
        lo, hi = buildings["units_per_building"][tpl]
        rows.append(dict(node=baseline["node"], income=income, density=to_density,
                         template=tpl, units=units,
                         n_buildings=max(1, int(round(units / rng.uniform(lo, hi))))))
    return dict(rows=pd.DataFrame(rows), total_units=new_total_units,
               volume_m3_month=new_volume,
               per_unit_m3_month=new_volume / new_total_units)


def calibration_sweep(portfolios_t0: pd.DataFrame, baseline: dict, mean_unit: dict,
                      buildings: dict, to_density: str,
                      multipliers: list[float]) -> pd.DataFrame:
    """Network-wide calibration ratio (sum anchor / sum volume, same formula
    as `build_portfolios`) if this ONE node's volume were replaced per
    `multiplier`, everything else held at today's values. Finds where the
    [0.5, 2.0] plausibility bound breaks."""
    per_node = portfolios_t0.drop_duplicates("node")
    anchor_total = per_node["anchor_m3_month"].sum()
    volume_total_others = anchor_total / per_node["calibration"].iloc[0] - baseline["volume_m3_month"]
    # ^ back out sum(volume) from the ALREADY-applied calibration constant,
    # then remove this node's own contribution so it can be swapped out.
    rows = []
    for m in multipliers:
        variant = density_variant(baseline, buildings, mean_unit, to_density, m)
        new_total_volume = volume_total_others + variant["volume_m3_month"]
        calib = anchor_total / new_total_volume
        rows.append(dict(multiplier=m, node_volume_m3_month=variant["volume_m3_month"],
                         network_calibration=calib, inside_bounds=0.5 <= calib <= 2.0))
    return pd.DataFrame(rows)


def plausibility_table(baseline: dict, buildings: dict, mean_unit: dict,
                       to_density: str, multipliers: list[float]) -> pd.DataFrame:
    """Per-template n_buildings at each multiplier -- eyeball check for
    implausible building counts, not an automatic pass/fail (there's no
    ground truth for "how many buildings is too many"). `multiplier=1.0` IS
    today's actual code behaviour for this density transition (income-only
    total_units, re-split by the new density's mix) -- include it in
    `multipliers` as the reference point, no separate baseline needed."""
    out = []
    for m in multipliers:
        variant = density_variant(baseline, buildings, mean_unit, to_density, m)
        out.append(variant["rows"].assign(multiplier=m))
    return pd.concat(out, ignore_index=True)


def scale_demand_series(demand_series: pd.DataFrame, node: str, multiplier: float) -> pd.DataFrame:
    """Level-only approximation: rescale the node's existing L/s series by
    `multiplier`, holding its diurnal/seasonal SHAPE fixed. A full
    implementation would let the extra units also feed the crowd-smoothing
    (sqrt-N) law in `synthesize_demands`, which would shift the shape a
    little too (more units -> lower peak factor, at the same density
    category) -- this isolates the level effect only, which dominates the
    pressure-floor question this tier is checking."""
    out = demand_series.copy()
    out[node] = out[node] * multiplier
    return out


def run_v1_v3(wn, demand_series: pd.DataFrame, validation: dict) -> dict:
    """The repo's own run_hydraulics + V1-V3, unmodified. V1-V3 normally
    RAISE (Kedro conventions) -- caught here so a sweep can find exactly
    where they break instead of dying on the first failure."""
    from fedwater.pipelines.hydraulics.nodes import run_hydraulics
    from fedwater.pipelines.sim_validation.nodes import check_mass_balance, check_pressures

    pressures, flows, demands_simulated = run_hydraulics(wn, demand_series)
    checks, error = [], None
    try:
        checks.append(check_mass_balance(demands_simulated, demand_series, validation))
    except AssertionError as e:
        error = str(e)
    try:
        checks.append(check_pressures(pressures, demand_series, validation))
    except AssertionError as e:
        error = error or str(e)
    report = pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()
    return dict(pressures=pressures, flows=flows, demands_simulated=demands_simulated,
               report=report, error=error)


def pressure_sweep(wn, demand_series: pd.DataFrame, node: str,
                   multipliers: list[float], validation: dict) -> pd.DataFrame:
    """One real EPANET run per multiplier (via run_v1_v3) -- the actual
    pressure-drop number, not an estimate. Slow: O(len(multipliers)) full
    hydraulic simulations over the whole horizon."""
    node_cols = [c for c in demand_series.columns if c != "month"]
    rows = []
    for m in multipliers:
        scaled = scale_demand_series(demand_series, node, m)
        out = run_v1_v3(wn, scaled, validation)
        p = out["pressures"][node_cols]
        rows.append(dict(multiplier=m, min_pressure_node=float(out["pressures"][node].min()),
                         min_pressure_network=float(p.to_numpy().min()),
                         v1_v3_error=out["error"]))
    return pd.DataFrame(rows)
