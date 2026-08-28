"""Urban scenario: block portfolios, drift schedule, assignment timeline.

Model
-----
A junction is a *city block*, not a single building: at the network's design
operating point each node carries hundreds of household units. Two factors
drive demand, mirroring the BEPE design:

* ``income``  -> building *standards* mix -> mean consumption per unit
                 -> the node's demand LEVEL (via ``income_factor``).
* ``density`` -> building *template* mix (houses vs apartment blocks)
                 -> the node's demand SHAPE (via peak-time dispersion,
                 handled in demand_synthesis).

Every node's baseline volume is anchored to the network's hydraulic design
point: ``anchor = inp_base_demand * anchor_scale``. A single global
calibration constant rescales income factors so the *initial* scenario hits
the anchor total exactly — recorded, never silent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400
ANCHOR_DAYS_PER_MONTH = 30  # anchors are monthly volumes on a 30-day basis


def _mean_unit_volume(standards_mix: dict, unit_m3_month: dict) -> float:
    """Expected m3/month of one unit under a standards mix."""
    return sum(unit_m3_month[int(s)] * w for s, w in standards_mix.items())


def build_income_factors(buildings: dict) -> pd.DataFrame:
    """Demand-level factor per income, derived from the standards table.

    factor(income) = E[unit volume | income] / E[unit volume | 'medium'] —
    the income->level link is *earned* from the standards table instead of
    being a free parameter.
    """
    unit_vol = {int(k): v for k, v in buildings["unit_m3_month"].items()}
    means = {
        inc: _mean_unit_volume(mix, unit_vol)
        for inc, mix in buildings["standards_by_income"].items()
    }
    ref = means["medium"]
    return pd.DataFrame(
        [
            {"income": inc, "mean_unit_m3_month": m, "income_factor": m / ref}
            for inc, m in means.items()
        ]
    )


def build_portfolios(
    wn, districts: dict, income_factors: pd.DataFrame, scenario: dict,
    buildings: dict, hydraulics: dict, seed: int,
) -> pd.DataFrame:
    """Initial (month-0) portfolio for every node: one row per cohort.

    A cohort groups a node's units by building template; its ``units`` count is
    what drives crowd-smoothing downstream (sqrt-N law).
    """
    rng = np.random.default_rng(seed)
    mapping = dict(zip(districts["districts"].keys(), scenario["income_density_mapping"]))
    factors = dict(zip(income_factors["income"], income_factors["income_factor"]))
    mean_unit = dict(zip(income_factors["income"], income_factors["mean_unit_m3_month"]))
    days = ANCHOR_DAYS_PER_MONTH

    rows = []
    for district, nodes in districts["districts"].items():
        income, density = mapping[district]
        for node in nodes:
            base_si = wn.get_node(node).demand_timeseries_list[0].base_value  # m3/s
            anchor_m3 = base_si * hydraulics["anchor_scale"] * SECONDS_PER_DAY * days
            volume_m3 = anchor_m3 * factors[income]  # pre-calibration
            total_units = volume_m3 / mean_unit[income]
            for tpl, share in buildings["unit_share_by_density"][density].items():
                units = float(total_units * share)
                if units < 0.5:
                    continue
                lo, hi = buildings["units_per_building"][tpl]
                rows.append({
                    "node": node, "district": district,
                    "income": income, "density": density,
                    "template": tpl, "units": units,
                    "n_buildings": max(1, int(round(units / rng.uniform(lo, hi)))),
                    "anchor_m3_month": anchor_m3,
                    "volume_m3_month": volume_m3,
                })
    df = pd.DataFrame(rows)

    # Global calibration: initial total volume == anchor total (design point).
    per_node = df.drop_duplicates("node")
    calib = per_node["anchor_m3_month"].sum() / per_node["volume_m3_month"].sum()
    if not 0.5 <= calib <= 2.0:
        raise ValueError(f"Calibration factor {calib:.3f} outside [0.5, 2.0] — "
                         "income mapping is hydraulically implausible.")
    df["calibration"] = calib
    # Calibration scales POPULATION, not thirst: unit counts absorb the
    # correction so per-unit consumption stays at the standards-table value.
    df["volume_m3_month"] *= calib
    df["units"] *= calib
    return df


def build_drift_schedule(wn, districts: dict, scenario: dict, seed: int) -> pd.DataFrame:
    """Drift as diffusion on the target district's subgraph — ground truth.

    From ``seed_node``, each post-warmup month converts (with probability
    ``growth_chance``) up to ``max_neighbors_per_month`` untouched neighbours.
    Output: one row per drifted node with the month it switches. This table IS
    the drift ground truth consumed by evaluation.
    """

    drift = scenario["drift"]
    rng = np.random.default_rng(seed)
    district = drift["tgt_district"]
    nodes = set(districts["districts"][district])

    G = wn.to_graph().to_undirected().subgraph(nodes)
    seed_node = str(drift["seed_node"])
    if seed_node not in nodes:
        raise ValueError(f"seed_node {seed_node} not in {district}")

    drifted = {seed_node: drift["warmup_months"]}
    frontier = {seed_node}
    for month in range(drift["warmup_months"] + 1, scenario["n_months"]):
        if rng.random() > drift["growth_chance"]:
            continue
        candidates = sorted(
            {nb for f in frontier for nb in G.neighbors(f)} - set(drifted)
        )
        if not candidates:
            break
        take = rng.choice(
            candidates,
            size=min(drift["max_neighbors_per_month"], len(candidates)),
            replace=False,
        )
        for n in take:
            drifted[str(n)] = month
            frontier.add(str(n))

    if len(drifted) < 2:
        raise ValueError("Drift never propagated beyond the seed — check "
                         "seed_node/growth_chance.")
    return pd.DataFrame(
        [{"node": n, "district": district, "drift_month": m,
          "to_income": drift["to_income"], "to_density": drift["to_density"]}
         for n, m in sorted(drifted.items(), key=lambda kv: kv[1])]
    )


def evolve_assignments(
    portfolios_t0: pd.DataFrame, drift_schedule: pd.DataFrame,
    income_factors: pd.DataFrame, buildings: dict, scenario: dict, seed: int,
) -> pd.DataFrame:
    """Tidy timeline: (month x node x cohort). Drifted nodes switch portfolio
    from their drift month onward; volume level moves with the new income via
    the same earned income factors (calibration constant unchanged — drift
    genuinely changes total demand, feasibility is checked downstream)."""
    rng = np.random.default_rng(seed)
    factors = dict(zip(income_factors["income"], income_factors["income_factor"]))
    mean_unit = dict(zip(income_factors["income"], income_factors["mean_unit_m3_month"]))
    drift_at = drift_schedule.set_index("node")["drift_month"].to_dict()

    frames = []
    for month in range(scenario["n_months"]):
        snap = portfolios_t0.copy()
        snap["month"] = month
        switch = [n for n, m in drift_at.items() if m <= month]
        if switch:
            to_inc = drift_schedule["to_income"].iloc[0]
            to_den = drift_schedule["to_density"].iloc[0]
            base = snap[snap["node"].isin(switch)].drop_duplicates("node")
            new_rows = []
            for _, r in base.iterrows():
                vol = r["anchor_m3_month"] * r["calibration"] * factors[to_inc]
                total_units = vol / mean_unit[to_inc]
                for tpl, share in buildings["unit_share_by_density"][to_den].items():
                    units = float(total_units * share)
                    if units < 0.5:
                        continue
                    lo, hi = buildings["units_per_building"][tpl]
                    new_rows.append({**r.to_dict(),
                                     "income": to_inc, "density": to_den,
                                     "template": tpl, "units": units,
                                     "n_buildings": max(1, int(round(units / rng.uniform(lo, hi)))),
                                     "volume_m3_month": vol})
            snap = pd.concat([snap[~snap["node"].isin(switch)],
                              pd.DataFrame(new_rows)], ignore_index=True)
            snap["month"] = month
        frames.append(snap)
    return pd.concat(frames, ignore_index=True)
