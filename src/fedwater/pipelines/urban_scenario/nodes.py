"""Urban scenario: land-use portfolios, drift schedule, assignment timeline.

Model
-----
A junction is a *city block*, not a single building: at the network's design
operating point each node carries hundreds of plots. Two factors drive demand:

* ``income``    -> building *standards* mix -> mean consumption of one
                   RESIDENTIAL plot. Income is residential-only: it enters the
                   level only through the residential intensity term.
* ``land_use``  -> the block's mix of residential / commercial / industrial
                   plots -> BOTH the demand LEVEL (via the plot-intensity
                   table, dialled by ``scenario.beta``) and the demand SHAPE
                   (via per-sector daily and weekly signatures, applied in
                   ``demand_synthesis``).

``land_use`` replaces the former ``density`` axis. Density was volume-neutral
by construction — node volume was a pure function of income and the month
series was renormalised to it — and its hydraulic direction was backwards:
higher density widened the peak-time dispersion, FLATTENING the aggregate at
constant volume and so RAISING minimum pressure. A densification drift
relieved the network instead of stressing it. Land use fixes both: sectors
differ in level and in temporal signature, and the signature is the part that
survives the FL pipeline's per-client MinMax scaler.

Every node's baseline volume is anchored to the network's hydraulic design
point: ``anchor = inp_base_demand * anchor_scale``. A single global
calibration constant rescales the initial scenario onto the anchor total —
recorded, never silent, and never a failure criterion (see ``build_portfolios``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400
ANCHOR_DAYS_PER_MONTH = 30  # anchors are monthly volumes on a 30-day basis


# --------------------------------------------------------------------------
# income -> residential plot intensity (unchanged from V0)
# --------------------------------------------------------------------------
def _mean_unit_volume(standards_mix: dict, unit_m3_month: dict) -> float:
    """Expected m3/month of one unit under a standards mix."""
    return sum(unit_m3_month[int(s)] * w for s, w in standards_mix.items())


def build_income_factors(buildings: dict) -> pd.DataFrame:
    """Mean consumption of one residential unit per income, from the standards
    table.

    Kept verbatim from V0. Under the land-use model ``income_factor`` is no
    longer the level driver — ``mean_unit_m3_month`` is, as the RESIDENTIAL
    entry of the plot-intensity table. The ratio column is retained because it
    is a cheap, auditable statement of the income ladder and V5 reads it.
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


# --------------------------------------------------------------------------
# land-use level model
# --------------------------------------------------------------------------
def plot_intensity(income_factors: pd.DataFrame, income: str,
                   land_use: dict) -> dict:
    """m3/month per plot, by sector, at a given income.

    ``intensity[residential]`` is NOT a free parameter: it is
    ``mean_unit_m3_month(income)`` from the standards table, which is what
    keeps income a residential-only attribute by construction. The two
    non-residential intensities are the only new assumptions in the model, and
    they are quoted relative to ONE MEDIUM-income residential plot so that a
    change of income map cannot silently rescale them.
    """
    m = income_factors.set_index("income")["mean_unit_m3_month"]
    return {"residential": float(m[income]),
            **{s: r * float(m["medium"])
               for s, r in land_use["intensity_rel"].items()}}


def build_landuse_factors(income_factors: pd.DataFrame, land_use: dict,
                          scenario: dict) -> pd.DataFrame:
    """Level factor per (income, land_use).

    Plot COUNT is conserved across a land-use change, so the level factor is
    simply the ratio of mean plot intensities. ``scenario.beta`` dials it:

        level_factor = (mean_plot_intensity / reference) ** beta

    ``beta = 0`` is shape-only — every factor is exactly 1.0, so the land-use
    map is volume-neutral and reproduces the V0 hydraulic loading. ``beta = 1``
    is the full level effect. Low beta is the default posture: at ``beta = 0``
    a residential->commercial drift still moves the MinMax-scaled weekly
    profile substantially, so the level effect carries hydraulic RISK without
    carrying most of the regime INFORMATION.
    """
    sectors = land_use["sectors"]
    mixes = land_use["mix"]
    beta = float(scenario["beta"])

    # Fail on a YAML typo here rather than with a KeyError deep inside EPANET.
    unknown = {s for mix in mixes.values() for s in mix} | set(land_use["intensity_rel"])
    if not unknown <= set(sectors):
        raise ValueError(
            f"land_use references sectors {sorted(unknown - set(sectors))} that "
            f"are not defined in land_use.sectors ({sorted(sectors)}).")
    for name, mix in mixes.items():
        total = sum(mix.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"land_use.mix['{name}'] plot shares sum to "
                             f"{total:.6f}, expected 1.0.")

    def mean_plot(inc: str, lu: str) -> float:
        intensity = plot_intensity(income_factors, inc, land_use)
        return sum(w * intensity[s] for s, w in mixes[lu].items())

    ref_income, ref_land_use = land_use["reference"]
    ref = mean_plot(ref_income, ref_land_use)
    return pd.DataFrame(
        [{"income": inc, "land_use": lu, "beta": beta,
          "mean_plot_m3_month": mean_plot(inc, lu),
          "level_factor": (mean_plot(inc, lu) / ref) ** beta}
         for inc in income_factors["income"] for lu in mixes]
    )


# --------------------------------------------------------------------------
# portfolios
# --------------------------------------------------------------------------
def _plots(node: str, district: str, income: str, land_use_code: str,
           anchor: float, volume: float, income_factors: pd.DataFrame,
           land_use: dict) -> list[dict]:
    """Split one node's monthly volume into sector cohorts.

    Two guards, each found by a failure:

    * **Zero-volume nodes are dropped entirely.** Zero-base-demand trunk
      junctions (node ``110`` is the one of record) would produce a cohort set
      whose volumes sum to zero; the volume-share weighting in
      ``synthesize_demands`` is then 0/0, the NaN reaches the EPANET pattern,
      and the solver dies with ``Error 200 - one or more errors in input
      file``. The synthesiser guards again, independently.
    * **Cohorts below ``min_plots`` are dropped and the remainder
      renormalised**, so the node's volume is preserved EXACTLY instead of
      leaking away with the dropped cohort.
    """
    if not np.isfinite(volume) or volume <= 0:
        return []
    intensity = plot_intensity(income_factors, income, land_use)
    mix = land_use["mix"][land_use_code]
    min_plots = float(land_use["min_plots"])

    total_plots = volume / sum(w * intensity[s] for s, w in mix.items())
    kept = [(s, w) for s, w in mix.items() if total_plots * w >= min_plots]
    if not kept:                       # degenerate node: keep the dominant sector
        kept = [max(mix.items(), key=lambda kv: kv[1])]
    norm = sum(w * intensity[s] for s, w in kept)

    rows = []
    for sector, w in kept:
        sector_volume = volume * w * intensity[sector] / norm
        rows.append({
            "node": node, "district": district,
            "income": income, "land_use": land_use_code, "sector": sector,
            "plots": sector_volume / intensity[sector],
            "sector_volume_m3_month": sector_volume,
            "anchor_m3_month": anchor, "volume_m3_month": volume,
        })
    return rows


def build_portfolios(wn, districts: dict, landuse_factors: pd.DataFrame,
                     income_factors: pd.DataFrame, scenario: dict,
                     land_use: dict, hydraulics: dict) -> pd.DataFrame:
    """Initial (month-0) portfolio for every node: one row per sector cohort.

    A cohort's ``plots`` count is what drives crowd smoothing downstream (the
    sqrt-N law). Establishments are few per block, so commercial and
    industrial cohorts come out naturally noisier than residential ones with
    no extra parameter.

    Calibration renormalises the month-0 total volume onto the anchor total.
    It is a DIAGNOSTIC, not a gate: because it rescales every node by the same
    constant it is hydraulically a no-op, and a value of, say, 0.4 only says
    the map's raw factors ran hot before renormalisation. V0 raised outside
    [0.5, 2.0], which short-circuited the feasibility search before it reached
    the real boundary (the pressure floor). The number is recorded on every row
    and reported by ``sim_validation``; nothing raises here.
    """
    level = landuse_factors.set_index(["income", "land_use"])["level_factor"].to_dict()
    mapping = dict(zip(districts["districts"].keys(),
                       scenario["income_landuse_mapping"]))
    anchor_scale = float(hydraulics["anchor_scale"])

    rows = []
    for district, nodes in districts["districts"].items():
        income, land_use_code = mapping[district]
        for node in nodes:
            base_si = wn.get_node(node).demand_timeseries_list[0].base_value  # m3/s
            anchor = base_si * anchor_scale * SECONDS_PER_DAY * ANCHOR_DAYS_PER_MONTH
            rows += _plots(node, district, income, land_use_code, anchor,
                           anchor * level[(income, land_use_code)],
                           income_factors, land_use)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No node carried a positive volume — check "
                         "anchor_scale and the income/land-use map.")

    per_node = df.drop_duplicates("node")
    calib = per_node["anchor_m3_month"].sum() / per_node["volume_m3_month"].sum()
    for col in ("volume_m3_month", "sector_volume_m3_month", "plots"):
        df[col] *= calib
    df["calibration"] = calib
    return df


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------
def build_drift_schedule(wn, districts: dict, scenario: dict,
                         seed: int) -> pd.DataFrame:
    """Drift as diffusion on the target district's subgraph — ground truth.

    From ``seed_node``, each post-warmup month converts (with probability
    ``growth_chance``) up to ``max_neighbors_per_month`` untouched neighbours.
    Note that ``max_neighbors_per_month`` limits the WIDTH of the front, not
    its depth: the front advances one BFS ring per month, so a short horizon
    leaves the district only partially converted.

    Output: one row per drifted node with the month it switches and its target
    ``(to_income, to_land_use)``. Targets are carried PER ROW — the timeline
    builder reads them per node rather than from a single scalar — so a
    multi-district drift is a schema-compatible extension later. The shipped
    config drives one district.

    This table IS the drift ground truth consumed by evaluation.
    """
    drift = scenario["drift"]
    warmup = int(drift["warmup_months"])
    if warmup < 1:
        # apply_drift_ramp blends the new regime against the month BEFORE the
        # switch; a node drifting at month 0 has no previous month to blend
        # from and raises a KeyError deep in demand_synthesis.
        raise ValueError(f"drift.warmup_months must be >= 1 (got {warmup}).")

    rng = np.random.default_rng(seed)
    district = drift["tgt_district"]
    nodes = set(districts["districts"][district])

    G = wn.to_graph().to_undirected().subgraph(nodes)
    seed_node = str(drift["seed_node"])
    if seed_node not in nodes:
        raise ValueError(f"seed_node {seed_node} not in {district}")

    drifted = {seed_node: warmup}
    frontier = {seed_node}
    for month in range(warmup + 1, scenario["n_months"]):
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
          "to_income": drift["to_income"], "to_land_use": drift["to_land_use"]}
         for n, m in sorted(drifted.items(), key=lambda kv: kv[1])]
    )


def evolve_assignments(portfolios_t0: pd.DataFrame,
                       drift_schedule: pd.DataFrame,
                       landuse_factors: pd.DataFrame,
                       income_factors: pd.DataFrame,
                       land_use: dict, scenario: dict) -> pd.DataFrame:
    """Tidy timeline: (month x node x sector cohort).

    Drifted nodes are rebuilt from their new ``(to_income, to_land_use)`` from
    their drift month onward, read PER ROW of the schedule. The calibration
    constant stays frozen at its month-0 value: drift genuinely changes total
    demand, and feasibility is checked downstream by the pressure floor rather
    than absorbed here.

    Note one inherited asymmetry, kept because the POC was validated with it:
    at month 0 the ``min_plots`` cohort test is evaluated on the PRE-calibration
    volume, while after a switch it sees the post-calibration volume. With
    ``calibration`` far from 1.0 a marginal cohort can therefore appear or
    disappear across the switch. It is a boundary effect on cohorts holding
    less than one plot, so it moves a negligible share of volume.
    """
    level = landuse_factors.set_index(["income", "land_use"])["level_factor"].to_dict()
    calib = float(portfolios_t0["calibration"].iloc[0])
    schedule = (drift_schedule.set_index("node")
                if len(drift_schedule) else None)

    frames = []
    for month in range(int(scenario["n_months"])):
        snap = portfolios_t0.copy()
        switch = ([n for n in schedule.index[schedule["drift_month"] <= month]]
                  if schedule is not None else [])
        # Nodes in the schedule that never entered the portfolio (zero-demand
        # trunk junctions) simply have no rows to switch — their "drift" has no
        # observable, which apply_drift_ramp also handles.
        if switch:
            base = snap[snap["node"].isin(switch)].drop_duplicates("node")
            new_rows = []
            for _, r in base.iterrows():
                to_income = schedule.loc[r["node"], "to_income"]
                to_land_use = schedule.loc[r["node"], "to_land_use"]
                anchor = r["anchor_m3_month"]
                new_rows += _plots(
                    r["node"], r["district"], to_income, to_land_use, anchor,
                    anchor * calib * level[(to_income, to_land_use)],
                    income_factors, land_use)
            snap = pd.concat(
                [snap[~snap["node"].isin(switch)],
                 pd.DataFrame(new_rows).assign(calibration=calib)],
                ignore_index=True)
        frames.append(snap.assign(month=month))
    return pd.concat(frames, ignore_index=True)
