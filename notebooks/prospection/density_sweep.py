"""density_sweep — drift-enabled feasibility sweep for a proposed
density -> consumption term.

Companion to ``density_poc``. That module asked "what does one node's density
multiplier do"; this one asks the question that actually gates the change:
**with density driving LEVEL, and a district fully drifted, how big can
``anchor_scale`` still be?**

Why drift matters here (and why a static sweep can't see it)
-----------------------------------------------------------
``build_portfolios`` sets ``calib = sum(anchor) / sum(volume)`` and then
applies ``volume *= calib``, so total network demand at t0 equals the anchor
total EXACTLY for every consumption_map -- only its distribution across nodes
varies. A static (no-drift) sweep therefore probes the calibration gate and
distributional effects, never total load, which is why V3 never fires there.
``evolve_assignments`` keeps ``calib`` frozen at its t0 value while
recomputing volume, so drift is the only place total demand genuinely departs
from the anchor total. That is the case this module builds.

Two gates, and they decouple
----------------------------
``calib`` is invariant to ``anchor_scale``: volume_pre_calib is proportional
to anchor, so scaling every anchor by k scales numerator and denominator
equally. Hence:

* the CALIBRATION gate [0.5, 2.0] depends only on the consumption_map and the
  factor tables -- ``anchor_scale`` is irrelevant to it;
* the V3 PRESSURE gate scales linearly with ``anchor_scale`` (post-calibration
  demand is ``anchor * f * calib``), so it is the one worth bisecting.

Nothing in the repo is modified: the real ``synthesize_demands``,
``configure_network``, ``run_hydraulics`` and ``check_pressures`` are called
as-is, only fed a portfolio/timeline this module constructs.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

ANCHOR_DAYS_PER_MONTH = 30  # mirrors urban_scenario.nodes


# ---------------------------------------------------------------------------
# the proposed factor
# ---------------------------------------------------------------------------
def mean_units_per_building(buildings: dict) -> dict:
    """E[units per building | density], from tables already in parameters.yml:
    ``unit_share_by_density`` weighted by the midpoint of
    ``units_per_building``. This is what makes the density factor DERIVED
    rather than a free parameter -- the same move ``build_income_factors``
    makes for income via ``standards_by_income``."""
    mean_upb = {tpl: (lo + hi) / 2.0
                for tpl, (lo, hi) in buildings["units_per_building"].items()}
    return {density: sum(share * mean_upb[tpl] for tpl, share in mix.items())
            for density, mix in buildings["unit_share_by_density"].items()}


def build_density_factors(buildings: dict, normalize: str = "medium",
                          reference_map: list | None = None,
                          anchor_weights: pd.Series | None = None,
                          alpha: float = 1.0) -> pd.DataFrame:
    """Density -> demand-LEVEL factor.

    ``alpha`` compresses the swing: the raw ratio E[upb|d] / E[upb|medium] is
    raised to ``alpha`` BEFORE normalization (order matters -- compressing
    after would move the reference point and break the calibration hold).
    ``alpha=1.0`` is the pure derivation; ``alpha=0.5`` roughly halves the
    high/low spread in log terms; ``alpha=0.0`` disables the density term
    entirely (every factor 1.0), which is today's behaviour and a useful
    control. Motivation: on the Graeme network the raw derivation makes
    density a STRONGER demand axis than income (high/low swing ~4.9x vs
    income's 3.1x), which V3 does not survive at anchor_scale 0.05.

    Two normalizations, because the choice decides which bound the calibration
    gate breaks on:

    ``normalize="medium"``
        factor(density) = (E[upb|density] / E[upb|medium]) ** alpha, exactly
        parallel to ``income_factor``. Clean and scenario-independent, but a
        mostly-low-density map then gets ~0.46x today's volume (at alpha=1),
        pushing ``calib`` UP toward the 2.0 ceiling.

    ``normalize="reference_map"``
        rescaled so the anchor-weighted mean factor over ``reference_map``
        (a list of [income, density] pairs, district order) is exactly 1.0 --
        i.e. today's scenario keeps its current calibration and only
        DEPARTURES from it move. Costs scenario-independence: the factor table
        now depends on which map you call "normal". ``anchor_weights`` is a
        district-indexed Series of summed anchor volume; None -> equal weights.
        Caveat seen in practice: if the reference map's high-density districts
        carry little anchor demand, the weighted reference sits near the
        low-density value and inflates ``density_factor[high]``.
    """
    E_raw = mean_units_per_building(buildings)
    # compress FIRST, on the medium-referenced ratio, then normalize
    E = {d: (v / E_raw["medium"]) ** alpha for d, v in E_raw.items()}
    if normalize == "medium":
        ref = E["medium"]
    elif normalize == "reference_map":
        if reference_map is None:
            raise ValueError("normalize='reference_map' needs reference_map")
        densities = [d for _, d in reference_map]
        if anchor_weights is None:
            w = np.ones(len(densities))
        else:
            w = anchor_weights.to_numpy(dtype=float)
            if len(w) != len(densities):
                raise ValueError("anchor_weights length != reference_map length")
        ref = float(np.average([E[d] for d in densities], weights=w))
    else:
        raise ValueError(f"unknown normalize={normalize!r}")
    return pd.DataFrame([{"density": d, "mean_units_per_building": E_raw[d],
                          "density_factor": v / ref} for d, v in E.items()])


# ---------------------------------------------------------------------------
# t0 portfolio with the density term applied
# ---------------------------------------------------------------------------
def apply_density_to_t0(portfolios_t0: pd.DataFrame, dfactors: pd.DataFrame,
                        raise_on_bound: bool = False) -> tuple[pd.DataFrame, float]:
    """``portfolios_t0`` as a patched ``build_portfolios`` would have produced
    it. At t0 no node's density changes, so template shares are untouched and
    the density term is a pure multiplier: undo the existing calibration,
    apply the factor, recompute ``calib = sum(anchor)/sum(volume)``, re-apply.
    Returns (patched frame, new calib). ``raise_on_bound=True`` mimics
    ``build_portfolios``'s own ValueError; default False so a sweep can record
    the failure instead of dying on it."""
    dfac = dict(zip(dfactors["density"], dfactors["density_factor"]))
    df = portfolios_t0.copy()
    old_calib = float(df["calibration"].iloc[0])
    scale = df["density"].map(dfac) / old_calib  # undo old calib, apply factor
    df["volume_m3_month"] = df["volume_m3_month"] * scale
    df["units"] = df["units"] * scale

    per_node = df.drop_duplicates("node")
    calib = per_node["anchor_m3_month"].sum() / per_node["volume_m3_month"].sum()
    if raise_on_bound and not 0.5 <= calib <= 2.0:
        raise ValueError(f"Calibration factor {calib:.3f} outside [0.5, 2.0]")
    df["calibration"] = calib
    df["volume_m3_month"] *= calib
    df["units"] *= calib
    return df, float(calib)


# ---------------------------------------------------------------------------
# the fully-drifted end state (the worst case, without simulating diffusion)
# ---------------------------------------------------------------------------
def fully_drifted_timeline(portfolios_t0: pd.DataFrame, districts: dict,
                          tgt_districts: list[str], to_income: str, to_density: str,
                          income_factors: pd.DataFrame, dfactors: pd.DataFrame,
                          buildings: dict, month: int = 0,
                          seed: int = 0) -> pd.DataFrame:
    """Every node of ``tgt_districts`` converted to (to_income, to_density) at
    once -- the diffusion END STATE, which is the worst case, so there's no
    need to walk months. Mirrors ``evolve_assignments``: volume recomputed
    from ``anchor * calibration * income_factor * density_factor`` with
    ``calibration`` FROZEN at its t0 value (this is exactly why drifted total
    demand departs from the anchor total), units re-split by
    ``unit_share_by_density[to_density]``.

    Single-month frame, shaped for ``synthesize_demands``."""
    rng = np.random.default_rng(seed)
    ifac = dict(zip(income_factors["income"], income_factors["income_factor"]))
    mean_unit = dict(zip(income_factors["income"], income_factors["mean_unit_m3_month"]))
    dfac = dict(zip(dfactors["density"], dfactors["density_factor"]))

    switch = {str(n) for d in tgt_districts for n in districts["districts"][d]}
    keep = portfolios_t0[~portfolios_t0["node"].astype(str).isin(switch)].copy()

    base = (portfolios_t0[portfolios_t0["node"].astype(str).isin(switch)]
            .drop_duplicates("node"))
    new_rows = []
    for _, r in base.iterrows():
        vol = (r["anchor_m3_month"] * r["calibration"]
               * ifac[to_income] * dfac[to_density])
        total_units = vol / mean_unit[to_income]
        for tpl, share in buildings["unit_share_by_density"][to_density].items():
            units = float(total_units * share)
            if units < 0.5:
                continue
            lo, hi = buildings["units_per_building"][tpl]
            new_rows.append({**r.to_dict(), "income": to_income, "density": to_density,
                             "template": tpl, "units": units,
                             "n_buildings": max(1, int(round(units / rng.uniform(lo, hi)))),
                             "volume_m3_month": vol})
    out = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True)
    out["month"] = month
    return out


# ---------------------------------------------------------------------------
# the V3 bisection on anchor_scale
# ---------------------------------------------------------------------------
def scale_anchor(timeline: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Rescale a timeline's volumes as a different ``anchor_scale`` would.
    Valid because anchor (and hence post-calibration volume) is exactly linear
    in ``anchor_scale``, and ``calib`` itself is invariant to it -- so this is
    a rescale, not an approximation. ``units`` deliberately NOT scaled: it
    feeds the sqrt-N crowd-smoothing law, and holding the population fixed
    isolates the pressure question from a simultaneous shape change."""
    out = timeline.copy()
    out["volume_m3_month"] = out["volume_m3_month"] * factor
    out["anchor_m3_month"] = out["anchor_m3_month"] * factor
    return out


def v3_passes(wn, timeline: pd.DataFrame, hydraulics: dict, patterns: dict,
              time: dict, validation: dict, seed: int = 0) -> tuple[bool, float, str | None]:
    """One real hydraulic evaluation of a timeline: real ``synthesize_demands``
    (seasonal peak forced, so the month tested IS the worst month) -> real
    ``configure_network`` -> real ``run_hydraulics`` -> real
    ``check_pressures``. Returns (passed, min_pressure, error)."""
    from fedwater.pipelines.demand_synthesis.nodes import synthesize_demands
    from fedwater.pipelines.network_prep.nodes import configure_network
    from fedwater.pipelines.hydraulics.nodes import run_hydraulics
    from fedwater.pipelines.sim_validation.nodes import check_pressures

    time_short = {**time, "n_months": 1}
    patterns_peak = {**patterns, "seasonal_peak_month": 0}
    tl = timeline.copy()
    tl["month"] = 0  # the single simulated month must BE the seasonal peak
    demand = synthesize_demands(tl, patterns_peak, time_short, seed)

    wn_cfg = configure_network(wn, hydraulics, time_short)
    pressures, _flows, _dem = run_hydraulics(wn_cfg, demand)
    node_cols = [c for c in demand.columns if c != "month"]
    min_p = float(pressures[node_cols].to_numpy().min())
    try:
        check_pressures(pressures, demand, validation)
        return True, min_p, None
    except AssertionError as e:
        return False, min_p, str(e)


def max_feasible_anchor_scale(wn, timeline: pd.DataFrame, hydraulics: dict,
                              patterns: dict, time: dict, validation: dict,
                              lo: float = 0.005, hi: float | None = None,
                              tol: float = 0.002, seed: int = 0) -> dict:
    """Largest ``anchor_scale`` at which this (already drifted) timeline still
    passes V3, by bisection. ``hi`` defaults to today's configured value, so
    a result AT ``hi`` means "today's anchor still works" and anything below
    means the envelope has to shrink. Returns dict with the bound and the
    endpoint diagnostics; ``max_anchor_scale`` is NaN if even ``lo`` fails."""
    base = float(hydraulics["anchor_scale"])
    hi = base if hi is None else float(hi)
    scale_at = lambda a: scale_anchor(timeline, a / base)
    ev = lambda a: v3_passes(wn, scale_at(a), {**hydraulics, "anchor_scale": a},
                             patterns, time, validation, seed)

    hi_pass, hi_min_p, hi_err = ev(hi)
    if hi_pass:
        return dict(max_anchor_scale=hi, capped_at_hi=True, min_pressure=hi_min_p,
                    error=None, today_anchor_scale=base)
    lo_pass, lo_min_p, lo_err = ev(lo)
    if not lo_pass:
        return dict(max_anchor_scale=float("nan"), capped_at_hi=False,
                    min_pressure=lo_min_p, error=lo_err, today_anchor_scale=base)

    best, best_min_p = lo, lo_min_p
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        ok, min_p, _err = ev(mid)
        if ok:
            lo, best, best_min_p = mid, mid, min_p
        else:
            hi = mid
    return dict(max_anchor_scale=best, capped_at_hi=False, min_pressure=best_min_p,
                error=None, today_anchor_scale=base)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep_drift_feasibility(art: dict, dfactors: pd.DataFrame,
                            cases: list[dict], seed: int = 0,
                            bisect: bool = True) -> pd.DataFrame:
    """One row per case. Each case: ``{"tgt_districts": [...], "to_income":
    ..., "to_density": ...}``.

    Per case: patch t0 with the density term (calibration gate), build the
    fully-drifted end state, then bisect ``anchor_scale`` against V3. The
    calibration gate is evaluated but NOT enforced -- a case outside
    [0.5, 2.0] is recorded and still simulated, so you can see both gates
    independently rather than having the first mask the second."""
    t0, calib = apply_density_to_t0(art["portfolios_t0"], dfactors)
    calib_ok = bool(0.5 <= calib <= 2.0)

    rows = []
    for case in cases:
        tl = fully_drifted_timeline(
            t0, art["districts"], case["tgt_districts"], case["to_income"],
            case["to_density"], art["income_factors"], dfactors, art["buildings"],
            seed=seed)
        per_node = tl.drop_duplicates("node")
        load_ratio = (per_node["volume_m3_month"].sum()
                      / per_node["anchor_m3_month"].sum())
        row = dict(tgt_districts="+".join(case["tgt_districts"]),
                   to_income=case["to_income"], to_density=case["to_density"],
                   t0_calibration=calib, t0_calibration_ok=calib_ok,
                   drifted_load_vs_anchor=float(load_ratio))
        if bisect:
            row.update(max_feasible_anchor_scale(
                art["wn"], tl, art["hydraulics"], art["patterns"], art["time"],
                art["validation"], seed=seed))
        else:
            ok, min_p, err = v3_passes(art["wn"], tl, art["hydraulics"],
                                       art["patterns"], art["time"],
                                       art["validation"], seed=seed)
            row.update(v3_passes_today=ok, min_pressure=min_p, error=err)
        rows.append(row)
    return pd.DataFrame(rows)


def single_district_cases(district_names: list[str],
                          combos: tuple = (("high", "low"), ("low", "high"),
                                           ("high", "high"))) -> list[dict]:
    """One case per (district, (to_income, to_density)) -- the usual starting
    grid: income-only drift, density-only drift, and both. Background is
    whatever ``portfolios_t0`` already carries (the world's real map), NOT
    forced to LL; use ``background_cases`` when you need to control it."""
    return [dict(tgt_districts=[d], to_income=inc, to_density=den)
            for d in district_names for inc, den in combos]


# ---------------------------------------------------------------------------
# background-controlled multi-district cases
# ---------------------------------------------------------------------------
_LETTER = {"L": "low", "M": "medium", "H": "high"}


def _stressed_background(pattern: str) -> str:
    """'LH' -> 'LM', 'HL' -> 'ML': bump the SAME axis the pattern stresses
    from 'L' to 'M', giving a harder-but-not-extreme background."""
    chars = list(pattern)
    for i, c in enumerate(chars):
        if c != "L":
            chars[i] = "M"
            break
    return "".join(chars)


_DENSITY_ORDER = ["low", "medium", "high"]


def is_single_step(from_density: str, to_density: str) -> bool:
    """True if the density transition moves at most one tier (low->medium,
    medium->high, or no change). The L->H jump is what makes density-driven
    drift so violent: it is the full 4.9x swing rather than ~2.3x."""
    return abs(_DENSITY_ORDER.index(to_density)
               - _DENSITY_ORDER.index(from_density)) <= 1


def background_cases(district_names: list[str], patterns: tuple = ("LH", "HL"),
                     k_values: tuple = (3, 4, 5),
                     background_codes: list[str] | None = None,
                     single_step_density: bool = False) -> list[dict]:
    """Cases where >= k districts drift to ``pattern`` while every other
    district SITS at ``background_code`` from t0.

    Unlike ``single_district_cases``, each case carries its own
    ``background_map``, so the t0 portfolio must be rebuilt per case (see
    ``sweep_background_feasibility``) rather than patched from the world's
    existing one.

    ``background_codes`` defaults per pattern to ['LL', <pattern's own axis
    bumped to M>] -- 'LH' -> ['LL','LM'], 'HL' -> ['LL','ML'] -- matching the
    static stress sweep's convention. Deduped on (drifted set, target,
    background), since at k == len(district_names) every district drifts and
    the background stops mattering.

    ``single_step_density=True`` drops any case whose density transition skips
    a tier (background low -> target high), a scenario-design constraint
    rather than a physics change. Empirically this is the difference between
    the LH/LM block (survives) and the LH/LL block (does not).
    """
    n = len(district_names)
    seen, cases = set(), []
    for pattern in patterns:
        to_income, to_density = _LETTER[pattern[0]], _LETTER[pattern[1]]
        bgs = background_codes or ["LL", _stressed_background(pattern)]
        for bg in bgs:
            bg_income, bg_density = _LETTER[bg[0]], _LETTER[bg[1]]
            if single_step_density and not is_single_step(bg_density, to_density):
                continue
            for k in k_values:
                for combo in combinations(range(n), k):
                    tgt = [district_names[i] for i in combo]
                    # every district STARTS at the background; the chosen ones drift
                    bg_map = [[bg_income, bg_density] for _ in range(n)]
                    key = (tuple(sorted(tgt)), pattern, bg if k < n else "-")
                    if key in seen:
                        continue
                    seen.add(key)
                    cases.append(dict(tgt_districts=tgt, to_income=to_income,
                                      to_density=to_density, pattern=pattern,
                                      background=bg, background_map=bg_map,
                                      k=k))
    return cases


def sweep_background_feasibility(art: dict, dfactors: pd.DataFrame,
                                 cases: list[dict], seed: int = 0,
                                 bisect: bool = True) -> pd.DataFrame:
    """Like ``sweep_drift_feasibility``, but each case rebuilds t0 from its own
    ``background_map`` via the real ``build_portfolios`` before applying the
    density term and drifting. That rebuild is why the background is
    controllable here and not in the other sweep.

    ``dfactors`` is deliberately a fixed input, NOT recomputed per case: the
    density factor table is a model parameter, so it must stay constant across
    the sweep or the cases aren't comparable. Normalize it once against the
    world's real map before calling.

    Both gates are recorded, neither enforced: a background map that already
    fails ``build_portfolios``'s own [0.5, 2.0] check is caught and reported
    with the hydraulic columns left NaN."""
    from fedwater.pipelines.urban_scenario.nodes import build_portfolios

    rows = []
    for case in cases:
        row = dict(tgt_districts="+".join(case["tgt_districts"]),
                   k=case.get("k", len(case["tgt_districts"])),
                   pattern=case.get("pattern"), background=case.get("background"),
                   to_income=case["to_income"], to_density=case["to_density"])
        try:
            raw = build_portfolios(art["wn"], art["districts"], art["income_factors"],
                                   {"income_density_mapping": case["background_map"]},
                                   art["buildings"], art["hydraulics"], seed)
        except ValueError as e:  # background alone is already implausible
            row.update(t0_calibration=float("nan"), t0_calibration_ok=False,
                       drifted_load_vs_anchor=float("nan"),
                       max_anchor_scale=float("nan"), error=str(e))
            rows.append(row)
            continue

        t0, calib = apply_density_to_t0(raw, dfactors)
        row.update(t0_calibration=calib, t0_calibration_ok=bool(0.5 <= calib <= 2.0))

        tl = fully_drifted_timeline(
            t0, art["districts"], case["tgt_districts"], case["to_income"],
            case["to_density"], art["income_factors"], dfactors, art["buildings"],
            seed=seed)
        per_node = tl.drop_duplicates("node")
        row["drifted_load_vs_anchor"] = float(
            per_node["volume_m3_month"].sum() / per_node["anchor_m3_month"].sum())

        if bisect:
            row.update(max_feasible_anchor_scale(
                art["wn"], tl, art["hydraulics"], art["patterns"], art["time"],
                art["validation"], seed=seed))
        else:
            ok, min_p, err = v3_passes(art["wn"], tl, art["hydraulics"],
                                       art["patterns"], art["time"],
                                       art["validation"], seed=seed)
            row.update(v3_passes_today=ok, min_pressure=min_p, error=err)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_alpha(art: dict, cases: list[dict], alphas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0),
                normalize: str = "reference_map", reference_map: list | None = None,
                anchor_weights: pd.Series | None = None, seed: int = 0) -> pd.DataFrame:
    """Pass rate vs ``alpha``: rebuilds the factor table at each compression
    level and re-runs the whole case list with ``bisect=False`` (one EPANET
    run per case, pass/fail at today's anchor_scale). ``alpha=0.0`` is the
    no-density-term control, so its pass rate is the bar the others have to
    reach. Returns one row per (alpha, case) -- groupby alpha for the summary.

    Cost: len(alphas) * len(cases) hydraulic runs. Trim ``cases`` first."""
    out = []
    for alpha in alphas:
        dfac = build_density_factors(art["buildings"], normalize=normalize,
                                     reference_map=reference_map,
                                     anchor_weights=anchor_weights, alpha=alpha)
        rep = sweep_background_feasibility(art, dfac, cases, seed=seed, bisect=False)
        rep.insert(0, "alpha", alpha)
        fac = dict(zip(dfac["density"], dfac["density_factor"]))
        rep["df_low"], rep["df_high"] = fac["low"], fac["high"]
        rep["density_swing"] = fac["high"] / fac["low"]
        out.append(rep)
    return pd.concat(out, ignore_index=True)
