"""Hydraulic capacity in mean-demand terms, dispatched on what the network has.

Why this module exists
----------------------
Capacity is the anchor for the whole scenario: it sets ``P_max``, which sets the
starting population, which sets every demand in the horizon. Measure it wrong
and every pressure conclusion downstream is measuring the measurement.

There are two honest ways to measure it and they are not interchangeable:

* **Steady state, constant demand.** Valid only for a network with no storage,
  where the worst instant is fully determined by the instantaneous demand. The
  answer is a *constant*-demand capacity, so the sustainable **mean** demand is
  that number divided by ``k1 * k2`` -- a real system must survive the peak hour
  of the peak day.
* **Dynamic, with the network's own patterns.** Required as soon as tanks exist,
  because storage buffers the peak: the binding constraint stops being "is the
  instantaneous pressure adequate" and becomes "do the tanks refill". The answer
  is already a **mean**-demand capacity with the peak inside it, so dividing by
  ``k1 * k2`` again double-counts the peak.

Getting that dispatch wrong is not a rounding error. Measured on D-Town
(7 tanks, 11 pumps, 24 level controls): the steady-state route returns roughly
0.55 x base demand and then divides by 1.8, while the network in fact survives
to lambda ~= 1.1 dynamically. That is a 3.6x understatement, which starts the
scenario at about a quarter of the design load, leaves the pumps barely cycling,
and quietly removes the control dynamics that any lagged-dependence analysis
depends on.

The reported basis
------------------
Capacity is always returned as a **pattern-weighted mean demand** in L/s, i.e.
``sum_j base_j * mean(pattern_j) * lambda``, never as ``base * lambda``. This
matters more than it looks: D-Town's five DMA patterns all have mean 0.625, so
``base`` overstates its true mean demand by 1/0.625 = 1.6x. Anchoring on the
wrong basis inflates the whole horizon by that factor.
"""
from __future__ import annotations

import copy
import os
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import wntr

LPS_PER_M3S = 1000.0
SEC_PER_DAY = 86400.0


# ==========================================================================
# helpers shared with the demand model
# ==========================================================================

def consumer_nodes(wn) -> list:
    """Junctions with non-zero base demand.

    Zero-demand junctions are modelling scaffolding -- pump suction dummies,
    valve interstitials, topology stubs. They are not consumers, and including
    them lets a sub-atmospheric pressure on a scaffold node be mistaken for
    unserved demand.
    """
    out = []
    for j in wn.junction_name_list:
        b = sum(float(ts.base_value or 0.0)
                for ts in wn.get_node(j).demand_timeseries_list)
        if b > 0:
            out.append(j)
    return out


def base_demand_of(wn, node) -> float:
    """Total base demand at a node, in m3/s, ignoring the global multiplier."""
    return sum(float(ts.base_value or 0.0)
               for ts in wn.get_node(node).demand_timeseries_list)


def pattern_mean_of(wn, node) -> float:
    """Base-weighted mean of the demand patterns attached to a node.

    A node with no pattern has mean 1 by definition. A node with several demand
    categories gets the base-weighted average, because that is what its mean
    demand actually is.
    """
    num = den = 0.0
    for ts in wn.get_node(node).demand_timeseries_list:
        b = float(ts.base_value or 0.0)
        if b == 0:
            continue
        if ts.pattern_name:
            m = float(np.mean(wn.get_pattern(ts.pattern_name).multipliers))
        else:
            m = 1.0
        num += b * m
        den += b
    return num / den if den else 1.0


def mean_demand_lps(wn, nodes=None) -> float:
    """Pattern-weighted mean demand in L/s -- the basis capacity is reported on."""
    nodes = nodes if nodes is not None else consumer_nodes(wn)
    return sum(base_demand_of(wn, j) * pattern_mean_of(wn, j)
               for j in nodes) * LPS_PER_M3S


def solve(wn):
    """Run EPANET in a scratch directory so no artefacts land in the cwd."""
    with tempfile.TemporaryDirectory() as scratch:
        cwd = os.getcwd()
        try:
            os.chdir(scratch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return wntr.sim.EpanetSimulator(wn).run_sim()
        finally:
            os.chdir(cwd)


# ==========================================================================
# regime detection
# ==========================================================================

def network_regime(wn) -> str:
    """Which capacity question this network poses.

    ``storage``    -- has tanks. Capacity is a storage-and-pumping question, so
                      it must be measured dynamically.
    ``controlled`` -- no tanks but has controls or pumps. Still stateful through
                      control switching, so still dynamic, but with no tank
                      criterion to apply.
    ``gravity``    -- no tanks, no controls, no pumps. The classic steady-state
                      case, and the only one where dividing by k1*k2 is correct.
    """
    if len(wn.tank_name_list) > 0:
        return "storage"
    if len(wn.control_name_list) > 0 or len(wn.pump_name_list) > 0:
        return "controlled"
    return "gravity"


@dataclass
class CapacityConfig:
    """How hard to push, and what counts as coping."""
    pressure_floor_m: float = 15.0
    lam_lo: float = 0.05
    lam_hi: float = 4.0
    tol: float = 0.02                 # relative bracket width
    max_iter: int = 20

    # dynamic probe
    probe_days: int = 14              # evaluated window
    warmup_days: int = 10             # discarded; removes the initial-condition
    timestep_s: int = 3600
    hydraulic_timestep_s: int = 900   # finer than reporting: control events

    # tank criteria (storage regime only)
    max_frac_at_min: float = 0.02     # a tank may skim its floor, not sit on it
    max_level_trend: float = 0.30     # half-window mean drop, as a fraction of depth

    # steady criteria (gravity regime only)
    steady_tank_level: str = "min"

    sanity_head_m: float = 1000.0
    verbose: bool = False

    def validate(self):
        if not 0 < self.lam_lo < self.lam_hi:
            raise ValueError("need 0 < lam_lo < lam_hi")
        if self.probe_days < 7:
            raise ValueError(
                "probe_days must be >= 7: a shorter probe cannot see a weekly "
                "cycle, and tank refill is a multi-day question")
        return self


# ==========================================================================
# feasibility
# ==========================================================================

def _prepare(wn, cfg: CapacityConfig, regime: str):
    """A copy of the network with the file multiplier folded in and the clock set."""
    w = copy.deepcopy(wn)
    m0 = float(w.options.hydraulic.demand_multiplier or 1.0)
    if abs(m0 - 1.0) > 1e-12:
        for n in w.junction_name_list:
            for ts in w.get_node(n).demand_timeseries_list:
                if ts.base_value:
                    ts.base_value = float(ts.base_value) * m0
    w.options.hydraulic.demand_multiplier = 1.0
    # capacity is a design question, not a service-degradation question: DD so
    # that an inadequate network shows up as low pressure rather than silently
    # shedding load, which is exactly what PDD would do
    w.options.hydraulic.demand_model = "DD"

    if regime == "gravity":
        for n in w.junction_name_list:
            for ts in w.get_node(n).demand_timeseries_list:
                ts.pattern_name = None            # constant demand
        w.options.time.duration = 0               # steady state
        if cfg.steady_tank_level == "min":
            for t in w.tank_name_list:
                tk = w.get_node(t)
                tk.init_level = float(tk.min_level)
    else:
        total_days = cfg.warmup_days + cfg.probe_days
        w.options.time.duration = int(total_days * 24 * 3600)
        w.options.time.hydraulic_timestep = int(cfg.hydraulic_timestep_s)
        w.options.time.report_timestep = int(cfg.timestep_s)
        w.options.time.pattern_timestep = int(cfg.timestep_s)
    return w


def _feasible(w, cfg: CapacityConfig, regime: str, cons: set, lam: float):
    """Does the network cope at demand multiplier ``lam``?"""
    w.options.hydraulic.demand_multiplier = float(lam)
    diag = {"lambda": lam}
    try:
        res = solve(w)
    except Exception as exc:
        diag["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return False, diag

    cols = [c for c in res.node["pressure"].columns if c in cons]
    if not cols:
        diag["error"] = "no consumer columns in result"
        return False, diag

    if regime == "gravity":
        keep = slice(None)
    else:
        t0 = cfg.warmup_days * 24 * 3600
        keep = res.node["pressure"].index >= t0

    p = res.node["pressure"].loc[keep, cols].to_numpy()
    diag["pressure_min_m"] = float(np.nanmin(p))
    diag["frac_below_floor"] = float((p < cfg.pressure_floor_m).mean())

    if not p.size or np.nanmax(np.abs(p)) > cfg.sanity_head_m:
        diag["error"] = "non-physical head"
        return False, diag
    ok = bool(np.nanmin(p) >= cfg.pressure_floor_m)

    tanks = list(w.tank_name_list)
    if tanks:
        lv = pd.DataFrame({t: res.node["head"][t] - w.get_node(t).elevation
                           for t in tanks}).loc[keep]
        # Endpoint-to-endpoint change is NOT usable here: once the tanks lock
        # into their control-driven cycle it reads exactly zero regardless of
        # load, and at very low load it reads large simply because the cycle has
        # not settled. Measured on D-Town: 0.000 at lambda 1.0 and 1.1, but
        # 0.26 at lambda 0.05. Use the fraction of time on the floor, which is
        # monotone in demand (0.000 at lambda <= 1.0, 0.095 at 1.1), and a
        # half-window mean comparison, which is insensitive to cycle phase.
        frac_min, trend = [], []
        half = len(lv) // 2
        for t in tanks:
            tk = w.get_node(t)
            lo, hi = float(tk.min_level), float(tk.max_level)
            depth = max(hi - lo, 1e-9)
            frac_min.append(float((lv[t] <= lo + 0.02).mean()))
            trend.append(float((lv[t].iloc[:half].mean()
                                - lv[t].iloc[half:].mean()) / depth))
        diag["tank_frac_at_min"] = float(np.max(frac_min))
        diag["tank_level_trend"] = float(np.max(trend))
        ok = ok and diag["tank_frac_at_min"] <= cfg.max_frac_at_min \
            and diag["tank_level_trend"] <= cfg.max_level_trend

    if w.pump_name_list and "status" in res.link:
        st = res.link["status"].loc[keep, list(w.pump_name_list)].to_numpy()
        diag["pump_duty"] = float(st.mean())
    return ok, diag


# ==========================================================================
# the measurement
# ==========================================================================

def measure_capacity(wn, cfg: CapacityConfig | None = None) -> dict:
    """Largest mean demand the network sustains, on the right basis for its regime.

    Returns ``apply_peak_division``, which callers must respect: it is True only
    for the steady-state gravity route, where the measured number is a constant
    demand and the sustainable mean is that divided by ``k1 * k2``. Dividing a
    dynamically measured capacity as well is the double-count this module exists
    to prevent.
    """
    cfg = (cfg or CapacityConfig()).validate()
    regime = network_regime(wn)
    cons = set(consumer_nodes(wn))
    if not cons:
        raise ValueError("network has no consumer junctions")

    w = _prepare(wn, cfg, regime)
    mean_lps = mean_demand_lps(w, cons)      # at lambda = 1, patterns included
    base_lps = sum(base_demand_of(w, j) for j in cons) * LPS_PER_M3S

    trace = []
    ok_lo, d_lo = _feasible(w, cfg, regime, cons, cfg.lam_lo)
    trace.append(d_lo)
    if not ok_lo:
        return {"capacity_lps": float("nan"), "lambda_star": float("nan"),
                "regime": regime, "basis": "mean_demand",
                "apply_peak_division": regime == "gravity",
                "mean_demand_lps_at_1": mean_lps, "base_demand_lps": base_lps,
                "status": "infeasible at lam_lo=%g -- the network has defects "
                          "independent of demand" % cfg.lam_lo,
                "trace": pd.DataFrame(trace)}

    ok_hi, d_hi = _feasible(w, cfg, regime, cons, cfg.lam_hi)
    trace.append(d_hi)
    if ok_hi:
        return {"capacity_lps": mean_lps * cfg.lam_hi, "lambda_star": cfg.lam_hi,
                "regime": regime, "basis": "mean_demand",
                "apply_peak_division": regime == "gravity",
                "mean_demand_lps_at_1": mean_lps, "base_demand_lps": base_lps,
                "status": "still feasible at lam_hi=%g; raise the bracket"
                          % cfg.lam_hi,
                "trace": pd.DataFrame(trace)}

    a, b, n = cfg.lam_lo, cfg.lam_hi, 2
    while (b - a) / max(a, 1e-9) > cfg.tol and n < cfg.max_iter:
        mid = 0.5 * (a + b)
        ok, d = _feasible(w, cfg, regime, cons, mid)
        trace.append(d)
        a, b = (mid, b) if ok else (a, mid)
        n += 1
        if cfg.verbose:
            print("    lambda=%.4f %s (pmin=%.1f)"
                  % (mid, "ok" if ok else "fail", d.get("pressure_min_m", np.nan)))

    return {"capacity_lps": mean_lps * a, "lambda_star": a,
            "regime": regime, "basis": "mean_demand",
            "apply_peak_division": regime == "gravity",
            "mean_demand_lps_at_1": mean_lps, "base_demand_lps": base_lps,
            "pattern_mean_effective": mean_lps / base_lps if base_lps else np.nan,
            "n_solves": n, "status": "bracketed",
            "trace": pd.DataFrame(trace)}


@dataclass
class DemandModelConfig:
    """Per-capita consumption, income effect, peaking, and non-residential PE."""
    litres_per_capita_day: float = 160.0
    income_factors: dict = field(default_factory=lambda: {
        "low": 0.80, "medium": 1.00, "high": 1.35})
    k1_daily_peak: float = 1.20
    k2_hourly_peak: float = 1.50
    pe_per_unit: dict = field(default_factory=lambda: {
        "residential": 1.0, "commercial": 1.25, "industrial": 2.0})

    def per_capita_m3s(self, income_band: str) -> float:
        f = self.income_factors.get(income_band)
        if f is None:
            raise ValueError(f"unknown income band {income_band!r}")
        return self.litres_per_capita_day * f / 1000.0 / SEC_PER_DAY


def capacity_to_population(capacity_lps: float, dm: DemandModelConfig,
                           income_band: str = "medium",
                           category_mix: dict | None = None,
                           apply_peak_division: bool = True) -> dict:
    """Convert a capacity in L/s into a population ceiling.

    ``apply_peak_division`` must come from ``measure_capacity``, not from a
    default. True divides by ``k1 * k2`` because the measurement was a constant
    demand; False leaves it alone because the measurement already contained the
    peak. Passing True for a dynamically measured capacity understates the
    ceiling by ~80% at the default factors.
    """
    mix = category_mix or {"residential": 1.0}
    tot = sum(mix.values())
    if tot <= 0:
        raise ValueError("category_mix must have positive total weight")
    pe_weighted = sum(w / tot * dm.pe_per_unit[c] for c, w in mix.items())
    q = dm.per_capita_m3s(income_band)
    peak = dm.k1_daily_peak * dm.k2_hourly_peak if apply_peak_division else 1.0
    mean_allowed = capacity_lps / LPS_PER_M3S / peak
    pop = mean_allowed / (q * pe_weighted) if q > 0 else np.nan
    return {"capacity_lps": capacity_lps,
            "peak_divisor_applied": peak,
            "sustainable_mean_lps": mean_allowed * LPS_PER_M3S,
            "population_equivalent_capacity": pop,
            "pe_weighted_intensity": pe_weighted,
            "per_capita_lps": q * LPS_PER_M3S}
