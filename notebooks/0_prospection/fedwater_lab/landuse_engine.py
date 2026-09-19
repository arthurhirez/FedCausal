"""landuse_engine.py — land-use demand model, proof-of-concept engine.

Replaces `density` with a three-sector land-use model. A block is a mix of
residential / commercial / industrial plots; the mix drives both the demand
*level* (via a plot-intensity table, dialled by `beta`) and the demand *shape*
(via per-sector daily and weekly signatures).

Nothing in `src/` is touched. This module overrides `build_portfolios`,
`evolve_assignments` and `synthesize_demands`; everything else in the pipeline
(`configure_network`, `apply_coupling`, `build_drift_schedule`,
`apply_drift_ramp`, `run_hydraulics`) is the shipped implementation.

Import AFTER `bootstrap_project(ROOT)` so `fedwater` is on the path.
"""
from __future__ import annotations

import copy
import itertools
import logging
import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import erf

from fedwater.pipelines.network_prep.nodes import configure_network, apply_coupling
from fedwater.pipelines.urban_scenario.nodes import (
    build_income_factors, build_drift_schedule, SECONDS_PER_DAY, ANCHOR_DAYS_PER_MONTH)
from fedwater.pipelines.demand_synthesis.nodes import apply_drift_ramp
from fedwater.pipelines.hydraulics.nodes import run_hydraulics

logging.getLogger("wntr").setLevel(logging.ERROR)   # DD reports overload as a warning per solve

BAND = (10.0, 50.0)          # ABNT NBR 12218
HARD_FLOOR = 5.0
CALIB_BAND = (0.5, 2.0)      # diagnostic only — calibration is hydraulically a no-op


# ============================================================== sector model
SECTORS = {
    "residential": dict(
        kind="peaks", night=0.15, dispersion_h=0.9, weekend_factor=1.0,
        day_sigma=0.12, seasonal_amplitude=0.14,
        peaks={"morning": (7.0, 3.5), "afternoon": (14.0, 1.0), "evening": (19.0, 3.5)}),
    "commercial": dict(
        kind="plateau", window=(8.0, 18.0), night=0.03, amp=1.00, dispersion_h=0.8,
        weekend_factor=0.15, day_sigma=0.05, seasonal_amplitude=0.06),
    "industrial": dict(
        kind="plateau", window=(6.0, 22.0), night=0.85, amp=0.22, dispersion_h=0.5,
        weekend_factor=0.90, day_sigma=0.02, seasonal_amplitude=0.01),
}

INTENSITY_REL = {"commercial": 2.5, "industrial": 8}   # x one medium-income residential unit

LAND_USE_MIX = {                                          # plot shares
    "residential": {"residential": 1.00},
    "mixed":       {"residential": 0.70, "commercial": 0.30},
    "commercial":  {"residential": 0.35, "commercial": 0.65},
    "industrial":  {"residential": 0.20, "commercial": 0.30, "industrial": 0.50},
}

_INC_C = {"L": "low", "M": "medium", "H": "high"}
_LU_C = {"R": "residential", "M": "mixed", "C": "commercial", "I": "industrial"}


def decode_map(code: str, n_districts: int = 5) -> list[list[str]]:
    """'LR_LM_LC_LR_LR' -> [[income, land_use], ...] in district order."""
    toks = code.strip().upper().split("_")
    assert len(toks) == n_districts, f"{code}: expected {n_districts} tokens"
    return [[_INC_C[t[0]], _LU_C[t[1]]] for t in toks]


def random_maps(n, seed=0, incomes="LLLMH"):
    rng = np.random.default_rng(seed)
    return ["_".join(rng.choice(list(incomes)) + rng.choice(list(_LU_C)) for _ in range(5))
            for _ in range(n)]


# ============================================================== level factors
def plot_intensity(income_factors: pd.DataFrame, income: str) -> dict:
    """m3/month per plot, by sector. Residential is not free — it comes from
    `standards_by_income`; only the two non-residential numbers are new."""
    m = income_factors.set_index("income")["mean_unit_m3_month"]
    return {"residential": float(m[income]),
            **{s: r * float(m["medium"]) for s, r in INTENSITY_REL.items()}}


def build_landuse_factors(income_factors, beta=1.0,
                          reference=("low", "residential")) -> pd.DataFrame:
    """Level factor per (income, land_use). Plot count is conserved across a land-use
    change, so the factor is the ratio of mean plot intensities; `beta` dials it
    (0 = shape-only / volume-neutral, 1 = full)."""
    def mean_plot(inc, lu):
        I = plot_intensity(income_factors, inc)
        return sum(w * I[s] for s, w in LAND_USE_MIX[lu].items())

    ref = mean_plot(*reference)
    return pd.DataFrame([{"income": i, "land_use": lu, "mean_plot_m3_month": mean_plot(i, lu),
                          "level_factor": (mean_plot(i, lu) / ref) ** beta}
                         for i in income_factors["income"] for lu in LAND_USE_MIX])


# ==================================================================== shapes
def _gauss(t, mu, w):
    return np.exp(-0.5 * ((t - mu) / w) ** 2)


def _boxcar(t, a, b, w):
    return 0.5 * (erf((t - a) / (np.sqrt(2) * w)) - erf((t - b) / (np.sqrt(2) * w)))


def sector_day_shape(t, sector, plots, rng, weekend, patterns):
    """One sector cohort's daily shape, arbitrary scale, > 0.

    A plateau is a boxcar convolved with the same Gaussian start-time jitter the
    residential peaks use, so crowd smoothing stays derived (1/sqrt(N)).
    """
    p = SECTORS[sector]
    w_eff = np.sqrt(patterns["peak_width_h"] ** 2 + p["dispersion_h"] ** 2)
    resid = 1.0 / np.sqrt(max(plots, 1.0))
    jit = rng.normal(0.0, p["dispersion_h"] * resid)
    amp_n = lambda a: max(a * (1.0 + rng.normal(0.0, patterns["amp_sigma"] * resid)), 0.0)

    if p["kind"] == "peaks":
        shape = np.full_like(t, p["night"], dtype=float)
        shift = patterns["weekend_morning_shift_h"] if weekend else 0.0
        for name, (mu, amp) in p["peaks"].items():
            a = amp_n(amp) * (patterns["weekend_morning_damp"]
                              if (weekend and name == "morning") else 1.0)
            shape += a * _gauss(t, mu + jit + (shift if name == "morning" else 0.0), w_eff)
    else:
        a, b = p["window"]
        shape = p["night"] + amp_n(p["amp"]) * _boxcar(t, a + jit, b + jit, w_eff)

    if weekend:
        shape = shape * p["weekend_factor"]
    return np.clip(shape, 1e-6, None)


# ================================================================ portfolios
def _plots(node, district, income, land_use, anchor, volume, income_factors):
    """Split a node's volume into sector cohorts.

    Zero-volume nodes (zero-base-demand trunk junctions) are dropped entirely — a
    cohort with no volume would make vol_w 0/0 in synthesis and put NaN into the
    EPANET pattern. Cohorts below half a plot are dropped and the remainder
    renormalised, so the node's volume is preserved exactly.
    """
    if not np.isfinite(volume) or volume <= 0:
        return []
    I = plot_intensity(income_factors, income)
    mix = LAND_USE_MIX[land_use]
    total = volume / sum(w * I[s] for s, w in mix.items())
    kept = [(s, w) for s, w in mix.items() if total * w >= 0.5]
    if not kept:
        kept = [max(mix.items(), key=lambda kv: kv[1])]
    norm = sum(w * I[s] for s, w in kept)
    out = []
    for s, w in kept:
        sv = volume * w * I[s] / norm
        out.append({"node": node, "district": district, "income": income,
                    "land_use": land_use, "sector": s, "plots": sv / I[s],
                    "sector_volume_m3_month": sv,
                    "anchor_m3_month": anchor, "volume_m3_month": volume})
    return out


def build_portfolios(wn, districts, landuse_factors, mapping, income_factors, anchor_scale):
    lvl = landuse_factors.set_index(["income", "land_use"])["level_factor"].to_dict()
    rows = []
    for district, nodes in districts["districts"].items():
        income, land_use = mapping[district]
        for node in nodes:
            base = wn.get_node(node).demand_timeseries_list[0].base_value
            anchor = base * anchor_scale * SECONDS_PER_DAY * ANCHOR_DAYS_PER_MONTH
            rows += _plots(node, district, income, land_use, anchor,
                           anchor * lvl[(income, land_use)], income_factors)
    df = pd.DataFrame(rows)
    per_node = df.drop_duplicates("node")
    calib = per_node["anchor_m3_month"].sum() / per_node["volume_m3_month"].sum()
    for c in ("volume_m3_month", "sector_volume_m3_month", "plots"):
        df[c] *= calib
    df["calibration"] = calib
    return df


def evolve_assignments(portfolios_t0, drift_schedule, landuse_factors, income_factors,
                       n_months):
    """Per-row to_income / to_land_use, so districts can drift differently."""
    lvl = landuse_factors.set_index(["income", "land_use"])["level_factor"].to_dict()
    ds = drift_schedule.set_index("node") if len(drift_schedule) else None
    calib = portfolios_t0["calibration"].iloc[0]
    frames = []
    for month in range(n_months):
        snap = portfolios_t0.copy()
        switch = list(ds.index[ds["drift_month"] <= month]) if ds is not None else []
        if switch:
            base, new = snap[snap["node"].isin(switch)].drop_duplicates("node"), []
            for _, r in base.iterrows():
                to_i, to_l = ds.loc[r["node"], "to_income"], ds.loc[r["node"], "to_land_use"]
                anchor = r["anchor_m3_month"]
                new += _plots(r["node"], r["district"], to_i, to_l, anchor,
                              anchor * calib * lvl[(to_i, to_l)], income_factors)
            snap = pd.concat([snap[~snap["node"].isin(switch)],
                              pd.DataFrame(new).assign(calibration=calib)], ignore_index=True)
        frames.append(snap.assign(month=month))
    return pd.concat(frames, ignore_index=True)


# ================================================================= synthesis
def synthesize_demands(assignments_timeline, patterns, time, seed):
    """Cohorts are mixed by VOLUME share (sectors differ ~10x in intensity) and each
    cohort is normalised over the WHOLE MONTH (so commercial's weekend collapse
    survives the mixing). Seasonality and the common-mode weather factor are
    volume-weighted across sectors."""
    res_h, days, n_months = time["resolution_h"], time["days_per_month"], time["n_months"]
    steps = int(round(24 / res_h))
    t = (np.arange(steps) + 0.5) * res_h
    nodes = sorted(assignments_timeline["node"].unique(), key=int)
    idx = {n: j for j, n in enumerate(nodes)}
    out = np.zeros((n_months * days * steps, len(nodes)))
    seas_scale = patterns.get("seasonality_scale", 1.0)

    for (month, node), coh in assignments_timeline.groupby(["month", "node"]):
        rng, month = np.random.default_rng([seed, int(month), int(node)]), int(month)
        sv = coh.set_index("sector")["sector_volume_m3_month"]
        if not np.isfinite(sv.sum()) or sv.sum() <= 0:
            continue
        vol_w = sv / sv.sum()
        plots = coh.set_index("sector")["plots"]
        wknd = np.array([((month * days + d) % 7) >= 5 for d in range(days)])

        series = np.zeros(days * steps)
        for sector, w in vol_w.items():
            s = np.concatenate([sector_day_shape(t, sector, float(plots[sector]), rng,
                                                 bool(wk), patterns) for wk in wknd])
            series += w * (s / s.mean())

        seas_a = seas_scale * sum(w * SECTORS[s]["seasonal_amplitude"] for s, w in vol_w.items())
        day_s = sum(w * SECTORS[s]["day_sigma"] for s, w in vol_w.items())
        series *= np.repeat(1.0 + rng.normal(0.0, day_s, days), steps)

        v_month = (sv.sum()
                   * (1.0 + seas_a * np.cos(2 * np.pi * (month % 12
                      - patterns["seasonal_peak_month"]) / 12.0))
                   * rng.lognormal(0.0, patterns["month_sigma"]))
        liters = v_month * 1000.0 * days / ANCHOR_DAYS_PER_MONTH
        series = np.clip(series, 1e-9, None)
        i = month * days * steps
        out[i:i + days * steps, idx[node]] = series / (series.sum() * res_h * 3600.0) * liters

    df = pd.DataFrame(out, columns=nodes)
    df["month"] = np.repeat(np.arange(n_months), days * steps)
    return df


# =================================================================== context
@dataclass
class Ctx:
    """Everything loaded once from the Kedro catalog."""
    wn_raw: object
    districts: dict
    params: dict
    income_factors: pd.DataFrame = field(init=False)
    node2dist: dict = field(init=False)

    def __post_init__(self):
        self.income_factors = build_income_factors(self.params["buildings"])
        self.node2dist = {n: d for d, ns in self.districts["districts"].items() for n in ns}

    def names(self):
        return list(self.districts["districts"])


# ==================================================================== runner
TIMELINE = dict(mode="timeline", n_months=10, days_per_month=14, resolution_h=1,
                anchor_scale=0.05, beta=0.35, drift_ramp_days=42, warmup_months=2,
                seasonality_scale=0.0, month_sigma=None, sim_seed=42)

# Envelope mode scores the fully-drifted state at the seasonal peak. The schedule is
# deterministic (every target node switches at warmup), so 3 months is exact — the
# diffusion front would otherwise need more months than its subgraph radius.
ENVELOPE = dict(mode="envelope", n_months=3, days_per_month=7, resolution_h=1,
                anchor_scale=0.05, beta=0.5, drift_ramp_days=5, warmup_months=1,
                seasonality_scale=1.0, month_sigma=None, sim_seed=42)


def _schedule(ctx, wn, drifts, scen, cfg):
    """Deterministic in envelope mode; the shipped diffusion in timeline mode.

    warmup_months must be >= 1: apply_drift_ramp blends against the month before the
    switch, so a node drifting at month 0 has nothing to blend from.
    """
    cols = ["node", "drift_month", "to_income", "to_land_use"]
    if not drifts:
        return pd.DataFrame(columns=cols)
    if cfg["mode"] == "envelope":
        return pd.DataFrame([{"node": n, "drift_month": cfg["warmup_months"],
                              "to_income": d["to_income"], "to_land_use": d["to_land_use"]}
                             for d in drifts
                             for n in ctx.districts["districts"][d["tgt_district"]]],
                            columns=cols)
    sched = pd.concat([
        build_drift_schedule(
            wn, ctx.districts,
            {**scen, "drift": {**scen["drift"], "tgt_district": d["tgt_district"],
                               "to_income": d["to_income"], "to_density": d["to_land_use"],
                               "warmup_months": cfg["warmup_months"],
                               "seed_node": d.get("seed_node") or seed_node(ctx, wn,
                                                                           d["tgt_district"])}},
            seed=cfg["sim_seed"] + i)
        for i, d in enumerate(drifts)], ignore_index=True)
    return sched.rename(columns={"to_density": "to_land_use"})


def seed_node(ctx, wn, district):
    """Largest-base-demand junction, skipping zero-demand trunk nodes."""
    base = {n: wn.get_node(n).demand_timeseries_list[0].base_value
            for n in ctx.districts["districts"][district]}
    nz = {n: v for n, v in base.items() if v > 0}
    return max(nz or base, key=(nz or base).get)


def run_world(ctx, map_code, drifts=(), preset=None, **over):
    cfg = {**(preset or TIMELINE), **over}
    p = copy.deepcopy(ctx.params)
    time = {k: cfg[k] for k in ("n_months", "days_per_month", "resolution_h")}
    assert time["days_per_month"] % 7 == 0, "keep days_per_month a multiple of 7"
    hyd = {**p["hydraulics"], "anchor_scale": cfg["anchor_scale"]}
    scen = {**p["scenario"], "n_months": cfg["n_months"]}
    mapping = dict(zip(ctx.names(), decode_map(map_code, len(ctx.names()))))

    wn = configure_network(copy.deepcopy(ctx.wn_raw), hyd, time)
    wn, _ = apply_coupling(wn, ctx.districts, p["coupling"], seed=cfg["sim_seed"])

    lf = build_landuse_factors(ctx.income_factors, cfg["beta"])
    pf = build_portfolios(wn, ctx.districts, lf, mapping, ctx.income_factors,
                          cfg["anchor_scale"])
    sched = _schedule(ctx, wn, list(drifts), scen, cfg)

    # the ramp cannot run past the end of the horizon
    last_switch = int(sched["drift_month"].max()) if len(sched) else 0
    avail = (cfg["n_months"] - last_switch) * cfg["days_per_month"]
    pat = {**p["patterns"],
           "drift_ramp_days": min(cfg["drift_ramp_days"], max(avail - 1, 1)),
           "seasonality_scale": cfg["seasonality_scale"],
           "seasonal_peak_month": (cfg["n_months"] - 1 if cfg["mode"] == "envelope"
                                   else p["patterns"]["seasonal_peak_month"])}
    if cfg["month_sigma"] is not None:
        pat["month_sigma"] = cfg["month_sigma"]

    tl = evolve_assignments(pf, sched, lf, ctx.income_factors, cfg["n_months"])
    dem = synthesize_demands(tl, pat, time, cfg["sim_seed"])
    if len(sched):
        dem = apply_drift_ramp(dem, sched, pat, time)
    pres, flows, dsim = run_hydraulics(wn, dem)

    tgt = {n for d in drifts for n in ctx.districts["districts"][d["tgt_district"]]}
    return dict(ctx=ctx, map=map_code, drifts=list(drifts), cfg=cfg, time=time,
                calibration=float(pf["calibration"].iloc[0]), landuse_factors=lf,
                portfolios=pf, schedule=sched, timeline=tl, demand=dem,
                pressures=pres, flows=flows, demands_sim=dsim,
                last=cfg["n_months"] - 1,
                drifted_frac=(len(set(sched["node"]) & tgt) / max(len(tgt), 1)
                              if len(sched) else 1.0))


# ==================================================================== views
def steps(r):
    return int(round(24 / r["time"]["resolution_h"]))


def pnodes(r):
    """Every district junction — pressure is checked at trunk nodes too."""
    return [c for c in r["pressures"].columns if c in r["ctx"].node2dist]


def dnodes(r):
    """Junctions that actually carry a demand column (zero-base-demand nodes never
    enter the portfolio, so they are absent from the demand frame)."""
    return [c for c in pnodes(r) if c in r["demand"].columns and c in r["demands_sim"].columns]


def tnodes(r):
    return sorted({n for d in r["drifts"]
                   for n in r["ctx"].districts["districts"][d["tgt_district"]]
                   if n in r["demand"].columns})


def phases(r):
    o = int(r["schedule"]["drift_month"].min()) if len(r["schedule"]) else 1
    e = int(r["schedule"]["drift_month"].max()) if len(r["schedule"]) else 1
    n = r["time"]["n_months"]
    return {"before": max(o - 1, 0), "during": min((o + e) // 2 + 1, n - 1), "after": n - 1}


def week_profile(r, month, nodes=None, scaled=False, ref_months=2):
    """Mean 7-day profile. scaled=True applies per-client MinMax fitted on the
    reference months, exactly as the FL preprocessing does."""
    st, dpm = steps(r), r["time"]["days_per_month"]
    s = r["demand"][nodes if nodes is not None else tnodes(r) or dnodes(r)].sum(axis=1)
    if scaled:
        ref = s[r["demand"]["month"] < ref_months]
        s = (s - ref.min()) / (ref.max() - ref.min())
    day = s.to_numpy().reshape(-1, st)
    mod = r["demand"]["month"].to_numpy()[::st]
    gday = mod * dpm + np.tile(np.arange(dpm), r["time"]["n_months"])
    sel = mod == month
    prof = []
    for w in range(7):
        m = sel & (gday % 7 == w)
        prof.append(day[m].mean(0) if m.any() else day[sel].mean(0))
    return np.concatenate(prof)


def night_day(r, month, nodes=None):
    """(night/day ratio, weekday/weekend ratio) — both scale-invariant."""
    st = steps(r)
    w = week_profile(r, month, nodes).reshape(7, st)
    h = (np.arange(st) + .5) * r["time"]["resolution_h"]
    return (w[:, h < 5].mean() / w[:, (h >= 9) & (h < 17)].mean(),
            w[:5].mean() / max(w[5:].mean(), 1e-9))


def shape_shift(r):
    """1 - corr between MinMax-scaled weekly profiles before and after the drift.
    Scale- and shift-invariant, so it isolates pure temporal signature change."""
    ph = phases(r)
    a, b = week_profile(r, ph["before"], scaled=True), week_profile(r, ph["after"], scaled=True)
    return float(1 - np.corrcoef(a, b)[0, 1])


# =================================================================== checks
def hydraulic_check(r, month=None):
    p, dem = r["pressures"], r["demand"]
    sel = (dem["month"].to_numpy() == month) if month is not None else slice(None)
    v = p.loc[sel, pnodes(r)].to_numpy()
    dj = dnodes(r)
    d_sim = r["demands_sim"][dj].to_numpy()
    supplied = -r["demands_sim"].drop(columns=["month"], errors="ignore").to_numpy() \
        .clip(max=0).sum()
    consumed = d_sim.clip(min=0).sum()
    inside = float(((v >= BAND[0]) & (v <= BAND[1])).mean())

    fails = []
    if not np.isfinite(v).all():
        fails.append("nonfinite")
    if v.min() < HARD_FLOOR:
        fails.append("V3_floor")
    if inside < 0.98:
        fails.append("V4_band")
    if v.max() > BAND[1]:
        fails.append("overpressure")
    if abs(supplied - consumed) / max(consumed, 1e-9) > 1e-4:
        fails.append("V1_mass")
    if (np.abs(d_sim - dem[dj].to_numpy()).max()
            / max(np.abs(dem[dj].to_numpy()).max(), 1e-9)) > 1e-3:
        fails.append("V2_dd")

    return {"pmin": float(v.min()), "pmax": float(v.max()), "inside_%": 100 * inside,
            "peak_lps": float(dem.loc[sel, dj].sum(axis=1).max()),
            "mean_lps": float(dem.loc[sel, dj].sum(axis=1).mean()),
            "calibration": r["calibration"], "drifted_%": 100 * r["drifted_frac"],
            "passed": not fails, "fail": ",".join(fails)}


def sanity_suite(ctx, map_code="LR_LM_LC_LR_LR", verbose=True):
    """Design invariants that must hold before any ablation is worth reading."""
    rows = []

    def chk(name, ok, detail=""):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})

    # --- level model ----------------------------------------------------
    lf1 = build_landuse_factors(ctx.income_factors, 1.0).query("income == 'low'")
    order = lf1.set_index("land_use")["level_factor"]
    chk("S1 level ordering R<M<C<I",
        order["residential"] < order["mixed"] < order["commercial"] < order["industrial"],
        " < ".join(f"{order[k]:.2f}" for k in ("residential", "mixed", "commercial", "industrial")))
    lf0 = build_landuse_factors(ctx.income_factors, 0.0)
    chk("S2 beta=0 is volume-neutral", np.allclose(lf0["level_factor"], 1.0),
        f"max dev {np.abs(lf0['level_factor'] - 1).max():.2e}")

    # --- shape model ----------------------------------------------------
    t = np.arange(24) + .5
    sig = {}
    for s in SECTORS:
        wd = sector_day_shape(t, s, 200, np.random.default_rng(0), False, ctx.params["patterns"])
        we = sector_day_shape(t, s, 200, np.random.default_rng(0), True, ctx.params["patterns"])
        sig[s] = (wd[t < 5].mean() / wd[(t >= 9) & (t < 17)].mean(), we.mean() / wd.mean())
    chk("S3 night/day: commercial < residential < industrial",
        sig["commercial"][0] < sig["residential"][0] < sig["industrial"][0],
        " < ".join(f"{sig[s][0]:.2f}" for s in ("commercial", "residential", "industrial")))
    chk("S4 commercial closes at weekends", sig["commercial"][1] < 0.3,
        f"we/wd {sig['commercial'][1]:.2f}")
    n_hi = sector_day_shape(t, "residential", 4000, np.random.default_rng(1), False,
                            ctx.params["patterns"])
    n_lo = sector_day_shape(t, "residential", 10, np.random.default_rng(1), False,
                            ctx.params["patterns"])
    chk("S5 crowd smoothing: small N is noisier",
        n_lo.std() / n_lo.mean() > n_hi.std() / n_hi.mean(),
        f"cv {n_lo.std()/n_lo.mean():.3f} vs {n_hi.std()/n_hi.mean():.3f}")

    # --- volume bookkeeping (deterministic world) ------------------------
    r = run_world(ctx, map_code, preset=ENVELOPE, month_sigma=0.0, seasonality_scale=0.0)
    pf = r["portfolios"].drop_duplicates("node").set_index("node")
    sec = r["portfolios"].groupby("node")["sector_volume_m3_month"].sum()
    gap = (sec - pf["volume_m3_month"]).abs().max() / pf["volume_m3_month"].max()
    chk("S6 sector volumes sum to node volume", gap < 1e-9, f"max rel gap {gap:.2e}")

    dpm = r["time"]["days_per_month"]
    sim_v = (r["demand"][dnodes(r)].loc[r["demand"]["month"] == 0].sum(axis=0)
             * 3600.0 / 1000.0 * ANCHOR_DAYS_PER_MONTH / dpm)
    tgt_v = (r["timeline"].query("month == 0").groupby("node")["sector_volume_m3_month"].sum()
             .reindex(sim_v.index))
    err = float((sim_v / tgt_v - 1).abs().max())
    chk("S7 simulated volume matches portfolio", err < 1e-6, f"max rel err {err:.2e}")

    per_node = r["portfolios"].drop_duplicates("node")
    chk("S8 calibration renormalises to anchor total",
        abs(per_node["volume_m3_month"].sum() / per_node["anchor_m3_month"].sum() - 1) < 1e-9,
        f"calib {r['calibration']:.3f}")
    missing = set(pnodes(r)) - set(dnodes(r))
    zero = {n for n in missing
            if r["ctx"].wn_raw.get_node(n).demand_timeseries_list[0].base_value == 0}
    chk("S9 demand-less nodes are exactly the zero-demand ones", missing == zero,
        f"{len(missing)} missing, {len(zero)} zero-demand")

    # --- hydraulics and determinism -------------------------------------
    h = hydraulic_check(r)
    chk("S10 V1 mass balance", "V1_mass" not in h["fail"], "")
    chk("S11 V2 DD reproduces input demand", "V2_dd" not in h["fail"], "")
    chk("S12 undrifted baseline is feasible", h["pmin"] >= HARD_FLOOR,
        f"pmin {h['pmin']:.1f} pmax {h['pmax']:.1f} mca")

    r2 = run_world(ctx, map_code, preset=ENVELOPE, month_sigma=0.0, seasonality_scale=0.0)
    chk("S13 same seed reproduces the world",
        np.allclose(r["demand"][dnodes(r)].to_numpy(), r2["demand"][dnodes(r2)].to_numpy()), "")

    # --- shape/level orthogonality --------------------------------------
    dr = [{"tgt_district": ctx.names()[3], "to_income": "low", "to_land_use": "commercial"}]
    a = run_world(ctx, map_code, dr, preset=TIMELINE, beta=0.0)
    b = run_world(ctx, map_code, dr, preset=TIMELINE, beta=0.5)
    ph = phases(a)
    ma = a["demand"].loc[a["demand"]["month"] == ph["before"], tnodes(a)].sum(axis=1).mean()
    mb = a["demand"].loc[a["demand"]["month"] == ph["after"], tnodes(a)].sum(axis=1).mean()
    chk("S14 beta=0 drift does not move the level", abs(mb / ma - 1) < 0.10,
        f"{100*(mb/ma-1):+.1f}%")
    chk("S15 shape shift survives the scaler and is beta-independent",
        shape_shift(a) > 0.3 and abs(shape_shift(a) - shape_shift(b)) < 0.15,
        f"beta0 {shape_shift(a):.3f} vs beta.5 {shape_shift(b):.3f}")

    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
        print(f"\n{int(df['pass'].sum())}/{len(df)} checks passed")
    return df, {"baseline": r, "beta0": a, "beta_half": b}


# ================================================================= ablation
CURATED_MAPS = [
    "LR_LR_LR_LR_LR", "LR_LM_LC_LR_LR", "LR_LR_LR_LM_LR", "LR_LR_LR_LC_LR",
    "LC_LC_LC_LC_LC", "LI_LI_LI_LI_LI", "LM_LM_LM_LM_LM", "HR_HR_HR_HR_HR",
    "LR_MR_HR_LR_MR", "LI_LR_LR_LR_LI",
]


def default_grid(ctx, n_random=6, betas=(0.0, 0.25, 0.5, 0.75, 1.0),
                 anchors=(0.05,), to_incomes=("low",)):
    names = ctx.names()
    return dict(maps=CURATED_MAPS + random_maps(n_random, seed=7),
                targets=[[n] for n in names] + [[names[0], names[3]]] + [names],
                to_lu=["mixed", "commercial", "industrial"],
                to_incomes=list(to_incomes), betas=list(betas), anchors=list(anchors))


def baselines(ctx, grid, cfg=ENVELOPE):
    """Does the initial map survive with no drift? Run before reading the ablation."""
    rows = []
    for mp, beta, anc in itertools.product(grid["maps"], sorted(grid["betas"]),
                                           grid["anchors"]):
        row = {"map": mp, "beta": beta, "anchor_scale": anc}
        try:
            row |= hydraulic_check(run_world(ctx, mp, preset=cfg, beta=beta,
                                             anchor_scale=anc))
        except Exception as e:
            row |= {"passed": False, "fail": f"EXC:{type(e).__name__}", "error": str(e)[:160]}
        rows.append(row)
    return pd.DataFrame(rows)


def ablation(ctx, grid=None, cfg=ENVELOPE, max_combos=None, csv="ablation.csv",
             verbose=True):
    """Beta ladder over (map, targets, land use, income, anchor). Ascending beta,
    stop at the first failure, so robust combos cost the full ladder and fragile
    ones short-circuit."""
    grid = grid or default_grid(ctx)
    combos = list(itertools.product(grid["maps"], range(len(grid["targets"])),
                                    grid["to_lu"], grid["to_incomes"], grid["anchors"]))
    if max_combos and len(combos) > max_combos:
        idx = np.random.default_rng(0).choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in sorted(idx)]

    rows, t0 = [], _time.time()
    for k, (mp, ti, lu, inc, anc) in enumerate(combos):
        tgts = grid["targets"][ti]
        drifts = [{"tgt_district": t, "to_income": inc, "to_land_use": lu} for t in tgts]
        label = "+".join(t.replace("District_", "") for t in tgts)
        for beta in sorted(grid["betas"]):
            row = {"map": mp, "targets": label, "n_targets": len(tgts), "to_land_use": lu,
                   "to_income": inc, "anchor_scale": anc, "beta": beta}
            try:
                row |= hydraulic_check(run_world(ctx, mp, drifts, preset=cfg, beta=beta,
                                                 anchor_scale=anc))
            except Exception as e:
                row |= {"passed": False, "fail": f"EXC:{type(e).__name__}",
                        "error": str(e)[:160]}
            rows.append(row)
            if not row["passed"]:
                break
        if verbose and (k + 1) % 20 == 0:
            el = _time.time() - t0
            print(f"{k+1}/{len(combos)}  {el/60:.1f} min  "
                  f"eta {(el/(k+1))*(len(combos)-k-1)/60:.1f} min", flush=True)
        if csv and (k + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    if csv:
        df.to_csv(csv, index=False)
    return df


def report(df, base=None):
    """Feasibility boundary. Pass `base` to drop combinations whose initial map was
    already infeasible — otherwise beta_max_pass conflates baseline and drift."""
    if base is not None:
        ok = base.loc[base["passed"], ["map", "beta"]]
        df = df.merge(ok.assign(base_ok=True), on=["map", "beta"], how="left")
        df = df[df["base_ok"].fillna(False)]

    key = ["map", "targets", "to_land_use", "to_income", "anchor_scale"]
    passing = df[df["passed"]].groupby(key)["beta"].max().rename("beta_max_pass")
    failing = (df[~df["passed"]].sort_values("beta").groupby(key)
               .first()[["beta", "fail"]].rename(columns={"beta": "beta_first_fail"}))
    edge = passing.to_frame().join(failing, how="outer").reset_index()
    edge["beta_max_pass"] = edge["beta_max_pass"].fillna(-1)

    print(f"runs {len(df)} | combos {len(edge)} | never pass "
          f"{int((edge['beta_max_pass'] < 0).sum())} | always pass "
          f"{int(edge['beta_first_fail'].isna().sum())}\n")
    print("failure modes:")
    print(df.loc[~df["passed"], "fail"].value_counts().to_string(), "\n")
    if "drifted_%" in df and (df["drifted_%"] < 99).any():
        print(f"WARNING: {int((df['drifted_%'] < 99).sum())} runs not fully drifted\n")
    print("beta_max_pass by target land use:")
    print(edge.groupby("to_land_use")["beta_max_pass"]
          .agg(["median", "min", "count"]).round(2).to_string(), "\n")
    print("beta_max_pass by number of drifting clients:")
    print(edge.merge(df[key + ["n_targets"]].drop_duplicates(), on=key)
          .groupby("n_targets")["beta_max_pass"].median().round(2).to_string(), "\n")
    print("tightest 15:")
    print(edge.sort_values(["beta_max_pass", "map"]).head(15).to_string(index=False))
    return edge


def plot_report(df, edge):
    import matplotlib.pyplot as plt
    if "peak_lps" not in df or df["peak_lps"].notna().sum() == 0:
        raise RuntimeError("no run produced hydraulic results — inspect df['error']")
    d = df[df["peak_lps"].notna()]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))

    edge.pivot_table(index="map", columns="to_land_use", values="beta_max_pass",
                     aggfunc="median").plot.bar(ax=ax[0])
    ax[0].set(title="median beta_max_pass by map", xlabel="")
    ax[0].tick_params(axis="x", labelsize=6, rotation=70)

    for sub, c, lab in ((d[d["passed"]], None, "pass"), (d[~d["passed"]], "crimson", "fail")):
        ax[1].scatter(sub["peak_lps"], sub["pmin"], s=8, alpha=.55, c=c, label=lab)
    ax[1].axhline(HARD_FLOOR, color="crimson", ls="--", lw=1)
    ax[1].axhline(BAND[0], color="orange", ls=":", lw=1)
    ax[1].set(xlabel="network peak demand (L/s)", ylabel="min pressure (mca)",
              title="feasibility frontier")
    ax[1].legend(fontsize=7)

    ax[2].scatter(d["peak_lps"], d["pmax"], s=8, alpha=.5)
    ax[2].axhline(BAND[1], color="crimson", ls="--", lw=1)
    ax[2].set(xlabel="network peak demand (L/s)", ylabel="max pressure (mca)",
              title="overpressure check")
    plt.tight_layout()
    plt.show()
    return fig
