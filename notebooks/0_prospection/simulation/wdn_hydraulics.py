"""Hydraulics: one monolithic EPANET run over the whole horizon.

The volume-exactness mechanism
------------------------------
Every consumer gets ``base_value = 1 L/s`` and a pattern whose multipliers *are*
its L/s series. EPANET then reproduces the requested demand identically, and the
whole class of scaling bugs -- a stray ``1/24``, an unfolded ``DEMAND
MULTIPLIER`` -- becomes structurally impossible rather than merely tested for.
There is no rescaling step, so there is nothing to drift.

Requested and delivered are two different series
------------------------------------------------
Under ``PDD``, EPANET sheds demand at nodes whose pressure falls below
``required_pressure``. That makes the demand you observe a **censored** version
of the demand the population asked for, and it saturates exactly when the drift
is largest. It also means a neighbour's growth reduces *my* delivered demand, so
hydraulic coupling shows up directly in the demand channel and not only in
pressure.

Both series are therefore returned. The drift ground truth is a statement about
*requested* demand, while every estimator sees *delivered* -- so any evaluation
that silently compares one to the other is measuring the censoring, and
``censoring_report`` exists to say how much of that there is.

Why nothing is chunked
----------------------
EPANET is never restarted, so tank levels and pump status are continuous by
construction: no state carry, no clipping, and none of the one-step stalls that
a chunked run injects at every boundary.
"""
from __future__ import annotations

import copy
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import wntr

LPS_PER_M3S = 1000.0


@dataclass
class RunConfig:
    """Execution options for the horizon run."""
    demand_model: str = "PDD"
    required_pressure_m: float = 15.0
    minimum_pressure_m: float = 0.0
    pressure_floor_m: float = 15.0        # reporting threshold, not a solver knob
    timestep_s: int = 3600
    hydraulic_timestep_s: int = 900       # finer than reporting: control events
    sanity_head_m: float = 1000.0
    record_links: bool = True             # flows: needed for every flow sensor

    # Solver convergence. These are deliberately tighter than what most .inp
    # files ship with, and the difference is not cosmetic. D-Town ships
    # ACCURACY 0.01 with UNBALANCED CONTINUE, which lets EPANET accept an
    # unbalanced solution and report it with no error. Measured at 0.8x
    # capacity under PDD, that leaves a per-node mass-balance residual of up to
    # 0.27 L/s -- about 18% of a node's demand, and indistinguishable from
    # signal to any demand-anomaly detector reading the delivered series.
    #   accuracy 0.01 (as shipped) -> worst residual 0.2675 L/s, 116 cells
    #   accuracy 1e-3 (EPANET default) -> 0.0163 L/s, 1 cell
    #   accuracy 1e-5 -> 9.3e-5 L/s, 0 cells (the single-precision floor)
    # The cost of the tightest setting was one second in seven.
    accuracy: float = 1e-5
    trials: int = 500
    unbalanced: str = "CONTINUE"
    unbalanced_trials: int = 10

    verbose: bool = True

    def validate(self):
        if self.hydraulic_timestep_s > self.timestep_s:
            raise ValueError("hydraulic timestep must not exceed the report step")
        if self.timestep_s % self.hydraulic_timestep_s:
            raise ValueError("report step must be a multiple of the hydraulic step")
        return self


def configure_network(wn, n_steps: int, run: RunConfig):
    """A copy with demands zeroed, the clock set, and the demand model pinned."""
    run = run.validate()
    w = copy.deepcopy(wn)

    # fold any file-level multiplier so our numbers mean what they say
    m0 = float(w.options.hydraulic.demand_multiplier or 1.0)
    if abs(m0 - 1.0) > 1e-12:
        for n in w.junction_name_list:
            for ts in w.get_node(n).demand_timeseries_list:
                if ts.base_value:
                    ts.base_value = float(ts.base_value) * m0
    w.options.hydraulic.demand_multiplier = 1.0

    w.options.hydraulic.accuracy = float(run.accuracy)
    w.options.hydraulic.trials = int(run.trials)
    w.options.hydraulic.unbalanced = run.unbalanced
    w.options.hydraulic.unbalanced_value = int(run.unbalanced_trials)

    w.options.hydraulic.demand_model = run.demand_model
    if run.demand_model.upper().startswith("P"):
        w.options.hydraulic.required_pressure = run.required_pressure_m
        w.options.hydraulic.minimum_pressure = run.minimum_pressure_m

    # EPANET reports at t = 0, ts, ..., duration, i.e. duration/ts + 1 points.
    # (n_steps - 1) steps therefore yields exactly n_steps points.
    w.options.time.duration = int((n_steps - 1) * run.timestep_s)
    w.options.time.hydraulic_timestep = int(run.hydraulic_timestep_s)
    w.options.time.report_timestep = int(run.timestep_s)
    w.options.time.pattern_timestep = int(run.timestep_s)

    for n in w.junction_name_list:
        for ts in w.get_node(n).demand_timeseries_list:
            ts.base_value = 0.0
            ts.pattern_name = None
    return w


def run_hydraulics(wn, demand_series: pd.DataFrame,
                   run: RunConfig | None = None,
                   block_months: int | None = None,
                   overlap_days: float = 7.0,
                   steps_per_month: int = 720) -> dict:
    """Solve the horizon and return every observable, on one uniform clock.

    ``block_months=None`` solves in a single EPANET run, which is the default
    and the preferred path: state is continuous because the solver never
    restarts. Peak memory is roughly linear in steps x nodes -- measured 613 MB
    at 8 months on this network, so about 3 GB at 40 months, because wntr
    materialises all four node and all seven link output variables whatever you
    ask for.

    Set ``block_months`` when that does not fit. Blocks are stitched with an
    **overlap that is discarded**, not butted together: each block re-simulates
    the preceding ``overlap_days`` from the previous block's recorded state and
    those steps are then dropped. The discarded warm-up absorbs both the restart
    transient and the one-step stall that a naive carry produces, so the
    reported series has no duplicated step at a seam. It is strictly worse than
    monolithic and only worth it under a memory ceiling.
    """
    run = (run or RunConfig()).validate()
    if block_months is None:
        return _run_window(wn, demand_series, run)
    return _run_blocked(wn, demand_series, run, block_months, overlap_days,
                        steps_per_month)


def _run_window(wn, demand_series: pd.DataFrame, run: RunConfig,
                offset: int = 0, init_tanks: dict | None = None) -> dict:
    """One EPANET solve over the given demand series."""
    nodes = [c for c in demand_series.columns if c != "month"]
    T = len(demand_series)
    w = configure_network(wn, T, run)
    if init_tanks:
        for t, lvl in init_tanks.items():
            tk = w.get_node(t)
            tk.init_level = float(np.clip(lvl, tk.min_level, tk.max_level))

    unknown = [n for n in nodes if n not in w.junction_name_list]
    if unknown:
        raise ValueError(f"demand_series has non-junction columns: {unknown[:5]}")

    t0 = time.time()
    for n in nodes:
        vals = demand_series[n].to_numpy(dtype=float)      # L/s
        pat = f"_d_{n}"
        if pat in w.pattern_name_list:
            w.remove_pattern(pat)
        w.add_pattern(pat, list(vals))
        node = w.get_node(n)
        node.demand_timeseries_list.clear()
        node.demand_timeseries_list.append((1.0 / LPS_PER_M3S, pat, "synth"))
    for n in w.junction_name_list:
        if len(w.get_node(n).demand_timeseries_list) == 0:
            w.get_node(n).demand_timeseries_list.append((0.0, None, "none"))
    t_build = time.time() - t0

    t0 = time.time()
    with tempfile.TemporaryDirectory() as scratch:
        cwd = os.getcwd()
        try:
            os.chdir(scratch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = wntr.sim.EpanetSimulator(w).run_sim()
        finally:
            os.chdir(cwd)
    t_solve = time.time() - t0

    step = np.arange(T) + offset
    pressures = res.node["pressure"].iloc[:T].set_axis(step).astype(np.float32)
    delivered = (res.node["demand"].iloc[:T].set_axis(step)
                 * LPS_PER_M3S).astype(np.float32)
    heads = res.node["head"].iloc[:T].set_axis(step).astype(np.float32)

    tanks = list(w.tank_name_list)
    tank_levels = pd.DataFrame(
        {t: heads[t] - float(w.get_node(t).elevation) for t in tanks},
        index=step).astype(np.float32) if tanks else pd.DataFrame(index=step)

    pumps = list(w.pump_name_list)
    pump_status = (res.link["status"].iloc[:T].set_axis(step)[pumps]
                   .astype(np.float32) if pumps and "status" in res.link
                   else pd.DataFrame(index=step))

    flows = (res.link["flowrate"].iloc[:T].set_axis(step) * LPS_PER_M3S
             ).astype(np.float32) if run.record_links else pd.DataFrame(index=step)

    if run.verbose:
        print("  built %d node patterns in %.0fs, solved %d steps in %.0fs"
              % (len(nodes), t_build, T, t_solve))

    return {"pressures": pressures,
            "flows": flows,
            "demands_delivered": delivered[nodes],
            "demands_requested": demand_series[nodes].set_axis(step),
            "tank_levels": tank_levels,
            "pump_status": pump_status,
            "month": demand_series.month.to_numpy(),
            "run": asdict(run),
            "seconds": {"build": round(t_build, 1), "solve": round(t_solve, 1)}}


def _run_blocked(wn, demand_series: pd.DataFrame, run: RunConfig,
                 block_months: int, overlap_days: float,
                 steps_per_month: int) -> dict:
    """Solve in blocks joined by a discarded overlap."""
    T = len(demand_series)
    blk = int(block_months * steps_per_month)
    ov = int(round(overlap_days * 24 * 3600 / run.timestep_s))
    if blk <= ov:
        raise ValueError("block_months must cover more than overlap_days")
    tanks = list(wn.tank_name_list)

    parts, init, seams = [], None, []
    starts = list(range(0, T, blk))
    for bi, s0 in enumerate(starts):
        s1 = min(s0 + blk, T)
        w0 = s0 if bi == 0 else max(0, s0 - ov)      # warm-up start
        win = demand_series.iloc[w0:s1].reset_index(drop=True)
        if run.verbose:
            print("block %d/%d: steps %d-%d (warm-up %d)"
                  % (bi + 1, len(starts), s0, s1 - 1, s0 - w0))
        part = _run_window(wn, win, run, offset=w0, init_tanks=init)
        keep = part["pressures"].index >= s0          # drop the overlap
        for k in ("pressures", "flows", "demands_delivered",
                  "demands_requested", "tank_levels", "pump_status"):
            if len(part[k].columns):
                part[k] = part[k].loc[keep]
        part["month"] = demand_series.month.to_numpy()[s0:s1]
        parts.append(part)
        if bi:
            seams.append(s0)
        if tanks:
            # carry the state from the point the next warm-up starts, so the
            # warm-up re-derives the trajectory rather than approximating it
            nxt = max(0, min(s1, T) - ov)
            lv = part["tank_levels"]
            row = lv.index[lv.index >= nxt]
            init = {t: float(lv.loc[row[0], t]) for t in tanks} if len(row) \
                else {t: float(lv.iloc[-1][t]) for t in tanks}

    def cat(key):
        fr = [p[key] for p in parts if len(p[key].columns)]
        return pd.concat(fr).sort_index() if fr else pd.DataFrame()

    return {"pressures": cat("pressures"), "flows": cat("flows"),
            "demands_delivered": cat("demands_delivered"),
            "demands_requested": cat("demands_requested"),
            "tank_levels": cat("tank_levels"),
            "pump_status": cat("pump_status"),
            "month": np.concatenate([p["month"] for p in parts]),
            # Seam steps are reported so downstream analysis can mask them.
            # Measured against a monolithic solve on this network, a 28-day
            # overlap brings the mean pressure error to 0.013 m but leaves
            # isolated ~20 m spikes where a pump switches one step differently.
            # That is pump-switching phase, not a decaying transient, so no
            # affordable overlap removes it -- which is the substantive reason
            # to prefer the monolithic path rather than a stylistic one.
            "seam_steps": seams,
            "run": asdict(run), "blocks": len(starts),
            "seconds": {"solve": round(sum(p["seconds"]["solve"]
                                           for p in parts), 1)}}


# ==========================================================================
# reporting
# ==========================================================================

def health_report(sim: dict, wn, consumers=None) -> pd.DataFrame:
    """Per-month pressure, service, tank and pump health."""
    P, month = sim["pressures"], sim["month"]
    cons = list(consumers) if consumers is not None \
        else list(sim["demands_requested"].columns)
    cols = [c for c in cons if c in P.columns]
    floor = sim["run"]["pressure_floor_m"]
    req, dlv = sim["demands_requested"], sim["demands_delivered"]
    TL, PS = sim["tank_levels"], sim["pump_status"]

    rows = []
    for m in np.unique(month):
        k = month == m
        p = P.loc[k, cols].to_numpy()
        r, d = req.loc[k].to_numpy().sum(), dlv.loc[k].to_numpy().sum()
        row = {"month": int(m), "n_steps": int(k.sum()),
               "pressure_min_m": float(np.nanmin(p)),
               "pressure_median_m": float(np.nanmedian(p)),
               "frac_below_floor": float((p < floor).mean()),
               "frac_negative": float((p < 0).mean()),
               "requested_lps": float(r / k.sum()),
               "delivered_lps": float(d / k.sum()),
               "unserved_frac": float(max(0.0, 1 - d / r)) if r > 0 else np.nan}
        if len(TL.columns):
            fmin, fmax = [], []
            for t in TL.columns:
                tk = wn.get_node(t)
                fmin.append(float((TL.loc[k, t] <= float(tk.min_level) + 0.02).mean()))
                fmax.append(float((TL.loc[k, t] >= float(tk.max_level) - 0.02).mean()))
            row |= {"tank_frac_at_min": float(np.max(fmin)),
                    "tank_frac_at_max": float(np.max(fmax))}
        if len(PS.columns):
            st = PS.loc[k].to_numpy()
            row |= {"pump_duty_cycle": float(st.mean()),
                    "pump_switches": int(np.abs(np.diff(st, axis=0)).sum())}
        rows.append(row)
    return pd.DataFrame(rows)


def pump_curve_report(sim: dict, wn) -> pd.DataFrame:
    """Where pumps operate beyond the right-hand end of their head curve.

    EPANET extrapolates past the last curve point rather than refusing, and
    only says so in the .rpt. On D-Town at 0.8x capacity, PU1 exceeds its
    maximum curve flow at ~120 steps, and that persists at every solver
    accuracy -- so it is a modelling condition of the network under load, not a
    convergence artefact. It matters because head on the extrapolated branch is
    an extension of the fitted curve, not measured pump behaviour.
    """
    flows, rows = sim["flows"], []
    for pn in sim["pump_status"].columns:
        if pn not in flows.columns:
            continue
        pump = wn.get_link(pn)
        cname = getattr(pump, "pump_curve_name", None)
        qmax = np.nan
        if cname:
            pts = wn.get_curve(cname).points
            qmax = max(x for x, _ in pts) * LPS_PER_M3S
        q = flows[pn].to_numpy()
        on = sim["pump_status"][pn].to_numpy() > 0
        rows.append({"pump": pn, "curve": cname,
                     "curve_max_flow_lps": qmax,
                     "max_flow_lps": float(np.nanmax(q)) if q.size else np.nan,
                     "duty_cycle": float(on.mean()),
                     "frac_steps_beyond_curve": float(
                         np.mean(on & (q > qmax))) if qmax == qmax else np.nan})
    return pd.DataFrame(rows)


def censoring_report(sim: dict) -> pd.DataFrame:
    """How much of the requested demand PDD sheds, and where.

    The point of this report is that unserved demand is not an error term. It is
    the mechanism by which the network's limits enter the observable, so it has
    to be quantified per month and per district-facing node before any estimator
    is trusted on the delivered series.
    """
    req, dlv, month = sim["demands_requested"], sim["demands_delivered"], sim["month"]
    gap = (req.to_numpy() - dlv.to_numpy())
    rows = []
    for m in np.unique(month):
        k = month == m
        g, r = gap[k], req.to_numpy()[k]
        rows.append({"month": int(m),
                     "unserved_lps": float(g.sum() / k.sum()),
                     "unserved_frac": float(g.sum() / r.sum()) if r.sum() else np.nan,
                     "frac_node_hours_censored": float((g > 1e-3).mean()),
                     "n_nodes_ever_censored": int((g > 1e-3).any(axis=0).sum()),
                     "worst_node_unserved_frac": float(
                         (g.sum(axis=0) / np.where(r.sum(axis=0) > 0,
                                                   r.sum(axis=0), np.nan)).max())})
    return pd.DataFrame(rows)


def validate_simulation(sim: dict, wn, run: RunConfig,
                        volume_atol_lps: float = 1e-3,
                        volume_rtol: float = 1e-3,
                        volume_floor_lps: float = 1e-2) -> pd.DataFrame:
    """The checks a long run can silently get wrong. V1-V3 are hard.

    V4-V6 warn: on a network at its service limit, low pressure and shed demand
    are findings, not bugs.
    """
    P, month = sim["pressures"], sim["month"]
    req, dlv = sim["demands_requested"], sim["demands_delivered"]
    checks = []

    # V1 clock integrity -- one uniform step, no gaps, no duplicates
    n_expected = len(req)
    dt = np.diff(P.index.to_numpy())
    checks.append({"check": "V1_clock", "hard": True,
                   "value": float(len(P)), "expected": float(n_expected),
                   "passed": bool(len(P) == n_expected
                                  and (dt == 1).all()),
                   "note": "uniform single-step index, no chunk boundaries"})

    # V2 volume exactness where service is adequate. Restricting to adequately
    # served node-hours is the whole point: under PDD a mismatch at a
    # pressure-deficient node is correct behaviour, so a global mass balance
    # would conflate a bookkeeping bug with a service finding.
    #
    # The tolerance is combined absolute-or-relative, not relative alone. A
    # purely relative test is meaningless here: EPANET's binary output is single
    # precision, and the smallest consumers on this network draw ~6e-5 L/s, so a
    # 2e-5 L/s rounding difference reads as a 34% error while the absolute error
    # never exceeds 9e-5 L/s. An absolute floor of 1e-3 L/s is far below any
    # meter's resolution and still catches every scaling bug this check exists
    # for -- a stray 1/24 shows up as 96% on every node, a x0.2 as 80%.
    ok = P[req.columns].to_numpy() >= run.required_pressure_m
    r, d = req.to_numpy(), dlv.to_numpy()
    slack = np.abs(d[ok] - r[ok]) - (volume_atol_lps + volume_rtol * np.abs(r[ok]))
    worst_abs = float(np.nanmax(np.abs(d[ok] - r[ok]))) if ok.any() else 0.0
    checks.append({"check": "V2_volume_exact_where_served", "hard": True,
                   "value": worst_abs, "expected": volume_atol_lps,
                   "passed": bool(not ok.any() or np.nanmax(slack) <= 0),
                   "note": "worst |requested - delivered| in L/s at served nodes"})

    # V2b the invariant as actually claimed: total volume, system and per node.
    # This is the number the "volume exact by construction" claim rests on, and
    # it is not implied by the per-step check above.
    vr, vd = r.sum(axis=0), d.sum(axis=0)
    sys_err = float(abs(vd.sum() - vr.sum()) / max(vr.sum(), 1e-12))
    checks.append({"check": "V2b_volume_system", "hard": True,
                   "value": sys_err, "expected": 1e-3,
                   "passed": bool(sys_err < 1e-3),
                   "note": "system-wide relative volume error over the horizon"})

    # Per-node volume error is reported only for nodes carrying non-trivial
    # demand. Below the floor the metric measures single-precision output, not
    # hydraulics: the smallest consumers here average ~6e-5 L/s, so accumulated
    # rounding of ~9e-5 L/s per step reaches ~10% of their total volume while
    # nothing is actually wrong. Above the floor, a real per-node discrepancy
    # means shed demand, which under PDD is a finding rather than a bug.
    keep = (r.mean(axis=0) >= volume_floor_lps)
    node_err = float(np.nanmax(np.abs(vd[keep] - vr[keep])
                               / np.where(vr[keep] > 0, vr[keep], np.nan))) \
        if keep.any() else 0.0
    checks.append({"check": "V2c_volume_worst_node", "hard": False,
                   "value": node_err, "expected": 1e-2,
                   "passed": bool(node_err < 1e-2),
                   "note": "worst per-node relative volume error among the %d "
                           "of %d nodes above %.3g L/s mean demand"
                           % (int(keep.sum()), len(keep), volume_floor_lps)})

    # V3 physical plausibility
    pmax = float(np.nanmax(np.abs(P.to_numpy())))
    checks.append({"check": "V3_head_sane", "hard": True,
                   "value": pmax, "expected": run.sanity_head_m,
                   "passed": bool(pmax < run.sanity_head_m),
                   "note": "no non-physical heads"})

    # V4 service level
    frac = float((P[req.columns].to_numpy() < run.pressure_floor_m).mean())
    checks.append({"check": "V4_pressure_floor", "hard": False,
                   "value": frac, "expected": 0.05, "passed": bool(frac < 0.05),
                   "note": "fraction of consumer node-hours below the floor"})

    # V5 tank sustainability
    TL = sim["tank_levels"]
    if len(TL.columns):
        f = max(float((TL[t] <= float(wn.get_node(t).min_level) + 0.02).mean())
                for t in TL.columns)
        checks.append({"check": "V5_tank_not_drained", "hard": False,
                       "value": f, "expected": 0.02, "passed": bool(f < 0.02),
                       "note": "worst tank's fraction of time on its floor"})

    # V6 censoring is bounded
    u = float(max(0.0, 1 - d.sum() / r.sum()))
    checks.append({"check": "V6_unserved_bounded", "hard": False,
                   "value": u, "expected": 0.02, "passed": bool(u < 0.02),
                   "note": "horizon-wide unserved fraction (a finding, not a bug)"})

    # V7 pumps operating on the extrapolated branch of their curve
    if len(sim["pump_status"].columns) and len(sim["flows"].columns):
        pc = pump_curve_report(sim, wn)
        f = float(np.nanmax(pc.frac_steps_beyond_curve)) if len(pc) else 0.0
        checks.append({"check": "V7_pump_within_curve", "hard": False,
                       "value": f, "expected": 0.01, "passed": bool(f < 0.01),
                       "note": "worst pump's fraction of steps beyond its "
                               "curve's maximum flow (head is extrapolated there)"})

    out = pd.DataFrame(checks)
    bad = out[out.hard & ~out.passed]
    if len(bad):
        raise AssertionError("hard validation checks failed:\n"
                             + bad.to_string(index=False))
    return out
