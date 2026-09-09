"""Network capacity under constant demand.

The question
-----------
How much demand can each network carry before consumers drop below a pressure
floor? Answering it needs three things kept separate, because conflating them
produces confident nonsense:

1. **What is broken regardless of demand.** At vanishing demand, pressure is
   pure static head: source HGL minus node elevation. A consumer below the floor
   *there* is failing on elevation, not on capacity, and no reduction in demand
   will save it. ``static_head_analysis`` finds these from the graph alone, with
   no solver, so a POWER pump's divergent head at zero flow cannot contaminate
   the answer. These nodes are excluded from the capacity criterion and reported
   as defects -- this is what ky15's J-465 at -99.6 m is.

2. **Whether capacity is even well defined.** Bisection assumes feasibility is
   monotone in the demand scale. PRVs, check valves and POWER pumps can break
   that. So a coarse sweep runs *first* and monotonicity is tested; bisection is
   only trusted when the sweep supports it. Otherwise the curve is the answer and
   the scalar is withheld.

3. **What "failure" means.** Under demand-driven analysis EPANET delivers the
   demand regardless and reports the shortfall as negative pressure -- a cliff.
   Under pressure-driven analysis it reports *unserved demand*, which is the
   physically meaningful quantity and gives a graded curve. Both are computed:
   DD locates the limit, PDD characterises the approach to it.

Method notes
------------
Everything runs at **constant demand** (patterns flattened) and **steady state**
(duration 0). With tanks present an extended run drifts regardless of demand, so
a time-dependent answer would depend on run length rather than on the network.

In steady state a tank is a fixed-head source at its current level, so capacity
depends on an assumed level. It is therefore reported at both the initial and the
minimum level; the gap between them is the storage-dependence of the answer, not
noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr

import network_conditioning as nc

LPS_PER_M3S = 1000.0
SEC_PER_DAY = 86400.0


@dataclass
class CapacityConfig:
    pressure_floor_m: float = 15.0
    lambda_grid: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
                                      3.0, 5.0, 8.0, 12.0, 20.0)
    bisect_tol: float = 0.02          # relative width of the final bracket
    bisect_max_iter: int = 16
    monotonic_tol_m: float = 0.5      # allowed non-monotone wobble in metres
    tank_levels: tuple[str, ...] = ("initial", "min")
    modes: tuple[str, ...] = ("DD", "PDD")
    sanity_head_m: float = nc.SANITY_HEAD_M
    freeze_tanks: bool = False        # freeze instead of holding level fixed


# --------------------------------------------------------------------------
# 1. static head: what is broken independent of demand
# --------------------------------------------------------------------------

def static_head_analysis(wn) -> pd.DataFrame:
    """Gravity-available head at every junction, computed from the graph.

    Walks the network *excluding pumps* (a pump's head is unknown without a
    flow, and a POWER pump's head diverges as flow goes to zero, so no solver
    can be trusted at vanishing demand). For each junction, the available HGL is
    the highest head among gravity-reachable sources: a reservoir's head, or a
    tank's elevation plus its level.

    Caveats recorded per node rather than hidden: a path crossing a PRV is
    *optimistic*, because the valve caps downstream head; a junction reachable
    only through a pump gets ``pump_fed`` and no static verdict.
    """
    pump_links = set(wn.pump_name_list)
    prv_links = {v for v in wn.valve_name_list
                 if wn.get_link(v).valve_type in ("PRV", "PSV", "FCV", "TCV")}

    g = nx.Graph()
    g.add_nodes_from(wn.node_name_list)
    for lname in wn.link_name_list:
        if lname in pump_links:
            continue
        l = wn.get_link(lname)
        if str(getattr(l, "initial_status", "")).upper() == "CLOSED":
            continue
        g.add_edge(l.start_node_name, l.end_node_name,
                   is_prv=lname in prv_links)

    sources = {}
    for r in wn.reservoir_name_list:
        sources[r] = float(wn.get_node(r).base_head)
    for t in wn.tank_name_list:
        tank = wn.get_node(t)
        sources[t] = float(tank.elevation) + float(tank.init_level)

    # highest gravity-reachable source head per component, and whether the
    # cheapest path to it crosses a pressure-reducing valve
    comp_of = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_of[n] = i
    comp_head = {}
    for s, h in sources.items():
        c = comp_of.get(s)
        if c is not None:
            comp_head[c] = max(comp_head.get(c, -np.inf), h)

    consumers = nc.consumer_junctions(wn)
    rows = []
    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        c = comp_of.get(jname)
        hgl = comp_head.get(c, np.nan)
        base = sum(float(ts.base_value or 0.0)
                   for ts in j.demand_timeseries_list) * LPS_PER_M3S
        via_prv = False
        if not np.isnan(hgl):
            best = max((s for s in sources if comp_of.get(s) == c),
                       key=lambda s: sources[s], default=None)
            if best is not None:
                try:
                    path = nx.shortest_path(g, jname, best)
                    via_prv = any(g[u][v].get("is_prv")
                                  for u, v in zip(path, path[1:]))
                except nx.NetworkXNoPath:
                    pass
        rows.append({
            "node": jname,
            "elevation_m": float(j.elevation),
            "base_demand_lps": base,
            "is_consumer": jname in consumers,
            "gravity_hgl_m": hgl,
            "static_pressure_m": hgl - float(j.elevation)
            if not np.isnan(hgl) else np.nan,
            "pump_fed": bool(np.isnan(hgl)),
            "path_crosses_prv": via_prv,
        })
    df = pd.DataFrame(rows)
    return df


def consumer_pressures(wn, lam: float, cfg: CapacityConfig) -> pd.Series:
    """Per-consumer minimum pressure at demand scale ``lam``."""
    wn.options.hydraulic.demand_multiplier = float(lam)
    r = nc.solve(wn, cfg.sanity_head_m)
    res, cons = r.get("_results"), r.get("_consumers")
    if res is None or not cons:
        return pd.Series(dtype=float)
    return res.node["pressure"][cons].min(axis=0)


def empirical_defects(wn, static: pd.DataFrame, cfg: CapacityConfig) -> pd.DataFrame:
    """Consumers that fail at vanishing demand -- the authoritative defect test.

    The graph-based static analysis cannot be the detector on its own: a path
    that crosses a PRV yields an optimistic head, because the valve caps
    everything downstream. ky15's J-465 shows +88.6 m of gravity head by graph
    and sits at -85 m in the solver at lambda=0.05. So the probe is a real solve
    at the smallest lambda on the grid, where friction loss is negligible and
    what remains is the head the network can actually deliver.

    The static table is still reported alongside, because it *explains* each
    failure (elevation vs available head, PRV on the path, pump-fed) in a way a
    single pressure number does not.
    """
    lam0 = float(min(cfg.lambda_grid))
    p = consumer_pressures(wn, lam0, cfg)
    if p.empty:
        return pd.DataFrame()
    bad = p[p < cfg.pressure_floor_m]
    if bad.empty:
        return pd.DataFrame()
    st = static.set_index("node")
    rows = []
    for node, pressure in bad.sort_values().items():
        s = st.loc[node] if node in st.index else None
        rows.append({
            "node": node,
            "probe_lambda": lam0,
            "probe_pressure_m": float(pressure),
            "deficit_m": cfg.pressure_floor_m - float(pressure),
            "negative": bool(pressure < 0),
            "elevation_m": float(s["elevation_m"]) if s is not None else np.nan,
            "gravity_hgl_m": float(s["gravity_hgl_m"]) if s is not None else np.nan,
            "static_pressure_m": float(s["static_pressure_m"]) if s is not None else np.nan,
            "base_demand_lps": float(s["base_demand_lps"]) if s is not None else np.nan,
            "pump_fed": bool(s["pump_fed"]) if s is not None else False,
            "path_crosses_prv": bool(s["path_crosses_prv"]) if s is not None else False,
        })
    df = pd.DataFrame(rows)
    df["explanation"] = np.select(
        [df.pump_fed,
         df.static_pressure_m < cfg.pressure_floor_m,
         df.path_crosses_prv],
        ["pump-fed: no gravity source reaches it",
         "elevation above available gravity head",
         "PRV on the supply path caps the head below the floor"],
        default="fails at negligible demand for another reason -- inspect")
    return df


def static_defects(static: pd.DataFrame, floor_m: float) -> pd.DataFrame:
    """Consumers that fail on static head alone -- defects, not capacity limits.

    ``pump_fed`` nodes are excluded: their head genuinely depends on pumping, so
    static analysis has nothing to say about them.
    """
    d = static[static.is_consumer & ~static.pump_fed].copy()
    d["static_deficit_m"] = floor_m - d["static_pressure_m"]
    d["negative_static"] = d["static_pressure_m"] < 0
    return d[d.static_deficit_m > 0].sort_values("static_deficit_m",
                                                 ascending=False)


# --------------------------------------------------------------------------
# 2. the lambda sweep
# --------------------------------------------------------------------------

def _prepare(source, cfg: CapacityConfig, tank_level: str, mode: str):
    """Constant demand, steady state, chosen tank level and demand model."""
    ccfg = nc.ConditioningConfig(
        fold_demand_multiplier=True,
        flatten_patterns=True,          # constant demand: lambda is the only knob
        duration_h=0.0,                 # steady state: no tank drift
        demand_model="PDD" if mode.upper().startswith("P") else "DD",
        required_pressure_m=cfg.pressure_floor_m,
        minimum_pressure_m=0.0,
        storage_policy="freeze_tanks" if cfg.freeze_tanks else "as_is",
        freeze_level=tank_level,
    )
    wn, recipe = nc.condition_network(source, ccfg)
    if not cfg.freeze_tanks and tank_level != "initial":
        # hold tanks at the chosen level without changing their type
        for t in wn.tank_name_list:
            tank = wn.get_node(t)
            tank.init_level = {"min": tank.min_level,
                               "max": tank.max_level}[tank_level]
    return wn, recipe


def _evaluate(wn, lam: float, excluded: set[str], cfg: CapacityConfig) -> dict:
    """One solve at demand scale ``lam``, scored over eligible consumers only."""
    wn.options.hydraulic.demand_multiplier = float(lam)
    r = nc.solve(wn, cfg.sanity_head_m)
    out = {"lambda": lam, "converged": r["converged"], "sane": r["sane"],
           "error": r["error"]}
    res = r.pop("_results", None)
    cons = r.pop("_consumers", None)
    if res is None or not cons:
        return out | {"pressure_min_m": np.nan, "n_below_floor": np.nan,
                      "frac_below_floor": np.nan, "frac_negative": np.nan,
                      "unserved_frac": np.nan, "worst_nodes": ""}
    eligible = [c for c in cons if c not in excluded]
    if not eligible:
        return out | {"pressure_min_m": np.nan, "n_below_floor": np.nan,
                      "frac_below_floor": np.nan, "frac_negative": np.nan,
                      "unserved_frac": np.nan, "worst_nodes": ""}
    p = res.node["pressure"][eligible]
    vals = p.to_numpy()
    mins = p.min(axis=0)

    expected = lam * sum(
        sum(float(ts.base_value or 0.0)
            for ts in wn.get_node(j).demand_timeseries_list) for j in eligible)
    delivered = np.nan
    if "demand" in res.node:
        delivered = float(res.node["demand"][eligible].to_numpy().mean(axis=0).sum())
    unserved = (1.0 - delivered / expected) if (expected > 0 and
                                                not np.isnan(delivered)) else np.nan

    return out | {
        "pressure_min_m": float(np.nanmin(vals)),
        "pressure_median_m": float(np.nanmedian(vals)),
        "n_eligible": len(eligible),
        "n_below_floor": int((mins < cfg.pressure_floor_m).sum()),
        "frac_below_floor": float((vals < cfg.pressure_floor_m).mean()),
        "frac_negative": float((vals < 0).mean()),
        "delivered_lps": delivered * LPS_PER_M3S if not np.isnan(delivered) else np.nan,
        "expected_lps": expected * LPS_PER_M3S,
        "unserved_frac": max(unserved, 0.0) if not np.isnan(unserved) else np.nan,
        "feasible": bool(np.nanmin(vals) >= cfg.pressure_floor_m),
        "worst_nodes": ", ".join(mins.sort_values().head(3).index),
    }


def check_monotonic(curve: pd.DataFrame, tol_m: float) -> dict:
    """Is min pressure non-increasing in lambda?

    If not, bisection would return an arbitrary point on a non-monotone
    function, so the scalar answer is withheld and the curve reported instead.
    """
    c = curve.dropna(subset=["pressure_min_m"]).sort_values("lambda")
    if len(c) < 3:
        return {"monotonic": False, "reason": "too few successful solves",
                "max_violation_m": np.nan}
    diffs = np.diff(c["pressure_min_m"].to_numpy())
    worst = float(diffs.max())
    ok = worst <= tol_m
    return {"monotonic": bool(ok),
            "max_violation_m": worst,
            "reason": "ok" if ok else
                      f"min pressure rises by {worst:.2f} m as demand grows "
                      f"between lambda="
                      f"{c['lambda'].to_numpy()[int(np.argmax(diffs))]:.3g} and "
                      f"{c['lambda'].to_numpy()[int(np.argmax(diffs)) + 1]:.3g}"}


def bisect_lambda(wn, excluded: set[str], cfg: CapacityConfig,
                  lo: float, hi: float) -> tuple[float, int]:
    """Refine the largest feasible lambda inside a known bracket."""
    n = 0
    for _ in range(cfg.bisect_max_iter):
        if (hi - lo) / max(lo, 1e-9) < cfg.bisect_tol:
            break
        mid = 0.5 * (lo + hi)
        r = _evaluate(wn, mid, excluded, cfg)
        n += 1
        if r["converged"] and r["sane"] and r["feasible"]:
            lo = mid
        else:
            hi = mid
    return lo, n


# --------------------------------------------------------------------------
# 3. binding constraint
# --------------------------------------------------------------------------

def binding_diagnosis(wn, lam: float, static: pd.DataFrame,
                      excluded: set[str], cfg: CapacityConfig) -> pd.DataFrame:
    """At the limit, which consumers bind and why.

    ``static_limited``   already short of the floor on static head alone.
    ``friction_limited`` static head is adequate; the loss to deliver flow is
                         what breaks it -- this is genuine network capacity.
    ``pump_limited``     no gravity source reaches it; supply depends on pumping.
    """
    r = _evaluate(wn, lam, excluded, cfg)
    if not r.get("converged") or np.isnan(r.get("pressure_min_m", np.nan)):
        return pd.DataFrame()
    wn.options.hydraulic.demand_multiplier = lam
    sol = nc.solve(wn, cfg.sanity_head_m)
    res = sol.get("_results")
    if res is None:
        return pd.DataFrame()
    cons = [c for c in sol["_consumers"] if c not in excluded]
    mins = res.node["pressure"][cons].min(axis=0).sort_values()
    st = static.set_index("node")

    rows = []
    for node, pmin in mins.head(25).items():
        s = st.loc[node] if node in st.index else None
        static_p = float(s["static_pressure_m"]) if s is not None else np.nan
        pump_fed = bool(s["pump_fed"]) if s is not None else False
        # Order matters. A PRV on the path caps downstream head, so the gap
        # between gravity head and delivered pressure is the valve setting plus
        # friction, not friction -- calling it friction_limited would blame the
        # pipes for a control decision.
        if pump_fed:
            kind = "pump_limited"
        elif not np.isnan(static_p) and static_p < cfg.pressure_floor_m:
            kind = "static_limited"
        elif s is not None and bool(s["path_crosses_prv"]):
            kind = "prv_limited"
        else:
            kind = "friction_limited"
        rows.append({"node": node, "pressure_at_limit_m": float(pmin),
                     "static_pressure_m": static_p,
                     "elevation_m": float(s["elevation_m"]) if s is not None else np.nan,
                     "base_demand_lps": float(s["base_demand_lps"]) if s is not None else np.nan,
                     # meaningful only when no PRV intervenes (see below)
                     "head_consumed_m": static_p - float(pmin)
                     if not np.isnan(static_p) else np.nan,
                     "binding_kind": kind,
                     "crosses_prv": bool(s["path_crosses_prv"]) if s is not None else False})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the assessment
# --------------------------------------------------------------------------

def assess_capacity(source: str | Path,
                    cfg: CapacityConfig | None = None) -> dict[str, pd.DataFrame]:
    """Full capacity assessment of one network.

    Returns tables: ``static``, ``defects``, ``curve``, ``summary``, ``binding``.
    Never raises: a network that cannot be assessed comes back with the reason in
    ``summary.status``.
    """
    cfg = cfg or CapacityConfig()
    source = Path(source)
    net = source.stem

    base = wntr.network.WaterNetworkModel(str(source))
    static = static_head_analysis(base)
    static.insert(0, "network", net)

    # Defects: what fails regardless of demand. Detected empirically at the
    # smallest lambda (authoritative), explained by the static analysis.
    probe_wn, _ = _prepare(source, cfg, cfg.tank_levels[0], "DD")
    defects = (empirical_defects(probe_wn, static, cfg)
               if nc.consumer_junctions(probe_wn) else pd.DataFrame())
    if not defects.empty:
        defects.insert(0, "network", net)
    excluded = set(defects["node"]) if not defects.empty else set()

    curves, summaries, bindings = [], [], []
    for tank_level in cfg.tank_levels:
        for mode in cfg.modes:
            wn, _ = _prepare(source, cfg, tank_level, mode)
            if not nc.consumer_junctions(wn):
                summaries.append({"network": net, "tank_level": tank_level,
                                  "mode": mode, "status": "no demand in file",
                                  "lambda_star": np.nan})
                continue

            grid = [_evaluate(wn, lam, excluded, cfg) for lam in cfg.lambda_grid]
            curve = pd.DataFrame(grid)
            curve.insert(0, "network", net)
            curve.insert(1, "tank_level", tank_level)
            curve.insert(2, "mode", mode)
            curves.append(curve)

            mono = check_monotonic(curve, cfg.monotonic_tol_m)
            feas = curve[curve.feasible.fillna(False)]
            infeas = curve[~curve.feasible.fillna(False)]

            row = {"network": net, "tank_level": tank_level, "mode": mode,
                   "n_consumers": int(static.is_consumer.sum()),
                   "n_excluded_defects": len(excluded),
                   "base_demand_lps": float(
                       static.loc[static.is_consumer, "base_demand_lps"].sum()),
                   "monotonic": mono["monotonic"],
                   "monotonic_note": mono["reason"],
                   "grid_solves": len(curve),
                   "n_failed_solves": int((~curve.converged).sum()
                                          + (~curve.sane.fillna(False)).sum())}

            if feas.empty:
                row |= {"status": "infeasible at every lambda on the grid",
                        "lambda_star": np.nan, "bisect_solves": 0}
            elif infeas.empty:
                row |= {"status": f"still feasible at lambda="
                                  f"{max(cfg.lambda_grid)} -- raise the grid",
                        "lambda_star": float(max(cfg.lambda_grid)),
                        "bisect_solves": 0}
            elif not mono["monotonic"]:
                row |= {"status": "non-monotone: scalar withheld, read the curve",
                        "lambda_star": np.nan, "bisect_solves": 0}
            else:
                lo = float(feas["lambda"].max())
                hi = float(infeas[infeas["lambda"] > lo]["lambda"].min())
                lam_star, n = bisect_lambda(wn, excluded, cfg, lo, hi)
                row |= {"status": "bracketed", "lambda_star": round(lam_star, 4),
                        "bisect_solves": n}

            if not np.isnan(row["lambda_star"]):
                row["max_demand_lps"] = round(
                    row["base_demand_lps"] * row["lambda_star"], 3)
                row["max_demand_cmd"] = round(
                    row["max_demand_lps"] * SEC_PER_DAY / 1000.0, 1)
                row["headroom_x"] = row["lambda_star"]
                if mode == "DD" and tank_level == cfg.tank_levels[0]:
                    b = binding_diagnosis(wn, row["lambda_star"], static,
                                          excluded, cfg)
                    if not b.empty:
                        b.insert(0, "network", net)
                        b.insert(1, "tank_level", tank_level)
                        bindings.append(b)
                        row["binding_kind_top"] = b["binding_kind"].iloc[0]
            summaries.append(row)

    return {
        "_floor": cfg.pressure_floor_m,
        "static": static,
        "defects": defects,
        "curve": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        "summary": pd.DataFrame(summaries),
        "binding": pd.concat(bindings, ignore_index=True) if bindings else pd.DataFrame(),
    }


def assess_corpus(paths, cfg: CapacityConfig | None = None,
                  verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Assess many networks; a failure on one never stops the batch."""
    import time
    cfg = cfg or CapacityConfig()
    keys = ("static", "defects", "curve", "summary", "binding")
    floor = cfg.pressure_floor_m
    acc = {k: [] for k in keys}
    for p in paths:
        t0 = time.time()
        try:
            out = assess_capacity(p, cfg)
        except Exception as exc:
            out = {"summary": pd.DataFrame([{
                "network": Path(p).stem, "status": f"{type(exc).__name__}: {exc}"[:200],
                "lambda_star": np.nan}])}
        for k in keys:
            df = out.get(k)
            if df is not None and len(df):
                acc[k].append(df)
        if verbose:
            s = out.get("summary", pd.DataFrame())
            lam = (s["lambda_star"].dropna().min()
                   if "lambda_star" in s and s["lambda_star"].notna().any() else np.nan)
            print(f"  {Path(p).stem:12s} {time.time() - t0:5.1f}s  "
                  f"lambda*={lam if not np.isnan(lam) else 'n/a'}")
    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
           for k, v in acc.items()}
    out["_floor"] = floor
    return out


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------

def plot_capacity(out: dict, network: str | None = None, figsize=(12, 4.2)):
    """Capacity curves: pressure, share below floor, unserved demand."""
    import matplotlib.pyplot as plt

    curve = out["curve"]
    if network:
        curve = curve[curve.network == network]
    net = network or (curve["network"].iloc[0] if len(curve) else "?")
    cfg_floor = None
    summ = out["summary"]
    if network:
        summ = summ[summ.network == network]

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for (lvl, mode), g in curve.groupby(["tank_level", "mode"]):
        g = g.sort_values("lambda")
        label = f"{mode}, tanks at {lvl}"
        ax[0].plot(g["lambda"], g["pressure_min_m"], marker="o", ms=3, label=label)
        ax[1].plot(g["lambda"], g["frac_below_floor"], marker="o", ms=3, label=label)
        if g["unserved_frac"].notna().any():
            ax[2].plot(g["lambda"], g["unserved_frac"], marker="o", ms=3, label=label)

    floor = out.get("_floor", 15.0)
    ax[0].axhline(floor, color="crimson", ls="--", lw=1, label=f"floor {floor} m")
    ax[0].axhline(0, color="k", lw=0.6)
    for lam in summ["lambda_star"].dropna().unique():
        for a in ax:
            a.axvline(lam, color="0.5", ls=":", lw=1)
    ax[0].set_ylabel("min consumer pressure (m)")
    ax[1].set_ylabel("share of consumer node-hours below floor")
    ax[2].set_ylabel("unserved demand fraction (PDD)")
    for a in ax:
        a.set_xlabel("demand scale $\\lambda$")
        a.set_xscale("log")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"{net} — capacity under constant demand (steady state)")
    fig.tight_layout()
    return fig
