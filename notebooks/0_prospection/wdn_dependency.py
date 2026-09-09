"""Interventional dependency mapping for water distribution networks.

What this is for
----------------
A water network is a rare setting in which the *true* causal graph is obtainable:
we can apply the do-operator directly, perturb one node's demand, and measure the
effect everywhere else. That interventional map is the ground truth against which
any observational estimator (Granger causality, transfer entropy) must be judged.
Without it, a causality claim on this system is an assertion.

Two maps, because the physics has two regimes
---------------------------------------------
``instantaneous_map`` -- steady-state sensitivity S[i, j] = d p_i / d d_j, one
solve per probed node. EPANET's extended-period simulation is *quasi-steady*:
each timestep solves an independent steady-state problem, so pressure responds to
demand with no lag by construction. S therefore captures **all** of the
same-timestep coupling.

A property worth stating because it is testable and consequential: for a network
of pipes only, S is the inverse of a symmetric matrix, so S must be symmetric
(reciprocity, the hydraulic analogue of Maxwell-Betti). Measured asymmetry is
therefore either numerical error or a genuinely directional element -- a PRV, a
check valve, a pump. On this corpus the valveless networks come in at ~0.001
relative asymmetry while PRV-heavy ones reach 0.1-0.2, so
``symmetry_diagnostics`` is not decoration: it *localizes* directional coupling.

The corollary matters for the whole project: **instantaneous hydraulic coupling
is undirected**. Any genuine direction in this system must come from storage
(tank mass) or from control switching. A network with no tanks and no controls
cannot produce lagged causality at all, whatever a time-series estimator reports.

``dynamic_map`` -- land-use perturbation. Every consumer runs a common diurnal
shape; one node is switched to a different shape *normalized to the same daily
volume*, so only the shape changes and not the total. An extended run then gives
both response magnitude and the lag at which it arrives. Volume normalization is
the control that makes this a shape experiment rather than a disguised magnitude
experiment.

Downfalls this module actively guards against
---------------------------------------------
* **Perturbation size.** Too small and the response is EPANET's convergence
  tolerance; too large and second-order terms corrupt a derivative
  interpretation. ``calibrate_delta`` picks the step from measured linearity
  rather than from a hardcoded constant.
* **Absolute vs relative probes.** A common absolute step gives a comparable
  Jacobian (and lets reciprocity be tested); a proportional step gives realistic
  influence but confounds "influential" with "large". Both are supported and the
  choice is recorded.
* **A noise floor that is asserted rather than measured.** Central differencing
  yields the second-order/noise term as a by-product, which is used as the floor.
* **Operating-point dependence.** Influence is nonlinear in flow, so a map made
  on a lightly loaded network need not describe it under growth. Maps can be
  built at several demand multipliers and compared.
* **Structural zeros are information.** An exact zero means hydraulic separation
  (a distinct pressure zone), not a weak link, and is reported separately.
"""
from __future__ import annotations

import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr

LPS_PER_M3S = 1000.0


# ==========================================================================
# configuration
# ==========================================================================

@dataclass
class DependencyConfig:
    """Every knob, recorded with each run. No magic numbers in the algorithms."""

    # --- probe design -----------------------------------------------------
    perturbation_mode: str = "absolute"   # "absolute" | "relative"
    delta_abs_lps: float | None = None    # None -> calibrate_delta decides
    delta_rel: float = 0.10               # used when mode == "relative"
    central_difference: bool = True       # +d and -d: signal + noise floor
    operating_lambdas: tuple[float, ...] = (1.0,)
    record_flow: bool = False   # also map link-flow sensitivity (path structure)

    # --- delta calibration ------------------------------------------------
    calibrate: bool = True
    calibration_fractions: tuple[float, ...] = (
        0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    calibration_ratio_tolerance: float = 0.25   # accept a step within 25% of
    calibration_probes: int = 5                 # the best-conditioned ratio

    # --- dynamic (shape) map ---------------------------------------------
    dynamic_hours: int = 96
    dynamic_warmup_h: int = 24
    dynamic_timestep_s: int = 3600
    max_lag_h: int = 12

    # --- subsetting -------------------------------------------------------
    max_probe_nodes: int | None = None    # None = every consumer
    probe_strategy: str = "stratified"    # "stratified" | "random" | "first"
    seed: int = 0

    # --- solver -----------------------------------------------------------
    demand_model: str = "DD"
    required_pressure_m: float = 15.0
    minimum_pressure_m: float = 0.0
    sanity_head_m: float = 1000.0

    # --- thresholds -------------------------------------------------------
    floor_safety_factor: float = 3.0      # affected iff |S| > factor * floor

    def validate(self):
        if self.perturbation_mode not in ("absolute", "relative"):
            raise ValueError("perturbation_mode must be 'absolute' or 'relative'")
        if self.probe_strategy not in ("stratified", "random", "first"):
            raise ValueError("probe_strategy must be stratified|random|first")
        if self.dynamic_warmup_h >= self.dynamic_hours:
            raise ValueError("dynamic_warmup_h must be < dynamic_hours")
        if self.max_lag_h >= self.dynamic_hours - self.dynamic_warmup_h:
            raise ValueError("max_lag_h too large for the run length")
        return self


# ==========================================================================
# solver plumbing
# ==========================================================================

def solve(wn):
    """Run EPANET in a scratch directory so no temp files touch the repo."""
    with tempfile.TemporaryDirectory() as scratch:
        cwd = os.getcwd()
        try:
            os.chdir(scratch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return wntr.sim.EpanetSimulator(wn).run_sim()
        finally:
            os.chdir(cwd)


def consumer_nodes(wn) -> list[str]:
    """Demand-bearing junctions that are not pump/valve scaffolding.

    Pump and valve links need two nodes, so model builders insert zero-demand
    junctions on either side (``I-Pump-3``, ``O-RV-25`` in the KY corpus). Those
    are not consumers: they carry no demand, represent nobody, and must not
    become probes or federated clients.
    """
    out = []
    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        base = sum(float(ts.base_value or 0.0) for ts in j.demand_timeseries_list)
        if base <= 0:
            continue
        links = list(wn.get_links_for_node(jname))
        types = {wn.get_link(l).link_type for l in links}
        if 0 < len(links) <= 2 and types and types.issubset({"Pump", "Valve"}):
            continue
        out.append(jname)
    return out


def base_demand_of(wn, node: str) -> float:
    return sum(float(ts.base_value or 0.0)
               for ts in wn.get_node(node).demand_timeseries_list)


def _set_extra_demand(wn, node: str, extra_m3s: float):
    """Add ``extra`` to the node's first demand category, returning the old value."""
    ts = wn.get_node(node).demand_timeseries_list[0]
    old = float(ts.base_value or 0.0)
    ts.base_value = old + extra_m3s
    return old


def _restore_demand(wn, node: str, old: float):
    wn.get_node(node).demand_timeseries_list[0].base_value = old


FLOAT32_EPS = float(np.finfo(np.float32).eps)   # 1.19e-7


def quantization_floor(pressures: pd.Series) -> float:
    """Analytic noise floor of a pressure difference read from EPANET output.

    EPANET writes results to its binary output file as 4-byte floats, so every
    reported pressure carries a relative quantization error of about 1.2e-7.
    A central-difference second-order term combines three such values, so the
    floor on that quantity is roughly ``2 * max|p| * eps32``.

    This is the real precision limit of the method and it cannot be lowered by
    tightening the solver: setting EPANET's ``accuracy`` from 1e-3 to 1e-8
    changes nothing, while the float64 pure-Python solver reproduces the same
    first-order term with a ~5x smaller second-order term. The practical
    consequence is that the perturbation step must be large enough to lift the
    response above this floor, which is exactly what ``calibrate_delta``
    enforces. In dense-grid networks, where many parallel paths absorb a single
    node's perturbation, that required step can approach the node's own demand.
    """
    p = np.abs(np.asarray(pressures, dtype=float))
    p = p[np.isfinite(p)]
    return float(2.0 * p.max() * FLOAT32_EPS) if p.size else np.nan


def prepare_steady(wn, cfg: DependencyConfig, lam: float = 1.0):
    """A steady-state copy with the demand multiplier set and options pinned.

    The file's own multiplier is folded into base demands first (Balerma ships
    0.45), so ``lam`` means what it says instead of compounding with a hidden
    scale.
    """
    import copy
    w = copy.deepcopy(wn)
    m = float(w.options.hydraulic.demand_multiplier or 1.0)
    if abs(m - 1.0) > 1e-12:
        for jname in w.junction_name_list:
            for ts in w.get_node(jname).demand_timeseries_list:
                if ts.base_value:
                    ts.base_value = float(ts.base_value) * m
    w.options.hydraulic.demand_multiplier = float(lam)
    w.options.hydraulic.demand_model = cfg.demand_model
    if cfg.demand_model.upper().startswith("P"):
        w.options.hydraulic.required_pressure = cfg.required_pressure_m
        w.options.hydraulic.minimum_pressure = cfg.minimum_pressure_m
    w.options.time.duration = 0
    return w


# ==========================================================================
# probe selection
# ==========================================================================

def select_probes(wn, cfg: DependencyConfig) -> tuple[list[str], pd.DataFrame]:
    """Choose which consumers to probe, and record how.

    ``stratified`` spreads the sample across demand magnitude and node degree so
    a subset map is not dominated by one kind of node. A full map is always
    preferable when affordable; this exists for ky17-scale networks.
    """
    cons = consumer_nodes(wn)
    meta = pd.DataFrame({
        "node": cons,
        "base_demand_lps": [base_demand_of(wn, n) * LPS_PER_M3S for n in cons],
        "degree": [len(list(wn.get_links_for_node(n))) for n in cons],
    })
    if cfg.max_probe_nodes is None or cfg.max_probe_nodes >= len(cons):
        meta["probed"] = True
        return cons, meta

    rng = np.random.default_rng(cfg.seed)
    if cfg.probe_strategy == "first":
        pick = cons[: cfg.max_probe_nodes]
    elif cfg.probe_strategy == "random":
        pick = list(rng.choice(cons, size=cfg.max_probe_nodes, replace=False))
    else:
        # stratify on demand quartile x degree class
        q = pd.qcut(meta.base_demand_lps.rank(method="first"), 4, labels=False)
        d = np.clip(meta.degree, 1, 4)
        meta["_stratum"] = q.astype(str) + "_" + d.astype(str)
        per = max(1, cfg.max_probe_nodes // meta._stratum.nunique())
        pick = []
        for _, grp in meta.groupby("_stratum"):
            take = min(per, len(grp))
            pick += list(rng.choice(grp.node.to_numpy(), size=take, replace=False))
        remaining = [n for n in cons if n not in set(pick)]
        short = cfg.max_probe_nodes - len(pick)
        if short > 0 and remaining:
            pick += list(rng.choice(remaining, size=min(short, len(remaining)),
                                    replace=False))
        pick = pick[: cfg.max_probe_nodes]
        meta = meta.drop(columns="_stratum")
    meta["probed"] = meta.node.isin(set(pick))
    return list(pick), meta


# ==========================================================================
# delta calibration
# ==========================================================================

def calibrate_delta(wn, cfg: DependencyConfig, lam: float = 1.0) -> dict:
    """Pick the perturbation size by measurement, balancing two opposing errors.

    A finite-difference derivative sits between two error sources that move in
    opposite directions:

    * **too small** -- the response is dominated by EPANET's convergence
      tolerance, so the "derivative" is numerical noise;
    * **too large** -- second-order curvature is no longer negligible and the
      difference stops approximating a derivative.

    Central differencing measures both at once. For a step ``d`` it yields the
    first-order term ``(p(+d) - p(-d)) / 2`` and the second-order term
    ``(p(+d) + p(-d) - 2 p(0)) / 2``. Their ratio

        r(d) = max|second order| / max|first order|

    is U-shaped in ``d``: noise inflates it at small steps (noise does not
    shrink with ``d`` while the signal does), curvature inflates it at large
    steps (true curvature grows as ``d^2``). The step minimising ``r`` is the
    best-conditioned choice, and the whole curve is returned so the decision is
    auditable rather than asserted.

    An initial calibration on this corpus showed the naive assumption
    ("smaller is more linear") to be exactly backwards, which is why the
    criterion is measured rather than assumed.
    """
    w = prepare_steady(wn, cfg, lam)
    cons = consumer_nodes(w)
    if not cons:
        return {"delta_m3s": np.nan, "evidence": pd.DataFrame(),
                "reason": "no consumers"}
    mean_base = float(np.mean([base_demand_of(w, n) for n in cons]))
    if mean_base <= 0:
        return {"delta_m3s": np.nan, "evidence": pd.DataFrame(),
                "reason": "zero mean demand"}

    rng = np.random.default_rng(cfg.seed)
    probes = list(rng.choice(cons, size=min(cfg.calibration_probes, len(cons)),
                             replace=False))
    p0 = solve(w).node["pressure"].iloc[0]

    rows = []
    for frac in cfg.calibration_fractions:
        d = frac * mean_base
        for node in probes:
            old = _set_extra_demand(w, node, d)
            p_plus = solve(w).node["pressure"].iloc[0]
            _restore_demand(w, node, old)
            _set_extra_demand(w, node, -d)
            p_minus = solve(w).node["pressure"].iloc[0]
            _restore_demand(w, node, old)

            first = (p_plus - p_minus).to_numpy() / 2.0
            second = (p_plus + p_minus - 2 * p0).to_numpy() / 2.0
            f1 = float(np.nanmax(np.abs(first)))
            f2 = float(np.nanmax(np.abs(second)))
            rows.append({"fraction": frac, "delta_m3s": d, "node": node,
                         "first_order_max_m": f1, "second_order_max_m": f2,
                         "ratio": f2 / f1 if f1 > 0 else np.nan})
    ev = pd.DataFrame(rows)
    curve = (ev.groupby(["fraction", "delta_m3s"], as_index=False)
               .agg(mean_ratio=("ratio", "mean"),
                    mean_first_order_m=("first_order_max_m", "mean"),
                    mean_second_order_m=("second_order_max_m", "mean")))

    ok = curve.dropna(subset=["mean_ratio"])
    if ok.empty:
        frac = cfg.calibration_fractions[len(cfg.calibration_fractions) // 2]
        chosen = curve.iloc[0]
        reason = "no usable response at any step; fell back to the middle step"
        delta = frac * mean_base
    else:
        best = ok.loc[ok.mean_ratio.idxmin()]
        # step down while the ratio stays within tolerance of the minimum,
        # preferring the smallest well-conditioned step
        tol = cfg.calibration_ratio_tolerance
        eligible = ok[ok.mean_ratio <= best.mean_ratio * (1.0 + tol)]
        chosen = eligible.loc[eligible.delta_m3s.idxmin()]
        delta = float(chosen.delta_m3s)
        frac = float(chosen.fraction)
        reason = ("smallest step whose second/first-order ratio is within "
                  f"{tol:.0%} of the minimum ({best.mean_ratio:.2e} at "
                  f"fraction {best.fraction:g})")

    at_edge = (not ok.empty) and frac in (min(cfg.calibration_fractions),
                                          max(cfg.calibration_fractions))
    warn = ("chosen step sits at the edge of the candidate grid -- widen "
            "calibration_fractions, the optimum may lie outside it"
            if at_edge else "")
    return {"delta_m3s": delta, "fraction": frac, "at_grid_edge": at_edge,
            "warning": warn,
            "mean_base_demand_m3s": mean_base,
            "chosen_ratio": float(chosen.mean_ratio)
            if np.isfinite(chosen.get("mean_ratio", np.nan)) else np.nan,
            "chosen_first_order_m": float(chosen.get("mean_first_order_m", np.nan)),
            "evidence": ev, "curve": curve, "reason": reason}


# ==========================================================================
# instantaneous (steady-state) influence map
# ==========================================================================

def instantaneous_map(wn, cfg: DependencyConfig | None = None,
                      lam: float = 1.0, probes: list[str] | None = None,
                      verbose: bool = False) -> dict:
    """S[i, j] = change in pressure at i per unit demand added at j.

    Units: metres per m3/s when ``perturbation_mode == "absolute"`` (a genuine
    Jacobian, comparable across nodes and testable for reciprocity); metres per
    unit *relative* demand change when ``"relative"`` (realistic influence, but
    confounded with node size).

    With ``central_difference`` the +d and -d responses give the antisymmetric
    part (the derivative) and the symmetric part (second-order curvature plus
    solver noise), the latter used as the significance floor.
    """
    cfg = (cfg or DependencyConfig()).validate()
    w = prepare_steady(wn, cfg, lam)
    cons = consumer_nodes(w)
    if probes is None:
        probes, probe_meta = select_probes(w, cfg)
    else:
        probe_meta = pd.DataFrame({"node": probes, "probed": True})

    delta_info = {}
    if cfg.perturbation_mode == "absolute":
        if cfg.delta_abs_lps is not None:
            delta = cfg.delta_abs_lps / LPS_PER_M3S
            delta_info = {"delta_m3s": delta, "reason": "given by config"}
        elif cfg.calibrate:
            delta_info = calibrate_delta(w, cfg, lam)
            delta = delta_info["delta_m3s"]
        else:
            raise ValueError("absolute mode needs delta_abs_lps or calibrate=True")

    base = solve(w)
    p0 = base.node["pressure"].iloc[0]
    q0 = base.link["flowrate"].iloc[0] if cfg.record_flow else None
    all_nodes = list(p0.index)

    S = pd.DataFrame(0.0, index=all_nodes, columns=probes, dtype=float)
    curv = pd.DataFrame(np.nan, index=all_nodes, columns=probes, dtype=float)
    Sq = (pd.DataFrame(0.0, index=list(q0.index), columns=probes, dtype=float)
          if cfg.record_flow else None)
    t0 = time.time()
    for k, j in enumerate(probes):
        d = delta if cfg.perturbation_mode == "absolute" else \
            cfg.delta_rel * base_demand_of(w, j)
        if d <= 0:
            continue
        old = _set_extra_demand(w, j, d)
        r_plus = solve(w)
        p_plus = r_plus.node["pressure"].iloc[0]
        if cfg.record_flow:
            Sq[j] = ((r_plus.link["flowrate"].iloc[0] - q0) / d).to_numpy()
        if cfg.central_difference:
            _restore_demand(w, j, old)
            _set_extra_demand(w, j, -d)
            p_minus = solve(w).node["pressure"].iloc[0]
            deriv = (p_plus - p_minus) / (2 * d)
            second = (p_plus + p_minus - 2 * p0) / 2.0
            curv[j] = second.to_numpy()
        else:
            deriv = (p_plus - p0) / d
        _restore_demand(w, j, old)
        if cfg.perturbation_mode == "relative":
            deriv = deriv * base_demand_of(w, j)   # per unit relative change
        S[j] = deriv.to_numpy()
        if verbose and (k + 1) % 100 == 0:
            print(f"    {k + 1}/{len(probes)} probes, {time.time() - t0:.0f}s")

    measured_floor = (float(np.nanquantile(np.abs(curv.to_numpy()), 0.99))
                      if cfg.central_difference and np.isfinite(curv.to_numpy()).any()
                      else np.nan)
    analytic_floor = quantization_floor(p0)
    # the floor lives in the same units as S, so divide the pressure-space floor
    # by the step actually used
    step = delta if cfg.perturbation_mode == "absolute" else np.nan
    floor_in_S_units = (max(measured_floor, analytic_floor) / step
                        if np.isfinite(step) and step > 0 else np.nan)

    return {
        "S": S,                              # rows = responders, cols = probes
        "S_flow": Sq,
        "curvature": curv,
        "measured_curvature_floor_m": measured_floor,
        "quantization_floor_m": analytic_floor,
        "baseline_pressure": p0,
        "consumers": cons,
        "probes": probes,
        "probe_meta": probe_meta,
        "delta_info": delta_info,
        "noise_floor": floor_in_S_units,
        "lambda": lam,
        "config": asdict(cfg),
        "seconds": round(time.time() - t0, 2),
    }


# ==========================================================================
# diagnostics on the instantaneous map
# ==========================================================================

def symmetry_diagnostics(res: dict, wn=None) -> dict:
    """Test reciprocity, and use its failure as a measurement.

    A pipes-only network has a symmetric sensitivity matrix. Departures are
    either numerical (expect ~1e-3 relative) or physical: a PRV, PSV, FCV or
    check valve regulates one direction only and genuinely decouples. So the
    asymmetric part is a *map of directional elements*, and the per-node
    asymmetry score points at which part of the network they govern.
    """
    S = res["S"]
    common = [n for n in S.columns if n in S.index]
    if len(common) < 3:
        return {"n_common": len(common), "relative_asymmetry": np.nan,
                "per_node": pd.DataFrame()}
    M = S.loc[common, common].to_numpy()
    A = 0.5 * (M - M.T)                       # antisymmetric part
    scale = float(np.nanmax(np.abs(M))) or 1.0
    per_node = pd.DataFrame({
        "node": common,
        "asymmetry_score": np.nanmax(np.abs(A), axis=1) / scale,
    }).sort_values("asymmetry_score", ascending=False)

    out = {
        "n_common": len(common),
        "relative_asymmetry": float(np.nanmax(np.abs(A)) / scale),
        "median_relative_asymmetry": float(np.nanmedian(np.abs(A)) / scale),
        "frobenius_asymmetry": float(np.linalg.norm(A) / max(np.linalg.norm(M), 1e-12)),
        "per_node": per_node,
    }
    if wn is not None:
        n_dir = len(wn.valve_name_list) + sum(
            1 for p in wn.pipe_name_list
            if bool(getattr(wn.get_link(p), "check_valve", False)))
        out["n_directional_elements"] = n_dir
        out["interpretation"] = (
            "consistent with numerical error only"
            if out["relative_asymmetry"] < 0.01 else
            f"asymmetry beyond numerical error with {n_dir} directional "
            f"elements present -- valve/check-valve mediated direction")
    return out


def sign_diagnostics(res: dict) -> dict:
    """Adding demand must not raise pressure anywhere. Violations are diagnostic.

    Monotonicity can genuinely fail where an active PRV or a POWER pump changes
    operating regime, so violations are counted and located rather than treated
    as a hard error.
    """
    S = res["S"].to_numpy()
    finite = np.isfinite(S)
    pos = (S > 0) & finite
    scale = float(np.nanmax(np.abs(S))) or 1.0
    strong = (S > 0.01 * scale) & finite
    return {"n_entries": int(finite.sum()),
            "n_positive": int(pos.sum()),
            "frac_positive": float(pos.sum() / max(finite.sum(), 1)),
            "n_positive_strong": int(strong.sum()),
            "max_positive": float(np.nanmax(S)) if finite.any() else np.nan}


def zone_structure(res: dict, cfg: DependencyConfig | None = None) -> dict:
    """Structural zeros: hydraulic separation, not weak coupling.

    An exact zero response means the probe cannot reach the responder at all --
    a distinct pressure zone, typically created by a closed link or a PRV. These
    are reported apart from small-but-nonzero couplings because they mean
    something categorically different for districting.
    """
    S = res["S"]
    A = np.abs(S.to_numpy())
    exact_zero = A == 0.0
    floor = res.get("noise_floor", np.nan)
    factor = (cfg or DependencyConfig()).floor_safety_factor
    thr = factor * floor if np.isfinite(floor) else np.nan
    affected = A > thr if np.isfinite(thr) else A > 0
    return {
        "frac_exact_zero": float(exact_zero.mean()),
        "threshold": thr,
        "reach_per_probe": pd.Series(affected.sum(axis=0), index=S.columns,
                                     name="n_nodes_affected"),
        "sensitivity_per_responder": pd.Series(affected.sum(axis=1), index=S.index,
                                               name="n_probes_felt"),
        "affected_mask": pd.DataFrame(affected, index=S.index, columns=S.columns),
    }


def influence_vs_distance(res: dict, wn) -> pd.DataFrame:
    """Does influence decay with hydraulic distance? A physical plausibility test.

    Distance is shortest path weighted by a resistance proxy (length / diameter^5,
    the Hazen-Williams scaling of head loss per unit flow). A map that shows no
    decay is almost certainly wrong.
    """
    g = nx.Graph()
    for lname in wn.link_name_list:
        l = wn.get_link(lname)
        length = float(getattr(l, "length", 0.0) or 0.0)
        diam = float(getattr(l, "diameter", 0.0) or 0.0)
        r = length / (diam ** 5) if (length > 0 and diam > 0) else 0.0
        a, b = l.start_node_name, l.end_node_name
        if g.has_edge(a, b):
            g[a][b]["r"] = min(g[a][b]["r"], r)
        else:
            g.add_edge(a, b, r=r)

    S = res["S"]
    probes = [p for p in S.columns if g.has_node(p)]
    rows = []
    for j in probes:
        dist = nx.single_source_dijkstra_path_length(g, j, weight="r")
        for i, d in dist.items():
            if i in S.index and i != j:
                v = S.at[i, j]
                if np.isfinite(v) and v != 0.0:
                    rows.append({"probe": j, "responder": i,
                                 "resistance_distance": d, "influence": abs(v)})
    df = pd.DataFrame(rows)
    if len(df) > 2:
        x = np.log10(df.resistance_distance.replace(0, np.nan))
        y = np.log10(df.influence.replace(0, np.nan))
        ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        df.attrs["log_log_corr"] = float(np.corrcoef(x[ok], y[ok])[0, 1]) \
            if ok.sum() > 2 else np.nan
    return df


# ==========================================================================
# dynamic (land-use shape) map
# ==========================================================================

def normalized_shape(values, name: str = "") -> np.ndarray:
    """Scale a diurnal shape to mean 1 so it carries no volume information.

    This is the control that makes a shape perturbation a shape perturbation.
    Without it, swapping shapes also changes the daily volume and the resulting
    map conflates "different rhythm" with "more water".
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"shape {name!r} must be a non-empty 1-D sequence")
    if np.any(v < 0):
        raise ValueError(f"shape {name!r} has negative multipliers")
    m = v.mean()
    if m <= 0:
        raise ValueError(f"shape {name!r} has zero mean")
    return v / m


def dynamic_map(wn, shape_base, shape_probe, cfg: DependencyConfig | None = None,
                lam: float = 1.0, probes: list[str] | None = None,
                verbose: bool = False) -> dict:
    """Response magnitude *and lag* from a volume-neutral shape swap.

    Every consumer runs ``shape_base``; the probed node runs ``shape_probe``.
    Both are mean-normalized, so total daily volume is unchanged and only the
    rhythm differs. The lag of the peak cross-correlation between the response
    and the injected demand deviation is the physical propagation lag.

    Expect lag 0 everywhere on a tankless, controlless network -- that is not a
    failure of the method, it is the quasi-steady assumption showing through.
    Non-zero lag can only arise via tank mass or control switching.
    """
    cfg = (cfg or DependencyConfig()).validate()
    import copy

    a = normalized_shape(shape_base, "shape_base")
    b = normalized_shape(shape_probe, "shape_probe")
    if a.size != b.size:
        raise ValueError("shape_base and shape_probe must have equal length")

    w = prepare_steady(wn, cfg, lam)          # folds multiplier, pins options
    w = copy.deepcopy(w)
    w.options.time.duration = int(cfg.dynamic_hours * 3600)
    w.options.time.hydraulic_timestep = int(cfg.dynamic_timestep_s)
    w.options.time.report_timestep = int(cfg.dynamic_timestep_s)
    w.options.time.pattern_timestep = int(cfg.dynamic_timestep_s)

    pat_a, pat_b = "_dep_base", "_dep_probe"
    for nm, vals in ((pat_a, a), (pat_b, b)):
        if nm in w.pattern_name_list:
            w.remove_pattern(nm)
        w.add_pattern(nm, list(vals))

    cons = consumer_nodes(w)
    for n in cons:                            # everyone on the base shape
        for ts in w.get_node(n).demand_timeseries_list:
            ts.pattern_name = pat_a

    if probes is None:
        probes, probe_meta = select_probes(w, cfg)
    else:
        probe_meta = pd.DataFrame({"node": probes, "probed": True})

    warm = int(cfg.dynamic_warmup_h * 3600 // cfg.dynamic_timestep_s)
    base_res = solve(w)
    p_base = base_res.node["pressure"].iloc[warm:]
    tanks = list(w.tank_name_list)

    mag, rms, lag = {}, {}, {}
    t0 = time.time()
    step_h = cfg.dynamic_timestep_s / 3600.0
    for k, j in enumerate(probes):
        tss = w.get_node(j).demand_timeseries_list
        prev = [ts.pattern_name for ts in tss]
        for ts in tss:
            ts.pattern_name = pat_b
        p_new = solve(w).node["pressure"].iloc[warm:]
        for ts, pv in zip(tss, prev):
            ts.pattern_name = pv

        dp = (p_new - p_base)
        mag[j] = dp.abs().max()
        rms[j] = np.sqrt((dp ** 2).mean())

        # injected demand deviation at the probe, on the same clock
        n_steps = dp.shape[0]
        idx = (np.arange(warm, warm + n_steps) * step_h).astype(int) % b.size
        drive = (b[idx] - a[idx]) * base_demand_of(w, j)
        lag[j] = _lag_of_peak_xcorr(dp, drive, cfg.max_lag_h, step_h)
        if verbose and (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(probes)} probes, {time.time() - t0:.0f}s")

    return {
        "magnitude": pd.DataFrame(mag),        # rows responders, cols probes
        "rms": pd.DataFrame(rms),
        "lag_h": pd.DataFrame(lag),
        "baseline_pressure": p_base,
        "tank_levels": (base_res.node["head"][tanks].iloc[warm:]
                        if tanks else pd.DataFrame()),
        "probes": probes,
        "probe_meta": probe_meta,
        "shape_base": a,
        "shape_probe": b,
        "has_lag_mechanism": bool(tanks) or len(w.control_name_list) > 0,
        "n_tanks": len(tanks),
        "n_controls": len(w.control_name_list),
        "lambda": lam,
        "config": asdict(cfg),
        "seconds": round(time.time() - t0, 2),
    }


def _lag_of_peak_xcorr(response: pd.DataFrame, drive: np.ndarray,
                       max_lag_h: float, step_h: float) -> pd.Series:
    """Lag (hours) maximising |corr(response_i(t), drive(t - lag))| per responder."""
    max_k = int(max_lag_h / step_h)
    d = np.asarray(drive, dtype=float)
    d = d - d.mean()
    out = {}
    R = response.to_numpy()
    for c, name in enumerate(response.columns):
        y = R[:, c].astype(float)
        y = y - np.nanmean(y)
        if not np.isfinite(y).all() or np.allclose(y, 0) or np.allclose(d, 0):
            out[name] = np.nan
            continue
        best_k, best = 0, -np.inf
        for k in range(0, max_k + 1):
            a = d[: len(d) - k] if k else d
            bvec = y[k:] if k else y
            if len(a) < 3:
                break
            sa, sb = a.std(), bvec.std()
            if sa <= 0 or sb <= 0:
                continue
            c_ = abs(float(np.mean(a * bvec) / (sa * sb)))
            if c_ > best:
                best, best_k = c_, k
        out[name] = best_k * step_h if np.isfinite(best) else np.nan
    return pd.Series(out)


# ==========================================================================
# from influence to federated districts
# ==========================================================================

def symmetrize(S: pd.DataFrame) -> pd.DataFrame:
    """|S| restricted to probed-and-responding nodes, symmetrized.

    Symmetrizing is honest here rather than lossy: the instantaneous map is
    theoretically symmetric, so the average of the two directions is the better
    estimate of a single physical coupling.
    """
    common = [n for n in S.columns if n in S.index]
    M = np.array(S.loc[common, common].to_numpy(), copy=True)
    W = 0.5 * (np.abs(M) + np.abs(M).T)
    np.fill_diagonal(W, 0.0)
    return pd.DataFrame(W, index=common, columns=common)


def discover_districts(S: pd.DataFrame, k: int, seed: int = 0,
                       normalize: str = "log") -> pd.Series:
    """Spectral clustering on the coupling matrix -> candidate districts.

    Influence spans orders of magnitude, so a log transform is applied by
    default; without it a handful of near-source nodes dominate the spectrum and
    the partition degenerates into one big cluster plus singletons.

    These are *candidate* districts from hydraulic coupling alone. A real
    partition also has to respect operational boundaries, and it should be
    compared against, not substituted for, whatever partition the study adopts.
    """
    from sklearn.cluster import SpectralClustering

    W = symmetrize(S)
    A = W.to_numpy().copy()
    if normalize == "log":
        pos = A[A > 0]
        floor = pos.min() if pos.size else 1.0
        A = np.log10(1.0 + A / floor)
    elif normalize == "rank":
        flat = A.flatten()
        order = flat.argsort().argsort().astype(float)
        A = (order / order.max()).reshape(A.shape)
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)

    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            random_state=seed, assign_labels="kmeans")
    labels = sc.fit_predict(A)
    return pd.Series(labels, index=W.index, name="district")


def district_coupling(S: pd.DataFrame, partition: pd.Series) -> dict:
    """Quantify inter-district dependence -- the federated coupling dial.

    ``coupling_matrix[A, B]`` is the mean absolute pressure sensitivity of
    district A's nodes to demand in district B. ``coupling_ratio`` is
    off-diagonal mass over total mass: 0 means fully separable clients (a
    federated problem with no cross-client information), 1 means district labels
    carry no hydraulic meaning at all.

    This is what makes "how dependent are the clients?" a measured quantity
    rather than a modelling assumption.
    """
    W = symmetrize(S)
    lab = partition.reindex(W.index).dropna()
    W = W.loc[lab.index, lab.index]
    groups = sorted(lab.unique())
    C = pd.DataFrame(0.0, index=groups, columns=groups, dtype=float)
    for a in groups:
        ia = lab[lab == a].index
        for b in groups:
            ib = lab[lab == b].index
            block = W.loc[ia, ib].to_numpy()
            if a == b and block.size > 1:
                m = block[~np.eye(len(ia), dtype=bool)]
                C.loc[a, b] = float(m.mean()) if m.size else 0.0
            else:
                C.loc[a, b] = float(block.mean()) if block.size else 0.0
    tot = float(C.to_numpy().sum())
    off = tot - float(np.trace(C.to_numpy()))
    return {"coupling_matrix": C,
            "coupling_ratio": off / tot if tot > 0 else np.nan,
            "sizes": lab.value_counts().sort_index(),
            "modularity_like": 1.0 - (off / tot) if tot > 0 else np.nan}


def influence_ranking(S: pd.DataFrame, damping: float = 0.85) -> pd.DataFrame:
    """Rank nodes by global influence via a random walk on the coupling graph.

    Two complementary scores: total outgoing influence (direct), and the
    stationary distribution of a damped random walk (indirect, PageRank-style),
    which rewards nodes that reach other *influential* nodes. Useful for sensor
    placement and for choosing which nodes to perturb in a scenario.
    """
    W = symmetrize(S)
    A = W.to_numpy()
    strength = A.sum(axis=1)
    g = nx.from_numpy_array(A)
    mapping = dict(enumerate(W.index))
    g = nx.relabel_nodes(g, mapping)
    try:
        pr = nx.pagerank(g, alpha=damping, weight="weight")
    except Exception:
        pr = {n: np.nan for n in W.index}
    return pd.DataFrame({
        "node": W.index,
        "total_influence": strength,
        "walk_influence": [pr.get(n, np.nan) for n in W.index],
    }).set_index("node").sort_values("walk_influence", ascending=False)


# ==========================================================================
# operating-point comparison
# ==========================================================================

def operating_point_stability(wn, cfg: DependencyConfig | None = None,
                              probes: list[str] | None = None) -> dict:
    """Build the map at several demand multipliers and compare.

    Influence is nonlinear in flow. If the map measured at low demand does not
    predict the map under growth, then any districting or sensor choice derived
    from a single operating point is provisional. This quantifies that rather
    than assuming it away.
    """
    cfg = (cfg or DependencyConfig()).validate()
    maps, rows = {}, []
    ref = None
    for lam in cfg.operating_lambdas:
        m = instantaneous_map(wn, cfg, lam=lam, probes=probes)
        maps[lam] = m
        if probes is None:
            probes = m["probes"]              # reuse the same probes across lambdas
        W = symmetrize(m["S"])
        v = W.to_numpy().flatten()
        if ref is None:
            ref = v
            rows.append({"lambda": lam, "pearson_vs_ref": 1.0,
                         "spearman_vs_ref": 1.0, "scale_vs_ref": 1.0})
        else:
            ok = np.isfinite(v) & np.isfinite(ref)
            from scipy.stats import spearmanr
            rows.append({
                "lambda": lam,
                "pearson_vs_ref": float(np.corrcoef(v[ok], ref[ok])[0, 1]),
                "spearman_vs_ref": float(spearmanr(v[ok], ref[ok]).statistic),
                "scale_vs_ref": float(np.nanmedian(
                    v[ok][ref[ok] > 0] / ref[ok][ref[ok] > 0]))
                if (ref[ok] > 0).any() else np.nan,
            })
    return {"maps": maps, "comparison": pd.DataFrame(rows), "probes": probes}


# ==========================================================================
# one-call assessment
# ==========================================================================

def assess_dependency(source, cfg: DependencyConfig | None = None,
                      shapes: dict | None = None, n_districts: int = 5,
                      run_dynamic: bool = True, verbose: bool = True) -> dict:
    """Full dependency assessment of one network. Never raises; reports instead."""
    cfg = (cfg or DependencyConfig()).validate()
    source = Path(source)
    net = source.stem
    out = {"network": net, "config": asdict(cfg)}
    t0 = time.time()
    try:
        wn = wntr.network.WaterNetworkModel(str(source))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if not consumer_nodes(wn):
        out["error"] = "no consumer junctions (network carries no demand)"
        return out

    inst = instantaneous_map(wn, cfg, lam=cfg.operating_lambdas[0], verbose=verbose)
    out["instantaneous"] = inst
    out["symmetry"] = symmetry_diagnostics(inst, wn)
    out["signs"] = sign_diagnostics(inst)
    out["zones"] = zone_structure(inst, cfg)
    out["decay"] = influence_vs_distance(inst, wn)
    try:
        part = discover_districts(inst["S"], n_districts, seed=cfg.seed)
        out["districts"] = part
        out["district_coupling"] = district_coupling(inst["S"], part)
    except Exception as exc:
        out["districts_error"] = f"{type(exc).__name__}: {exc}"
    out["ranking"] = influence_ranking(inst["S"])

    if run_dynamic and shapes:
        try:
            out["dynamic"] = dynamic_map(
                wn, shapes["base"], shapes["probe"], cfg,
                lam=cfg.operating_lambdas[0],
                probes=inst["probes"], verbose=verbose)
        except Exception as exc:
            out["dynamic_error"] = f"{type(exc).__name__}: {exc}"

    out["seconds"] = round(time.time() - t0, 2)
    return out


def summarize(assessments: list[dict]) -> pd.DataFrame:
    """One row per network: the numbers worth putting in a table."""
    rows = []
    for a in assessments:
        if "error" in a:
            rows.append({"network": a["network"], "status": a["error"]})
            continue
        inst, sym, zones = a["instantaneous"], a["symmetry"], a["zones"]
        dc = a.get("district_coupling", {})
        row = {
            "network": a["network"], "status": "ok",
            "n_consumers": len(inst["consumers"]),
            "n_probes": len(inst["probes"]),
            "delta_lps": round(inst["delta_info"].get("delta_m3s", np.nan)
                               * LPS_PER_M3S, 4)
            if inst["delta_info"] else np.nan,
            "noise_floor": inst.get("noise_floor", np.nan),
            "rel_asymmetry": sym.get("relative_asymmetry", np.nan),
            "frob_asymmetry": sym.get("frobenius_asymmetry", np.nan),
            "n_directional_elements": sym.get("n_directional_elements", np.nan),
            "frac_positive_S": a["signs"]["frac_positive"],
            "frac_exact_zero": zones["frac_exact_zero"],
            "median_reach": float(zones["reach_per_probe"].median()),
            "max_reach": int(zones["reach_per_probe"].max()),
            "decay_loglog_corr": a["decay"].attrs.get("log_log_corr", np.nan),
            "coupling_ratio": dc.get("coupling_ratio", np.nan),
            "seconds": a["seconds"],
        }
        if "dynamic" in a:
            d = a["dynamic"]
            lag = d["lag_h"].to_numpy()
            row |= {
                "has_lag_mechanism": d["has_lag_mechanism"],
                "n_tanks": d["n_tanks"], "n_controls": d["n_controls"],
                "median_lag_h": float(np.nanmedian(lag)),
                "frac_lag_nonzero": float(np.nanmean(lag > 0)),
                "max_shape_response_m": float(np.nanmax(d["magnitude"].to_numpy())),
            }
        rows.append(row)
    return pd.DataFrame(rows)
