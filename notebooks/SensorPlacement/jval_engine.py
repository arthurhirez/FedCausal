"""jval_engine.py -- windowed validation of interventional unmixing.

Implements section 7 of the v2 handoff against the jval_* worlds:

  * per-world gating          (7.7)  three assertions, run before inclusion
  * pre/transition/post       (7.3)  windows verified equal-length and
                                     calendar-aligned
  * M1 structure / M2 level   (7.2)  deliberately orthogonal measurands
  * Jacobian arms             (7.6)  J_pre / J_post / J_pre_wpost / J_perm
                                     (+ optional J_inp), no new simulation
  * free-method baselines     (5)    raw / PC1-removed / robust common mode,
                                     so `gain` is defined the same way as in
                                     the pinned table
  * predictions P1..P7        (7.8)  scored, with cluster-aware inference
                                     rather than n=41 paired tests (5.3)

Depends on `sensitivity_probe` (probe, Jacobian, unmixing) and
`placement_poc` (world discovery, candidates, placement strategies).
Call `api_report()` first: it fails loudly if a signature has drifted
instead of blowing up mid-sweep.

Two deliberate local reimplementations, both recorded in handoff 6.9 / 8.5:
  * `unmix` forwards rcond to pinv; `sensitivity_probe.unmix_demands` does
    not (the parameter is dead there). jval's uniform-weight and decoupled
    arms push cond up, so this matters here.
  * `state_map` remaps regimes per window. The commissioning consumption_map
    is wrong for the drifting district after onset.
"""
from __future__ import annotations

import copy
import inspect
import json
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

import placement_poc as P
import sensitivity_probe as S

# --------------------------------------------------------------------------
# 0. adapter surface -- everything this module borrows, in one place
# --------------------------------------------------------------------------
NEEDED = {
    "sensitivity_probe": ["flat_baseline", "perturbation_sweep", "jacobian",
                          "district_jacobian", "unmix_demands",
                          "unmixing_quality", "node_demands", "_set_demand"],
    "placement_poc": ["discover_worlds", "load_world", "candidate_table",
                      "topology_features", "residual_stats", "strategies",
                      "WORLD_ARTIFACTS"],
}


def api_report(verbose=True) -> pd.DataFrame:
    """Assert the borrowed API exists and print its signatures.

    Cheap insurance: a renamed kwarg upstream otherwise surfaces as a
    TypeError forty minutes into a sweep.
    """
    rows, missing = [], []
    for mod, names in NEEDED.items():
        m = {"sensitivity_probe": S, "placement_poc": P}[mod]
        for n in names:
            obj = getattr(m, n, None)
            if obj is None:
                missing.append(f"{mod}.{n}")
                rows.append(dict(module=mod, name=n, signature="MISSING"))
                continue
            try:
                sig = str(inspect.signature(obj))
            except TypeError:
                sig = f"<{type(obj).__name__}>"
            rows.append(dict(module=mod, name=n, signature=sig))
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
    if missing:
        raise ImportError(f"adapter surface incomplete: {missing}")
    return df


# --------------------------------------------------------------------------
# 1. world discovery and arm assignment
# --------------------------------------------------------------------------
CLONE = "clone/data"
PATHS = dict(
    demand=f"{CLONE}/02_intermediate/demand_series.parquet",
    flows=f"{CLONE}/02_intermediate/flows.parquet",
    pressures=f"{CLONE}/02_intermediate/pressures.parquet",
    schedule=f"{CLONE}/03_primary/gt_drift_schedule.csv",
)


def _arm(w: pd.Series) -> str:
    """Label a world by which jval_* study produced it.

    Rule-based rather than read from experiments.yml, because several cells
    are deliberate cache hits shared between studies (jval_coupling's
    baseline arm and jval_beta's 0.35 cell are literally jval_s1_oddone
    worlds). The label is therefore the world's *identity*, and a world can
    legitimately serve more than one study; `arms_for` gives the full set.
    """
    cmap, var = w.get("consumption_map"), w.get("variant")
    beta = float(w.get("beta", np.nan))
    mnpm = w.get("max_neighbors_per_month")
    if cmap == "LR_LR_LR_LR_HR":
        return "income"
    if cmap == "LR_LR_LR_LI_LI":
        return "s2_split"
    if cmap == "LR_LR_LR_LR_LI":
        if var == "isolated":
            return "coupling_isolated"
        if pd.notna(beta) and not np.isclose(beta, 0.35):
            return "beta"
        if pd.notna(mnpm) and int(mnpm) != 2:
            return "s1_equalised"
        return "s1_oddone"
    return "other"


def arms_for(w: pd.Series) -> list[str]:
    """Every study a world participates in (cache hits serve several)."""
    a = [_arm(w)]
    if a[0] == "s1_oddone":
        if int(w.get("sim_seed", -1)) == 42:
            a.append("beta")                      # the beta=0.35 cell
            if w.get("drift_district") in ("District_A", "District_E"):
                a.append("coupling_baseline")     # the paired baseline
    return a


def jval_index(proj, n_months=48, experiments_root=None) -> pd.DataFrame:
    """Discover worlds, keep the jval horizon, label the arms."""
    kw = {} if experiments_root is None else dict(experiments_root=experiments_root)
    idx = P.discover_worlds(proj, **kw)
    if not len(idx):
        raise RuntimeError("discover_worlds returned nothing")
    if n_months is not None:
        if "n_months" not in idx.columns:
            raise KeyError("manifest 'world' block has no n_months -- pass "
                           f"n_months=None. Columns: {sorted(idx.columns)}")
        idx = idx[idx["n_months"] == n_months].copy()
    idx["arm"] = idx.apply(_arm, axis=1)
    idx["arms"] = idx.apply(lambda r: ",".join(arms_for(r)), axis=1)
    return idx.sort_values(["arm", "consumption_map", "drift_district",
                            "sim_seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. per-world gating (handoff 7.7)
# --------------------------------------------------------------------------
def read_schedule(wdir) -> pd.DataFrame:
    return pd.read_csv(pathlib.Path(wdir) / PATHS["schedule"])


def gate_world(w: pd.Series, districts: dict, post_start=36,
               ramp_months=1.4, pre_start=6) -> dict:
    """The three assertions that would have caught the resolution_probe
    truncation the day it happened.

    Returned as a row rather than raised, so a failure is *recorded* and
    excluded rather than aborting a sweep.
    """
    tgt = w["drift_district"]
    n_nodes = len(districts["districts"][tgt])
    out = dict(sim_hash=w["sim_hash"], arm=w.get("arm"), tgt=tgt,
               n_nodes=n_nodes)
    try:
        sch = read_schedule(w["dir"])
    except Exception as e:                                    # noqa: BLE001
        return {**out, "ok": False, "why": f"schedule unreadable: {e}"}
    mcol = "drift_month" if "drift_month" in sch.columns else None
    if mcol is None:
        return {**out, "ok": False, "why": f"no drift_month in {list(sch.columns)}"}
    n_conv, m0, m1 = len(sch), float(sch[mcol].min()), float(sch[mcol].max())
    c = dict(front_complete=n_conv == n_nodes,
             clears_post=m1 + ramp_months < post_start,
             pre_clean=m0 >= pre_start)
    out.update(n_converted=n_conv, onset=m0, last=m1, span=m1 - m0, **c)
    out["ok"] = all(c.values())
    out["why"] = "" if out["ok"] else ";".join(k for k, v in c.items() if not v)
    return out


def gate_table(idx: pd.DataFrame, districts: dict, **kw) -> pd.DataFrame:
    rows = [gate_world(w, districts, **kw) for _, w in idx.iterrows()]
    return pd.DataFrame(rows)


def pin(idx: pd.DataFrame, gates: pd.DataFrame, path=None) -> pd.DataFrame:
    """Freeze the analysed world set to an explicit sim_hash list (6.9).

    Every table in the thesis should be reconstructible from this file;
    `.head(n)` on a growing pool already cost this project one headline.
    """
    keep = set(gates.loc[gates["ok"], "sim_hash"])
    out = idx[idx["sim_hash"].isin(keep) & idx["usable"]].copy()
    if path:
        cols = [c for c in ("sim_hash", "arm", "arms", "variant", "sim_seed",
                            "beta", "consumption_map", "drift_district",
                            "drift_to_income", "drift_to_land_use",
                            "n_months") if c in out.columns]
        out[cols].to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------
# 3. windows (handoff 7.3)
# --------------------------------------------------------------------------
WINDOWS = dict(pre=(0, 6), transition=(6, 36), post=(36, 42))


def month_vector(art: dict, dem: pd.DataFrame, n: int) -> np.ndarray:
    """Month label per step, preferring the simulator's own column."""
    if "month" in dem.columns and len(dem) >= n:
        return dem["month"].to_numpy()[:n]
    spm = int(art.get("steps_day", 24)) * 30
    return np.arange(n) // spm


def window_masks(months: np.ndarray, windows=None) -> dict:
    windows = windows or WINDOWS
    return {k: (months >= a) & (months < b) for k, (a, b) in windows.items()}


def window_report(masks: dict) -> dict:
    """pre and post must have identical step counts, else the two
    correlation estimators do not have identical variance (7.3)."""
    n = {k: int(v.sum()) for k, v in masks.items()}
    return dict(**n, pre_eq_post=n.get("pre") == n.get("post"))


# --------------------------------------------------------------------------
# 4. truth, estimates, regime labels
# --------------------------------------------------------------------------
PAIRKEY = lambda a, b: (a, b) if a < b else (b, a)          # noqa: E731


def district_demand(dem: pd.DataFrame, districts: dict) -> pd.DataFrame:
    """(time x district) aggregate nodal demand -- the ground truth."""
    n2d = {n: d for d, nn in districts["districts"].items() for n in nn}
    ds = sorted(districts["districts"])
    return pd.DataFrame({d: dem[[c for c in dem.columns
                                 if c != "month" and n2d.get(c) == d]]
                        .sum(axis=1) for d in ds})


def corr_pairs(F: pd.DataFrame) -> dict:
    ds = list(F.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {PAIRKEY(a, b): float(np.corrcoef(F[a], F[b])[0, 1])
                for i, a in enumerate(ds) for b in ds[i + 1:]}


def state_map(w: pd.Series, districts: dict, window: str,
              axis="land_use") -> dict:
    """Regime token per district, remapped for the window.

    The commissioning `consumption_map` is WRONG for the drifting district
    after onset -- the `_regime_map` bug. `post` therefore substitutes the
    drift target; `transition` is genuinely mixed and is labelled with the
    pre state and must not carry a gap claim.
    """
    ds = sorted(districts["districts"])
    toks = w["consumption_map"].split("_")
    pos = 0 if axis == "income" else 1
    m = {d: t[pos] for d, t in zip(ds, toks)}
    if window == "post":
        key = "drift_to_income" if axis == "income" else "drift_to_land_use"
        m[w["drift_district"]] = str(w[key])[0].upper()
    return m


def gap(rho: dict, states: dict) -> float:
    """G = mean rho(same-state pairs) - mean rho(different-state pairs)."""
    same = [v for (a, b), v in rho.items() if states[a] == states[b]]
    diff = [v for (a, b), v in rho.items() if states[a] != states[b]]
    if not same or not diff:
        return np.nan
    return float(np.mean(same) - np.mean(diff))


def truth_audit(worlds: pd.DataFrame, districts: dict, windows=None,
                axis="land_use", progress=10) -> pd.DataFrame:
    """Pair-level ground truth per window. Demand only -- no hydraulics.

    Run this BEFORE scoring any estimator. The invariance null and the
    signed prediction are asserted "by construction", but the construction
    is a claim about the simulator, and the demand series carries seasonal
    and stochastic variation on top of it. If rho_true(A,B) moves pre->post
    when neither A nor B drifted, then P3 is not a clean null and the
    threshold has to come from here rather than from 0.02.
    """
    rows = []
    for i, (_, w) in enumerate(worlds.iterrows(), 1):
        dem = _load(w["dir"], "demand")
        n = len(dem)
        months = month_vector({"steps_day": 24}, dem, n)
        masks = window_masks(months, windows)
        D = district_demand(dem, districts)
        for win, m in masks.items():
            if m.sum() < 48:
                continue
            st = state_map(w, districts, win, axis=axis)
            for k, v in corr_pairs(D.loc[np.flatnonzero(m)]).items():
                rows.append(dict(sim_hash=w["sim_hash"], arm_study=w.get("arm"),
                                 drift_district=w["drift_district"],
                                 placement="truth", method="truth",
                                 window=win, a=k[0], b=k[1], rho_true=v,
                                 rho_hat=v, state_a=st[k[0]],
                                 state_b=st[k[1]]))
        if progress and i % progress == 0:
            print(f"  {i}/{len(worlds)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. the two measurands (handoff 7.2)
# --------------------------------------------------------------------------
def M1(D_true: pd.DataFrame, D_hat: pd.DataFrame, states: dict) -> dict:
    """Structure: scale-blind, shape-sensitive. Reads land use.

    MAE is primary. With four identical districts 6 of 10 true correlations
    are tied at ~1, so a Spearman over 10 points is near-uninformative on
    S1 -- reported, never load-bearing.
    """
    t, h = corr_pairs(D_true), corr_pairs(D_hat)
    keys = sorted(t)
    tv, hv = [t[k] for k in keys], [h[k] for k in keys]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = sps.spearmanr(hv, tv).statistic
    g_t, g_h = gap(t, states), gap(h, states)
    n_ties = int(np.sum(np.abs(np.subtract.outer(tv, tv)) < 1e-3) - len(tv)) // 2
    return dict(rank_corr=float(rc), mae=float(np.mean(np.abs(
        np.array(hv) - np.array(tv)))),
        max_err=float(np.max(np.abs(np.array(hv) - np.array(tv)))),
        G_true=g_t, G_hat=g_h,
        gap_retention=(g_h / g_t if g_t not in (0, np.nan) and
                       np.isfinite(g_t) and abs(g_t) > 1e-9 else np.nan),
        n_true_ties=n_ties)


def M2(D_true: pd.DataFrame, D_hat: pd.DataFrame) -> dict:
    """Level: shape-blind, scale-sensitive. Reads income.

    Jd carries a global scale factor, so mean(r) is UNIDENTIFIABLE and
    diagnostic only. std(r) and the amplitude rank are the reportable
    quantities.
    """
    ds = list(D_true.columns)
    st, sh = D_true[ds].std().to_numpy(), D_hat[ds].std().to_numpy()
    ok = (st > 0) & (sh > 0)
    r = np.full(len(ds), np.nan)
    r[ok] = np.log(sh[ok] / st[ok])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = sps.spearmanr(sh[ok], st[ok]).statistic if ok.sum() > 2 else np.nan
    return dict(std_r=float(np.nanstd(r)), mean_r=float(np.nanmean(r)),
                amp_rank_corr=float(rho),
                r_per_district={d: (None if np.isnan(v) else float(v))
                                for d, v in zip(ds, r)},
                amp_hat={d: float(v) for d, v in zip(ds, sh)},
                amp_true={d: float(v) for d, v in zip(ds, st)})


# --------------------------------------------------------------------------
# 6. Jacobian arms (handoff 7.6) -- no new simulation
# --------------------------------------------------------------------------
L_PER_M3 = 1000.0

# The arms that are anchored to a WORLD operating point. These are the ones
# section 7.6 actually needs, and none of them touches the .inp base demands.
ARMS_DEFAULT = ("J_pre", "J_post", "J_pre_wpost", "J_perm")

# `J_inp` is the old `J_flat`: probe the network at its own base demands.
# It is OPT-IN, because on the world networks (`wn_variant`, multiplier
# pinned to 1.0) those bases sit ~5x above the demand the worlds actually
# run at, and every solve comes back infeasible. See `probe_diag`.
ARMS_ALL = ("J_inp",) + ARMS_DEFAULT


class ProbeInfeasible(RuntimeError):
    """The perturbation sweep produced no usable rows.

    Raised rather than returned, because `perturbation_sweep` drops
    infeasible solves SILENTLY: an overloaded operating point yields an
    empty sweep, an empty Jacobian, and then a zero-size reduction inside
    `unmixing_quality`, which is a long way from the cause.
    """


def probe_at(wn, op_m3s: pd.Series | None = None, delta_frac=0.5,
             quantity="flowrate", demand_scale=1.0, label="",
             require_feasible=True) -> dict:
    """Flat-baseline perturbation sweep at a chosen operating point.

    `op_m3s` overrides each junction's constant demand before probing --
    that is what "flat probe at pre-drift mean demand" means. delta is
    `delta_frac` x mean nodal demand, the reciprocity-calibrated step
    (handoff 4); the mean is over ALL junctions including zero-demand ones,
    which is what reproduces the 0.45%-of-total figure.

    Three things are checked before returning, because each has already
    failed once on this project:
      * the demand overrides actually applied (a silent failure would leave
        the probe at the .inp point while the diag table claims otherwise);
      * the flat baseline itself is feasible;
      * the sweep is non-empty.
    """
    flat = S.flat_baseline(wn, demand_multiplier=1.0)
    if op_m3s is not None:
        bad = []
        for n, v in op_m3s.items():
            try:
                S._set_demand(flat, str(n), float(v) * demand_scale)
            except Exception as e:                            # noqa: BLE001
                bad.append((str(n), type(e).__name__))
        if len(bad) > 0.05 * max(len(op_m3s), 1):
            raise ProbeInfeasible(
                f"{label or 'probe'}: {len(bad)}/{len(op_m3s)} demand "
                f"overrides failed (e.g. {bad[:3]}) -- the operating point "
                "is NOT the one the diag table would report")
    elif demand_scale != 1.0:
        d = S.node_demands(flat)
        for n, v in d.items():
            S._set_demand(flat, str(n), float(v) * demand_scale)

    d0 = S.node_demands(flat)
    arr = np.asarray(d0, dtype=float)
    mean_d = float(arr.mean())
    delta = delta_frac * mean_d
    sweep, meta = S.perturbation_sweep(flat, [delta])
    out = dict(sweep=sweep, delta=delta, mean_d=mean_d, d0=d0,
               total=float(arr.sum()), meta=meta, label=label)
    out.update(_meta_summary(meta))
    out["sweep_rows"] = int(len(sweep))
    out["ok"] = bool(len(sweep))
    if not out["ok"] and require_feasible:
        raise ProbeInfeasible(
            f"{label or 'probe'}: sweep is EMPTY -- every solve was dropped "
            f"as infeasible. total demand {out['total']:.3f} m3/s, "
            f"delta {delta:.5g}, baseline min_pressure "
            f"{out.get('baseline_min_pressure')}. The operating point is "
            "outside the network's feasible envelope; probe at a world "
            "operating point (J_pre/J_post) or pass demand_scale.")
    return out


def _meta_summary(meta) -> dict:
    """Pull feasibility out of whatever shape `perturbation_sweep` returns."""
    if not isinstance(meta, pd.DataFrame) or not len(meta):
        return dict(n_solves=np.nan, n_feasible=np.nan,
                    baseline_min_pressure=np.nan, min_pressure=np.nan)
    fea = meta["feasible"] if "feasible" in meta.columns else pd.Series(dtype=bool)
    mp = meta["min_pressure"] if "min_pressure" in meta.columns else pd.Series(dtype=float)
    base = meta[meta.get("kind", pd.Series(index=meta.index)) == "baseline"]
    return dict(n_solves=int(len(meta)),
                n_feasible=int(fea.sum()) if len(fea) else np.nan,
                baseline_min_pressure=(float(base["min_pressure"].iloc[0])
                                       if len(base) and "min_pressure" in base
                                       else np.nan),
                min_pressure=float(mp.min()) if len(mp) else np.nan)


def op_from_window(dem: pd.DataFrame, mask: np.ndarray,
                   scale=1.0 / L_PER_M3) -> pd.Series:
    """Mean nodal demand over a window, in m3/s.

    `demand_series` is L/s (the simulator injects it as a pattern on a
    1 L/s base), so the /1000 here is the unit conversion. `probe_diag`
    prints the resulting total against the network's own base demands --
    this is exactly the class of bug the 0.2 demand multiplier was.
    """
    cols = [c for c in dem.columns if c != "month"]
    return dem.loc[mask, cols].mean() * scale


def probe_diag(art: dict, dem: pd.DataFrame, masks: dict, delta_frac=0.5,
               districts=None, sensors=None) -> pd.DataFrame:
    """Total demand and feasibility at every candidate operating point.

    RUN THIS FIRST on any new world set. It answers, in one table, the
    question that an empty sweep does not: is the network's own base demand
    anywhere near the demand the world actually runs at?

    On the jval worlds the answer is no -- `wn_variant` carries base demands
    about 5x the pre-window mean, so the .inp-anchored arm is infeasible
    while every world-anchored arm is fine. That ratio is a finding about
    which operating point the section-5 Jacobian was measured at, not a bug
    to scale away silently.
    """
    cands = {"inp_base": None,
             "pre_mean": op_from_window(dem, masks["pre"]),
             "post_mean": op_from_window(dem, masks["post"]),
             "all_mean": op_from_window(dem, np.ones(len(dem), bool))}
    rows = []
    for name, op in cands.items():
        r = dict(operating_point=name)
        try:
            p = probe_at(art["wn"], op, delta_frac, label=name,
                         require_feasible=False)      # report, do not raise
            r.update(total_m3s=p["total"], mean_d=p["mean_d"],
                     delta=p["delta"], n_solves=p["n_solves"],
                     n_feasible=p["n_feasible"],
                     baseline_min_p=p["baseline_min_pressure"],
                     sweep_rows=p["sweep_rows"], ok=p["ok"],
                     why="" if p["ok"] else "sweep empty: all solves infeasible")
            if p["ok"] and districts is not None and sensors is not None:
                Jd = S.district_jacobian(p["sweep"], districts, sensors,
                                         delta=p["delta"],
                                         node_weights=p["d0"] if op is None else op)
                r.update(n_matched=Jd.shape[0], **_quality(Jd))
        except Exception as e:                                # noqa: BLE001
            r.update(ok=False, why=f"{type(e).__name__}: {str(e)[:80]}")
        rows.append(r)
    df = pd.DataFrame(rows)
    if "total_m3s" in df.columns and df.total_m3s.notna().any():
        ref = df.loc[df.operating_point == "pre_mean", "total_m3s"]
        if len(ref) and ref.iloc[0]:
            df["x_pre_mean"] = (df.total_m3s / ref.iloc[0]).round(3)
    return df


def _quality(Jd: pd.DataFrame) -> dict:
    """`unmixing_quality` on an empty Jd reduces over a zero-size array."""
    if Jd.shape[0] == 0:
        return dict(n_sensors=0, n_districts=int(Jd.shape[1]),
                    cond=np.inf, sigma_min=0.0, sigma_max=0.0)
    return S.unmixing_quality(Jd)


def jd_arms(art: dict, districts: dict, sensors: list, dem: pd.DataFrame,
            masks: dict, arms=ARMS_DEFAULT, delta_frac=0.5, seed=0,
            cache: dict | None = None, cache_key=None, demand_scale=1.0
            ) -> tuple[dict, pd.DataFrame]:
    """Build the analysis arms. Topology never changes under drift, so a
    Jacobian moves only through the operating point and through
    within-district composition -- and those are separable.

    `J_perm` is built by shuffling the district columns of `J_pre`, i.e. of
    the arm it falsifies, so the two differ only in the column assignment.
    """
    wn = art["wn"]
    w_pre = op_from_window(dem, masks["pre"])
    w_post = op_from_window(dem, masks["post"])
    spec = dict(J_inp=("inp", None), J_pre=("pre", w_pre),
                J_post=("post", w_post), J_pre_wpost=("pre", w_post),
                J_perm=("pre", w_pre))
    unknown = set(arms) - set(spec)
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {sorted(spec)}")

    need = {spec[a][0] for a in arms}
    pr = {}
    if "inp" in need:                       # world-independent given topology
        ck = ("inp", cache_key, delta_frac, demand_scale)
        if cache is not None and ck in cache:
            pr["inp"] = cache[ck]
        else:
            pr["inp"] = probe_at(wn, None, delta_frac,
                                 demand_scale=demand_scale, label="inp_base")
            if cache is not None:
                cache[ck] = pr["inp"]
    if "pre" in need:
        pr["pre"] = probe_at(wn, w_pre, delta_frac,
                             demand_scale=demand_scale, label="pre_mean")
    if "post" in need:
        pr["post"] = probe_at(wn, w_post, delta_frac,
                              demand_scale=demand_scale, label="post_mean")

    out, diag = {}, []
    rng = np.random.default_rng(seed)
    for a in arms:
        src, nw = spec[a]
        p = pr[src]
        nw = p["d0"] if nw is None else nw
        Jd = S.district_jacobian(p["sweep"], districts, sensors,
                                 delta=p["delta"], node_weights=nw)
        if Jd.shape[0] < Jd.shape[1]:
            Jn = S.jacobian(p["sweep"], delta=p["delta"])
            raise ValueError(
                f"{a}: district_jacobian matched {Jd.shape[0]} of "
                f"{len(sensors)} sensors -> underdetermined "
                f"({Jd.shape[0]}x{Jd.shape[1]}). sweep {p['sweep'].shape}, "
                f"jacobian index {list(Jn.index)[:5]}, "
                f"wanted {list(sensors)[:5]}")
        if a == "J_perm":
            cols = list(Jd.columns)
            perm = list(rng.permutation(len(cols)))
            while perm == list(range(len(cols))):
                perm = list(rng.permutation(len(cols)))
            Jd = pd.DataFrame(Jd.to_numpy()[:, perm], index=Jd.index,
                              columns=cols)
        out[a] = Jd
        diag.append(dict(arm=a, probe=src, delta=p["delta"],
                         mean_d=p["mean_d"], op_total=p["total"],
                         baseline_min_p=p["baseline_min_pressure"],
                         n_feasible=p["n_feasible"], **_quality(Jd)))
    return out, pd.DataFrame(diag)



def unmix(Y: pd.DataFrame, Jd: pd.DataFrame, rcond=1e-3) -> pd.DataFrame:
    """d_hat = Jd^+ y, with rcond ACTUALLY forwarded to pinv.

    `sensitivity_probe.unmix_demands` accepts rcond and drops it, so no
    regularisation runs there (handoff 8.5). Harmless at cond 5-15; not
    harmless on jval's badly-conditioned arms.
    """
    cols = [c for c in Jd.index if c in Y.columns]
    if len(cols) < Jd.shape[1]:
        raise ValueError(f"{len(cols)} sensors for {Jd.shape[1]} districts")
    A = Jd.loc[cols].to_numpy()
    y = (Y[cols] - Y[cols].mean()).to_numpy()
    return pd.DataFrame(y @ np.linalg.pinv(A, rcond=rcond).T,
                        index=Y.index, columns=Jd.columns)


# --------------------------------------------------------------------------
# 7. Jacobian-free baselines -- so `gain` means what it did in section 5
# --------------------------------------------------------------------------
def free_estimates(Y: pd.DataFrame, sens_by_district: dict) -> dict:
    """raw / PC1-removed / robust common-mode district signals.

    Each is a (time x district) frame built WITHOUT the Jacobian, so
    `gain = unmixed - best(free)` is comparable to the pinned table.
    Robust CM uses the per-step median across standardised sensors: for a
    scalar common mode the geometric median is the median, and the mean
    version destroys exactly the structure we are trying to keep.
    """
    Z = (Y - Y.mean()) / Y.std().replace(0, np.nan)
    Z = Z.dropna(axis=1, how="all").fillna(0.0)

    def agg(F):
        return pd.DataFrame({d: F[[s for s in ss if s in F.columns]].mean(axis=1)
                             for d, ss in sens_by_district.items()
                             if any(s in F.columns for s in ss)})

    A = Z.to_numpy()
    A = A - A.mean(0)
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    pc1 = pd.DataFrame(A - np.outer(U[:, 0] * sv[0], Vt[0]),
                       index=Z.index, columns=Z.columns)
    rcm = Z.sub(Z.median(axis=1), axis=0)
    return dict(raw=agg(Z), pc1_removed=agg(pc1), robust_cm=agg(rcm))


# --------------------------------------------------------------------------
# 8. placements
# --------------------------------------------------------------------------
def placements(art: dict, names=("variance", "boundary", "manual"),
               seed=0) -> dict:
    """Flow-only placements from the observational engine. `variance` is the
    deployment recommendation: purely local, no Jacobian, no coordination,
    and the most robust to every form of model error tested (5.1)."""
    c = P.candidate_table(art["wn"], art["districts"])
    f, dt = P.topology_features(art["wn"], art["districts"], c)
    st = P.residual_stats(art["pressures"], art["flows"], c, art["steps_day"])
    base = P.strategies(c, f, st, dt, art["params"], seed=seed)
    return {n: {d: list(v["flow"]) for d, v in base[n].items()}
            for n in names if n in base}


def flatten(pl: dict) -> list:
    return [e for d in sorted(pl) for e in pl[d]]


# --------------------------------------------------------------------------
# 9. one world, end to end
# --------------------------------------------------------------------------
def _load(wdir, key) -> pd.DataFrame:
    p = pathlib.Path(wdir) / PATHS[key]
    return pd.read_parquet(p)


def run_world(w: pd.Series, place_names=("variance",), arms=ARMS_DEFAULT,
              windows=None, axis="land_use", rcond=1e-3, delta_frac=0.5,
              cache: dict | None = None, seed=0, demand_scale=1.0,
              verbose=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (results, pairs, diag).

    results: one row per (placement, arm, window, method) with M1 + M2.
    pairs:   one row per pair per window -- needed for P3/P4, which are
             statements about individual pairs, not about aggregates.
    diag:    Jacobian conditioning and operating-point totals.
    """
    wdir = w["dir"]
    art = P.load_world(wdir)
    districts = art["districts"]
    ds = sorted(districts["districts"])
    dem = _load(wdir, "demand")
    flows = art["flows"] if "flows" in art else _load(wdir, "flows")

    n = min(len(dem), len(flows))
    months = month_vector(art, dem, n)
    masks = window_masks(months, windows)
    D_true_all = district_demand(dem.iloc[:n], districts)
    fcols = [c for c in flows.columns if c != "month"]
    Yall = flows.iloc[:n][fcols].reset_index(drop=True)

    res, pairs, diags = [], [], []
    for pname in place_names:
        pl = placements(art, names=(pname,), seed=seed).get(pname)
        if pl is None:
            continue
        sens = [e for e in flatten(pl) if e in Yall.columns]
        if len(sens) < len(ds):
            # the usual cause is a naming mismatch: candidate_table gives
            # bare pipe names, client CSVs prefix them `q_<pipe>`.
            print(f"  {w['sim_hash'][:8]} {pname}: only {len(sens)} of "
                  f"{len(flatten(pl))} sensors found in flows. "
                  f"wanted {flatten(pl)[:4]}... have {list(Yall.columns)[:4]}...")
            continue
        Jds, dg = jd_arms(art, districts, sens, dem.iloc[:n], masks, arms=arms,
                          delta_frac=delta_frac, seed=seed, cache=cache,
                          demand_scale=demand_scale,
                          cache_key=(w.get("variant"), w.get("close_fraction")))
        diags.append(dg.assign(sim_hash=w["sim_hash"], placement=pname))

        free = {win: free_estimates(Yall[masks[win]], pl) for win in masks}
        for win, m in masks.items():
            if m.sum() < 48:
                continue
            Dt = D_true_all.loc[np.flatnonzero(m)]
            st_map = state_map(w, districts, win, axis=axis)
            ests = {f"free:{k}": v for k, v in free[win].items()}
            for a, Jd in Jds.items():
                ests[a] = unmix(Yall[m], Jd, rcond=rcond)
            for mname, Dh in ests.items():
                Dh = Dh.reindex(columns=ds)
                if Dh.isna().all().any():
                    continue
                Dh.index = Dt.index
                m1, m2 = M1(Dt, Dh, st_map), M2(Dt, Dh)
                res.append(dict(sim_hash=w["sim_hash"], arm_study=w.get("arm"),
                                variant=w.get("variant"),
                                sim_seed=w.get("sim_seed"),
                                beta=w.get("beta"),
                                consumption_map=w.get("consumption_map"),
                                drift_district=w.get("drift_district"),
                                drift_to_land_use=w.get("drift_to_land_use"),
                                drift_to_income=w.get("drift_to_income"),
                                placement=pname, method=mname, window=win,
                                n_steps=int(m.sum()),
                                **{k: v for k, v in m1.items()},
                                std_r=m2["std_r"], mean_r=m2["mean_r"],
                                amp_rank_corr=m2["amp_rank_corr"],
                                r_json=json.dumps(m2["r_per_district"])))
                t, h = corr_pairs(Dt), corr_pairs(Dh)
                for k in sorted(t):
                    pairs.append(dict(sim_hash=w["sim_hash"],
                                      arm_study=w.get("arm"),
                                      drift_district=w.get("drift_district"),
                                      placement=pname, method=mname,
                                      window=win, a=k[0], b=k[1],
                                      rho_true=t[k], rho_hat=h[k],
                                      state_a=st_map[k[0]],
                                      state_b=st_map[k[1]]))
    return (pd.DataFrame(res), pd.DataFrame(pairs),
            pd.concat(diags) if diags else pd.DataFrame())


def run_sweep(worlds: pd.DataFrame, progress=5, **kw):
    """Loop run_world, sharing the flat-probe cache across worlds.

    The `J_inp` probe depends only on (variant, close_fraction) because the
    topology is the same .inp everywhere, so it is computed once per coupling
    variant rather than 41 times. The world-anchored probes cannot be cached.

    A ProbeInfeasible on one world is reported and skipped, not fatal: an
    infeasible operating point is a recorded outcome for that world (the
    beta=1.0 cell was flagged as the feasibility risk in the design).
    """
    cache = kw.pop("cache", {})
    R, Pr, Dg, skip = [], [], [], []
    for i, (_, w) in enumerate(worlds.iterrows(), 1):
        try:
            r, p, d = run_world(w, cache=cache, **kw)
            R.append(r); Pr.append(p); Dg.append(d)
        except Exception as e:                                # noqa: BLE001
            skip.append(dict(sim_hash=w["sim_hash"], arm=w.get("arm"),
                             error=type(e).__name__, why=str(e)))
            print(f"  !! {w['sim_hash'][:8]} ({w.get('arm')}): "
                  f"{type(e).__name__}: {e}")
        if progress and i % progress == 0:
            print(f"  {i}/{len(worlds)}")
    cat = lambda L: (pd.concat(L, ignore_index=True) if L else pd.DataFrame())
    if skip:
        print(f"\n{len(skip)} world(s) skipped -- returned as SKIPPED, "
              "record them rather than dropping them silently")
    run_sweep.SKIPPED = pd.DataFrame(skip)
    return cat(R), cat(Pr), cat(Dg), cache


# --------------------------------------------------------------------------
# 10. aggregation, gain, cluster-aware inference (handoff 5.3)
# --------------------------------------------------------------------------
FREE = ["free:raw", "free:pc1_removed", "free:robust_cm"]


def gain_table(R: pd.DataFrame, arm="J_pre", metric="rank_corr") -> pd.DataFrame:
    """gain = unmixed - best Jacobian-free method, per (world, placement,
    window). Defined against the BEST free method, as in section 5, so the
    comparison cannot be flattered by a weak baseline."""
    idx = ["sim_hash", "placement", "window"]
    piv = R.pivot_table(index=idx, columns="method", values=metric)
    have = [c for c in FREE if c in piv.columns]
    if arm not in piv.columns or not have:
        return pd.DataFrame()
    out = piv[[arm] + have].copy()
    out["best_free"] = out[have].max(axis=1) if metric == "rank_corr" \
        else out[have].min(axis=1)
    out["gain"] = (out[arm] - out["best_free"]) if metric == "rank_corr" \
        else (out["best_free"] - out[arm])
    meta = R.drop_duplicates("sim_hash").set_index("sim_hash")[
        [c for c in ("arm_study", "variant", "beta", "consumption_map",
                     "sim_seed", "drift_district") if c in R.columns]]
    return out.reset_index().merge(meta, on="sim_hash", how="left")


def cluster_summary(df: pd.DataFrame, value="gain",
                    cluster=("consumption_map", "sim_seed")) -> pd.DataFrame:
    """Sign test over clusters, not a paired t over correlated worlds.

    Section 5.3: the n=40 p-values treated 4 consumption maps as 40
    independent draws. The cluster is the unit that actually varies.
    """
    cl = [c for c in cluster if c in df.columns]
    g = df.groupby(cl)[value].mean().reset_index()
    k = int((g[value] > 0).sum()); nn = int(g[value].notna().sum())
    p = sps.binomtest(k, nn, 0.5, alternative="greater").pvalue if nn else np.nan
    return pd.DataFrame([dict(n_clusters=nn, n_positive=k,
                              mean=float(g[value].mean()),
                              min=float(g[value].min()),
                              max=float(g[value].max()),
                              sign_test_p=float(p))]), g


# --------------------------------------------------------------------------
# 11. pair-level predictions P3 / P4
# --------------------------------------------------------------------------
def delta_rho(Pr: pd.DataFrame, method, placement="variance",
              w0="pre", w1="post") -> pd.DataFrame:
    """Pre->post change per pair, with the drift-implied sign prediction.

    predicted: +1 if the pair BECOMES same-state, -1 if it stops being
    same-state, 0 if the pair does not involve the drifter (the invariance
    null -- unchanged by construction, so any estimated movement is pure
    method error).
    """
    q = Pr[(Pr.method == method) & (Pr.placement == placement)]
    key = [c for c in ("sim_hash", "arm_study", "a", "b", "drift_district")
           if c in q.columns]
    A = q[q.window == w0].set_index(key)
    B = q[q.window == w1].set_index(key)
    j = A[["rho_true", "rho_hat", "state_a", "state_b"]].join(
        B[["rho_true", "rho_hat", "state_a", "state_b"]],
        lsuffix="_0", rsuffix="_1", how="inner").reset_index()
    j["d_true"] = j.rho_true_1 - j.rho_true_0
    j["d_hat"] = j.rho_hat_1 - j.rho_hat_0
    j["involves_drifter"] = (j.a == j.drift_district) | (j.b == j.drift_district)
    same0 = j.state_a_0 == j.state_b_0
    same1 = j.state_a_1 == j.state_b_1
    j["predicted"] = np.where(~same0 & same1, 1,
                              np.where(same0 & ~same1, -1, 0))
    j.loc[~j.involves_drifter, "predicted"] = 0
    return j


# --------------------------------------------------------------------------
# 12. the pre-registered scoreboard (handoff 7.8)
# --------------------------------------------------------------------------
def predictions(R: pd.DataFrame, Pr: pd.DataFrame, main_arm="J_pre",
                placement="variance", noise_floor=0.02,
                p1_target=0.05) -> pd.DataFrame:
    """Score P1..P7. Written before the numbers existed, on purpose."""
    out = []

    def add(pid, claim, value, verdict, note=""):
        out.append(dict(P=pid, claim=claim, value=value, verdict=verdict,
                        note=note))

    q = R[(R.placement == placement) & (R.method == main_arm)]

    # P1 -- pre-window MAE reproduces the pinned table
    v = q.loc[q.window == "pre", "mae"].mean()
    add("P1", f"pre-window MAE ~ {p1_target}", round(float(v), 4),
        "PASS" if np.isfinite(v) and v < 2 * p1_target else "FAIL",
        "failure implicates the windowing, not the method")

    # P2 -- transition is the worst window
    piv = q.pivot_table(index="sim_hash", columns="window", values="mae")
    if {"pre", "transition", "post"} <= set(piv.columns):
        worst = piv[["pre", "transition", "post"]].idxmax(axis=1)
        frac = float((worst == "transition").mean())
        add("P2", "transition window worst of three", round(frac, 3),
            "PASS" if frac > 0.5 else "FAIL",
            f"{int((worst=='transition').sum())}/{len(worst)} worlds; "
            "failure -> look at the operating point, not composition")
    else:
        add("P2", "transition window worst of three", np.nan, "NO DATA")

    # P3 -- invariance null on non-drifter pairs
    d = delta_rho(Pr, main_arm, placement)
    nd = d[~d.involves_drifter]
    if len(nd):
        v = float(nd.d_hat.abs().mean())
        add("P3", f"|d rho_hat| non-drifter pairs < {noise_floor}",
            round(v, 4), "PASS" if v < noise_floor else "FAIL",
            f"n={len(nd)} pairs; true |d| = {nd.d_true.abs().mean():.4f}")
    else:
        add("P3", "invariance null", np.nan, "NO DATA")

    # P4 -- signed change on drifter pairs
    sd = d[d.predicted != 0]
    if len(sd):
        hit = float((np.sign(sd.d_hat) == sd.predicted).mean())
        add("P4", "signed change correct in >=90%", round(hit, 3),
            "PASS" if hit >= 0.90 else "FAIL",
            f"n={len(sd)} pairs; true-sign agreement "
            f"{float((np.sign(sd.d_true)==sd.predicted).mean()):.3f}")
    else:
        add("P4", "signed prediction", np.nan, "NO DATA")

    # P5 -- isolated worlds show near-zero gain  (LOAD-BEARING)
    G = gain_table(R, arm=main_arm)
    if len(G) and "variant" in G.columns:
        g = G[(G.placement == placement) & (G.window == "pre")]
        iso = g.loc[g.variant == "isolated", "gain"]
        base = g.loc[g.variant == "baseline", "gain"]
        if len(iso) and len(base):
            add("P5", "isolated gain ~ 0 while baseline gain > 0",
                f"iso {iso.mean():+.3f} / base {base.mean():+.3f}",
                "PASS" if iso.mean() < 0.5 * base.mean() else "FAIL",
                "load-bearing: failure means the gain is not hydraulic mixing")
        else:
            add("P5", "isolated vs baseline gain", np.nan,
                "NO DATA", "needs jval_coupling worlds")
    else:
        add("P5", "isolated vs baseline gain", np.nan, "NO DATA")

    # P6 -- monotone degradation in beta
    b = q[(q.window == "post") & q.beta.notna()]
    if b.beta.nunique() >= 3:
        s = b.groupby("beta")["mae"].mean().sort_index()
        mono = bool(np.all(np.diff(s.to_numpy()) >= -1e-9))
        add("P6", "MAE monotone increasing in beta",
            " / ".join(f"{k}:{v:.3f}" for k, v in s.items()),
            "PASS" if mono else "FAIL",
            "failure -> the operating point is not the mechanism")
    else:
        add("P6", "monotone in beta", np.nan, "NO DATA",
            f"{b.beta.nunique()} beta levels present")

    # P7 -- M2 ranks districts by amplitude
    inc = R[(R.placement == placement) & (R.method == main_arm) &
            (R.arm_study == "income") & (R.window == "pre")]
    if len(inc):
        v = float(inc.amp_rank_corr.mean())
        add("P7", "M2 amplitude rank correct", round(v, 3),
            "PASS" if v > 0.8 else "FAIL",
            f"n={len(inc)} income worlds; std_r={inc.std_r.mean():.3f}. "
            "A negative answer to Q3 is still worth reporting.")
    else:
        add("P7", "M2 amplitude rank", np.nan, "NO DATA",
            "needs jval_income worlds")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 13. persistence
# --------------------------------------------------------------------------
def save(outdir, **frames):
    d = pathlib.Path(outdir); d.mkdir(parents=True, exist_ok=True)
    for k, v in frames.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.to_parquet(d / f"{k}.parquet", index=False)
    return sorted(p.name for p in d.glob("*.parquet"))


def save_cache(path, cache):
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def load_cache(path):
    p = pathlib.Path(path)
    return pickle.load(open(p, "rb")) if p.exists() else {}


# --------------------------------------------------------------------------
# 14. self-test -- known-answer construction, no world needed
# --------------------------------------------------------------------------
def selftest(verbose=True) -> bool:
    """Validate the measurands against constructions whose answer is known.

    Same discipline as the rest of the project: every method is checked on
    a case with an answer by construction before it touches Graeme.
    """
    rng = np.random.default_rng(0)
    ds = [f"District_{c}" for c in "ABCDE"]
    T = 4320
    z = rng.normal(size=T)
    # A,B,C,D residential-like (share z strongly); E industrial-like
    cols = {d: 0.95 * z + 0.31 * rng.normal(size=T) for d in ds[:4]}
    cols[ds[4]] = rng.normal(size=T)
    amp = dict(zip(ds, [1.0, 1.0, 1.0, 1.0, 3.6]))
    Dt = pd.DataFrame({d: cols[d] * amp[d] for d in ds})

    # (a) M1 is scale-blind: multiply every district by its own constant
    Dh = Dt * pd.Series({d: 0.3 * (i + 1) for i, d in enumerate(ds)})
    m1 = M1(Dt, Dh, {d: ("R" if d != ds[4] else "I") for d in ds})
    assert m1["mae"] < 1e-9 and abs(m1["rank_corr"] - 1) < 1e-9, m1
    assert abs(m1["gap_retention"] - 1) < 1e-9, m1

    # (b) M1 gap is positive when same-state pairs really are more similar
    assert m1["G_true"] > 0.3, m1

    # (c) M2 is shape-blind but scale-sensitive; global factor unidentifiable
    Dh2 = Dt * 7.0
    m2 = M2(Dt, Dh2)
    assert m2["std_r"] < 1e-9, m2
    assert abs(m2["mean_r"] - np.log(7.0)) < 1e-9, m2
    assert abs(m2["amp_rank_corr"] - 1) < 1e-9, m2

    # (d) M2 catches a district-specific amplitude error. Note the two
    #     statistics are independent: /3.6 flattens E to the others and
    #     moves std_r while leaving the RANK intact, so the rank inversion
    #     has to be forced separately.
    m2b = M2(Dt, Dt.assign(**{ds[4]: Dt[ds[4]] / 3.6}))
    assert m2b["std_r"] > 0.3 and abs(m2b["amp_rank_corr"] - 1) < 1e-6, m2b
    m2c = M2(Dt, Dt.assign(**{ds[4]: Dt[ds[4]] / 20.0}))
    assert m2c["amp_rank_corr"] < 0.9, m2c

    # (e) unmix inverts a known mixing matrix exactly
    A = rng.normal(size=(9, 5)); A[np.abs(A) < .2] += .5
    Jd = pd.DataFrame(A, index=[f"s{i}" for i in range(9)], columns=ds)
    Y = pd.DataFrame(Dt.to_numpy() @ A.T + 3.0, columns=Jd.index)
    Dh4 = unmix(Y, Jd, rcond=1e-12)
    err = float(np.abs(np.corrcoef(Dh4.T) - np.corrcoef(Dt.T)).max())
    assert err < 1e-6, err

    # (f) state_map remaps the drifter post-drift
    w = pd.Series(dict(consumption_map="LR_LR_LR_LR_LI",
                       drift_district="District_D", drift_to_land_use="industrial",
                       drift_to_income="low"))
    dd = {"districts": {d: [] for d in ds}}
    assert state_map(w, dd, "pre")["District_D"] == "R"
    assert state_map(w, dd, "post")["District_D"] == "I"
    assert state_map(w, dd, "post", axis="income")["District_D"] == "L"

    # (g) signed prediction matches the handoff's S1 statement
    pr = pd.DataFrame([
        dict(sim_hash="h", a=x, b=y, drift_district="District_D",
             placement="variance", method="J_pre", window=win,
             rho_true=0.0, rho_hat=0.0,
             state_a=state_map(w, dd, win)[x], state_b=state_map(w, dd, win)[y])
        for win in ("pre", "post")
        for i, x in enumerate(ds) for y in ds[i + 1:]])
    d = delta_rho(pr, "J_pre")
    got = {(r.a, r.b): r.predicted for r in d.itertuples()}
    assert got[("District_D", "District_E")] == 1, got
    assert got[("District_A", "District_D")] == -1, got
    assert got[("District_A", "District_B")] == 0, got

    if verbose:
        print("selftest OK  |  M1 scale-blind, M2 scale-sensitive, "
              "unmix exact, regime remap and signed prediction correct")
    return True


if __name__ == "__main__":
    selftest()
