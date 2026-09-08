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


# The jval_* design, transcribed from experiments_jval.yml. Arms are matched
# against THIS rather than inferred from the spec, because a rule ("cmap ends
# LI and beta != 0.35 -> beta arm") also swallows every pre-existing 48-month
# world that happens to look similar. The first run of this notebook labelled
# 20 worlds `s1_oddone` against a design of 15 and mixed five foreign worlds
# into the beta arm, which is what broke P6.
_S1 = [("District_A", "low", "industrial"), ("District_B", "low", "industrial"),
       ("District_C", "low", "industrial"), ("District_D", "low", "industrial"),
       ("District_E", "low", "residential")]
_S2 = [("District_A", "low", "industrial"), ("District_B", "low", "industrial"),
       ("District_C", "low", "industrial"), ("District_D", "low", "residential"),
       ("District_E", "low", "residential")]
_INC = [("District_C", "high", "residential"), ("District_E", "low", "residential")]
_CPL = [("District_A", "low", "industrial"), ("District_E", "low", "residential")]
SEEDS = (42, 43, 44)


def design_table() -> pd.DataFrame:
    """One row per designed world; `cell` is the design identity."""
    rows = []

    def add(study, seed, beta, cmap, variant, drift):
        d, inc, lu = drift
        rows.append(dict(study=study, sim_seed=seed, beta=beta,
                         consumption_map=cmap, variant=variant,
                         drift_district=d, drift_to_income=inc,
                         drift_to_land_use=lu))

    for sd in SEEDS:
        for dr in _S1:
            add("s1_oddone", sd, 0.35, "LR_LR_LR_LR_LI", "baseline", dr)
        for dr in _S2:
            add("s2_split", sd, 0.35, "LR_LR_LR_LI_LI", "baseline", dr)
        for dr in _INC:
            add("income", sd, 0.35, "LR_LR_LR_LR_HR", "baseline", dr)
    for b in (0.0, 0.35, 1.0):
        add("beta", 42, b, "LR_LR_LR_LR_LI", "baseline",
            ("District_D", "low", "industrial"))
    for var in ("baseline", "isolated"):
        for dr in _CPL:
            add("coupling", 42, 0.35, "LR_LR_LR_LR_LI", var, dr)
    df = pd.DataFrame(rows)
    df["cell"] = (df.consumption_map + "|" + df.variant + "|b"
                  + df.beta.map("{:.2f}".format) + "|s"
                  + df.sim_seed.astype(str) + "|"
                  + df.drift_district.str[-1] + df.drift_to_income.str[0]
                  + df.drift_to_land_use.str[0])
    # a cell can serve several studies (jval_coupling's baseline arm and
    # jval_beta's 0.35 cell ARE jval_s1_oddone worlds -- deliberate cache hits)
    g = df.groupby("cell")["study"].apply(lambda x: ",".join(sorted(set(x))))
    df = df.drop_duplicates("cell").drop(columns="study").merge(
        g.rename("studies"), on="cell")
    df["arm"] = df.studies.str.split(",").str[0]
    return df


KEYS = ["consumption_map", "variant", "beta", "sim_seed", "drift_district",
        "drift_to_income", "drift_to_land_use"]


def _cell(w) -> str:
    return (f"{w['consumption_map']}|{w['variant']}|b{float(w['beta']):.2f}"
            f"|s{int(w['sim_seed'])}|{str(w['drift_district'])[-1]}"
            f"{str(w['drift_to_income'])[0]}{str(w['drift_to_land_use'])[0]}")


def label_arms(idx: pd.DataFrame, onset: pd.Series | None = None
               ) -> pd.DataFrame:
    """Attach `cell` / `arm` / `studies` by matching the design.

    Worlds that match no design cell get arm "other" and must be excluded.
    When a design cell has several matching sim_hashes the extras are marked
    `dup=True`: the design says one world per cell, so a second is a foreign
    world colliding on the spec fields (usually a different warmup_months,
    which is hashed but not surfaced in the manifest's world block).
    """
    D = design_table()
    out = idx.copy()
    miss = [k for k in KEYS if k not in out.columns]
    if miss:
        raise KeyError(f"index missing design keys {miss}")
    out["cell"] = out.apply(_cell, axis=1)
    out = out.merge(D[["cell", "arm", "studies"]], on="cell", how="left")
    out["arm"] = out["arm"].fillna("other")
    out["studies"] = out["studies"].fillna("")
    if onset is not None:
        out["onset"] = out["sim_hash"].map(onset)
        # warmup_months is hashed but absent from the manifest world block;
        # onset == warmup_months exactly (handoff 7.3), so onset != 6 marks a
        # world built under a different warmup than the design specifies.
        out.loc[out.onset.notna() & (out.onset != 6), "arm"] = "other"
    key = ["onset"] if "onset" in out.columns else []
    out = out.sort_values(["cell"] + key + ["sim_hash"])
    out["dup"] = out.duplicated("cell") & (out.arm != "other")
    return out.sort_values(["arm", "cell"]).reset_index(drop=True)


def design_audit(labelled: pd.DataFrame) -> pd.DataFrame:
    """Designed vs found vs kept, per study. The count check the first run
    of this notebook needed and did not have."""
    D = design_table()
    rows = []
    for st in ("s1_oddone", "s2_split", "income", "beta", "coupling"):
        want = int(D.studies.str.contains(st).sum())
        m = labelled.studies.str.contains(st) & ~labelled.dup
        rows.append(dict(study=st, designed=want, matched=int(m.sum()),
                         duplicates=int((labelled.studies.str.contains(st)
                                         & labelled.dup).sum()),
                         complete=int(m.sum()) == want))
    rows.append(dict(study="(unmatched)", designed=0,
                     matched=int((labelled.arm == "other").sum()),
                     duplicates=0, complete=True))
    return pd.DataFrame(rows)


def jval_index(proj, n_months=48, experiments_root=None) -> pd.DataFrame:
    """Discover worlds and keep the jval horizon. Arms are attached by
    `label_arms` after gating, because the warmup check needs the onset."""
    kw = {} if experiments_root is None else dict(experiments_root=experiments_root)
    idx = P.discover_worlds(proj, **kw)
    if not len(idx):
        raise RuntimeError("discover_worlds returned nothing")
    if n_months is not None:
        if "n_months" not in idx.columns:
            raise KeyError("manifest 'world' block has no n_months -- pass "
                           f"n_months=None. Columns: {sorted(idx.columns)}")
        idx = idx[idx["n_months"] == n_months].copy()
    return idx.sort_values(["consumption_map", "drift_district",
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


def pin(idx: pd.DataFrame, gates: pd.DataFrame, path=None,
        keep_other=False) -> pd.DataFrame:
    """Freeze the analysed world set to an explicit sim_hash list (6.9).

    Excludes, in order: worlds that fail the 7.7 gate, worlds matching no
    design cell (`arm == "other"`), and duplicate matches on a design cell.
    Every table in the thesis should be reconstructible from this file;
    `.head(n)` on a growing pool already cost this project one headline.
    """
    ok = set(gates.loc[gates["ok"], "sim_hash"])
    out = idx[idx["sim_hash"].isin(ok) & idx["usable"]].copy()
    if "dup" in out.columns:
        out = out[~out["dup"]]
    if not keep_other and "arm" in out.columns:
        out = out[out["arm"] != "other"]
    if path:
        cols = [c for c in ("sim_hash", "cell", "arm", "studies", "variant",
                            "sim_seed", "beta", "anchor_scale",
                            "consumption_map", "drift_district",
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
ARMS_DEFAULT = ("J_inp", "J_pre", "J_post", "J_pre_wpost", "J_perm")

# `J_inp` is the old `J_flat`: the network at its own base demands, scaled
# by `anchor_scale`. This is the section-5 Jacobian, so it is IN the default
# set -- the pre-window J_inp vs J_pre gap is exactly the misspecification
# penalty that section 5's 0.052 MAE turned out to be.
#
# It is feasible only if `anchor_scale` is forwarded to `flat_baseline`,
# whose default is 1.0 while the worlds are built at 0.05. Probing at the
# unscaled bases puts 20x the world's demand on the network and every solve
# comes back infeasible (min pressure -865 mca). `probe_diag` shows this as
# `x_pre_mean ~ 20`.
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
             require_feasible=True, anchor_scale=1.0, **fb_kw) -> dict:
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
    flat = S.flat_baseline(wn, anchor_scale=anchor_scale,
                           demand_multiplier=1.0, **fb_kw)
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
               districts=None, sensors=None, anchor_scale=1.0) -> pd.DataFrame:
    """Total demand and feasibility at every candidate operating point.

    RUN THIS FIRST on any new world set. It answers, in one table, the
    question that an empty sweep does not: is the network's own base demand
    anywhere near the demand the world actually runs at?

    Read `x_pre_mean`: the ratio of each candidate's total demand to the
    pre-window mean. With `anchor_scale` forwarded correctly it lands at
    ~1.0 for `inp_base`; at ~20 (= 1 / 0.05) the anchor was not forwarded
    and `flat_baseline`'s default of 1.0 was used, which puts 20x the
    world's demand on the network and makes every solve infeasible.
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
                         anchor_scale=anchor_scale,
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
    df["x_pre_mean"] = np.nan
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


def world_probes(art: dict, dem: pd.DataFrame, masks: dict, arms=ARMS_DEFAULT,
                 delta_frac=0.5, cache: dict | None = None, cache_key=None,
                 demand_scale=1.0, anchor_scale=1.0) -> tuple[dict, dict]:
    """The perturbation sweeps a world needs, one per operating point.

    Hoisted out of the placement loop on purpose: a sweep is 113
    zero-duration solves and does not depend on where the sensors are, so
    comparing four placements must not cost four times the hydraulics.
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
        ck = ("inp", cache_key, delta_frac, demand_scale, anchor_scale)
        if cache is not None and ck in cache:
            pr["inp"] = cache[ck]
        else:
            pr["inp"] = probe_at(wn, None, delta_frac, label="inp_base",
                                 demand_scale=demand_scale,
                                 anchor_scale=anchor_scale)
            if cache is not None:
                cache[ck] = pr["inp"]
    for k, op, lab in (("pre", w_pre, "pre_mean"), ("post", w_post, "post_mean")):
        if k in need:
            pr[k] = probe_at(wn, op, delta_frac, label=lab,
                             demand_scale=demand_scale,
                             anchor_scale=anchor_scale)
    return pr, spec


def jd_from_probes(pr: dict, spec: dict, districts: dict, sensors: list,
                   arms=ARMS_DEFAULT, seed=0) -> tuple[dict, pd.DataFrame]:
    """Aggregate node columns to districts, one Jacobian per arm.

    `J_perm` shuffles the district columns of `J_pre`, i.e. of the arm it
    falsifies, so the two are the same matrix with the wrong assignment --
    identical sensors, identical conditioning.
    """
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


def jd_arms(art: dict, districts: dict, sensors: list, dem: pd.DataFrame,
            masks: dict, arms=ARMS_DEFAULT, delta_frac=0.5, seed=0,
            cache: dict | None = None, cache_key=None, demand_scale=1.0,
            anchor_scale=1.0) -> tuple[dict, pd.DataFrame]:
    """Convenience wrapper: probe then build. Topology never changes under
    drift, so a Jacobian moves only through the operating point and through
    within-district composition -- and those are separable."""
    pr, spec = world_probes(art, dem, masks, arms, delta_frac, cache,
                            cache_key, demand_scale, anchor_scale)
    return jd_from_probes(pr, spec, districts, sensors, arms, seed)


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
PLACEMENTS_ALL = ("variance", "boundary", "manual", "spread", "random")


def placements(art: dict, names=PLACEMENTS_ALL, seed=0) -> dict:
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
              anchor_scale=None, verbose=False
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (results, pairs, diag).

    results: one row per (placement, arm, window, method) with M1 + M2.
    pairs:   one row per pair per window -- needed for P3/P4, which are
             statements about individual pairs, not about aggregates.
    diag:    Jacobian conditioning, operating point and feasibility, per
             placement (conditioning depends on placement; the probe does
             not, and is computed once).

    `anchor_scale` defaults to the world's own spec value, which is the
    only correct choice: `flat_baseline`'s default is 1.0 and the worlds are
    built at 0.05.
    """
    wdir = w["dir"]
    if anchor_scale is None:
        anchor_scale = float(w.get("anchor_scale", 1.0) or 1.0)
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

    # one set of sweeps for the whole world, shared by every placement
    pr, spec = world_probes(art, dem.iloc[:n], masks, arms, delta_frac, cache,
                            (w.get("variant"), w.get("close_fraction")),
                            demand_scale, anchor_scale)

    all_pl = placements(art, names=tuple(place_names), seed=seed)
    res, pairs, diags = [], [], []
    for pname in place_names:
        pl = all_pl.get(pname)
        if pl is None:
            if verbose:
                print(f"  {w['sim_hash'][:8]}: placement '{pname}' not "
                      f"offered by P.strategies (have {sorted(all_pl)})")
            continue
        sens = [e for e in flatten(pl) if e in Yall.columns]
        if len(sens) < len(ds):
            # the usual cause is a naming mismatch: candidate_table gives
            # bare pipe names, client CSVs prefix them `q_<pipe>`.
            print(f"  {w['sim_hash'][:8]} {pname}: only {len(sens)} of "
                  f"{len(flatten(pl))} sensors found in flows. "
                  f"wanted {flatten(pl)[:4]}... have {list(Yall.columns)[:4]}...")
            continue
        Jds, dg = jd_from_probes(pr, spec, districts, sens, arms, seed)
        diags.append(dg.assign(sim_hash=w["sim_hash"], placement=pname,
                               arm_study=w.get("arm"), n_placed=len(sens)))

        free = {win: free_estimates(Yall[masks[win]], pl) for win in masks}
        for win, m in masks.items():
            if m.sum() < 48:
                continue
            Dt = D_true_all.loc[np.flatnonzero(m)]
            st_map = state_map(w, districts, win, axis=axis)
            ests = {f"free:{k}": v for k, v in free[win].items()}
            for a, Jd in Jds.items():
                ests[a] = unmix(Yall[m], Jd, rcond=rcond)
            sig = {a: _quality(Jd)["sigma_min"] for a, Jd in Jds.items()}
            for mname, Dh in ests.items():
                Dh = Dh.reindex(columns=ds)
                if Dh.isna().all().any():
                    continue
                Dh.index = Dt.index
                m1, m2 = M1(Dt, Dh, st_map), M2(Dt, Dh)
                res.append(dict(sim_hash=w["sim_hash"], arm_study=w.get("arm"),
                                cell=w.get("cell"),
                                variant=w.get("variant"),
                                sim_seed=w.get("sim_seed"),
                                beta=w.get("beta"),
                                consumption_map=w.get("consumption_map"),
                                drift_district=w.get("drift_district"),
                                drift_to_land_use=w.get("drift_to_land_use"),
                                drift_to_income=w.get("drift_to_income"),
                                placement=pname, method=mname, window=win,
                                n_steps=int(m.sum()), n_placed=len(sens),
                                sigma_min=sig.get(mname, np.nan),
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
# 10b. placement comparison (handoff 5.1)
# --------------------------------------------------------------------------
def placement_table(R: pd.DataFrame, DG: pd.DataFrame, arm="J_pre",
                    window="pre") -> pd.DataFrame:
    """One row per placement: conditioning, recovery, and gain.

    Section 5.1's claim is specific and worth testing separately: sigma_min
    predicts ROBUSTNESS to Jacobian error, not the LEVEL of recovery. So
    this table reports sigma_min alongside recovery in the well-specified
    arm (`J_pre`) and in the misspecified one (`J_inp`, whose node weights
    are the .inp proportions rather than the window's). The difference
    between those two columns is the misspecification penalty, and THAT is
    what sigma_min should order -- not the J_pre column.
    """
    q = R[R.window == window]
    piv = q.pivot_table(index="placement", columns="method",
                        values=["mae", "rank_corr"])
    sig = (DG[DG.arm == arm].groupby("placement")[["sigma_min", "cond"]].mean()
           if len(DG) else pd.DataFrame())
    rows = []
    for pl in sorted(q.placement.unique()):
        r = dict(placement=pl,
                 n_worlds=int(q[q.placement == pl].sim_hash.nunique()))
        if len(sig) and pl in sig.index:
            r.update(sigma_min=float(sig.loc[pl, "sigma_min"]),
                     cond=float(sig.loc[pl, "cond"]))
        for m in (arm, "J_inp", "J_perm"):
            if ("mae", m) in piv.columns:
                r[f"mae_{m}"] = float(piv.loc[pl, ("mae", m)])
                r[f"rc_{m}"] = float(piv.loc[pl, ("rank_corr", m)])
        frees = [c for c in piv.columns.get_level_values(1).unique()
                 if str(c).startswith("free:")]
        if frees:
            r["mae_best_free"] = min(float(piv.loc[pl, ("mae", f)]) for f in frees)
            r["rc_best_free"] = max(float(piv.loc[pl, ("rank_corr", f)])
                                    for f in frees)
            r["gain_mae"] = r["mae_best_free"] - r.get(f"mae_{arm}", np.nan)
            r["gain_rc"] = r.get(f"rc_{arm}", np.nan) - r["rc_best_free"]
        if f"mae_{arm}" in r and "mae_J_inp" in r:
            r["misspec_penalty"] = r["mae_J_inp"] - r[f"mae_{arm}"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values(f"mae_{arm}").reset_index(drop=True)


def placement_ranking(R: pd.DataFrame, DG: pd.DataFrame, arm="J_pre",
                      window="pre") -> pd.DataFrame:
    """Does sigma_min order recovery, or only order the penalty?

    Spearman across placements, which is 4-5 points -- a direction, not a
    result. Reported because 6.7 downgraded the original claim from
    'sigma_min predicts recovery' to 'predicts robustness', and this is the
    cheapest available check of the downgraded version.
    """
    T = placement_table(R, DG, arm, window)
    out = []
    for col, sign in ((f"mae_{arm}", -1), ("misspec_penalty", -1),
                      (f"rc_{arm}", +1), ("gain_mae", +1)):
        if col not in T.columns or "sigma_min" not in T.columns:
            continue
        d = T[["sigma_min", col]].dropna()
        if len(d) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = sps.spearmanr(d.sigma_min, d[col]).statistic
        out.append(dict(target=col, n_placements=len(d),
                        spearman_vs_sigma_min=float(rho),
                        expected_sign="higher sigma_min -> "
                        + ("lower " if sign < 0 else "higher ") + col))
    return pd.DataFrame(out, columns=["target", "n_placements",
                                      "spearman_vs_sigma_min",
                                      "expected_sign"])


def placement_by_window(R: pd.DataFrame, arm="J_pre", value="mae"
                        ) -> pd.DataFrame:
    q = R[R.method == arm]
    return q.pivot_table(index="placement", columns="window", values=value)


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
# 11b. structured answers -- one record per question, derived not narrated
# --------------------------------------------------------------------------
# Every threshold used to turn a number into an answer is declared here, so
# a verdict can be traced to a rule rather than to a sentence someone typed.
THRESH = dict(
    # 2 correlation points. A Spearman over 10 pairs has a ~0.02 noise floor
    # (handoff 5.2), so MAE differences below this are not distinguishable
    # from estimator noise and must not be reported as effects.
    mae_negligible=0.02,
    gap_retention_ok=0.90,     # G_hat/G_true within 10%
    amp_rank_ok=0.80,          # M2 orders districts by amplitude
    std_r_ok=0.10,             # spread of log-amplitude error across districts
    perm_collapse=5.0,         # J_perm MAE must exceed the main arm by this x
    sign_ok=0.90,              # P4
    excursion_min=1.10,        # below this beta has no reach on the operating pt
    cluster_p=0.05,
    placement_equiv=0.02,      # placements closer than this are equivalent
)


class Ledger:
    """Accumulates one row per answered question.

    Exists so the notebook reports ANSWERS rather than prints: each row
    carries the question, the statistic it was decided on, the value, the
    verdict, and the rule that produced the verdict. Tables stay in the
    notebook underneath for sanity checking; this is what gets read.
    """

    COLS = ["exp", "question", "metric", "value", "answer", "basis"]

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, exp, question, metric, value, answer, basis=""):
        self.rows.append(dict(exp=exp, question=question, metric=metric,
                              value=value, answer=answer, basis=basis))
        return self

    def extend(self, df):
        if isinstance(df, pd.DataFrame) and len(df):
            self.rows.extend(df.to_dict("records"))
        return self

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLS)

    def sheet(self, wrap=None) -> pd.DataFrame:
        """The answer sheet: questions and verdicts, basis dropped."""
        f = self.frame()
        return f[["exp", "question", "metric", "value", "answer"]]


def _fmt(v, n=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{n}f}" if isinstance(v, (int, float, np.floating)) else str(v)


def _rows(exp, *specs) -> pd.DataFrame:
    return pd.DataFrame([dict(exp=exp, question=q, metric=m, value=v,
                              answer=a, basis=b)
                         for q, m, v, a, b in specs],
                        columns=Ledger.COLS)


def answer_inventory(lab: pd.DataFrame, worlds: pd.DataFrame) -> pd.DataFrame:
    aud = design_audit(lab)
    des = aud[aud.study != "(unmatched)"]
    complete = bool(des.complete.all())
    n_other = int((lab.arm == "other").sum())
    n_dup = int(lab.dup.sum())
    return _rows(
        "E1",
        ("Did we build the design, and only the design?",
         "studies complete / total",
         f"{int(des.complete.sum())}/{len(des)}",
         "YES" if complete else "NO -- arms incomplete",
         "design_table() cell-by-cell match"),
        ("How many 48-month worlds are foreign to the design?",
         "unmatched + duplicate-cell", f"{n_other} + {n_dup}",
         "none" if n_other + n_dup == 0 else f"{n_other + n_dup} excluded",
         "arm=='other' or dup=True; rule-based labelling had swept these in"),
        ("How many worlds carry the analysis?", "pinned", int(len(worlds)),
         f"{len(worlds)} worlds, {worlds.arm.nunique()} arms",
         "gate 7.7 passed AND design-matched AND not duplicate"),
    )


def answer_truth(DT: pd.DataFrame, TA: pd.DataFrame, floor: float
                 ) -> pd.DataFrame:
    nd = DT[~DT.involves_drifter]
    sd = DT[DT.predicted != 0]
    ceil = (np.nan if not len(sd)
            else float((np.sign(sd.d_true) == sd.predicted).mean()))
    gs = []
    for win in ("pre", "post"):
        q = TA[TA.window == win]
        gs.append(pd.Series({h: (x.loc[x.state_a == x.state_b, "rho_true"].mean()
                                 - x.loc[x.state_a != x.state_b, "rho_true"].mean())
                             for h, x in q.groupby("sim_hash")}))
    g_sd = float(pd.concat(gs).std())
    eff = float(sd.d_true.abs().mean()) if len(sd) else np.nan
    return _rows(
        "E4",
        ("Is the invariance null clean enough in the TRUTH to score P3?",
         "mean |d rho_true|, non-drifter pairs",
         float(nd.d_true.abs().mean()),
         "YES" if nd.d_true.abs().mean() < THRESH["mae_negligible"]
         else "NO -- the null itself moves",
         f"threshold {THRESH['mae_negligible']}; P3 floor set to {floor:.4f}"),
        ("Does the intervention move the truth in the predicted direction?",
         "sign agreement in truth (P4 ceiling)", ceil,
         "YES" if ceil is not None and ceil >= THRESH["sign_ok"]
         else "NO -- P4 indicts the prediction, not the estimator",
         f"ceiling for P4; threshold {THRESH['sign_ok']}"),
        ("How demanding is the gap test?", "effect size |d rho_true|", eff,
         "large" if eff > 0.2 else "small",
         f"sd(G_true) across worlds = {g_sd:.4f}; near-zero means G is an "
         "almost deterministic function of the land-use token, so passing "
         "the gap test is cheap"),
    )


def answer_reproduction(R: pd.DataFrame, placement="variance",
                        main_arm="J_pre", pinned_mae=0.052) -> pd.DataFrame:
    q = R[(R.placement == placement) & (R.window == "pre")]
    piv = q.pivot_table(index="sim_hash", columns="method", values="mae")
    v = float(piv[main_arm].mean()) if main_arm in piv else np.nan
    inp = float(piv["J_inp"].mean()) if "J_inp" in piv else np.nan
    perm = float(piv["J_perm"].mean()) if "J_perm" in piv else np.nan
    return _rows(
        "E7",
        ("Does recovery survive windowed analysis?",
         f"pre-window MAE ({main_arm})", v,
         "YES" if v < 2 * pinned_mae else "NO",
         f"section 5 pinned {pinned_mae} on the whole 30-month series"),
        ("How much of section 5's error was misspecification, not method?",
         "MAE(J_inp) - MAE(main)", inp - v,
         ("misspecification dominates" if inp - v > v
          else "method-limited" if np.isfinite(inp) else "n/a"),
         "J_inp uses .inp node weights at the .inp operating point; the main "
         "arm uses window means for both. Same sensors, same worlds."),
        ("Does the falsification control collapse?",
         "MAE(J_perm) / MAE(main)", perm / v if v else np.nan,
         "YES" if perm > THRESH["perm_collapse"] * v else "NO -- the result "
         "is not coming from the Jacobian",
         f"J_perm is the main arm with district columns shuffled; needs "
         f">{THRESH['perm_collapse']}x"),
    )


def answer_q2(R: pd.DataFrame, TA: pd.DataFrame, placement="variance"
              ) -> pd.DataFrame:
    q = R[(R.placement == placement) & (R.window == "post")]
    piv = q.pivot_table(index="sim_hash", columns="method",
                        values="mae")
    need = {"J_pre", "J_post", "J_pre_wpost"}
    if not need <= set(piv.columns):
        return _rows("E8", ("Must a client recommission the Jacobian after a "
                            "regime change?", "post-window MAE", np.nan,
                            "NO DATA", f"needs arms {sorted(need)}"))
    piv = piv.dropna(subset=list(need))
    pen = float((piv.J_pre - piv.J_post).mean())
    comp = float((piv.J_pre_wpost - piv.J_pre).mean())
    op = float((piv.J_post - piv.J_pre_wpost).mean())
    span = float(TA.rho_true.abs().max()) if len(TA) else 1.0
    return _rows(
        "E8",
        ("Must a client recommission the Jacobian after a regime change?",
         "MAE(J_pre) - MAE(J_post), post window", pen,
         ("NO -- commission once" if abs(pen) < THRESH["mae_negligible"]
          else "YES -- recommissioning helps materially"),
         f"{int((piv.J_pre > piv.J_post).sum())}/{len(piv)} worlds worse "
         f"without; judged against a truth spanning |rho| up to {span:.2f} "
         f"and a {THRESH['mae_negligible']} noise floor"),
        ("Which term carries the penalty: composition or operating point?",
         "composition / operating-point split", f"{comp:+.4f} / {op:+.4f}",
         ("operating point" if abs(op) > abs(comp) else "composition"),
         "topology is invariant under drift, so these two are exhaustive; "
         "J_pre_wpost holds the operating point and moves only the weights"),
    )


def answer_q1(G: pd.DataFrame, placement="variance", main_arm="J_pre",
              window="pre") -> pd.DataFrame:
    g = G[(G.placement == placement) & (G.window == window)
          & (G.variant == "baseline")]
    if not len(g):
        return _rows("E9", ("Does the gain hold with a defensible number of "
                            "independent clusters?", "sign test", np.nan,
                            "NO DATA", ""))
    summ, per = cluster_summary(g, "gain")
    p = float(summ.sign_test_p.iloc[0])
    n = int(summ.n_clusters.iloc[0])
    k = int(summ.n_positive.iloc[0])
    return _rows(
        "E9",
        ("Does the gain over Jacobian-free methods hold across clusters?",
         "sign test over (consumption_map, sim_seed)", p,
         "YES" if p < THRESH["cluster_p"] else "NO -- still underpowered",
         f"{k}/{n} clusters positive, mean gain "
         f"{float(summ['mean'].iloc[0]):+.3f}; section 5.3 had 4 clusters "
         f"at p=0.0625"),
        ("What does this NOT establish?", "clusters for Jacobian validity", 1,
         "external validity unchanged",
         "clusters here are maps and seeds; for the Jacobian-validity claim "
         "the cluster is the NETWORK, and that is still n=1 (handoff 9)"),
    )


def answer_q3(R: pd.DataFrame, placement="variance", main_arm="J_pre",
              window="pre") -> pd.DataFrame:
    inc = R[(R.arm_study == "income") & (R.placement == placement)
            & (R.window == window)]
    if not len(inc):
        return _rows("E10", ("Is the level (income) axis recoverable?",
                             "M2 amp_rank_corr", np.nan, "NO DATA",
                             "needs jval_income worlds"))
    a = inc[inc.method == main_arm]
    fr = inc[inc.method.str.startswith("free:")]
    perm = inc[inc.method == "J_perm"]
    rk = float(a.amp_rank_corr.mean())
    sr = float(a.std_r.mean())
    frk = float(fr.amp_rank_corr.max()) if len(fr) else np.nan
    return _rows(
        "E10",
        ("Is the level (income) axis recoverable at all?",
         "M2 amp_rank_corr", rk,
         "YES" if rk >= THRESH["amp_rank_ok"] else "NO",
         f"std_r {sr:.4f} (threshold {THRESH['std_r_ok']}); mean_r is a "
         "global scale factor and is not interpretable"),
        ("Does any Jacobian-free method see level?",
         "best free amp_rank_corr", frk,
         "NO" if not np.isfinite(frk) or frk < THRESH["amp_rank_ok"] else "YES",
         "this is the contrast with the FL path, where per-client MinMax "
         "removes level by construction"),
        ("Is M1 blind on this arm, as pre-registered?",
         "MAE(J_perm) / MAE(main), income arm",
         (float(perm.mae.mean()) / float(a.mae.mean())
          if len(perm) and float(a.mae.mean()) else np.nan),
         "YES -- M1 has no power here" if len(perm) and
         float(perm.mae.mean()) < THRESH["perm_collapse"] * float(a.mae.mean())
         else "NO -- M1 moves, which is method error",
         "land use is constant so rho_true does not move; shuffling the "
         "columns should therefore cost M1 nothing"),
    )


def _vs_random(T, col, ref):
    r = T[T.placement == "random"]
    v = T[T.placement == ref]
    if not len(r) or not len(v):
        return np.nan
    return float(v[col].iloc[0]) - float(r[col].iloc[0])


def answer_placement(R: pd.DataFrame, DG: pd.DataFrame, main_arm="J_pre",
                     window="pre", ref="variance") -> pd.DataFrame:
    T = placement_table(R, DG, main_arm, window)
    if not len(T):
        return _rows("E11", ("Does sensor placement matter?", "MAE spread",
                             np.nan, "NO DATA", ""))
    col = f"mae_{main_arm}"
    best, worst = T.iloc[0], T.iloc[-1]
    spread = float(worst[col] - best[col])
    rnd = T[T.placement == "random"]
    rk = placement_ranking(R, DG, main_arm, window)

    def rho_of(target):
        if not len(rk) or "target" not in rk.columns:
            return np.nan
        r = rk[rk["target"] == target]
        return float(r.spearman_vs_sigma_min.iloc[0]) if len(r) else np.nan

    rows = [
        ("Does sensor placement change recovery on this network?",
         f"MAE spread across placements ({best.placement} -> {worst.placement})",
         spread,
         ("NO -- all placements equivalent"
          if spread < THRESH["placement_equiv"] else "YES"),
         f"threshold {THRESH['placement_equiv']} (the ~0.02 noise floor of a "
         "Spearman over 10 pairs)"),
        ("Is a principled placement better than a random one?",
         f"MAE({ref}) - MAE(random)", _vs_random(T, col, ref),
         ("no -- placement is not the binding constraint"
          if np.isfinite(_vs_random(T, col, ref) or np.nan)
          and abs(_vs_random(T, col, ref)) < THRESH["placement_equiv"]
          else "yes" if np.isfinite(_vs_random(T, col, ref) or np.nan)
          else "NO DATA -- no random placement in the sweep"),
         "if random ties the designed strategies, the placement literature "
         "the study started from answers a question that does not bind here"),
        ("Does sigma_min order the LEVEL of recovery? (6.7 says no)",
         f"Spearman(sigma_min, {col}) across placements", rho_of(col),
         "confirmed: does not order level"
         if abs(rho_of(col)) < 0.6 or not np.isfinite(rho_of(col))
         else "orders level too",
         "expected null; 4-5 points, so a direction not a result"),
        ("Does sigma_min order ROBUSTNESS to Jacobian error? (6.7 says yes)",
         "Spearman(sigma_min, misspec_penalty)", rho_of("misspec_penalty"),
         ("confirmed: higher sigma_min, smaller penalty"
          if rho_of("misspec_penalty") < -0.6
          else "not confirmed on these placements"),
         "misspec_penalty = MAE(J_inp) - MAE(main): the cost of the wrong "
         "node weights at the wrong operating point"),
        ("Which placement to deploy?", "recommended", best.placement,
         best.placement,
         "cheapest that is not measurably worse; `variance` is purely local, "
         "needs no Jacobian and no coordination to compute"),
    ]
    return _rows("E11", *rows)


def answer_pairs(D: pd.DataFrame, floor: float) -> pd.DataFrame:
    nd, sd = D[~D.involves_drifter], D[D.predicted != 0]
    ih = float(nd.d_hat.abs().mean()) if len(nd) else np.nan
    sh = (float((np.sign(sd.d_hat) == sd.predicted).mean())
          if len(sd) else np.nan)
    st = (float((np.sign(sd.d_true) == sd.predicted).mean())
          if len(sd) else np.nan)
    return _rows(
        "E12",
        ("Does the estimator leave undisturbed pairs undisturbed?",
         "mean |d rho_hat|, non-drifter pairs", ih,
         "YES" if ih < floor else "NO -- districts are not being isolated",
         f"floor {floor:.4f} taken from the truth in E4; "
         f"truth moves {float(nd.d_true.abs().mean()):.4f}"),
        ("Does it recover the SIGNED change a known intervention causes?",
         "sign agreement, drifter pairs", sh,
         "YES -- causal, not correlational" if sh >= THRESH["sign_ok"]
         else "NO -- claim withdrawn, recovery stays correlational",
         f"ceiling from truth is {st:.3f}; effect size "
         f"|d rho_true| = {float(sd.d_true.abs().mean()):.3f}"),
    )


def answer_beta(R: pd.DataFrame, EX: pd.DataFrame, placement="variance",
                main_arm="J_pre", window="post") -> pd.DataFrame:
    b = R[(R.placement == placement) & (R.window == window)
          & (R.method == main_arm)]
    per = b.groupby("beta")["mae"].mean().sort_index()
    exc = (float(EX.excursion.max()) if isinstance(EX, pd.DataFrame)
           and len(EX) and "excursion" in EX else np.nan)
    reach = np.isfinite(exc) and exc >= THRESH["excursion_min"]
    mono = (bool(np.all(np.diff(per.to_numpy()) >= -1e-9))
            if len(per) >= 3 else None)
    return _rows(
        "E13",
        ("Does beta move the operating point enough to test the mechanism?",
         "max post/pre demand ratio across the ladder", exc,
         "YES" if reach else "NO -- arm underpowered by construction",
         f"threshold {THRESH['excursion_min']}; only one district in five "
         "drifts, so the excursion is bounded regardless of beta"),
        ("Is degradation monotone in beta?",
         "MAE by beta", " / ".join(f"{k}:{v:.4f}" for k, v in per.items()),
         ("yes" if mono else "no" if mono is not None else "NO DATA")
         + ("" if reach else " (but see above: not interpretable)"),
         "matched cell only (District_D, seed 42); a flat curve with no "
         "reach is evidence about the dial, not about the mechanism"),
    )


def answer_coupling(G: pd.DataFrame, paired, placement="variance",
                    main_arm="J_pre", window="pre") -> pd.DataFrame:
    g = G[(G.window == window) & (G.placement == placement)]
    if paired is not None:
        g = g[g.sim_hash.isin(paired)]
    iso, bas = g[g.variant == "isolated"], g[g.variant == "baseline"]
    if not len(iso) or not len(bas):
        return _rows("E14", ("Is the gain attributable to hydraulic mixing?",
                             "isolated vs baseline gain", np.nan, "NO DATA",
                             "needs paired jval_coupling worlds"))
    d_un = float(iso[main_arm].mean() - bas[main_arm].mean())
    d_fr = float(iso.best_free.mean() - bas.best_free.mean())
    gi, gb = float(iso.gain.mean()), float(bas.gain.mean())
    premise = abs(d_fr) > abs(d_un)
    return _rows(
        "E14",
        ("Does the gain vanish when there is no mixing to undo?",
         "gain: baseline -> isolated", f"{gb:+.3f} -> {gi:+.3f}",
         "YES" if gi < 0.5 * gb else "NO",
         f"n={len(bas)} baseline vs {len(iso)} isolated, paired cells only"),
        ("Which term moved -- the estimator or the baseline?",
         "delta(unmixed) / delta(best_free) across the dial",
         f"{d_un:+.3f} / {d_fr:+.3f}",
         ("the BASELINE moved: P5's premise failed, not the method"
          if premise else "the estimator moved: the gain is mixing-dependent"),
         "a pipe is not a district-demand meter even without inter-district "
         "mixing -- it carries a routing-signed subset of its own district, "
         "and a per-district reservoir changes that routing"),
        ("So what is the gain attributable to?", "interpretation",
         "pipe->district inversion" if premise else "hydraulic unmixing",
         ("broader than inter-district unmixing" if premise
          else "inter-district unmixing, as claimed"),
         "if the premise failed, section 5's attribution needs narrowing and "
         "6.5 has a second reason to be ill-posed"),
    )


def answer_windows(R: pd.DataFrame, placement="variance", main_arm="J_pre"
                   ) -> pd.DataFrame:
    order = ["pre", "transition", "post"]
    q = R[(R.placement == placement) & (R.method == main_arm)]
    piv = q.pivot_table(index="sim_hash", columns="window", values="mae")
    if not set(order) <= set(piv.columns):
        return _rows("E15", ("Is the transition window the worst?", "share",
                             np.nan, "NO DATA", ""))
    piv = piv[order]
    frac = float((piv.idxmax(axis=1) == "transition").mean())
    ties = q.pivot_table(index="sim_hash", columns="window",
                         values="n_true_ties")[order].mean()
    return _rows(
        "E15",
        ("Does recovery dip while a district is mid-conversion?",
         "share of worlds where transition is worst on MAE", frac,
         "YES" if frac > 0.5 else "NO -- composition is not the dominant "
         "failure mode",
         "mid-conversion breaks the proportional-weights assumption, which "
         "is the spatially coherent node_weights error that was the only "
         "misspecification to measurably hurt recovery"),
        ("Is the rank_corr window ordering trustworthy?",
         "mean n_true_ties by window",
         " / ".join(f"{k}:{v:.1f}" for k, v in ties.items()),
         ("no -- ties drive it" if ties["transition"] < ties["pre"] - 1
          else "yes"),
         "ties drop mid-conversion, which inflates Spearman there even when "
         "MAE says recovery is worse; MAE is primary for this reason"),
    )


# --------------------------------------------------------------------------
# 12. the pre-registered scoreboard (handoff 7.8)
# --------------------------------------------------------------------------
def predictions(R: pd.DataFrame, Pr: pd.DataFrame, main_arm="J_pre",
                placement="variance", noise_floor=0.02, p1_target=0.05,
                paired_cells=None, excursion=np.nan) -> pd.DataFrame:
    """Score P1..P7. Written before the numbers existed, on purpose."""
    out = []

    def add(pid, claim, value, verdict, note=""):
        out.append(dict(P=pid, claim=claim, value=value, verdict=verdict,
                        note=note))

    q = R[(R.placement == placement) & (R.method == main_arm)]

    # P1 -- pre-window MAE reproduces the pinned table
    #
    # The prediction was written as "reproduces ~0.05". It is one-sided:
    # BEATING the pinned figure is not a failure, and if J_inp is present the
    # J_inp-vs-main_arm gap says how much of section 5's 0.052 was node-weight
    # and operating-point misspecification rather than method error.
    v = q.loc[q.window == "pre", "mae"].mean()
    qi = R[(R.placement == placement) & (R.method == "J_inp") &
           (R.window == "pre")]
    extra = (f" J_inp {qi.mae.mean():.4f} -> misspecification accounts for "
             f"{qi.mae.mean() - v:+.4f} of it." if len(qi) else "")
    add("P1", f"pre-window MAE <= ~{p1_target}", round(float(v), 4),
        "PASS" if np.isfinite(v) and v < 2 * p1_target else "FAIL",
        "failure implicates the windowing, not the method." + extra)

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
        add("P3", f"|d rho_hat| non-drifter pairs < {noise_floor:.4f}",
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
    #
    # Scored on the PAIRED coupling cells only, and decomposed: a gain is a
    # difference of two terms, and the first run showed the movement was
    # entirely in the free-method term, not in the unmixed one. So report
    # both. If unmixed recovery is flat across the dial while the free
    # methods collapse on isolated worlds, the premise "no mixing => raw
    # already works" is what failed, not the method -- a pipe is not a
    # district-demand meter even with no inter-district mixing, because it
    # carries a routing-signed SUBSET of its own district.
    G = gain_table(R, arm=main_arm)
    if len(G) and "variant" in G.columns:
        g = G[(G.placement == placement) & (G.window == "pre")]
        if paired_cells is not None:
            g = g[g.sim_hash.isin(paired_cells)]
        iso = g[g.variant == "isolated"]
        base = g[g.variant == "baseline"]
        if len(iso) and len(base):
            d_un = iso[main_arm].mean() - base[main_arm].mean()
            d_fr = iso["best_free"].mean() - base["best_free"].mean()
            ok = iso.gain.mean() < 0.5 * base.gain.mean()
            add("P5", "isolated gain ~ 0 while baseline gain > 0",
                f"iso {iso.gain.mean():+.3f} / base {base.gain.mean():+.3f}",
                "PASS" if ok else "FAIL",
                f"n={len(iso)}v{len(base)}. decomposition: unmixed moves "
                f"{d_un:+.3f}, best_free moves {d_fr:+.3f} across the dial. "
                "If |free| >> |unmixed| the PREMISE failed, not the method: "
                "the gain is pipe->district inversion, of which inter-"
                "district unmixing is only one part.")
        else:
            add("P5", "isolated vs baseline gain", np.nan,
                "NO DATA", "needs paired jval_coupling worlds")
    else:
        add("P5", "isolated vs baseline gain", np.nan, "NO DATA")

    # P6 -- monotone degradation in beta, on the MATCHED cell only
    #
    # The beta arm is one drift origin (District_D) at one seed (42): the
    # only thing allowed to vary is beta. Aggregating every seed-42 world
    # into the 0.35 cell mixes five drift districts against two single-world
    # cells and the comparison means nothing.
    b = q[(q.window == "post") & q.beta.notna()
          & (q.drift_district == "District_D") & (q.sim_seed == 42)
          & (q.consumption_map == "LR_LR_LR_LR_LI")
          & (q.variant == "baseline")]
    per = b.groupby("beta")["mae"].agg(["mean", "size"]).sort_index()
    if len(per) >= 3:
        v = per["mean"].to_numpy()
        mono = bool(np.all(np.diff(v) >= -1e-9))
        add("P6", "MAE monotone increasing in beta",
            " / ".join(f"{k}:{r['mean']:.4f}(n{int(r['size'])})"
                       for k, r in per.iterrows()),
            "PASS" if mono else "FAIL",
            f"excursion {excursion:.3f}x pre-window demand. Below ~1.1x "
            "there is no operating-point movement for curvature to act on "
            "and the arm is underpowered BY CONSTRUCTION, whichever way it "
            "comes out -- a flat result is not evidence against the "
            "mechanism, it is evidence the dial has no reach here.")
    else:
        add("P6", "monotone in beta", np.nan, "NO DATA",
            f"{len(per)} beta levels on the matched District_D/seed-42 cell")

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
# 12b. explorer support: one loaded world, clustering, and its controls
# --------------------------------------------------------------------------
def district_adjacency(wn, districts: dict) -> pd.DataFrame:
    """Districts sharing at least one pipe.

    Adjacency is the standing confound: in the federated-artifact sweep it
    explained prototype and latent structure better than the regime labels
    did. Clustering therefore has to be scored against BOTH label sets --
    high agreement with regime AND low agreement with adjacency is the
    de-confounding claim; the reverse is the confound reasserting itself.
    """
    ds = sorted(districts["districts"])
    n2d = {str(n): d for d, nn in districts["districts"].items() for n in nn}
    A = pd.DataFrame(False, index=ds, columns=ds)
    for name in wn.link_name_list:
        lk = wn.get_link(name)
        a = n2d.get(str(lk.start_node_name))
        b = n2d.get(str(lk.end_node_name))
        if a and b and a != b:
            A.loc[a, b] = A.loc[b, a] = True
    return A


def ari(a, b) -> float:
    """Adjusted Rand index, implemented locally to avoid a sklearn import."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return np.nan
    ct = pd.crosstab(a, b).to_numpy()
    comb = lambda x: x * (x - 1) / 2.0                        # noqa: E731
    sij = comb(ct).sum()
    sa = comb(ct.sum(1)).sum()
    sb = comb(ct.sum(0)).sum()
    exp = sa * sb / comb(n)
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def cluster_labels(rho: pd.DataFrame, k=2, method="average") -> np.ndarray:
    """Agglomerative clustering on 1 - |rho|.

    |rho| because a loop-differential sensor pair can be strongly ANTI-
    correlated while carrying the same information; sign is geometry, not
    regime.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = 1.0 - rho.abs().to_numpy()
    np.fill_diagonal(d, 0.0)
    d = 0.5 * (d + d.T)
    Z = linkage(squareform(d, checks=False), method=method)
    return fcluster(Z, t=k, criterion="maxclust")


def regime_labels(w: pd.Series, districts: dict, window="pre",
                  axis="land_use") -> tuple[np.ndarray, list]:
    m = state_map(w, districts, window, axis=axis)
    ds = sorted(districts["districts"])
    return np.array([m[d] for d in ds]), ds


def adjacency_labels(A: pd.DataFrame, k=2, method="average") -> np.ndarray:
    """Cluster the districts by hydraulic proximity alone -- the confound's
    own answer, to score against."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    import scipy.sparse.csgraph as csg
    hops = csg.shortest_path(A.to_numpy().astype(float), unweighted=True)
    hops[~np.isfinite(hops)] = A.shape[0] + 1.0
    np.fill_diagonal(hops, 0.0)
    Z = linkage(squareform(0.5 * (hops + hops.T), checks=False), method=method)
    return fcluster(Z, t=k, criterion="maxclust")


# ---- naming: one place that decides how a world is described -----------
EXP_SHORT = {"s1_oddone": "oddone", "s2_split": "split", "income": "income",
             "beta": "beta", "coupling": "coupling",
             "s1_equalised": "equalised", "other": "other"}


def drift_states(w: pd.Series, districts: dict | None = None) -> tuple:
    """(initial, final) two-char regime token of the DRIFTING client.

    Position 0 is income {L,M,H}, position 1 land use {R,M,C,I}. The initial
    token is read out of the commissioning consumption_map at the drifter's
    index; the final one is assembled from the drift target.
    """
    ds = (sorted(districts["districts"]) if districts
          else [f"District_{c}" for c in "ABCDE"])
    toks = str(w["consumption_map"]).split("_")
    k = ds.index(str(w["drift_district"]))
    final = (str(w["drift_to_income"])[0] + str(w["drift_to_land_use"])[0]).upper()
    return toks[k], final


def map_str(w: pd.Series, districts: dict | None = None,
            window="pre", mark=True) -> str:
    """The consumption map for a window, with the drifter's token bracketed."""
    ds = (sorted(districts["districts"]) if districts
          else [f"District_{c}" for c in "ABCDE"])
    toks = list(str(w["consumption_map"]).split("_"))
    k = ds.index(str(w["drift_district"]))
    if window == "post":
        toks[k] = drift_states(w, districts)[1]
    if mark:
        toks[k] = f"[{toks[k]}]"
    return "_".join(toks)


def world_label(w: pd.Series, districts: dict | None = None) -> str:
    """CLIENT_INIT_FINAL__MAPPING__HASH6_EXPERIMENT.

    e.g. E_LI_LR__LI_LI_LR_LR_LI__dqw34q_oddone -- district E drifting from
    low-industrial to low-residential, on that commissioning map. Sorts
    usefully: same drifter and same transition group together.
    """
    init, final = drift_states(w, districts)
    exp = EXP_SHORT.get(str(w.get("arm")), str(w.get("arm")))
    return (f"{str(w['drift_district'])[-1]}_{init}_{final}"
            f"__{w['consumption_map']}"
            f"__{str(w['sim_hash'])[:6]}_{exp}")


def plot_title(w: pd.Series, districts: dict | None = None, extra="") -> str:
    """Two lines: who drifts and how, then the map before and after."""
    init, final = drift_states(w, districts)
    c = str(w["drift_district"])[-1]
    exp = EXP_SHORT.get(str(w.get("arm")), str(w.get("arm")))
    beta = w.get("beta")
    l1 = (f"{w['drift_district']} ({c}):  {init} \u2192 {final}"
          f"   |   {exp}  \u00b7  seed {w.get('sim_seed')}"
          f"  \u00b7  beta {beta}  \u00b7  {w.get('variant')}"
          f"  \u00b7  {str(w['sim_hash'])[:6]}")
    l2 = (f"map  {map_str(w, districts, 'pre')}  \u2192  "
          f"{map_str(w, districts, 'post')}")
    return l1 + "\n" + l2 + (("\n" + extra) if extra else "")


class WorldView:
    """One world, loaded once, with every signal the explorer needs.

    Deliberately eager on the cheap things (demand, flows, windows) and lazy
    on the expensive one (the perturbation sweep), so a widget can re-render
    on a slider without re-solving hydraulics.
    """

    def __init__(self, w: pd.Series, districts: dict, placement="variance",
                 arms=ARMS_DEFAULT, delta_frac=0.5, rcond=1e-3, seed=0):
        self.w, self.districts = w, districts
        self.ds = sorted(districts["districts"])
        self.rcond = rcond
        self.art = P.load_world(w["dir"])
        self.dem = _load(w["dir"], "demand")
        flows = self.art.get("flows")
        if flows is None:
            flows = _load(w["dir"], "flows")
        self.n = min(len(self.dem), len(flows))
        self.dem = self.dem.iloc[:self.n]
        self.months = month_vector(self.art, self.dem, self.n)
        self.masks = window_masks(self.months)
        self.Y = flows.iloc[:self.n][
            [c for c in flows.columns if c != "month"]].reset_index(drop=True)
        self.D_true = district_demand(self.dem, districts)
        self.placement = placement
        self.pl = placements(self.art, names=(placement,), seed=seed)[placement]
        self.sensors = [e for e in flatten(self.pl) if e in self.Y.columns]
        self.anchor = float(w.get("anchor_scale", 1.0) or 1.0)
        self.Jd, self.diag = jd_arms(
            self.art, districts, self.sensors, self.dem, self.masks, arms=arms,
            delta_frac=delta_frac, seed=seed, anchor_scale=self.anchor)
        self.A = district_adjacency(self.art["wn"], districts)

    # ---- masks ----------------------------------------------------------
    def mask(self, key="pre") -> np.ndarray:
        if key in self.masks:
            return self.masks[key]
        if key == "all":
            return np.ones(self.n, bool)
        return self.months == int(key)                       # a single month

    # ---- signals --------------------------------------------------------
    def unmixed(self, arm="J_pre", key="pre") -> pd.DataFrame:
        m = self.mask(key)
        return unmix(self.Y[m], self.Jd[arm], rcond=self.rcond).reindex(
            columns=self.ds)

    def raw(self, key="pre") -> pd.DataFrame:
        m = self.mask(key)
        return free_estimates(self.Y[m], self.pl)["raw"].reindex(
            columns=self.ds)

    def truth(self, key="pre") -> pd.DataFrame:
        return self.D_true.loc[np.flatnonzero(self.mask(key))]

    def signal(self, kind="unmixed", arm="J_pre", key="pre") -> pd.DataFrame:
        return dict(truth=lambda: self.truth(key),
                    unmixed=lambda: self.unmixed(arm, key),
                    raw=lambda: self.raw(key))[kind]()

    # ---- forward and inverse checks -------------------------------------
    def reconstruct(self, arm="J_pre", key="pre") -> tuple:
        """y_hat = Jd * D_true against the measured y, on the same window.

        The FORWARD direction. It tests the model itself rather than the
        inverse: if y_hat tracks y, then `y = Jd D` holds on this window and
        the proportional-weights assumption is satisfied. A forward failure
        localises the problem to the Jacobian or the weights, not to pinv.
        """
        Jd = self.Jd[arm]
        m = self.mask(key)
        cols = [c for c in Jd.index if c in self.Y.columns]
        act = self.Y[m][cols]
        act = act - act.mean()
        Dt = self.truth(key)
        pred = pd.DataFrame(Dt[self.ds].to_numpy()
                            @ Jd.loc[cols, self.ds].to_numpy().T,
                            index=act.index, columns=cols)
        pred = pred - pred.mean()
        return pred, act

    def fit_report(self, arm="J_pre", key="pre") -> pd.DataFrame:
        """Per-sensor forward fit and per-district inverse fit, one table."""
        # column names avoid `corr` and `name`, which shadow DataFrame
        # attributes and turn `fr.corr` into a bound method
        pred, act = self.reconstruct(arm, key)
        rows = [dict(kind="sensor", element=c,
                     fit_corr=float(np.corrcoef(pred[c], act[c])[0, 1]),
                     sd_ratio=float(act[c].std() / pred[c].std())
                     if pred[c].std() else np.nan)
                for c in act.columns]
        Dt, Dh = self.truth(key), self.unmixed(arm, key)
        Dh.index = Dt.index
        for d in self.ds:
            rows.append(dict(kind="district", element=d,
                             fit_corr=float(np.corrcoef(Dh[d], Dt[d])[0, 1]),
                             sd_ratio=float(Dh[d].std() / Dt[d].std())
                             if Dt[d].std() else np.nan))
        return pd.DataFrame(rows)

    # ---- clustering -----------------------------------------------------
    def rho(self, kind="unmixed", arm="J_pre", key="pre") -> pd.DataFrame:
        F = self.signal(kind, arm, key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.DataFrame(np.corrcoef(F[self.ds].to_numpy().T),
                                index=self.ds, columns=self.ds).fillna(0.0)

    def cluster_scores(self, kind="unmixed", arm="J_pre", key="pre", k=2,
                       axis="land_use", method="average") -> dict:
        """ARI of the clustering against the regime labels AND against
        adjacency. The de-confounding claim is high on the first and low on
        the second; the confound reasserting itself is the reverse."""
        win = key if key in self.masks else (
            "pre" if str(key).isdigit() and int(key) < 6 else
            "post" if str(key).isdigit() and int(key) >= 36 else "transition")
        lab = cluster_labels(self.rho(kind, arm, key), k=k, method=method)
        reg, _ = regime_labels(self.w, self.districts, win, axis=axis)
        adj = adjacency_labels(self.A, k=k, method=method)
        # Score against BOTH label sets. Mid-conversion the drifting district
        # belongs to neither, so agreement with the commissioning labels
        # falls while agreement with the post-drift labels rises: the
        # crossing IS the front advancing, and a single label set hides it.
        pre_l, _ = regime_labels(self.w, self.districts, "pre", axis=axis)
        post_l, _ = regime_labels(self.w, self.districts, "post", axis=axis)
        return dict(kind=kind, arm=arm, window=key, k=k,
                    ari_regime=ari(lab, reg), ari_adjacency=ari(lab, adj),
                    ari_regime_pre=ari(lab, pre_l),
                    ari_regime_post=ari(lab, post_l),
                    labels="".join(map(str, lab)),
                    regime="".join(map(str, reg)))


def cluster_scan(view: WorldView, arm="J_pre", k=2, kinds=("truth", "unmixed",
                                                           "raw"),
                 axis="land_use", months=None) -> pd.DataFrame:
    """Month-by-month clustering agreement. The dynamic view of the claim.

    A drifting district should MIGRATE between clusters as its front
    advances, so ari_regime traces the conversion when it is measured on a
    signal that can see regime, and stays flat when it cannot.
    """
    months = range(int(view.months.max()) + 1) if months is None else months
    rows = []
    for mo in months:
        if (view.months == mo).sum() < 48:
            continue
        for kind in kinds:
            try:
                r = view.cluster_scores(kind, arm, str(mo), k=k, axis=axis)
                rows.append(dict(month=mo, **r))
            except Exception:                                 # noqa: BLE001,S112
                continue
    return pd.DataFrame(rows)


def clustering_sanity(worlds: pd.DataFrame, districts: dict,
                      placement="variance", arm="J_pre", k=2, key="pre",
                      axis="land_use", arms=("J_pre", "J_perm"),
                      progress=5) -> pd.DataFrame:
    """The controls, across the pinned set. Four rows per world:

      truth    -- the ceiling; regime is recoverable at all
      unmixed  -- the claim
      raw      -- the Jacobian-free baseline, i.e. the confounded signal
      J_perm   -- unmixing with the district columns shuffled; MUST collapse

    If `unmixed` beats `raw` on ari_regime while not beating it on
    ari_adjacency, the physics confound has been removed rather than
    relabelled.
    """
    rows = []
    for i, (_, w) in enumerate(worlds.iterrows(), 1):
        try:
            v = WorldView(w, districts, placement=placement, arms=arms)
            for kind, a in (("truth", arm), ("unmixed", arm), ("raw", arm),
                            ("unmixed", "J_perm")):
                if a not in v.Jd:
                    continue
                r = v.cluster_scores(kind, a, key, k=k, axis=axis)
                rows.append(dict(sim_hash=w["sim_hash"], arm_study=w.get("arm"),
                                 variant=w.get("variant"),
                                 signal=kind if a == arm else f"{kind}:{a}",
                                 **{kk: vv for kk, vv in r.items()
                                    if kk not in ("kind", "arm")}))
        except Exception as e:                                # noqa: BLE001
            print(f"  !! {w['sim_hash'][:8]}: {type(e).__name__}: {e}")
        if progress and i % progress == 0:
            print(f"  {i}/{len(worlds)}")
    return pd.DataFrame(rows)


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
