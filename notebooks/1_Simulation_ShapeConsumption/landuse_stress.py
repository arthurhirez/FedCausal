"""Stress test for the land-use POC — find the combinations that break.

Run inside the POC notebook so it inherits its namespace:

    %run -i landuse_stress.py
    results = stress_sweep()                 # ~10-20 min at default grid
    report(results)

Design
------
Each run is a *worst-case envelope*, not a timeline: the drift front is forced to
convert the whole target district immediately (growth_chance=1, no warm-up), the
seasonal peak is pinned to the final month, and only that final month is scored.
Three 7-day months is therefore enough, which is what makes a large grid affordable.

For each (map, drift, anchor) combination the sweep walks beta upward and stops at
the first failure, recording the largest beta that still passes. Passing combos
cost the full ladder; failing ones short-circuit.
"""
import copy
import itertools
import time as _time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- envelope cfg
# warmup_months must be >= 1: apply_drift_ramp blends against the month BEFORE the
# switch, so a node drifting at month 0 has nothing to blend from (KeyError).
# n_months must leave room for the drift front to cross the district: the frontier
# advances one BFS ring per month regardless of max_neighbors, so the horizon has to
# exceed the subgraph radius or the envelope is only partially drifted.
STRESS = dict(n_months=6, days_per_month=7, resolution_h=1,
              warmup_months=1, drift_ramp_days=5,
              seasonality_scale=1.0, sim_seed=42)

BAND = (10.0, 50.0)      # ABNT NBR 12218
HARD_FLOOR = 5.0
CALIB_BAND = (0.5, 2.0)


# --------------------------------------------------------------------- runner
def run_stress(map_code, drifts, beta, anchor_scale=0.05, cfg=STRESS):
    """One envelope world. Returns a dict consumed by check_world."""
    assert cfg["warmup_months"] >= 1, "warmup_months=0 breaks apply_drift_ramp"
    p = copy.deepcopy(PARAMS)
    time = {k: cfg[k] for k in ("n_months", "days_per_month", "resolution_h")}
    hyd = {**p["hydraulics"], "anchor_scale": anchor_scale}
    pat = {**p["patterns"],
           "drift_ramp_days": cfg["drift_ramp_days"],
           "seasonality_scale": cfg["seasonality_scale"],
           "seasonal_peak_month": cfg["n_months"] - 1}      # worst month is the last
    scen = {**p["scenario"], "n_months": cfg["n_months"],
            "income_land_use_mapping": decode_landuse_map(map_code)}

    wn = configure_network(copy.deepcopy(WN_RAW), hyd, time)
    wn, _ = apply_coupling(wn, DISTRICTS, p["coupling"], seed=cfg["sim_seed"])

    lf = build_landuse_factors(beta)
    pf = poc_build_portfolios(wn, DISTRICTS, lf, scen, hyd, cfg["sim_seed"])

    sched = pd.DataFrame([{"node": n, "drift_month": cfg["warmup_months"],
                           "to_income": d["to_income"], "to_land_use": d["to_land_use"]}
                          for d in drifts
                          for n in DISTRICTS["districts"][d["tgt_district"]]],
                         columns=["node", "drift_month", "to_income", "to_land_use"])

    if sched is None:                                        # no-drift baseline
        sched = pd.DataFrame(columns=["node", "drift_month", "to_income", "to_density"])


    tl = poc_evolve_assignments(pf, sched, lf, scen)
    dem = poc_synthesize_demands(tl, pat, time, cfg["sim_seed"])
    if len(sched):
        dem = apply_drift_ramp(dem, sched, pat, time)
    pres, _, dsim = run_hydraulics(wn, dem)

    tgt_nodes = {n for d in drifts for n in DISTRICTS["districts"][d["tgt_district"]]}
    last = cfg["n_months"] - 1
    conv = (len(set(sched["node"]) & tgt_nodes) / max(len(tgt_nodes), 1)) if len(sched) else 1.0
    return dict(dem=dem, pres=pres, dsim=dsim, calib=float(pf["calibration"].iloc[0]),
                last=last, drifted_frac=conv,
                front_done=bool(len(sched) == 0 or sched["drift_month"].max() <= last))


# --------------------------------------------------------------------- checks
def check_world(w):
    """V1-V4 style checks scored on the worst (final, fully drifted, seasonal peak) month."""
    dem, pres, dsim, calib, last = w["dem"], w["pres"], w["dsim"], w["calib"], w["last"]
    # pressure is checked at every district junction, including zero-demand trunk
    # nodes; mass/volume only where a demand column exists.
    pj = [c for c in pres.columns if c in NODE2DIST]
    dj = [c for c in pj if c in dem.columns and c in dsim.columns]
    sel = dem["month"].to_numpy() == last
    v = pres.loc[sel, pj].to_numpy()
    lo, hi = BAND
    inside = float(((v >= lo) & (v <= hi)).mean())

    d_j = dsim[dj].to_numpy()
    supplied = -dsim.drop(columns=["month"], errors="ignore").to_numpy().clip(max=0).sum()
    consumed = d_j.clip(min=0).sum()
    mass_err = abs(supplied - consumed) / max(consumed, 1e-9)
    dd_err = float(np.abs(d_j - dem[dj].to_numpy()).max()
                   / max(np.abs(dem[dj].to_numpy()).max(), 1e-9))

    fails = []
    if not np.isfinite(v).all():          fails.append("nonfinite_pressure")
    if v.min() < HARD_FLOOR:              fails.append("V3_floor")
    if inside < 0.98:                     fails.append("V4_band")
    if v.max() > hi:                      fails.append("overpressure")
    if mass_err > 1e-4:                   fails.append("V1_mass")
    if dd_err > 1e-3:                     fails.append("V2_dd_mismatch")
    # if not CALIB_BAND[0] <= calib <= CALIB_BAND[1]:
    #     fails.append("calibration")

    return {"pmin": float(v.min()), "pmax": float(v.max()), "inside_%": 100 * inside,
            "drifted_%": 100 * w["drifted_frac"], "front_done": w["front_done"],
                        "peak_lps": float(dem.loc[sel, dj].sum(axis=1).max()),
            "mean_lps": float(dem.loc[sel, dj].sum(axis=1).mean()),
            "calibration": calib, "mass_err": float(mass_err), "dd_err": dd_err,
            "passed": not fails, "fail": ",".join(fails)}


# ----------------------------------------------------------------------- grid
LU = {"R": "residential", "M": "mixed", "C": "commercial", "I": "industrial"}
DISTRICT_NAMES = None          # filled at call time from DISTRICTS


def random_maps(n, seed=0, incomes="LLLMH"):
    rng = np.random.default_rng(seed)
    return ["_".join(rng.choice(list(incomes)) + rng.choice(list(LU)) for _ in range(5))
            for _ in range(n)]


CURATED_MAPS = [
    "LR_LR_LR_LR_LR",      # all residential — the soft baseline
    "LR_LM_LC_LR_LR",      # the demo map
    "LR_LR_LR_LM_LR",      # target starts mixed
    "LR_LR_LR_LC_LR",      # target starts commercial
    "LC_LC_LC_LC_LC",      # city-wide commercial
    "LI_LI_LI_LI_LI",      # city-wide industrial — expected to break first
    "LM_LM_LM_LM_LM",
    "HR_HR_HR_HR_HR",      # high income everywhere
    "LR_MR_HR_LR_MR",      # income gradient, residential land use
    "LI_LR_LR_LR_LI",      # industry on the edges
]


def default_grid(n_random=6, betas=(0.0, 0.25, 0.5, 0.75, 1.0),
                 anchors=(0.05,), to_incomes=("low",)):
    names = list(DISTRICTS["districts"])
    targets = ([[n] for n in names]                                  # each district alone
               + [[names[0], names[3]]]                              # two clients
               + [names])                                            # every client at once
    to_lu = ["mixed", "commercial", "industrial"]
    maps = CURATED_MAPS + random_maps(n_random, seed=7)
    return dict(maps=maps, targets=targets, to_lu=to_lu, to_incomes=list(to_incomes),
                betas=list(betas), anchors=list(anchors))


# ---------------------------------------------------------------------- sweep
def stress_sweep(grid=None, cfg=STRESS, max_combos=None, csv="stress_results.csv",
                 verbose=True):
    """Beta ladder over every (map, target set, land use, income, anchor) combination.

    Ascending beta, stop at first failure. Every rung is recorded, so the output
    contains both the passing envelope and the exact rung that broke.
    """
    grid = grid or default_grid()
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
                row |= check_world(run_stress(mp, drifts, beta, anc, cfg))
            except Exception as e:
                row |= {"passed": False, "fail": f"EXC:{type(e).__name__}",
                        "error": str(e)[:160]}
            rows.append(row)
            if not row["passed"]:
                break                                    # ladder short-circuits
        if verbose and (k + 1) % 10 == 0:
            el = _time.time() - t0
            print(f"{k+1}/{len(combos)} combos  {el/60:.1f} min  "
                  f"eta {(el/(k+1))*(len(combos)-k-1)/60:.1f} min", flush=True)
        if csv and (k + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    if csv:
        df.to_csv(csv, index=False)
    return df


def baselines(grid=None, cfg=STRESS, anchor=0.05):
    """Does the initial map alone survive, before any drift? Run this first."""
    grid = grid or default_grid()
    rows = []
    for mp in grid["maps"]:
        for beta in sorted(grid["betas"]):
            row = {"map": mp, "beta": beta, "anchor_scale": anchor}
            try:
                row |= check_world(run_stress(mp, [], beta, anchor, cfg))
            except Exception as e:
                row |= {"passed": False, "fail": f"EXC:{type(e).__name__}"}
            rows.append(row)
            if not row["passed"]:
                break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- report
def report(df):
    """Where is the feasibility boundary, and what breaks first?"""
    key = ["map", "targets", "to_land_use", "to_income", "anchor_scale"]
    passing = df[df["passed"]].groupby(key)["beta"].max().rename("beta_max_pass")
    failing = (df[~df["passed"]].sort_values("beta").groupby(key)
               .first()[["beta", "fail"]].rename(columns={"beta": "beta_first_fail"}))
    edge = passing.to_frame().join(failing, how="outer").reset_index()
    edge["beta_max_pass"] = edge["beta_max_pass"].fillna(-1)     # -1 = fails even at beta 0

    print(f"runs {len(df)} | combos {len(edge)} | "
          f"never pass {int((edge['beta_max_pass'] < 0).sum())} | "
          f"pass at every beta {int(edge['beta_first_fail'].isna().sum())}\n")

    print("failure modes:")
    print(df.loc[~df["passed"], "fail"].value_counts().to_string(), "\n")
    if "error" in df.columns and df["error"].notna().any():
        print("exception messages (top 5):")
        print(df["error"].dropna().value_counts().head().to_string(), "\n")
    if "drifted_%" in df.columns and (df["drifted_%"] < 99).any():
        n = int((df["drifted_%"] < 99).sum())
        print(f"WARNING: {n} runs were not fully drifted at the scored month — "
              f"raise STRESS['n_months'].\n")

    print("median beta_max_pass by target land use:")
    print(edge.groupby("to_land_use")["beta_max_pass"].agg(["median", "min", "count"])
          .round(2).to_string(), "\n")

    print("median beta_max_pass by number of drifting clients:")
    print(edge.merge(df[key + ["n_targets"]].drop_duplicates(), on=key)
          .groupby("n_targets")["beta_max_pass"].median().round(2).to_string(), "\n")

    print("tightest 15 combinations:")
    print(edge.sort_values(["beta_max_pass", "map"]).head(15).to_string(index=False))
    return edge


def plot_report(df, edge):
    import matplotlib.pyplot as plt
    if "peak_lps" not in df.columns or df["peak_lps"].notna().sum() == 0:
        raise RuntimeError("no run produced hydraulic results — inspect "
                           "results['error'] before plotting")
    df = df[df["peak_lps"].notna()]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    edge.pivot_table(index="map", columns="to_land_use",
                     values="beta_max_pass", aggfunc="median").plot.bar(ax=ax[0])
    ax[0].set(title="median beta_max_pass by map", xlabel="")
    ax[0].tick_params(axis="x", labelsize=6, rotation=70)

    ok = df[df["passed"]]
    ax[1].scatter(ok["peak_lps"], ok["pmin"], s=8, alpha=.5, label="pass")
    bad = df[~df["passed"] & df["pmin"].notna()]
    ax[1].scatter(bad["peak_lps"], bad["pmin"], s=8, alpha=.6, c="crimson", label="fail")
    ax[1].axhline(HARD_FLOOR, color="crimson", ls="--", lw=1)
    ax[1].axhline(BAND[0], color="orange", ls=":", lw=1)
    ax[1].set(xlabel="network peak demand (L/s)", ylabel="min pressure (mca)",
              title="feasibility frontier")
    ax[1].legend(fontsize=7)

    ax[2].scatter(df["peak_lps"], df["pmax"], s=8, alpha=.5)
    ax[2].axhline(BAND[1], color="crimson", ls="--", lw=1)
    ax[2].set(xlabel="network peak demand (L/s)", ylabel="max pressure (mca)",
              title="overpressure check (the other band edge)")
    plt.tight_layout()
    return fig
