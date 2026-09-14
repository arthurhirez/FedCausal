"""landuse_sweep.py -- drift one district at a time, at several strengths.

The grid is (network) x (which district drifts) x (drift strength). The
initial map is ALL residential, so exactly one district differs from the rest
in the final phase and every "which regime does this look like" statistic has
a real contrast to resolve.

Why `beta` is the strength axis and income is not
-------------------------------------------------
`level_factor = (mean_plot_intensity / reference) ** beta`. For a
residential -> industrial conversion at low income the intensity ratio is
9.19, so beta walks the level change from 1.00x (beta 0) to 9.19x (beta 1):

    beta      0.00   0.15   0.35   0.60   1.00
    level     1.00   1.39   2.17   3.78   9.19

`to_income` cannot do this. Income enters only through the RESIDENTIAL plot
intensity, and an industrial block is 20% residential by plot share, so the
same ratio at medium income is 9.37 and at high income 9.70 -- a 5% move
against beta's 9x. Income is the wrong knob for an industrial target; it
would be the right one for a residential -> residential level drift, which is
a different experiment.

`beta` is global, but with an all-residential initial map every non-drifting
district sits at ratio 1.0 and therefore at level_factor 1.0 for ANY beta. So
in this design beta moves the drifting district alone -- it is a clean
strength dial, not a confound. That stops being true the moment the initial
map contains a second non-residential district.

**beta = 0 is the rung that matters.** It is volume-neutral: the drift changes
the temporal signature and nothing else. If the signal is only detectable at
beta >= 0.6 then what is being detected is a level change, the shape model is
not carrying the regime, and the consumption-attribution protocol is what
needs work -- not the sensors.

Caching
-------
Cells are content-addressed by `sim_hash`. A cell whose world directory
already exists is loaded from disk instead of re-solved, so an interrupted
sweep resumes and a widened grid only pays for the new cells.
"""
from __future__ import annotations

import pathlib
import time as _time
import traceback
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import landuse_world as lw
import signal_probe as sp

__all__ = ["SweepCfg", "run_sweep", "cells", "district_names", "leak_matrix",
           "signal_carriers", "drift_carriers", "stability",
           "proposed_sensors", "plot_leak", "plot_carriers"]

BETAS = (0.35,)


# ==========================================================================
# grid
# ==========================================================================
@dataclass
class SweepCfg:
    """The axes, plus the world settings held fixed across every cell.

    Horizon defaults are chosen for cost, not for realism: 16 months of
    14-day months is 5 376 steps against the POC's 33 600, which is the
    difference between a 15-minute cell and a 15-minute sweep. `max_neighbors`
    is raised so the diffusion front finishes early enough to leave a clean
    final phase inside that horizon -- a WIDTH change to the front, not a
    change to the drift mechanism.
    """
    networks: tuple = ("dtown",)
    betas: tuple = BETAS
    districts: dict | None = None          # network -> [district, ...]; None = all
    to_land_use: str = "industrial"
    to_income: str = "low"

    n_months: int = 16
    days_per_month: int = 14
    warmup_months: int = 2
    max_neighbors_per_month: int = 8
    growth_chance: float = 1.0
    drift_ramp_days: int = 7
    seasonality_scale: float = 0.0
    coupling_variant: str = "baseline"
    seed: int = 42
    extra: dict = field(default_factory=dict)

    # Passed EXPLICITLY rather than looked up. `WorldCfg.resolved_anchor`
    # raises a KeyError for any network name absent from `lw.ANCHOR_SCALE`,
    # and the district sweep runs on synthetic per-partition bundle names that
    # are not in that dict by construction. `None` restores the lookup.
    anchor_scale: float | None = 0.25

    def base(self, network: str) -> lw.WorldCfg:
        return lw.WorldCfg(
            network=network, anchor_scale=self.anchor_scale,
            to_income=self.to_income,
            to_land_use=self.to_land_use, n_months=self.n_months,
            days_per_month=self.days_per_month, warmup_months=self.warmup_months,
            max_neighbors_per_month=self.max_neighbors_per_month,
            growth_chance=self.growth_chance, drift_ramp_days=self.drift_ramp_days,
            seasonality_scale=self.seasonality_scale,
            coupling_variant=self.coupling_variant, seed=self.seed,
            extra=self.extra)


def district_names(root, network: str) -> list[str]:
    return list(lw.load_bundle(root, network)["districts"]["districts"])


def cells(cfg: SweepCfg, root) -> list[lw.WorldCfg]:
    """The full grid, as concrete world configs."""
    out = []
    for network in cfg.networks:
        names = district_names(root, network)
        wanted = (cfg.districts or {}).get(network, names)
        allres = "_".join(["LR"] * len(names))          # all residential at t0
        for d in wanted:
            for beta in cfg.betas:
                out.append(replace(cfg.base(network), map_code=allres,
                                   tgt_district=d, beta=float(beta)))
        # beta is only a strength dial while every other district is LR --
        # see the module docstring.
    return out


# ==========================================================================
# execution
# ==========================================================================
def run_sweep(cfg: SweepCfg, root, out_root, cache: bool = True,
              phase: str = "final", max_lag_h: float = 6.0,
              verbose: bool = True):
    """Run the grid. Returns (cells, scores, deltas).

    `phase="final"` is the default because the INITIAL map is all residential:
    at t0 every district carries the same regime token, so every regime
    statistic is degenerate there by construction. `init` is still computed
    and stored -- it is the drift-free reference for sensor resemblance.

    A cell that fails (usually V3, the pressure floor, at high beta) is
    RECORDED with its error and the sweep continues. An infeasible high-beta
    corner is a result about the network's capacity, not a reason to lose the
    rest of the grid.
    """
    root, out_root = pathlib.Path(root), pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    grid = cells(cfg, root)
    rows, score_frames, delta_frames = [], [], []
    t_all = _time.time()

    for k, wc in enumerate(grid):
        key = dict(network=wc.network, drift_district=wc.tgt_district,
                   beta=wc.beta, sim_hash=lw.sim_hash(wc))
        tag = f"{wc.network}/{wc.tgt_district}/beta={wc.beta}"
        t0 = _time.time()
        try:
            P, src = _cell_probe(wc, root, out_root, cache)
            P.params.setdefault("patterns", {})
            cand = sp.candidates(P)
            sc = sp.battery(P, cand, max_lag_h=max_lag_h)
            sc = sc.merge(cand[["id", "served_own"]], on="id", how="left")
            for c, v in key.items():
                sc[c] = v
            score_frames.append(sc)

            dl = sp.phase_delta(P).assign(**key)
            delta_frames.append(dl)
            rows.append({**key, **_cell_row(P, sc, dl, phase),
                         "source": src, "status": "ok", "error": "",
                         "seconds": round(_time.time() - t0, 1)})
        except Exception as exc:                      # noqa: BLE001 - recorded
            rows.append({**key, "source": "-", "status": "fail",
                         "error": f"{type(exc).__name__}: {exc}"[:200],
                         "seconds": round(_time.time() - t0, 1)})
            if verbose:
                print(f"  [{k+1}/{len(grid)}] {tag}  FAILED  "
                      f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
                traceback.print_exc(limit=1)
            continue
        if verbose:
            r = rows[-1]
            el = _time.time() - t_all
            print(f"  [{k+1}/{len(grid)}] {tag}  {r['source']:6s} "
                  f"dDemand {r['d_demand_%']:+7.1f}%  shift {r['shape_shift']:.2f}  "
                  f"flow gap {r['gap_flow']:.2f}  resp {r['resp_flow_top']:.2f}  "
                  f"{r['seconds']:.0f}s  | eta {(el/(k+1))*(len(grid)-k-1)/60:.0f} min",
                  flush=True)

    C = pd.DataFrame(rows)
    S = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    D = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    C.to_csv(out_root / "sweep_cells.csv", index=False)
    if len(S):
        S.to_parquet(out_root / "sweep_scores.parquet", index=False)
        D.to_csv(out_root / "sweep_deltas.csv", index=False)
    if verbose:
        ok = int((C["status"] == "ok").sum())
        print(f"\n{ok}/{len(C)} cells ok in {(_time.time()-t_all)/60:.1f} min "
              f"-> {out_root}")
    return C, S, D


def _cell_probe(wc: lw.WorldCfg, root, out_root, cache: bool):
    """Load the cell from the world cache, or simulate and persist it."""
    d = pathlib.Path(out_root) / lw.sim_hash(wc)
    if cache and (d / "manifest.json").exists():
        from worlds import load_world
        return sp.from_lab_world(load_world(d)), "cache"
    world = lw.build(wc, root, verbose=False)
    lw.persist(world, out_root)
    return sp.from_world(world), "solved"


def _cell_row(P: sp.Probe, sc: pd.DataFrame, dl: pd.DataFrame,
              phase: str) -> dict:
    """One line per cell: did the drift land, and did anything see it."""
    tgt = P.drift["tgt_district"]
    t = dl[dl["district"] == tgt].iloc[0]
    g = sp.discriminative_gap(sc)
    g = g[g["phase"] == phase].set_index("kind")

    own = sc[(sc["phase"] == phase) & (sc["district"] == tgt)]
    top = {}
    for kind in ("flow", "pressure"):
        k = own[own["kind"] == kind].nlargest(3, "r_res_own")
        top[kind] = float(k["response"].mean()) if len(k) else np.nan
    # The leak: the strongest response anywhere OUTSIDE the drifting district.
    # A closed network guarantees it is non-zero; how big it is relative to
    # the target's own response is the confounding this whole thesis is about.
    other = sc[(sc["phase"] == phase) & (sc["district"] != tgt)
               & (sc["kind"] == "flow")]
    return {
        "d_demand_%": float(t["d_demand_%"]), "shape_shift": float(t["shape_shift"]),
        "d_pressure": float(t["d_pressure"]), "d_boundary_%": float(t["d_boundary_%"]),
        "gap_flow": float(g.loc["flow", "gap_regime"]) if "flow" in g.index else np.nan,
        "gap_pressure": (float(g.loc["pressure", "gap_regime"])
                         if "pressure" in g.index else np.nan),
        "hit_flow": float(g.loc["flow", "regime_hit_rate"]) if "flow" in g.index else np.nan,
        "hit_pressure": (float(g.loc["pressure", "regime_hit_rate"])
                         if "pressure" in g.index else np.nan),
        "resp_flow_top": top["flow"], "resp_pressure_top": top["pressure"],
        "resp_leak_max": float(other["response"].abs().max()) if len(other) else np.nan,
        "resp_leak_p95": (float(other["response"].abs().quantile(0.95))
                          if len(other) else np.nan),
    }


# ==========================================================================
# read-outs
#
# `beta_ladder`, `beta_threshold` and `plot_beta_ladder` were removed with the
# beta axis. They answered "how much level change does detection need", which
# is not a question a single-beta grid can ask. The consequence is worth
# stating where it will be read: with one operating point there is no
# linearity check, so every dependence number below is an operating-point
# statement and cannot be quoted at another beta.
# ==========================================================================
def leak_matrix(S: pd.DataFrame, phase: str = "final", kind: str = "flow",
                top: int = 3, agg: str = "mean") -> dict:
    """Rows = which district drifted, cols = where the response was measured.

    Each entry is the mean |response| of that column-district's `top`
    best-resembling gauges. The diagonal is the signal; everything off it is
    hydraulic contamination arriving through shared pipes. The ratio of the
    two is the de-confounding problem stated in numbers, per network.
    """
    out = {}
    d = S[(S["phase"] == phase) & (S["kind"] == kind)]
    for network, gn in d.groupby("network"):
        rows = {}
        for drifted, gd in gn.groupby("drift_district"):
            rows[drifted] = {}
            for district, gg in gd.groupby("district"):
                # top-`top` WITHIN each beta, then pooled -- taking the
                # globally largest rows would silently weight whichever beta
                # happened to produce the cleanest resemblance.
                best = (gg.groupby("beta", group_keys=False)
                        .apply(lambda x: x.nlargest(top, "r_res_own"),
                               include_groups=False))
                v = best["response"].abs()
                rows[drifted][district] = float(getattr(v, agg)())
        M = pd.DataFrame(rows).T
        out[network] = M.reindex(index=sorted(M.index), columns=sorted(M.columns))
    return out


def signal_carriers(S: pd.DataFrame, phase: str = "init", top: int = 5) -> pd.DataFrame:
    """Gauges that RESEMBLE their district, averaged over the whole grid.

    Scored in the drift-free `init` phase and averaged across every cell, so
    this is a property of the network and the placement, not of any one drift.
    `std` is the stability: a gauge with a high mean and a low std is a
    placement you can commit to across scenarios.
    """
    g = (S[S["phase"] == phase]
         .groupby(["network", "district", "kind", "id", "role"])["r_res_own"]
         .agg(["mean", "std", "min", "count"]).reset_index()
         .rename(columns={"mean": "r_mean", "std": "r_std", "min": "r_min"}))
    g["rank"] = g.groupby(["network", "district", "kind"])["r_mean"] \
        .rank(ascending=False, method="first")
    return (g[g["rank"] <= top].sort_values(["network", "district", "kind", "rank"])
            .reset_index(drop=True).round(4))


def drift_carriers(S: pd.DataFrame, phase: str = "final", top: int = 5) -> pd.DataFrame:
    """Gauges that RESPOND when their own district is the one that drifts.

    The complement of `signal_carriers`, and the stricter test: restricted to
    the cells where this gauge's district is the drifting one, and ranked by
    response rather than resemblance. `beta_min` is the smallest strength at
    which that gauge already responded above 0.5.
    """
    d = S[(S["phase"] == phase) & (S["district"] == S["drift_district"])]
    g = (d.groupby(["network", "district", "kind", "id", "role"])
         .agg(resp_mean=("response", "mean"), resp_std=("response", "std"),
              r_mean=("r_res_own", "mean"), n=("response", "size")).reset_index())
    strong = d[d["response"] >= 0.5].groupby(["network", "district", "kind", "id"])["beta"] \
        .min().rename("beta_min").reset_index()
    g = g.merge(strong, on=["network", "district", "kind", "id"], how="left")
    g["rank"] = g.groupby(["network", "district", "kind"])["resp_mean"] \
        .rank(ascending=False, method="first")
    return (g[g["rank"] <= top].sort_values(["network", "district", "kind", "rank"])
            .reset_index(drop=True).round(4))


def stability(S: pd.DataFrame, phase: str = "init", top: int = 5) -> pd.DataFrame:
    """How often a gauge lands in its district's top-`top` across all cells.

    `share` near 1.0 means the ranking does not depend on which district
    drifted or how hard -- i.e. placement is a property of the network and can
    be decided once. A low share everywhere means placement has to be
    re-derived per scenario, which would be a finding in its own right.
    """
    d = S[S["phase"] == phase].copy()
    d["rk"] = d.groupby(["network", "sim_hash", "district", "kind"])["r_res_own"] \
        .rank(ascending=False, method="first")
    n = d.groupby(["network", "district", "kind"])["sim_hash"].nunique() \
        .rename("cells")
    hits = (d[d["rk"] <= top].groupby(["network", "district", "kind", "id"])
            .size().rename("in_top"))
    out = hits.reset_index().merge(n.reset_index(),
                                   on=["network", "district", "kind"])
    out["share"] = out["in_top"] / out["cells"]
    return out.sort_values(["network", "district", "kind", "share"],
                           ascending=[True, True, True, False]).reset_index(drop=True)


def proposed_sensors(S: pd.DataFrame, network: str, n_pressure: int = 2,
                     n_flow: int = 3, phase: str = "init") -> dict:
    """A placement chosen on the WHOLE grid rather than on one world.

    Ranked by mean resemblance across every cell and tie-broken by stability,
    so the result is the placement that works whatever drifts next.
    """
    car = signal_carriers(S[S["network"] == network], phase=phase, top=50)
    st = stability(S[S["network"] == network], phase=phase, top=5)
    car = car.merge(st[["district", "kind", "id", "share"]],
                    on=["district", "kind", "id"], how="left")
    car["share"] = car["share"].fillna(0.0)
    out = {}
    for district, g in car.groupby("district"):
        pick = {}
        for kind, k in (("pressure", n_pressure), ("flow", n_flow)):
            sub = (g[g["kind"] == kind]
                   .sort_values(["share", "r_mean"], ascending=False).head(k))
            pick[kind] = [i[len(sp._PREFIX[kind]):] for i in sub["id"]]
        out[district] = pick
    return out


# ==========================================================================
# figures
# ==========================================================================
def plot_leak(M: dict, axes=None, vmax: float | None = None):
    """The leak matrices. Diagonal = signal, off-diagonal = contamination."""
    import matplotlib.pyplot as plt
    nets = sorted(M)
    if axes is None:
        _, axes = plt.subplots(1, len(nets), figsize=(4.6 * len(nets), 4.0),
                               squeeze=False)
        axes = axes[0]
    for ax, net in zip(np.atleast_1d(axes), nets):
        A = M[net]
        im = ax.imshow(A.to_numpy(), cmap="magma", vmin=0,
                       vmax=vmax or float(np.nanmax(A.to_numpy())))
        ax.set(xticks=range(A.shape[1]), yticks=range(A.shape[0]),
               xlabel="measured in", ylabel="drifted",
               title=f"{net}: |response| leak")
        ax.set_xticklabels([c.replace("District_", "") for c in A.columns])
        ax.set_yticklabels([c.replace("District_", "") for c in A.index])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A.to_numpy()[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v < 0.6 * im.get_clim()[1] else "k")
    return axes


def plot_carriers(car: pd.DataFrame, network: str, kind: str = "flow", ax=None):
    """Mean resemblance with its spread, per district's top gauges."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    d = car[(car["network"] == network) & (car["kind"] == kind)]
    x = np.arange(len(d))
    ax.errorbar(x, d["r_mean"], yerr=d["r_std"].fillna(0), fmt="o", ms=4,
                capsize=2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.district.replace('District_','')}:{r.id}"
                        for r in d.itertuples()], rotation=75, fontsize=6)
    ax.axhline(0, color="k", lw=.6)
    ax.set(ylabel="r_res_own (mean +/- sd over cells)",
           title=f"{network} -- {kind} signal carriers")
    return ax
