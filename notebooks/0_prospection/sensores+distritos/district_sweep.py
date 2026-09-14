"""district_sweep.py -- sweep the PARTITION, not the drift strength.

The grid
--------
    (district configuration) x (which district drifts)

`beta` is a fixed hyperparameter now. The inner axis cannot be dropped: the
mixture estimator identifies a gauge's district mix by stacking worlds in which
exactly one district drifted, so a configuration of K districts needs K worlds
and reads out one gain matrix. Four pipeline partitions at K=5 is 20 worlds.

Two analyses run on each configuration's stack, and they answer different
questions:

* **resemblance and response** (`signal_probe`, via `landuse_sweep`) -- which
  gauges look like their own district's demand shape, how stably, and which
  ones move when it drifts. This is the placement question.
* **mixture and dependence** (`mixture_probe`) -- of what a gauge saw, how much
  came from each district, and how far a district's drift travels. This is the
  quantified-dependence question, and it is what `placement` selects on.

What this module does NOT do
----------------------------
No estimator lives here. Every number comes from `signal_probe` or
`mixture_probe` unchanged; this is the grid, the cache, the pre-flight and the
store. The one judgement it makes is horizon sizing, and it makes it visibly
(`plan`) rather than by picking a constant.

Consequence of the fixed beta, stated where it will be read: with a single
operating point there is no linearity check, so every dependence number this
module persists is an operating-point statement and cannot be quoted at
another beta.
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

import landuse_sweep as ls
import landuse_world as lw
import mixture_probe as mp
import signal_probe as sp
from bundles import Bundle

__all__ = ["SweepPlan", "plan", "ConfigResult", "run_config", "run", "load",
           "summarise", "NULL_FACTORS"]

# The tiering threshold is swept, not chosen once. `null_factor` decides which
# responses enter the simplex at all, so a purity that moves across this range
# is a statement about the threshold rather than about the network -- which is
# exactly what `placement` filters the target set on. 1.5 is the working
# default: mixture_lab found 3.0 manufactures separability on D-Town, driving
# the off-diagonal to zero by construction.
NULL_FACTORS = (1.0, 1.5, 3.0)
NULL_FACTOR = 1.5
CORE_PURITY = 0.85
FOREIGN_MAX = 0.15


# ==========================================================================
# 0. horizon planning
# ==========================================================================
@dataclass
class SweepPlan:
    """The world settings every cell shares, sized against the partitions.

    `convert_months` is the quantity everything else hangs off: the drift front
    must finish converting the largest district early enough to leave a settled
    tail that `mixture_probe.settled_window` will accept.
    """
    cfg: ls.SweepCfg
    convert_months: int
    largest_district: int
    max_eccentricity: int
    note: str = ""

    def show(self):
        c = self.cfg
        print(f"horizon plan: n_months={c.n_months} x {c.days_per_month}d "
              f"| warmup {c.warmup_months} | front {c.max_neighbors_per_month}"
              f" nodes/month -> converts {self.largest_district} nodes in "
              f"~{self.convert_months} months (eccentricity floor "
              f"{self.max_eccentricity})")
        if self.note:
            print(f"  {self.note}")
        return self.cfg


def plan(bundles: list[Bundle], beta: float = 0.35, anchor_scale: float = 0.25,
         days_per_month: int = 14, warmup_months: int = 2,
         convert_months: int = 12, settled_months: int = 8,
         drift_ramp_days: int = 7, seed: int = 42,
         max_neighbors_per_month: int | str = "auto",
         n_months: int | None = None, verbose: bool = True) -> SweepPlan:
    """Size the horizon and the diffusion front against the partitions in hand.

    The problem this exists to solve is concrete. Under a partition whose
    districts are wildly uneven -- D-Town's walktrap cut puts 265 of 399
    junctions in one district and 12 in another -- a single
    `max_neighbors_per_month` makes the small districts convert in two months
    and the large one in thirty-three. At a fixed horizon the large cell then
    has no settled tail at all and `settled_window` raises, while the small
    cells are fine. Deriving the front width from the largest district makes
    every cell of a configuration finish converting at about the same month.

    This is a WIDTH change to the diffusion front, which the drift engine
    already parameterises; the mechanism is untouched. `max_neighbors_per_month`
    accepts an int to pin it instead.
    """
    sizes = [len(ns) for b in bundles for ns in b.districts.values()]
    eccs = [int(e) for b in bundles for e in b.seeds["eccentricity"].dropna()]
    largest = max(sizes) if sizes else 1
    max_ecc = max(eccs) if eccs else 0

    if max_neighbors_per_month == "auto":
        front = max(1, math.ceil(largest / max(1, convert_months)))
    else:
        front = int(max_neighbors_per_month)

    # Two independent lower bounds on conversion time, and the front cannot
    # beat either: the front width, and the graph eccentricity from the seed
    # (the front advances at most one hop per month however wide it is).
    conv = max(math.ceil(largest / front), max_ecc)
    if max_neighbors_per_month == "auto" and conv > 0:
        # When eccentricity binds, a wide front buys nothing -- it just
        # converts a whole annulus the month it reaches it. Narrow it back to
        # the width that finishes in the same `conv` months, so the front
        # advances smoothly instead of in jumps.
        front = max(1, math.ceil(largest / conv))
    ramp_months = math.ceil(drift_ramp_days / days_per_month)
    need = warmup_months + conv + ramp_months + 1 + settled_months
    months = int(n_months) if n_months else int(need)

    note = ""
    if months < need:
        note = (f"WARNING: n_months={months} is below the {need} this plan "
                "needs; expect settled_window to raise on the large districts")
    cfg = ls.SweepCfg(
        networks=tuple(b.network for b in bundles), betas=(float(beta),),
        districts=None, to_land_use="industrial", to_income="low",
        n_months=months, days_per_month=days_per_month,
        warmup_months=warmup_months, max_neighbors_per_month=front,
        growth_chance=1.0, drift_ramp_days=drift_ramp_days,
        seasonality_scale=0.0, coupling_variant="baseline", seed=seed,
        anchor_scale=anchor_scale)
    p = SweepPlan(cfg=cfg, convert_months=conv, largest_district=largest,
                  max_eccentricity=max_ecc, note=note)
    if verbose:
        p.show()
    return p


# ==========================================================================
# 1. one configuration
# ==========================================================================
@dataclass
class ConfigResult:
    """Everything one partition produced. The unit the store reads and writes."""
    config_id: str
    method: str
    network: str
    districts: dict
    assets: dict
    seeds: pd.DataFrame

    cells: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    deltas: pd.DataFrame = field(default_factory=pd.DataFrame)

    horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    settle: pd.DataFrame = field(default_factory=pd.DataFrame)
    mixture: pd.DataFrame = field(default_factory=pd.DataFrame)
    gains: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    dependence: dict = field(default_factory=dict)     # kind -> {"D","R","n"}
    response: dict = field(default_factory=dict)       # (kind, use) -> frame
    connectivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    carriers: pd.DataFrame = field(default_factory=pd.DataFrame)
    carrier_stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    leak: dict = field(default_factory=dict)           # kind -> frame

    status: str = "ok"
    error: str = ""
    seconds: float = 0.0
    # Live handles, not persisted.
    base: dict | None = field(default=None, repr=False, compare=False)
    probes: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def names(self) -> list[str]:
        return list(self.districts)

    def summary(self) -> pd.Series:
        row = {"config_id": self.config_id, "method": self.method,
               "status": self.status, "seconds": round(self.seconds, 1)}
        if len(self.cells):
            row["cells_ok"] = int((self.cells["status"] == "ok").sum())
            row["cells"] = int(len(self.cells))
        for kind in ("flow", "pressure"):
            dep = self.dependence.get(kind)
            if dep is None:
                continue
            A = dep["R"].to_numpy(dtype=float).copy()
            off = A[~np.eye(len(A), dtype=bool)]
            row[f"{kind}_offdiag"] = round(float(np.nanmean(off)), 4)
            row[f"{kind}_offdiag_max"] = round(float(np.nanmax(off)), 4)
        if len(self.mixture):
            for tier in ("core", "transition", "foreign", "unusable"):
                row[f"n_{tier}"] = int((self.mixture["tier"] == tier).sum())
        if len(self.seeds):
            row["min_coverage"] = round(float(self.seeds["coverage"].min()), 3)
        row["error"] = self.error
        return pd.Series(row)


def run_config(bundle: Bundle, cfg: ls.SweepCfg, root, worlds_root,
               null_factor: float = NULL_FACTOR,
               core_purity: float = CORE_PURITY,
               foreign_max: float = FOREIGN_MAX,
               top_carriers: int = 5, cache: bool = True,
               verbose: bool = True) -> ConfigResult:
    """Solve one configuration's K worlds and run both analyses over them."""
    t0 = _time.time()
    res = ConfigResult(config_id=bundle.config_id, method=bundle.method,
                       network=bundle.network, districts=bundle.districts,
                       assets=bundle.assets, seeds=bundle.seeds)
    one = ls.SweepCfg(**{**cfg.__dict__, "networks": (bundle.network,)})
    world_dir = pathlib.Path(worlds_root) / bundle.config_id
    if verbose:
        print(f"\n=== {bundle.config_id} ({bundle.method}) "
              f"-- {len(bundle.districts)} worlds ===", flush=True)

    # ---- the K worlds, plus the resemblance/response battery --------------
    res.cells, res.scores, res.deltas = ls.run_sweep(
        one, root, world_dir, cache=cache, phase="final", verbose=verbose)

    bad = res.cells[res.cells["status"] != "ok"]
    if len(bad):
        # `build_mixture` stacks one world per district: a missing column is a
        # missing district, not a smaller sample. The configuration has no
        # mixture at all, and saying so is more useful than a partial matrix.
        res.status = "incomplete"
        res.error = ("worlds failed: "
                     + ", ".join(f"{r.drift_district}({r.error[:60]})"
                                 for r in bad.itertuples()))
        res.seconds = _time.time() - t0
        if verbose:
            print(f"  !! {res.error}")
        return res

    # ---- §4/§5/§6 read-outs, on this configuration's grid -----------------
    res.leak = {k: ls.leak_matrix(res.scores, phase="final", kind=k, top=3)
                .get(bundle.network, pd.DataFrame())
                for k in ("flow", "pressure")}
    res.carriers = ls.signal_carriers(res.scores, phase="init", top=top_carriers)
    res.carrier_stability = ls.stability(res.scores, phase="init",
                                         top=top_carriers)

    # ---- the mixture stack -----------------------------------------------
    probes = mp.load_cells(bundle.network, float(cfg.betas[0]), one, root,
                           world_dir)
    if len(probes) != len(bundle.districts):
        res.status = "incomplete"
        res.error = (f"{len(probes)}/{len(bundle.districts)} worlds present in "
                     "the cache; the mixture needs one per district")
        res.seconds = _time.time() - t0
        return res
    res.probes = probes

    res.horizon = mp.horizon_check(probes)
    if not res.horizon["ok"].all():
        short = list(res.horizon.loc[~res.horizon["ok"], "drift_district"])
        res.status = "incomplete"
        res.error = (f"settled window too short for {short}; rerun with "
                     f"n_months >= {int(res.horizon['need_n_months'].max())}")
        res.seconds = _time.time() - t0
        if verbose:
            print(f"  !! {res.error}")
        return res

    res.settle = pd.concat(
        [mp.settle_report(P).assign(drift_district=d) for d, P in probes.items()],
        ignore_index=True)

    base = mp.build_mixture(probes, null_factor=null_factor,
                            core_purity=core_purity, foreign_max=foreign_max,
                            verbose=verbose)
    res.base = base
    res.mixture = base["mixture"]
    res.gains = base["gains"]
    res.stability = _purity_stability(base, core_purity, foreign_max)
    res.diagnostics = mp.mixture_diagnostics(base)

    for kind in ("flow", "pressure"):
        D, R, n = mp.dependence(base, kind=kind)
        res.dependence[kind] = {"D": D, "R": R, "n": n.to_frame()}
        res.response[(kind, "all")] = mp.response_matrix(base, kind, top=None)
        for tier in ("core", "transition"):
            if (res.mixture["tier"] == tier).any():
                res.response[(kind, tier)] = mp.response_matrix(
                    base, kind, top=None, tier=tier, mixture=res.mixture)

    P0 = probes[sorted(probes)[0]]
    res.connectivity = pd.concat(
        [mp.core_connectivity(P0, base, k) for k in ("flow", "pressure")],
        ignore_index=True)
    res.checks = pd.concat([mp.estimator_checks(base, k, space=s)
                            for k in ("flow", "pressure")
                            for s in ("shape", "native")], ignore_index=True)
    res.seconds = _time.time() - t0
    if verbose:
        print(f"  done in {res.seconds/60:.1f} min | "
              + res.mixture.groupby("tier").size().to_dict().__str__())
    return res


def _purity_stability(base: dict, core_purity: float,
                      foreign_max: float) -> pd.DataFrame:
    """Per gauge: how far its purity and its tier move across `NULL_FACTORS`.

    `remix` recomputes the simplex from stored gains -- no re-profiling, no
    solving -- so this costs nothing and answers the question the target set
    depends on. A gauge whose purity swings from 0.4 to 0.9 between
    null_factor 1.5 and 3.0 does not carry a quantified mixture; it carries a
    threshold artefact, and `placement` drops it.
    """
    frames = {}
    for f in NULL_FACTORS:
        m = mp.remix(base, null_factor=f, core_purity=core_purity,
                     foreign_max=foreign_max)["mixture"]
        frames[f] = m.set_index("id")[["purity", "tier", "w_second"]]
    ids = frames[NULL_FACTORS[0]].index
    P = pd.DataFrame({f: frames[f]["purity"].reindex(ids) for f in NULL_FACTORS})
    T = pd.DataFrame({f: frames[f]["tier"].reindex(ids) for f in NULL_FACTORS})
    return pd.DataFrame({
        "id": ids,
        "purity_min": P.min(axis=1).to_numpy(),
        "purity_max": P.max(axis=1).to_numpy(),
        "purity_swing": (P.max(axis=1) - P.min(axis=1)).to_numpy(),
        "tier_stable": T.nunique(axis=1).eq(1).to_numpy(),
    }).round(4)


# ==========================================================================
# 2. the grid
# ==========================================================================
def run(bundles: list[Bundle], cfg: ls.SweepCfg, root, out_root,
        worlds_root=None, persist_results: bool = True, **kw) -> dict:
    """Every configuration, in order. Returns {config_id: ConfigResult}.

    A configuration that fails is RECORDED and the grid continues: an
    infeasible partition is a result about that partition, not a reason to
    lose the others.
    """
    out_root = pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    worlds_root = pathlib.Path(worlds_root or out_root / "worlds")
    results, t0 = {}, _time.time()
    for b in bundles:
        r = run_config(b, cfg, root, worlds_root, **kw)
        results[b.config_id] = r
        if persist_results:
            persist(r, out_root)
    S = summarise(results)
    S.to_csv(out_root / "configs.csv", index=False)
    elements(results).to_parquet(out_root / "elements.parquet", index=False)
    print(f"\n{int((S['status'] == 'ok').sum())}/{len(S)} configurations ok "
          f"in {(_time.time()-t0)/60:.1f} min -> {out_root}")
    return results


def summarise(results: dict) -> pd.DataFrame:
    return pd.DataFrame([r.summary() for r in results.values()])


def elements(results: dict) -> pd.DataFrame:
    """Gauge rows across configurations, keyed by ELEMENT rather than by home.

    Per-gauge rows cannot be pooled across configurations the obvious way:
    `home` is partition-dependent, so `p_J123` is District_A under one cut and
    District_C under another and the two rows are not the same observation.
    The element is shared, though -- every configuration runs the same .inp --
    so keying on it answers the one cross-configuration question that is well
    posed: which elements are readable whatever the cut.
    """
    keep = ["id", "element", "kind", "home", "role", "tier", "purity",
            "second", "w_second", "snr_home", "n_live", "n_mass", "neg_mass"]
    out = []
    for cid, r in results.items():
        if not len(r.mixture):
            continue
        m = r.mixture[[c for c in keep if c in r.mixture.columns]].copy()
        if len(r.stability):
            m = m.merge(r.stability[["id", "purity_swing", "tier_stable"]],
                        on="id", how="left")
        out.append(m.assign(config_id=cid, method=r.method))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ==========================================================================
# 3. store
# ==========================================================================
def persist(res: ConfigResult, out_root) -> pathlib.Path:
    """Write one configuration's tables. The FL step reads this directory."""
    out = pathlib.Path(out_root) / res.config_id
    out.mkdir(parents=True, exist_ok=True)
    # Clear only what THIS function owns. An earlier version removed the whole
    # directory, which quietly deleted `placements/` whenever a re-persist
    # followed a `placement.emit` -- the two write to the same config folder
    # and only the run order kept them apart.
    for stale in list(out.glob("*.csv")) + list(out.glob("*.parquet")):
        stale.unlink()
    if (out / "matrices").exists():
        shutil.rmtree(out / "matrices")

    (out / "districts.yml").write_text(yaml.safe_dump(
        {"districts": res.districts, "assets": res.assets}, sort_keys=False))
    (out / "manifest.json").write_text(json.dumps({
        "config_id": res.config_id, "method": res.method,
        "network": res.network, "districts": res.names,
        "status": res.status, "error": res.error,
        "seconds": round(res.seconds, 1),
        "created": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    for name, frame in (("cells", res.cells), ("deltas", res.deltas),
                        ("seeds", res.seeds), ("horizon", res.horizon),
                        ("settle", res.settle), ("diagnostics", res.diagnostics),
                        ("connectivity", res.connectivity), ("checks", res.checks),
                        ("carrier_stability", res.carrier_stability)):
        if len(frame):
            frame.to_csv(out / f"{name}.csv", index=False)
    for name, frame in (("scores", res.scores), ("mixture", res.mixture),
                        ("gains", res.gains), ("stability", res.stability),
                        ("carriers", res.carriers)):
        if len(frame):
            frame.to_parquet(out / f"{name}.parquet", index=False)

    mat = out / "matrices"
    mat.mkdir(exist_ok=True)
    for kind, d in res.dependence.items():
        d["D"].to_csv(mat / f"dependence_D_{kind}.csv")
        d["R"].to_csv(mat / f"dependence_R_{kind}.csv")
    for (kind, use), A in res.response.items():
        A.to_csv(mat / f"response_{kind}_{use}.csv")
    for kind, A in res.leak.items():
        if len(A):
            A.to_csv(mat / f"leak_{kind}.csv")
    return out


def load(path) -> ConfigResult:
    """Rebuild a `ConfigResult` from disk. `base` and `probes` stay empty.

    Everything the figures and the browsers read is a persisted table, so a
    loaded result drives them exactly as a live one does. `placement` also
    works from the tables alone; only `mixture_probe.remix` needs `base`, and
    its output is already persisted as `stability`.
    """
    path = pathlib.Path(path)
    mf = json.loads((path / "manifest.json").read_text())
    dy = yaml.safe_load((path / "districts.yml").read_text())
    res = ConfigResult(config_id=mf["config_id"], method=mf["method"],
                       network=mf["network"], districts=dy["districts"],
                       assets=dy.get("assets") or {}, seeds=pd.DataFrame(),
                       status=mf.get("status", "ok"), error=mf.get("error", ""),
                       seconds=float(mf.get("seconds", 0.0)))
    for name in ("cells", "deltas", "seeds", "horizon", "settle", "diagnostics",
                 "connectivity", "checks", "carrier_stability"):
        f = path / f"{name}.csv"
        if f.exists():
            setattr(res, name, pd.read_csv(f))
    for name in ("scores", "mixture", "gains", "stability", "carriers"):
        f = path / f"{name}.parquet"
        if f.exists():
            setattr(res, name, pd.read_parquet(f))

    mat = path / "matrices"
    if mat.exists():
        for f in sorted(mat.glob("dependence_R_*.csv")):
            kind = f.stem.split("dependence_R_")[1]
            R = pd.read_csv(f, index_col=0)
            D = pd.read_csv(mat / f"dependence_D_{kind}.csv", index_col=0)
            res.dependence[kind] = {"D": D, "R": R, "n": pd.DataFrame()}
        for f in sorted(mat.glob("response_*.csv")):
            kind, use = f.stem.split("response_")[1].rsplit("_", 1)
            res.response[(kind, use)] = pd.read_csv(f, index_col=0)
        for f in sorted(mat.glob("leak_*.csv")):
            res.leak[f.stem.split("leak_")[1]] = pd.read_csv(f, index_col=0)
    return res


def load_all(out_root) -> dict:
    out_root = pathlib.Path(out_root)
    return {p.name: load(p) for p in sorted(out_root.iterdir())
            if (p / "manifest.json").exists()}
