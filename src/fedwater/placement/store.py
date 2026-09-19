"""store.py -- probe stacks: built once per operating point, cached, reloaded.

Layout under the store root (``globals.probe_store``, default
``data/09_experiments/probes``)::

    <network>/<partition method>/<stack_hash>/
        manifest.json         status, spec, timings, src hash
        districts.yml         the partition the stack was measured on
        seeds.csv horizon.csv settle.csv cells.csv deltas.csv diagnostics.csv
        connectivity.csv checks.csv channels.csv carrier_stability.csv
        tier_sensitivity.csv purity_histogram.csv dependence_summary.csv
        scores.parquet mixture.parquet gains.parquet stability.parquet
        carriers.parquet elasticity_shape.parquet elasticity_native.parquet
        matrices/             dependence D/R (shape and native), response,
                              leak, asymmetry -- one csv per kind
        worlds/<District_X>/  the K probe worlds' series (retain_series)

Everything a figure or browser reads is a persisted table, so
:func:`load_stack` drives them exactly as a freshly built stack does (the
POC's ``district_sweep.load`` contract). Only ``mixture_probe.remix`` needs
the live ``base``, and its output is already persisted as ``stability`` and
``tier_sensitivity``.

Identity
--------
``stack_hash`` is ``canonical_hash`` of the spec: network, partition id,
``.inp`` content, initial consumption map, the resolved operating point
(hydraulics, beta, land use, buildings), the probe horizon and patterns,
coupling, seed, transitions and the analysis thresholds. It does NOT include
the world's drift target, its seed node, or the SELECTION settings (slots,
fill ladder) -- selection is cheap and re-run per world. Every world that
shares an initial map at one operating point therefore reuses one stack.

A failed build writes ``status: failed`` and is rebuilt on the next request;
the manifest records ``src_hash`` and a stack reloaded under different code is
flagged ``src_drift`` (reported, not invalidated -- the engine's policy).
"""
from __future__ import annotations

import json
import pickle
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fedwater.hashing import canonical_hash

from . import classification as cl
from . import excite
from . import mixture_probe as mp
from . import readouts as ro
from . import signal_probe as sp

__all__ = ["ProbeStack", "ConfigResult", "stack_spec", "stack_dir",
           "ensure_stack", "load_stack", "load_world", "load_probes",
           "src_hash"]

_CSV = ("cells", "deltas", "seeds", "horizon", "settle", "diagnostics",
        "connectivity", "checks", "channels", "carrier_stability",
        "tier_sensitivity", "purity_histogram", "dependence_summary")
_PARQUET = ("scores", "mixture", "gains", "stability", "carriers")
KINDS = ("flow", "pressure")
SPACES = ("shape", "native")


# ==========================================================================
# the result object -- the POC's ConfigResult, same attribute names
# ==========================================================================
@dataclass
class ProbeStack:
    """Everything one stack produced. Figures and browsers read this."""
    config_id: str
    method: str
    network: str
    districts: dict
    assets: dict
    seeds: pd.DataFrame = field(default_factory=pd.DataFrame)

    cells: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    deltas: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    settle: pd.DataFrame = field(default_factory=pd.DataFrame)
    mixture: pd.DataFrame = field(default_factory=pd.DataFrame)
    gains: pd.DataFrame = field(default_factory=pd.DataFrame)
    stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    dependence: dict = field(default_factory=dict)   # kind -> {"D","R","n"}
    dependence_native: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)     # (kind, use) -> frame
    connectivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    channels: pd.DataFrame = field(default_factory=pd.DataFrame)
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    carriers: pd.DataFrame = field(default_factory=pd.DataFrame)
    carrier_stability: pd.DataFrame = field(default_factory=pd.DataFrame)
    leak: dict = field(default_factory=dict)         # kind -> frame
    tier_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    purity_histogram: pd.DataFrame = field(default_factory=pd.DataFrame)
    dependence_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    asymmetry: dict = field(default_factory=dict)    # kind -> frame
    elasticity: dict = field(default_factory=dict)   # space -> frame

    status: str = "ok"
    error: str = ""
    seconds: float = 0.0
    stack_hash: str = ""
    path: Path | None = None
    spec: dict = field(default_factory=dict)
    src_drift: bool = False
    # live handles, never persisted
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
        for kind in KINDS:
            dep = self.dependence.get(kind)
            if dep is None:
                continue
            A = dep["R"].to_numpy(dtype=float).copy()
            off = A[~np.eye(len(A), dtype=bool)]
            row[f"{kind}_offdiag"] = round(float(np.nanmean(off)), 4)
            row[f"{kind}_offdiag_max"] = round(float(np.nanmax(off)), 4)
            # R is a RATIO to the diagonal, not a fraction: > 1 means j's
            # meters respond more to k's drift than to j's own
            row[f"{kind}_offdiag_gt1"] = int(np.nansum(off > 1.0))
        for _, r in self.channels.iterrows():
            row[f"{r['kind']}_channel"] = r["verdict"]
            row[f"{r['kind']}_tv"] = r["mean_pairwise_tv"]
        if len(self.mixture):
            for tier in ("core", "transition", "foreign", "unusable"):
                row[f"n_{tier}"] = int((self.mixture["tier"] == tier).sum())
        if len(self.seeds):
            row["min_coverage"] = round(float(self.seeds["coverage"].min()), 3)
        row["error"] = self.error
        return pd.Series(row)


ConfigResult = ProbeStack   # the POC's name, for notebooks written against it


# ==========================================================================
# identity
# ==========================================================================
def src_hash() -> str:
    """Hash of the code a stack's numbers depend on."""
    here = Path(__file__).resolve().parent
    pipes = here.parent / "pipelines"
    files = sorted(here.glob("*.py"))
    for p in ("network_prep", "urban_scenario", "demand_synthesis",
              "hydraulics", "sim_validation"):
        files += sorted((pipes / p).glob("*.py"))
    h = sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def stack_spec(*, network: str, partition: dict, world: dict, probe: dict,
               classes: dict, plan: dict, coupling: dict, seed: int) -> dict:
    """The full identity of one stack (see module docstring)."""
    scen = world["scenario"]
    drift_mech = {k: v for k, v in (scen.get("drift") or {}).items()
                  if k not in ("tgt_district", "seed_node", "to_income",
                               "to_land_use", "warmup_months", "growth_chance",
                               "max_neighbors_per_month")}
    return {
        "network": network,
        "partition": {"method": partition["method"],
                      "partition_id": partition["partition_id"],
                      "inp_sha": partition.get("inp_sha")},
        "init_map": [list(x) for x in scen["income_landuse_mapping"]],
        "beta": float(scen["beta"]),
        "drift_mechanism": drift_mech,
        "hydraulics": world["hydraulics"],
        "land_use": world["land_use"],
        "buildings": world["buildings"],
        # the probe's actual time grid and schedule: the plan decided them,
        # from the world or from the probe block, per the mode switches
        "patterns": {**world["patterns"],
                     "drift_ramp_days": int(plan["drift_ramp_days"]),
                     "seasonality_scale": float(plan["seasonality_scale"])},
        "time": {**world["time"], "n_months": int(plan["n_months"]),
                 "days_per_month": int(plan["days_per_month"])},
        "coupling": coupling,
        "seed": int(seed),
        "probe": {**{k: probe[k] for k in (
            "transitions", "settle", "battery_max_lag_h", "top_carriers")},
            **plan["modes"]},
        "plan": {k: plan[k] for k in (
            "n_months", "days_per_month", "warmup_months", "drift_ramp_days",
            "seasonality_scale", "season_aligned", "growth_chance",
            "max_neighbors_per_month")}
        | {"seeds": dict(zip(plan["seeds"]["district"],
                             plan["seeds"]["seed_node"]))},
        "analysis": {k: classes[k] for k in (
            "null_factor", "null_factors", "core_purity", "foreign_max")},
    }


def stack_dir(store_root, network: str, method: str, stack_hash: str) -> Path:
    return Path(store_root) / network / method / stack_hash


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


# ==========================================================================
# probe-world persistence
# ==========================================================================
def _save_world(world: dict, wdir: Path) -> None:
    wdir.mkdir(parents=True, exist_ok=True)
    world["demand"].to_parquet(wdir / "demand.parquet")
    world["pressures"].to_parquet(wdir / "pressures.parquet")
    world["flows"].to_parquet(wdir / "flows.parquet")
    world["schedule"].to_csv(wdir / "schedule.csv", index=False)
    world["validation"].to_csv(wdir / "validation.csv", index=False)
    with open(wdir / "wn.pkl", "wb") as fh:
        pickle.dump(world["wn"], fh)
    (wdir / "params.yml").write_text(
        yaml.safe_dump(world["params"], sort_keys=False))


def load_world(path, district: str, districts: dict | None = None) -> dict:
    """One persisted probe world, in the shape ``Probe.from_world`` reads."""
    path = Path(path)
    wdir = path / "worlds" / district
    if not (wdir / "params.yml").exists():
        raise FileNotFoundError(
            f"{wdir}: probe world not retained (sensor_placement.probe."
            "retain_series was false when the stack was built)")
    with open(wdir / "wn.pkl", "rb") as fh:
        wn = pickle.load(fh)
    districts = districts or yaml.safe_load((path / "districts.yml").read_text())
    return {"params": yaml.safe_load((wdir / "params.yml").read_text()),
            "districts": districts,
            "demand": pd.read_parquet(wdir / "demand.parquet"),
            "pressures": pd.read_parquet(wdir / "pressures.parquet"),
            "flows": pd.read_parquet(wdir / "flows.parquet"),
            "schedule": pd.read_csv(wdir / "schedule.csv", dtype={"node": str}),
            "validation": pd.read_csv(wdir / "validation.csv"),
            "wn": wn, "label": f"{path.name}:{district}"}


def load_probes(path) -> dict:
    """``{district: Probe}`` for every retained world of a stack."""
    path = Path(path)
    districts = yaml.safe_load((path / "districts.yml").read_text())
    return {d: sp.Probe.from_world(load_world(path, d, districts))
            for d in districts["districts"]}


# ==========================================================================
# analysis -- the POC's district_sweep.run_config, minus the grid
# ==========================================================================
def _purity_stability(base: dict, factors, core_purity: float,
                      foreign_max: float) -> pd.DataFrame:
    """Per gauge: how far purity and tier move across the null factors.

    A gauge whose purity swings between thresholds carries a threshold
    artefact, not a quantified mixture; selection drops it on
    ``max_purity_swing``.
    """
    frames = {}
    for f in factors:
        m = mp.remix(base, null_factor=f, core_purity=core_purity,
                     foreign_max=foreign_max)["mixture"]
        frames[f] = m.set_index("id")[["purity", "tier", "w_second"]]
    ids = frames[factors[0]].index
    P = pd.DataFrame({f: frames[f]["purity"].reindex(ids) for f in factors})
    T = pd.DataFrame({f: frames[f]["tier"].reindex(ids) for f in factors})
    return pd.DataFrame({
        "id": ids,
        "purity_min": P.min(axis=1).to_numpy(),
        "purity_max": P.max(axis=1).to_numpy(),
        "purity_swing": (P.max(axis=1) - P.min(axis=1)).to_numpy(),
        "tier_stable": T.nunique(axis=1).eq(1).to_numpy(),
    }).round(4)


def analyze(stack: ProbeStack, probes: dict, beta: float, classes: dict,
            probe: dict, verbose: bool = True,
            season_aligned: bool = False) -> ProbeStack:
    """Every quantity the POC measured, over one stack's K worlds.

    ``season_aligned``: the probes carry seasonality, so the horizon check
    and the mixture use whole-year windows (``mixture_probe.season_windows``).
    The resemblance battery and the settle report keep their phase windows;
    both are descriptive, and neither feeds the gain matrix.
    """
    settle = probe["settle"]
    kw = dict(ref_months=int(settle["ref_months"]), slack=float(settle["slack"]),
              pad=int(settle["pad"]))
    null_factor = float(classes["null_factor"])
    core, foreign = float(classes["core_purity"]), float(classes["foreign_max"])
    factors = tuple(float(f) for f in classes["null_factors"])
    net = stack.network

    # -- resemblance / response battery, per world (POC §4-§5) ---------------
    cells, scores, deltas = [], [], []
    for d, P in probes.items():
        P.params.setdefault("patterns", {})
        cand = sp.candidates(P)
        sc = sp.battery(P, cand, max_lag_h=float(probe["battery_max_lag_h"]))
        sc = sc.merge(cand[["id", "served_own"]], on="id", how="left")
        key = dict(network=net, drift_district=d, beta=float(beta),
                   sim_hash=f"{stack.stack_hash}:{d}")
        sc = sc.assign(**key)
        dl = sp.phase_delta(P).assign(**key)
        scores.append(sc)
        deltas.append(dl)
        cells.append({**key, **ro.cell_row(P, sc, dl, "final"),
                      "source": "solved", "status": "ok", "error": ""})
    stack.cells = pd.DataFrame(cells)
    stack.scores = pd.concat(scores, ignore_index=True)
    stack.deltas = pd.concat(deltas, ignore_index=True)
    top = int(probe["top_carriers"])
    stack.leak = {k: ro.leak_matrix(stack.scores, phase="final", kind=k, top=3)
                  .get(net, pd.DataFrame()) for k in KINDS}
    stack.carriers = ro.signal_carriers(stack.scores, phase="init", top=top)
    stack.carrier_stability = ro.stability(stack.scores, phase="init", top=top)

    # -- settling (POC §2b) ------------------------------------------------
    stack.horizon = mp.horizon_check(probes, min_months=int(settle["min_months"]),
                                     season_aligned=season_aligned, **kw)
    if not stack.horizon["ok"].all():
        short = list(stack.horizon.loc[~stack.horizon["ok"], "drift_district"])
        raise ValueError(
            f"probe stack {stack.stack_hash}: settled window too short for "
            f"{short}; raise sensor_placement.probe.settled_months (or "
            f"n_months) so n_months >= "
            f"{int(stack.horizon['need_n_months'].max())}")
    stack.settle = pd.concat(
        [mp.settle_report(P, ref_months=kw["ref_months"], slack=kw["slack"])
         .assign(drift_district=d) for d, P in probes.items()],
        ignore_index=True)

    # -- the mixture (POC §2c) -------------------------------------------------
    base = mp.build_mixture(probes, null_factor=null_factor, core_purity=core,
                            foreign_max=foreign, verbose=verbose,
                            season_aligned=season_aligned, **kw)
    stack.base = base
    stack.mixture = base["mixture"]
    stack.gains = base["gains"]
    stack.stability = _purity_stability(base, factors, core, foreign)
    stack.diagnostics = mp.mixture_diagnostics(base)
    stack.tier_sensitivity = mp.tier_sensitivity(
        base, factors=factors, purities=(core,), foreign_max=foreign)
    stack.purity_histogram = mp.purity_histogram(base)

    # -- elasticity, dependence, response (POC §2d) ----------------------------
    stack.elasticity = {s: mp.elasticity(base, space=s) for s in SPACES}
    summaries = []
    for kind in KINDS:
        D, R, n = mp.dependence(base, kind=kind)
        stack.dependence[kind] = {"D": D, "R": R, "n": n.to_frame()}
        Dn, Rn, nn = mp.dependence(base, kind=kind, space="native")
        stack.dependence_native[kind] = {"D": Dn, "R": Rn, "n": nn.to_frame()}
        stack.asymmetry[kind] = mp.asymmetry(R)
        summaries.append(mp.dependence_summary(R).reset_index()
                         .rename(columns={"index": "district"})
                         .assign(kind=kind))
        stack.response[(kind, "all")] = mp.response_matrix(base, kind, top=None)
        for tier in ("core", "transition"):
            if ((stack.mixture["tier"] == tier)
                    & (stack.mixture["kind"] == kind)).any():
                stack.response[(kind, tier)] = mp.response_matrix(
                    base, kind, top=None, tier=tier, mixture=stack.mixture)
    stack.dependence_summary = pd.concat(summaries, ignore_index=True)

    # -- verification (POC §2e, §2f) ------------------------------------------
    P0 = probes[sorted(probes)[0]]
    stack.connectivity = pd.concat(
        [mp.core_connectivity(P0, base, k) for k in KINDS], ignore_index=True)
    stack.checks = pd.concat([mp.estimator_checks(base, k, space=s)
                              for k in KINDS for s in SPACES],
                             ignore_index=True)
    stack.channels = cl.channel_diversity(cl.classification(stack))
    if verbose:
        for _, r in stack.channels.iterrows():
            if r["verdict"] != "ok":
                print(f"  !! {r['kind']} channel DEGENERATE: pairwise tv "
                      f"{r['mean_pairwise_tv']:.3f}, {r['distinct_rows']} "
                      f"distinct rows in {r['n_live']} gauges -- kept and "
                      "flagged; fl.preprocessing.channels decides whether it "
                      "feeds the model")
    return stack


# ==========================================================================
# table persistence
# ==========================================================================
def _persist_tables(stack: ProbeStack, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in _CSV:
        frame = getattr(stack, name)
        if len(frame):
            frame.to_csv(path / f"{name}.csv", index=False)
    for name in _PARQUET:
        frame = getattr(stack, name)
        if len(frame):
            frame.to_parquet(path / f"{name}.parquet", index=False)
    for space, E in stack.elasticity.items():
        E.to_parquet(path / f"elasticity_{space}.parquet")
    mat = path / "matrices"
    mat.mkdir(exist_ok=True)
    for tag, deps in (("", stack.dependence),
                      ("native_", stack.dependence_native)):
        for kind, d in deps.items():
            d["D"].to_csv(mat / f"dependence_D_{tag}{kind}.csv")
            d["R"].to_csv(mat / f"dependence_R_{tag}{kind}.csv")
    for kind, A in stack.asymmetry.items():
        A.to_csv(mat / f"asymmetry_{kind}.csv")
    for (kind, use), A in stack.response.items():
        A.to_csv(mat / f"response_{kind}_{use}.csv")
    for kind, A in stack.leak.items():
        if len(A):
            A.to_csv(mat / f"leak_{kind}.csv")


def load_stack(path, with_probes: bool = False) -> ProbeStack:
    """Rebuild a :class:`ProbeStack` from disk (``base`` stays empty)."""
    path = Path(path)
    mf = _read_json(path / "manifest.json") or {}
    if mf.get("status") != "ok":
        raise ValueError(f"{path}: stack status is {mf.get('status')!r}: "
                         f"{mf.get('error', '')}")
    dy = yaml.safe_load((path / "districts.yml").read_text())
    stack = ProbeStack(
        config_id=mf["config_id"], method=mf["method"], network=mf["network"],
        districts=dy["districts"], assets=dy.get("assets") or {},
        status=mf["status"], error=mf.get("error", ""),
        seconds=float(mf.get("seconds", 0.0)), stack_hash=mf["stack_hash"],
        path=path, spec=mf.get("spec", {}),
        src_drift=mf.get("src_hash") != src_hash())
    for name in _CSV:
        f = path / f"{name}.csv"
        if f.exists():
            setattr(stack, name, pd.read_csv(f))
    for name in _PARQUET:
        f = path / f"{name}.parquet"
        if f.exists():
            setattr(stack, name, pd.read_parquet(f))
    for space in SPACES:
        f = path / f"elasticity_{space}.parquet"
        if f.exists():
            stack.elasticity[space] = pd.read_parquet(f)
    mat = path / "matrices"
    for kind in KINDS:
        for tag, deps in (("", stack.dependence),
                          ("native_", stack.dependence_native)):
            fr = mat / f"dependence_R_{tag}{kind}.csv"
            if fr.exists():
                deps[kind] = {
                    "D": pd.read_csv(mat / f"dependence_D_{tag}{kind}.csv",
                                     index_col=0),
                    "R": pd.read_csv(fr, index_col=0), "n": pd.DataFrame()}
        fa = mat / f"asymmetry_{kind}.csv"
        if fa.exists():
            stack.asymmetry[kind] = pd.read_csv(fa, index_col=0)
        fl = mat / f"leak_{kind}.csv"
        if fl.exists():
            stack.leak[kind] = pd.read_csv(fl, index_col=0)
    for f in sorted(mat.glob("response_*.csv")):
        kind, use = f.stem[len("response_"):].rsplit("_", 1)
        stack.response[(kind, use)] = pd.read_csv(f, index_col=0)
    if with_probes:
        stack.probes = load_probes(path)
    return stack


# ==========================================================================
# cache-or-build
# ==========================================================================
def ensure_stack(*, store_root, spec: dict, inputs: dict, world: dict,
                 probe: dict, classes: dict, plan: dict, transitions: dict,
                 coupling: dict, seed: int, verbose: bool = True) -> ProbeStack:
    """Load the stack for ``spec`` if a good one exists, else build it.

    ``inputs`` are the catalog objects a probe world starts from (see
    ``excite.simulate_probe``); ``world`` the target world's resolved blocks.
    """
    h = canonical_hash(spec)
    network, method = spec["network"], spec["partition"]["method"]
    path = stack_dir(store_root, network, method, h)
    mf = _read_json(path / "manifest.json")
    if mf and mf.get("stack_hash") == h and mf.get("status") == "ok":
        if verbose:
            print(f"probe stack {network}/{method}/{h}: cached")
        return load_stack(path)

    t0 = _time.time()
    districts = inputs["districts"]
    names = list(districts["districts"])
    stack = ProbeStack(
        config_id=f"{method}__{h}", method=method, network=network,
        districts=districts["districts"], assets=districts.get("assets") or {},
        seeds=plan["seeds"].assign(
            stack_hash=h,
            to_land_use=[transitions[r[1]]
                         for r in world["scenario"]["income_landuse_mapping"]]),
        stack_hash=h, path=path, spec=spec)
    manifest = {"stack_hash": h, "config_id": stack.config_id,
                "network": network, "method": method, "districts": names,
                "spec": spec, "src_hash": src_hash(),
                "created_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds")}
    path.mkdir(parents=True, exist_ok=True)
    (path / "districts.yml").write_text(yaml.safe_dump(
        {"districts": districts["districts"],
         "assets": districts.get("assets") or {}}, sort_keys=False))
    try:
        regimes = dict(zip(names, world["scenario"]["income_landuse_mapping"]))
        probes = {}
        for d in names:
            if verbose:
                ex = excite.excitation(regimes[d], transitions)
                print(f"probe stack {network}/{method}/{h}: {d} "
                      f"{regimes[d][1]} -> {ex['to_land_use']} "
                      f"({plan['n_months']} x {plan['days_per_month']} d, "
                      f"warm-up {plan['warmup_months']}, seasonality "
                      f"{plan['seasonality_scale']})", flush=True)
            params = excite.probe_params(world, probe, plan, d, regimes[d],
                                         transitions, coupling, seed)
            w = excite.simulate_probe(inputs, params,
                                      label=f"{network}:{h}:{d}")
            if probe.get("retain_series", True):
                _save_world(w, path / "worlds" / d)
            probes[d] = sp.Probe.from_world(w)
        analyze(stack, probes, beta=float(world["scenario"]["beta"]),
                classes=classes, probe=probe, verbose=verbose,
                season_aligned=bool(plan["season_aligned"]))
        stack.probes = probes
        stack.seconds = _time.time() - t0
        _persist_tables(stack, path)
        _write_json(path / "manifest.json",
                    {**manifest, "status": "ok", "error": "",
                     "seconds": round(stack.seconds, 1)})
    except Exception as exc:
        _write_json(path / "manifest.json",
                    {**manifest, "status": "failed",
                     "error": f"{type(exc).__name__}: {exc}"[:2000],
                     "seconds": round(_time.time() - t0, 1)})
        raise
    if verbose:
        print(f"probe stack {network}/{method}/{h}: built in "
              f"{stack.seconds / 60:.1f} min | "
              + str(stack.mixture.groupby("tier").size().to_dict()))
    return stack
