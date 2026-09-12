"""landuse_world.py -- build ONE world in-process, for a chosen network.

Why this exists
---------------
`landuse_engine.py` was written when the land-use model lived only in the
notebook: it *overrode* `build_portfolios`, `evolve_assignments` and
`synthesize_demands`. Those overrides are now dead weight -- the shipped
`fedwater.pipelines.*` nodes carry the sector model, the network-scoped
parameter resolver, natural-sorted node order and `stable_hash`-keyed RNG.
So this module re-implements NOTHING. It calls the shipped nodes in the
order `__default__` does, with the parameters assembled in Python instead of
by Kedro, and writes the result in the layout `fedwater_lab.load_world`
expects.

Two deliberate departures from `kedro run`:

* **the network is an argument, not `conf/base/globals.yml`.** A notebook that
  has to edit a config file to switch networks cannot hold two networks in one
  session, which is exactly what a placement study needs.
* **`anchor_scale` comes from `ANCHOR_SCALE` below, not from the bundle.** It
  is passed EXPLICITLY, so `resolve_params` finds nothing to fill and leaves
  it alone (the resolver only ever fills nulls). Everything else still
  resolves from `profile.yml` in the normal way.

Usage
-----
    import landuse_world as lw
    cfg   = lw.WorldCfg(network="ky7", map_code="LR_LR_LR_LR",
                        tgt_district="District_A", to_land_use="commercial")
    world = lw.build(cfg, root=ROOT)          # ~1-3 min depending on network
    path  = lw.persist(world, out_root)       # -> fedwater_lab.load_world(path)
"""
from __future__ import annotations

import json
import pathlib
import shutil
import time as _time
from dataclasses import dataclass, field, asdict

import pandas as pd
import yaml

# --------------------------------------------------------------------------
# the one seam to fedwater. `bridge.nodes_module` imports a pipeline's
# nodes.py whether or not kedro is installed, so this module works in a bare
# analysis venv exactly as it does inside the project.
# --------------------------------------------------------------------------
from bridge import nodes_module
from worlds import deep_merge

__all__ = ["ANCHOR_SCALE", "WorldCfg", "decode_map", "encode_map", "build",
           "persist", "summary"]


# ======================================================================
# network-scoped constants
# ======================================================================
# HARDCODED on purpose (requested): the notebook chooses the network, so it
# must also carry the one network-scoped value that changes the operating
# point. Provenance, from each bundle's profile.yml:
#   graeme 0.05  -- 5557 L/s of .inp base demand against a ~1100 L/s frontier
#   ky7    0.5   -- 67 L/s of base demand against ~90; tanks clean to ~0.75
#   dtown  0.25  -- measured: mean 107 / peak 268 L/s, pmin 19.5 m, no tank on
#                   its floor. The cliff is between 0.35 and 0.50.
ANCHOR_SCALE = {"graeme": 0.05, "ky7": 0.5, "dtown": 0.25}

# Compact consumption-map codec. Income in position 0, land use in position 1;
# 'M' is disambiguated BY POSITION (medium income vs mixed land use).
_INCOME = {"L": "low", "M": "medium", "H": "high"}
_LANDUSE = {"R": "residential", "M": "mixed", "C": "commercial", "I": "industrial"}
_INCOME_INV = {v: k for k, v in _INCOME.items()}
_LANDUSE_INV = {v: k for k, v in _LANDUSE.items()}


def decode_map(code: str, n_districts: int) -> list[list[str]]:
    """'LR_LM_LC_LR' -> [[income, land_use], ...] in district name order.

    `n_districts` is REQUIRED and never defaulted: `income_landuse_mapping` is
    positional, so a five-token map on a four-district network is a different
    scenario, not a wrong one. `build_portfolios` raises on a length mismatch
    anyway; failing here gives a better message.
    """
    toks = str(code).strip().upper().split("_")
    if len(toks) != n_districts:
        raise ValueError(f"consumption map {code!r} has {len(toks)} tokens for "
                         f"{n_districts} districts")
    bad = [t for t in toks if len(t) != 2 or t[0] not in _INCOME or t[1] not in _LANDUSE]
    if bad:
        raise ValueError(f"unreadable token(s) {bad} in {code!r} "
                         f"(income {sorted(_INCOME)}, land use {sorted(_LANDUSE)})")
    return [[_INCOME[t[0]], _LANDUSE[t[1]]] for t in toks]


def encode_map(mapping) -> str:
    """[[income, land_use], ...] -> 'LR_LM_...'. Inverse of `decode_map`."""
    return "_".join(_INCOME_INV[str(i)] + _LANDUSE_INV[str(lu)] for i, lu in mapping)


def final_map_code(code: str, districts: list[str], tgt: str,
                   to_income: str, to_land_use: str) -> str:
    """The post-drift consumption map -- the world's `final` regime label."""
    toks = code.strip().upper().split("_")
    if tgt in districts:
        toks[districts.index(tgt)] = (_INCOME_INV[to_income] + _LANDUSE_INV[to_land_use])
    return "_".join(toks)


# ======================================================================
# configuration
# ======================================================================
@dataclass
class WorldCfg:
    """Everything that identifies a world. Hashed into `sim_hash`.

    `map_code` is the INITIAL state; (`tgt_district`, `to_income`,
    `to_land_use`) is the FINAL state of the one district that moves. The
    drift *mechanism* (diffusion front, ramp) is untouched -- this round does
    not change the drift engine.
    """
    network: str = "ky7"
    map_code: str = "LR_LR_LR_LR"

    # --- final state -----------------------------------------------------
    tgt_district: str | None = None      # None -> the bundle's own default
    to_income: str = "low"
    to_land_use: str = "commercial"
    seed_node: str | None = None         # None -> profile's drift_seed_nodes, then auto

    # --- dials -----------------------------------------------------------
    beta: float = 0.35
    anchor_scale: float | None = None    # None -> ANCHOR_SCALE[network]
    coupling_variant: str = "baseline"
    close_fraction: float = 0.5

    # --- horizon ---------------------------------------------------------
    n_months: int = 12
    days_per_month: int = 28             # multiple of 7 keeps whole weeks per
                                         # month; the weekday counter is global
                                         # so any value is *correct*, but 28
                                         # makes the weekly means balanced.
    resolution_h: int = 1
    warmup_months: int = 2
    seed: int = 42

    # --- scenario extras (left at the shipped values unless overridden) ---
    growth_chance: float | None = None
    max_neighbors_per_month: int | None = None
    drift_ramp_days: int | None = None
    seasonality_scale: float | None = None
    extra: dict = field(default_factory=dict)   # free-form deep-merge escape

    def resolved_anchor(self) -> float:
        if self.anchor_scale is not None:
            return float(self.anchor_scale)
        if self.network not in ANCHOR_SCALE:
            raise KeyError(f"no ANCHOR_SCALE entry for {self.network!r}; pass "
                           "anchor_scale explicitly")
        return float(ANCHOR_SCALE[self.network])

    def identity(self) -> dict:
        d = asdict(self)
        d["anchor_scale"] = self.resolved_anchor()
        return d


# ======================================================================
# parameter assembly
# ======================================================================
def load_bundle(root, network: str) -> dict:
    """The three files that make a network, plus the .inp path."""
    raw = pathlib.Path(root) / "data" / "01_raw" / network
    inp = raw / "network.inp"
    for p in (inp, raw / "districts.yml", raw / "profile.yml"):
        if not p.exists():
            raise FileNotFoundError(f"{network}: missing {p}")
    return {"network": network, "dir": raw, "inp": inp,
            "districts": yaml.safe_load((raw / "districts.yml").read_text()),
            "profile": yaml.safe_load((raw / "profile.yml").read_text())}


def load_params(root) -> dict:
    """`conf/base/parameters.yml`, read directly.

    Deliberately NOT through a KedroSession: the session resolves
    `${globals:network}`, which is the thing this module exists to bypass.
    """
    p = pathlib.Path(root) / "conf" / "base" / "parameters.yml"
    if not p.exists():
        raise FileNotFoundError(f"no parameters at {p}")
    return yaml.safe_load(p.read_text())


def _overrides(cfg: WorldCfg, districts: list[str]) -> dict:
    """The blocks this world changes, as a deep-merge patch over the base."""
    scen = {
        "n_months": int(cfg.n_months),
        "beta": float(cfg.beta),
        "income_landuse_mapping": decode_map(cfg.map_code, len(districts)),
        "drift": {
            "tgt_district": cfg.tgt_district,     # None leaves the bundle's own
            "to_income": cfg.to_income,
            "to_land_use": cfg.to_land_use,
            "seed_node": cfg.seed_node,
            "warmup_months": int(cfg.warmup_months),
        },
    }
    # A None here would OVERWRITE the bundle/base value with null, which the
    # resolver would then have to fill -- fine for the network-scoped keys
    # (tgt_district, seed_node) and wrong for everything else.
    for k, v in (("growth_chance", cfg.growth_chance),
                 ("max_neighbors_per_month", cfg.max_neighbors_per_month)):
        if v is not None:
            scen["drift"][k] = v
    if cfg.tgt_district is None:
        scen["drift"].pop("tgt_district")         # let the profile fill it

    pat = {}
    if cfg.drift_ramp_days is not None:
        pat["drift_ramp_days"] = int(cfg.drift_ramp_days)
    if cfg.seasonality_scale is not None:
        pat["seasonality_scale"] = float(cfg.seasonality_scale)

    patch = {
        "seed": int(cfg.seed),
        "time": {"n_months": int(cfg.n_months),
                 "days_per_month": int(cfg.days_per_month),
                 "resolution_h": int(cfg.resolution_h)},
        "hydraulics": {"anchor_scale": cfg.resolved_anchor()},
        "coupling": {"variant": cfg.coupling_variant,
                     "close_fraction": float(cfg.close_fraction)},
        "scenario": scen,
    }
    if pat:
        patch["patterns"] = pat
    return deep_merge(patch, cfg.extra or {})


# ======================================================================
# the chain
# ======================================================================
def build(cfg: WorldCfg, root, verbose: bool = True) -> dict:
    """Run the `__default__` simulation chain for one world, in-process.

    Node order and inputs are copied from the shipped `pipeline.py` files, so
    a divergence between this and `kedro run` is a code change upstream, not a
    drift in this module.
    """
    t0 = _time.time()
    NP = nodes_module("network_prep")
    US = nodes_module("urban_scenario")
    DS = nodes_module("demand_synthesis")
    HY = nodes_module("hydraulics")
    SE = nodes_module("sensing")
    SV = nodes_module("sim_validation")
    from fedwater.networks.partition import validate_partition

    bundle = load_bundle(root, cfg.network)
    profile, districts_yml = bundle["profile"], bundle["districts"]
    names = list(districts_yml["districts"])

    base = load_params(root)
    params = deep_merge(base, _overrides(cfg, names))
    wn_raw = _load_inp(bundle["inp"])

    # --- network_prep ---------------------------------------------------
    hyd, scen, val, np_report = NP.resolve_network_parameters(
        profile, params["hydraulics"], params["scenario"], params["validation"])
    params["hydraulics"], params["scenario"], params["validation"] = hyd, scen, val

    wn0, prep_report = NP.configure_network(wn_raw, profile, hyd, params["time"])
    partition_report = validate_partition(wn0, districts_yml)
    wn, gt_boundaries = NP.apply_coupling(wn0, districts_yml, params["coupling"],
                                          params["seed"])

    # --- urban_scenario --------------------------------------------------
    income_factors = US.build_income_factors(params["buildings"])
    landuse_factors = US.build_landuse_factors(income_factors, params["land_use"], scen)
    portfolios = US.build_portfolios(wn, districts_yml, landuse_factors,
                                     income_factors, scen, params["land_use"], hyd)
    schedule = US.build_drift_schedule(wn, districts_yml, profile, scen, params["seed"])
    timeline = US.evolve_assignments(portfolios, schedule, landuse_factors,
                                     income_factors, params["land_use"], scen)

    # --- demand_synthesis + hydraulics -----------------------------------
    demand_raw = DS.synthesize_demands(timeline, params["land_use"],
                                       params["patterns"], params["time"],
                                       params["seed"])
    demand = DS.apply_drift_ramp(demand_raw, schedule, params["patterns"],
                                 params["time"])
    if verbose:
        print(f"[{cfg.network}] solving {len(demand):,} steps x "
              f"{demand.shape[1] - 1} nodes ...", flush=True)
    pressures, flows, demands_sim = HY.run_hydraulics(wn, demand)

    # --- sensing ---------------------------------------------------------
    sensor_true = SE.extract_sensor_series(pressures, flows, profile)
    sensor_series = SE.add_measurement_noise(sensor_true, params["noise"],
                                             params["seed"])
    clients = SE.package_client_datasets(sensor_series, params["time"],
                                         params["start_date"])

    # --- sim_validation --------------------------------------------------
    validation = _validate(SV, wn, demand, demands_sim, pressures, timeline,
                           income_factors, params, val)

    world = {
        "cfg": cfg, "params": params, "bundle": bundle, "profile": profile,
        "districts": districts_yml, "names": names,
        "wn": wn, "wn_configured": wn0,
        "income_factors": income_factors, "landuse_factors": landuse_factors,
        "portfolios": portfolios, "schedule": schedule, "timeline": timeline,
        "demand": demand, "pressures": pressures, "flows": flows,
        "demands_sim": demands_sim, "sensor_series": sensor_series,
        "clients": clients, "gt_boundaries": gt_boundaries,
        "reports": {"network_params": np_report, "network_prep": prep_report,
                    "partition": partition_report, "validation": validation},
        "seconds": _time.time() - t0,
    }
    world["sim_hash"] = sim_hash(cfg)
    if verbose:
        print(summary(world).to_string(), flush=True)
    return world


def _load_inp(path):
    """The .inp as a wntr model. Uses the project's dataset when available."""
    try:
        from fedwater.datasets import WntrNetworkDataset
        return WntrNetworkDataset(filepath=str(path)).load()
    except Exception:
        import wntr
        return wntr.network.WaterNetworkModel(str(path))


def _validate(SV, wn, demand, demands_sim, pressures, timeline, income_factors,
              params, val) -> pd.DataFrame:
    """The five shipped checks. V1-V3 raise upstream; this keeps the report."""
    mass = SV.check_mass_balance(wn, demands_sim, demand, val)
    pres = SV.check_pressures(pressures, demand, val)
    store = SV.check_storage(pressures, wn, val)
    cons = SV.check_consumption_sanity(timeline, income_factors, val)
    peaks = SV.check_peak_factors(demand, timeline, params["time"], val)
    return SV.compile_validation_report(mass, pres, store, cons, peaks)


def sim_hash(cfg: WorldCfg) -> str:
    """Content address of a world. 8 hex chars, stable across processes."""
    from fedwater.hashing import stable_hash
    return f"{stable_hash(json.dumps(cfg.identity(), sort_keys=True, default=str)):08x}"


# ======================================================================
# persistence -- the layout `fedwater_lab.load_world` reads
# ======================================================================
def persist(world: dict, out_root, overwrite: bool = True) -> pathlib.Path:
    """Write a world directory: `manifest.json` + a `clone/` data tree.

    The tree mirrors the catalog paths exactly, so anything already written
    against `data/02_intermediate/pressures.parquet` and friends -- including
    `fedwater_lab.World` -- reads this world with no special case.

    ONE COMPATIBILITY WART, flagged rather than hidden: `World.districts_file`
    is hardcoded to `01_raw/districts_graeme.yml`. The partition is written to
    BOTH that path and the bundle path, so `World.districts` is populated on a
    non-Graeme network without editing the lab package. Remove the duplicate
    once `worlds.py` takes the districts filename from the manifest.
    """
    cfg, params = world["cfg"], world["params"]
    out = pathlib.Path(out_root).expanduser().resolve() / world["sim_hash"]
    if out.exists() and overwrite:
        shutil.rmtree(out)
    clone = out / "clone"
    for sub in ("conf/base", "data/01_raw", "data/02_intermediate",
                "data/03_primary", "data/07_model_output/clients",
                "data/08_reporting"):
        (clone / sub).mkdir(parents=True, exist_ok=True)

    # -- inputs, so the world is self-describing --------------------------
    shutil.copy(world["bundle"]["inp"], clone / "data" / "01_raw" / "network.inp")
    dtext = yaml.safe_dump(world["districts"], sort_keys=False)
    (clone / "data" / "01_raw" / "districts.yml").write_text(dtext)
    (clone / "data" / "01_raw" / "districts_graeme.yml").write_text(dtext)  # see docstring
    (clone / "data" / "01_raw" / "profile.yml").write_text(
        yaml.safe_dump(world["profile"], sort_keys=False))
    (clone / "conf" / "base" / "parameters.yml").write_text(
        yaml.safe_dump(params, sort_keys=False))

    # -- tables -----------------------------------------------------------
    inter, prim = clone / "data" / "02_intermediate", clone / "data" / "03_primary"
    world["portfolios"].to_parquet(inter / "portfolios_t0.parquet", index=False)
    world["demand"].to_parquet(inter / "demand_series.parquet")
    world["pressures"].to_parquet(inter / "pressures.parquet")
    world["flows"].to_parquet(inter / "flows.parquet")
    world["demands_sim"].to_parquet(inter / "demands_simulated.parquet")
    import pickle
    with open(inter / "wn_variant.pkl", "wb") as fh:
        pickle.dump(world["wn"], fh)

    world["schedule"].to_csv(prim / "gt_drift_schedule.csv", index=False)
    world["gt_boundaries"].to_csv(prim / "gt_boundaries.csv", index=False)
    world["income_factors"].to_csv(prim / "income_factors.csv", index=False)
    world["landuse_factors"].to_csv(prim / "landuse_factors.csv", index=False)
    world["timeline"].to_parquet(prim / "assignments_timeline.parquet", index=False)
    world["sensor_series"].to_parquet(prim / "sensor_series.parquet", index=False)

    for district, frame in world["clients"].items():
        frame.to_csv(clone / "data" / "07_model_output" / "clients"
                     / f"{district}.csv", index=False)
    for name, rep in world["reports"].items():
        if isinstance(rep, pd.DataFrame):
            rep.to_csv(clone / "data" / "08_reporting" / f"{name}_report.csv",
                       index=False)

    # -- manifest ---------------------------------------------------------
    names = world["names"]
    init = encode_map(params["scenario"]["income_landuse_mapping"])
    tgt = params["scenario"]["drift"]["tgt_district"]
    manifest = {
        "sim_hash": world["sim_hash"],
        "status": "ok",
        "created": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(float(world["seconds"]), 1),
        "steps_day": int(round(24 / params["time"]["resolution_h"])),
        "world": {
            "network": cfg.network,
            "consumption_map": init,
            "consumption_map_final": final_map_code(
                init, names, tgt, params["scenario"]["drift"]["to_income"],
                params["scenario"]["drift"]["to_land_use"]),
            "drift_district": tgt,
            "variant": params["coupling"]["variant"],
            "beta": params["scenario"]["beta"],
            "anchor_scale": params["hydraulics"]["anchor_scale"],
            "districts": names,
        },
        # `effective` is what the simulator actually ran with. `load_world`
        # deep-merges it OVER the clone's parameters.yml, so anything here
        # wins and anything absent (notably `fl`) falls back to that file.
        "effective": {
            "seed": params["seed"], "start_date": params["start_date"],
            "time": params["time"], "scenario": params["scenario"],
            "hydraulics": params["hydraulics"], "validation": params["validation"],
            "patterns": params["patterns"], "land_use": params["land_use"],
            "coupling": params["coupling"], "noise": params["noise"],
            "sensors": world["profile"]["sensors"],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


# ======================================================================
# read-out
# ======================================================================
def summary(world: dict) -> pd.Series:
    """One line per world: identity, loading, feasibility, drift extent."""
    p, cfg = world["params"], world["cfg"]
    dem, pres = world["demand"], world["pressures"]
    nodes = [c for c in dem.columns if c != "month"]
    total = dem[nodes].sum(axis=1)
    consumers = [n for n in world["portfolios"]["node"].unique() if n in pres.columns]
    v = pres[consumers].to_numpy()
    band = p["validation"]["pressure_band_mca"]
    sch = world["schedule"]
    tgt = p["scenario"]["drift"]["tgt_district"]
    tgt_nodes = set(world["districts"]["districts"][tgt])
    rep = world["reports"]["validation"]
    failed = ""
    if isinstance(rep, pd.DataFrame) and "passed" in rep.columns:
        col = "check" if "check" in rep.columns else rep.columns[0]
        failed = ",".join(rep.loc[~rep["passed"].astype(bool), col].astype(str))
    return pd.Series({
        "sim_hash": world["sim_hash"], "network": cfg.network,
        "map": encode_map(p["scenario"]["income_landuse_mapping"]),
        "drift": f"{tgt}->{p['scenario']['drift']['to_land_use']}",
        "beta": p["scenario"]["beta"], "anchor": p["hydraulics"]["anchor_scale"],
        "months": p["time"]["n_months"], "steps": len(dem),
        "mean_lps": round(float(total.mean()), 1),
        "peak_lps": round(float(total.max()), 1),
        "pmin": round(float(v.min()), 2), "pmax": round(float(v.max()), 2),
        "in_band_%": round(100 * float(((v >= band[0]) & (v <= band[1])).mean()), 2),
        "drifted_nodes": f"{len(set(sch['node']) & tgt_nodes)}/{len(tgt_nodes)}",
        "switch_months": f"{int(sch['drift_month'].min())}..{int(sch['drift_month'].max())}",
        "validation_failed": failed or "-",
        "seconds": round(float(world["seconds"]), 1),
    })
