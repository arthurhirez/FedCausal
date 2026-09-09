"""Network conditioning — turn an arbitrary ``.inp`` into a simulation-ready model.

Principles
----------
1. **The source file is never edited.** Conditioning is a function from a source
   ``.inp`` to (a) a new model, (b) a *recipe* recording every action and its
   rationale, and optionally (c) a conditioned ``.inp`` written elsewhere. The
   recipe is the artifact: it is diffable, hand-editable, and replayable, so a
   per-network fix is documented rather than folded invisibly into a file.
2. **Every action is logged, including the ones that did nothing.** A recipe that
   says "storage_policy: as_is, reason: no tanks" is more useful than silence.
3. **Nothing is silently repaired.** Structural defects (a consumer above every
   available head, a network with no demand at all) are reported and tier the
   network; they are not patched behind the caller's back.
4. **The demand multiplier is freed.** Any global ``DEMAND MULTIPLIER`` in the
   file is folded into the base demands and the option pinned to 1.0. This is
   physically a no-op, and it makes the multiplier available as *our* capacity
   scaling knob without double-counting the file's own scale (Balerma ships
   0.45).

Storage policy is the real decision. Most of this corpus ships ``Duration 0``
with few or no controls, so extending it drains the tanks and produces failures
that say nothing about the network:

* ``as_is``          -- change nothing. Honest, but long horizons are unusable.
* ``freeze_tanks``   -- replace each tank with a fixed-head reservoir at a chosen
  level. Removes storage dynamics entirely; every horizon becomes stable and
  cross-network comparable. Controls referencing tank levels are dropped (and
  recorded). This is the right default when the object of study is the demand
  signal, not tank operation.
* ``synth_controls`` -- keep the tanks and synthesize level-based pump controls
  (the ky16 template: pump ON below a low level, OFF above a high one), pairing
  each pump with the tank it feeds. Keeps storage realistic at the cost of more
  per-network tuning.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr
from wntr.network.controls import Control, ControlAction, ValueCondition

LPS_PER_M3S = 1000.0
SEC_PER_DAY = 86400.0

STORAGE_POLICIES = ("as_is", "freeze_tanks", "synth_controls")


@dataclass
class ConditioningConfig:
    """All conditioning choices in one place. Serialized into every recipe."""

    # options
    fold_demand_multiplier: bool = True
    demand_model: str = "DD"                  # "DD" or "PDD"
    required_pressure_m: float = 15.0         # PDD only
    minimum_pressure_m: float = 0.0           # PDD only

    # time
    duration_h: float | None = 24.0           # None = leave the file's own
    timestep_s: int = 3600

    # demand shape
    flatten_patterns: bool = False            # constant demand (phase 1)
    normalize_pattern_length: int | None = 24  # fix the 23-step KY patterns

    # storage
    storage_policy: str = "as_is"
    freeze_level: str = "initial"             # initial | min | max
    control_on_frac: float = 0.15             # ON below min + frac*range
    control_off_frac: float = 0.85            # OFF above min + frac*range
    # False by default: the author's operating logic beats a synthesized guess.
    # Overriding ky15's 16 shipped controls with generated ones drove its
    # POWER pumps against closed PRVs and produced -438,920 m of head.
    drop_conflicting_controls: bool = False

    # demand synthesis, for files that carry none (C-Town)
    synth_demand_total_cmd: float | None = None
    demand_allocation: str = "pipe_length"    # uniform | pipe_length

    def validate(self):
        if self.storage_policy not in STORAGE_POLICIES:
            raise ValueError(f"storage_policy must be one of {STORAGE_POLICIES}")
        if self.freeze_level not in ("initial", "min", "max"):
            raise ValueError("freeze_level must be initial|min|max")
        if not 0.0 <= self.control_on_frac < self.control_off_frac <= 1.0:
            raise ValueError("need 0 <= control_on_frac < control_off_frac <= 1")
        if self.demand_allocation not in ("uniform", "pipe_length"):
            raise ValueError("demand_allocation must be uniform|pipe_length")
        return self


class Recipe:
    """Ordered log of conditioning actions, with rationale."""

    def __init__(self, network: str, source: str, cfg: ConditioningConfig):
        self.network = network
        self.source = str(source)
        self.wntr_version = wntr.__version__
        self.config = asdict(cfg)
        self.actions: list[dict] = []

    def log(self, step: str, changed: bool, detail: str, **params):
        self.actions.append({"step": step, "changed": bool(changed),
                             "detail": detail, "params": params})

    def to_dict(self) -> dict:
        return {"network": self.network, "source": self.source,
                "wntr_version": self.wntr_version, "config": self.config,
                "actions": self.actions}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"network": self.network, "order": i, "step": a["step"],
             "changed": a["changed"], "detail": a["detail"],
             "params": json.dumps(a["params"], default=str)}
            for i, a in enumerate(self.actions)])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        except ImportError:
            path.with_suffix(".json").write_text(
                json.dumps(self.to_dict(), indent=2, default=str))
        return path


# --------------------------------------------------------------------------
# individual transforms
# --------------------------------------------------------------------------

def fold_demand_multiplier(wn, recipe: Recipe):
    """Move a global DEMAND MULTIPLIER into the base demands; pin the option to 1.

    Physically a no-op. Its purpose is to leave ``demand_multiplier`` free as the
    capacity-scaling knob, so a later lambda sweep is not silently multiplied by
    the file's own hidden scale.
    """
    m = float(wn.options.hydraulic.demand_multiplier or 1.0)
    if abs(m - 1.0) < 1e-12:
        recipe.log("fold_demand_multiplier", False,
                   "multiplier already 1.0", multiplier=m)
        return wn
    n = 0
    for jname in wn.junction_name_list:
        for ts in wn.get_node(jname).demand_timeseries_list:
            if ts.base_value:
                ts.base_value = float(ts.base_value) * m
                n += 1
    wn.options.hydraulic.demand_multiplier = 1.0
    recipe.log("fold_demand_multiplier", True,
               f"folded {m} into {n} demand entries; option pinned to 1.0",
               multiplier=m, n_entries=n)
    return wn


def flatten_demand_patterns(wn, recipe: Recipe):
    """Constant demand: drop every junction demand pattern.

    Removes time as a confound, which is what makes capacity a one-dimensional
    question. Reservoir head patterns and pump speed patterns are left alone --
    they are not consumption and flattening them would change the supply side.
    """
    n = 0
    for jname in wn.junction_name_list:
        for ts in wn.get_node(jname).demand_timeseries_list:
            if ts.pattern_name is not None:
                ts.pattern_name = None
                n += 1
    recipe.log("flatten_demand_patterns", n > 0,
               f"cleared {n} junction demand pattern references "
               f"(reservoir/pump patterns untouched)", n_cleared=n)
    return wn


def normalize_pattern_lengths(wn, recipe: Recipe, target: int):
    """Pad or truncate patterns to ``target`` steps.

    The KY files ship 23-step diurnal patterns. EPANET wraps a pattern when it
    runs out, so a 23-step "daily" pattern walks the cycle backwards one hour
    per day -- which would quietly destroy any hour-of-day or month-aware
    residualization downstream.
    """
    fixed = []
    for pname in wn.pattern_name_list:
        pat = wn.get_pattern(pname)
        mult = np.asarray(pat.multipliers, dtype=float)
        if mult.size == target or mult.size <= 1:
            continue
        if mult.size < target:                       # pad by cyclic repetition
            reps = int(np.ceil(target / mult.size))
            new = np.tile(mult, reps)[:target]
            how = "padded cyclically"
        else:
            new = mult[:target]
            how = "truncated"
        pat.multipliers = new
        fixed.append(f"{pname}:{mult.size}->{target} ({how})")
    recipe.log("normalize_pattern_lengths", bool(fixed),
               "; ".join(fixed) or f"all patterns already {target} steps or scalar",
               target=target, n_fixed=len(fixed))
    return wn


def set_time_and_solver_options(wn, recipe: Recipe, cfg: ConditioningConfig):
    """Pin everything that affects physics, from config, never from file defaults."""
    before = {"duration_h": float(wn.options.time.duration or 0) / 3600,
              "hydraulic_timestep_s": float(wn.options.time.hydraulic_timestep or 0),
              "demand_model": wn.options.hydraulic.demand_model}
    if cfg.duration_h is not None:
        wn.options.time.duration = int(cfg.duration_h * 3600)
        wn.options.time.hydraulic_timestep = int(cfg.timestep_s)
        wn.options.time.report_timestep = int(cfg.timestep_s)
        wn.options.time.pattern_timestep = int(cfg.timestep_s)
    wn.options.hydraulic.demand_model = cfg.demand_model
    if cfg.demand_model.upper().startswith("P"):
        wn.options.hydraulic.required_pressure = cfg.required_pressure_m
        wn.options.hydraulic.minimum_pressure = cfg.minimum_pressure_m
    after = {"duration_h": float(wn.options.time.duration or 0) / 3600,
             "hydraulic_timestep_s": float(wn.options.time.hydraulic_timestep or 0),
             "demand_model": wn.options.hydraulic.demand_model}
    recipe.log("set_time_and_solver_options", before != after,
               f"{before} -> {after}", **after)
    return wn


def _tank_freeze_head(tank, level: str) -> float:
    lvl = {"initial": tank.init_level, "min": tank.min_level,
           "max": tank.max_level}[level]
    return float(tank.elevation) + float(lvl)


def freeze_tanks(wn, recipe: Recipe, level: str = "initial"):
    """Replace every tank with a fixed-head reservoir.

    Links are rewired onto the new reservoir, then the tank is removed. Controls
    that reference a frozen tank's level can no longer fire and are dropped --
    recorded explicitly, because dropping controls is exactly the kind of change
    that must never be invisible.

    The new node is named ``<tank>_FIXED`` rather than reusing the tank's name,
    so nothing in a downstream ID map silently changes meaning.
    """
    tanks = list(wn.tank_name_list)
    if not tanks:
        recipe.log("freeze_tanks", False, "no tanks", level=level)
        return wn

    dropped, mapping = [], {}
    for name in list(wn.control_name_list):
        if any(t in str(wn.get_control(name)) for t in tanks):
            dropped.append(f"{name}: {wn.get_control(name)}")
            wn.remove_control(name)

    for tname in tanks:
        t = wn.get_node(tname)
        head = _tank_freeze_head(t, level)
        new = f"{tname}_FIXED"
        wn.add_reservoir(new, base_head=head, coordinates=t.coordinates)
        node_obj = wn.get_node(new)
        for lname in wn.link_name_list:
            l = wn.get_link(lname)
            if l.start_node_name == tname:
                l.start_node = node_obj
            if l.end_node_name == tname:
                l.end_node = node_obj
        wn.remove_node(tname)
        mapping[tname] = {"reservoir": new, "head_m": round(head, 4)}

    recipe.log("freeze_tanks", True,
               f"froze {len(tanks)} tanks at their {level} level as fixed-head "
               f"reservoirs; dropped {len(dropped)} level-based controls",
               level=level, mapping=mapping, dropped_controls=dropped)
    return wn


def _pump_to_tank_pairs(wn) -> dict[str, list[str]]:
    """Which tank does each pump fill?

    Walk downstream from the pump's discharge node without crossing another
    pump; any tank reached is a tank that pump can fill. Cheap, name-agnostic,
    and it reproduces the pairing in ky16's shipped controls.
    """
    g = nx.Graph()
    pump_links = set(wn.pump_name_list)
    for lname in wn.link_name_list:
        if lname in pump_links:
            continue
        l = wn.get_link(lname)
        if str(getattr(l, "initial_status", "")).upper() == "CLOSED":
            continue
        g.add_edge(l.start_node_name, l.end_node_name)

    pairs = {}
    tanks = set(wn.tank_name_list)
    for pname in wn.pump_name_list:
        p = wn.get_link(pname)
        reached = set()
        for side in (p.end_node_name, p.start_node_name):
            if g.has_node(side):
                comp = nx.node_connected_component(g, side)
                if side == p.end_node_name:
                    reached |= comp & tanks
        pairs[pname] = sorted(reached)
    return pairs


def synthesize_controls(wn, recipe: Recipe, cfg: ConditioningConfig):
    """Add level-based pump controls following the ky16 template.

    ``IF TANK t LEVEL BELOW lo THEN PUMP p OPEN`` /
    ``IF TANK t LEVEL ABOVE hi THEN PUMP p CLOSED``, with lo/hi expressed as
    fractions of each tank's own usable range so the rule ports across networks
    instead of carrying ky16's absolute metres.

    A pump that already appears in an existing control is left alone unless
    ``drop_conflicting_controls`` is set: the author's operating logic beats a
    synthesized guess.
    """
    if not wn.tank_name_list or not wn.pump_name_list:
        recipe.log("synthesize_controls", False,
                   f"needs both tanks ({len(wn.tank_name_list)}) and pumps "
                   f"({len(wn.pump_name_list)})")
        return wn

    existing = {name: str(wn.get_control(name)) for name in wn.control_name_list}
    controlled_pumps = {p for p in wn.pump_name_list
                        if any(f"PUMP {p} " in c for c in existing.values())}

    pairs = _pump_to_tank_pairs(wn)
    added, skipped = [], []
    for pname, tanks in pairs.items():
        if not tanks:
            skipped.append(f"{pname}: fills no tank")
            continue
        if pname in controlled_pumps and not cfg.drop_conflicting_controls:
            skipped.append(f"{pname}: already controlled by the author")
            continue
        tname = tanks[0]                      # nearest/only tank it can fill
        t = wn.get_node(tname)
        lo_lvl = float(t.min_level) + cfg.control_on_frac * float(
            t.max_level - t.min_level)
        hi_lvl = float(t.min_level) + cfg.control_off_frac * float(
            t.max_level - t.min_level)
        pump = wn.get_link(pname)
        for tag, lvl, op, status in [("on", lo_lvl, "<", 1),
                                     ("off", hi_lvl, ">", 0)]:
            cname = f"synth_{pname}_{tag}".replace("~", "").replace("@", "")
            if cname in wn.control_name_list:
                wn.remove_control(cname)
            wn.add_control(cname, Control(
                ValueCondition(t, "level", op, lvl),
                [ControlAction(pump, "status", status)], name=cname))
        added.append(f"{pname}<-{tname} on<{lo_lvl:.2f}m off>{hi_lvl:.2f}m")

    recipe.log("synthesize_controls", bool(added),
               f"added {len(added)} pump on/off pairs; skipped {len(skipped)}",
               on_frac=cfg.control_on_frac, off_frac=cfg.control_off_frac,
               added=added, skipped=skipped)
    return wn


def synthesize_demand(wn, recipe: Recipe, cfg: ConditioningConfig):
    """Allocate a target total demand over junctions that carry none.

    Only fires when the file has zero demand (C-Town, whose demands were the
    withheld part of a calibration challenge) *and* a target total is supplied.
    Without a target the network stays at zero and is tiered as needing demand
    synthesis -- inventing a total would be fabricating the dataset.

    ``pipe_length`` weights each junction by half the length of its incident
    pipes, the standard proxy for served area. ``uniform`` splits evenly.
    """
    current = sum(sum(float(ts.base_value or 0.0)
                      for ts in wn.get_node(j).demand_timeseries_list)
                  for j in wn.junction_name_list)
    if current > 0:
        recipe.log("synthesize_demand", False,
                   f"file already carries {current * LPS_PER_M3S:.2f} L/s")
        return wn
    if cfg.synth_demand_total_cmd is None:
        recipe.log("synthesize_demand", False,
                   "file carries no demand and no target total was given -- "
                   "left at zero, network tiered as needs_demand")
        return wn

    total_m3s = cfg.synth_demand_total_cmd / SEC_PER_DAY
    structural = set()
    for jname in wn.junction_name_list:
        types = set()
        for lname in wn.get_links_for_node(jname):
            types.add(wn.get_link(lname).link_type)
        if types and types.issubset({"Pump", "Valve"}):
            structural.add(jname)
    candidates = [j for j in wn.junction_name_list if j not in structural]

    if cfg.demand_allocation == "pipe_length":
        w = {}
        for jname in candidates:
            L = 0.0
            for lname in wn.get_links_for_node(jname):
                l = wn.get_link(lname)
                if l.link_type == "Pipe":
                    L += 0.5 * float(l.length)
            w[jname] = L
        if sum(w.values()) <= 0:
            w = {j: 1.0 for j in candidates}
    else:
        w = {j: 1.0 for j in candidates}
    tot = sum(w.values())

    for jname, weight in w.items():
        wn.get_node(jname).demand_timeseries_list[0].base_value = \
            total_m3s * weight / tot
    recipe.log("synthesize_demand", True,
               f"allocated {cfg.synth_demand_total_cmd:.0f} m3/day "
               f"({total_m3s * LPS_PER_M3S:.2f} L/s) over {len(candidates)} "
               f"junctions by {cfg.demand_allocation}; {len(structural)} "
               f"structural junctions excluded",
               total_cmd=cfg.synth_demand_total_cmd,
               method=cfg.demand_allocation, n_junctions=len(candidates))
    return wn


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

def condition_network(source: str | Path, cfg: ConditioningConfig | None = None,
                      wn=None) -> tuple[object, Recipe]:
    """Apply the full conditioning chain. Returns (model, recipe).

    Order matters: the multiplier is folded before any demand is read or scaled;
    demand synthesis runs before storage policy so synthesized controls see real
    flows; storage policy runs before options are pinned so a frozen-tank model
    still gets its duration set.
    """
    cfg = (cfg or ConditioningConfig()).validate()
    source = Path(source)
    if wn is None:
        wn = wntr.network.WaterNetworkModel(str(source))
    else:
        wn = copy.deepcopy(wn)
    recipe = Recipe(source.stem, source, cfg)

    if cfg.fold_demand_multiplier:
        fold_demand_multiplier(wn, recipe)
    synthesize_demand(wn, recipe, cfg)
    if cfg.flatten_patterns:
        flatten_demand_patterns(wn, recipe)
    elif cfg.normalize_pattern_length:
        normalize_pattern_lengths(wn, recipe, cfg.normalize_pattern_length)

    if cfg.storage_policy == "freeze_tanks":
        freeze_tanks(wn, recipe, cfg.freeze_level)
    elif cfg.storage_policy == "synth_controls":
        synthesize_controls(wn, recipe, cfg)
    else:
        recipe.log("storage_policy", False, "as_is: tanks and controls untouched",
                   policy="as_is", n_tanks=len(wn.tank_name_list),
                   n_controls=len(wn.control_name_list))

    set_time_and_solver_options(wn, recipe, cfg)
    return wn, recipe


def save_conditioned(wn, path: str | Path) -> Path:
    """Write the conditioned model as a fresh ``.inp`` (never over the source)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wntr.network.write_inpfile(wn, str(path), version=2.2)
    return path


# --------------------------------------------------------------------------
# canonical ID map (item 4, emitted here because conditioning is where the
# role of every element becomes known)
# --------------------------------------------------------------------------

def canonical_id_map(network: str, wn) -> pd.DataFrame:
    """A mapping layer, not a rename.

    IDs inside the ``.inp`` are left alone -- rewriting them would break controls
    and rules and destroy traceability to the source dataset. Downstream code
    (districting, FL clients) speaks the canonical ID; EPANET keeps its own.

    Canonical IDs are type-qualified because Balerma reuses 324 IDs across
    element types (a pipe ``1`` and a junction ``1`` both exist), so a bare ID
    is not a key.
    """
    rows = []
    codes = {"junction": "J", "reservoir": "R", "tank": "T",
             "pipe": "P", "pump": "U", "valve": "V"}

    def role_of_junction(jname) -> str:
        j = wn.get_node(jname)
        base = sum(float(ts.base_value or 0.0) for ts in j.demand_timeseries_list)
        types = {wn.get_link(l).link_type for l in wn.get_links_for_node(jname)}
        deg = len(list(wn.get_links_for_node(jname)))
        if abs(base) < 1e-12 and 0 < deg <= 2 and (types & {"Pump", "Valve"}):
            return "structural"
        if base > 0:
            return "consumer"
        if base < 0:
            return "injection"
        return "transit"

    groups = [("junction", wn.junction_name_list),
              ("reservoir", wn.reservoir_name_list),
              ("tank", wn.tank_name_list),
              ("pipe", wn.pipe_name_list),
              ("pump", wn.pump_name_list),
              ("valve", wn.valve_name_list)]
    for etype, names in groups:
        for i, name in enumerate(sorted(names), start=1):
            rows.append({
                "network": network, "element_type": etype, "source_id": name,
                "canonical_id": f"{network}:{codes[etype]}:{i:05d}",
                "role": role_of_junction(name) if etype == "junction" else etype,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# validation and automatic policy selection
# --------------------------------------------------------------------------

# Pressures beyond this magnitude are not physics, they are solver artifacts.
# A POWER-defined pump forced open with no flow path has head = P/(rho*g*Q),
# which diverges as Q -> 0; ky15 produced -451,836 m that way. Any result past
# this bound invalidates the run rather than being reported as a pressure.
SANITY_HEAD_M = 1000.0


def consumer_junctions(wn) -> set[str]:
    """Junctions with positive base demand that are not pump/valve scaffolding."""
    out = set()
    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        base = sum(float(ts.base_value or 0.0) for ts in j.demand_timeseries_list)
        if base <= 0:
            continue
        types = {wn.get_link(l).link_type for l in wn.get_links_for_node(jname)}
        deg = len(list(wn.get_links_for_node(jname)))
        if 0 < deg <= 2 and types and types.issubset({"Pump", "Valve"}):
            continue
        out.add(jname)
    return out


def solve(wn, sanity: float = SANITY_HEAD_M) -> dict:
    """Run EPANET in a scratch directory and summarise consumer pressures."""
    import os
    import tempfile

    res = {"converged": False, "sane": False, "error": "",
           "pressure_min_m": np.nan, "frac_negative": np.nan,
           "frac_below_floor": np.nan, "worst_nodes": "",
           "delivered_lps": np.nan, "expected_lps": np.nan}
    try:
        with tempfile.TemporaryDirectory() as scratch:
            cwd = os.getcwd()
            try:
                os.chdir(scratch)
                out = wntr.sim.EpanetSimulator(wn).run_sim()
            finally:
                os.chdir(cwd)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return res

    res["converged"] = True
    cons = sorted(consumer_junctions(wn) & set(out.node["pressure"].columns))
    if not cons:
        res["error"] = "no consumer junctions"
        return res
    p = out.node["pressure"][cons]
    vals = p.to_numpy()
    res["sane"] = bool(np.nanmax(np.abs(vals)) <= sanity)
    res["pressure_min_m"] = float(np.nanmin(vals))
    # The upper tail matters as much as the lower one. ky14 run for 24 h puts
    # consumers at 6,494 m (640 bar) because a 335 kW POWER pump deadheads
    # against full tanks -- a floor-only check calls that network healthy.
    res["pressure_max_m"] = float(np.nanmax(vals))
    res["frac_negative"] = float((vals < 0).mean())
    res["worst_nodes"] = ", ".join(p.min(axis=0).sort_values().head(3).index)
    mult = float(wn.options.hydraulic.demand_multiplier or 1.0)
    expected = mult * sum(
        sum(float(ts.base_value or 0.0)
            for ts in wn.get_node(j).demand_timeseries_list) for j in cons)
    res["expected_lps"] = expected * LPS_PER_M3S
    if "demand" in out.node:
        res["delivered_lps"] = float(
            out.node["demand"][cons].to_numpy().mean(axis=0).sum()) * LPS_PER_M3S
    res["_results"] = out
    res["_consumers"] = cons
    return res


def validate_model(wn, pressure_floor_m: float = 15.0,
                   sanity: float = SANITY_HEAD_M) -> dict:
    """Is this model usable? Returns the verdict plus the numbers behind it."""
    r = solve(wn, sanity)
    r.pop("_results", None)
    r.pop("_consumers", None)
    r["frac_below_floor"] = np.nan
    r["overpressure"] = bool(
        not np.isnan(r.get("pressure_max_m", np.nan))
        and r["pressure_max_m"] > sanity)
    r["usable"] = bool(
        r["converged"] and r["sane"]
        and not np.isnan(r["pressure_min_m"])
        and r["frac_negative"] == 0.0)
    r["meets_floor"] = bool(r["usable"]
                            and r["pressure_min_m"] >= pressure_floor_m)
    return r


DEFAULT_POLICY_LADDER = ("as_is", "synth_controls", "freeze_tanks")


def auto_condition(source: str | Path, cfg: ConditioningConfig | None = None,
                   ladder: tuple[str, ...] = DEFAULT_POLICY_LADDER,
                   pressure_floor_m: float = 15.0) -> tuple[object, Recipe, pd.DataFrame]:
    """Try storage policies in increasing intrusiveness; keep the first that works.

    This is the structured answer to "make simulations run on a new network":
    the choice is made by evidence, and every attempt -- including the rejected
    ones and why -- is recorded in the returned attempt log and in the recipe.

    Tiers assigned:
      ``T0_as_is``        usable with no intervention
      ``T1_conditioned``  usable after a storage policy (named)
      ``T2_needs_demand`` carries no demand at all, nothing to assess
      ``T3_infeasible``   no policy on the ladder produces a usable model
    """
    cfg = (cfg or ConditioningConfig()).validate()
    source = Path(source)
    base = wntr.network.WaterNetworkModel(str(source))

    attempts = []
    chosen = None
    for policy in ladder:
        trial_cfg = ConditioningConfig(**{**asdict(cfg), "storage_policy": policy})
        wn, recipe = condition_network(source, trial_cfg, wn=base)
        if not consumer_junctions(wn):
            attempts.append({"network": source.stem, "policy": policy,
                             "usable": False, "reason": "no demand in file",
                             "pressure_min_m": np.nan, "frac_negative": np.nan})
            chosen = (wn, recipe, "T2_needs_demand", policy)
            break
        v = validate_model(wn, pressure_floor_m)
        attempts.append({"network": source.stem, "policy": policy,
                         "usable": v["usable"], "meets_floor": v["meets_floor"],
                         "pressure_min_m": v["pressure_min_m"],
                         "pressure_max_m": v.get("pressure_max_m", np.nan),
                         "frac_negative": v["frac_negative"],
                         "sane": v["sane"], "worst_nodes": v["worst_nodes"],
                         "reason": v["error"] or (
                             "ok" if v["usable"] else
                             "negative consumer pressure" if v["sane"] else
                             f"non-physical overpressure "
                             f"({v.get('pressure_max_m', float('nan')):.0f} m)")})
        if v["usable"]:
            tier = "T0_as_is" if policy == "as_is" else "T1_conditioned"
            chosen = (wn, recipe, tier, policy)
            break

    log = pd.DataFrame(attempts)
    if chosen is None:
        # nothing worked: return the least intrusive model, tiered as infeasible
        wn, recipe = condition_network(
            source, ConditioningConfig(**{**asdict(cfg),
                                          "storage_policy": ladder[0]}), wn=base)
        tier, policy = "T3_infeasible", ladder[0]
    else:
        wn, recipe, tier, policy = chosen

    recipe.log("auto_condition", True,
               f"tier={tier} via storage_policy={policy} after "
               f"{len(attempts)} attempt(s)",
               tier=tier, policy=policy, ladder=list(ladder),
               attempts=attempts)
    return wn, recipe, log


def condition_corpus(paths, cfg: ConditioningConfig | None = None,
                     ladder: tuple[str, ...] = DEFAULT_POLICY_LADDER,
                     pressure_floor_m: float = 15.0,
                     recipe_dir: str | Path | None = None,
                     inp_dir: str | Path | None = None,
                     verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Condition many networks, tier them, and persist recipes and conditioned files.

    Returns ``tiers`` (one row per network), ``attempts`` (every policy tried and
    why it was accepted or rejected), ``recipes`` (the full action log) and
    ``id_map`` (canonical IDs). Nothing raises: a failure is a row.
    """
    import time
    cfg = (cfg or ConditioningConfig()).validate()
    tiers, attempts, recipes, id_maps = [], [], [], []

    for p in paths:
        p = Path(p)
        t0 = time.time()
        try:
            wn, recipe, log = auto_condition(p, cfg, ladder, pressure_floor_m)
            chosen = [a for a in recipe.actions if a["step"] == "auto_condition"][0]
            tier, policy = chosen["params"]["tier"], chosen["params"]["policy"]
            v = validate_model(wn, pressure_floor_m)
            tiers.append({
                "network": p.stem, "tier": tier, "storage_policy": policy,
                "n_attempts": len(log),
                "n_junctions": len(wn.junction_name_list),
                "n_consumers": len(consumer_junctions(wn)),
                "n_tanks": len(wn.tank_name_list),
                "n_controls": len(wn.control_name_list),
                "duration_h": float(wn.options.time.duration or 0) / 3600,
                "usable": v["usable"], "meets_floor": v["meets_floor"],
                "pressure_min_m": v["pressure_min_m"],
                "pressure_max_m": v.get("pressure_max_m", np.nan),
                "frac_negative": v["frac_negative"],
                "overpressure": v.get("overpressure", False),
                "worst_nodes": v["worst_nodes"],
                "seconds": round(time.time() - t0, 2),
            })
            if len(log):
                attempts.append(log)
            recipes.append(recipe.to_frame())
            id_maps.append(canonical_id_map(p.stem, wn))
            if recipe_dir:
                recipe.save(Path(recipe_dir) / f"{p.stem}.yml")
            if inp_dir and tier in ("T0_as_is", "T1_conditioned"):
                save_conditioned(wn, Path(inp_dir) / f"{p.stem}.inp")
        except Exception as exc:
            tiers.append({"network": p.stem, "tier": "T3_infeasible",
                          "storage_policy": "-", "usable": False,
                          "worst_nodes": f"{type(exc).__name__}: {exc}"[:200],
                          "seconds": round(time.time() - t0, 2)})
        if verbose:
            print(f"  {p.stem:12s} {tiers[-1]['seconds']:5.1f}s  "
                  f"{tiers[-1]['tier']:16s} {tiers[-1].get('storage_policy', '-')}")

    def cat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {"tiers": pd.DataFrame(tiers), "attempts": cat(attempts),
            "recipes": cat(recipes), "id_map": cat(id_maps)}
