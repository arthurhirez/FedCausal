"""Experiment specifications: what identifies a *world* and a *run*.

The experiments engine factors every study into two nested identities:

* **World** — everything that determines the simulated physics: sim seed,
  coupling variant/fraction, anchor scale, horizon, consumption map, level
  dial (``beta``), drift block, oracle settings. Expensive; simulated once;
  cached by content hash.
* **Run** — everything that determines learning + analysis on a fixed
  world: FL seed, window stride, batch size, rounds, surrogate counts.
  Cheap; many runs reuse one cached world (this is what makes seed
  replication affordable).

Identity is the **effective configuration**, not the delta: a world's hash
covers the full merged sim parameter blocks (base ``conf/base/parameters.yml``
top-level-replaced by the world override, exactly as Kedro merges
``conf/local``), so editing base parameters correctly invalidates caches.
The run hash covers the full effective ``fl`` block plus the pipeline list.

Consumption maps are written as compact 5-token strings in district order
A..E, each token = income initial + LAND-USE initial. The two alphabets are
distinct — income over {L, M, H}, land use over {R, M, C, I} — and ``M`` is
disambiguated by position (medium income vs mixed land use). The shipped
scenario is ``LR_LM_LC_LR_LR``: all incomes low, land use
residential / mixed / commercial / residential / residential.

NETWORK AXIS: ``network`` names a bundle under ``data/01_raw/`` and is part of
the world identity, so a world hash cannot be reused across networks. It is
NOT a parameter override -- it is written to ``conf/local/globals.yml``, which
is what the catalog interpolates. Adding it changes every existing world hash;
``data/09_experiments/worlds/`` must be re-simulated.

PARTITION AXIS: ``districting`` names the partition method a world is built on
(``manual`` | ``spectral`` | ``girvan_newman`` | ``fast_greedy`` |
``walktrap``); omitted, it is ``globals.districting.active``. Like
``network`` it reaches the catalog through ``conf/local/globals.yml``, and the
world hash folds in the partition's CONTENT id, so rebuilding a partition that
changes the cut invalidates every world built on it.

SENSORS are no longer part of the input: ``sensor_placement`` chooses them per
world, after the probes. Their identity is the ``sensor_placement`` parameter
block (minus the store path, which is deployment, not physics). A world with
``sensor_placement.source: manual`` still hashes the profile's hand-placed
block, since that is what it reads.

The consumption map is POSITIONAL over districts in name order, so a map
written for one network means something different on another. The district
count is now read from the selected bundle's ``districts.yml`` and validated
rather than assumed to be 5.

EVERY-DISTRICT DESIGN: ``drift_all_districts: true`` on a world expands it into
K worlds, one per district of its partition, each drifting that district to
the target ``sensor_placement.probe.transitions`` gives its initial land use
(income kept) -- the probe stack's design as independent worlds. It refuses
an explicit drift target, and a ``sim_seed`` equal to the probe seed (which
would re-simulate the probe worlds: in-sample for the sensors chosen on them).

AUTO HORIZON: ``n_months: auto`` sizes a world from its own drift schedule,
replayed exactly (``urban_scenario.drift_completion``): ``auto_horizon.pre``
drift-free months, the conversion, the last ramp, the settle pad,
``auto_horizon.post`` settled months and ``auto_horizon.spare``. Worlds that
would share a probe stack -- the same world up to the drift target and
``sim_seed`` -- get ONE horizon, the largest need in the group, so they keep
sharing it. The resolved ``n_months`` is identity; ``auto_horizon`` is not.

LAND-USE REFACTOR: this module previously encoded a density axis, with both
token positions drawn from {L, M, H}. Every consumption map and every drift
target changed, so every world hash changed — ``data/09_experiments/worlds/``
must be re-simulated in full.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
from pathlib import Path

import yaml

from fedwater.config import active_partition, load_base_params, load_globals
from fedwater.hashing import _canon, canonical_hash  # noqa: F401  (re-export)
from fedwater.networks import partitions as pstore
from fedwater.networks import profile as nprofile
from fedwater.networks.partition import district_nodes

# Income initials and land-use initials are SEPARATE alphabets. 'M' means
# medium income in position 0 and mixed land use in position 1 — position
# disambiguates, so the two tables must never be merged back into one.
_INCOME = {"L": "low", "M": "medium", "H": "high"}
_LAND_USE = {"R": "residential", "M": "mixed", "C": "commercial",
             "I": "industrial"}
_INCOME_INV = {v: k for k, v in _INCOME.items()}
_LAND_USE_INV = {v: k for k, v in _LAND_USE.items()}
COUPLING_VARIANTS = ("baseline", "partial", "isolated")

# Run-spec keys promoted to first-class axes -> their path inside the fl block.
RUN_KEY_PATHS = {
    "fl_seed": ("training", "seed"),
    "rounds": ("training", "rounds"),
    "batch_size": ("training", "batch_size"),
    "learning_rate": ("training", "learning_rate"),
    "local_epochs": ("training", "local_epochs"),
    "participation": ("training", "participation"),
    "lstm_units": ("model", "lstm_units"),
    "step_size": ("preprocessing", "step_size"),
    "n_surrogates": ("dependence", "n_surrogates"),
    "n_surrogates_expensive": ("dependence", "n_surrogates_expensive"),
}
DEFAULT_PIPELINES = ("fl", "dependence_detection", "drift_attribution")


# --------------------------------------------------------------------------
# consumption-map codec
# --------------------------------------------------------------------------
def decode_map(code: str, n_districts: int) -> list[list[str]]:
    """``'LR_LM_LC_LR_LR' -> [['low','residential'], ['low','mixed'], ...]``."""
    tokens = code.strip().upper().split("_")
    if len(tokens) != n_districts:
        raise ValueError(
            f"consumption_map '{code}' has {len(tokens)} tokens, "
            f"expected {n_districts} (district order A..)."
        )
    out = []
    for t in tokens:
        if len(t) != 2 or t[0] not in _INCOME or t[1] not in _LAND_USE:
            raise ValueError(
                f"consumption_map token '{t}' invalid: expected an income "
                f"initial from {sorted(_INCOME)} followed by a land-use "
                f"initial from {sorted(_LAND_USE)}."
            )
        out.append([_INCOME[t[0]], _LAND_USE[t[1]]])
    return out


def encode_map(mapping: list) -> str:
    """Inverse of :func:`decode_map` (accepts lists or tuples)."""
    return "_".join(_INCOME_INV[i] + _LAND_USE_INV[lu] for i, lu in mapping)


# --------------------------------------------------------------------------
# network bundles
# --------------------------------------------------------------------------
def default_network(project_root: Path) -> str:
    """The active network, resolved the way Kedro resolves it.

    ``conf/local/globals.yml`` merges OVER ``conf/base/globals.yml`` -- that is
    how the catalog picks its bundle, so spec expansion must read it the same
    way. Reading base alone let the engine hash and build a world as one
    network while a plain ``kedro run`` in the same project used another.
    """
    network = load_globals(project_root).get("network")
    if network is None:
        raise FileNotFoundError(
            f"No `network` key found in {project_root}/conf/base/globals.yml "
            "(or conf/local/globals.yml). It names a bundle directory under "
            "data/01_raw/.")
    return network


def default_partition(project_root: Path) -> str:
    """The active partition method (``globals.districting.active``)."""
    return active_partition(load_globals(project_root))


def bundle(project_root: Path, network: str, method: str | None = None) -> dict:
    """One network bundle, read through partition ``method``.

    ``districts`` is the PARTITION's document (``partitions/<method>/``), which
    is what a world is built on; ``manual_districts`` is the hand-authored
    root file. A partition that has not been built raises -- the engine
    builds it first (``ExperimentEngine.ensure_partitions``).
    """
    root = pstore.bundle_dir(project_root, network)
    inp = root / "network.inp"
    districts_path = root / pstore.DISTRICTS_FILE
    profile_path = root / "profile.yml"
    for path in (inp, districts_path, profile_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Network bundle '{network}' is incomplete: {path} is missing. "
                f"A bundle needs network.inp, districts.yml and profile.yml.")
    method = pstore.check_method(method or default_partition(project_root))
    part = pstore.read_partition(project_root, network, method)
    profile = yaml.safe_load(profile_path.read_text())
    return {"network": network, "inp": inp, "method": method,
            "districts": part["districts"],
            "manual_districts": yaml.safe_load(districts_path.read_text()),
            "partition": pstore.partition_meta(part["manifest"], profile),
            "profile": profile}


# --------------------------------------------------------------------------
# canonical hashing — identity of effective configuration
# --------------------------------------------------------------------------
# `_canon` / `canonical_hash` now live in fedwater.hashing (same algorithm, so
# every existing hash is unchanged) and are re-exported above.


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (patch or {}).items():
        out[k] = _deep_merge(out[k], v) if (
            isinstance(v, dict) and isinstance(out.get(k), dict)) else copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------
# axis expansion — how a study section becomes a list of specs
# --------------------------------------------------------------------------
def _axis_values(v):
    """A grid axis is a list, or ``{range: N}`` / ``{range: [a, b]}``."""
    if isinstance(v, dict) and set(v) == {"range"}:
        r = v["range"]
        return list(range(*r)) if isinstance(r, (list, tuple)) else list(range(int(r)))
    if isinstance(v, list):
        return v
    return [v]


def expand_axes(section) -> list[dict]:
    """Expand a ``worlds``/``runs`` section into explicit spec dicts.

    Accepted forms:
      * a list of explicit dicts (used verbatim), or
      * a dict with any of:
          ``fixed`` — constants applied to every spec;
          ``grid``  — cartesian product over its axes;
          ``zip``   — a list of dicts treated as *paired* profiles
                      (crossed against the grid, never against each other).
    Precedence on key collision: fixed < grid < zip.
    """
    if section is None:
        return [{}]
    if isinstance(section, list):
        return [copy.deepcopy(s) for s in section]
    fixed = section.get("fixed", {}) or {}
    grid = section.get("grid", {}) or {}
    zipped = section.get("zip", None) or [{}]
    axes = {k: _axis_values(v) for k, v in grid.items()}
    keys = list(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)) if keys else [()]:
        g = dict(zip(keys, combo))
        for z in zipped:
            out.append({**copy.deepcopy(fixed), **copy.deepcopy(g), **copy.deepcopy(z)})
    return out


# --------------------------------------------------------------------------
# resolution — spec + base params -> full-block override + effective config
# --------------------------------------------------------------------------
STORE_REF = "${globals:probe_store}"


def resolve_world(world: dict, base_params: dict, network: str,
                  n_districts: int, profile: dict | None = None,
                  partition: dict | None = None) -> dict:
    """Build the FULL top-level parameter blocks a world writes to
    ``conf/local`` (Kedro merges destructively at the top level — partial
    blocks silently erase sibling keys, the documented gotcha), plus the
    effective sim configuration and its content hash.

    NETWORK-SCOPED PARAMETERS are resolved FIRST, from the bundle's
    ``profile.yml``, before anything else reads ``base_params``. Keys such as
    ``hydraulics.anchor_scale`` and ``scenario.income_landuse_mapping`` are
    ``null`` in ``conf/base/parameters.yml`` precisely because their correct
    value is a fact about the network; filling them here is what puts the
    resolved value into ``effective`` and therefore into the world's content
    hash, so editing a profile correctly invalidates that network's worlds.
    ``networks.profile.resolve_params`` only ever fills nulls, so a study that
    sets one of those keys explicitly still wins, and applying it again inside
    a plain ``kedro run`` changes nothing.

    Note on ``land_use``: the sector table, intensities and mixes live in a
    base-only block and are NOT part of the override, exactly like
    ``buildings`` and ``patterns``. They still reach the hash through
    ``effective``, so editing them invalidates caches correctly; a study that
    wants to sweep them goes through ``sim_overrides``.

    ``partition`` is ``networks.partitions.partition_meta`` for the partition
    the world is built on (``bundle()["partition"]``). Omitted, the world is
    described as the manual partition with no content id -- enough for
    hashing tests, never for a real build.
    """
    base, _ = nprofile.resolve_params(base_params, profile or {})
    partition = partition or {"method": pstore.MANUAL, "partition_id": None}
    scenario = copy.deepcopy(base["scenario"])
    n_months = int(world.get("n_months", base["time"]["n_months"]))
    scenario["n_months"] = n_months
    if "consumption_map" in world:
        scenario["income_landuse_mapping"] = decode_map(
            world["consumption_map"], n_districts)
    if "beta" in world:
        scenario["beta"] = float(world["beta"])
    drift_patch = world.get("drift", {}) or {}
    scenario["drift"] = _deep_merge(scenario["drift"], drift_patch)
    if ("tgt_district" in drift_patch and "seed_node" not in drift_patch
            and drift_patch["tgt_district"]
            != base["scenario"]["drift"].get("tgt_district")):
        # Retargeted drift must not inherit the base district's seed node
        # (node "2" belongs to District_D); leave unset for the auto-picker.
        scenario["drift"]["seed_node"] = None
    if scenario["drift"].get("seed_node") is not None:
        scenario["drift"]["seed_node"] = str(scenario["drift"]["seed_node"])

    override = {
        "seed": int(world.get("sim_seed", base["seed"])),
        "time": {**copy.deepcopy(base["time"]), "n_months": n_months},
        "hydraulics": {**copy.deepcopy(base["hydraulics"]),
                       "anchor_scale": float(world.get(
                           "anchor_scale", base["hydraulics"]["anchor_scale"]))},
        "coupling": {**copy.deepcopy(base["coupling"]),
                     **(world.get("coupling") or {})},
        "scenario": scenario,
    }
    # time grid and demand-pattern keys a study may set directly (they also
    # reach the probes when sensor_placement.probe.horizon / seasonality is
    # `world`)
    if "days_per_month" in world:
        override["time"]["days_per_month"] = int(world["days_per_month"])
    patterns_patch = {k: world[k] for k in ("seasonality_scale",
                                            "drift_ramp_days") if k in world}
    if patterns_patch:
        override["patterns"] = _deep_merge(base["patterns"], patterns_patch)
    if world.get("oracle"):
        override["oracle"] = _deep_merge(base["oracle"], world["oracle"])
    # `placement` is the sensor_placement block's shorthand
    if world.get("placement"):
        override["sensor_placement"] = _deep_merge(
            base["sensor_placement"], world["placement"])
    for key, patch in (world.get("sim_overrides") or {}).items():
        if key == "fl":
            raise ValueError("sim_overrides cannot touch 'fl' (run-level).")
        if key == "districting":
            raise ValueError("sim_overrides cannot touch 'districting': the "
                             "partition is chosen with the `districting` axis "
                             "and configured in parameters.yml.")
        override[key] = _deep_merge(override.get(key, base.get(key, {})),
                                    patch) \
            if isinstance(patch, dict) else copy.deepcopy(patch)
    if "sensor_placement" in override:
        # base was read with ${globals:...} RESOLVED; written back literally,
        # the clone would store probe stacks under its own relative path
        # instead of the engine's shared store.
        override["sensor_placement"]["store"] = STORE_REF

    effective = {k: v for k, v in base.items() if k != "fl"}
    effective = {**effective, **override}
    # The network is part of the world's identity but NOT of its parameter
    # override: it reaches the catalog through conf/local/globals.yml. Putting
    # it in `effective` is what stops a Graeme world being reused under a
    # D-Town label.
    effective["network"] = network
    # THE PARTITION. Its method reaches the catalog through globals; its
    # CONTENT id is what makes the hash honest: a rebuilt partition with a
    # different cut is a different world, one that reproduces the cut is not.
    # The districting dials themselves are not identity -- only their result
    # is -- so the block is dropped.
    effective.pop("districting", None)
    # How an auto horizon is SIZED is not physics; the n_months (and
    # warm-up) it resolves to are, and they are already in the blocks.
    effective.pop("auto_horizon", None)
    effective["partition"] = {"method": partition["method"],
                              "partition_id": partition.get("partition_id")}
    # SENSOR PLACEMENT. Sensors are an OUTPUT of the world now (chosen by
    # `sensor_placement` from the probe stacks), so the world's identity is
    # the rule that chooses them, i.e. the `sensor_placement` block -- minus
    # `store`, which is where the stacks live, not what they are. A manual
    # placement reads the profile's block, so for `source: manual` that block
    # IS identity, exactly as before the refactor: a world simulated under an
    # old hand placement must not be served under a new one.
    placement = copy.deepcopy(effective.get("sensor_placement") or {})
    placement.pop("store", None)
    effective["sensor_placement"] = placement
    source = placement.get("source", "dynamic")
    manual_sensors = (copy.deepcopy((profile or {}).get("sensors") or {})
                      if source == "manual" else {})
    if source == "manual":
        effective["manual_sensors"] = manual_sensors
    slots = placement.get("slots") or {}
    per_district = {k: int(v.get("pure", 0)) + int(v.get("mixed", 0))
                    for k, v in slots.items()}
    flat = {
        "network": network,
        "sim_seed": override["seed"],
        "n_months": n_months,
        "variant": override["coupling"]["variant"],
        "close_fraction": float(override["coupling"].get("close_fraction", 0.0)),
        "anchor_scale": override["hydraulics"]["anchor_scale"],
        "beta": float(scenario["beta"]),
        "consumption_map": encode_map(scenario["income_landuse_mapping"]),
        "drift_district": scenario["drift"]["tgt_district"],
        "drift_seed_node": scenario["drift"].get("seed_node"),
        "drift_to_income": scenario["drift"]["to_income"],
        "drift_to_land_use": scenario["drift"]["to_land_use"],
        # PROVENANCE, not identity: `flat` becomes a column of runs.parquet
        # (engine.collect flattens `manifest["world"]`, which IS `flat`), and
        # that table is one row per run, so every entry here must be a
        # scalar. The chosen sensors themselves are a world ARTIFACT
        # (data/03_primary/sensor_placement.csv); these are the counts.
        "districting": partition["method"],
        "partition_id": partition.get("partition_id"),
        "placement_source": source,
        "slots_pressure": per_district.get("pressure", 0),
        "days_per_month": int(effective["time"]["days_per_month"]),
        "seasonality_scale": float(effective["patterns"].get(
            "seasonality_scale", 1.0)),
        "probe_horizon": (placement.get("probe") or {}).get("horizon",
                                                            "planned"),
        "probe_seasonality": (placement.get("probe") or {}).get("seasonality",
                                                                "none"),
        "slots_flow": per_district.get("flow", 0),
        "n_pressure_sensors": (
            sum(len(c.get("pressure", [])) for c in manual_sensors.values())
            if source == "manual"
            else per_district.get("pressure", 0) * n_districts),
        "n_flow_sensors": (
            sum(len(c.get("flow", [])) for c in manual_sensors.values())
            if source == "manual"
            else per_district.get("flow", 0) * n_districts),
    }
    return {"sim_hash": canonical_hash(effective), "override": override,
            "effective": effective, "flat": flat, "network": network,
            "globals": {"network": network,
                        "districting": {"active": partition["method"],
                                        "methods": [partition["method"]]}}}


def resolve_run(run: dict, base_params: dict,
                pipelines=DEFAULT_PIPELINES) -> dict:
    """Build the effective ``fl`` block for one run and its content hash.
    Promoted keys map through :data:`RUN_KEY_PATHS`; anything else goes via
    ``fl_overrides`` (deep-merged last)."""
    fl = copy.deepcopy(base_params["fl"])
    unknown = set(run) - set(RUN_KEY_PATHS) - {"fl_overrides"}
    if unknown:
        raise ValueError(
            f"Unknown run-spec keys {sorted(unknown)}; promoted keys are "
            f"{sorted(RUN_KEY_PATHS)}, everything else via 'fl_overrides'.")
    for key, path in RUN_KEY_PATHS.items():
        if key in run and run[key] is not None:
            node = fl
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = run[key]
    fl = _deep_merge(fl, run.get("fl_overrides", {}))
    identity = {"fl": fl, "pipelines": list(pipelines)}
    flat = {k: run.get(k) for k in RUN_KEY_PATHS if k in run}
    return {"run_hash": canonical_hash(identity), "fl": fl,
            "pipelines": list(pipelines), "flat": flat}


# --------------------------------------------------------------------------
# validation — fail at spec time, not two minutes into EPANET
# --------------------------------------------------------------------------
def validate_world(resolved: dict, districts: dict) -> None:
    """Raise ``ValueError`` on specs that cannot produce a meaningful world.

    Drift rule (per design review, carried over from the density axis): the
    drift target must change **at least one** of (income, land_use) relative
    to the district's initial state on the consumption map — either alone is
    enough, both is fine, neither is a no-op drift and is rejected.
    """
    eff, flat = resolved["effective"], resolved["flat"]
    names = list(district_nodes(districts))
    net = resolved.get("network", "?")
    if flat["variant"] not in COUPLING_VARIANTS:
        raise ValueError(f"coupling.variant '{flat['variant']}' not in "
                         f"{COUPLING_VARIANTS}.")
    if not 0.0 <= flat["close_fraction"] <= 1.0:
        raise ValueError(f"close_fraction {flat['close_fraction']} outside [0, 1].")
    if flat["beta"] < 0.0:
        raise ValueError(f"scenario.beta {flat['beta']} must be >= 0 "
                         "(0 = shape-only, 1 = full level effect).")
    tgt = flat["drift_district"]
    if tgt not in names:
        raise ValueError(
            f"[network={net}] drift.tgt_district '{tgt}' not one of {names}.")
    seed_node = flat["drift_seed_node"]
    tgt_nodes = district_nodes(districts)[tgt]
    if seed_node is not None and str(seed_node) not in tgt_nodes:
        raise ValueError(
            f"[network={net}] drift.seed_node '{seed_node}' is not a node of "
            f"{tgt}, whose junctions look like {tgt_nodes[:5]}... "
            f"({len(tgt_nodes)} total).\n"
            "A seed node id is NETWORK-SPECIFIC. Either drop `seed_node` from "
            "the study and let the auto-picker choose per network (the "
            "district's largest-base-demand junction), or pin the study to the "
            "network the id belongs to with `network: <name>` under "
            "worlds.fixed.")

    from fedwater.pipelines.urban_scenario.nodes import drift_front
    drift_front(eff["scenario"]["drift"], 1)      # int >= 1, or auto + months

    land_use_codes = set(eff["land_use"]["mix"])
    if flat["drift_to_land_use"] not in land_use_codes:
        raise ValueError(
            f"drift.to_land_use '{flat['drift_to_land_use']}' is not a class "
            f"in land_use.mix ({sorted(land_use_codes)}).")

    mapping = eff["scenario"]["income_landuse_mapping"]
    init_income, init_land_use = mapping[names.index(tgt)]
    if (flat["drift_to_income"] == init_income
            and flat["drift_to_land_use"] == init_land_use):
        raise ValueError(
            f"Drift on {tgt} is a no-op: to_income/to_land_use "
            f"({flat['drift_to_income']}, {flat['drift_to_land_use']}) equal "
            f"the initial map state ({init_income}, {init_land_use}). At least "
            f"one of income or land use must change.")

    validate_placement(eff)

    warmup = int(eff["scenario"]["drift"].get("warmup_months", 0))
    if warmup < 1:
        # apply_drift_ramp blends the new regime against the month BEFORE the
        # switch, so a node drifting at month 0 has nothing to blend from.
        raise ValueError(f"drift.warmup_months must be >= 1 (got {warmup}).")
    if flat["n_months"] < warmup + 2:
        raise ValueError(
            f"n_months={flat['n_months']} leaves no room for drift "
            f"(warmup_months={warmup}); need at least warmup + 2.")


def validate_placement(eff: dict) -> None:
    """The ``sensor_placement`` block, checked before any probe is solved."""
    from fedwater.placement.excite import validate_transitions

    sp = eff.get("sensor_placement") or {}
    source = sp.get("source", "dynamic")
    if source not in ("dynamic", "manual"):
        raise ValueError(f"sensor_placement.source {source!r} is not "
                         "'dynamic' or 'manual'.")
    if source == "manual":
        if not eff.get("manual_sensors"):
            raise ValueError("sensor_placement.source is 'manual' but the "
                             "bundle profile has no `sensors:` block.")
        return
    slots = sp.get("slots") or {}
    if not slots or set(slots) - {"flow", "pressure"}:
        raise ValueError(f"sensor_placement.slots must be keyed by flow / "
                         f"pressure, got {sorted(slots)}.")
    for kind, v in slots.items():
        if min(int(v.get("pure", 0)), int(v.get("mixed", 0))) < 0 \
                or int(v.get("pure", 0)) + int(v.get("mixed", 0)) == 0:
            raise ValueError(f"sensor_placement.slots.{kind} needs a "
                             f"non-negative, non-zero pure + mixed, got {v}.")
    from fedwater.placement.excite import probe_modes
    from fedwater.placement.mixture_probe import SEASON_MONTHS
    modes = probe_modes(sp.get("probe") or {})
    seasonal = (modes["seasonality"] == "world"
                and float(eff["patterns"].get("seasonality_scale", 1.0)) > 0)
    if seasonal:
        warmup = (int(eff["scenario"]["drift"]["warmup_months"])
                  if modes["horizon"] == "world"
                  else int(sp["probe"]["warmup_months"]))
        if warmup < SEASON_MONTHS:
            raise ValueError(
                "sensor_placement.probe.seasonality is `world` and the world "
                f"has seasonality, but the probe warm-up is {warmup} month(s): "
                f"whole-year windows need >= {SEASON_MONTHS} (raise "
                "drift.warmup_months, or set probe.seasonality: none).")
    try:
        int((sp.get("probe") or {}).get("seed"))
    except (TypeError, ValueError):
        raise ValueError("sensor_placement.probe.seed must be an int, got "
                         f"{(sp.get('probe') or {}).get('seed')!r}.") from None
    from fedwater.placement.store import gate, mixture_rule
    gate(sp.get("classes") or {})
    mixture_rule(sp.get("classes") or {})
    ref = (sp.get("selection_coupling") or {}).get("variant")
    if ref not in ("baseline", "isolated"):
        raise ValueError("sensor_placement.selection_coupling.variant must be "
                         f"baseline or isolated, got {ref!r}.")
    validate_transitions(sp["probe"]["transitions"], eff["land_use"])


def with_anchor(resolved: dict, anchor_scale: float) -> dict:
    """A copy of a resolved world with a different ``anchor_scale`` (and the
    correspondingly different content hash). Used by the engine's optional
    auto-anchor retry ladder — policy (b) of the physics-constraint design."""
    out = copy.deepcopy(resolved)
    out["override"]["hydraulics"]["anchor_scale"] = float(anchor_scale)
    out["effective"]["hydraulics"]["anchor_scale"] = float(anchor_scale)
    out["flat"]["anchor_scale"] = float(anchor_scale)
    out["sim_hash"] = canonical_hash(out["effective"])
    return out


def with_beta(resolved: dict, beta: float) -> dict:
    """A copy of a resolved world at a different level dial ``beta`` (and the
    correspondingly different content hash).

    The land-use analogue of :func:`with_anchor`, and the cheaper of the two
    feasibility levers: ``beta`` moves only the demand LEVEL implied by the
    land-use map, leaving the temporal signatures — the part that survives the
    FL scaler — untouched. Lowering it therefore buys hydraulic headroom at
    little cost in regime legibility, whereas lowering ``anchor_scale`` moves
    the whole network's operating point and is a cross-world confound.
    """
    out = copy.deepcopy(resolved)
    out["override"]["scenario"]["beta"] = float(beta)
    out["effective"]["scenario"]["beta"] = float(beta)
    out["flat"]["beta"] = float(beta)
    out["sim_hash"] = canonical_hash(out["effective"])
    return out


def auto_seed_node(district: str, districts: dict, inp_path: Path) -> str:
    """Deterministic drift seed for a district: its junction with the largest
    base demand in the ``.inp``. Skips zero-demand trunk junctions (the
    node-110 class of bug the label factory exposed).

    The rule itself lives in ``fedwater.networks.partition`` and is shared with
    ``build_drift_schedule``, so the engine and a plain ``kedro run`` cannot
    pick different seeds. This wrapper reads base demands straight out of the
    ``.inp`` text so spec expansion stays free of a wntr load.
    """
    demands = _inp_base_demands(Path(inp_path))
    nodes = district_nodes(districts)[district]
    carrying = {n: demands.get(n, 0.0) for n in nodes if demands.get(n, 0.0) > 0}
    if not carrying:
        raise ValueError(f"No demand-carrying junction found in {district}.")
    return max(sorted(carrying), key=carrying.get)


def _inp_base_demands(inp_path: Path) -> dict[str, float]:
    demands, in_junctions = {}, False
    for line in inp_path.read_text().splitlines():
        s = line.strip()
        if s.upper().startswith("[JUNCTIONS]"):
            in_junctions = True
            continue
        if in_junctions:
            if s.startswith("["):
                break
            if not s or s.startswith(";"):
                continue
            parts = re.split(r"[\s\t]+", s.split(";")[0].strip())
            if len(parts) >= 3:
                try:
                    demands[parts[0]] = float(parts[2])
                except ValueError:
                    continue
            elif len(parts) == 2:
                demands[parts[0]] = 0.0
    return demands


# --------------------------------------------------------------------------
# study loading
# --------------------------------------------------------------------------
def load_studies(project_root: Path) -> dict:
    path = Path(project_root) / "conf/base/experiments.yml"
    if not path.exists():
        raise FileNotFoundError(f"No study file at {path}.")
    cfg = yaml.safe_load(path.read_text()) or {}
    return cfg


def study_partitions(name: str, project_root: Path) -> list[tuple[str, str]]:
    """``(network, method)`` pairs a study's worlds are built on.

    Read from the axes alone, so the engine can build missing partitions
    BEFORE ``expand_study`` needs to read them.
    """
    cfg = load_studies(project_root)
    study = cfg["studies"][name]
    fallback = default_network(project_root)
    fallback_method = default_partition(project_root)
    return sorted({(w.get("network", fallback),
                    pstore.check_method(w.get("districting", fallback_method)))
                   for w in expand_axes(study.get("worlds"))})


# --------------------------------------------------------------------------
# every-district design and auto horizon
# --------------------------------------------------------------------------
_DRIFT_TARGET_KEYS = ("tgt_district", "to_income", "to_land_use", "seed_node")
AUTO_HORIZON_KEYS = ("pre_months", "post_months", "spare_months")


def all_district_worlds(world: dict, base_params: dict, network: str,
                        districts: dict, profile: dict | None,
                        partition: dict | None) -> list[dict]:
    """``drift_all_districts: true`` -> one world per district, else ``[world]``.

    District ``k`` drifts to ``probe.transitions[its initial land use]`` with
    its income kept -- exactly the excitation its probe world applies, so
    ``verify`` compares like with like. The world's own front, grid and seed
    are untouched. Refused: an explicit drift target (it would be silently
    overwritten), and ``sim_seed == probe.seed`` for dynamic sensors (the
    worlds would reproduce the probe worlds the sensors were chosen on).
    """
    from fedwater.placement.excite import validate_transitions

    world = copy.deepcopy(world)
    if not world.pop("drift_all_districts", False):
        return [world]
    drift = dict(world.get("drift") or {})
    clash = sorted(set(drift) & set(_DRIFT_TARGET_KEYS))
    if clash:
        raise ValueError(
            f"drift_all_districts sets the drift target of every world; "
            f"remove drift.{clash} from the study.")
    names = list(district_nodes(districts))
    probe = {k: v for k, v in world.items() if k != "n_months"}
    eff = resolve_world(probe, base_params, network, len(names), profile,
                        partition)["effective"]
    sp = eff["sensor_placement"]
    transitions = validate_transitions(sp["probe"]["transitions"],
                                       eff["land_use"])
    if (sp.get("source", "dynamic") == "dynamic"
            and int(eff["seed"]) == int(sp["probe"]["seed"])):
        raise ValueError(
            f"drift_all_districts with sim_seed {eff['seed']} equal to "
            "sensor_placement.probe.seed: these worlds would re-simulate the "
            "probe worlds the sensors were selected on (in-sample). Use a "
            "different sim_seed.")
    out = []
    for d, (income, land_use) in zip(names,
                                     eff["scenario"]["income_landuse_mapping"]):
        w = copy.deepcopy(world)
        w["drift"] = {**drift, "tgt_district": d, "to_income": str(income),
                      "to_land_use": transitions[str(land_use)]}
        out.append(w)
    return out


def auto_horizon_config(base_params: dict, world: dict) -> dict:
    """``auto_horizon`` (parameters.yml) patched by the world's own block."""
    cfg = {**(base_params.get("auto_horizon") or {}),
           **(world.get("auto_horizon") or {})}
    missing = [k for k in AUTO_HORIZON_KEYS if k not in cfg]
    unknown = sorted(set(cfg) - set(AUTO_HORIZON_KEYS))
    if missing or unknown:
        raise ValueError(f"auto_horizon needs exactly {AUTO_HORIZON_KEYS}; "
                         f"missing {missing}, unknown {unknown}.")
    cfg = {k: int(v) for k, v in cfg.items()}
    if cfg["pre_months"] < 1 or cfg["post_months"] < 1 or cfg["spare_months"] < 0:
        raise ValueError(f"auto_horizon {cfg}: pre/post >= 1, spare >= 0.")
    return cfg


def horizon_need(resolved: dict, wn, districts: dict, partition: dict,
                 horizon: dict) -> dict:
    """One world's minimal horizon, from its EXACT drift schedule.

    ``need = last_switch + ceil(drift_ramp_days / days_per_month) + pad
    + post_months + spare_months`` -- the settled window of
    ``mixture_probe.season_windows`` (final phase + ``settle.pad``) followed by
    ``post_months`` whole months and the spare. ``pre`` is the warm-up the
    world already carries: the first switch is at ``warmup_months``.
    """
    from fedwater.pipelines.urban_scenario.nodes import drift_completion

    eff = resolved["effective"]
    done = drift_completion(wn, districts, partition, eff["scenario"],
                            int(eff["seed"]))
    dpm = int(eff["time"]["days_per_month"])
    ramp_days = float(eff["patterns"].get("drift_ramp_days", 0) or 0)
    ramp = math.ceil(ramp_days / dpm) if ramp_days > 0 else 0
    pad = int(eff["sensor_placement"]["probe"]["settle"]["pad"])
    need = (done["last_switch"] + ramp + pad + horizon["post_months"]
            + horizon["spare_months"])
    return {**done, "ramp_months": ramp, "pad": pad, "need": int(need)}


def _group_key(network: str, method: str, world: dict) -> str:
    """Worlds that differ only in drift target and sim_seed share a stack."""
    k = copy.deepcopy(world)
    for key in ("sim_seed", "n_months"):
        k.pop(key, None)
    k["drift"] = {a: b for a, b in (k.get("drift") or {}).items()
                  if a not in _DRIFT_TARGET_KEYS}
    return canonical_hash({"network": network, "method": method, "world": k})


def _probe_fits(wn, districts, partition, resolved) -> bool:
    """Would the world's probe schedule fit its horizon (engine pre-flight)?"""
    from fedwater.placement import excite

    eff = resolved["effective"]
    sp = eff.get("sensor_placement") or {}
    if sp.get("source", "dynamic") != "dynamic":
        return True
    try:
        excite.horizon_plan(wn, districts, partition, sp["probe"],
                            {k: eff[k] for k in ("time", "scenario",
                                                 "patterns")})
    except ValueError:
        return False
    return True


def expand_study(name: str, project_root: Path) -> dict:
    """Resolve one named study into validated world specs x run specs.

    ``network`` is an ordinary world axis: put it under ``worlds.fixed`` or
    ``worlds.grid`` to sweep it. When absent it falls back to
    ``conf/base/globals.yml``. Each world carries its own bundle, so a study
    MAY mix networks -- though a positional ``consumption_map`` is only
    comparable across networks with the same district count and ordering.

    ``drift_all_districts`` and ``n_months: auto`` are resolved here (module
    docstring); an auto world needs the network's graph, loaded once per
    network.
    """
    project_root = Path(project_root)
    cfg = load_studies(project_root)
    if name not in cfg.get("studies", {}):
        raise KeyError(f"Study '{name}' not found; available: "
                       f"{sorted(cfg.get('studies', {}))}.")
    study = cfg["studies"][name]
    base = load_base_params(project_root)
    fallback = default_network(project_root)
    fallback_method = default_partition(project_root)

    pipelines = tuple(study.get("pipelines", DEFAULT_PIPELINES))
    entries, bundles = [], {}
    for w in expand_axes(study.get("worlds")):
        w = copy.deepcopy(w)
        network = w.pop("network", fallback)
        method = w.pop("districting", fallback_method)
        key = (network, method)
        if key not in bundles:
            bundles[key] = bundle(project_root, network, method)
        b = bundles[key]
        districts = b["districts"]
        n_districts = len(district_nodes(districts))
        expanded = all_district_worlds(w, base, network, districts,
                                       b["profile"], b["partition"])
        for wk in expanded:
            horizon = None
            if wk.get("n_months") == "auto":
                horizon = auto_horizon_config(base, wk)
                wk.pop("n_months")
                wk.pop("auto_horizon", None)
                if "warmup_months" not in (wk.get("drift") or {}):
                    wk = _deep_merge(wk, {"drift": {
                        "warmup_months": horizon["pre_months"]}})
            elif "auto_horizon" in wk:
                raise ValueError("auto_horizon is read only with "
                                 "n_months: auto.")
            wk, resolved = _resolve_seeded(wk, base, network, n_districts, b)
            entries.append({"w": wk, "key": key, "resolved": resolved,
                            "horizon": horizon,
                            "all_districts": len(expanded) > 1
                            or bool(w.get("drift_all_districts"))})

    # -- auto horizons: one per stack-sharing group ---------------------------
    groups: dict[str, list] = {}
    for e in entries:
        if e["horizon"] is not None:
            groups.setdefault(_group_key(*e["key"], e["w"]), []).append(e)
    models = {}
    for members in groups.values():
        network, method = members[0]["key"]
        b = bundles[(network, method)]
        if network not in models:
            import wntr
            models[network] = wntr.network.WaterNetworkModel(
                str(pstore.bundle_dir(project_root, network) / "network.inp"))
        wn = models[network]
        for e in members:
            e["need"] = horizon_need(e["resolved"], wn, b["districts"],
                                     b["partition"], e["horizon"])
        n = max(e["need"]["need"] for e in members)
        n_districts = len(district_nodes(b["districts"]))

        def at(e, n):
            return resolve_world({**e["w"], "n_months": n}, base, network,
                                 n_districts, b["profile"], b["partition"])

        # the probes may need more than the worlds (horizon: world): grow the
        # group's horizon until the pre-flight's own plan fits
        for _ in range(600):
            if _probe_fits(wn, b["districts"], b["partition"],
                           at(members[0], n)):
                break
            n += 1
        else:
            raise ValueError(f"auto horizon: no n_months <= {n} fits the "
                             f"probe schedule of {network}/{method}.")
        for e in members:
            e["w"] = {**e["w"], "n_months": n}
            e["resolved"] = at(e, n)

    worlds = []
    for e in entries:
        resolved = e["resolved"]
        flat = resolved["flat"]
        flat["n_months_auto"] = e["horizon"] is not None
        flat["drift_last_switch"] = (e["need"]["last_switch"]
                                     if e["horizon"] is not None else None)
        flat["all_districts"] = e["all_districts"]
        if e["horizon"] is not None:
            _check_auto_horizon(resolved, e["horizon"], e["need"])
        validate_world(resolved, bundles[e["key"]]["districts"])
        worlds.append(resolved)
    runs = [resolve_run(r, base, pipelines) for r in expand_axes(study.get("runs"))]

    if len({w["sim_hash"] for w in worlds}) != len(worlds):
        raise ValueError(f"Study '{name}' contains duplicate world specs.")
    if len({r["run_hash"] for r in runs}) != len(runs):
        raise ValueError(f"Study '{name}' contains duplicate run specs.")
    return {"name": name, "worlds": worlds, "runs": runs,
            "pipelines": pipelines,
            "networks": sorted({n for n, _ in bundles}),
            "partitions": sorted(bundles),
            "harvest": tuple(study.get("harvest",
                             ("validation", "drift", "ladder", "c4",
                              "dependence"))),
            "retain": tuple(study.get("retain", ())),
            "root": cfg.get("root", "data/09_experiments"),
            "description": study.get("description", "")}


def _resolve_seeded(w: dict, base: dict, network: str, n_districts: int,
                    b: dict) -> tuple[dict, dict]:
    """``resolve_world`` with the drift seed node pinned explicitly."""
    resolved = resolve_world(w, base, network, n_districts, b["profile"],
                             b["partition"])
    if resolved["flat"]["drift_seed_node"] is None:
        # Precedence (explicit > partition seed source > auto) lives in
        # ONE function, shared with build_drift_schedule, so the engine
        # and a plain `kedro run` cannot pick different origins. The seed
        # source is `partition_meta` -- the profile's drift_seed_nodes for
        # the manual partition, none for a generated one. The auto-picker
        # passed here is the .inp-TEXT one, which keeps spec expansion
        # free of a wntr load; it agrees with the model-based picker in
        # networks.partition on every district of every bundle.
        tgt = resolved["flat"]["drift_district"]
        node = nprofile.resolve_seed_node(
            None, b["partition"], tgt,
            lambda: auto_seed_node(tgt, b["districts"], b["inp"]))
        w = _deep_merge(w, {"drift": {"seed_node": node}})
        resolved = resolve_world(w, base, network, n_districts,
                                 b["profile"], b["partition"])
    return w, resolved


def _check_auto_horizon(resolved: dict, horizon: dict, need: dict) -> None:
    """With seasonality on, both sides must hold whole years (season_windows)."""
    from fedwater.placement.mixture_probe import SEASON_MONTHS

    eff = resolved["effective"]
    if float(eff["patterns"].get("seasonality_scale", 0) or 0) <= 0:
        return
    warmup = int(eff["scenario"]["drift"]["warmup_months"])
    if warmup < SEASON_MONTHS or horizon["post_months"] < SEASON_MONTHS:
        raise ValueError(
            f"n_months: auto with seasonality on needs >= {SEASON_MONTHS} "
            f"drift-free months (warm-up {warmup}) and auto_horizon."
            f"post_months >= {SEASON_MONTHS} (got {horizon['post_months']}).")
