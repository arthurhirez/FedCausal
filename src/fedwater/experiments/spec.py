"""Experiment specifications: what identifies a *world* and a *run*.

The experiments engine factors every study into two nested identities:

* **World** — everything that determines the simulated physics: sim seed,
  coupling variant/fraction, anchor scale, horizon, consumption map, drift
  block, oracle settings. Expensive; simulated once; cached by content hash.
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
A..E, each token = income initial + density initial over {L, M, H}; e.g.
the shipped scenario is ``LL_LM_LH_LL_LL`` (all incomes low; densities
low/medium/high/low/low).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

import yaml

_CODE = {"L": "low", "M": "medium", "H": "high"}
_CODE_INV = {v: k for k, v in _CODE.items()}
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
def decode_map(code: str, n_districts: int = 5) -> list[list[str]]:
    """``'LL_LM_LH_LL_LL' -> [['low','low'], ['low','medium'], ...]``."""
    tokens = code.strip().upper().split("_")
    if len(tokens) != n_districts:
        raise ValueError(
            f"consumption_map '{code}' has {len(tokens)} tokens, "
            f"expected {n_districts} (district order A..)."
        )
    out = []
    for t in tokens:
        if len(t) != 2 or t[0] not in _CODE or t[1] not in _CODE:
            raise ValueError(
                f"consumption_map token '{t}' invalid: expected two of "
                f"{sorted(_CODE)} (income initial + density initial)."
            )
        out.append([_CODE[t[0]], _CODE[t[1]]])
    return out


def encode_map(mapping: list) -> str:
    """Inverse of :func:`decode_map` (accepts lists or tuples)."""
    return "_".join(_CODE_INV[i] + _CODE_INV[d] for i, d in mapping)


# --------------------------------------------------------------------------
# canonical hashing — identity of effective configuration
# --------------------------------------------------------------------------
def _canon(obj):
    """Normalize to hash-stable primitives: sort-insensitive dicts, lists,
    ints-for-integral-floats (0.0 == 0), numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return int(obj) if obj.is_integer() else obj
    if hasattr(obj, "item"):  # numpy scalar
        return _canon(obj.item())
    return obj


def canonical_hash(obj, n: int = 12) -> str:
    payload = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:n]


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
def resolve_world(world: dict, base_params: dict) -> dict:
    """Build the FULL top-level parameter blocks a world writes to
    ``conf/local`` (Kedro merges destructively at the top level — partial
    blocks silently erase sibling keys, the documented gotcha), plus the
    effective sim configuration and its content hash."""
    base = base_params
    scenario = copy.deepcopy(base["scenario"])
    n_months = int(world.get("n_months", base["time"]["n_months"]))
    scenario["n_months"] = n_months
    if "consumption_map" in world:
        scenario["income_density_mapping"] = decode_map(world["consumption_map"])
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
    if world.get("oracle"):
        override["oracle"] = _deep_merge(base["oracle"], world["oracle"])
    for key, patch in (world.get("sim_overrides") or {}).items():
        if key == "fl":
            raise ValueError("sim_overrides cannot touch 'fl' (run-level).")
        override[key] = _deep_merge(base.get(key, {}), patch) \
            if isinstance(patch, dict) else copy.deepcopy(patch)

    effective = {k: v for k, v in base.items() if k != "fl"}
    effective = {**effective, **override}
    flat = {
        "sim_seed": override["seed"],
        "n_months": n_months,
        "variant": override["coupling"]["variant"],
        "close_fraction": float(override["coupling"].get("close_fraction", 0.0)),
        "anchor_scale": override["hydraulics"]["anchor_scale"],
        "consumption_map": encode_map(scenario["income_density_mapping"]),
        "drift_district": scenario["drift"]["tgt_district"],
        "drift_seed_node": scenario["drift"].get("seed_node"),
        "drift_to_income": scenario["drift"]["to_income"],
        "drift_to_density": scenario["drift"]["to_density"],
    }
    return {"sim_hash": canonical_hash(effective), "override": override,
            "effective": effective, "flat": flat}


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

    Drift rule (per design review): the drift target must change **at least
    one** of (income, density) relative to the district's initial state on
    the consumption map — either alone is enough, both is fine, neither is
    a no-op drift and is rejected.
    """
    eff, flat = resolved["effective"], resolved["flat"]
    names = list(districts["districts"])
    if flat["variant"] not in COUPLING_VARIANTS:
        raise ValueError(f"coupling.variant '{flat['variant']}' not in "
                         f"{COUPLING_VARIANTS}.")
    if not 0.0 <= flat["close_fraction"] <= 1.0:
        raise ValueError(f"close_fraction {flat['close_fraction']} outside [0, 1].")
    tgt = flat["drift_district"]
    if tgt not in names:
        raise ValueError(f"drift.tgt_district '{tgt}' not one of {names}.")
    seed_node = flat["drift_seed_node"]
    if seed_node is not None and str(seed_node) not in [
            str(n) for n in districts["districts"][tgt]]:
        raise ValueError(f"drift.seed_node '{seed_node}' is not a node of {tgt}.")

    mapping = eff["scenario"]["income_density_mapping"]
    init_income, init_density = mapping[names.index(tgt)]
    if (flat["drift_to_income"] == init_income
            and flat["drift_to_density"] == init_density):
        raise ValueError(
            f"Drift on {tgt} is a no-op: to_income/to_density "
            f"({flat['drift_to_income']}, {flat['drift_to_density']}) equal the "
            f"initial map state ({init_income}, {init_density}). At least one "
            f"of income or density must change.")

    warmup = int(eff["scenario"]["drift"].get("warmup_months", 0))
    if flat["n_months"] < warmup + 2:
        raise ValueError(
            f"n_months={flat['n_months']} leaves no room for drift "
            f"(warmup_months={warmup}); need at least warmup + 2.")


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


def auto_seed_node(district: str, districts: dict, inp_path: Path) -> str:
    """Deterministic drift seed for a district: its junction with the largest
    base demand in the ``.inp``. Skips zero-demand trunk junctions (the
    node-110 class of bug the label factory exposed)."""
    demands = _inp_base_demands(Path(inp_path))
    nodes = [str(n) for n in districts["districts"][district]]
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


def expand_study(name: str, project_root: Path) -> dict:
    """Resolve one named study into validated world specs x run specs."""
    project_root = Path(project_root)
    cfg = load_studies(project_root)
    if name not in cfg.get("studies", {}):
        raise KeyError(f"Study '{name}' not found; available: "
                       f"{sorted(cfg.get('studies', {}))}.")
    study = cfg["studies"][name]
    base = yaml.safe_load((project_root / "conf/base/parameters.yml").read_text())
    districts = yaml.safe_load(
        (project_root / "data/01_raw/districts_graeme.yml").read_text())
    inp = project_root / "data/01_raw/Graeme.inp"

    pipelines = tuple(study.get("pipelines", DEFAULT_PIPELINES))
    worlds = []
    for w in expand_axes(study.get("worlds")):
        resolved = resolve_world(w, base)
        if resolved["flat"]["drift_seed_node"] is None:
            node = auto_seed_node(resolved["flat"]["drift_district"],
                                  districts, inp)
            w = _deep_merge(w, {"drift": {"seed_node": node}})
            resolved = resolve_world(w, base)
        validate_world(resolved, districts)
        worlds.append(resolved)
    runs = [resolve_run(r, base, pipelines) for r in expand_axes(study.get("runs"))]

    if len({w["sim_hash"] for w in worlds}) != len(worlds):
        raise ValueError(f"Study '{name}' contains duplicate world specs.")
    if len({r["run_hash"] for r in runs}) != len(runs):
        raise ValueError(f"Study '{name}' contains duplicate run specs.")
    return {"name": name, "worlds": worlds, "runs": runs,
            "pipelines": pipelines,
            "harvest": tuple(study.get("harvest",
                             ("validation", "drift", "ladder", "c4",
                              "dependence"))),
            "retain": tuple(study.get("retain", ())),
            "root": cfg.get("root", "data/09_experiments"),
            "description": study.get("description", "")}
