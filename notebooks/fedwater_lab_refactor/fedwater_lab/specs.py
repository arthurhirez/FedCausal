"""The experiment axes, as data.

A leaf module: it imports nothing from the rest of the package, so every
other module can depend on it. A `CellSpec` is the complete description of
one trainable cell -- upstream (placement, channels, transform, scaling,
window geometry) plus model and training. Two specs that hash the same
describe the same computation, which is what makes the store a cache instead
of an append-only log.

Deliberately *not* in the spec: anything derived from the world (warm-up,
drift target, regime map, phase boundaries). Those are read from the
manifest at load time, so a spec can never disagree with the world it ran
against. The world's `sim_hash` enters the key separately.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Sequence

__all__ = ["CLASSES", "TRANSFORMS", "SCALINGS", "KINDS", "Geometry",
           "ModelCfg", "TrainCfg", "LossTerms", "Schedule", "CellSpec",
           "POC", "expand"]

# Which SLOTS a spec draws from (`sensor_placement.csv`'s `slot_class`).
# `manual` is not a class to filter on -- it is the world's own placement
# source (see `World.placement_source`), not something a spec chooses.
CLASSES = ("pure", "mixed")

TRANSFORMS = ("none", "slog", "diff", "deseason_own", "deseason_shared")

# `minmax_ref` is fl_preprocessing's own per-client scaler (level destroyed
# per client); `shared` refits one pooled MinMax on the pooled reference
# windows (level preserved across clients); `none` returns physical units;
# `zscore` standardises per client-channel.
SCALINGS = ("minmax_ref", "shared", "none", "zscore")

KINDS = ("pressure", "flow")


def _norm(v):
    """Canonical form for hashing: sorted keys, plain types, no float noise."""
    if isinstance(v, dict):
        return {k: _norm(v[k]) for k in sorted(v) if v[k] is not None}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 10)
    return v


@dataclass(frozen=True)
class Geometry:
    """Window geometry, in `fl.preprocessing` terms."""

    interval_agg_h: int = 3
    window_size: int = 84
    step_size: int = 12
    reference_months: int | None = None      # None -> the world's warm-up
    label_threshold: float = 0.75
    feature_range: tuple[float, float] = (-1.0, 1.0)

    @property
    def window_hours(self) -> int:
        return int(self.interval_agg_h) * int(self.window_size)

    @property
    def whole_days(self) -> bool:
        return self.window_hours % 24 == 0

    def label(self) -> str:
        return (f"a{self.interval_agg_h}w{self.window_size}"
                f"s{self.step_size}")

    def as_preprocessing(self, reference_months: int) -> dict:
        """The `fl.preprocessing` block this geometry implies."""
        return {"interval_agg_h": int(self.interval_agg_h),
                "window_size": int(self.window_size),
                "step_size": int(self.step_size),
                "reference_months": int(self.reference_months
                                        if self.reference_months is not None
                                        else reference_months),
                "label_threshold": float(self.label_threshold),
                "feature_range": [float(self.feature_range[0]),
                                  float(self.feature_range[1])]}


@dataclass(frozen=True)
class ModelCfg:
    lstm_units: int = 30
    reg_ratio: float = 0.5

    @property
    def latent_dim(self) -> int:
        return 2 * int(self.lstm_units)

    def as_fl(self) -> dict:
        return {"lstm_units": int(self.lstm_units),
                "reg_ratio": float(self.reg_ratio)}


@dataclass(frozen=True)
class TrainCfg:
    rounds: int = 5
    local_epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 64
    participation: float = 1.0
    averaging: str = "equal"
    proto_alpha: float = 0.2
    infonce_temperature: float = 0.02
    fl_seed: int = 0
    device: str = "auto"
    # UPSTREAM knob (`fl.training.proto_weight`, read by `FPLTrainer`):
    # weight of the FPL prototype loss. None = not set = the protocol's own
    # weight 1.0, and the key is left out of the spec entirely, so every run
    # key computed before this field existed is unchanged.
    proto_weight: float | None = None

    def as_fl(self) -> dict:
        out = self._as_fl()
        if self.proto_weight is not None:
            out["proto_weight"] = float(self.proto_weight)
        return out

    def _as_fl(self) -> dict:
        return {"rounds": int(self.rounds),
                "local_epochs": int(self.local_epochs),
                "learning_rate": float(self.learning_rate),
                "batch_size": int(self.batch_size),
                "participation": float(self.participation),
                "averaging": str(self.averaging),
                "proto_alpha": float(self.proto_alpha),
                "infonce_temperature": float(self.infonce_temperature),
                "seed": int(self.fl_seed),
                "device": str(self.device)}


@dataclass(frozen=True)
class LossTerms:
    """LAB-SIDE local objective (`train.SnapshotFPLTrainer` re-implements
    `FPLTrainer._local_update` when a spec carries one; None = upstream loop).

        loss = aer_loss
             + proto_weight * proto_loss        (from round `proto_start`)
             + diff * MSE(d/dt dec, d/dt x)      over the full window
             + spec * MSE(|F dec_y|, |F x_y|)    FFT magnitudes, steps 1..W-2,
                                                 per-window mean removed,
                                                 orthonormal (Parseval scale)
             + var  * mean_d relu(var_gamma - std_batch(z_d))

    `diff` and `spec` penalise exactly what the flat-line minimum loses (the
    within-window swing); a flat output pays the target's full derivative /
    spectral energy. `var` is the VICReg variance hinge: it keeps each latent
    dimension's batch spread above `var_gamma`, against the prototype term's
    pull of every window onto its month mean. `LossTerms()` reproduces the
    upstream objective exactly (asserted by the lab's equivalence check).
    """

    proto_weight: float = 1.0
    proto_start: int = 1
    diff: float = 0.0
    spec: float = 0.0
    var: float = 0.0
    var_gamma: float = 0.1

    def __post_init__(self):
        for k in ("proto_weight", "diff", "spec", "var", "var_gamma"):
            if getattr(self, k) < 0:
                raise ValueError(f"LossTerms.{k} must be >= 0")
        if self.proto_start < 0:
            raise ValueError("LossTerms.proto_start must be >= 0")

    def as_dict(self) -> dict:
        return {"proto_weight": float(self.proto_weight),
                "proto_start": int(self.proto_start),
                "diff": float(self.diff), "spec": float(self.spec),
                "var": float(self.var), "var_gamma": float(self.var_gamma)}

    def label(self) -> str:
        d = LossTerms().as_dict()
        parts = [f"{k[0]}{v:g}" if k != "proto_start" else f"ps{v}"
                 for k, v in self.as_dict().items()
                 if v != d[k] and k != "var_gamma"]
        if self.var and self.var_gamma != d["var_gamma"]:
            parts.append(f"g{self.var_gamma:g}")
        return "L" + ("-".join(parts) if parts else "default")


@dataclass(frozen=True)
class Schedule:
    """WHEN the model sees which months. None on a spec = the default: every
    month at once, every round (what POC_02 trains).

    mode="commissioning"  train only on months [lo, hi] (`months`), then
        freeze. The rest of the horizon is only ever ENCODED, so the model
        never sees the drift -- the realistic counterpart of a scaler fitted
        on the commissioning period. `rounds` / `local_epochs` still come
        from `training`.
    mode="streaming"  prequential: the horizon is cut into blocks of
        `block_months`; for each block in order the CURRENT model is scored
        on it first (unseen), then trained on it for `rounds_per_block`
        rounds, carrying the global and local weights and the global
        prototypes forward. Adam state restarts each block.

    `warm_months` (streaming) trains the first block for `warm_rounds`
    instead, so the model does not enter block 1 untrained.
    """

    mode: str = "commissioning"
    months: tuple[int, int] | None = None      # commissioning: inclusive
    block_months: int = 6                      # streaming
    rounds_per_block: int = 5                  # streaming
    warm_rounds: int | None = None             # streaming: first block

    def __post_init__(self):
        if self.mode not in ("commissioning", "streaming"):
            raise KeyError(f"schedule mode {self.mode!r}")
        if self.mode == "commissioning":
            if self.months is None or len(self.months) != 2:
                raise ValueError("commissioning needs months=(lo, hi)")
            if self.months[0] > self.months[1]:
                raise ValueError(f"months out of order: {self.months}")
        if self.block_months < 1 or self.rounds_per_block < 1:
            raise ValueError("block_months and rounds_per_block must be >= 1")

    def as_dict(self) -> dict:
        d = {"mode": self.mode,
             "months": None if self.months is None else list(self.months)}
        if self.mode == "streaming":
            d.update(block_months=int(self.block_months),
                     rounds_per_block=int(self.rounds_per_block),
                     warm_rounds=(None if self.warm_rounds is None
                                  else int(self.warm_rounds)))
        return d

    def label(self) -> str:
        if self.mode == "commissioning":
            return f"C{self.months[0]}-{self.months[1]}"
        w = "" if self.warm_rounds is None else f"w{self.warm_rounds}"
        return f"S{self.block_months}x{self.rounds_per_block}{w}"


@dataclass(frozen=True)
class CellSpec:
    """One trainable cell. `key()` is its identity; `label()` is for humans.

    Sensor selection is three filters, applied in this order against a
    world's `sensor_placement.csv` (see `data.sensor_table`):

    1. `kinds`   -- pressure / flow.
    2. `classes` -- pure / mixed (`slot_class`). Ignored on a world whose
       placement source is `manual` (it has no classes to filter).
    3. `include` / `exclude` -- explicit sensor ids (`sensor_placement.csv`'s
       `sensor` column, e.g. `"q_P-13"`), applied AFTER 1-2: `include` adds
       matching sensors back even if `kinds`/`classes` would have dropped
       them, `exclude` removes them even if kinds/classes would have kept
       them. The two must be disjoint -- a sensor cannot be asked for and
       refused in the same spec.

    Which world a spec is run against -- and therefore whether it was built
    with dynamic or manual sensors -- is NOT part of the spec (see the module
    docstring: nothing derived from the world belongs here). `World` carries
    that; `data.sensor_table`/`resolve_placement` are the join of a `World`
    and a `CellSpec`.
    """

    kinds: tuple[str, ...] = KINDS
    classes: tuple[str, ...] = CLASSES
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    transform: str = "none"
    scaling: str = "minmax_ref"
    geometry: Geometry = field(default_factory=Geometry)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainCfg = field(default_factory=TrainCfg)
    noise: bool = False                  # sensor noise when building series
    cap: int | None = None               # windows per client, None = all
    snapshot_rounds: str | tuple[int, ...] = "sparse"   # sparse|all|explicit
    snapshot_points: int = 1500
    keep_weights: str = "final"          # final|none|all_rounds
    loss_terms: LossTerms | None = None  # None = upstream `_local_update`
    # Flip each flow sensor so its dominant direction over the reference
    # months is positive (`data.orientation`). Applied client-side, before
    # the transform and the scaler. False = the raw EPANET sign (pipe
    # orientation), and the key is left out of the spec.
    orient: bool = False
    schedule: Schedule | None = None     # None = all months, every round

    # ------------------------------------------------------------ validation
    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise KeyError(f"transform {self.transform!r} not in {TRANSFORMS}")
        if self.scaling not in SCALINGS:
            raise KeyError(f"scaling {self.scaling!r} not in {SCALINGS}")
        bad = set(self.kinds) - set(KINDS)
        if bad or not self.kinds:
            raise KeyError(f"kinds must be a non-empty subset of {KINDS}, "
                           f"got {self.kinds!r}")
        bad = set(self.classes) - set(CLASSES)
        if bad or not self.classes:
            raise KeyError(f"classes must be a non-empty subset of {CLASSES}, "
                           f"got {self.classes!r}")
        overlap = set(self.include) & set(self.exclude)
        if overlap:
            raise KeyError(f"include and exclude overlap: {sorted(overlap)}")
        if self.keep_weights not in ("final", "none", "all_rounds"):
            raise KeyError(f"keep_weights {self.keep_weights!r}")

    # ------------------------------------------------------------- identity
    def to_dict(self) -> dict:
        # kinds/classes/include/exclude are SETS, not sequences -- sorted so
        # two specs that name the same selection in a different order hash
        # the same (an order-sensitive hash here would be a silent cache-key
        # bug: the same selection, two different, un-deduplicated entries).
        return {"kinds": sorted(self.kinds),
                "classes": sorted(self.classes),
                "include": sorted(self.include),
                "exclude": sorted(self.exclude),
                "transform": self.transform,
                "scaling": self.scaling,
                "geometry": {"interval_agg_h": self.geometry.interval_agg_h,
                             "window_size": self.geometry.window_size,
                             "step_size": self.geometry.step_size,
                             "reference_months": self.geometry.reference_months,
                             "label_threshold": self.geometry.label_threshold,
                             "feature_range": list(self.geometry.feature_range)},
                "model": self.model.as_fl(),
                "training": self.training.as_fl(),
                "noise": bool(self.noise),
                "cap": self.cap,
                # PRE-EXISTING GAP, fixed here: these three were computed and
                # applied at train time but never round-tripped through
                # spec.json, so a reloaded spec silently reverted to their
                # defaults -- "so a cached run is re-runnable" (this class's
                # own docstring) wasn't true whenever any of them was
                # non-default. Since `key()` hashes `to_dict()`, adding them
                # also makes them part of cache identity going forward: two
                # specs that keep/snapshot differently now get different
                # run keys instead of silently sharing one. Existing
                # run_keys on disk are unaffected (they were computed under
                # the old, narrower to_dict()); only NEW keys are stricter.
                "snapshot_rounds": (list(self.snapshot_rounds)
                                    if not isinstance(self.snapshot_rounds, str)
                                    else self.snapshot_rounds),
                "snapshot_points": int(self.snapshot_points),
                "keep_weights": self.keep_weights,
                # None is dropped by `_norm`, so specs without a lab loss hash
                # exactly as before this field existed.
                "loss_terms": (None if self.loss_terms is None
                               else self.loss_terms.as_dict()),
                "orient": True if self.orient else None,
                "schedule": (None if self.schedule is None
                             else self.schedule.as_dict())}

    def key(self, world_hash: str, sensor_hash: str = "") -> str:
        """Deterministic run key: 12 hex chars over the normalized spec.

        `sensor_hash` folds in the sensors the placement actually resolved to,
        so two worlds whose `boundary` rule picked different sensors cannot
        collide, and a placement rule that silently changes shows up as a new
        key rather than a corrupted cache hit.
        """
        payload = {"world": str(world_hash), "sensors": str(sensor_hash),
                   "spec": _norm(self.to_dict())}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def label(self) -> str:
        ch = "".join(k[0] for k in sorted(self.kinds))
        cl = "".join(c[0] for c in sorted(self.classes))
        ov = ""
        if self.include or self.exclude:
            digest = hashlib.sha256(
                json.dumps([sorted(self.include), sorted(self.exclude)],
                          separators=(",", ":")).encode()).hexdigest()[:6]
            ov = f"~{digest}"
        tr = self.training
        ep = f"e{tr.local_epochs}" if tr.local_epochs != 1 else ""
        pw = f"pw{tr.proto_weight:g}" if tr.proto_weight is not None else ""
        lt = f"__{self.loss_terms.label()}" if self.loss_terms is not None else ""
        lt += f"__{self.schedule.label()}" if self.schedule is not None else ""
        tf = self.transform + ("+or" if self.orient else "")
        return (f"{ch}{cl}{ov}__{tf}__{self.scaling}"
                f"__{self.geometry.label()}"
                f"__u{self.model.lstm_units}r{tr.rounds}{ep}{pw}"
                f"s{tr.fl_seed}{lt}")

    def with_(self, **kw) -> "CellSpec":
        """Copy with overrides; nested blocks by dotted key.

        >>> spec.with_(transform="diff", **{"training.rounds": 10})
        """
        nested: dict[str, dict] = {}
        flat: dict = {}
        for k, v in kw.items():
            if "." in k:
                block, field_ = k.split(".", 1)
                nested.setdefault(block, {})[field_] = v
            else:
                flat[k] = v
        for block, over in nested.items():
            flat[block] = replace(getattr(self, block), **over)
        return replace(self, **flat)

    # ---------------------------------------------------------------- config
    def as_fl(self, base_fl: dict, reference_months: int) -> dict:
        """`params.fl` with this spec's blocks substituted in.

        Everything the spec does not own (`dependence`, `personalization`,
        `drift`, ...) is carried through untouched, so nodes that read those
        blocks keep working.
        """
        import copy as _copy
        fl = _copy.deepcopy(base_fl or {})
        fl["preprocessing"] = {**fl.get("preprocessing", {}),
                               **self.geometry.as_preprocessing(reference_months)}
        fl["model"] = {**fl.get("model", {}), **self.model.as_fl()}
        fl["training"] = {**fl.get("training", {}), **self.training.as_fl()}
        return fl


# The POC cell: every sensor a world offers, one geometry, short training.
# Works against any world regardless of its placement source -- `classes` is
# simply ignored when the world's sensors are `manual`.
POC = CellSpec(kinds=KINDS, classes=CLASSES, transform="none",
               scaling="minmax_ref",
               geometry=Geometry(interval_agg_h=3, window_size=84, step_size=12),
               model=ModelCfg(lstm_units=30),
               training=TrainCfg(rounds=5, fl_seed=0),
               cap=3000, snapshot_rounds="sparse", snapshot_points=1500)


def expand(base: CellSpec, axes: dict[str, Sequence]) -> list[CellSpec]:
    """Cartesian product over dotted axis names.

    >>> expand(POC, {"transform": ["none", "diff"], "training.rounds": [5, 15]})
    """
    import itertools
    names = list(axes)
    out = []
    for combo in itertools.product(*(list(axes[n]) for n in names)):
        out.append(base.with_(**dict(zip(names, combo))))
    return out
