"""World -> client frames -> transform -> windows -> rescale.

The single data path for every cell. `preprocess_clients` is imported from
`fedwater` and run verbatim -- the two insertion points around it are the
ones the notebooks established:

* **before windowing**, a transform on the client frame
  (`none | slog | diff | deseason_own | deseason_shared`);
* **after windowing**, the scaling axis. `preprocess_clients` always fits a
  per-client MinMax on the reference months; `rescale_windows` inverts that
  affine map with `fl_scalers` and re-applies whatever was asked for, so
  `none`, `zscore` and `shared` are reachable without forking pipeline code.

`shared` is the one genuinely new value: one MinMax fitted on the *pooled*
reference windows of all clients. Per-client `minmax_ref` pins every client
to exactly the feature range and therefore deletes cross-client level;
`shared` keeps it. `ref_range_per_client` is the check that says which of
those two actually happened.

Sensor SELECTION is always explicit and always built the same way from
`pressures`/`flows`, whatever the world's placement SOURCE (`dynamic` or
`manual` -- a fact about the world, see `worlds.World.placement_source`).
`sensor_table` reads `world.sensor_placement` and applies a `CellSpec`'s
`kinds`/`classes`/`include`/`exclude` filters; `client_frames` builds the
series for whatever it resolves to. The client CSVs in
`07_model_output/clients` are *not* read: they carry `observed` (noisy)
values, so mixing them with rebuilt series would confound the selection axis
with a noise axis.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import pandas as pd

from .bridge import add_measurement_noise, deseasonalize, preprocess_clients
from .specs import CLASSES, CellSpec, KINDS
from .worlds import World

__all__ = ["client_frames", "apply_transform", "build_windows",
           "orientation", "apply_orientation", "orientation_table",
           "channel_alignment",
           "rescale_windows", "ref_range_per_client", "assert_equal_channels",
           "window_metadata", "sensor_hash", "sensor_table", "dropped_table",
           "resolve_placement", "WindowCache", "CACHE"]


# --------------------------------------------------------------------------
# selection -- world.sensor_placement x CellSpec -> which sensors, and why
# --------------------------------------------------------------------------
def sensor_table(world: World, spec: CellSpec | None = None) -> pd.DataFrame:
    """The rows of `world.sensor_placement` this spec SELECTS, with a `note`
    saying why each one survived: `"kind+class"` (passed the ordinary
    filter), or `"include"` (forced in by `spec.include` despite the
    filter -- only possible when the world is `dynamic`; a `manual` world has
    no filter for `include` to override).

    `spec.classes` is only applied when `world.placement_source == "dynamic"`
    -- a `manual` world's rows have no `slot_class` to filter on (they are
    all literally `"manual"`), so every one of them is a candidate before
    `kinds`/`include`/`exclude`.
    """
    spec = spec or CellSpec()
    sp = world.sensor_placement
    by_kind = sp[sp["kind"].isin(spec.kinds)]
    if world.placement_source == "dynamic":
        base = by_kind[by_kind["slot_class"].isin(spec.classes)]
    else:
        base = by_kind
    base = base.assign(note="kind+class")

    forced = sp[sp["sensor"].isin(spec.include) & ~sp["sensor"].isin(base["sensor"])]
    forced = forced.assign(note="include")
    out = pd.concat([base, forced], ignore_index=True)
    out = out[~out["sensor"].isin(spec.exclude)]
    return out.sort_values(["district", "kind", "slot"]).reset_index(drop=True)


def dropped_table(world: World, spec: CellSpec | None = None) -> pd.DataFrame:
    """The complement of `sensor_table`: every candidate NOT selected, with a
    `note` saying why (`"kind"`, `"class"`, or `"exclude"`)."""
    spec = spec or CellSpec()
    sp = world.sensor_placement
    kept = set(sensor_table(world, spec)["sensor"])
    out = sp[~sp["sensor"].isin(kept)].copy()

    def why(r):
        if r["sensor"] in spec.exclude:
            return "exclude"
        if r["kind"] not in spec.kinds:
            return "kind"
        if world.placement_source == "dynamic"                 and r["slot_class"] not in spec.classes:
            return "class"
        return "?"                       # should not happen; visible if it does

    out["note"] = out.apply(why, axis=1) if len(out) else []
    return out.sort_values(["district", "kind", "slot"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# selection -> client frames
# --------------------------------------------------------------------------
def resolve_placement(world: World, spec: CellSpec | None = None
                      ) -> dict[str, dict[str, list[str]]]:
    """{district: {"pressure": [...], "flow": [...]}} for a spec's selection.

    Always both kind keys present (possibly empty), matching the shape
    `client_frames`/`sensor_hash` expect regardless of `spec.kinds`.
    """
    spec = spec or CellSpec()
    t = sensor_table(world, spec)
    out: dict[str, dict[str, list[str]]] = {
        d: {k: [] for k in KINDS} for d in sorted(t["district"].unique())}
    for (d, k), grp in t.groupby(["district", "kind"]):
        out[d][k] = [str(x) for x in grp["element"]]
    return out


def sensor_hash(sensors: dict, kinds=("pressure", "flow")) -> str:
    """Stable 8-hex digest of the resolved sensor set (order-insensitive).

    Folded into the run key so "the same placement" across worlds is a
    checkable claim rather than an assumption.
    """
    import hashlib
    parts = []
    for d in sorted(sensors):
        for k in sorted(kinds):
            ids = sorted(str(x) for x in (sensors[d].get(k) or []))
            parts.append(f"{d}|{k}|{','.join(ids)}")
    return hashlib.sha256(";".join(parts).encode()).hexdigest()[:8]


def client_frames(world: World, spec: CellSpec | None = None,
                  noise: bool | None = None, seed: int | None = None
                  ) -> "OrderedDict[str, pd.DataFrame]":
    """{district: wide frame} in the `client_datasets` schema.

    Columns: `timestamp, month` + one column per selected sensor, named by
    its SLOT (`q_pure_0`, `p_mixed_1`, ...) exactly as the pipeline's
    `package_client_datasets` names them, so channel i means the same slot
    in every client -- which is what makes a global (FedAvg / FINCH)
    prototype comparable channel by channel across clients. On a `manual`
    world the slot IS the sensor id (`q_P-163`), as in the pipeline.

    Values are the *clean* simulated series unless `noise=True` (default:
    `spec.noise`), in which case the pipeline's own `add_measurement_noise`
    is applied, keyed on the ELEMENT id (`sensor`), as the pipeline keys it.
    """
    spec = spec or CellSpec()
    noise = spec.noise if noise is None else bool(noise)
    t = sensor_table(world, spec)
    if not len(t):
        raise AssertionError(
            f"no sensors selected: kinds={spec.kinds} classes={spec.classes} "
            f"(world placement_source={world.placement_source!r})")
    true = _true_series(world, t)
    if noise:
        seed = seed if seed is not None else int(world.params.get("seed", 42))
        true = add_measurement_noise(true, world.params.get("noise") or {},
                                     seed)
        col = "observed"
    else:
        col = "value"
    return _wide_by_slot(world, true, col)


def _true_series(world: World, table: pd.DataFrame) -> pd.DataFrame:
    """Long-form `step, month, district, sensor, slot, kind, value` -- the
    shape the pipeline's `extract_sensor_series` produces (and that
    `add_measurement_noise` consumes)."""
    pres, flow = world.pressures, world.flows
    month = _month_series(pres, flow)
    step = np.arange(len(month))
    rows = []
    for r in table.itertuples(index=False):
        src = pres if r.kind == "pressure" else flow
        el = str(r.element)
        if el not in src.columns:
            raise KeyError(f"{r.district}: {r.kind} sensor {el!r} absent from "
                           f"{'pressures' if r.kind == 'pressure' else 'flows'}")
        rows.append(pd.DataFrame({
            "step": step, "month": np.asarray(month), "district": r.district,
            "sensor": r.sensor, "slot": r.slot, "kind": r.kind,
            "value": src[el].to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


def _wide_by_slot(world: World, long: pd.DataFrame, col: str):
    month = (long.drop_duplicates("step").set_index("step")["month"]
             .sort_index())
    ts = _timestamps(world, len(month))
    out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for d, grp in long.groupby("district", sort=True):
        wide = grp.pivot(index="step", columns="slot", values=col).sort_index()
        f = wide[sorted(wide.columns)].reset_index(drop=True)
        f.insert(0, "month", month.to_numpy())
        f.insert(0, "timestamp", ts)
        out[d] = f
    slots = {d: tuple(f.columns[2:]) for d, f in out.items()}
    if len(set(slots.values())) > 1 and world.placement_source == "dynamic":
        print("WARNING: clients do not share one slot set -- channel i means "
              "a different slot in different clients (include/exclude made "
              "the selection asymmetric); global/FINCH prototypes are then "
              "not comparable channel by channel.")
    return out


def _month_series(pres: pd.DataFrame, flow: pd.DataFrame) -> np.ndarray:
    for src in (pres, flow):
        if "month" in src.columns:
            return src["month"].to_numpy()
    raise AssertionError("neither pressures nor flows carries a `month` column")


def _timestamps(world: World, n: int) -> np.ndarray:
    """Epoch seconds, matching the shipped client CSVs.

    `DatetimeIndex.astype('int64')` is unit dependent in pandas >= 2 (it can
    be us, not ns), so the cast goes through `datetime64[s]` explicitly.
    """
    res_h = max(1, int(round(float(world.time["resolution_h"]))))
    start = str(world.params.get("start_date") or "2024-01-01")
    idx = pd.date_range(start, periods=n, freq=f"{res_h}h")
    return idx.values.astype("datetime64[s]").astype("int64")


def assert_equal_channels(frames: dict[str, pd.DataFrame]) -> int:
    """FedAvg needs identical architectures: same channel count everywhere."""
    n = {c: len([x for x in f.columns if x not in ("timestamp", "month")])
         for c, f in frames.items()}
    if len(set(n.values())) > 1:
        raise AssertionError(f"channel count differs across clients "
                             f"(breaks FedAvg): {n}")
    return next(iter(n.values()))


# --------------------------------------------------------------------------
# orientation (before the transform)
# --------------------------------------------------------------------------
def orientation(frames: dict, ref_months: int, bidir_tol: float = 0.05
                ) -> pd.DataFrame:
    """Per (client, slot): the sign that makes the sensor's dominant
    direction positive.

    EPANET reports a pipe flow with the sign of the pipe's own orientation,
    so the same physical behaviour (more demand downstream) raises one
    sensor and lowers another. With per-client MinMax that inverts the
    demand shape on some channels only, and a FedAvg model shared across
    clients is asked to produce opposite shapes on the same channel.

    sign            sign(median over months < ref_months); +1 if the median
                    is exactly 0
    frac_rev_ref    share of reference steps with the opposite sign
    frac_rev_all    the same over the whole series (drift can reverse flow)
    bidirectional   frac_rev_ref > bidir_tol or frac_rev_all > bidir_tol:
                    a flip does not make these one-directional -- the sign
                    carries information there (direction of supply)
    Pressure slots get sign +1 (they are positive by construction)."""
    rows = []
    for c, f in frames.items():
        ref = f["month"].to_numpy() < ref_months
        for x in (x for x in f.columns if x not in ("timestamp", "month")):
            v = f[x].to_numpy(float)
            med = float(np.median(v[ref])) if ref.any() else float(np.median(v))
            sgn = 1.0 if med >= 0 else -1.0
            nz = np.abs(v) > 1e-9
            rev = (np.sign(v) == -sgn) & nz
            fr = float(rev[ref].mean()) if ref.any() else np.nan
            fa = float(rev.mean())
            rows.append({"client": c, "slot": x, "median_ref": med,
                         "sign": sgn, "frac_rev_ref": fr, "frac_rev_all": fa,
                         "bidirectional": bool((fr > bidir_tol)
                                               or (fa > bidir_tol))})
    return pd.DataFrame(rows)


def apply_orientation(frames: dict, table: pd.DataFrame) -> dict:
    """Multiply every slot by its `sign` from `orientation`."""
    sg = {(r.client, r.slot): r.sign for r in table.itertuples(index=False)}
    out = {}
    for c, f in frames.items():
        g = f.copy()
        for x in (x for x in g.columns if x not in ("timestamp", "month")):
            g[x] = g[x].to_numpy(float) * sg[(c, x)]
        out[c] = g
    return out


def _reference_months(world: World, spec: CellSpec) -> int:
    warmup = world.warmup_months or int(
        (world.fl.get("preprocessing") or {}).get("reference_months", 2))
    return int(spec.as_fl(world.fl, reference_months=warmup)
               ["preprocessing"]["reference_months"])


def orientation_table(world: World, spec: CellSpec | None = None,
                      bidir_tol: float = 0.05) -> pd.DataFrame:
    """`orientation` on the spec's RAW frames (whether or not the spec
    orients), joined with each slot's placement row (sensor, tier, purity,
    second)."""
    spec = spec or CellSpec()
    t = orientation(client_frames(world, spec), _reference_months(world, spec),
                    bidir_tol)
    place = sensor_table(world, spec).rename(columns={"district": "client"})
    keep = [c for c in ("client", "slot", "sensor", "kind", "slot_class", "tier",
                        "purity", "second", "w_second") if c in place.columns]
    return t.merge(place[keep], on=["client", "slot"], how="left")


def channel_alignment(world: World, spec: CellSpec | None = None) -> pd.DataFrame:
    """Channel index x client: what channel i is in each client.

    Channels are ordered by SLOT name in every client (`client_frames`), so
    `slot` is equal across a row by construction. What the slot does NOT
    fix is the sensor's role (tier, purity, whom it mixes with) and its
    flow sign -- this table puts them side by side, one row per channel,
    one column block per client."""
    t = orientation_table(world, spec)
    for col, fill in (("tier", ""), ("purity", np.nan), ("second", None)):
        if col not in t.columns:
            t[col] = fill
    t["channel"] = t.groupby("client").cumcount()
    t["role"] = [f"{s} · {tr} · p{p:.2f}" + (f" · +{sec}" if isinstance(sec, str)
                                               else "")
                 + f" · {'+' if g > 0 else '-'}" + (" · bidir" if b else "")
                 for s, tr, p, sec, g, b in zip(
                     t["slot"], t["tier"], t["purity"], t["second"],
                     t["sign"], t["bidirectional"])]
    return t.pivot(index="channel", columns="client", values="role")


# --------------------------------------------------------------------------
# transforms (before windowing)
# --------------------------------------------------------------------------
def _cell_index(n: int, steps_day: int, month) -> pd.DataFrame:
    day = np.arange(n) // steps_day
    return pd.DataFrame({"month": np.asarray(month),
                         "hour": np.arange(n) % steps_day,
                         "weekend": (day % 7) >= 5})


def _slog(x):
    """Signed log1p -- flows are signed, so log1p goes on the magnitude."""
    return np.sign(x) * np.log1p(np.abs(x))


def deseason_shared(frames: dict, steps_day: int, k: int = 4) -> dict:
    """Remove the top-k PCs of the POOLED (month, hour, weekend) profile matrix.

    Each component is mean-zero across cells by construction, so a channel's
    own LEVEL survives -- the difference from `deseason_own`, which subtracts
    each channel's own cell means and therefore deletes exactly the level
    ladder the regime codes assert.
    """
    cols = {c: [x for x in f.columns if x not in ("timestamp", "month")]
            for c, f in frames.items()}
    any_f = next(iter(frames.values()))
    cells = _cell_index(len(any_f), steps_day, any_f["month"].to_numpy())
    gid = cells.groupby(["month", "hour", "weekend"], sort=True).ngroup().to_numpy()
    ncell = gid.max() + 1

    stack, names = [], []
    for c, f in frames.items():
        for x in cols[c]:
            v = f[x].to_numpy(float)
            stack.append(np.bincount(gid, v, ncell)
                         / np.bincount(gid, minlength=ncell))
            names.append((c, x))
    Pm = np.column_stack(stack)                            # cells x channels
    mu, sd = Pm.mean(0), Pm.std(0, ddof=0)
    Z = (Pm - mu) / np.where(sd > 0, sd, 1.0)
    U, _, _ = np.linalg.svd(Z, full_matrices=False)
    B = U[:, :k]
    fitted = (B @ (B.T @ Z)) * np.where(sd > 0, sd, 1.0)   # channel units

    out = {c: f.copy() for c, f in frames.items()}
    for j, (c, x) in enumerate(names):
        out[c][x] = out[c][x].to_numpy(float) - fitted[gid, j]
    return out


def apply_transform(frames: dict, name: str, steps_day: int) -> dict:
    """One of `none | slog | diff | deseason_own | deseason_shared`."""
    if name == "none":
        return frames
    if name == "deseason_shared":
        return deseason_shared(frames, steps_day)

    out = {}
    for c, f in frames.items():
        g = f.copy()
        cols = [x for x in g.columns if x not in ("timestamp", "month")]
        if name == "slog":
            g[cols] = _slog(g[cols].to_numpy(float))
        elif name == "diff":
            g[cols] = g[cols].diff().fillna(0.0)
        elif name == "deseason_own":
            wide = g[cols].copy()
            wide.index = np.arange(len(g))
            res = deseasonalize(wide, steps_day, g["month"].copy(),
                                month_aware=True)
            g[cols] = np.asarray(res)
        else:
            raise KeyError(name)
        out[c] = g
    return out


# --------------------------------------------------------------------------
# scaling (after windowing)
# --------------------------------------------------------------------------
def _scaler_bounds(scalers: pd.DataFrame, client: str,
                   sensors) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) per channel from `fl_scalers`, whatever it calls its columns."""
    low = {c.lower(): c for c in scalers.columns}
    ccol = next(low[k] for k in ("client", "district", "partition", "name")
                if k in low)
    scol = next(low[k] for k in ("sensor", "feature", "channel", "column")
                if k in low)
    mncol = next(c for c in scalers.columns if "min" in c.lower())
    mxcol = next(c for c in scalers.columns if "max" in c.lower())
    g = scalers[scalers[ccol] == client].drop_duplicates(scol).set_index(scol)
    return (g.loc[list(sensors), mncol].to_numpy(float),
            g.loc[list(sensors), mxcol].to_numpy(float))


def _to_physical(d: dict, scalers: pd.DataFrame, client: str,
                 feature_range) -> np.ndarray:
    """Invert fl_preprocessing's per-client MinMax -> physical units."""
    lo, hi = feature_range
    mn, mx = _scaler_bounds(scalers, client, d["sensors"])
    return ((d["windows"].astype(np.float64) - lo) / (hi - lo) * (mx - mn) + mn)


def rescale_windows(fl_windows: dict, scalers: pd.DataFrame, mode: str,
                    fl: dict) -> dict:
    """`minmax_ref` (as-is) | `shared` | `none` (physical) | `zscore`.

    `shared` fits ONE MinMax on the pooled reference windows of all clients,
    so cross-client level survives into the model input. It is the only mode
    here that is not client-local; that is the point of it, and it is why the
    channel order must match across clients (asserted).
    """
    if mode == "minmax_ref":
        return fl_windows

    prep = fl["preprocessing"]
    frange = tuple(prep["feature_range"])
    lo, hi = frange
    ref_m = int(prep["reference_months"])
    raw = {c: _to_physical(d, scalers, c, frange) for c, d in fl_windows.items()}

    if mode == "none":
        return {c: {**d, "windows": raw[c].astype(np.float32)}
                for c, d in fl_windows.items()}

    if mode == "zscore":
        out = {}
        for c, d in fl_windows.items():
            ref = d["labels"] < ref_m
            src = raw[c][ref] if ref.any() else raw[c]
            mu = src.reshape(-1, src.shape[-1]).mean(0)
            sd = src.reshape(-1, src.shape[-1]).std(0)
            w = (raw[c] - mu) / np.where(sd > 0, sd, 1.0)
            out[c] = {**d, "windows": w.astype(np.float32)}
        return out

    if mode == "shared":
        order = [tuple(fl_windows[c]["sensors"]) for c in sorted(fl_windows)]
        if len({len(o) for o in order}) > 1:
            raise AssertionError("shared scaling needs the same channel count "
                                 f"on every client: {order}")
        pooled = []
        for c, d in fl_windows.items():
            ref = d["labels"] < ref_m
            src = raw[c][ref] if ref.any() else raw[c]
            pooled.append(src.reshape(-1, src.shape[-1]))
        P = np.concatenate(pooled, axis=0)
        mn, mx = P.min(0), P.max(0)
        span = np.where((mx - mn) > 0, mx - mn, 1.0)
        return {c: {**d, "windows": (((raw[c] - mn) / span) * (hi - lo) + lo
                                     ).astype(np.float32)}
                for c, d in fl_windows.items()}

    raise KeyError(mode)


def ref_range_per_client(fl_windows: dict, reference_months: int) -> pd.DataFrame:
    """Min/max of each client's COMMISSIONING windows -- the scaling check.

    Per-client MinMax on the reference months puts every client at exactly
    the feature range, so identical (min, max) rows across clients means
    `minmax_ref`. A shared or inverted scaler generally does not, so this
    says whether a scaling override actually did anything -- without
    depending on how `fl_preprocessing` spells it.
    """
    rows = []
    for c in sorted(fl_windows):
        d = fl_windows[c]
        w = d["windows"][d["labels"] < reference_months]
        rows.append({"client": c, "n_ref_windows": int(len(w)),
                     "min": float(w.min()) if len(w) else np.nan,
                     "max": float(w.max()) if len(w) else np.nan,
                     "mean": float(w.mean()) if len(w) else np.nan})
    out = pd.DataFrame(rows)
    out.attrs["identical_bounds"] = bool(
        out[["min", "max"]].round(6).drop_duplicates().shape[0] == 1)
    return out


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def _subsample(fl_windows: dict, cap: int | None) -> dict:
    """Even-in-time subsample to `cap` windows per client."""
    if not cap:
        return fl_windows
    for c, d in fl_windows.items():
        n = len(d["windows"])
        if n > cap:
            i = np.linspace(0, n - 1, cap).round().astype(int)
            d["windows"], d["labels"] = d["windows"][i], d["labels"][i]
            d["window_start_step"] = d["window_start_step"][i]
    return fl_windows


def build_windows(world: World, spec: CellSpec, clients=None,
                  cache: "WindowCache | None" = None):
    """(fl_windows, fl_scalers, fl) for one cell.

    `reference_months` defaults to the world's own warm-up, and is asserted
    to fit inside it: a scaler fitted past the first node switch would have
    seen the drift it is supposed to make visible.
    """
    # `fl` is ALWAYS rebuilt from this spec. The cache holds windows and
    # scalers only: it is keyed on the upstream part of the spec, so a cached
    # `fl` would hand the training / model blocks of whichever spec filled
    # the entry to every later spec with the same windows (the bug that made
    # 30-round cells train for 5).
    warmup = world.warmup_months or int(
        (world.fl.get("preprocessing") or {}).get("reference_months", 2))
    fl = spec.as_fl(world.fl, reference_months=warmup)
    if cache is not None:
        hit = cache.get(world, spec, clients)
        if hit is not None:
            fw, scalers = hit
            return fw, scalers, fl

    ref_m = int(fl["preprocessing"]["reference_months"])
    first = world.first_switch
    if first is not None and ref_m > first:
        raise AssertionError(
            f"reference_months={ref_m} reaches past the first node switch "
            f"(month {first}): the commissioning scaler would see the drift. "
            f"warm-up is {world.warmup_months} months.")

    frames = client_frames(world, spec)
    if clients:
        frames = {c: frames[c] for c in clients}
    assert_equal_channels(frames)
    if spec.orient:
        frames = apply_orientation(frames, orientation(frames, ref_m))
    frames = apply_transform(frames, spec.transform, world.steps_day)

    fw, scalers, _report = preprocess_clients(frames, fl, world.time)
    fw = _subsample(fw, spec.cap)
    fw = rescale_windows(fw, scalers, spec.scaling, fl)

    if cache is not None:
        cache.put(world, spec, clients, (fw, scalers))
    return fw, scalers, fl


def window_metadata(world: World, fl_windows: dict) -> pd.DataFrame:
    """One row per window, keyed (district, window): the calendar decoding."""
    res_h = float(world.time["resolution_h"])
    steps_day = world.steps_day
    rows = []
    for c in sorted(fl_windows):
        d = fl_windows[c]
        start = np.asarray(d["window_start_step"])
        day = (start * res_h) // 24
        rows.append(pd.DataFrame({
            "district": c, "window": start, "month": d["labels"],
            "hour": ((start * res_h) % 24).astype(int),
            "day": day.astype(int),
            "weekend": ((day % 7) >= 5),
            "phase": [world.phase_of(int(m)) for m in d["labels"]],
        }))
    del steps_day
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
class WindowCache:
    """Bounded LRU over `(world, spec-upstream, clients)` -> (windows,
    scalers). Never the resolved `fl`: that depends on the whole spec.

    Explicit and bounded on purpose: the notebooks kept a module-level dict,
    so a stray widget interaction could silently reprocess a world and grow
    without limit. Only the *upstream* part of the spec is keyed -- model and
    training parameters do not change the windows.
    """

    def __init__(self, maxsize: int = 6):
        self.maxsize = int(maxsize)
        self._store: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.hits = self.misses = 0

    @staticmethod
    def _key(world: World, spec: CellSpec, clients):
        g = spec.geometry
        return (world.sim_hash, tuple(sorted(spec.kinds)),
                tuple(sorted(spec.classes)), tuple(sorted(spec.include)),
                tuple(sorted(spec.exclude)),
                spec.transform, spec.scaling, bool(spec.noise), spec.cap,
                bool(spec.orient),
                g.interval_agg_h, g.window_size, g.step_size,
                g.reference_months, g.label_threshold,
                tuple(g.feature_range), tuple(clients or ()))

    def get(self, world, spec, clients=None):
        k = self._key(world, spec, clients)
        if k in self._store:
            self._store.move_to_end(k)
            self.hits += 1
            return copy.deepcopy(self._store[k])
        self.misses += 1
        return None

    def put(self, world, spec, clients, value):
        k = self._key(world, spec, clients)
        self._store[k] = copy.deepcopy(value)
        self._store.move_to_end(k)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()
        self.hits = self.misses = 0

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return (f"WindowCache(n={len(self._store)}/{self.maxsize}, "
                f"hits={self.hits}, misses={self.misses})")


CACHE = WindowCache()
