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

Placement is always explicit and always built the same way from
`pressures`/`flows`, for every rule including `manual`. The client CSVs in
`07_model_output/clients` are *not* read: they carry `observed` (noisy)
values, so mixing them with rebuilt placements would confound the placement
axis with a noise axis.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import pandas as pd

from bridge import deseasonalize, preprocess_clients
from specs import CellSpec
from worlds import World

__all__ = ["client_frames", "apply_transform", "build_windows",
           "rescale_windows", "ref_range_per_client", "assert_equal_channels",
           "window_metadata", "sensor_hash", "resolve_placement",
           "WindowCache", "CACHE"]

_PLACEMENT_MODULE = None


def _placement_poc():
    """The placement POC module, imported lazily and only when needed.

    `manual` is read from the manifest, so nothing here (and therefore
    neither wntr nor the notebooks package) is required for the default path.
    """
    global _PLACEMENT_MODULE
    if _PLACEMENT_MODULE is None:
        try:
            from notebooks.SensorPlacement import placement_poc as P
        except ImportError as exc:      # pragma: no cover - environment dependent
            raise ImportError(
                "topology/statistics placements need the placement POC "
                "(notebooks/SensorPlacement/placement_poc.py) on sys.path; "
                "placement='manual' works without it") from exc
        _PLACEMENT_MODULE = P
    return _PLACEMENT_MODULE


# --------------------------------------------------------------------------
# placement -> client frames
# --------------------------------------------------------------------------
def resolve_placement(world: World, placement: str) -> dict[str, dict[str, list[str]]]:
    """{district: {"pressure": [...], "flow": [...]}} for a placement rule.

    `manual` is the placement the world was simulated with, read straight
    from `manifest.effective.sensors`. Other rules are resolved by the
    placement POC against this world's network and residual statistics.
    """
    if placement == "manual":
        sensors = world.sensors_manual
        if not sensors:
            raise AssertionError(f"{world.sim_hash}: manifest carries no sensors")
        return sensors

    P = _placement_poc()
    art = world.artifacts()
    cands = P.candidate_table(art["wn"], art["districts"])
    feats, dist = P.topology_features(art["wn"], art["districts"], cands)
    stats = P.residual_stats(art["pressures"], art["flows"], cands,
                             art["steps_day"])
    rules = P.strategies(cands, feats, stats, dist, art["params"],
                         seed=int(world.params.get("seed", 42)))
    if placement not in rules:
        raise KeyError(f"placement {placement!r} not offered by the POC "
                       f"(have {sorted(rules)})")
    return _as_sensor_map(rules[placement])


def _as_sensor_map(placement) -> dict[str, dict[str, list[str]]]:
    """Normalise a POC placement into the manifest's sensor-map shape."""
    if isinstance(placement, dict) and all(
            isinstance(v, dict) for v in placement.values()):
        return {d: {k: [str(x) for x in (v.get(k) or [])] for k in ("pressure", "flow")}
                for d, v in placement.items()}
    df = pd.DataFrame(placement)
    cols = {c.lower(): c for c in df.columns}
    dcol = next(cols[k] for k in ("district", "client") if k in cols)
    kcol = next(cols[k] for k in ("kind", "type") if k in cols)
    icol = next(cols[k] for k in ("id", "node", "link", "sensor", "name")
                if k in cols)
    out: dict[str, dict[str, list[str]]] = {}
    for d, grp in df.groupby(dcol):
        out[str(d)] = {k: [str(x) for x in grp.loc[grp[kcol] == k, icol]]
                       for k in ("pressure", "flow")}
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


def _column_name(kind: str, ident: str) -> str:
    return ("p_" if kind == "pressure" else "q_") + str(ident)


def client_frames(world: World, placement: str = "manual",
                  kinds=("pressure", "flow"), noise: bool = False,
                  seed: int | None = None) -> "OrderedDict[str, pd.DataFrame]":
    """{district: wide frame} in the `client_datasets` schema.

    Columns: `timestamp, month, p_*/q_*` sorted, exactly what
    `preprocess_clients` expects. Values are the *clean* simulated series
    unless `noise=True`, in which case the world's own noise block is applied
    through the POC (which keys its RNG on `hash(sensor)` -- see the
    determinism note in `train.assert_determinism`).
    """
    sensors = resolve_placement(world, placement)
    kinds = tuple(kinds)

    if noise:
        return _client_frames_noisy(world, sensors, kinds,
                                    seed if seed is not None
                                    else int(world.params.get("seed", 42)))

    pres, flow = world.pressures, world.flows
    month = _month_series(pres, flow)
    n = len(month)
    ts = _timestamps(world, n)

    out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for d in sorted(sensors):
        series = {}
        for kind in kinds:
            src = pres if kind == "pressure" else flow
            for ident in sensors[d].get(kind) or []:
                col = str(ident)
                if col not in src.columns:
                    raise KeyError(f"{d}: {kind} sensor {col!r} absent from "
                                   f"{'pressures' if kind == 'pressure' else 'flows'}")
                series[_column_name(kind, col)] = src[col].to_numpy(float)
        if not series:
            raise AssertionError(f"{d}: no sensors left after kinds={kinds}")
        f = pd.DataFrame({k: series[k] for k in sorted(series)})
        f.insert(0, "month", np.asarray(month))
        f.insert(0, "timestamp", ts)
        out[d] = f
    return out


def _client_frames_noisy(world: World, sensors: dict, kinds, seed: int):
    """Noisy variant, via the POC's `build_series` (the only noise source)."""
    P = _placement_poc()
    art = world.artifacts()
    ss = P.build_series(art["pressures"], art["flows"], sensors,
                        art["params"].get("noise"), seed)
    ss = ss[ss["kind"].isin(list(kinds))]
    wide = ss.pivot(index="step", columns="sensor", values="observed").sort_index()
    month = ss.drop_duplicates("step").set_index("step")["month"].sort_index()
    key = ss.drop_duplicates("sensor").set_index("sensor")["district"]
    ts = _timestamps(world, len(wide))

    out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for d, grp in key.groupby(key):
        cols = sorted(grp.index)
        f = wide[cols].copy().reset_index(drop=True)
        f.insert(0, "month", month.to_numpy())
        f.insert(0, "timestamp", ts)
        out[d] = f
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
    if cache is not None:
        hit = cache.get(world, spec, clients)
        if hit is not None:
            return hit

    warmup = world.warmup_months or int(
        (world.fl.get("preprocessing") or {}).get("reference_months", 2))
    fl = spec.as_fl(world.fl, reference_months=warmup)
    ref_m = int(fl["preprocessing"]["reference_months"])
    first = world.first_switch
    if first is not None and ref_m > first:
        raise AssertionError(
            f"reference_months={ref_m} reaches past the first node switch "
            f"(month {first}): the commissioning scaler would see the drift. "
            f"warm-up is {world.warmup_months} months.")

    frames = client_frames(world, spec.placement, spec.kinds, spec.noise)
    if clients:
        frames = {c: frames[c] for c in clients}
    assert_equal_channels(frames)
    frames = apply_transform(frames, spec.transform, world.steps_day)

    fw, scalers, _report = preprocess_clients(frames, fl, world.time)
    fw = _subsample(fw, spec.cap)
    fw = rescale_windows(fw, scalers, spec.scaling, fl)

    if cache is not None:
        cache.put(world, spec, clients, (fw, scalers, fl))
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
    """Bounded LRU over `(world, spec-upstream, clients)` -> windows.

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
        return (world.sim_hash, spec.placement, tuple(sorted(spec.kinds)),
                spec.transform, spec.scaling, bool(spec.noise), spec.cap,
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
