"""The single entry point. Everything goes through `run_cell`.

`run_cell(world, spec, store)` is a **cache read** when that key already
exists on disk and a train otherwise, so re-running a grid costs nothing and
adding a cell to a grid trains only the new cell. This is the behaviour PP_01
had (deterministic `cell_id`) and the AER sandbox lacked (timestamped
`run_id`, so every re-run retrained and duplicated).

`run_grid` maps it over specs and reports one line each. Failures are
collected rather than raised, so one bad cell does not abandon a sweep.

Scoring is deliberately *not* done here. `store.scores/` is a separate layer
precisely so a new method can be scored over already-trained runs; see
`metrics.score_run` and `cluster.score_run`.
"""
from __future__ import annotations

import traceback

import pandas as pd

from data import CACHE, build_windows
from specs import CellSpec
from store import Store
from worlds import World

__all__ = ["run_cell", "run_grid"]


def run_cell(world: World, spec: CellSpec, store: Store | None = None,
             overwrite: bool = False, cache=CACHE, verbose: bool = False,
             load_tables: bool = True) -> dict:
    """Train (or load) one cell.

    Returns the training result dict, plus `run_key`, `cache_hit` and (when
    a store was given) `dir`. With `load_tables=False` a cache hit returns
    only the metadata -- useful when sweeping just to check what exists.
    """
    import train as T                     # deferred: imports torch

    key = sensor = None
    if store is not None:
        key, sensor = store.key_for(world, spec)
        if not overwrite and store.exists(world, spec):
            out = (store.load_run(world.sim_hash, key) if load_tables
                   else {"meta": {}, "spec": spec})
            out.update(run_key=key, sensor_hash=sensor, cache_hit=True,
                       snapshots=out.get("latents_by_round"))
            if verbose:
                print(f"  cached  {spec.label()}  [{key}]")
            return out

    fl_windows, scalers, fl = build_windows(world, spec, cache=cache)
    res = T.run_cell(world, spec, fl_windows=fl_windows, scalers=scalers,
                     fl=fl, verbose=verbose)
    res.update(run_key=key, sensor_hash=sensor, cache_hit=False)

    if store is not None:
        res["dir"] = store.save_run(world, spec, res)
    return res


def run_grid(world: World, specs, store: Store | None = None,
             overwrite: bool = False, cache=CACHE,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every spec. -> (index of what ran, failures).

    Nothing is scored; the returned frame is provenance only, so this stays
    usable while the scoring layer is still moving.
    """
    rows, fails = [], []
    for i, spec in enumerate(specs, 1):
        label = spec.label()
        try:
            res = run_cell(world, spec, store=store, overwrite=overwrite,
                           cache=cache, verbose=False, load_tables=False)
            rows.append({"n": i, "run_key": res.get("run_key"),
                         "label": label, "cache_hit": res.get("cache_hit"),
                         "seconds": res.get("seconds")})
            if verbose:
                flag = "cached" if res.get("cache_hit") else f"{res['seconds']}s"
                print(f"[{i}/{len(specs)}] {label:<62} {flag}")
        except Exception as exc:
            fails.append({"n": i, "label": label,
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(limit=3)})
            if verbose:
                print(f"[{i}/{len(specs)}] {label:<62} "
                      f"FAILED {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows), pd.DataFrame(fails)
