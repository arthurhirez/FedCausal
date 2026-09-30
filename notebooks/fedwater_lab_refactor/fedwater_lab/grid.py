"""The wider pull: score a dependence method over every run already on disk.

`store.scores/` exists so a new method can be scored across an existing grid
with no GPU, and that is what the factor study is -- schedule, loss terms,
alignment, scaling, orientation and slot class are read off the stored specs,
not re-trained.

Two things this module refuses to let the notebook do silently.

**Unbalanced by construction.** The grid exists because cells were
interesting, not because a design called for them. `coverage` reports the
cell count per axis level and `axis_effects` marks an axis as unestimable
when its levels are collinear with another axis rather than quietly
reporting a small effect. What comes out is a regression over available
cells with a coverage table attached, never a balanced ANOVA.

**Comparable only within a block.** Two cells can be differenced only when
they share the world, the sensor selection, the window geometry, the
transform and the scaler -- everything upstream of the model. Across those,
you may RANK but not subtract, because the target `phi` is in different
units. `comparable_blocks` assigns the block id and `check_contrast` raises
on a cross-block difference.

Every score row carries `method` and `method_hash` -- a digest of the
estimator's own configuration -- so re-scoring with a changed estimator
lands in new rows instead of pooling silently with the old ones. That was
the failure mode the registry schema drift used to produce.
"""
from __future__ import annotations

import json
import zlib

import pandas as pd

__all__ = ["method_hash", "pull", "scan", "load_tables", "locate",
           "assert_cached", "spec_axes",
           "comparable_blocks", "check_contrast", "coverage", "score_grid",
           "axis_effects", "AXES", "BLOCK_KEYS"]

AXES = ("schedule", "schedule_months", "loss", "proto_weight", "spec_term",
        "diff_term", "var_term", "scaling", "transform", "orient", "kinds",
        "classes", "geometry", "rounds", "local_epochs", "lstm_units",
        "fl_seed")

BLOCK_KEYS = ("world_hash", "sensor_hash", "interval_agg_h", "window_size",
              "step_size", "transform", "scaling", "kinds", "classes")


def method_hash(name: str, params: dict) -> str:
    """Stable digest of an estimator's configuration.

    Uses `fedwater.hashing.stable_hash` when the repo package is importable
    (so lab rows and pipeline rows agree), else the same crc32 recipe.
    """
    payload = json.dumps({"name": name, "params": params}, sort_keys=True,
                         default=str)
    try:
        from fedwater.hashing import stable_hash
        return f"{stable_hash(payload):08x}"
    except Exception:
        return f"{zlib.crc32(payload.encode()) % 2**31:08x}"


# --------------------------------------------------------------------------
# the pull
# --------------------------------------------------------------------------
def spec_axes(spec) -> dict:
    """The experiment axes of one `CellSpec`, flattened for a table."""
    lt = getattr(spec, "loss_terms", None)
    sch = getattr(spec, "schedule", None)
    return {
        "label": spec.label(),
        "schedule": "all" if sch is None else sch.mode,
        "schedule_months": None if sch is None else (
            f"{sch.months[0]}-{sch.months[1]}" if sch.mode == "commissioning"
            else f"B{sch.block_months}x{sch.rounds_per_block}"),
        "loss": "upstream" if lt is None else lt.label(),
        "proto_weight": None if lt is None else lt.proto_weight,
        "spec_term": None if lt is None else lt.spec,
        "diff_term": None if lt is None else lt.diff,
        "var_term": None if lt is None else lt.var,
        "scaling": spec.scaling, "transform": spec.transform,
        "orient": bool(getattr(spec, "orient", False)),
        "kinds": "+".join(sorted(spec.kinds)),
        "classes": "+".join(sorted(spec.classes)),
        "geometry": spec.geometry.label(),
        "interval_agg_h": spec.geometry.interval_agg_h,
        "window_size": spec.geometry.window_size,
        "step_size": spec.geometry.step_size,
        "rounds": spec.training.rounds,
        "local_epochs": spec.training.local_epochs,
        "lstm_units": spec.model.lstm_units,
        "fl_seed": spec.training.fl_seed,
    }


def pull(store, world_hash: str | None = None, **filters) -> pd.DataFrame:
    """`runs.csv` + the axes that live only in `spec.json`.

    `filters` are equality filters on either source (e.g. `network="ky7"`,
    `schedule="commissioning"`).
    """
    idx = store.index(world_hash)
    if not len(idx):
        return idx
    rows = []
    for r in idx.itertuples(index=False):
        rec = r._asdict()
        try:
            run = store.load_run(r.world_hash, r.run_key, tables=())
            rec.update(spec_axes(run["spec"]))
            rec["world_path"] = run["meta"].get("world_path")
            rec["world_tag"] = run["meta"].get("world_tag")
            rec["drift_district"] = run["meta"].get("drift_district")
            rec["readable"] = True
        except Exception as exc:
            rec.update(readable=False,
                       error=f"{type(exc).__name__}: {exc}")
        rows.append(rec)
    out = pd.DataFrame(rows)
    for k, v in filters.items():
        if k in out.columns:
            out = out[out[k] == v]
    return out.reset_index(drop=True)


def comparable_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Add `cmp_block`: cells that may be differenced against each other."""
    keys = [k for k in BLOCK_KEYS if k in df.columns]
    out = df.copy()
    out["cmp_block"] = (out[keys].astype(str).agg("|".join, axis=1)
                        .astype("category").cat.codes)
    out["cmp_key"] = out[keys].astype(str).agg(" · ".join, axis=1)
    return out


def check_contrast(df: pd.DataFrame, run_keys) -> None:
    """Raise unless these runs sit in one comparability block."""
    sub = df[df["run_key"].isin(list(run_keys))]
    if sub["cmp_block"].nunique() > 1:
        raise AssertionError(
            "cross-block contrast: these cells differ upstream of the model "
            f"({sub['cmp_key'].nunique()} distinct keys). Rank them, do not "
            "difference them.\n"
            + sub[["run_key", "cmp_key"]].to_string(index=False))


def coverage(df: pd.DataFrame, axes=AXES) -> pd.DataFrame:
    """Cells per axis level -- the table that says which effects exist."""
    rows = []
    for a in axes:
        if a not in df.columns:
            continue
        vc = df[a].astype(str).value_counts()
        rows.append({"axis": a, "levels": int(vc.size),
                     "counts": vc.to_dict(),
                     "min_cells": int(vc.min()) if vc.size else 0,
                     "estimable": bool(vc.size > 1 and vc.min() >= 2)})
    return pd.DataFrame(rows)


def score_grid(store, df: pd.DataFrame, fn, name: str, params: dict,
               save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Apply `fn(store, row) -> DataFrame | dict | None` to every pulled run.

    Failures are collected as rows with `status="failed"`, never raised: one
    unreadable cell must not abandon a sweep. Rows are tagged with the run
    identity and the method hash and appended to `store.scores/<name>`.
    """
    mh = method_hash(name, params)
    frames = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        base = {"world_hash": r.world_hash, "run_key": r.run_key,
                "label": getattr(r, "label", None),
                "method": name, "method_hash": mh}
        try:
            out = fn(store, r)
            if out is None:
                continue
            if isinstance(out, dict):
                out = pd.DataFrame([out])
            out = out.copy()
            for k, v in base.items():
                out[k] = v
            out["status"] = "ok"
            frames.append(out)
            if verbose:
                print(f"[{i}/{len(df)}] {base['label']:<48} {len(out)} rows")
        except Exception as exc:
            frames.append(pd.DataFrame([{**base, "status": "failed",
                                         "error": f"{type(exc).__name__}: {exc}"}]))
            if verbose:
                print(f"[{i}/{len(df)}] {base['label']:<48} FAILED "
                      f"{type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame()
    scores = pd.concat(frames, ignore_index=True)
    if save:
        store.save_scores(name, scores,
                          keys=("world_hash", "run_key", "method",
                                "method_hash", "phase", "source", "target",
                                "arm", "client_a", "client_b"))
    return scores


def axis_effects(scores: pd.DataFrame, value: str, axes=AXES,
                 by: str | None = "cmp_block") -> pd.DataFrame:
    """Mean of `value` per axis level, within comparability blocks.

    Reported as a direction with its cell count, not as an estimate: with an
    unbalanced grid and a handful of independent worlds, the ordering is the
    claim and the spread is the caveat.
    """
    rows = []
    d = scores[scores.get("status", "ok") == "ok"] if "status" in scores else scores
    for a in axes:
        if a not in d.columns or d[a].nunique() < 2:
            continue
        grp = [a] if by is None or by not in d.columns else [by, a]
        g = d.groupby(grp)[value].agg(["mean", "std", "count"]).reset_index()
        for lvl, sub in g.groupby(a):
            rows.append({"axis": a, "level": lvl,
                         "mean": float(sub["mean"].mean()),
                         "spread_across_blocks": float(sub["mean"].std()),
                         "n_cells": int(sub["count"].sum()),
                         "n_blocks": int(sub.shape[0])})
    out = pd.DataFrame(rows)
    return out.sort_values(["axis", "mean"], ascending=[True, False]) \
        .reset_index(drop=True) if len(out) else out


def scan(root, world_hash: str | None = None) -> pd.DataFrame:
    """Index a directory of saved runs directly, without `runs.csv`.

    A run directory carries everything the index does (`meta.json`,
    `spec.json`), so a store copied without its index is still readable.
    `root` may be the store root (`runs/<world>/<key>/`) or a flat directory
    of run folders, which is what a zipped export usually is.
    """
    import pathlib
    root = pathlib.Path(root)
    dirs = sorted(p.parent for p in root.rglob("meta.json"))
    rows = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        rec = {"world_hash": meta.get("world_hash", d.parent.name),
               "run_key": meta.get("run_key", d.name), "dir": str(d)}
        rec.update({k: meta.get(k) for k in
                    ("label", "world_tag", "world_path", "network",
                     "districting", "placement_source", "sensor_hash",
                     "n_windows", "seconds", "drift_district", "rounds",
                     "warmup_months")})
        try:
            from .specs import CellSpec                     # noqa: F401
            from .store import _spec_from_dict
            rec.update(spec_axes(_spec_from_dict(
                json.loads((d / "spec.json").read_text()))))
        except Exception:
            sched = meta.get("schedule") or {}
            lt = meta.get("loss_terms") or {}
            rec.update(schedule=sched.get("mode", "all"),
                       schedule_months=(None if not sched.get("months") else
                                        f"{sched['months'][0]}-{sched['months'][1]}"),
                       proto_weight=lt.get("proto_weight"),
                       spec_term=lt.get("spec"), diff_term=lt.get("diff"),
                       var_term=lt.get("var"))
        rec["phases"] = meta.get("phases")
        rec["clients"] = meta.get("clients")
        rows.append(rec)
    out = pd.DataFrame(rows)
    if world_hash is not None and len(out):
        out = out[out["world_hash"] == world_hash].reset_index(drop=True)
    return out


def load_tables(row, tables=("latent_trajectories", "prototype_history",
                             "prototypes", "update_gram")) -> dict:
    """Read one scanned run's artifacts (parquet or gzipped csv)."""
    import pathlib
    d = pathlib.Path(row["dir"] if isinstance(row, dict) else row.dir)
    out = {"meta": json.loads((d / "meta.json").read_text()),
           "fl": json.loads((d / "fl.json").read_text()), "dir": d}
    for name in tables:
        for suf in (".parquet", ".csv.gz"):
            f = (d / name).with_suffix(suf)
            if f.exists():
                out[name] = (pd.read_parquet(f) if suf == ".parquet"
                             else pd.read_csv(f))
                break
        else:
            out[name] = None
    return out


# --------------------------------------------------------------------------
# the guard: an analysis cell must never train
# --------------------------------------------------------------------------
# `aligned.analyse` -> `runner.run_cell` is a cache read ONLY when
# `Store.exists` finds `<root>/runs/<world_hash>/<run_key>/meta.json`; on a
# miss it TRAINS (silently, and for a commissioning or streaming cell for a
# long time). `scan` above walks any layout, so a notebook that indexes with
# `scan` and analyses with the Store can see a run and still retrain it. These
# two functions close that gap: call `assert_cached` before `analyse`.


def locate(store, world_hash: str, run_key: str) -> dict:
    """Where the Store expects a run, whether it is there, and where else a
    directory with that run key sits under the same root (a flat export, a
    copy one level too shallow)."""
    import pathlib
    expected = store.run_dir(world_hash, run_key) / "meta.json"
    root = pathlib.Path(store.root)
    elsewhere = [str(m.parent) for m in root.rglob("meta.json")
                 if m.parent.name == run_key and m != expected]
    return {"expected": str(expected), "exists": expected.exists(),
            "found_elsewhere": elsewhere}


def assert_cached(store, world, spec, expected_key: str | None = None) -> dict:
    """Refuse to proceed unless `run_cell(world, spec, store)` is a cache hit.

    Two independent failures, reported separately because they have
    different fixes:

    * the KEY differs from the one on disk -- the spec round-trips (checked
      on all nine POC_04 runs), so this means the sensor selection resolved
      to a different `sensor_hash` (placement re-resolved, or a different
      world object);
    * the key matches but the DIRECTORY is not where the Store looks --
      the run must live at `runs/<world_hash>/<run_key>/`.
    """
    key, sh = store.key_for(world, spec)
    loc = locate(store, world.sim_hash, key)
    problems = []
    if expected_key is not None and key != expected_key:
        problems.append(
            f"recomputed run key {key} != {expected_key} on disk "
            f"(sensor_hash here: {sh}). The spec round-trips, so the sensor "
            "selection resolved differently -- check the world object and "
            "its placement.")
    if not loc["exists"]:
        if loc["found_elsewhere"]:
            problems.append(
                f"run {key} exists at {loc['found_elsewhere'][0]} but the Store "
                f"only reads {loc['expected']} -- move the directory to "
                "runs/<world_hash>/<run_key>/.")
        else:
            problems.append(f"no run {key} anywhere under {store.root}.")
    if problems:
        raise FileNotFoundError(
            "refusing to call analyse(): on a cache miss it TRAINS.\n  - "
            + "\n  - ".join(problems))
    return {"run_key": key, "sensor_hash": sh, **loc}
