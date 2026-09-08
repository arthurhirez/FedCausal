"""Persistence, in two layers.

```
<sandbox>/
  runs/<world_hash>/<run_key>/        immutable, written once by `save_run`
    spec.json  meta.json  sensors.json  training_log.csv  scalers.csv
    prototype_history.parquet          every round -- what similarity reads
    global_prototype_history.parquet   FINCH centroids, every round
    latents_by_round.parquet           stage=pre|post, snapshot rounds only
    latent_trajectories.parquet        final model, all windows
    prototypes.parquet                 the compiled scopes
    drift_signals.parquet  update_gram.parquet
    fed_model.pt                       final-round weights (keep_weights=final)
  scores/<name>.parquet                long-form, appended by `save_scores`
  runs.csv                             thin index: keys + axes + status
```

The split is by *recompute cost*, not storage cost. Training artifacts are
expensive and never change, so they are written once and treated as a cache
whose source of truth is `(spec, world, seed)` -- any run is exactly
reproducible from `spec.json`. Scores are cheap and change every time a
method changes, so they live apart and are keyed back to `run_key`: a new
similarity channel is scored across the whole existing grid with no GPU.

`runs.csv` deliberately holds no metrics. That was what made the notebooks'
registry schema drift every time a scorer gained a column.
"""
from __future__ import annotations

import json
import pathlib
import time

import pandas as pd

from .specs import CellSpec, Geometry, ModelCfg, TrainCfg
from .worlds import World

__all__ = ["Store", "RUN_TABLES"]

RUN_TABLES = ("prototype_history", "global_prototype_history",
              "latents_by_round", "latent_trajectories", "prototypes",
              "drift_signals", "update_gram")

_INDEX_COLS = ["world_hash", "run_key", "label", "tag", "placement", "kinds",
               "transform", "scaling", "interval_agg_h", "window_size",
               "step_size", "reference_months", "lstm_units", "rounds",
               "fl_seed", "sensor_hash", "n_windows", "seconds", "saved_at",
               "status"]


def _write_table(df: pd.DataFrame, base: pathlib.Path) -> pathlib.Path:
    """Parquet, falling back to gzipped CSV where pyarrow is unavailable."""
    try:
        p = base.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = base.with_suffix(".csv.gz")
        df.to_csv(p, index=False)
        return p


def _read_table(base: pathlib.Path):
    for suf in (".parquet", ".csv.gz"):
        p = base.with_suffix(suf)
        if p.exists():
            return pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p)
    return None


def _spec_from_dict(d: dict) -> CellSpec:
    """Rebuild a `CellSpec` from `spec.json` -- so a cached run is re-runnable."""
    g, m, t = d["geometry"], d["model"], d["training"]
    return CellSpec(
        placement=d["placement"], kinds=tuple(d["kinds"]),
        transform=d["transform"], scaling=d["scaling"],
        geometry=Geometry(interval_agg_h=g["interval_agg_h"],
                          window_size=g["window_size"],
                          step_size=g["step_size"],
                          reference_months=g.get("reference_months"),
                          label_threshold=g.get("label_threshold", 0.75),
                          feature_range=tuple(g.get("feature_range", (-1.0, 1.0)))),
        model=ModelCfg(lstm_units=m["lstm_units"], reg_ratio=m["reg_ratio"]),
        training=TrainCfg(rounds=t["rounds"], local_epochs=t["local_epochs"],
                          learning_rate=t["learning_rate"],
                          batch_size=t["batch_size"],
                          participation=t["participation"],
                          averaging=t["averaging"],
                          proto_alpha=t["proto_alpha"],
                          infonce_temperature=t["infonce_temperature"],
                          fl_seed=t.get("seed", 0),
                          device=t.get("device", "auto")),
        noise=bool(d.get("noise", False)), cap=d.get("cap"))


class Store:
    """A sandbox directory. Construct once per session and pass it around."""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root).expanduser().resolve()
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        (self.root / "scores").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- geometry
    def run_dir(self, world_hash: str, run_key: str) -> pathlib.Path:
        return self.root / "runs" / world_hash / run_key

    @property
    def index_path(self) -> pathlib.Path:
        return self.root / "runs.csv"

    def key_for(self, world: World, spec: CellSpec) -> tuple[str, str]:
        """(run_key, sensor_hash) -- resolves the placement to get the hash."""
        from .data import resolve_placement, sensor_hash
        sh = sensor_hash(resolve_placement(world, spec.placement), spec.kinds)
        return spec.key(world.sim_hash, sh), sh

    def exists(self, world: World, spec: CellSpec) -> bool:
        key, _ = self.key_for(world, spec)
        return (self.run_dir(world.sim_hash, key) / "meta.json").exists()

    # ----------------------------------------------------------------- save
    def save_run(self, world: World, spec: CellSpec, res: dict) -> pathlib.Path:
        """Write one training result. Overwrites the same key idempotently."""
        key, sh = self.key_for(world, spec)
        out = self.run_dir(world.sim_hash, key)
        out.mkdir(parents=True, exist_ok=True)

        (out / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
        (out / "sensors.json").write_text(json.dumps(
            {"sensor_hash": sh, "placement": spec.placement,
             "kinds": list(spec.kinds),
             "sensors": res["meta"].get("sensors")}, indent=2, default=str))

        n_windows = int(sum(len(d["windows"])
                            for d in (res.get("fl_windows") or {}).values()))
        meta = {"run_key": key, "world_hash": world.sim_hash,
                "world_tag": world.tag, "world_path": str(world.path),
                "label": spec.label(), "sensor_hash": sh,
                "n_windows": n_windows, "seconds": res.get("seconds"),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phases": world.phases(),
                "regimes_init": world.regimes(phase="init"),
                "regimes_final": world.regimes(phase="final"),
                "drift_district": world.drift_district,
                "warmup_months": world.warmup_months,
                **{k: v for k, v in res["meta"].items() if k != "sensors"}}
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        (out / "fl.json").write_text(json.dumps(res["fl"], indent=2, default=str))

        if res.get("training_log") is not None:
            res["training_log"].to_csv(out / "training_log.csv", index=False)
        if res.get("scalers") is not None:
            res["scalers"].to_csv(out / "scalers.csv", index=False)

        tables = {"prototype_history": res.get("prototype_history"),
                  "global_prototype_history": res.get("global_prototype_history"),
                  "latents_by_round": res.get("snapshots"),
                  "latent_trajectories": res.get("latent_trajectories"),
                  "prototypes": res.get("prototypes"),
                  "drift_signals": res.get("drift_signals"),
                  "update_gram": res.get("update_gram")}
        for name, df in tables.items():
            if df is not None and len(df):
                _write_table(df, out / name)

        if spec.keep_weights != "none" and res.get("models"):
            import torch
            torch.save({**res["models"], "meta": res["meta"],
                        "spec": spec.to_dict(), "world_hash": world.sim_hash},
                       out / "fed_model.pt")

        self._index_upsert({
            "world_hash": world.sim_hash, "run_key": key,
            "label": spec.label(), "tag": world.tag,
            "placement": spec.placement, "kinds": "+".join(sorted(spec.kinds)),
            "transform": spec.transform, "scaling": spec.scaling,
            "interval_agg_h": spec.geometry.interval_agg_h,
            "window_size": spec.geometry.window_size,
            "step_size": spec.geometry.step_size,
            "reference_months": res["fl"]["preprocessing"]["reference_months"],
            "lstm_units": spec.model.lstm_units,
            "rounds": spec.training.rounds, "fl_seed": spec.training.fl_seed,
            "sensor_hash": sh, "n_windows": n_windows,
            "seconds": res.get("seconds"), "saved_at": meta["saved_at"],
            "status": "ok"})
        return out

    def _index_upsert(self, row: dict) -> None:
        p = self.index_path
        reg = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=_INDEX_COLS)
        if len(reg):
            reg = reg[~((reg["world_hash"] == row["world_hash"])
                        & (reg["run_key"] == row["run_key"]))]
        reg = pd.concat([reg, pd.DataFrame([row])], ignore_index=True)
        reg.reindex(columns=_INDEX_COLS).to_csv(p, index=False)

    # ----------------------------------------------------------------- load
    def load_run(self, world_hash: str, run_key: str, tables=RUN_TABLES) -> dict:
        d = self.run_dir(world_hash, run_key)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"no saved run at {d}")
        out = {"meta": json.loads((d / "meta.json").read_text()),
               "spec": _spec_from_dict(json.loads((d / "spec.json").read_text())),
               "sensors": json.loads((d / "sensors.json").read_text()),
               "fl": json.loads((d / "fl.json").read_text()),
               "dir": d}
        if (d / "training_log.csv").exists():
            out["training_log"] = pd.read_csv(d / "training_log.csv")
        for name in tables:
            out[name] = _read_table(d / name)
        return out

    def load_model(self, world_hash: str, run_key: str, client: str = "global"):
        """Rebuild an `AER` with the saved weights (final round)."""
        import torch
        from . import bridge
        d = self.run_dir(world_hash, run_key)
        state = torch.load(d / "fed_model.pt", map_location="cpu",
                           weights_only=False)
        m = state["meta"]
        mdl = bridge.aer_class()(m["n_features"], m["window_size"],
                                 m["lstm_units"])
        mdl.load_state_dict(state[client])
        mdl.eval()
        return mdl

    def index(self, world_hash: str | None = None) -> pd.DataFrame:
        p = self.index_path
        reg = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=_INDEX_COLS)
        return reg if not world_hash or not len(reg) \
            else reg[reg["world_hash"] == world_hash].reset_index(drop=True)

    # --------------------------------------------------------------- scores
    def save_scores(self, name: str, df: pd.DataFrame,
                    keys=("world_hash", "run_key", "scope", "round", "phase",
                          "method")) -> pathlib.Path:
        """Append long-form score rows, replacing any prior rows with the
        same key tuple. Idempotent re-scoring, no duplicate accumulation."""
        if df is None or not len(df):
            return self.root / "scores" / f"{name}.parquet"
        base = self.root / "scores" / name
        prev = _read_table(base)
        keys = [k for k in keys if k in df.columns]
        if prev is not None and len(prev) and keys:
            common = [k for k in keys if k in prev.columns]
            if common:
                mark = prev[common].astype(str).agg("|".join, axis=1)
                new = set(df[common].astype(str).agg("|".join, axis=1))
                prev = prev[~mark.isin(new)]
            df = pd.concat([prev, df], ignore_index=True)
        return _write_table(df, base)

    def load_scores(self, name: str) -> pd.DataFrame | None:
        return _read_table(self.root / "scores" / name)

    # ------------------------------------------------------------- reporting
    def disk_usage(self, world_hash: str | None = None) -> pd.DataFrame:
        """Bytes per artifact kind -- the input to the retention policy."""
        rows = []
        base = self.root / "runs"
        for wdir in sorted(p for p in base.glob("*") if p.is_dir()):
            if world_hash and wdir.name != world_hash:
                continue
            for rdir in sorted(p for p in wdir.glob("*") if p.is_dir()):
                for f in sorted(rdir.iterdir()):
                    rows.append({"world_hash": wdir.name, "run_key": rdir.name,
                                 "artifact": f.stem, "bytes": f.stat().st_size})
        d = pd.DataFrame(rows)
        if not len(d):
            return d
        return (d.groupby("artifact")["bytes"]
                .agg(["count", "sum", "mean"])
                .assign(MB=lambda x: (x["sum"] / 2**20).round(2))
                .sort_values("sum", ascending=False)
                .drop(columns="sum"))

    def __repr__(self):
        n = len(self.index())
        return f"Store({self.root}, runs={n})"
