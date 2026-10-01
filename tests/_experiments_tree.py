"""A REAL experiments tree, in the engine's on-disk layout, built in-process.

The engine runs each world as ``kedro run`` in a subprocess of a project
clone, which needs the project's ``pyproject.toml``. This builder does the
same work in-process: the same clone layout (``conf/base`` + the engine's own
``_write_local`` overrides + ``data/01_raw``), the project's own
``__default__`` pipeline over a catalog built from the clone's conf, the same
``WORLD_REQUIRED`` check, and a manifest with the keys ``ensure_world``
writes. Micro horizons (7-day months) keep it to a couple of minutes.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Graeme (the network the stack integration test builds on the same micro
# probe horizon): two partitions, and on `manual` two drift targets that
# share one probe stack (planned probe horizon ignores the drift target)
NETWORK = "graeme"
WORLDS = (
    ("demo_study", "manual", "District_A"),
    ("demo_study", "manual", "District_B"),
    ("demo_study", "spectral", "District_A"),
    ("other_study", "manual", "District_A"),     # a world two studies share
)


def world_spec(method: str, target: str) -> dict:
    return {"network": NETWORK, "districting": method, "sim_seed": 43,
            "n_months": 6, "days_per_month": 7, "drift_ramp_days": 3,
            "seasonality_scale": 0.0,
            "drift": {"tgt_district": target, "warmup_months": 2,
                      "to_income": "low", "to_land_use": "commercial"},
            "placement": {"source": "dynamic",
                          "probe": {"warmup_months": 4, "settled_months": 5,
                                    "days_per_month": 7}},
            "oracle": {"tiers": [1]}}


def _run_world(clone: Path) -> float:
    from kedro.framework.project import configure_project
    from kedro.runner import SequentialRunner

    from fedwater.pipeline_registry import register_pipelines
    from fedwater.reporting import world_catalog

    configure_project("fedwater")
    catalog, params, _ = world_catalog(clone)
    catalog["parameters"] = params
    for k, v in params.items():
        catalog[f"params:{k}"] = v
    t0 = time.time()
    SequentialRunner().run(register_pipelines()["__default__"], catalog)
    return time.time() - t0


def build(root: Path) -> Path:
    from fedwater.config import load_base_params
    from fedwater.experiments import spec as spec_mod
    from fedwater.experiments.engine import WORLD_REQUIRED, ExperimentEngine

    root = Path(root)
    base = load_base_params(ROOT)
    engine = ExperimentEngine.__new__(ExperimentEngine)       # for its helpers
    engine.root = root
    for study, method, target in WORLDS:
        b = spec_mod.bundle(ROOT, NETWORK, method)
        w = world_spec(method, target)
        net, meth = w.pop("network"), w.pop("districting")
        n = len(b["districts"]["districts"])
        w, resolved = spec_mod._resolve_seeded(w, base, net, n, b)
        h = resolved["sim_hash"]
        wdir = root / "worlds" / h
        if not (wdir / "manifest.json").exists():
            clone = wdir / "clone"
            shutil.copytree(ROOT / "conf" / "base", clone / "conf" / "base")
            shutil.copytree(ROOT / "data" / "01_raw" / NETWORK,
                            clone / "data" / "01_raw" / NETWORK)
            ExperimentEngine._write_local(clone, resolved["override"],
                                          engine._globals(resolved))
            seconds = _run_world(clone)
            missing = [r for r in WORLD_REQUIRED if not (clone / r).exists()]
            (wdir / "manifest.json").write_text(json.dumps({
                "sim_hash": h, "status": "ok" if not missing else
                "sim_incomplete:" + ",".join(missing),
                "world": resolved["flat"], "override": resolved["override"],
                "effective": resolved["effective"], "seconds": seconds,
                "created_utc": "2026-09-30T00:00:00+00:00",
                "src_hash": "test"}, default=str, indent=1))
        rdir = root / study / "runs" / f"{h}__run0"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "manifest.json").write_text(json.dumps(
            {"sim_hash": h, "run_hash": "run0", "status": "ok",
             "world": resolved["flat"], "run": {}}))
    return root
