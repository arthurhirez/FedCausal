"""fedwater_lab -- the unified sandbox for placement x preprocessing x FL.

One data path, one trainer, one store, one set of metrics. `fedwater`'s
pipeline code is imported through `bridge` and never re-implemented.

Typical use:

    from fedwater_lab import load_world, Store, POC, run_cell
    world = load_world("data/09_experiments/worlds/<hash>")
    store = Store("notebooks/prospection/lab_sandbox")
    res   = run_cell(world, POC)
    store.save_run(world, POC, res)
"""
from __future__ import annotations

from .specs import (POC, CellSpec, Geometry, ModelCfg, TrainCfg, expand,
                    KINDS, PLACEMENTS, SCALINGS, TRANSFORMS)
from .store import Store
from .worlds import World, load_world

__all__ = ["POC", "CellSpec", "Geometry", "ModelCfg", "TrainCfg", "expand",
           "KINDS", "PLACEMENTS", "SCALINGS", "TRANSFORMS",
           "Store", "World", "load_world", "run_cell", "run_grid"]


def __getattr__(name):
    """Defer torch-importing modules until first use."""
    if name in ("run_cell", "run_grid"):
        from . import runner
        return getattr(runner, name)
    raise AttributeError(name)
