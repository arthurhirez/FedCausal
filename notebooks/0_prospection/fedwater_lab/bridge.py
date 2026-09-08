"""The single seam to `fedwater`. Nothing upstream is ever re-implemented.

Every symbol this package borrows from the pipeline is resolved here, so
there is exactly one place to look when the upstream API moves.

One wrinkle it absorbs: several `fedwater.pipelines.*` packages import kedro
in their `__init__.py`, so a plain `from fedwater.pipelines.x.nodes import f`
drags kedro in even though `nodes.py` itself only needs numpy/pandas/torch.
When kedro is absent (a bare analysis venv, CI) the fallback loads the very
same `nodes.py` file directly by path -- same source, same behaviour, no
package side effects. If `fedwater` itself is missing, the error says so.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from types import ModuleType

__all__ = ["nodes_module", "preprocess_clients", "deseasonalize",
           "AER", "FPLTrainer", "aggregate_prototypes", "extract_prototypes",
           "effective_fl_seed", "compute_drift_signals",
           "deconfounded_prototypes", "FINCH", "fedwater_root"]


def fedwater_root():
    """Path of the installed `fedwater` package, or None."""
    try:
        spec = importlib.util.find_spec("fedwater")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    import pathlib
    return pathlib.Path(list(spec.submodule_search_locations)[0])


def _stub_package(dotted: str, path) -> ModuleType:
    """Register `dotted` as a package rooted at `path` WITHOUT running its
    `__init__.py`.

    Needed because some pipeline packages import kedro at `__init__` time
    while their `nodes.py` does not. Registering the stub first means a
    relative import inside `nodes.py` (`from . import methods`) resolves
    against `__path__` and never triggers the real `__init__`.
    """
    if dotted in sys.modules:
        return sys.modules[dotted]
    mod = ModuleType(dotted)
    mod.__path__ = [str(path)]           # marks it a package for the finders
    mod.__package__ = dotted
    sys.modules[dotted] = mod
    parent, _, leaf = dotted.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)
    return mod


@lru_cache(maxsize=None)
def nodes_module(pipeline: str, module: str = "nodes") -> ModuleType:
    """`fedwater.pipelines.<pipeline>.<module>`, kedro or no kedro."""
    dotted = f"fedwater.pipelines.{pipeline}.{module}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    try:
        return importlib.import_module(dotted)
    except ImportError as exc:
        if "kedro" not in str(exc):
            raise
        root = fedwater_root()
        if root is None:
            raise ImportError(
                "the `fedwater` package is not importable -- put <repo>/src on "
                "sys.path (the lab modules never re-implement pipeline code)"
            ) from exc
        path = root / "pipelines" / pipeline / f"{module}.py"
        if not path.exists():
            raise ImportError(f"{dotted}: no such module at {path}") from exc

        _stub_package("fedwater", root)
        _stub_package("fedwater.pipelines", root / "pipelines")
        _stub_package(f"fedwater.pipelines.{pipeline}", root / "pipelines" / pipeline)

        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod


# --------------------------------------------------------------- borrowed API
def preprocess_clients(*args, **kw):
    """`fl_preprocessing.nodes.preprocess_clients`, verbatim."""
    return nodes_module("fl_preprocessing").preprocess_clients(*args, **kw)


def deseasonalize(*args, **kw):
    """`dependence_oracle.nodes.deseasonalize` -- fedwater invariant 5."""
    return nodes_module("dependence_oracle").deseasonalize(*args, **kw)


def compute_drift_signals(*args, **kw):
    """`drift_detection.nodes.compute_drift_signals`, verbatim."""
    return nodes_module("drift_detection").compute_drift_signals(*args, **kw)


def deconfounded_prototypes(*args, **kw):
    """`personalization.nodes._deconfounded_prototypes` (per-month median removed)."""
    return nodes_module("personalization")._deconfounded_prototypes(*args, **kw)


def effective_fl_seed(*args, **kw):
    return nodes_module("fl_training").effective_fl_seed(*args, **kw)


def _training(module: str):
    return nodes_module("fl_training", module)


def AER(*args, **kw):                    # noqa: N802 - mirrors the class name
    return _training("aer").AER(*args, **kw)


def aer_class():
    return _training("aer").AER


def FPLTrainer():                        # noqa: N802 - returns the class
    """The `FPLTrainer` CLASS (subclassed by `train.SnapshotFPLTrainer`)."""
    return _training("federated").FPLTrainer


def aggregate_prototypes(*args, **kw):
    return _training("federated").aggregate_prototypes(*args, **kw)


def extract_prototypes(*args, **kw):
    return _training("federated").extract_prototypes(*args, **kw)


def FINCH():                             # noqa: N802 - returns the function
    return _training("finch").FINCH
