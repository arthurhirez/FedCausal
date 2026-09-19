"""Reading ``conf/`` outside a KedroSession, the way Kedro reads it.

Why this exists
---------------
The experiments engine, the spec expander and the partition-freshness check
all read ``conf/base/*.yml`` with plain ``yaml.safe_load``, because opening a
KedroSession per world would be slow and would resolve ``${globals:network}``
to whatever the PROJECT selects rather than to the world being built. Two
things Kedro does for us are therefore redone here, and only these two:

* **env layering** -- ``conf/local/globals.yml`` merges over
  ``conf/base/globals.yml``. Kedro's merge is destructive at the TOP level, so
  a local ``districting:`` block replaces the base one whole; the same rule
  is applied here.
* **``${globals:...}`` interpolation** in parameters. ``parameters.yml``
  forwards two globals into nodes (``districting.methods`` and the probe
  store path). A value that is EXACTLY one interpolation is substituted; an
  interpolation embedded in a longer string is left alone, because nothing in
  this project writes one and supporting it would mean re-implementing
  OmegaConf rather than mirroring the one case we use.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

_GLOBAL_REF = re.compile(r"^\$\{globals:([A-Za-z0-9_.]+)(?:,\s*(.*))?\}$")


def _lookup(tree: dict, dotted: str):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def load_globals(project_root, override: dict | None = None) -> dict:
    """``base`` then ``local`` globals, then ``override`` -- each top-level
    key replaced whole, as Kedro does."""
    out: dict = {}
    for env in ("base", "local"):
        path = Path(project_root) / "conf" / env / "globals.yml"
        if path.exists():
            out.update(yaml.safe_load(path.read_text()) or {})
    out.update(copy.deepcopy(override or {}))
    return out


def resolve_globals(tree, globals_: dict):
    """Substitute every value that is exactly ``${globals:key[, default]}``."""
    if isinstance(tree, dict):
        return {k: resolve_globals(v, globals_) for k, v in tree.items()}
    if isinstance(tree, list):
        return [resolve_globals(v, globals_) for v in tree]
    if isinstance(tree, str):
        m = _GLOBAL_REF.match(tree.strip())
        if m:
            try:
                return copy.deepcopy(_lookup(globals_, m.group(1)))
            except KeyError:
                if m.group(2) is not None:
                    return yaml.safe_load(m.group(2))
                raise KeyError(
                    f"parameters reference ${{globals:{m.group(1)}}} but "
                    "conf/base/globals.yml does not define it") from None
    return tree


def load_base_params(project_root, globals_: dict | None = None) -> dict:
    """``conf/base/parameters.yml`` with its globals interpolations resolved."""
    root = Path(project_root)
    raw = yaml.safe_load((root / "conf/base/parameters.yml").read_text()) or {}
    g = load_globals(root) if globals_ is None else globals_
    return resolve_globals(raw, g)


def active_partition(globals_: dict) -> str:
    """The partition method a world is built on (``districting.active``)."""
    method = ((globals_ or {}).get("districting") or {}).get("active")
    if not method:
        raise KeyError(
            "conf/base/globals.yml has no `districting.active`. It names the "
            "partition under data/01_raw/<network>/partitions/ that worlds are "
            "built on (manual | spectral | girvan_newman | fast_greedy | "
            "walktrap).")
    return str(method)
