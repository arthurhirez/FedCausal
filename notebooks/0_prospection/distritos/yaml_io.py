"""The output contract: ``districts.yml``, and the bundle it lives in.

Bundle layout
-------------
Each network is a self-describing directory::

    data/01_raw/<network>/
        network.inp
        districts.yml

:func:`find_bundle` walks up from wherever it is called, so a notebook in a POC
subdirectory and a Kedro node running from the project root resolve the same
bundle without either of them hardcoding ``../..``.

The schema, and why the split is load-bearing
---------------------------------------------
::

    districts:                 # MUST partition the junction set EXACTLY
      District_A: [J511, ...]
    assets:                    # tanks and reservoirs belonging to each district
      District_A: [T4]

``districts`` is the simulation domain: demand portfolios, drift diffusion and
inter-district boundary pipes are all derived from it, so a junction appearing
twice or not at all corrupts everything downstream. ``assets`` records storage
and sources. The split is not cosmetic -- a tank listed under ``districts``
would break portfolio building and make every tank feed read as an
inter-district boundary pipe.

:func:`validate_districts` checks the contract instead of trusting it, and is
run by the sanity suite on every write.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

BUNDLE_ROOT = Path("data/01_raw")
NETWORK_FILE = "network.inp"
DISTRICTS_FILE = "districts.yml"


# ==========================================================================
# locating the bundle
# ==========================================================================

def find_bundle(network: str, start: str | Path | None = None,
                max_up: int = 6) -> Path:
    """Resolve ``data/01_raw/<network>/`` by walking up from ``start``.

    Walking up rather than accepting a relative path means the same call works
    from the project root, from ``notebooks/``, and from a POC subdirectory --
    the three places this code gets run from -- with no ``../..`` anywhere.
    """
    here = Path(start or Path.cwd()).resolve()
    for base in [here, *here.parents][:max_up + 1]:
        cand = base / BUNDLE_ROOT / network
        if (cand / NETWORK_FILE).exists():
            return cand
    raise FileNotFoundError(
        "no bundle %s/%s/%s found walking up %d levels from %s"
        % (BUNDLE_ROOT, network, NETWORK_FILE, max_up, here))


def load_network(path: str | Path):
    """Read an ``.inp`` into a wntr model."""
    import wntr

    return wntr.network.WaterNetworkModel(str(path))


# ==========================================================================
# the schema
# ==========================================================================

def districts_mapping(wn, labels: pd.Series) -> dict:
    """Split a full-node labelling into the ``districts`` / ``assets`` schema.

    Junctions go to ``districts``; tanks and reservoirs to ``assets``. Anything
    else a model might carry is dropped from the file rather than guessed at,
    and :func:`validate_districts` will say so if that loses a node that
    downstream needs.
    """
    from .graph import source_nodes

    juncs, assets = {}, {}
    junction_set, source_set = set(wn.junction_name_list), set(source_nodes(wn))
    for node, d in labels.items():
        d = str(d)
        if node in junction_set:
            juncs.setdefault(d, []).append(node)
        elif node in source_set:
            assets.setdefault(d, []).append(node)
    return {"districts": {d: sorted(v) for d, v in sorted(juncs.items())},
            "assets": {d: sorted(v) for d, v in sorted(assets.items()) if v}}


def write_districts_yml(path, wn, labels: pd.Series, header: str = "") -> Path:
    """Write the project ``districts.yml`` schema, flow-style, one line per district.

    The formatting matches the file already consumed downstream -- flow
    sequences with single-quoted names -- so a regenerated file is a clean diff
    against a hand-maintained one rather than a wholesale reformat.
    """
    path = Path(path)
    mapping = districts_mapping(wn, labels)

    lines = [l if l.startswith("#") else "# " + l for l in header.splitlines()]
    if lines:
        lines.append("")
    for section in ("districts", "assets"):
        lines.append("%s:" % section)
        for d, members in mapping[section].items():
            lines.append("  %s: [%s]" % (d, ", ".join("'%s'" % n for n in members)))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def read_districts_yml(path) -> dict:
    """Read a ``districts.yml`` into ``{"districts": {...}, "assets": {...}}``."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    return {"districts": doc.get("districts") or {}, "assets": doc.get("assets") or {}}


def labels_from_mapping(mapping: dict) -> pd.Series:
    """Flatten ``{district: [nodes]}`` (or a full document) into a node->district Series."""
    if "districts" in mapping:
        mapping = mapping["districts"]
    return pd.Series({n: d for d, members in mapping.items() for n in members},
                     name="district")


# ==========================================================================
# the contract, checked
# ==========================================================================

def validate_districts(wn, doc: dict) -> pd.DataFrame:
    """Check a ``districts.yml`` document against the schema contract.

    Returns one row per check with ``ok`` and a human-readable ``detail``, so a
    failure says which nodes are wrong rather than only that something is.
    """
    from .graph import source_nodes

    districts = doc.get("districts") or {}
    assets = doc.get("assets") or {}
    listed = [n for members in districts.values() for n in members]
    listed_set, junction_set = set(listed), set(wn.junction_name_list)
    asset_listed = [n for members in assets.values() for n in members]
    source_set = set(source_nodes(wn))

    dupes = sorted({n for n in listed if listed.count(n) > 1})
    missing = sorted(junction_set - listed_set)
    extra = sorted(listed_set - junction_set)
    asset_dupes = sorted({n for n in asset_listed if asset_listed.count(n) > 1})
    asset_missing = sorted(source_set - set(asset_listed))
    asset_wrong = sorted(set(asset_listed) - source_set)
    empty = sorted(d for d, m in districts.items() if not m)

    checks = [
        ("districts cover every junction", not missing,
         "ok" if not missing else "%d junction(s) unassigned: %s" % (len(missing), missing[:8])),
        ("districts contain only junctions", not extra,
         "ok" if not extra else "%d non-junction(s) listed: %s" % (len(extra), extra[:8])),
        ("no junction listed twice", not dupes,
         "ok" if not dupes else "%d duplicate(s): %s" % (len(dupes), dupes[:8])),
        ("no empty district", not empty,
         "ok" if not empty else "empty: %s" % empty),
        ("assets cover every tank/reservoir", not asset_missing,
         "ok" if not asset_missing else "unrecorded: %s" % asset_missing[:8]),
        ("assets contain only tanks/reservoirs", not asset_wrong,
         "ok" if not asset_wrong else "not a source: %s" % asset_wrong[:8]),
        ("no asset listed twice", not asset_dupes,
         "ok" if not asset_dupes else "duplicate(s): %s" % asset_dupes[:8]),
        ("assets never appear under districts", not (set(asset_listed) & listed_set),
         "ok" if not (set(asset_listed) & listed_set)
         else "in both: %s" % sorted(set(asset_listed) & listed_set)[:8]),
    ]
    return pd.DataFrame([{"check": c, "ok": bool(ok), "detail": d} for c, ok, d in checks])
