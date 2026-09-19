"""The partition store: one district partition per (network, method).

Layout
------
::

    data/01_raw/<network>/
        network.inp  profile.yml  districts.yml       hand-authored inputs
        partitions/                                   written by --pipeline districting
            manual/        districts.yml  partition.yml  boundary.csv  sanity.csv
            spectral/      ...
            fast_greedy/   ...

``manual`` is the hand-authored ``districts.yml`` copied VERBATIM (validated,
then scored like every other method), so the catalog reads every partition
through one path: ``partitions/${globals:districting.active}/districts.yml``.

Nothing here copies ``profile.yml``. Every network-scoped parameter --
``anchor_scale`` first among them -- resolves from the ONE root profile
whatever the partition. The derived-bundle scheme this replaces duplicated the
``.inp``, carried a stale copy of the profile, and wrote a top-level
``hydraulics.anchor_scale`` that the resolver never read.

Identity
--------
``partition.yml`` carries two hashes and they answer different questions:

* ``config_hash`` -- was this built from the CURRENT inputs? It covers the
  method's configuration, ``k``, the ``.inp`` content and, for ``manual``, the
  hand-authored file. A mismatch means "rebuild"; :func:`status` says so and
  the engine acts on it.
* ``partition_id`` -- what the partition IS: a crc over its content (the same
  function the placement POC named bundles with, so ``fast_greedy`` on KY7
  still reads ``2ab65bc6``). World hashes fold this in, so a rebuild that
  changes the cut invalidates every world built on the old one, and a rebuild
  that reproduces it invalidates nothing.

Drift seeds
-----------
``profile.yml``'s ``drift_seed_nodes`` name nodes chosen for the MANUAL
partition. Under a generated cut those nodes can land in another district
(KY7's J-507 is District_A by hand and District_C under fast_greedy), so
:func:`seed_source` hands the profile's seeds to the resolver only for
``manual``; every generated partition falls through to the auto-picker
(largest-base-demand junction), which is the one rule the project uses.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from fedwater.hashing import canonical_hash, stable_hash, text_sha

MANUAL = "manual"
GENERATED = ("spectral", "girvan_newman", "fast_greedy", "walktrap")
METHODS = (MANUAL,) + GENERATED

PARTITIONS_DIR = "partitions"
DISTRICTS_FILE = "districts.yml"
MANIFEST_FILE = "partition.yml"


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
def bundle_dir(project_root, network: str) -> Path:
    return Path(project_root) / "data" / "01_raw" / network


def partition_dir(project_root, network: str, method: str) -> Path:
    return bundle_dir(project_root, network) / PARTITIONS_DIR / method


def check_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"districting method {method!r} is not one of {METHODS}")
    return method


# --------------------------------------------------------------------------
# content identity
# --------------------------------------------------------------------------
def partition_id(doc: dict) -> str:
    """8 hex chars over the partition CONTENT, stable across processes.

    District ORDER is part of the content on purpose: the consumption map is
    positional, so a reordering is a different scenario. Node order within a
    district is not, so it is sorted out of the key.
    """
    districts = doc.get("districts", doc)
    canon = [(str(d), tuple(sorted(str(n) for n in ns)))
             for d, ns in districts.items()]
    return f"{stable_hash(canon):08x}"


def resolve_k(districting: dict, manual_doc: dict) -> int:
    """``k: auto`` -> the manual partition's district count.

    That default is what keeps every positional consumption map written for a
    network (4 tokens on KY7, 5 on D-Town and Graeme) valid under every
    method. ``k`` is a fixed choice, never optimised.
    """
    k = (districting or {}).get("k", "auto")
    if k in (None, "auto"):
        return len((manual_doc or {}).get("districts") or {})
    return int(k)


def method_config(districting: dict, method: str, k: int) -> dict:
    """The part of ``params:districting`` that identifies ONE method's build."""
    check_method(method)
    if method == MANUAL:
        return {"method": MANUAL}
    d = districting or {}
    cfg = {"method": method, "k": int(k), "seed": int(d.get("seed", 0)),
           "on_disconnected": d.get("on_disconnected", "raise"),
           "postprocess": d.get("postprocess") or {},
           "phase3": d.get("phase3") or {}}
    if method == "spectral":
        cfg["weights"] = (d.get("spectral") or {}).get("weights") or {}
    return cfg


def config_hash(network: str, cfg: dict, inp_text: str,
                manual_doc: dict | None) -> str:
    identity = {"network": network, "config": cfg,
                "inp_sha": text_sha(inp_text)}
    if cfg.get("method") == MANUAL:
        identity["manual"] = manual_doc or {}
    return canonical_hash(identity)


# --------------------------------------------------------------------------
# reading the store
# --------------------------------------------------------------------------
def read_manifest(project_root, network: str, method: str) -> dict | None:
    path = partition_dir(project_root, network, method) / MANIFEST_FILE
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or None


def read_partition(project_root, network: str, method: str) -> dict:
    """``{"districts": <doc>, "manifest": <partition.yml>}``, or raise."""
    d = partition_dir(project_root, network, method)
    doc, man = d / DISTRICTS_FILE, d / MANIFEST_FILE
    if not (doc.exists() and man.exists()):
        raise FileNotFoundError(
            f"partition '{method}' of network '{network}' is not built "
            f"({d} is missing {DISTRICTS_FILE} or {MANIFEST_FILE}). Build it "
            "once with `kedro run --pipeline districting` (with "
            f"districting.methods including '{method}'); the experiments "
            "engine does this automatically.")
    return {"districts": yaml.safe_load(doc.read_text()),
            "manifest": yaml.safe_load(man.read_text())}


def expected_hash(project_root, network: str, method: str,
                  districting: dict) -> str:
    """The ``config_hash`` a build from the CURRENT inputs would carry."""
    root = bundle_dir(project_root, network)
    manual = yaml.safe_load((root / DISTRICTS_FILE).read_text())
    k = resolve_k(districting, manual)
    cfg = method_config(districting, method, k)
    return config_hash(network, cfg, (root / "network.inp").read_text(),
                       manual if method == MANUAL else None)


def status(project_root, network: str, method: str, districting: dict) -> str:
    """``current`` | ``missing`` | ``stale``."""
    man = read_manifest(project_root, network, method)
    doc = partition_dir(project_root, network, method) / DISTRICTS_FILE
    if man is None or not doc.exists():
        return "missing"
    want = expected_hash(project_root, network, method, districting)
    return "current" if man.get("config_hash") == want else "stale"


# --------------------------------------------------------------------------
# what a world needs from its partition
# --------------------------------------------------------------------------
def partition_meta(manifest: dict, network_profile: dict) -> dict:
    """The partition facts downstream nodes read, in one small dict.

    It is shaped so that it can stand in for ``network_profile`` wherever the
    drift-seed precedence is resolved (``name`` + ``drift_seed_nodes``), which
    is how ``build_drift_schedule`` stays unchanged.
    """
    manifest = manifest or {}
    method = manifest.get("method")
    if method is None:
        raise ValueError("partition.yml carries no `method`; rebuild the "
                         "partition with `kedro run --pipeline districting`")
    return {
        "name": (network_profile or {}).get("name", manifest.get("network")),
        "network": manifest.get("network"),
        "method": method,
        "partition_id": manifest.get("partition_id"),
        "config_hash": manifest.get("config_hash"),
        "inp_sha": manifest.get("inp_sha"),
        "k": manifest.get("k"),
        "districts": list(manifest.get("sizes") or {}),
        "drift_seed_nodes": seed_source(method, network_profile),
    }


def seed_source(method: str, network_profile: dict) -> dict:
    """Per-district seed overrides valid for ``method``: the profile's for
    ``manual``, none for a generated partition (auto-picker decides)."""
    if method != MANUAL:
        return {}
    return dict((network_profile or {}).get("drift_seed_nodes") or {})
