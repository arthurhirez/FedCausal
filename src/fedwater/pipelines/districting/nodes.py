"""District attribution nodes: build, document, report.

The partitioning itself lives in :mod:`fedwater.districting` and is used as
shipped -- one ``run`` per method on one shared graph, then the shared sanity
suite and comparison. What this module adds is the pipeline contract:

* **the gate.** A partition is written only if the simulation can consume it:
  :func:`fedwater.networks.partition.validate_partition` must pass (the check
  ``network_prep`` runs on every world), no node may be left in
  ``District_UNASSIGNED``, and a generated partition must reach the requested
  ``k`` -- consumption maps are positional, so a 3-district cut on a network
  whose maps have 4 tokens is not a smaller world, it is an invalid one.
  Everything else the sanity suite reports (contiguity, rebalancing
  tolerance, ...) is recorded, not raised, exactly as the POC treated it.
* **``manual`` is adopted, not re-derived.** The hand-authored
  ``districts.yml`` is copied byte-for-byte; it is only *scored* through the
  ``external`` method, with post-processing forced off so the report
  describes the file as written.
* **identity.** Each partition's ``partition.yml`` records the build
  configuration hash and the content id (see
  :mod:`fedwater.networks.partitions`).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yaml

from fedwater import districting as dst
from fedwater.hashing import text_sha
from fedwater.networks import partitions as store
from fedwater.networks.partition import validate_partition

_AUTO = (None, "auto")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def _methods(districting: dict) -> list[str]:
    methods = list(districting.get("methods") or [])
    if not methods:
        raise ValueError("districting.methods is empty -- list the methods to "
                         "build in conf/base/globals.yml (districting.methods)")
    for m in methods:
        store.check_method(m)
    if len(set(methods)) != len(methods):
        raise ValueError(f"districting.methods has duplicates: {methods}")
    return methods


def _weights(districting: dict) -> dst.DistrictingWeights:
    w = dict((districting.get("spectral") or {}).get("weights") or {})
    if w.get("elevation_bandwidth_m") in _AUTO:
        w["elevation_bandwidth_m"] = None     # DistrictingWeights: None = auto
    return dst.DistrictingWeights(**w).validate()


def district_config(districting: dict, method: str, k: int) -> dst.DistrictingConfig:
    """``params:districting`` -> the POC's ``DistrictingConfig`` for one method.

    ``auto`` is this project's spelling of the POC's ``None`` / ``"default"``
    (per-method defaults): ``null`` is reserved in ``parameters.yml`` for
    network-scoped keys filled from the bundle.
    """
    post = districting.get("postprocess") or {}
    p3 = districting.get("phase3") or {}
    repair = post.get("repair", "auto")
    balance = post.get("balance_target", "auto")
    optimizer = p3.get("optimizer", "none")
    criteria = (dst.HydraulicCriteria(**p3["criteria"])
                if optimizer != "none" and p3.get("criteria") else None)
    return dst.DistrictingConfig(
        k=int(k), method=method, seed=int(districting.get("seed", 0)),
        weights=_weights(districting),
        repair=None if repair in _AUTO else bool(repair),
        balance_target="default" if balance in _AUTO else balance,
        balance_tol=float(post.get("balance_tol", 0.15)),
        on_disconnected=districting.get("on_disconnected", "raise"),
        criteria=criteria, optimizer=optimizer,
        budget=int(p3.get("budget", 20000)), nsga2=dict(p3.get("nsga2") or {}),
        force_optimize=bool(p3.get("force_optimize", False)),
    )


def _coupling(network: str, districting: dict):
    """The optional measured coupling matrix, only when its dial is on."""
    w = ((districting.get("spectral") or {}).get("weights") or {})
    if float(w.get("w_coupling", 0.0) or 0.0) <= 0:
        return None
    S = dst.coupling.load_coupling(network)
    if S is None:
        raise FileNotFoundError(
            f"districting.spectral.weights.w_coupling > 0 but no coupling "
            f"matrix exists for '{network}' under "
            "data/08_reporting/dependency/ground_truth/. Set w_coupling: 0.0 "
            "or supply the matrix.")
    return S


# --------------------------------------------------------------------------
# 1. build
# --------------------------------------------------------------------------
def _gate(wn, method: str, doc: dict, k_requested: int, achieved: int) -> None:
    unassigned = [n for n, ns in doc["districts"].items()
                  if n == "District_UNASSIGNED" and ns]
    if unassigned:
        raise ValueError(
            f"[{method}] nodes were left in District_UNASSIGNED (the graph is "
            "disconnected and on_disconnected='largest'); a world cannot be "
            "built on that partition.")
    validate_partition(wn, doc)   # the same contract network_prep enforces
    if method != store.MANUAL and achieved != k_requested:
        raise ValueError(
            f"[{method}] produced {achieved} districts, {k_requested} were "
            "requested. Consumption maps are positional, so this partition "
            "would silently change what every map means; not written.")


def build_partitions(wn, districts_manual: dict, network_profile: dict,
                     districting: dict) -> dict:
    """Run every requested method on one shared graph. ``{method: result}``."""
    network = network_profile["name"]
    methods = _methods(districting)
    k = store.resolve_k(districting, districts_manual)
    reference = dst.metrics.published_dma_tags(wn) or None
    coupling = _coupling(network, districting) if "spectral" in methods else None
    g = dst.build_graph(wn)

    results = {}
    for m in methods:
        if m == store.MANUAL:
            manual = {d: [str(n) for n in ns]
                      for d, ns in districts_manual["districts"].items()}
            cfg = district_config(districting, "external", len(manual))
            # adopted as written: no repair, no rebalancing, no Phase 3
            cfg.repair, cfg.balance_target = False, None
            cfg.criteria, cfg.optimizer = None, "none"
            res = dst.run(wn, cfg, network=network, mapping=manual,
                          reference=reference, graph=g)
            res.method = store.MANUAL
            doc = districts_manual
        else:
            res = dst.run(wn, district_config(districting, m, k),
                          network=network, coupling=coupling,
                          reference=reference, graph=g)
            doc = res.districts_mapping(wn)
        _gate(wn, m, doc, k, len(doc["districts"]))
        results[m] = res
        print(f"districting[{network}] {m:14s} k={res.k} "
              f"sizes={'/'.join(str(len(v)) for v in doc['districts'].values())} "
              f"boundary={len(res.boundary)}")
    return results


# --------------------------------------------------------------------------
# 2. documents
# --------------------------------------------------------------------------
def _doc(res, wn, districts_manual: dict) -> dict:
    return districts_manual if res.method == store.MANUAL \
        else res.districts_mapping(wn)


def partition_documents(results: dict, wn, districts_manual: dict,
                        districts_manual_text: str, network_inp_text: str,
                        network_profile: dict, districting: dict) -> dict:
    """``{"<method>/districts": text, "<method>/partition": text}``."""
    network = network_profile["name"]
    k = store.resolve_k(districting, districts_manual)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = {}
    for m, res in results.items():
        doc = _doc(res, wn, districts_manual)
        if m == store.MANUAL:
            text = districts_manual_text          # verbatim, comments and all
            cfg = store.method_config(districting, m, len(doc["districts"]))
        else:
            text = dst.yaml_io.render_districts_yml(doc, res.default_header())
            cfg = store.method_config(districting, m, k)
        manifest = {
            "network": network,
            "method": m,
            "k": len(doc["districts"]),
            "config_hash": store.config_hash(
                network, cfg, network_inp_text,
                districts_manual if m == store.MANUAL else None),
            "partition_id": store.partition_id(doc),
            "inp_sha": text_sha(network_inp_text),
            "config": cfg,
            "sizes": {d: len(ns) for d, ns in doc["districts"].items()},
            "assets": {d: list(ns) for d, ns in (doc.get("assets") or {}).items()},
            "n_boundary_links": int(len(res.boundary)),
            "all_contiguous": bool(res.report["contiguity"].contiguous.all()),
            "created_utc": created,
        }
        out[f"{m}/{store.DISTRICTS_FILE[:-4]}"] = text
        out[f"{m}/{store.MANIFEST_FILE[:-4]}"] = yaml.safe_dump(
            manifest, sort_keys=False)
    return out


# --------------------------------------------------------------------------
# 3. reports -- POC sections 6 (sanity) and 7 (summary)
# --------------------------------------------------------------------------
def partition_reports(results: dict, wn, districts_manual: dict,
                      network_profile: dict, districting: dict):
    """Per-method boundary/sanity tables, and the cross-method comparison."""
    network = network_profile["name"]
    manual = {d: [str(n) for n in ns]
              for d, ns in districts_manual["districts"].items()}
    tables, sanity = {}, []
    for m, res in results.items():
        sc = dst.sanity_check(
            wn, res, mapping=manual if m == store.MANUAL else None)
        sc.insert(0, "method", m)
        sanity.append(sc)
        tables[f"{m}/boundary"] = res.status.reset_index(drop=True)
        tables[f"{m}/sanity"] = sc

    summary = dst.compare(results)
    summary.insert(2, "partition_id", [
        store.partition_id(_doc(results[m], wn, districts_manual))
        for m in summary["method"]])

    g = dst.build_graph(wn)
    k = store.resolve_k(districting, districts_manual)
    order = [dst.partition.order_sensitivity(g, wn, k, m) for m in results
             if m in dst.partition.COMMUNITY_METHODS]
    order = pd.DataFrame(order, columns=[
        "method", "ami_between_orders", "modularity_declared",
        "modularity_lexical", "modularity_gap", "order_sensitive"])
    order.insert(0, "network", network)
    return tables, summary, pd.concat(sanity, ignore_index=True), order
