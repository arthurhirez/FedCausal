"""placement.py -- turn a configuration's mixture into shippable sensor sets.

What this produces
------------------
Per configuration, four complete `sensors:` blocks, each of shape
`{district: {"pressure": [...], "flow": [...]}}` with at most `n_max` of each
kind per district. Together with `districts.yml` they are the
`(district attribution, sensor placement)` tuple the federated step consumes,
and having several of them means the FL side can run matched arms rather than
one placement and a shrug.

    control_high   gauges that read almost only their own district
                   (tier `core`, purity -> 1). The best case for a client
                   whose signal is its own.
    control_low    gauges that respond strongly and to the WRONG district
                   (tier `foreign`, purity -> 0). Not "noisy": these are
                   transit pipes whose flow reverses when the local district
                   draws more, so a client metered here is reading a
                   neighbour. The informative failure.
    target         gauges carrying a QUANTIFIED BLEND -- purity inside a band
                   with a named second district holding real mass. The set the
                   thesis needs: dependence that is measured rather than
                   assumed.
    unusable       gauges whose signature did not move for any district's
                   drift. Report-only, never a pipeline arm: a client metered
                   here gets a flat series and the FL run degenerates rather
                   than failing informatively.

Why every arm ranks on purity
-----------------------------
`signal_probe.r_res_own` would be the natural score for "resembles its own
district" and is a perfectly good number -- it is reported alongside. But the
target set is mixture-defined by construction, so controls scored on a
different axis would not be matched controls, and the three arms would not be
comparable. One axis, three regions of it.

Ragged sets are shipped, not padded
-----------------------------------
A district can simply have no gauge meeting the target criteria -- pressure
especially, where the discriminative gap is two orders of magnitude below
flow's. Padding to `n_max` from outside the criteria would put gauges in the
target arm that are not targets, which is the one thing this module must not
do. So short sets ship, and every row carries `n_selected` against `n_eligible`
so the downstream knows. The cost is real and is named here rather than
discovered later: client CSVs then have different column counts per kind, and
`preprocess_clients` reads that as a different feature geometry per client.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import yaml

import signal_probe as sp

__all__ = ["ARMS", "PIPELINE_ARMS", "ArmSpec", "classification", "eligible",
           "select", "selection_table", "coverage", "emit", "sensors_yaml"]


# ==========================================================================
# arm definitions
# ==========================================================================
class ArmSpec:
    """One arm: which gauges qualify, and in what order they are preferred."""

    def __init__(self, name, tier=None, purity_band=None, min_w_second=0.0,
                 min_n_mass=1, require_stable=False, max_purity_swing=None,
                 order=(), ascending=(), doc=""):
        self.name = name
        self.tier = tier
        self.purity_band = purity_band
        self.min_w_second = float(min_w_second)
        self.min_n_mass = int(min_n_mass)
        self.require_stable = bool(require_stable)
        self.max_purity_swing = max_purity_swing
        self.order = tuple(order)
        self.ascending = tuple(ascending)
        self.doc = doc


ARMS = {
    "control_high": ArmSpec(
        "control_high", tier="core", min_n_mass=1,
        order=("purity", "snr_home"), ascending=(False, False),
        doc="reads almost only its own district"),
    "control_low": ArmSpec(
        "control_low", tier="foreign", min_n_mass=1,
        order=("purity", "snr_home"), ascending=(True, False),
        doc="responds strongly, and to another district"),
    "target": ArmSpec(
        "target", tier="transition", purity_band=(0.25, 0.75),
        min_w_second=0.20, min_n_mass=2, require_stable=True,
        max_purity_swing=0.20,
        # Most stable blend first, strongest response as the tie-break. Not
        # "closest to 50/50": a 25/75 that holds across thresholds is a better
        # quantified mixture than a 50/50 that is an artefact of one.
        order=("purity_swing", "snr_home"), ascending=(True, False),
        doc="a quantified blend of two named districts"),
    "unusable": ArmSpec(
        "unusable", tier="unusable",
        order=("snr_home",), ascending=(True,),
        doc="did not move for any drift -- report only"),
}

# The arms the automated pipeline ships. `unusable` is built and reported as
# an edge case, in the same spirit as the walktrap partition.
PIPELINE_ARMS = ("control_high", "control_low", "target")


# ==========================================================================
# the classification table
# ==========================================================================
def classification(res) -> pd.DataFrame:
    """Every node and link of the configuration, tiered.

    This is the full read-out Goal 3 asks for -- not just the selected gauges.
    `tier` is `mixture_probe.classify`'s verdict; `purity_swing` and
    `tier_stable` say whether that verdict survives the threshold sweep, which
    is what separates a measured mixture from a threshold artefact.
    """
    if not len(res.mixture):
        return pd.DataFrame()
    cols = ["id", "element", "kind", "home", "role", "tier", "purity",
            "second", "w_second", "snr_home", "n_live", "n_mass", "mass",
            "hhi", "entropy", "neg_mass", "degenerate_null"]
    m = res.mixture[[c for c in cols if c in res.mixture.columns]].copy()
    wcols = [c for c in res.mixture.columns if c.startswith("w_District")]
    m = pd.concat([m, res.mixture[wcols]], axis=1)
    if len(res.stability):
        m = m.merge(res.stability, on="id", how="left")
    if len(res.scores):
        # the resemblance axis, carried alongside so an arm can be audited
        # against the score it was NOT chosen on
        s = (res.scores[res.scores["phase"] == "init"]
             .groupby("id")[["r_res_own", "response"]].mean().reset_index()
             .rename(columns={"r_res_own": "r_res_own_init",
                              "response": "response_mean"}))
        m = m.merge(s, on="id", how="left")
    return m.assign(config_id=res.config_id, method=res.method)


# ==========================================================================
# selection
# ==========================================================================
def eligible(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Rows of `classification` that qualify for `arm`, unordered and uncapped."""
    spec = ARMS[arm]
    d = table
    if spec.tier is not None:
        d = d[d["tier"] == spec.tier]
    if spec.purity_band is not None:
        lo, hi = spec.purity_band
        d = d[d["purity"].between(lo, hi)]
    if spec.min_w_second > 0:
        d = d[d["w_second"].fillna(0.0) >= spec.min_w_second]
    if spec.min_n_mass > 1 and "n_mass" in d:
        d = d[d["n_mass"].fillna(0) >= spec.min_n_mass]
    if spec.require_stable and "tier_stable" in d:
        d = d[d["tier_stable"].fillna(False).astype(bool)]
    if spec.max_purity_swing is not None and "purity_swing" in d:
        d = d[d["purity_swing"].fillna(1.0) <= spec.max_purity_swing]
    if "degenerate_null" in d:
        # A gauge whose init series is constant has a null of ~0 and an SNR
        # that is not a measurement. It cannot be ranked on `snr_home`.
        d = d[~d["degenerate_null"].fillna(False).astype(bool)]
    return d


def select(res, arm: str, n_max: int = 5,
           table: pd.DataFrame | None = None) -> dict:
    """The arm's `sensors:` block: <= `n_max` per district per kind."""
    table = classification(res) if table is None else table
    spec = ARMS[arm]
    d = eligible(table, arm)
    out = {}
    for district in res.names:
        pick = {}
        for kind in ("pressure", "flow"):
            sub = d[(d["home"] == district) & (d["kind"] == kind)]
            order = [c for c in spec.order if c in sub.columns]
            asc = [a for c, a in zip(spec.order, spec.ascending)
                   if c in sub.columns]
            if order:
                sub = sub.sort_values(order, ascending=asc)
            pick[kind] = [i[len(sp._PREFIX[kind]):] for i in sub["id"].head(n_max)]
        out[district] = pick
    return out


def selection_table(res, arms=PIPELINE_ARMS, n_max: int = 5,
                    table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every selected gauge, with the values it was selected on.

    One row per (arm, district, kind, rank). This is the audit trail: an arm
    is only trustworthy if you can see the purity and the second district of
    each gauge it put in front of the trainer.
    """
    table = classification(res) if table is None else table
    keep = ["id", "element", "kind", "home", "role", "tier", "purity",
            "second", "w_second", "snr_home", "purity_swing", "tier_stable",
            "r_res_own_init", "response_mean"]
    rows = []
    for arm in arms:
        spec = ARMS[arm]
        d = eligible(table, arm)
        for district in res.names:
            for kind in ("pressure", "flow"):
                sub = d[(d["home"] == district) & (d["kind"] == kind)]
                order = [c for c in spec.order if c in sub.columns]
                asc = [a for c, a in zip(spec.order, spec.ascending)
                       if c in sub.columns]
                if order:
                    sub = sub.sort_values(order, ascending=asc)
                head = sub.head(n_max)
                for r, (_, row) in enumerate(head.iterrows(), start=1):
                    rows.append({"config_id": res.config_id, "arm": arm,
                                 "district": district, "kind": kind, "rank": r,
                                 "n_eligible": int(len(sub)),
                                 **{c: row.get(c) for c in keep if c in row}})
    return pd.DataFrame(rows)


def coverage(res, arms=PIPELINE_ARMS, n_max: int = 5,
             table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (arm, district, kind): how full the set is, and whether it is ragged.

    `complete` false is not an error; it is the honest outcome for a district
    with no gauge meeting the criteria. It is recorded because the FL step has
    to know that its clients do not share a feature geometry.
    """
    table = classification(res) if table is None else table
    rows = []
    for arm in arms:
        d = eligible(table, arm)
        for district in res.names:
            for kind in ("pressure", "flow"):
                sub = d[(d["home"] == district) & (d["kind"] == kind)]
                n = min(len(sub), n_max)
                rows.append({"config_id": res.config_id, "arm": arm,
                             "district": district, "kind": kind,
                             "n_selected": int(n), "n_eligible": int(len(sub)),
                             "n_max": int(n_max), "complete": bool(n == n_max),
                             "empty": bool(n == 0)})
    out = pd.DataFrame(rows)
    return out


def sensors_yaml(sel: dict) -> str:
    """The selection in the shape a bundle's `profile.yml` wants."""
    return sp.sensors_yaml(sel)


# ==========================================================================
# emit -- what the federated step reads
# ==========================================================================
def emit(res, out_root, arms=PIPELINE_ARMS, n_max: int = 5,
         extra_arms=("unusable",), verbose: bool = True) -> pathlib.Path:
    """Write the arms, the classification and the handoff manifest.

    Layout, under `<out_root>/<config_id>/`:

        districts.yml                    (already written by district_sweep)
        classification.parquet           every node and link, tiered
        placements/<arm>.yml             a `sensors:` block per arm
        placements/selection.parquet     the audit trail
        placements/coverage.csv          completeness per (arm, district, kind)
        handoff.json                     the tuple the FL step consumes
    """
    out = pathlib.Path(out_root) / res.config_id
    pl = out / "placements"
    pl.mkdir(parents=True, exist_ok=True)
    table = classification(res)
    if not len(table):
        raise ValueError(f"{res.config_id}: no mixture to select from "
                         f"(status={res.status!r}: {res.error})")
    table.to_parquet(out / "classification.parquet", index=False)

    all_arms = tuple(arms) + tuple(a for a in extra_arms if a not in arms)
    sels = {}
    for arm in all_arms:
        sel = select(res, arm, n_max=n_max, table=table)
        sels[arm] = sel
        (pl / f"{arm}.yml").write_text(
            f"# {res.config_id} -- arm {arm}: {ARMS[arm].doc}\n"
            f"# <= {n_max} per district per kind; short sets are SHIPPED RAGGED.\n"
            + sensors_yaml(sel) + "\n")

    sel_tab = selection_table(res, arms=all_arms, n_max=n_max, table=table)
    cov = coverage(res, arms=all_arms, n_max=n_max, table=table)
    if len(sel_tab):
        sel_tab.to_parquet(pl / "selection.parquet", index=False)
    cov.to_csv(pl / "coverage.csv", index=False)

    (out / "handoff.json").write_text(json.dumps({
        "config_id": res.config_id, "method": res.method,
        "network": res.network, "districts_file": "districts.yml",
        "districts": res.names, "n_max_per_kind": int(n_max),
        "pipeline_arms": list(arms),
        "report_only_arms": [a for a in all_arms if a not in arms],
        "placements": {a: f"placements/{a}.yml" for a in all_arms},
        "ragged": sorted({f"{r.arm}/{r.district}/{r.kind}"
                          for r in cov.itertuples() if not r.complete}),
    }, indent=2))

    if verbose:
        for arm in all_arms:
            c = cov[cov["arm"] == arm]
            print(f"  {arm:13s} {int(c['n_selected'].sum()):4d} gauges | "
                  f"{int((~c['complete']).sum())}/{len(c)} sets short | "
                  f"{int(c['empty'].sum())} empty")
    return out
