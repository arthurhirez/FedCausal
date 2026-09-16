"""select.py -- the shipped placement: equal slots per district, filled by class.

The federated side needs every client to have the SAME feature geometry, so
the placement is a set of slots, identical for every district::

    sensor_placement.slots:
      flow:     {pure: 3, mixed: 2}      ->  q_pure_0..2, q_mixed_0..1
      pressure: {pure: 3, mixed: 2}      ->  p_pure_0..2, p_mixed_0..1

Classes (read off the SELECTION stack's mixture)
------------------------------------------------
pure   tier ``core``: purity >= ``core_purity`` -- reads almost only its own
       district.
mixed  largest foreign share (``w_second``) >= ``mixed_min_foreign``, at least
       ``min_n_mass`` districts carrying mass, and ``purity_swing`` <=
       ``max_purity_swing`` -- a blend that holds across null thresholds. No
       home-share floor and no top-two concentration gate.

The two sets are disjoint whenever ``mixed_min_foreign > 1 - core_purity``.

The fill ladder, per district and kind
--------------------------------------
1. pure slots   <- ``pure``: core gauges, best first (purity, then response
                   SNR), skipping near-duplicates (links in series carry the
                   same flow; distance measured on the response profile).
2. mixed slots  <- ``mixed``: eligible blends taken ROUND-ROBIN over the
                   partner district (``second``), skipping duplicate mixtures
                   (total variation < ``dedupe_min_distance``).
3. pure slots   <- ``pure_redundant``: the near-duplicates step 1 skipped.
4. mixed slots  <- ``mixed_redundant``: the duplicate mixtures step 2 skipped.
5. mixed slots  <- ``pure_fill``: unused core gauges.
6. any slot     <- ``best_available``: the best remaining responsive gauge
                   (by purity for a pure slot, by foreign share for a mixed
                   one). KY7 pressure, whose channel carries one reading, lands
                   here.
7. any slot     <- ``unresponsive``: a gauge whose signature moved for no
                   probe (tier ``unusable``). Only so that pressure is always
                   collected; flagged, and never chosen while anything else
                   remains.

A district fails the world only when it has fewer non-degenerate candidates of
a kind than slots. Gauges whose init series is constant (``degenerate_null``)
are never used: their SNR is not a measurement and their MinMax scale is zero.

Degenerate CHANNELS (every gauge reports the same blend) are kept and flagged
in the placement (``channel_verdict``); which channels feed the model is a
run-level choice (``fl.preprocessing.channels``).

Labels
------
Selection reads the selection stack (reference coupling), so the same sensors
are chosen in every coupling arm. The mixture columns written for each chosen
sensor come from the LABEL stack (the world's own coupling), because they are
the ground truth for the physics that world actually has. When the world runs
at the reference coupling the two stacks are the same object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import classification as cl
from . import signal_probe as sp

__all__ = ["CLASSES", "FILLS", "slot_name", "select_slots",
           "placement_frame", "manual_placement", "sensors_block",
           "InsufficientCandidates"]

CLASSES = ("pure", "mixed")
FILLS = ("pure", "mixed", "pure_redundant", "mixed_redundant", "pure_fill",
         "best_available", "unresponsive")
_KIND_PREFIX = sp._PREFIX                     # {"pressure": "p_", "flow": "q_"}

# label / selection columns carried per sensor
_LABEL_COLS = ("tier", "purity", "second", "w_second", "n_live", "n_mass",
               "mass", "hhi", "entropy", "neg_mass", "snr_home",
               "degenerate_null", "purity_swing", "tier_stable")
_SEL_COLS = ("tier", "purity", "second", "w_second", "n_mass", "snr_home",
             "purity_swing", "served_own", "r_res_own_init", "response_mean")


class InsufficientCandidates(ValueError):
    """A district has fewer usable gauges of a kind than it has slots."""


def slot_name(kind: str, slot_class: str, i: int) -> str:
    return f"{_KIND_PREFIX[kind]}{slot_class}_{i}"


# ==========================================================================
# ranking
# ==========================================================================
def _rank(d: pd.DataFrame, by, ascending) -> pd.DataFrame:
    by = [c for c in by if c in d.columns]
    asc = [a for c, a in zip(by, ascending) if c in d.columns]
    return d.sort_values(by + ["id"], ascending=asc + [True]) if by else d


def _round_robin(d: pd.DataFrame) -> pd.DataFrame:
    """Interleave partner districts: best of each, then second best, ..."""
    if not len(d):
        return d
    groups = [g for _, g in d.groupby("second", sort=False)]
    groups.sort(key=lambda g: list(d.index).index(g.index[0]))
    order, i = [], 0
    while any(i < len(g) for g in groups):
        order += [g.index[i] for g in groups if i < len(g)]
        i += 1
    return d.loc[order]


def _take(d: pd.DataFrame, used: set, n: int) -> list[str]:
    ids = [i for i in d["id"] if i not in used][:max(n, 0)]
    used.update(ids)
    return ids


def _dedupe_take(d: pd.DataFrame, used: set, n: int, min_distance: float,
                 on: str) -> list[str]:
    d = d[~d["id"].isin(used)]
    ids = list(cl.dedupe(d, n, min_distance, on=on)["id"]) if n > 0 else []
    used.update(ids)
    return ids


# ==========================================================================
# one district, one kind
# ==========================================================================
def _fill(pool: pd.DataFrame, n_pure: int, n_mixed: int, classes: dict,
          min_distance: float) -> tuple[list, dict]:
    """Returns ``[(slot_class, id, filled_as), ...]`` and the eligible counts."""
    live = pool[pool["tier"] != "unusable"]
    pure = _rank(live[live["tier"] == "core"],
                 ("purity", "snr_home"), (False, False))
    mixed = live[(live["w_second"].fillna(0.0) >= float(classes["mixed_min_foreign"]))
                 & (live["n_mass"].fillna(0) >= int(classes["min_n_mass"]))]
    if "purity_swing" in mixed.columns:
        mixed = mixed[mixed["purity_swing"].fillna(1.0)
                      <= float(classes["max_purity_swing"])]
    mixed = _round_robin(_rank(mixed, ("purity_swing", "snr_home"),
                               (True, False)))

    used: set = set()
    got = {"pure": [], "mixed": []}

    def add(cls, ids, how):
        got[cls] += [(i, how) for i in ids]

    add("pure", _dedupe_take(pure, used, n_pure, min_distance, "profile"), "pure")
    add("mixed", _dedupe_take(mixed, used, n_mixed, min_distance, "mixture"),
        "mixed")
    add("pure", _take(pure, used, n_pure - len(got["pure"])), "pure_redundant")
    add("mixed", _take(mixed, used, n_mixed - len(got["mixed"])),
        "mixed_redundant")
    add("mixed", _take(pure, used, n_mixed - len(got["mixed"])), "pure_fill")

    by_purity = _rank(live, ("purity", "snr_home"), (False, False))
    by_foreign = _rank(live, ("w_second", "snr_home"), (False, False))
    add("pure", _take(by_purity, used, n_pure - len(got["pure"])),
        "best_available")
    add("mixed", _take(by_foreign, used, n_mixed - len(got["mixed"])),
        "best_available")

    dead = _rank(pool[pool["tier"] == "unusable"], ("snr_home",), (False,))
    add("pure", _take(dead, used, n_pure - len(got["pure"])), "unresponsive")
    add("mixed", _take(dead, used, n_mixed - len(got["mixed"])), "unresponsive")

    rows = [(c, i, how) for c in CLASSES for i, how in got[c]]
    counts = {"n_candidates": int(len(pool)), "n_live": int(len(live)),
              "n_eligible_pure": int(len(pure)),
              "n_eligible_mixed": int(len(mixed))}
    return rows, counts


def select_slots(table: pd.DataFrame, districts, slots: dict, classes: dict,
                 min_distance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill every district's slots from a ``classification`` table.

    Returns ``(selection, coverage)``: one row per filled slot, and one row per
    (district, kind, slot_class) saying how the slots were filled.
    """
    rows, cov = [], []
    base = table.copy()
    if "degenerate_null" in base.columns:
        base = base[~base["degenerate_null"].fillna(False).astype(bool)]
    for district in districts:
        for kind, spec in slots.items():
            n_pure, n_mixed = int(spec.get("pure", 0)), int(spec.get("mixed", 0))
            if n_pure + n_mixed == 0:
                continue
            pool = base[(base["home"] == district) & (base["kind"] == kind)]
            if pool["id"].nunique() < n_pure + n_mixed:
                raise InsufficientCandidates(
                    f"{district} has {pool['id'].nunique()} usable {kind} "
                    f"gauge(s) for {n_pure + n_mixed} slots "
                    f"(sensor_placement.slots.{kind}). Lower the slot count, "
                    "or use a partition without such a small district.")
            picks, counts = _fill(pool, n_pure, n_mixed, classes, min_distance)
            if len(picks) < n_pure + n_mixed:      # defensive: ladder is total
                raise InsufficientCandidates(
                    f"{district}/{kind}: filled {len(picks)} of "
                    f"{n_pure + n_mixed} slots")
            idx = pool.set_index("id")
            seen = {c: 0 for c in CLASSES}
            for slot_class, gid, how in picks:
                r = idx.loc[gid]
                rows.append({
                    "district": district, "kind": kind,
                    "slot": slot_name(kind, slot_class, seen[slot_class]),
                    "slot_class": slot_class, "filled_as": how,
                    "rank": seen[slot_class], "sensor": gid,
                    "element": r["element"], "role": r.get("role"),
                    **{f"sel_{c}": r.get(c) for c in _SEL_COLS},
                })
                seen[slot_class] += 1
            for slot_class, n in (("pure", n_pure), ("mixed", n_mixed)):
                hows = [h for c, _, h in picks if c == slot_class]
                cov.append({"district": district, "kind": kind,
                            "slot_class": slot_class, "n_slots": n,
                            **counts,
                            "n_as_class": sum(h == slot_class for h in hows),
                            "n_filled": sum(h != slot_class for h in hows),
                            **{f"n_{f}": sum(h == f for h in hows)
                               for f in FILLS}})
    return pd.DataFrame(rows), pd.DataFrame(cov)


# ==========================================================================
# the downstream contract
# ==========================================================================
def placement_frame(selection: pd.DataFrame, label_table: pd.DataFrame,
                    districts, channels: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """``sensor_placement.csv``: selection + label mixture, one row per slot."""
    wcols = [f"w_{d}" for d in districts]
    lab = label_table.set_index("id")
    keep = [c for c in _LABEL_COLS if c in lab.columns]
    labels = lab.reindex(selection["sensor"])[keep + [c for c in wcols
                                                      if c in lab.columns]]
    labels = labels.reset_index(drop=True)
    for c in wcols:
        if c not in labels.columns:
            labels[c] = np.nan
    verdict = dict(zip(channels.get("kind", []), channels.get("verdict", [])))
    out = pd.concat([selection.reset_index(drop=True), labels], axis=1)
    out["channel_verdict"] = out["kind"].map(verdict).fillna("n/a")
    for k, v in meta.items():
        out[k] = v
    lead = ["district", "kind", "slot", "slot_class", "filled_as", "rank",
            "sensor", "element", "role"] + keep + wcols
    rest = [c for c in out.columns if c not in lead]
    return out[lead + rest]


def manual_placement(network_profile: dict, meta: dict) -> pd.DataFrame:
    """The profile's hand-placed ``sensors:`` block as a placement table.

    Slot == sensor id (``p_J-109``), so client CSVs keep their historical
    column names and a manual world reproduces its pre-refactor artifacts.
    """
    rows = []
    for district, cfg in (network_profile.get("sensors") or {}).items():
        for kind in ("pressure", "flow"):
            for r, el in enumerate(cfg.get(kind) or []):
                sid = f"{_KIND_PREFIX[kind]}{el}"
                rows.append({"district": district, "kind": kind, "slot": sid,
                             "slot_class": "manual", "filled_as": "manual",
                             "rank": r, "sensor": sid, "element": str(el),
                             "role": None, "channel_verdict": "n/a", **meta})
    if not rows:
        raise ValueError("sensor_placement.source is 'manual' but the bundle "
                         "profile has no `sensors:` block")
    return pd.DataFrame(rows)


def sensors_block(placement: pd.DataFrame) -> dict:
    """``{district: {"pressure": [...], "flow": [...]}}`` view of a placement,
    in slot order -- the shape the old ``profile.yml`` block had."""
    out = {}
    for district, g in placement.groupby("district", sort=False):
        out[district] = {k: [str(e) for e in g.loc[g["kind"] == k, "element"]]
                         for k in ("pressure", "flow")}
    return out
