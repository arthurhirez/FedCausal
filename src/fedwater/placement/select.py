"""select.py -- the shipped placement: equal slots per district, chosen on the exact layer.

The federated side needs every client to have the SAME feature geometry, so the placement is a set of
slots, identical for every district::

    sensor_placement.slots:
      flow:     {branch: 2, transit: 2}   ->  q_branch_0..1, q_transit_0..1
      pressure: {branch: 0, transit: 0}   ->  (no pressure slots; see "Pressure" below)

Everything is read from the stack's exact layer (:mod:`fedwater.placement.exact`): topology, the
no-drift composition of every link's water (E1), its change under each probe drift (dm_k), the paired
responses (S, Nr) and the storage share (E2 sigma). Candidates are the links of the client's district
that are not a pump, not a valve, not dead, and carry mean |q| >= ``flow_floor`` x the network median.

The flow rule, per client h
---------------------------
branch slots   branches (a bridge with a source on one side only: the flow is exactly the demand
               downstream), ranked by the demand they carry; a branch whose downstream node set overlaps
               a chosen one by Jaccard > ``j_max`` is skipped.
transit slots  loop / source_bridge links, in three tiers:
  1. pool      some foreign district k changes the link's water mix, |dm_k| >= ``dm_min``, with
               dm_k = m_k(probe world k) - m_k(no-drift city) on k's settled window (paired, exact).
               Picked in order of the largest |dm_k|; every pick at least ``d_min`` from the chosen
               transits on the signature (composition + dm vector) -- the distance is a gate, not the
               objective; ``new_partner`` prefers a pick credited to a district no chosen transit has.
               A link carrying only its own water in both worlds has dm = 0 and never enters; a link
               that carries k's water only once k drifts (rerouting) does.
  2. mixed     short pool: a link that carries a foreign district's water in the no-drift city (largest
               foreign share >= ``m_carry``; the origin is exact by E1). Responders attributable to one
               district first, then an uncovered partner, then the larger foreign share; ``d_min`` applies.
  3. branch    otherwise a further branch (same Jaccard gate). A client whose loops carry no foreign
               water ends with only branches: there is nothing to choose.
A client that cannot fill every slot raises :class:`InsufficientCandidates` (the geometry must be equal).

What the rule optimises -- mixture change, not response
-------------------------------------------------------
The dependence LABEL is the response (S in shape space, Nr in native space: how much client j's signal
moves when district k drifts -- the interventional notion). It counts two channels: demand transit (k's
water passes through the gauge) and the hub (pumps / valves / tanks react to k's drift and move a gauge
that carries none of k's water). The selection criterion is narrower on purpose: dm_k, dependence through
water actually carried. It guarantees a transit gauge carries a neighbour's water and moves with it. The
two rankings differ where the hub dominates (D-Town: the strongest foreign responders are hub-mediated),
so a chosen transit is the largest MIXTURE change in its client, not necessarily the largest response.
The response labels of every chosen sensor are still written (``S_<k>``, ``Nr_<k>``, ``sigma_<k>``).

Pressure
--------
No pressure rule exists yet: branch / transit are LINK classes. Pressure needs its own node topology
(does a node sit in a tree subtree or in the looped core), so ``slots.pressure`` must be 0 for a dynamic
placement; a non-zero count raises.

Rules
-----
``sensor_placement.rule`` picks the rule: ``exact`` (this module, the default) or ``legacy`` (the
pre-exact pure / mixed ladder on the stack's own estimator, :mod:`fedwater.placement.select_legacy`,
kept to reproduce earlier worlds and compare). The slot classes in ``slots`` must be the rule's own:
branch / transit for ``exact``, pure / mixed for ``legacy``.

Labels
------
Selection reads the selection stack (reference coupling), so every coupling arm of a study gets the same
sensors. The label columns come from the LABEL stack's exact layer (the world's realised coupling); when
the world runs at the reference coupling the two are the same stack.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import signal_probe as sp

__all__ = ["RULES", "rule_of", "SLOT_CLASSES", "FILLS", "RULE_DEFAULTS", "TAXONOMY_DEFAULTS", "slot_name",
           "prep",
           "taxonomy", "select_flow", "select_slots", "placement_frame", "manual_placement",
           "sensors_block", "signature_distance", "InsufficientCandidates"]

SLOT_CLASSES = ("branch", "transit")
RULES = {"exact": SLOT_CLASSES, "legacy": ("pure", "mixed")}      # rule -> its slot classes
FILLS = ("branch", "pool", "mixed", "branch_fill")
TOPO_TRANSIT = ("loop", "source_bridge")
_KIND_PREFIX = sp._PREFIX                     # {"pressure": "p_", "flow": "q_"}

RULE_DEFAULTS = {"flow_floor": 0.05, "j_max": 0.2, "dm_min": 0.005, "new_partner": True,
                 "d_min": 0.1, "dist": "js", "m_carry": 0.05, "rho": 1.5, "attr": "S"}
TAXONOMY_DEFAULTS = {"S_floor": 0.25, "Nr_floor": 0.05, "ch_lo": 1 / 3, "ch_hi": 2 / 3,
                     "m_carry": 0.05, "m_trace": 0.01}
CLASSES = ("own", "none", "undefined", "transit_demand", "transit_hub", "trace",
           "reroute_demand", "hub_only")


class InsufficientCandidates(ValueError):
    """A district cannot fill its slots."""


def rule_of(sensor_placement: dict) -> str:
    """``sensor_placement.rule`` (default ``exact``), checked."""
    r = (sensor_placement or {}).get("rule") or "exact"
    if r not in RULES:
        raise ValueError(f"sensor_placement.rule must be one of {sorted(RULES)}, got {r!r}")
    return r


def slot_name(kind: str, slot_class: str, i: int) -> str:
    return f"{_KIND_PREFIX[kind]}{slot_class}_{i}"


def _cfg(rule: dict | None, tax: dict | None) -> tuple[dict, dict]:
    r = {**RULE_DEFAULTS, **(rule or {})}
    t = {**TAXONOMY_DEFAULTS, **(tax or {})}
    if r["dist"] not in ("js", "cosine"):
        raise ValueError(f"sensor_placement.selection.dist must be js or cosine, got {r['dist']!r}")
    if r["attr"] not in ("S", "Nr"):
        raise ValueError(f"sensor_placement.selection.attr must be S or Nr, got {r['attr']!r}")
    return r, t


# ==========================================================================
# the exact layer, as link x district matrices
# ==========================================================================
def taxonomy(labels: pd.DataFrame, tax: dict | None = None, attr: str = "S") -> np.ndarray:
    """Channel class of every (link, drifting district) cell (as ``epanet_recovery``; ASCII labels,
    since they are written to ``sensor_placement.csv`` and read back by Kedro as UTF-8):
    own; none (attr below its floor); transit_demand (carries k's water, sigma < ch_lo); transit_hub
    (carries k's water); trace (m_trace <= share < m_carry); reroute_demand (no k water, sigma < ch_lo);
    hub_only; undefined (sigma or share missing). Carrier share: raw share of k's demand in the link's
    water, drift world."""
    _, t = _cfg(None, tax)
    floor = t["S_floor"] if attr == "S" else t["Nr_floor"]
    resp = (labels[attr] >= floor).to_numpy()
    sh = labels["sigma"].to_numpy(float) if "sigma" in labels.columns else np.full(len(labels), np.nan)
    ch = np.select([sh < t["ch_lo"], sh > t["ch_hi"], np.isfinite(sh)], ["demand", "storage", "both"],
                   "undefined")
    mk = labels["c_k_drift"].to_numpy(float)
    carry = np.select([mk >= t["m_carry"], mk >= t["m_trace"], np.isfinite(mk)],
                      ["carrier", "trace", "non-carrier"], "unknown")
    return np.select([labels["own"].to_numpy(bool), ~resp, (ch == "undefined") | (carry == "unknown"),
                      (carry == "carrier") & (ch == "demand"), carry == "carrier", carry == "trace",
                      ch == "demand"],
                     ["own", "none", "undefined", "transit_demand", "transit_hub", "trace",
                      "reroute_demand"], "hub_only")


def prep(X: dict, tax: dict | None = None) -> dict:
    """Everything the rule reads from one stack's exact layer (``exact.load_exact`` output)."""
    names = list(X["manifest"]["districts"])
    L = X["links"].copy()
    L["element"] = L["element"].astype(str)
    L = L.set_index("element")
    B = X["labels"].copy()
    B["element"] = B["element"].astype(str)
    piv = lambda c: B.pivot(index="element", columns="k", values=c).reindex(index=L.index, columns=names)
    comp = np.nan_to_num(L[[f"c_{d}" for d in names] + ["c_storage", "c_reservoir"]].to_numpy(float))
    s = comp.sum(axis=1, keepdims=True)
    comp = comp / np.where(s > 0, s, 1.0)
    D = X["downstream"].copy()
    D["element"] = D["element"].astype(str)
    down = D.groupby("element")["node"].apply(lambda x: frozenset(map(str, x))).to_dict()
    has_sig = "sigma" in B.columns
    return {"names": names, "L": L, "B": B, "down": down,
            "DM": piv("dm_k"), "S": piv("S"), "Nr": piv("Nr"), "GW": piv("g_w"), "AMP": piv("amp"),
            "SIG": piv("sigma") if has_sig else piv("S") * np.nan, "CK": piv("c_k_drift"),
            "M0": L[[f"m_{d}" for d in names]].set_axis(names, axis=1),
            "COMP": pd.DataFrame(comp, index=L.index),
            "CLS": {a: B.assign(cls=taxonomy(B, tax, a)).pivot(index="element", columns="k", values="cls")
                    .reindex(index=L.index, columns=names) for a in ("S", "Nr")}}


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def signature_distance(PS: dict, a: str, b: str, how: str = "js") -> float:
    """Distance in [0, 1] between two links' signatures: composition (districts, filling tanks,
    reservoir; no-drift city) and the dm vector over every drifting district.
    js     = 1/2 JS(composition) + 1/2 (1 - cos(dm_a, dm_b)) / 2
    cosine = (1 - cos) / 2 of [composition, dm / ||dm||]"""
    from scipy.spatial.distance import jensenshannon
    ca, cb = PS["COMP"].loc[a].to_numpy(), PS["COMP"].loc[b].to_numpy()
    ra = np.nan_to_num(PS["DM"].loc[a].to_numpy(float))
    rb = np.nan_to_num(PS["DM"].loc[b].to_numpy(float))
    na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
    cos_r = float(ra @ rb / (na * nb)) if na > 0 and nb > 0 else 0.0
    if how == "js":
        js = float(jensenshannon(ca, cb, base=2)) if ca.sum() > 0 and cb.sum() > 0 else 1.0
        return 0.5 * js + 0.5 * (1.0 - cos_r) / 2.0
    sa = np.concatenate([ca, ra / na if na > 0 else ra])
    sb = np.concatenate([cb, rb / nb if nb > 0 else rb])
    den = np.linalg.norm(sa) * np.linalg.norm(sb)
    return (1.0 - float(sa @ sb / den)) / 2.0 if den > 0 else 1.0


def _attribution(PS: dict, r: dict, t: dict, e: str, foreign: list):
    """The foreign district k* a link's response can be credited to, or None: responses
    R = {k: A_k >= floor} non-empty, not every foreign district, the largest >= rho x the second, and
    the link carries k*'s water (no-drift or drift world share >= m_trace)."""
    floor = t["S_floor"] if r["attr"] == "S" else t["Nr_floor"]
    a = PS[r["attr"]].loc[e, foreign].astype(float).fillna(0.0).sort_values(ascending=False)
    R = a[a >= floor]
    if not len(R) or len(R) == len(foreign):
        return None
    if a.iloc[0] < r["rho"] * (float(a.iloc[1]) if len(a) > 1 else 0.0):
        return None
    k = a.index[0]
    carried = max(np.nan_to_num(PS["M0"].at[e, k]), np.nan_to_num(PS["CK"].at[e, k]))
    return k if carried >= t["m_trace"] else None


def candidates(PS: dict, rule: dict | None = None) -> pd.DataFrame:
    """The live links: home district known, not a pump / valve / dead, mean |q| above the flow floor."""
    r, _ = _cfg(rule, None)
    L = PS["L"]
    ok = ~L["dead"].astype(bool) & ~L["is_pump"].astype(bool) & ~L["is_valve"].astype(bool) \
        & L["home"].notna()
    floor = r["flow_floor"] * float(np.median(L.loc[~L["dead"].astype(bool), "mean_abs_flow"]))
    return L[ok & (L["mean_abs_flow"] >= floor)]


# ==========================================================================
# the rule
# ==========================================================================
def select_flow(PS: dict, rule: dict | None = None, tax: dict | None = None,
                n_branch: int = 2, n_transit: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flow slots of every client. Returns (picks: one row per filled slot, log: one row per client).
    Slots a client cannot fill are left out of ``picks`` and counted in ``log`` (short_*)."""
    r, t = _cfg(rule, tax)
    names, down = PS["names"], PS["down"]
    live = candidates(PS, r)
    rows, logs = [], []
    for h in names:
        foreign = [k for k in names if k != h]
        mine = live[live["home"] == h]
        br = mine[mine["topo"] == "branch"].sort_values(["down_demand", "mean_abs_flow"], ascending=False)
        Te = list(mine.index[mine["topo"].isin(TOPO_TRANSIT)])
        if Te:
            val = PS["DM"].loc[Te, foreign].abs()
            val = val.where(val >= r["dm_min"])
            score = val.max(axis=1)
            credit = val.fillna(-np.inf).idxmax(axis=1).where(score.notna())
            pool = score.dropna().sort_values(ascending=False, kind="stable")
            m0f = PS["M0"].loc[Te, foreign]
            fshare, fpartner = m0f.max(axis=1), m0f.fillna(-1.0).idxmax(axis=1)
            mixed = fshare[fshare >= r["m_carry"]]
        else:
            pool, credit, fshare, mixed = (pd.Series(dtype=float) for _ in range(4))
        picks = []                                           # (slot class, element, how, credit)

        def branch_ok(e, chosen):
            return all(jaccard(down.get(e, frozenset()), down.get(c, frozenset())) <= r["j_max"]
                       for c in chosen)

        def far(e, chosen):
            return not chosen or min(signature_distance(PS, e, c, r["dist"]) for c in chosen) >= r["d_min"]

        chosen_b = []
        for e in br.index:
            if len(chosen_b) == n_branch:
                break
            if branch_ok(e, chosen_b):
                chosen_b.append(e)
        picks += [("branch", e, "branch", None) for e in chosen_b]
        # 1. pool
        chosen_t, partners = [], set()
        while len(chosen_t) < n_transit:
            ok_ = [e for e in pool.index if e not in chosen_t and far(e, chosen_t)]
            if not ok_:
                break
            if r["new_partner"]:
                ok_ = [e for e in ok_ if credit[e] not in partners] or ok_
            e = ok_[0]
            chosen_t.append(e)
            partners.add(credit[e])
            picks.append(("transit", e, "pool", credit[e]))
        n_pool = len(chosen_t)
        # 2. mixed by composition
        covered = set(partners)
        while len(chosen_t) < n_transit:
            cands = []
            for e in mixed.index:
                if e in chosen_t or not far(e, chosen_t):
                    continue
                k = _attribution(PS, r, t, e, foreign)
                cr = k if k is not None else fpartner[e]
                cands.append((k is None, cr in covered, -float(mixed[e]), e, cr))
            if not cands:
                break
            *_, e, cr = min(cands)
            chosen_t.append(e)
            covered.add(cr)
            picks.append(("transit", e, "mixed", cr))
        n_mixed = len(chosen_t) - n_pool
        # 3. a further branch
        n_fill = 0
        for e in br.index:
            if len(chosen_t) >= n_transit:
                break
            if e in chosen_b or e in chosen_t:
                continue
            if branch_ok(e, chosen_b + [x for x in chosen_t if x in down]):
                chosen_t.append(e)
                n_fill += 1
                picks.append(("transit", e, "branch_fill", None))
        seen = {c: 0 for c in SLOT_CLASSES}
        L = PS["L"]
        for cls, e, how, cr in picks:
            row = L.loc[e]
            rows.append({"district": h, "kind": "flow", "slot": slot_name("flow", cls, seen[cls]),
                         "slot_class": cls, "filled_as": how, "rank": seen[cls],
                         "sensor": _KIND_PREFIX["flow"] + e, "element": e, "role": row["role"],
                         "topo": row["topo"], "credit": cr,
                         "sel_dm_credit": PS["DM"].at[e, cr] if cr else np.nan,
                         "sel_m_credit": PS["M0"].at[e, cr] if cr else np.nan,
                         "sel_m_home": row["m_home"], "sel_n_eff": row["n_eff"],
                         "sel_c_storage": row["c_storage"], "sel_down_demand": row["down_demand"],
                         "sel_mean_abs_flow": row["mean_abs_flow"]})
            seen[cls] += 1
        logs.append({"district": h, "kind": "flow", "n_candidates": len(mine), "n_branch_pool": len(br),
                     "n_transit_cand": len(Te), "n_transit_pool": len(pool), "n_mixed_cand": len(mixed),
                     "n_branch": len(chosen_b), "n_pool": n_pool, "n_mixed": n_mixed, "n_branch_fill": n_fill,
                     "short_branch": n_branch - len(chosen_b), "short_transit": n_transit - len(chosen_t),
                     "transit_distance": signature_distance(PS, chosen_t[0], chosen_t[1], r["dist"])
                     if len(chosen_t) >= 2 else np.nan,
                     "branch_jaccard": max((jaccard(down[a], down[b]) for a, b in itertools.combinations(
                         [e for e in chosen_b + chosen_t if e in down], 2)), default=np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(logs)


def select_slots(PS: dict, districts, slots: dict, rule: dict | None = None,
                 tax: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill every district's slots. Returns ``(selection, coverage)``: one row per slot, and one row per
    (district, kind) saying how they were filled. Raises when a district cannot fill its slots."""
    p = slots.get("pressure") or {}
    if int(p.get("branch", 0)) + int(p.get("transit", 0)) > 0:
        raise NotImplementedError(
            "sensor_placement.slots.pressure must be 0: branch / transit are link classes, and pressure "
            "needs its own node topology (tree subtree vs looped core) before a rule can be defined.")
    f = slots.get("flow") or {}
    nb, nt = int(f.get("branch", 0)), int(f.get("transit", 0))
    if nb + nt == 0:
        raise ValueError("sensor_placement.slots.flow is empty: no client would observe anything")
    picks, log = select_flow(PS, rule, tax, n_branch=nb, n_transit=nt)
    picks = picks[picks["district"].isin(list(districts))]
    short = log[(log["short_branch"] > 0) | (log["short_transit"] > 0)]
    if len(short):
        raise InsufficientCandidates(
            "cannot fill every flow slot for " + ", ".join(
                f"{r.district} (branch short {r.short_branch}, transit short {r.short_transit}; "
                f"{r.n_branch_pool} branches, {r.n_transit_cand} loop/source_bridge candidates)"
                for r in short.itertuples())
            + ". Lower sensor_placement.slots.flow, relax selection.j_max / d_min, or use a partition "
              "without such a district.")
    return picks.reset_index(drop=True), log


# ==========================================================================
# the downstream contract
# ==========================================================================
def placement_frame(selection: pd.DataFrame, PS_label: dict, districts, meta: dict) -> pd.DataFrame:
    """``sensor_placement.csv``: the selection plus, for every drifting district k, the chosen link's
    exact labels on the LABEL stack: ``S_<k>``, ``Nr_<k>`` (response), ``dm_<k>`` (mixture change),
    ``sigma_<k>`` (storage share), ``cls_<k>`` (channel class, by S) and ``m_<k>`` (no-drift
    composition, district-renormalised), plus ``c_storage`` (share of its water filling tanks)."""
    names = list(districts)
    el_ = selection["element"].astype(str).tolist()
    lab = {}
    for k in names:
        for col, M in (("S", PS_label["S"]), ("Nr", PS_label["Nr"]), ("dm", PS_label["DM"]),
                       ("sigma", PS_label["SIG"]), ("cls", PS_label["CLS"]["S"])):
            lab[f"{col}_{k}"] = M.reindex(el_)[k].to_numpy()
    for k in names:
        lab[f"m_{k}"] = PS_label["M0"].reindex(el_)[k].to_numpy()
    lab["c_storage"] = PS_label["L"]["c_storage"].reindex(el_).to_numpy()
    out = pd.concat([selection.reset_index(drop=True), pd.DataFrame(lab)], axis=1)
    for k, v in meta.items():
        out[k] = v
    return out


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
                             "role": None, **meta})
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
