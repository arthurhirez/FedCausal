"""The flow-selection rule (placement.select) on the exact layer of the toy probe stack.

The stack is real EPANET physics (the toy network of ``_toy_network``: one pump, one tank, a loop and two
branches; two districts, each drifting in its own probe world). Rule properties are checked against the
exact tables the rule reads; where a case the toy network does not contain is needed (a second branch in
the same district, a link that carries only its own water), the table is edited in the one cell that
creates the case and the edit is stated.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from fedwater.placement import exact as ex
from fedwater.placement import select as sel

from _toy_network import fake_stack

A, B = "District_A", "District_B"


@pytest.fixture(scope="module")
def ps(tmp_path_factory):
    path = fake_stack(tmp_path_factory.mktemp("stack"))
    X = ex.ensure_exact(path, verbose=False)
    assert X["gates"]["passed"].all()
    return sel.prep(X)


def _edit(ps):
    return copy.deepcopy(ps)


def test_candidates_exclude_pumps_valves_dead_and_low_flow(ps):
    p = _edit(ps)
    p["L"].loc["PU", "home"] = A                         # give the pump a home: still not a candidate
    p["L"].loc["P1", "dead"] = True
    c = sel.candidates(p)
    assert "PU" not in c.index and "P1" not in c.index
    hi = sel.candidates(p, {"flow_floor": 1.0})          # >= the network median only
    med = float(np.median(p["L"].loc[~p["L"]["dead"], "mean_abs_flow"]))
    assert (hi["mean_abs_flow"] >= med).all() and len(hi) < len(c)


def test_pool_is_ordered_by_mixture_change_and_gated_by_distance(ps):
    picks, log = sel.select_flow(ps, {"d_min": 0.0, "new_partner": False}, n_branch=1, n_transit=2)
    a = picks[(picks["district"] == A) & (picks["slot_class"] == "transit")]
    T = [e for e in sel.candidates(ps).index
         if ps["L"].at[e, "home"] == A and ps["L"].at[e, "topo"] in ("loop", "source_bridge")]
    order = ps["DM"].loc[T, B].abs().sort_values(ascending=False)
    assert list(a["element"]) == list(order.index[:2])    # largest |dm_B| first
    assert (a["filled_as"] == "pool").all() and (a["credit"] == B).all()
    # d_min: the second pick is the largest change at least d_min from the first
    first = order.index[0]
    d = {e: sel.signature_distance(ps, e, first) for e in order.index[1:]}
    d_min = float(np.median(list(d.values()))) + 1e-9
    picks2, _ = sel.select_flow(ps, {"d_min": d_min, "new_partner": False}, n_branch=1, n_transit=2)
    a2 = picks2[(picks2["district"] == A) & (picks2["slot_class"] == "transit")]
    want = next(e for e in order.index[1:] if d[e] >= d_min)
    assert list(a2["element"])[:2] == [first, want]


def test_dm_min_empties_the_pool_and_the_mixed_tier_takes_over(ps):
    picks, log = sel.select_flow(ps, {"dm_min": 1.0}, n_branch=1, n_transit=1)
    t = picks[picks["slot_class"] == "transit"].set_index("district")
    assert (t["filled_as"] == "mixed").all()
    for h, r in t.iterrows():                             # carries the credited district's water
        assert ps["M0"].at[r["element"], r["credit"]] >= sel.RULE_DEFAULTS["m_carry"]
    assert (log["n_transit_pool"] == 0).all()


def test_a_link_carrying_only_its_own_water_never_fills_a_transit_slot(ps):
    p = _edit(ps)
    T = ["P0", "P1", "P6"]                                # every transit candidate of District_A
    p["DM"].loc[T, :] = 0.0                               # no mixture change under any drift ...
    p["M0"].loc[T, B] = 0.0                               # ... and no foreign water in the no-drift city
    picks, log = sel.select_flow(p, {}, n_branch=1, n_transit=1)
    a = log.set_index("district").loc[A]
    assert a["n_pool"] == 0 and a["n_mixed"] == 0
    assert not set(picks.loc[picks["district"] == A, "element"]) & set(T)


def test_branches_are_ranked_by_downstream_demand_and_gated_by_overlap(ps):
    p = _edit(ps)
    p["L"].loc["P5", "home"] = A                          # a second branch in A, nested in P4 (Jaccard 1/2)
    assert sel.jaccard(p["down"]["P4"], p["down"]["P5"]) == pytest.approx(0.5)
    picks, log = sel.select_flow(p, {"j_max": 0.4}, n_branch=2, n_transit=1)
    a = log.set_index("district").loc[A]
    assert a["short_branch"] == 1
    assert list(picks.loc[(picks["district"] == A) & (picks["slot_class"] == "branch"), "element"]) == ["P4"]
    picks, _ = sel.select_flow(p, {"j_max": 0.6}, n_branch=2, n_transit=1)
    assert list(picks.loc[(picks["district"] == A) & (picks["slot_class"] == "branch"), "element"]) \
        == ["P4", "P5"]                                   # P4 carries more demand downstream


def test_a_further_branch_fills_a_transit_slot_when_nothing_carries_foreign_water(ps):
    p = _edit(ps)
    p["L"].loc["P5", "home"] = A
    picks, log = sel.select_flow(p, {"dm_min": 1.0, "m_carry": 1.1, "j_max": 0.6}, n_branch=1, n_transit=1)
    t = picks[(picks["district"] == A) & (picks["slot_class"] == "transit")]
    assert list(t["element"]) == ["P5"] and list(t["filled_as"]) == ["branch_fill"]


def test_slots_must_fill_and_pressure_must_be_zero(ps):
    names = ps["names"]
    sel_, cov = sel.select_slots(ps, names, {"flow": {"branch": 1, "transit": 1}})
    assert sel_.groupby("district").size().to_dict() == {A: 2, B: 2}
    assert list(sel_.loc[sel_["district"] == A, "slot"]) == ["q_branch_0", "q_transit_0"]
    with pytest.raises(sel.InsufficientCandidates, match="District_"):
        sel.select_slots(ps, names, {"flow": {"branch": 2, "transit": 2}})   # one branch per district
    with pytest.raises(NotImplementedError, match="pressure"):
        sel.select_slots(ps, names, {"flow": {"branch": 1, "transit": 1},
                                     "pressure": {"branch": 1, "transit": 0}})


def test_placement_frame_takes_labels_from_the_label_stack(ps):
    names = ps["names"]
    picks, _ = sel.select_slots(ps, names, {"flow": {"branch": 1, "transit": 1}})
    lab = _edit(ps)
    lab["S"] = lab["S"] * 2.0                             # a label stack whose physics differs
    out = sel.placement_frame(picks, lab, names, {"network": "toy", "label_stack": "h"})
    r = out.set_index("sensor")
    s = picks.set_index("sensor")
    for sid in s.index:
        e = s.at[sid, "element"]
        assert r.at[sid, f"S_{B}"] == pytest.approx(2.0 * ps["S"].at[e, B])
        assert r.at[sid, f"dm_{B}"] == pytest.approx(ps["DM"].at[e, B])
    assert {f"{c}_{k}" for c in ("S", "Nr", "dm", "sigma", "cls", "m") for k in names} <= set(out.columns)
    assert list(out.columns[:9]) == ["district", "kind", "slot", "slot_class", "filled_as", "rank",
                                     "sensor", "element", "role"]
    assert sel.sensors_block(out)[A]["flow"] == list(s.loc[s["district"] == A, "element"])
    # written to CSV and read back by Kedro as UTF-8 whatever the platform's encoding: ASCII only
    text = out.astype(str).to_csv(index=False)
    assert text.isascii()
    assert set(out[f"cls_{B}"].dropna()) <= set(sel.CLASSES)
