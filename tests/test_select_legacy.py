"""The legacy placement rule (``sensor_placement.rule: legacy``, placement.select_legacy).

The pre-exact pure / mixed ladder on the probe stack's own classification table, restored as a selectable
rule so earlier worlds can be reproduced. These are its original tests (moved from test_placement.py),
unchanged except for the module they call, plus the dispatch checks.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fedwater.placement import select as sel
from fedwater.placement import select_legacy as leg


def _row(id_, home, kind="flow", tier="core", purity=1.0, second=None,
         w_second=0.0, n_mass=1, swing=0.0, snr=10.0, **kw):
    return {"id": id_, "element": id_[2:], "kind": kind, "home": home,
            "role": "internal", "tier": tier, "purity": purity,
            "second": second, "w_second": w_second, "n_mass": n_mass,
            "purity_swing": swing, "snr_home": snr, "served_own": 1.0,
            "response_mean": 0.5, "degenerate_null": False, **kw}


CLASSES = {"core_purity": 0.85, "mixed_min_foreign": 0.25, "min_n_mass": 2,
           "max_purity_swing": 0.20}


def test_ladder_prefers_class_then_diversity_then_redundancy():
    rows = [
        # pure: four core gauges; q_c2 duplicates q_c1's profile exactly
        _row("q_c1", "A", snr=9.0), _row("q_c2", "A", snr=9.0),
        _row("q_c3", "A", purity=0.95, snr=5.0),
        _row("q_c4", "A", purity=0.90, snr=1.0, served_own=0.1),
        # mixed: two with partner B (the second one is a near-copy), one C
        _row("q_m1", "A", tier="transition", purity=.6, second="B",
             w_second=.4, n_mass=2, snr=8),
        _row("q_m2", "A", tier="transition", purity=.6, second="B",
             w_second=.4, n_mass=2, snr=7),
        _row("q_m3", "A", tier="transition", purity=.5, second="C",
             w_second=.3, n_mass=2, snr=6),
        # not mixed: foreign share below 0.25 / swinging / one-district mass
        _row("q_x1", "A", tier="transition", purity=.8, second="B",
             w_second=.2, n_mass=2),
        _row("q_x2", "A", tier="transition", purity=.6, second="B",
             w_second=.4, n_mass=2, swing=.5),
        _row("q_x3", "A", tier="foreign", purity=0.0, second="B",
             w_second=1.0, n_mass=1),
    ]
    t = pd.DataFrame(rows)
    for c in ("w_A", "w_B", "w_C"):
        t[c] = 0.0
    t.loc[t["id"] == "q_m1", ["w_A", "w_B"]] = [.6, .4]
    t.loc[t["id"] == "q_m2", ["w_A", "w_B"]] = [.6, .4]
    t.loc[t["id"] == "q_m3", ["w_A", "w_C"]] = [.5, .3]
    t.loc[t["home"] == "A", "w_A"] = t.loc[t["home"] == "A", "purity"]
    picks, cov = leg.select_slots(t, ["A"], {"flow": {"pure": 3, "mixed": 3}},
                                  CLASSES, 0.10)
    got = dict(zip(picks["slot"], zip(picks["sensor"], picks["filled_as"])))
    # pure: best first, near-duplicate skipped in favour of the next reading
    assert got["q_pure_0"][0] == "q_c1" and got["q_pure_0"][1] == "pure"
    assert {got["q_pure_1"][0], got["q_pure_2"][0]} <= {"q_c3", "q_c4"}
    # mixed: partner round-robin (B then C), then the duplicate as redundant
    assert got["q_mixed_0"] == ("q_m1", "mixed")
    assert got["q_mixed_1"] == ("q_m3", "mixed")
    assert got["q_mixed_2"] == ("q_m2", "mixed_redundant")
    assert set(picks["sensor"]).isdisjoint({"q_x1", "q_x2", "q_x3"})
    c = cov.set_index("slot_class")
    assert c.loc["mixed", "n_eligible_mixed"] == 3
    assert c.loc["mixed", "n_mixed_redundant"] == 1


def test_ladder_fills_short_classes_and_keeps_geometry_equal():
    rows = []
    # District A: 2 core only (short on pure and on mixed)
    rows += [_row("q_a1", "A"), _row("q_a2", "A", purity=.95, snr=3)]
    rows += [_row("q_a3", "A", tier="foreign", purity=0.1, second="B",
                  w_second=.9, n_mass=1)]
    rows += [_row("q_a4", "A", tier="unusable", purity=0.0, n_mass=0, snr=.1)]
    rows += [_row("q_a5", "A", degenerate_null=True)]
    # District B: plenty
    rows += [_row(f"q_b{i}", "B", snr=10 - i) for i in range(3)]
    rows += [_row(f"q_bm{i}", "B", tier="transition", purity=.5, second="A",
                  w_second=.5, n_mass=2, snr=5 - i) for i in range(2)]
    t = pd.DataFrame(rows)
    t["w_A"] = 0.0
    t["w_B"] = 0.0
    t.loc[t["home"] == "A", "w_A"] = t.loc[t["home"] == "A", "purity"]
    t.loc[t["home"] == "B", "w_B"] = t.loc[t["home"] == "B", "purity"]
    t.loc[t["id"].str.startswith("q_bm"), "w_A"] = .5
    slots = {"flow": {"pure": 2, "mixed": 2}}
    picks, cov = leg.select_slots(t, ["A", "B"], slots, CLASSES, 0.10)
    assert picks.groupby("district").size().to_dict() == {"A": 4, "B": 4}
    a = picks[picks["district"] == "A"].set_index("slot")
    assert a.loc["q_pure_0", "filled_as"] == "pure"
    # A has no mixed and only two core: the ladder walks down to the end
    assert set(a.loc[["q_mixed_0", "q_mixed_1"], "filled_as"]) \
        == {"best_available", "unresponsive"}
    assert "q_a5" not in set(picks["sensor"])            # degenerate null
    assert list(picks["slot"][picks["district"] == "B"]) \
        == ["q_pure_0", "q_pure_1", "q_mixed_0", "q_mixed_1"]
    with pytest.raises(sel.InsufficientCandidates, match="District|A"):
        leg.select_slots(t, ["A"], {"flow": {"pure": 3, "mixed": 2}},
                         CLASSES, 0.10)


def test_placement_frame_takes_labels_from_the_label_stack():
    s_table = pd.DataFrame([_row("q_1", "A"), _row("p_1", "A", kind="pressure")])
    picks, _ = leg.select_slots(s_table, ["A"],
                                {"flow": {"pure": 1, "mixed": 0},
                                 "pressure": {"pure": 1, "mixed": 0}},
                                CLASSES, 0.10)
    l_table = pd.DataFrame([
        _row("q_1", "A", tier="transition", purity=.4, second="B",
             w_second=.6, n_mass=2),
        _row("p_1", "A")]).assign(w_A=[.4, 1.0], w_B=[.6, 0.0])
    channels = pd.DataFrame({"kind": ["flow", "pressure"],
                             "verdict": ["ok", "degenerate"]})
    out = leg.placement_frame(picks, l_table, ["A", "B"], channels,
                              {"network": "x", "label_stack": "h"})
    r = out.set_index("sensor")
    assert r.loc["q_1", "sel_tier"] == "core"            # how it was chosen
    assert r.loc["q_1", "tier"] == "transition"          # what it reads here
    assert r.loc["q_1", "w_B"] == pytest.approx(.6)
    assert r.loc["p_1", "channel_verdict"] == "degenerate"
    assert list(out.columns[:9]) == ["district", "kind", "slot", "slot_class",
                                     "filled_as", "rank", "sensor", "element",
                                     "role"]
    assert sel.sensors_block(out) == {"A": {"pressure": ["1"], "flow": ["1"]}}


def test_legacy_defaults_are_the_values_shipped_before_the_exact_rule():
    assert leg.DEFAULTS == {"mixed_min_foreign": 0.25, "min_n_mass": 2, "max_purity_swing": 0.20}
    assert {k: CLASSES[k] for k in leg.DEFAULTS} == leg.DEFAULTS


def test_rule_switch_and_slot_classes():
    assert sel.rule_of({}) == "exact" and sel.rule_of({"rule": "legacy"}) == "legacy"
    assert sel.RULES["legacy"] == leg.CLASSES and sel.RULES["exact"] == sel.SLOT_CLASSES
    with pytest.raises(ValueError, match="rule"):
        sel.rule_of({"rule": "greedy"})
    assert leg.InsufficientCandidates is sel.InsufficientCandidates
