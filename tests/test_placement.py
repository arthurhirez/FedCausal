"""Tests for the dynamic-input refactor: partition store, probe stacks, slot
selection, and the downstream contract (sensing, oracle, FL feature filter).

House discipline as elsewhere: every claim is pinned against a case whose
answer is known by construction -- a synthetic classification table whose
ladder outcome is written down, a partition whose content id is recorded in
the POC's bundle name, a closure set chosen by hand.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fedwater.config import load_base_params, resolve_globals
from fedwater.networks import partitions as pstore
from fedwater.placement import excite
from fedwater.placement import select as sel
from fedwater.placement import store

ROOT = Path(__file__).resolve().parents[1]
BASE = load_base_params(ROOT)
NETWORK = "ky7"


def _bundle_files(network=NETWORK):
    root = ROOT / "data/01_raw" / network
    return {"profile": yaml.safe_load((root / "profile.yml").read_text()),
            "manual": yaml.safe_load((root / "districts.yml").read_text()),
            "inp_text": (root / "network.inp").read_text()}


@pytest.fixture(scope="module")
def wn():
    import wntr
    return wntr.network.WaterNetworkModel(
        str(ROOT / "data/01_raw" / NETWORK / "network.inp"))


# ==========================================================================
# config
# ==========================================================================
def test_globals_interpolation_is_whole_value_only():
    g = {"districting": {"methods": ["manual", "spectral"]},
         "probe_store": "/x"}
    tree = {"a": "${globals:districting.methods}", "b": ["${globals:probe_store}"],
            "c": "prefix ${globals:probe_store}", "d": "${globals:nope, 3}"}
    out = resolve_globals(tree, g)
    assert out == {"a": ["manual", "spectral"], "b": ["/x"],
                   "c": "prefix ${globals:probe_store}", "d": 3}
    with pytest.raises(KeyError, match="globals.yml"):
        resolve_globals({"x": "${globals:missing}"}, g)
    assert BASE["districting"]["methods"] == [
        "manual", "spectral", "girvan_newman", "fast_greedy", "walktrap"]


# ==========================================================================
# partition store
# ==========================================================================
def test_partition_id_is_content_and_matches_the_poc_bundle(built_partitions):
    from conftest import _write_partitions
    _write_partitions(NETWORK, ("fast_greedy",))
    part = pstore.read_partition(ROOT, NETWORK, "fast_greedy")
    # the POC named this bundle ky72__fast_greedy__2ab65bc6
    assert part["manifest"]["partition_id"] == "2ab65bc6"
    assert pstore.partition_id(part["districts"]) == "2ab65bc6"
    doc = copy.deepcopy(part["districts"])
    first = next(iter(doc["districts"]))
    doc["districts"][first] = list(reversed(doc["districts"][first]))
    assert pstore.partition_id(doc) == "2ab65bc6"      # node order: not content
    reordered = {"districts": dict(reversed(list(doc["districts"].items())))}
    assert pstore.partition_id(reordered) != "2ab65bc6"  # district order: is


def test_manual_partition_is_the_hand_authored_file_verbatim(built_partitions):
    root = ROOT / "data/01_raw" / NETWORK
    assert (root / "partitions/manual/districts.yml").read_text() \
        == (root / "districts.yml").read_text()
    man = pstore.read_manifest(ROOT, NETWORK, "manual")
    assert man["k"] == 4 and man["config"] == {"method": "manual"}


def test_freshness_tracks_config_inp_and_manual_file(built_partitions):
    dp = BASE["districting"]
    assert pstore.status(ROOT, NETWORK, "manual", dp) == "current"
    f = _bundle_files()
    k = pstore.resolve_k(dp, f["manual"])
    assert k == 4
    cfg = pstore.method_config(dp, "spectral", k)
    h = pstore.config_hash(NETWORK, cfg, f["inp_text"], None)
    assert pstore.config_hash(NETWORK, pstore.method_config(
        {**dp, "seed": dp["seed"] + 1}, "spectral", k),
        f["inp_text"], None) != h
    assert pstore.config_hash(NETWORK, cfg, f["inp_text"] + "\n;edit",
                              None) != h
    # CRLF and LF copies of one .inp are the same content
    assert pstore.config_hash(NETWORK, cfg, f["inp_text"].replace("\r\n", "\n"),
                              None) == pstore.config_hash(
        NETWORK, cfg, f["inp_text"].replace("\n", "\r\n"), None)
    # manual's identity is the file; the dials do not touch it
    m = pstore.method_config(dp, "manual", 4)
    assert pstore.method_config({**dp, "seed": 99}, "manual", 4) == m
    edited = copy.deepcopy(f["manual"])
    edited["districts"]["District_A"] = edited["districts"]["District_A"][:-1]
    assert pstore.config_hash(NETWORK, m, f["inp_text"], edited) \
        != pstore.config_hash(NETWORK, m, f["inp_text"], f["manual"])
    assert pstore.status(ROOT, NETWORK, "manual", {**dp, "seed": 99}) \
        == "current"
    assert pstore.status(ROOT, "graeme", "manual", dp) == "current"
    with pytest.raises(ValueError):
        pstore.check_method("kmeans")


def test_resolve_partition_rejects_an_edited_generated_file(built_partitions):
    from fedwater.pipelines.network_prep.nodes import resolve_partition
    part = pstore.read_partition(ROOT, NETWORK, "manual")
    profile = _bundle_files()["profile"]
    meta = resolve_partition(part["manifest"], profile, part["districts"])
    assert meta["method"] == "manual" and meta["name"] == NETWORK
    assert meta["drift_seed_nodes"] == {"District_A": "J-507"}
    tampered = copy.deepcopy(part["districts"])
    tampered["districts"]["District_A"].append(
        tampered["districts"]["District_B"].pop())
    with pytest.raises(ValueError, match="edited after it was built"):
        resolve_partition(part["manifest"], profile, tampered)
    generated = {**part["manifest"], "method": "fast_greedy"}
    assert resolve_partition(generated, profile,
                             part["districts"])["drift_seed_nodes"] == {}


def test_districting_gate_refuses_a_partition_with_the_wrong_k(wn):
    from fedwater.pipelines.districting.nodes import _gate
    manual = _bundle_files()["manual"]
    _gate(wn, "fast_greedy", manual, 4, 4)
    with pytest.raises(ValueError, match="requested"):
        _gate(wn, "fast_greedy", manual, 5, 4)
    _gate(wn, "manual", manual, 5, 4)                  # manual is adopted
    broken = {"districts": {**manual["districts"],
                            "District_UNASSIGNED": ["J-1"]}}
    with pytest.raises(ValueError, match="UNASSIGNED"):
        _gate(wn, "spectral", broken, 5, 5)


# ==========================================================================
# excitation
# ==========================================================================
def test_transitions_must_map_every_class_elsewhere():
    lu = BASE["land_use"]
    t = BASE["sensor_placement"]["probe"]["transitions"]
    assert excite.validate_transitions(t, lu) == t
    for bad in ({**t, "industrial": "industrial"},
                {k: v for k, v in t.items() if k != "mixed"},
                {**t, "farm": "industrial"},
                {**t, "commercial": "farm"}):
        with pytest.raises(ValueError, match="transitions"):
            excite.validate_transitions(bad, lu)
    assert excite.excitation(("medium", "industrial"), t) == {
        "to_income": "medium", "to_land_use": "residential"}


def test_probe_params_inherit_the_operating_point(wn, built_partitions):
    from fedwater.networks.profile import resolve_params
    f = _bundle_files()
    resolved, _ = resolve_params(BASE, f["profile"])
    part = pstore.read_partition(ROOT, NETWORK, "manual")
    meta = pstore.partition_meta(part["manifest"], f["profile"])
    probe = BASE["sensor_placement"]["probe"]
    plan = excite.horizon_plan(wn, part["districts"], meta, probe)
    seeds = plan["seeds"].set_index("district")["seed_node"]
    assert seeds["District_A"] == "J-507"               # profile, manual cut
    assert plan["largest_district"] == 316
    assert plan["n_months"] == plan["need_n_months"] >= (
        probe["warmup_months"] + plan["max_eccentricity"]
        + probe["settled_months"])
    world = {k: resolved[k] for k in ("hydraulics", "scenario", "validation",
                                      "time", "land_use", "buildings",
                                      "patterns")}
    world["scenario"]["drift"]["seed_node"] = "J-507"
    p = excite.probe_params(world, probe, plan, "District_C",
                            ("low", "residential"), probe["transitions"],
                            {"variant": "baseline"}, 42)
    assert p["hydraulics"]["anchor_scale"] == 0.5        # the profile's
    assert p["scenario"]["drift"]["tgt_district"] == "District_C"
    assert p["scenario"]["drift"]["seed_node"] is None   # not the world's
    assert p["scenario"]["drift"]["to_land_use"] == "industrial"
    assert p["time"]["days_per_month"] == probe["days_per_month"]
    assert p["patterns"]["seasonality_scale"] == 0.0
    assert p["scenario"]["income_landuse_mapping"] \
        == world["scenario"]["income_landuse_mapping"]


def test_stack_identity_is_the_operating_point_not_the_drift(wn,
                                                             built_partitions):
    from fedwater.hashing import canonical_hash
    from fedwater.networks.profile import resolve_params
    f = _bundle_files()
    resolved, _ = resolve_params(BASE, f["profile"])
    part = pstore.read_partition(ROOT, NETWORK, "manual")
    meta = pstore.partition_meta(part["manifest"], f["profile"])
    probe = BASE["sensor_placement"]["probe"]
    classes = BASE["sensor_placement"]["classes"]
    plan = excite.horizon_plan(wn, part["districts"], meta, probe)
    world = {k: copy.deepcopy(resolved[k]) for k in (
        "hydraulics", "scenario", "validation", "time", "land_use",
        "buildings", "patterns")}

    def h(world=world, coupling=None, probe=probe):
        return canonical_hash(store.stack_spec(
            network=NETWORK, partition=meta, world=world, probe=probe,
            classes=classes, plan=plan,
            coupling=coupling or {"variant": "baseline"}, seed=42))

    h0 = h()
    w = copy.deepcopy(world)
    w["scenario"]["drift"].update(tgt_district="District_C", seed_node="J-1",
                                  to_land_use="industrial")
    w["time"]["n_months"] = 99                           # world horizon
    w["validation"]["hard_pressure_floor_mca"] = 1.0
    assert h(world=w) == h0                               # shared stack
    w = copy.deepcopy(world)
    w["scenario"]["income_landuse_mapping"][1] = ["low", "industrial"]
    assert h(world=w) != h0                               # init map
    w = copy.deepcopy(world)
    w["hydraulics"]["anchor_scale"] = 0.4
    assert h(world=w) != h0                               # anchor
    assert h(coupling={"variant": "isolated"}) != h0
    pr = copy.deepcopy(probe)
    pr["transitions"]["residential"] = "commercial"
    assert h(probe=pr) != h0
    pr = copy.deepcopy(probe)
    pr["retain_series"] = False
    assert h(probe=pr) == h0                              # storage, not physics


# ==========================================================================
# coupling
# ==========================================================================
def test_label_coupling_is_the_realised_closure_set():
    from fedwater.pipelines.sensor_placement.nodes import (
        _realised_coupling, _reference_coupling)
    none = pd.DataFrame({"pipe": ["P1", "P2"], "closed": [False, False]})
    some = pd.DataFrame({"pipe": ["P2", "P1", "P3"],
                         "closed": [True, False, True]})
    assert _realised_coupling({"variant": "partial", "close_fraction": .5},
                              none) == {"variant": "baseline"}
    assert _realised_coupling({"variant": "partial", "close_fraction": .9},
                              some) == {"variant": "explicit",
                                        "closed": ["P2", "P3"]}
    assert _realised_coupling({"variant": "isolated"}, some) \
        == {"variant": "isolated"}
    assert _realised_coupling({"variant": "baseline"}, none) \
        == _reference_coupling({"variant": "baseline", "close_fraction": 0.3})
    with pytest.raises(ValueError, match="selection_coupling"):
        _reference_coupling({"variant": "partial"})


def test_apply_coupling_explicit_closes_exactly_the_given_pipes():
    import wntr
    from fedwater.pipelines.network_prep.nodes import apply_coupling
    root = ROOT / "data/01_raw/graeme"
    wn = wntr.network.WaterNetworkModel(str(root / "network.inp"))
    districts = yaml.safe_load((root / "districts.yml").read_text())
    _, drawn = apply_coupling(wn, districts, {"variant": "partial",
                                              "close_fraction": 0.5}, seed=11)
    closed = sorted(drawn.loc[drawn["closed"], "pipe"])
    assert closed
    wn_x, rebuilt = apply_coupling(
        wn, districts, {"variant": "explicit", "closed": closed}, seed=0)
    assert sorted(rebuilt.loc[rebuilt["closed"], "pipe"]) == closed
    shut = sorted(p for p in wn_x.pipe_name_list
                  if wn_x.get_link(p).initial_status
                  == wntr.network.LinkStatus.Closed)
    assert shut == closed
    with pytest.raises(ValueError, match="inter-district"):
        apply_coupling(wn, districts, {"variant": "explicit",
                                       "closed": ["not-a-pipe"]}, seed=0)


# ==========================================================================
# selection ladder
# ==========================================================================
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
    picks, cov = sel.select_slots(t, ["A"], {"flow": {"pure": 3, "mixed": 3}},
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
    picks, cov = sel.select_slots(t, ["A", "B"], slots, CLASSES, 0.10)
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
        sel.select_slots(t, ["A"], {"flow": {"pure": 3, "mixed": 2}},
                         CLASSES, 0.10)


def test_placement_frame_takes_labels_from_the_label_stack():
    s_table = pd.DataFrame([_row("q_1", "A"), _row("p_1", "A", kind="pressure")])
    picks, _ = sel.select_slots(s_table, ["A"],
                                {"flow": {"pure": 1, "mixed": 0},
                                 "pressure": {"pure": 1, "mixed": 0}},
                                CLASSES, 0.10)
    l_table = pd.DataFrame([
        _row("q_1", "A", tier="transition", purity=.4, second="B",
             w_second=.6, n_mass=2),
        _row("p_1", "A")]).assign(w_A=[.4, 1.0], w_B=[.6, 0.0])
    channels = pd.DataFrame({"kind": ["flow", "pressure"],
                             "verdict": ["ok", "degenerate"]})
    out = sel.placement_frame(picks, l_table, ["A", "B"], channels,
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


def test_manual_placement_keeps_element_names_and_order():
    profile = _bundle_files()["profile"]
    m = sel.manual_placement(profile, {"network": NETWORK})
    assert (m["slot"] == m["sensor"]).all()
    a = m[m["district"] == "District_A"]
    assert list(a["sensor"]) == (
        [f"p_{x}" for x in profile["sensors"]["District_A"]["pressure"]]
        + [f"q_{x}" for x in profile["sensors"]["District_A"]["flow"]])
    with pytest.raises(ValueError, match="sensors"):
        sel.manual_placement({}, {})


# ==========================================================================
# downstream contract
# ==========================================================================
def _toy_series():
    idx = pd.RangeIndex(4, name="step")
    pressures = pd.DataFrame({"J1": [1., 2, 3, 4], "J2": [5., 6, 7, 8],
                              "month": [0, 0, 1, 1]}, index=idx)
    flows = pd.DataFrame({"P1": [.1, .2, .3, .4], "P2": [1., 1, 1, 1],
                          "month": [0, 0, 1, 1]}, index=idx)
    return pressures, flows


def test_sensing_reads_the_placement_and_names_columns_by_slot():
    from fedwater.pipelines.sensing.nodes import (
        add_measurement_noise, extract_sensor_series, package_client_datasets)
    pressures, flows = _toy_series()
    placement = pd.DataFrame({
        "district": ["A", "A", "B", "B"], "kind": ["flow", "pressure"] * 2,
        "slot": ["q_pure_0", "p_pure_0"] * 2,
        "sensor": ["q_P1", "p_J1", "q_P2", "p_J2"],
        "element": ["P1", "J1", "P2", "J2"]})
    true = extract_sensor_series(pressures, flows, placement)
    assert set(true.columns) >= {"sensor", "slot", "kind", "value"}
    noisy = add_measurement_noise(true, BASE["noise"], seed=1)
    # noise is keyed on the ELEMENT: re-slotting a gauge (or moving it to
    # another district's placement) does not change its realisation
    reslotted = true.assign(slot="q_mixed_7", district="Z")
    again = add_measurement_noise(reslotted, BASE["noise"], seed=1)
    pd.testing.assert_series_equal(noisy["observed"], again["observed"])
    clients = package_client_datasets(
        noisy, {"resolution_h": 1}, "2024-01-01")
    assert list(clients["A"].columns) == list(clients["B"].columns)
    assert {"p_pure_0", "q_pure_0"} <= set(clients["A"].columns)
    np.testing.assert_allclose(clients["A"]["q_pure_0"],
                               flows["P1"], atol=1.5)
    with pytest.raises(ValueError, match="empty"):
        extract_sensor_series(pressures, flows, placement.iloc[:0])


def test_topology_features_measure_between_the_placed_pressure_gauges():
    import wntr
    from fedwater.pipelines.dependence_oracle.nodes import topology_features
    from fedwater.pipelines.network_prep.nodes import apply_coupling
    root = ROOT / "data/01_raw/graeme"
    wn = wntr.network.WaterNetworkModel(str(root / "network.inp"))
    districts = yaml.safe_load((root / "districts.yml").read_text())
    profile = yaml.safe_load((root / "profile.yml").read_text())
    wn_b, gt = apply_coupling(wn, districts, {"variant": "baseline"}, seed=0)
    manual = sel.manual_placement(profile, {})
    a = topology_features(wn_b, districts, gt, manual)
    moved = manual.copy()
    moved.loc[(moved["district"] == "District_A")
              & (moved["kind"] == "pressure"), "element"] = "110"
    b = topology_features(wn_b, districts, gt, moved)
    ab = ["district_a", "district_b"]
    ja = a.set_index(ab)["hydraulic_distance_m"]
    jb = b.set_index(ab)["hydraulic_distance_m"]
    touched = [k for k in ja.index if "District_A" in k]
    assert (ja.drop(touched) == jb.drop(touched)).all()
    assert (ja[touched] != jb[touched]).any()


def test_fl_feature_filter_is_run_level_and_class_aware():
    from fedwater.pipelines.fl_preprocessing.nodes import select_feature_columns
    cols = ["timestamp", "month", "p_mixed_0", "p_pure_0",
            "q_mixed_0", "q_pure_0", "q_pure_1"]
    assert select_feature_columns(cols, {}) == [
        "p_mixed_0", "p_pure_0", "q_mixed_0", "q_pure_0", "q_pure_1"]
    assert select_feature_columns(cols, BASE["fl"]["preprocessing"]) \
        == select_feature_columns(cols, {})
    assert select_feature_columns(cols, {"channels": ["flow"],
                                         "classes": ["pure"]}) \
        == ["q_pure_0", "q_pure_1"]
    with pytest.raises(ValueError, match="slot-named"):
        select_feature_columns(["p_J-1", "q_P-2"], {"classes": ["pure"]})
    with pytest.raises(ValueError, match="channels"):
        select_feature_columns(cols, {"channels": ["acoustic"]})
    with pytest.raises(ValueError, match="survives"):
        select_feature_columns(["q_pure_0"], {"channels": ["pressure"]})


# ==========================================================================
# the whole stack, small (real EPANET)
# ==========================================================================
@pytest.mark.integration
def test_probe_stack_builds_caches_and_reloads(tmp_path, built_partitions):
    """Graeme, shortened probe horizon: a stack is built once, reloads with
    the same tables, is served from cache, and a failed build is retried."""
    import wntr
    from fedwater.networks.profile import resolve_params
    from fedwater.pipelines.sensor_placement.nodes import (
        ensure_probe_stacks, select_sensors)
    network = "graeme"
    root = ROOT / "data/01_raw" / network
    wn = wntr.network.WaterNetworkModel(str(root / "network.inp"))
    profile = yaml.safe_load((root / "profile.yml").read_text())
    part = pstore.read_partition(ROOT, network, "manual")
    meta = pstore.partition_meta(part["manifest"], profile)
    params, _ = resolve_params(BASE, profile)
    sp = copy.deepcopy(params["sensor_placement"])
    sp["store"] = str(tmp_path / "probes")
    sp["probe"].update(warmup_months=4, settled_months=5, days_per_month=7)
    sp["slots"] = {"flow": {"pure": 2, "mixed": 1},
                   "pressure": {"pure": 1, "mixed": 1}}
    gt = pd.DataFrame({"pipe": ["6"], "closed": [False]})
    args = (wn, profile, part["districts"], part["manifest"], meta,
            params["hydraulics"], params["scenario"], params["validation"],
            params["time"], {"variant": "partial", "close_fraction": .5}, gt,
            params["land_use"], params["buildings"], params["patterns"], sp)
    ptr = ensure_probe_stacks(*args)
    assert ptr["same_stack"] and ptr["label"]["coupling"] == {
        "variant": "baseline"}
    path = Path(ptr["selection"]["path"])
    first = store.load_stack(path)
    assert (first.mixture["tier"] != "").all() and len(first.channels) == 2
    assert set(first.dependence) == {"flow", "pressure"}
    assert set(first.elasticity) == {"shape", "native"}
    stamp = (path / "manifest.json").stat().st_mtime_ns
    assert ensure_probe_stacks(*args)["selection"]["stack_hash"] \
        == ptr["selection"]["stack_hash"]
    assert (path / "manifest.json").stat().st_mtime_ns == stamp   # cached
    probes = store.load_probes(path)
    assert sorted(probes) == sorted(part["districts"]["districts"])

    placement, coverage, channels, arms = select_sensors(
        ptr, part["districts"], profile, meta, sp)
    assert placement.groupby("district").size().nunique() == 1
    assert len(placement) == 5 * (2 + 1 + 1 + 1)
    assert {f"w_{d}" for d in part["districts"]["districts"]} \
        <= set(placement.columns)

    # a failed manifest never serves; the next request rebuilds
    m = (path / "manifest.json")
    m.write_text(m.read_text().replace('"status": "ok"',
                                       '"status": "failed"'))
    with pytest.raises(ValueError, match="failed"):
        store.load_stack(path)
    ensure_probe_stacks(*args)
    assert store.load_stack(path).status == "ok"


# ==========================================================================
# one ground for world and probes: horizon / seasonality / diffusion modes
# ==========================================================================
def _world_blocks(n_months=50, dpm=30, warmup=12, seasonality=1.0, ramp=42):
    from fedwater.networks.profile import resolve_params
    resolved, _ = resolve_params(BASE, _bundle_files()["profile"])
    w = {k: copy.deepcopy(resolved[k]) for k in (
        "hydraulics", "scenario", "validation", "time", "land_use",
        "buildings", "patterns")}
    w["time"].update(n_months=n_months, days_per_month=dpm)
    w["scenario"]["n_months"] = n_months
    w["scenario"]["drift"]["warmup_months"] = warmup
    w["patterns"].update(seasonality_scale=seasonality, drift_ramp_days=ramp)
    return w


def _probe(**modes):
    p = copy.deepcopy(BASE["sensor_placement"]["probe"])
    p.update(modes)
    return p


@pytest.fixture(scope="module")
def manual_part(built_partitions):
    part = pstore.read_partition(ROOT, NETWORK, "manual")
    meta = pstore.partition_meta(part["manifest"], _bundle_files()["profile"])
    return part, meta


def test_world_horizon_puts_probes_on_the_world_grid(wn, manual_part):
    part, meta = manual_part
    world = _world_blocks()
    plan = excite.horizon_plan(wn, part["districts"], meta,
                               _probe(horizon="world", seasonality="world"),
                               world)
    assert (plan["n_months"], plan["days_per_month"], plan["warmup_months"],
            plan["drift_ramp_days"]) == (50, 30, 12, 42)
    assert plan["seasonality_scale"] == 1.0 and plan["season_aligned"]
    assert plan["settled_months"] == 12
    # the planned front makes the LARGEST district convert and settle in time
    assert plan["need_n_months"] <= 50
    assert plan["convert_months"] >= plan["max_eccentricity"]
    p = excite.probe_params(world, _probe(horizon="world"), plan, "District_B",
                            ("low", "residential"), BASE["sensor_placement"][
                                "probe"]["transitions"],
                            {"variant": "baseline"}, 42)
    assert p["time"]["n_months"] == 50 and p["time"]["days_per_month"] == 30
    assert p["scenario"]["drift"]["warmup_months"] == 12
    assert p["patterns"]["seasonality_scale"] == 1.0
    assert p["patterns"]["drift_ramp_days"] == 42
    # seasonality none: same grid, no season, the probe block's settled rule
    flat = excite.horizon_plan(wn, part["districts"], meta,
                               _probe(horizon="world", seasonality="none"),
                               world)
    assert flat["seasonality_scale"] == 0.0 and not flat["season_aligned"]
    assert flat["settled_months"] == BASE["sensor_placement"]["probe"][
        "settled_months"]


def test_world_modes_refuse_what_cannot_settle(wn, manual_part):
    part, meta = manual_part
    with pytest.raises(ValueError, match="whole-year"):
        excite.horizon_plan(wn, part["districts"], meta,
                            _probe(horizon="world", seasonality="world"),
                            _world_blocks(warmup=2))
    with pytest.raises(ValueError, match="needs n_months >="):
        excite.horizon_plan(wn, part["districts"], meta,
                            _probe(horizon="world", seasonality="world"),
                            _world_blocks(n_months=30))
    # the world's own front (2 nodes/month, growth 0.8) cannot convert the
    # 316-junction district in 50 months
    with pytest.raises(ValueError, match="diffusion=world"):
        excite.horizon_plan(wn, part["districts"], meta,
                            _probe(horizon="world", seasonality="world",
                                   diffusion="world"), _world_blocks())
    with pytest.raises(ValueError, match="horizon"):
        excite.probe_modes({"horizon": "probe"})
    with pytest.raises(ValueError, match="world"):
        excite.horizon_plan(wn, part["districts"], meta,
                            _probe(horizon="world"), None)
    # defaults stay on the planned grid
    assert excite.probe_modes(BASE["sensor_placement"]["probe"]) == {
        "horizon": "planned", "diffusion": "planned", "seasonality": "none"}


def test_stack_identity_follows_the_world_grid_in_world_mode(wn, manual_part):
    from fedwater.hashing import canonical_hash
    part, meta = manual_part
    classes = BASE["sensor_placement"]["classes"]

    def h(world, probe):
        plan = excite.horizon_plan(wn, part["districts"], meta, probe, world)
        return canonical_hash(store.stack_spec(
            network=NETWORK, partition=meta, world=world, probe=probe,
            classes=classes, plan=plan, coupling={"variant": "baseline"},
            seed=42))

    wm = _probe(horizon="world", seasonality="world")
    pl = _probe()
    base = h(_world_blocks(), wm)
    assert h(_world_blocks(n_months=60), wm) != base      # world grid is identity
    assert h(_world_blocks(seasonality=0.5), wm) != base
    assert h(_world_blocks(n_months=60), pl) == h(_world_blocks(), pl)  # planned
    assert h(_world_blocks(seasonality=0.5), pl) == h(_world_blocks(), pl)
    assert h(_world_blocks(), pl) != base


def test_season_windows_cover_whole_years():
    from fedwater.placement import mixture_probe as mp
    from fedwater.placement import signal_probe as sp
    sch = pd.DataFrame({"node": ["n1", "n2"], "drift_month": [12, 20],
                        "district": ["A", "A"]})
    P = sp.Probe(districts={"A": ["n1", "n2"]}, assets={}, demand=pd.DataFrame(),
                 pressures=pd.DataFrame(), flows=pd.DataFrame(), schedule=sch,
                 time={"n_months": 50, "days_per_month": 30, "resolution_h": 1},
                 params={"patterns": {"drift_ramp_days": 42}})
    sm = P.steps_month
    (b0, b1), (s0, s1) = mp.season_windows(P, pad=1)
    assert (b0 // sm, b1 // sm) == (0, 12)
    # final starts at 20 + ceil(42/30) = 22, +1 pad -> 23; 27 months left -> 24
    assert (s0 // sm, s1 // sm) == (26, 50)
    assert ((b1 - b0) // sm) % 12 == 0 and ((s1 - s0) // sm) % 12 == 0
    P.time["n_months"] = 34
    with pytest.raises(ValueError, match="settled months"):
        mp.season_windows(P, pad=1)


def test_study_keys_reach_the_world_and_the_store_stays_shared(built_partitions):
    from fedwater.experiments import spec as spec_mod
    b = spec_mod.bundle(ROOT, NETWORK, "manual")
    w = spec_mod.resolve_world(
        {"n_months": 50, "days_per_month": 30, "seasonality_scale": 0.0,
         "drift_ramp_days": 21, "drift": {"warmup_months": 12},
         "placement": {"probe": {"horizon": "world", "seasonality": "world"},
                       "slots": {"pressure": {"pure": 1, "mixed": 1}}}},
        BASE, NETWORK, 4, b["profile"], b["partition"])
    o = w["override"]
    assert o["time"]["days_per_month"] == 30
    assert o["patterns"]["seasonality_scale"] == 0.0
    assert o["patterns"]["drift_ramp_days"] == 21
    assert set(o["patterns"]) == set(BASE["patterns"])       # full block
    assert o["sensor_placement"]["probe"]["horizon"] == "world"
    assert o["sensor_placement"]["slots"]["flow"] == BASE["sensor_placement"][
        "slots"]["flow"]                                       # deep merge
    # written back as the interpolation, so clones share the engine's store
    assert o["sensor_placement"]["store"] == "${globals:probe_store}"
    assert "store" not in w["effective"]["sensor_placement"]
    assert w["flat"]["probe_horizon"] == "world"
    assert w["flat"]["days_per_month"] == 30
    same = spec_mod.resolve_world(
        {"n_months": 50, "days_per_month": 30, "seasonality_scale": 0.0,
         "drift_ramp_days": 21, "drift": {"warmup_months": 12},
         "sim_overrides": {"sensor_placement": {
             "probe": {"horizon": "world", "seasonality": "world"},
             "slots": {"pressure": {"pure": 1, "mixed": 1}}}}},
        BASE, NETWORK, 4, b["profile"], b["partition"])
    assert same["sim_hash"] == w["sim_hash"]                   # one meaning
    with pytest.raises(ValueError, match="districting"):
        spec_mod.resolve_world({"sim_overrides": {"districting": {"k": 5}}},
                               BASE, NETWORK, 4, b["profile"], b["partition"])
    seasonal = spec_mod.resolve_world(
        {"drift": {"warmup_months": 2},
         "placement": {"probe": {"horizon": "world",
                                 "seasonality": "world"}}},
        BASE, NETWORK, 4, b["profile"], b["partition"])
    with pytest.raises(ValueError, match="whole-year"):
        spec_mod.validate_world(seasonal, b["districts"])


def test_engine_preflight_refuses_a_horizon_the_probes_cannot_fill(
        tmp_path, built_partitions):
    from fedwater.experiments import spec as spec_mod
    from fedwater.experiments.engine import ExperimentEngine
    b = spec_mod.bundle(ROOT, NETWORK, "manual")
    engine = ExperimentEngine(ROOT, root=tmp_path / "exp")

    def world(n):
        return spec_mod.resolve_world(
            {"n_months": n, "days_per_month": 30, "drift_ramp_days": 42,
             "drift": {"warmup_months": 12},
             "placement": {"probe": {"horizon": "world",
                                     "seasonality": "world"}}},
            BASE, NETWORK, 4, b["profile"], b["partition"])

    with pytest.raises(ValueError, match="does not fit for 1 world"):
        engine.preflight_placement({"worlds": [world(40)]})
    engine.preflight_placement({"worlds": [world(50)]})
    manual = copy.deepcopy(world(40))
    manual["effective"]["sensor_placement"]["source"] = "manual"
    engine.preflight_placement({"worlds": [manual]})       # nothing to plan
