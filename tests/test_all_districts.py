"""Every-district worlds, the auto front and horizon, and the mixture gate.

Real data only: KY7 (manual partition) for the design and the horizon, whose
answers are checked against the project's OWN pipeline (network_prep ->
urban_scenario run in-process), and a shortened Graeme probe stack for the
gate, whose rule is checked cell by cell against the stack's stored numbers.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from fedwater.config import load_base_params
from fedwater.experiments import spec as spec_mod
from fedwater.hashing import canonical_hash
from fedwater.networks import partitions as pstore
from fedwater.pipelines.urban_scenario.nodes import (build_drift_schedule,
                                                     drift_completion,
                                                     drift_front)
from fedwater.placement import excite, store

ROOT = Path(__file__).resolve().parents[1]
BASE = load_base_params(ROOT)
STUDY = "ky7_all_districts"


@pytest.fixture(scope="module")
def ky7(built_partitions):
    import wntr
    b = spec_mod.bundle(ROOT, "ky7", "manual")
    wn = wntr.network.WaterNetworkModel(
        str(ROOT / "data/01_raw/ky7/network.inp"))
    return {"bundle": b, "wn": wn}


@pytest.fixture(scope="module")
def study(built_partitions):
    return spec_mod.expand_study(STUDY, ROOT)


# ==========================================================================
# front
# ==========================================================================
def test_front_auto_is_sized_per_target_district(ky7):
    districts = ky7["bundle"]["districts"]["districts"]
    auto = {"max_neighbors_per_month": "auto", "convert_months": 12}
    for d, nodes in districts.items():
        assert drift_front(auto, len(nodes)) == math.ceil(len(nodes) / 12)
    assert drift_front({"max_neighbors_per_month": 2}, 316) == 2  # manual
    with pytest.raises(ValueError, match="convert_months"):
        drift_front({"max_neighbors_per_month": "auto"}, 30)
    with pytest.raises(ValueError, match="int >= 1"):
        drift_front({"max_neighbors_per_month": 0}, 30)


# ==========================================================================
# the every-district design
# ==========================================================================
def test_all_districts_expands_to_the_probe_design(study, ky7):
    b = ky7["bundle"]
    names = list(b["districts"]["districts"])
    transitions = BASE["sensor_placement"]["probe"]["transitions"]
    init = spec_mod.decode_map("LI_LR_LI_LI", len(names))
    worlds = study["worlds"]
    assert [w["flat"]["drift_district"] for w in worlds] == names
    for w, (income, land_use) in zip(worlds, init):
        assert w["flat"]["drift_to_income"] == income           # income kept
        assert w["flat"]["drift_to_land_use"] == transitions[land_use]
        assert w["flat"]["all_districts"] and w["flat"]["n_months_auto"]
        assert w["flat"]["sim_seed"] != BASE["sensor_placement"]["probe"]["seed"]
    assert len({w["sim_hash"] for w in worlds}) == len(names)


def test_all_districts_refuses_what_it_would_overwrite(ky7):
    b = ky7["bundle"]
    base_world = {"network": "ky7", "sim_seed": 43, "drift_all_districts": True,
                  "consumption_map": "LI_LR_LI_LI"}

    def expand(**patch):
        w = {**copy.deepcopy(base_world), **patch}
        w.pop("network")
        return spec_mod.all_district_worlds(w, BASE, "ky7", b["districts"],
                                            b["profile"], b["partition"])

    assert len(expand()) == 4
    with pytest.raises(ValueError, match="drift target"):
        expand(drift={"tgt_district": "District_A"})
    with pytest.raises(ValueError, match="in-sample"):
        expand(sim_seed=int(BASE["sensor_placement"]["probe"]["seed"]))
    # manual sensors have no probe stack to be in-sample against
    assert len(expand(sim_seed=42, placement={"source": "manual"})) == 4
    single = spec_mod.all_district_worlds({"sim_seed": 43}, BASE, "ky7",
                                          b["districts"], b["profile"],
                                          b["partition"])
    assert single == [{"sim_seed": 43}]


# ==========================================================================
# the auto horizon
# ==========================================================================
def test_replayed_completion_is_the_pipelines_own_schedule(study, ky7):
    """The predicted last switch is what the world's pipeline produces."""
    root = ROOT / "data/01_raw/ky7"
    part = pstore.read_partition(ROOT, "ky7", "manual")
    pipe = excite.core_pipeline(targets=("gt_drift_schedule",))
    for w in study["worlds"]:
        eff = w["effective"]
        data = {"network_inp": copy.deepcopy(ky7["wn"]),
                "network_profile": yaml.safe_load(
                    (root / "profile.yml").read_text()),
                "districts": part["districts"],
                "partition_manifest": part["manifest"]}
        for k in ("hydraulics", "scenario", "validation", "time", "coupling",
                  "seed", "land_use", "buildings", "patterns"):
            data[f"params:{k}"] = eff[k]
        sch = excite._run_inprocess(pipe, data)["gt_drift_schedule"]
        assert int(sch["drift_month"].max()) == w["flat"]["drift_last_switch"]
        # and the replay (horizon lifted) is the same table, row for row
        replay = build_drift_schedule(
            ky7["wn"], part["districts"], ky7["bundle"]["partition"],
            {**eff["scenario"], "n_months": 10_000}, int(eff["seed"]))
        assert replay.reset_index(drop=True).equals(
            sch.reset_index(drop=True))


def test_auto_horizon_is_one_group_max_with_whole_years(study, ky7,
                                                         tmp_path):
    b, wn = ky7["bundle"], ky7["wn"]
    worlds = study["worlds"]
    ah = BASE["auto_horizon"]
    months = {w["flat"]["n_months"] for w in worlds}
    assert len(months) == 1                                  # one horizon
    n = months.pop()
    ramp = math.ceil(42 / 30)
    pad = BASE["sensor_placement"]["probe"]["settle"]["pad"]
    last = max(w["flat"]["drift_last_switch"] for w in worlds)
    assert n == last + ramp + pad + ah["post_months"] + ah["spare_months"]
    for w in worlds:
        eff = w["effective"]
        assert eff["scenario"]["drift"]["warmup_months"] == ah["pre_months"]
        assert eff["time"]["n_months"] == eff["scenario"]["n_months"] == n
        # the season-aligned windows verify / the probes use both fit
        start = w["flat"]["drift_last_switch"] + ramp + pad
        assert n - start >= 12 and eff["scenario"]["drift"][
            "warmup_months"] >= 12
        done = drift_completion(wn, b["districts"], b["partition"],
                                eff["scenario"], int(eff["seed"]))
        assert done["n_converted"] == done["n_reachable"]
    # the whole group shares ONE selection stack, and the pre-flight passes
    hashes = set()
    for w in worlds:
        eff = w["effective"]
        sp = eff["sensor_placement"]
        world = {k: eff[k] for k in ("hydraulics", "scenario", "validation",
                                     "time", "land_use", "buildings",
                                     "patterns")}
        plan = excite.horizon_plan(wn, b["districts"], b["partition"],
                                   sp["probe"], world)
        hashes.add(canonical_hash(store.stack_spec(
            network="ky7", partition=b["partition"], world=world,
            probe=sp["probe"], classes=sp["classes"], plan=plan,
            coupling={"variant": "baseline"}, seed=int(sp["probe"]["seed"]))))
    assert len(hashes) == 1
    from fedwater.experiments.engine import ExperimentEngine
    ExperimentEngine(ROOT, root=tmp_path / "exp").preflight_placement(study)


def test_auto_horizon_config_is_checked_and_not_identity(ky7):
    with pytest.raises(ValueError, match="missing"):
        spec_mod.auto_horizon_config({"auto_horizon": {"pre_months": 12}}, {})
    cfg = spec_mod.auto_horizon_config(
        BASE, {"auto_horizon": {"spare_months": 5}})
    assert cfg["spare_months"] == 5 and cfg["post_months"] == 12
    b = ky7["bundle"]
    a = spec_mod.resolve_world({}, BASE, "ky7", 4, b["profile"], b["partition"])
    other = copy.deepcopy(BASE)
    other["auto_horizon"]["spare_months"] = 9
    c = spec_mod.resolve_world({}, other, "ky7", 4, b["profile"],
                               b["partition"])
    assert a["sim_hash"] == c["sim_hash"]


def test_auto_front_reaches_the_probes_in_world_diffusion(ky7):
    """diffusion: world under an auto front: each probe district gets its own
    width, and the front's convert_months is stack identity."""
    b, wn = ky7["bundle"], ky7["wn"]
    from fedwater.networks.profile import resolve_params
    resolved, _ = resolve_params(BASE, b["profile"])
    world = {k: copy.deepcopy(resolved[k]) for k in (
        "hydraulics", "scenario", "validation", "time", "land_use",
        "buildings", "patterns")}
    world["time"].update(n_months=60, days_per_month=30)
    world["scenario"]["drift"].update(warmup_months=12,
                                      max_neighbors_per_month="auto",
                                      convert_months=12)
    world["patterns"].update(seasonality_scale=1.0, drift_ramp_days=42)
    probe = {**copy.deepcopy(BASE["sensor_placement"]["probe"]),
             "horizon": "world", "seasonality": "world", "diffusion": "world"}
    plan = excite.horizon_plan(wn, b["districts"], b["partition"], probe, world)
    assert plan["max_neighbors_per_month"] == "auto"
    p = excite.probe_params(world, probe, plan, "District_C",
                            ("low", "residential"), probe["transitions"],
                            {"variant": "baseline"}, 42)
    assert p["scenario"]["drift"]["max_neighbors_per_month"] == "auto"
    assert p["scenario"]["drift"]["convert_months"] == 12
    classes = BASE["sensor_placement"]["classes"]

    def h(w):
        pl = excite.horizon_plan(wn, b["districts"], b["partition"], probe, w)
        return canonical_hash(store.stack_spec(
            network="ky7", partition=b["partition"], world=w, probe=probe,
            classes=classes, plan=pl, coupling={"variant": "baseline"},
            seed=42))
    w2 = copy.deepcopy(world)
    w2["scenario"]["drift"]["convert_months"] = 10
    assert h(w2) != h(world)


# ==========================================================================
# the gate
# ==========================================================================
def test_gate_default_keeps_every_stack_identity(ky7):
    b, wn = ky7["bundle"], ky7["wn"]
    from fedwater.networks.profile import resolve_params
    resolved, _ = resolve_params(BASE, b["profile"])
    world = {k: copy.deepcopy(resolved[k]) for k in (
        "hydraulics", "scenario", "validation", "time", "land_use",
        "buildings", "patterns")}
    probe = BASE["sensor_placement"]["probe"]
    plan = excite.horizon_plan(wn, b["districts"], b["partition"], probe)

    def h(classes):
        return canonical_hash(store.stack_spec(
            network="ky7", partition=b["partition"], world=world, probe=probe,
            classes=classes, plan=plan, coupling={"variant": "baseline"},
            seed=42))
    # a stack built before the keys existed: no gate, no mixture rule
    classes = {k: v for k, v in BASE["sensor_placement"]["classes"].items()
               if k not in ("gate", "mixture")}
    assert h({**classes, "gate": "norm"}) == h(classes)
    assert h({**classes, "mixture": "positive"}) == h(classes)
    assert h({**classes, "gate": "projection"}) != h(classes)
    assert h({**classes, "mixture": "magnitude"}) != h(classes)
    with pytest.raises(ValueError, match="gate"):
        store.gate({"gate": "both"})


@pytest.mark.integration
def test_gate_rule_holds_cell_by_cell_on_a_real_stack(tmp_path,
                                                      built_partitions):
    """Graeme, shortened probe horizon (as the stack integration test): each
    gate admits exactly the cells its rule says, on the stack's own numbers."""
    import wntr
    from fedwater.networks.profile import resolve_params
    from fedwater.placement import mixture_probe as mp
    network = "graeme"
    root = ROOT / "data/01_raw" / network
    wn = wntr.network.WaterNetworkModel(str(root / "network.inp"))
    profile = yaml.safe_load((root / "profile.yml").read_text())
    part = pstore.read_partition(ROOT, network, "manual")
    meta = pstore.partition_meta(part["manifest"], profile)
    params, _ = resolve_params(BASE, profile)
    sp = copy.deepcopy(params["sensor_placement"])
    sp["probe"].update(warmup_months=4, settled_months=5, days_per_month=7)
    # built under the norm gate so each gate's rule is checked from one base
    sp["classes"].update(gate="norm", mixture="positive")
    world = {k: params[k] for k in ("hydraulics", "scenario", "validation",
                                    "time", "land_use", "buildings",
                                    "patterns")}
    plan = excite.horizon_plan(wn, part["districts"], meta, sp["probe"], world)
    transitions = excite.validate_transitions(sp["probe"]["transitions"],
                                              params["land_use"])
    spec = store.stack_spec(network=network, partition=meta, world=world,
                            probe=sp["probe"], classes=sp["classes"],
                            plan=plan, coupling={"variant": "baseline"},
                            seed=42)
    st = store.ensure_stack(
        store_root=tmp_path, spec=spec,
        inputs={"network_inp": wn, "network_profile": profile,
                "districts": part["districts"],
                "partition_manifest": part["manifest"]},
        world=world, probe=sp["probe"], classes=sp["classes"], plan=plan,
        transitions=transitions, coupling={"variant": "baseline"}, seed=42,
        verbose=False)
    base = st.base
    f = float(sp["classes"]["null_factor"])
    ids = base["G"].index
    norm = mp.remix(base, null_factor=f)["mixture"].set_index("id")
    proj = mp.remix(base, null_factor=f, gate="projection")["mixture"] \
        .set_index("id")
    want_norm = (base["DZ"].to_numpy()
                 > f * base["null"].to_numpy()[:, None]).sum(axis=1)
    want_proj = (np.abs(base["PROJ"].to_numpy())
                 > f * base["NU"].to_numpy()).sum(axis=1)
    assert (norm.loc[ids, "n_live"].to_numpy() == want_norm).all()
    assert (proj.loc[ids, "n_live"].to_numpy() == want_proj).all()
    assert base["gate"] == "norm" and st.mixture.equals(
        mp.remix(base, null_factor=f)["mixture"])
    # the gate reaches a stack build through classes, with no re-simulation
    classes = {**sp["classes"], "gate": "projection"}
    st2 = copy.copy(st)
    store.analyze(st2, st.probes, beta=float(world["scenario"]["beta"]),
                  classes=classes, probe=sp["probe"], verbose=False)
    assert st2.base["gate"] == "projection"
    assert (st2.mixture.set_index("id").loc[ids, "n_live"].to_numpy()
            == want_proj).all()
