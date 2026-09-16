"""Tests for fedwater.experiments: spec resolution, engine mechanics (with a
stubbed kedro), harvest extraction, and — when torch/wntr are installed — a
micro-world integration run through the real pipelines.

House discipline: every correctness claim is pinned against a case whose
ground truth is known by construction (hand-built CSVs with a known AUROC,
a spec whose hash must/must-not move, a fake kedro whose failure mode we
chose)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fedwater.experiments import harvest as harvest_mod
from fedwater.experiments import spec as spec_mod
from fedwater.experiments.engine import ExperimentEngine
from fedwater.pipelines.fl_training.nodes import effective_fl_seed

ROOT = Path(__file__).resolve().parents[1]
NETWORK = "ky7"           # default bundle; Graeme where a study pins it

from fedwater.config import load_base_params  # noqa: E402

BASE = load_base_params(ROOT)


@pytest.fixture(scope="module")
def ctx(built_partitions):
    """KY7 through its manual partition: bundle + a resolve_world shortcut."""
    b = spec_mod.bundle(ROOT, NETWORK, "manual")
    n = len(b["districts"]["districts"])

    def world(spec=None, base=BASE):
        return spec_mod.resolve_world(spec or {}, base, NETWORK, n,
                                      b["profile"], b["partition"])
    return {"bundle": b, "districts": b["districts"], "inp": b["inp"],
            "world": world, "n": n}


# ---------------------------------------------------------------- spec ----
def test_map_codec_matches_the_bundle_scenario(ctx):
    """The compact code and the spelled-out mapping must agree -- this is what
    stops a profile and experiments.yml drifting apart silently. The mapping
    is network-scoped, so it is read from the resolved bundle."""
    mapping = ctx["bundle"]["profile"]["network_params"]["scenario"][
        "income_landuse_mapping"]
    code = spec_mod.encode_map(mapping)
    assert code == "LR_LR_LR_LR"
    assert spec_mod.decode_map(code, ctx["n"]) == [list(x) for x in mapping]


def test_canonical_hash_is_order_and_numeric_type_stable():
    a = {"b": 1, "a": [1, 2, {"x": 0.0}]}
    b = {"a": [1, 2, {"x": 0}], "b": 1.0}
    assert spec_mod.canonical_hash(a) == spec_mod.canonical_hash(b)
    assert spec_mod.canonical_hash({"a": 1}) != spec_mod.canonical_hash({"a": 2})


def test_expand_axes_fixed_grid_zip_and_range():
    section = {"fixed": {"rounds": 8},
               "grid": {"fl_seed": {"range": 3}, "k": ["x", "y"]},
               "zip": [{"step_size": 1, "batch_size": 64},
                       {"step_size": 2, "batch_size": 128}]}
    out = spec_mod.expand_axes(section)
    assert len(out) == 3 * 2 * 2                       # grid x grid x zip
    assert all(o["rounds"] == 8 for o in out)
    assert {(o["step_size"], o["batch_size"]) for o in out} \
        == {(1, 64), (2, 128)}                          # zip stays paired
    assert spec_mod.expand_axes([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]
    assert spec_mod.expand_axes(None) == [{}]


def test_resolve_world_hash_moves_with_the_physics_only(ctx):
    world = ctx["world"]
    w0 = world()                                       # pure base + bundle
    assert w0["flat"]["consumption_map"] == "LR_LR_LR_LR"
    assert w0["flat"]["drift_district"] == "District_A"
    assert w0["flat"]["drift_to_land_use"] == "commercial"
    assert w0["flat"]["anchor_scale"] == 0.5           # from the ky7 profile
    w_map = world({"consumption_map": "LR_MR_HR_LR"})
    w_seed = world({"sim_seed": 7})
    w_beta = world({"beta": 0.0})
    w_anchor = spec_mod.with_anchor(w0, 0.04)
    hashes = {w["sim_hash"] for w in (w0, w_map, w_seed, w_beta, w_anchor)}
    assert len(hashes) == 5
    assert world()["sim_hash"] == w0["sim_hash"]
    # beta is a world axis, and with_beta must agree with the spec route
    assert spec_mod.with_beta(w0, 0.0)["sim_hash"] == w_beta["sim_hash"]
    # full-block override discipline (the Kedro destructive-merge gotcha):
    assert set(w_map["override"]["scenario"]) == set(BASE["scenario"])
    assert set(w_map["override"]["hydraulics"]) == set(BASE["hydraulics"])
    # the world carries its partition to the catalog, whole
    assert w0["globals"] == {"network": "ky7", "districting": {
        "active": "manual", "methods": ["manual"]}}


def test_world_identity_tracks_partition_and_placement_rule(ctx):
    """Dynamic sensors: the RULE is identity, the profile block is not; the
    store path is deployment; districting dials are identity only through
    the partition they produce."""
    import copy
    world, b = ctx["world"], ctx["bundle"]
    w0 = world()
    assert w0["flat"]["partition_id"] == b["partition"]["partition_id"]
    assert w0["flat"]["placement_source"] == "dynamic"
    assert "sensors" not in w0["effective"]

    other = {**b["partition"], "partition_id": "deadbeef"}
    moved = spec_mod.resolve_world({}, BASE, NETWORK, ctx["n"],
                                   b["profile"], other)
    assert moved["sim_hash"] != w0["sim_hash"]

    def with_base(mutate):
        base = copy.deepcopy(BASE)
        mutate(base)
        return world(base=base)["sim_hash"]

    assert with_base(lambda p: p["sensor_placement"].update(
        store="/somewhere/else")) == w0["sim_hash"]
    assert with_base(lambda p: p["districting"].update(seed=99)) \
        == w0["sim_hash"]
    assert with_base(lambda p: p["sensor_placement"]["slots"]["flow"].update(
        mixed=1)) != w0["sim_hash"]
    assert with_base(lambda p: p["sensor_placement"]["probe"][
        "transitions"].update(industrial="commercial")) != w0["sim_hash"]

    profile2 = copy.deepcopy(b["profile"])
    profile2["sensors"]["District_A"]["flow"] = ["P-1"]
    same = spec_mod.resolve_world({}, BASE, NETWORK, ctx["n"], profile2,
                                  b["partition"])
    assert same["sim_hash"] == w0["sim_hash"]         # dynamic: not read

    manual = copy.deepcopy(BASE)
    manual["sensor_placement"]["source"] = "manual"
    h1 = spec_mod.resolve_world({}, manual, NETWORK, ctx["n"], b["profile"],
                                b["partition"])
    h2 = spec_mod.resolve_world({}, manual, NETWORK, ctx["n"], profile2,
                                b["partition"])
    assert h1["sim_hash"] != h2["sim_hash"]           # manual: it IS read
    assert h1["flat"]["n_flow_sensors"] == sum(
        len(v["flow"]) for v in b["profile"]["sensors"].values())


def test_validator_checks_the_placement_block(ctx):
    import copy
    for mutate, match in (
            (lambda p: p["sensor_placement"]["probe"]["transitions"].pop(
                "mixed"), "transitions"),
            (lambda p: p["sensor_placement"]["probe"]["transitions"].update(
                industrial="industrial"), "transitions"),
            (lambda p: p["sensor_placement"].update(source="auto"), "source"),
            (lambda p: p["sensor_placement"]["selection_coupling"].update(
                variant="partial"), "selection_coupling"),
            (lambda p: p["sensor_placement"]["slots"].update(
                acoustic={"pure": 1, "mixed": 0}), "slots")):
        base = copy.deepcopy(BASE)
        mutate(base)
        with pytest.raises(ValueError, match=match):
            spec_mod.validate_world(ctx["world"](base=base), ctx["districts"])


def test_resolve_run_promotes_keys_and_hashes_fl_identity():
    r0 = spec_mod.resolve_run({}, BASE)
    assert "seed" not in r0["fl"]["training"]           # base behaviour kept
    r1 = spec_mod.resolve_run({"fl_seed": 0, "step_size": 2,
                               "batch_size": 128}, BASE)
    assert r1["fl"]["training"]["seed"] == 0            # 0 is a valid seed
    assert r1["fl"]["preprocessing"]["step_size"] == 2
    assert r1["run_hash"] != r0["run_hash"]
    assert spec_mod.resolve_run(
        {"fl_seed": 0, "batch_size": 128, "step_size": 2},
        BASE)["run_hash"] == r1["run_hash"]             # order-stable
    r2 = spec_mod.resolve_run(
        {"fl_overrides": {"drift": {"persistence": 2}}}, BASE)
    assert r2["fl"]["drift"]["persistence"] == 2
    assert r2["fl"]["drift"]["k_sigma"] == BASE["fl"]["drift"]["k_sigma"]
    with pytest.raises(ValueError):
        spec_mod.resolve_run({"stride": 2}, BASE)       # unknown key


def test_validator_income_or_land_use_rule(ctx):
    """Drift must change at least ONE of (income, land_use) vs the map's
    initial state for the target district — either alone suffices."""
    D = ctx["districts"]

    def world(**drift):
        return ctx["world"]({"drift": {**drift}})

    # District_A starts (low, residential) on LR_LR_LR_LR
    with pytest.raises(ValueError, match="no-op"):
        spec_mod.validate_world(
            world(to_income="low", to_land_use="residential"), D)
    spec_mod.validate_world(
        world(to_income="high", to_land_use="residential"), D)  # income alone
    spec_mod.validate_world(
        world(to_income="low", to_land_use="commercial"), D)    # land use alone
    spec_mod.validate_world(
        world(to_income="high", to_land_use="industrial"), D)   # both
    with pytest.raises(ValueError, match="land_use.mix"):
        spec_mod.validate_world(
            world(to_income="low", to_land_use="agricultural"), D)


def test_map_codec_roundtrips_and_rejects_bad_tokens():
    """The two token positions draw on DIFFERENT alphabets; 'M' is medium
    income in position 0 and mixed land use in position 1."""
    code = "LR_MM_HC_LI_LM"
    decoded = spec_mod.decode_map(code, 5)
    assert decoded[1] == ["medium", "mixed"]
    assert spec_mod.encode_map(decoded) == code
    for bad in ("LR_LM_LC_LR", "LX_LM_LC_LR_LR", "LL_LM_LC_LR_LR",
                "L_LM_LC_LR_LR"):
        with pytest.raises(ValueError):
            spec_mod.decode_map(bad, 5)


def test_validator_rejects_zero_warmup(ctx):
    with pytest.raises(ValueError, match="warmup_months"):
        spec_mod.validate_world(ctx["world"](
            {"drift": {"warmup_months": 0}}), ctx["districts"])


def test_validator_rejects_bad_targets_and_horizons(ctx):
    D = ctx["districts"]
    foreign = str(D["districts"]["District_B"][0])
    bad_node = ctx["world"](
        {"drift": {"tgt_district": "District_A", "seed_node": foreign}})
    with pytest.raises(ValueError, match="not a node"):
        spec_mod.validate_world(bad_node, D)            # belongs to B
    with pytest.raises(ValueError, match="tgt_district"):
        spec_mod.validate_world(ctx["world"](
            {"drift": {"tgt_district": "District_X"}}), D)
    with pytest.raises(ValueError, match="n_months"):
        spec_mod.validate_world(ctx["world"]({"n_months": 3}), D)  # warmup 2


def test_auto_seed_node_skips_zero_demand_trunk_junctions(ctx):
    inp, D = ctx["inp"], ctx["districts"]
    demands = spec_mod._inp_base_demands(inp)
    for d in D["districts"]:
        node = spec_mod.auto_seed_node(d, D, inp)
        assert node in [str(n) for n in D["districts"][d]]
        assert demands[node] > 0
        assert node == spec_mod.auto_seed_node(d, D, inp)


def test_seed_source_is_the_profile_for_manual_only(ctx, built_partitions):
    """KY7's profile seeds District_A at J-507. That id belongs to the MANUAL
    cut; a generated partition must fall through to the auto-picker."""
    from fedwater.networks import partitions as pstore
    profile = ctx["bundle"]["profile"]
    assert profile["drift_seed_nodes"]["District_A"] == "J-507"
    assert pstore.seed_source("manual", profile) == {"District_A": "J-507"}
    assert pstore.seed_source("fast_greedy", profile) == {}
    w = ctx["world"]()
    node = spec_mod.nprofile.resolve_seed_node(
        None, ctx["bundle"]["partition"], "District_A", lambda: "AUTO")
    assert node == "J-507" and w["flat"]["drift_seed_node"] is None


def test_expand_study_resolves_the_declared_studies(built_partitions):
    d0 = spec_mod.expand_study("d0_replication", Path.cwd())
    assert len(d0["worlds"]) == 2 and len(d0["runs"]) == 40
    assert d0["partitions"] == [("graeme", "manual")]
    variants = {w["flat"]["variant"] for w in d0["worlds"]}
    assert variants == {"baseline", "isolated"}
    seeds = {r["fl"]["training"]["seed"] for r in d0["runs"]}
    assert seeds == set(range(20))
    graeme = spec_mod.bundle(ROOT, "graeme", "manual")["districts"]
    sweep = spec_mod.expand_study("drift_origin_sweep", Path.cwd())
    assert len(sweep["worlds"]) == 5
    for w in sweep["worlds"]:                # auto seed nodes: valid, demand>0
        d, n = w["flat"]["drift_district"], w["flat"]["drift_seed_node"]
        assert n in [str(x) for x in graeme["districts"][d]]

    betas = spec_mod.expand_study("beta_sweep", Path.cwd())
    assert len(betas["worlds"]) == 5 and len(betas["runs"]) == 3
    assert {w["flat"]["beta"] for w in betas["worlds"]} == {0.0, 0.25, 0.5, 0.75, 1.0}

    ky7 = spec_mod.expand_study("ky7_component_probe", Path.cwd())
    assert ky7["worlds"][0]["flat"]["drift_seed_node"] == "J-507"
    assert ky7["worlds"][0]["flat"]["anchor_scale"] == 0.5


def test_study_partitions_reads_the_districting_axis():
    pairs = spec_mod.study_partitions("ky7_districting_probe", Path.cwd())
    assert pairs == [("ky7", m) for m in
                     ("fast_greedy", "girvan_newman", "manual", "spectral")]


def test_effective_fl_seed_fallback_and_zero():
    assert effective_fl_seed({"training": {}}, 42) == 42
    assert effective_fl_seed({"training": {"seed": 0}}, 42) == 0
    assert effective_fl_seed({"training": {"seed": 7}}, 42) == 7


# ------------------------------------------------------------- harvest ----
def _fake_run_data(tmp_path: Path) -> Path:
    """A minimal data tree with schemas matching the real reports, built so
    the correct answers are known by construction."""
    data = tmp_path / "data"
    (data / "08_reporting").mkdir(parents=True, exist_ok=True)
    (data / "03_primary").mkdir(exist_ok=True)
    (data / "07_model_output").mkdir(exist_ok=True)
    pd.DataFrame({"check": ["V1_mass_balance"], "value": [0.0],
                  "hard": [True], "passed": [True]}).to_csv(
        data / "08_reporting/validation_report.csv", index=False)
    pd.DataFrame({
        "client": ["District_D", "District_E"], "is_drifted": [True, False],
        "mean_delta_post": [0.9, 0.5], "threshold": [0.7, 0.7],
        "detection_month": [12, pd.NA], "rank": [1, 2],
        "false_alarms": [0, pd.NA], "separation_ratio": [1.8, pd.NA],
        "detection_delay_months": [3, pd.NA]}).to_csv(
        data / "08_reporting/drift_report.csv", index=False)
    ladder = pd.DataFrame({
        "corrector": ["C0_uncorrected"] * 2, "client": ["District_D", "District_E"],
        "is_drifted": [True, False], "rank": [1, 2],
        "separation_ratio": [1.8, pd.NA]})
    ladder.to_csv(data / "08_reporting/corrector_ladder_report.csv", index=False)
    pd.DataFrame({"iteration": [0, 0, 1, 1],
                  "client": ["District_D", "District_E"] * 2,
                  "weight": [0.5, 0.5, 0.3, 0.6],
                  "gain": [1.0] * 4, "sparse_score": [0.2, 0.1, 0.7, 0.2]}
                 ).to_csv(data / "08_reporting/loop_diagnostics.csv", index=False)
    pd.DataFrame({"node": ["2"], "district": ["District_D"],
                  "drift_month": [2], "to_income": ["high"],
                  "to_land_use": ["commercial"]}).to_csv(
        data / "03_primary/gt_drift_schedule.csv", index=False)
    pd.DataFrame({"kind": ["flow"], "tier": [1], "method": ["corr"],
                  "district_a": ["District_A"], "district_b": ["District_B"],
                  "statistic": [0.5], "direction": ["sym"],
                  "p_value": [0.01]}).to_csv(
        data / "03_primary/gt_dependence_battery.csv", index=False)
    # 3 districts -> 3 pairs; boundary pairs (A,B) and (B,C); statistics
    # rank boundaries strictly above the non-boundary pair => AUROC 1, and
    # margin = 0.30 - 0.20 = 0.10 by construction.
    pd.DataFrame({"district_a": ["District_A", "District_A", "District_B"],
                  "district_b": ["District_B", "District_C", "District_C"],
                  "boundary_pipes": [1, 0, 2],
                  "open_boundary_pipes": [1, 0, 2],
                  "hydraulic_distance_m": [100.0, 300.0, 150.0]}).to_csv(
        data / "03_primary/gt_topology.csv", index=False)
    pd.DataFrame({"level": ["T"] * 3, "kind": ["aer_latent"] * 3,
                  "district_a": ["District_A", "District_A", "District_B"],
                  "district_b": ["District_B", "District_C", "District_C"],
                  "comm_floats_per_client": [10] * 3,
                  "method": ["partial_rv"] * 3,
                  "statistic": [0.40, 0.20, 0.30],
                  "p_value": [0.01, 0.20, 0.01], "extra": [None] * 3,
                  "q_value": [0.03, 0.20, 0.03]}).to_csv(
        data / "08_reporting/client_dependence.csv", index=False)
    # Same method under BOTH kinds: the mapped (level, method) merge must
    # attach only the aer_latent row to level T — and never duplicate rows.
    pd.DataFrame({"evaluation": ["structure_recovery"] * 2,
                  "kind": ["aer_latent", "prototype_displacement"],
                  "tier": [6, 5], "method": ["partial_rv"] * 2,
                  "auroc": [1.0, 0.5],
                  "spearman_vs_proximity": [0.7, -0.1],
                  "n_pairs": [3, 3]}).to_csv(
        data / "08_reporting/dependence_evaluation.csv", index=False)
    return data


def test_harvest_groups_extract_known_answers(tmp_path):
    data = _fake_run_data(tmp_path)
    dep = harvest_mod.harvest_dependence_summary(data)
    assert len(dep) == 1                    # mapped merge: no duplication
    row = dep.set_index(["level", "method"]).loc[("T", "partial_rv")]
    assert row["auroc"] == 1.0
    assert row["n_sig_q05"] == 2
    assert abs(row["margin"] - 0.10) < 1e-12
    assert row["spearman_vs_proximity"] == 0.7
    c4 = harvest_mod.harvest_c4(data)
    assert set(c4["iteration"]) == {1}                  # final iteration only
    fingered = c4[c4["is_fingered"]]
    assert list(fingered["client"]) == ["District_D"]
    assert bool(fingered["is_origin"].iloc[0])
    assert harvest_mod.resolve_groups(("dependence", "drift")) \
        == ["dependence_pairs", "dependence_summary", "drift"]
    with pytest.raises(KeyError):
        harvest_mod.resolve_groups(("nope",))


# -------------------------------------------------- engine (stub kedro) ----
def _stub_kedro(behaviour):
    """A fake ExperimentEngine._kedro: `behaviour(cwd, pipeline)` returns
    (returncode, stdout) and creates whatever artifacts the stage 'made'."""
    def _fake(self, cwd, pipeline=None):
        rc, out = behaviour(Path(cwd), pipeline)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr=""), 0.1
    return _fake


def _sim_artifacts(cwd: Path):
    _fake_run_data(cwd)         # reuse the same known-answer tree
    pd.DataFrame({"district": ["District_A"], "kind": ["flow"],
                  "slot": ["q_pure_0"], "sensor": ["q_P-1"],
                  "element": ["P-1"]}).to_csv(
        cwd / "data/03_primary/sensor_placement.csv", index=False)
    (cwd / "data/07_model_output/clients").mkdir(parents=True, exist_ok=True)
    (cwd / "data/07_model_output/clients/District_A.csv").write_text("t,v\n0,1\n")


def test_engine_caches_by_content_hash_and_records_failures(tmp_path,
                                                            monkeypatch, ctx):
    calls = []

    def behaviour(cwd, pipeline):
        calls.append(pipeline)
        _sim_artifacts(cwd)     # world build AND run stage: harvest reads
        return 0, "ok"          # the run clone, so both must have artifacts

    monkeypatch.setattr(ExperimentEngine, "_kedro", _stub_kedro(behaviour))
    engine = ExperimentEngine(Path.cwd(), root=tmp_path / "exp")
    world = ctx["world"]({"sim_seed": 7})
    run = spec_mod.resolve_run({"fl_seed": 1}, BASE, pipelines=("fl",))

    assert engine.ensure_world(world)["status"] == "ok"
    assert engine.ensure_world(world)["cached"] is True  # content-hash hit
    r = engine.ensure_run("study", world, run,
                          harvest=("validation", "dependence", "c4"))
    assert r["status"] == "ok"
    assert engine.ensure_run("study", world, run)["cached"] is True
    assert calls == [None, "fl"]                         # one sim, one run
    globals_ = yaml.safe_load(
        (tmp_path / "exp/worlds" / world["sim_hash"] / "clone/conf/local/"
         "globals.yml").read_text())
    assert globals_["probe_store"] == str(tmp_path / "exp" / "probes")
    assert globals_["districting"] == {"active": "manual",
                                       "methods": ["manual"]}

    manifest = json.loads((r["dir"] / "manifest.json").read_text())
    assert manifest["sim_hash"] == world["sim_hash"]
    assert manifest["run_hash"] == run["run_hash"]
    assert manifest["fl_effective"]["training"]["seed"] == 1
    assert "python" in manifest["versions"]
    assert not (r["dir"] / "clone").exists()             # clone cleaned up
    index = engine.collect("study")
    assert len(index) == 1 and index.loc[0, "status"] == "ok"
    assert index.loc[0, "partial_rv_auroc_T"] == 1.0
    assert index.loc[0, "c4_fingered_client"] == "District_D"

    # a different run spec is a different identity — no stale reuse
    run2 = spec_mod.resolve_run({"fl_seed": 2}, BASE, pipelines=("fl",))
    assert engine.ensure_run("study", world, run2)["cached"] is False


def test_engine_records_validation_failure_and_downstream_runs_skip(
        tmp_path, monkeypatch, ctx):
    monkeypatch.setattr(
        ExperimentEngine, "_kedro",
        _stub_kedro(lambda cwd, pl: (1, "AssertionError: V3_pressure floor")))
    engine = ExperimentEngine(Path.cwd(), root=tmp_path / "exp")
    world = ctx["world"]({"sim_seed": 9})
    built = engine.ensure_world(world)
    assert built["status"] == "sim_validation_failed"    # policy (a): recorded
    manifest = json.loads((built["dir"] / "manifest.json").read_text())
    assert "V3" in manifest["stdout_tail"]
    run = spec_mod.resolve_run({}, BASE, pipelines=("fl",))
    r = engine.ensure_run("study", world, run)
    assert r["status"] == "world_sim_validation_failed"  # skipped, not crashed


def test_engine_refuses_ok_status_when_world_artifacts_missing(tmp_path,
                                                               monkeypatch, ctx):
    """Exit code 0 with missing required artifacts must be recorded as
    sim_incomplete, and downstream runs must skip — the D0 lesson: 40 runs
    paid 13 minutes each for a battery a millisecond stat() would have
    caught."""
    monkeypatch.setattr(ExperimentEngine, "_kedro",
                        _stub_kedro(lambda cwd, pl: (0, "ok")))  # writes nothing
    engine = ExperimentEngine(Path.cwd(), root=tmp_path / "exp")
    world = ctx["world"]({"sim_seed": 13})
    built = engine.ensure_world(world)
    assert built["status"].startswith("sim_incomplete:")
    assert "gt_dependence_battery.csv" in built["status"]
    assert "sensor_placement.csv" in built["status"]
    run = spec_mod.resolve_run({}, BASE, pipelines=("fl",))
    r = engine.ensure_run("study", world, run)
    assert r["status"].startswith("world_sim_incomplete:")
    # and a later rebuild attempt is NOT blocked by the failed manifest
    assert engine.ensure_world(world)["cached"] is False


def test_no_builtin_hash_seeding_in_pipelines():
    """Tripwire: ``hash()`` of strings is randomized per process, so seeding
    RNGs with it makes results irreproducible across runs. This bit us twice
    (personalization fixture, dependence surrogates) — keep it out."""
    import re
    # the BUILTIN only: `stable_hash((...))` is the sanctioned replacement
    builtin = re.compile(r"(?<![\w.])hash\(\(")
    root = ROOT / "src" / "fedwater"
    offenders = [p for p in root.rglob("*.py") if builtin.search(p.read_text())]
    assert not offenders, f"builtin hash() used for seeding in: {offenders}"


@pytest.mark.xfail(strict=True, reason=(
    "label_factory predates network bundles: it calls resolve_world without "
    "a profile, so network-scoped nulls (anchor_scale) cannot resolve. Out of "
    "scope for the dynamic-input refactor; strict so a fix is noticed."))
def test_label_factory_assembles_from_engine_harvests(tmp_path, monkeypatch):
    from fedwater.pipelines.label_factory import nodes as factory

    def fake_world(self, world):
        return {"status": "ok", "dir": tmp_path, "cached": True,
                "sim_hash": world["sim_hash"]}

    def fake_run(self, study, world, run, harvest=(), retain=(),
                 keep_clone=False):
        rdir = tmp_path / f"{world['sim_hash']}__{run['run_hash']}"
        (rdir / "harvest").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"district_a": ["District_A"], "district_b": ["District_B"],
                      "label_connected": [1]}).to_parquet(
            rdir / "harvest/pairs.parquet")
        pd.DataFrame({"client": [f"District_{c}" for c in "ABCDE"],
                      "label_is_origin": [0, 0, 0, 1, 0],
                      "label_onset_month": [2] * 5}).to_parquet(
            rdir / "harvest/clients.parquet")
        return {"status": "ok", "dir": rdir, "cached": False, "id": "x"}

    monkeypatch.setattr(ExperimentEngine, "ensure_world", fake_world)
    monkeypatch.setattr(ExperimentEngine, "ensure_run", fake_run)
    fl = yaml.safe_load(Path("conf/base/parameters.yml").read_text())["fl"]
    fl["label_factory"]["scratch_dir"] = str(tmp_path / "scratch")
    # No district on the base map is industrial, so no draw can produce a
    # no-op drift — generate_labeled_worlds now validates before simulating.
    fl["label_factory"]["drift_land_uses"] = ["industrial"]
    # label_factory is Graeme-only (it hardcodes five districts)
    graeme = yaml.safe_load((ROOT / "data/01_raw/graeme/districts.yml").read_text())
    specs = factory.build_world_specs(
        fl, graeme, seed=42).head(2)
    pairs, clients = factory.generate_labeled_worlds(specs, fl)
    assert set(pairs["world"]) == {0, 1}
    assert {"variant", "close_fraction", "label_connected"} <= set(pairs.columns)
    assert clients.groupby("world")["label_is_origin"].sum().eq(1).all()


# ----------------------------------------------------------- integration ----
@pytest.mark.integration
def test_micro_world_through_real_pipelines(tmp_path, built_partitions):
    """A 4-month, 2-round KY7 world end-to-end through the real engine, with
    dynamic sensor placement (its probe stack is built once and shared):
    cache hit on the second call, manifests + harvests on disk. Skipped when
    the heavy dependencies are absent."""
    pytest.importorskip("torch")
    pytest.importorskip("wntr")
    pytest.importorskip("kedro")

    b = spec_mod.bundle(ROOT, NETWORK, "manual")
    world = spec_mod.resolve_world(
        {"sim_seed": 11, "n_months": 4,
         "drift": {"tgt_district": "District_A",
                   "to_income": "low", "to_land_use": "commercial"},
         "oracle": {"tiers": [1], "n_surrogates": 5,
                    "n_surrogates_expensive": 3}},
        BASE, NETWORK, len(b["districts"]["districts"]), b["profile"],
        b["partition"])
    spec_mod.validate_world(world, b["districts"])
    run = spec_mod.resolve_run(
        {"fl_seed": 1, "step_size": 6, "batch_size": 128, "rounds": 2,
         "n_surrogates": 8, "n_surrogates_expensive": 4,
         "fl_overrides": {"drift": {"reference_months": 2, "persistence": 2},
                          "attribution": {"reference_months": 2}}},
        BASE)
    engine = ExperimentEngine(Path.cwd(), root=tmp_path / "exp")
    built = engine.ensure_world(world)
    assert built["status"] == "ok", \
        (json.loads((built["dir"] / "manifest.json").read_text())
         .get("stdout_tail") or "")[-1500:]
    placement = pd.read_csv(built["dir"] / "clone/data/03_primary/"
                            "sensor_placement.csv")
    assert placement.groupby("district").size().nunique() == 1
    executed = engine.ensure_run(
        "micro", world, run,
        harvest=("validation", "drift", "ladder", "c4", "dependence",
                 "pairs", "clients"),
        retain=("prototype_history",))
    assert executed["status"] == "ok", \
        (json.loads((executed["dir"] / "manifest.json").read_text())
         .get("stdout_tail") or "")[-1500:]
    assert engine.ensure_run("micro", world, run)["cached"] is True
    assert (executed["dir"] / "retained/prototype_history.parquet").exists()
    index = engine.collect("micro")
    assert index.loc[0, "status"] == "ok"
    assert 0.0 <= index.loc[0, "partial_rv_auroc_T"] <= 1.0
