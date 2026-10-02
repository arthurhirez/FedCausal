"""The magnitude mixture and analysis-only re-use, on a real Graeme stack.

One stack is simulated (shortened probe horizon, as the stack integration
test) with the OLD rules (norm gate, positive mixture); every assertion is
then made on its own stored numbers: the rule each simplex follows, cell by
cell, the direction columns, and that a gate/mixture change re-analyses the
SAME probe worlds instead of simulating new ones.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from fedwater.config import load_base_params
from fedwater.hashing import canonical_hash
from fedwater.networks import partitions as pstore
from fedwater.networks.profile import resolve_params
from fedwater.placement import excite, store
from fedwater.placement import mixture_probe as mp

ROOT = Path(__file__).resolve().parents[1]
BASE = load_base_params(ROOT)
pytestmark = pytest.mark.integration
OLD = {"gate": "norm", "mixture": "positive"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory, built_partitions):
    import wntr
    net = "graeme"
    raw = ROOT / "data/01_raw" / net
    wn = wntr.network.WaterNetworkModel(str(raw / "network.inp"))
    profile = yaml.safe_load((raw / "profile.yml").read_text())
    part = pstore.read_partition(ROOT, net, "manual")
    meta = pstore.partition_meta(part["manifest"], profile)
    params, _ = resolve_params(BASE, profile)
    sp = copy.deepcopy(params["sensor_placement"])
    sp["probe"].update(warmup_months=4, settled_months=5, days_per_month=7)
    world = {k: params[k] for k in ("hydraulics", "scenario", "validation",
                                    "time", "land_use", "buildings",
                                    "patterns")}
    plan = excite.horizon_plan(wn, part["districts"], meta, sp["probe"], world)
    transitions = excite.validate_transitions(sp["probe"]["transitions"],
                                              params["land_use"])
    inputs = {"network_inp": wn, "network_profile": profile,
              "districts": part["districts"],
              "partition_manifest": part["manifest"]}

    def build(classes, root):
        spec = store.stack_spec(network=net, partition=meta, world=world,
                                probe=sp["probe"], classes=classes, plan=plan,
                                coupling={"variant": "baseline"}, seed=42)
        return store.ensure_stack(
            store_root=root, spec=spec, inputs=inputs, world=world,
            probe=sp["probe"], classes=classes, plan=plan,
            transitions=transitions, coupling={"variant": "baseline"},
            seed=42, verbose=False)

    root = tmp_path_factory.mktemp("store")
    old = build({**sp["classes"], **OLD}, root)
    return {"build": build, "root": root, "old": old, "classes": sp["classes"]}


def test_defaults_are_projection_and_magnitude():
    c = BASE["sensor_placement"]["classes"]
    assert store.gate(c) == "projection" and store.mixture_rule(c) == "magnitude"
    assert store.mixture_rule({}) == "positive"          # absent == old rule
    with pytest.raises(ValueError, match="mixture"):
        store.mixture_rule({"mixture": "signed"})


def test_each_rule_builds_its_simplex_cell_by_cell(setup):
    base = setup["old"].base
    f = float(setup["classes"]["null_factor"])
    G = base["G"].to_numpy()
    live = base["DZ"].to_numpy() > f * base["null"].to_numpy()[:, None]
    names = base["districts"]
    W = [f"w_{d}" for d in names]
    for rule, cell in (("positive", np.clip(G, 0, None)),
                       ("magnitude", np.abs(G))):
        mix = mp.remix(base, null_factor=f, mixture=rule)["mixture"] \
            .set_index("id").loc[base["G"].index]
        V = np.where(live, cell, 0.0)
        tot = V.sum(axis=1)
        want = np.divide(V, tot[:, None], out=np.zeros_like(V),
                         where=tot[:, None] > 1e-12)
        assert np.allclose(mix[W].to_numpy(), want)
        ph = mix[[f"phase_{d}" for d in names]].to_numpy()
        assert np.array_equal(ph, np.where(live & (np.abs(G) > 1e-12),
                                           np.sign(G), 0.0))
        neg = -np.where(live, np.clip(G, None, 0), 0.0).sum(axis=1)
        absm = np.where(live, np.abs(G), 0.0).sum(axis=1)
        share = np.divide(neg, absm, out=np.zeros_like(neg), where=absm > 1e-12)
        assert np.allclose(mix["anti_phase_share"].to_numpy(), share)
    # the old rule is untouched by the change
    assert setup["old"].mixture.equals(
        mp.remix(base, null_factor=f, mixture="positive")["mixture"])


def test_magnitude_ends_foreign_by_anti_phase(setup):
    """A gauge whose own-district response is anti-phase and dominant is
    'foreign' under positive, and reads its own district under magnitude."""
    base = setup["old"].base
    f = float(setup["classes"]["null_factor"])
    pos = mp.remix(base, null_factor=f, mixture="positive")["mixture"].set_index("id")
    mag = mp.remix(base, null_factor=f, mixture="magnitude")["mixture"].set_index("id")
    G = base["G"]
    own = np.array([G.loc[i, h] for i, h in zip(pos.index, pos["home"])])
    others = np.array([np.abs(G.loc[i].drop(h)).max()
                       for i, h in zip(pos.index, pos["home"])])
    flip = (pos["tier"] == "foreign").to_numpy() & (own < 0) & \
        (np.abs(own) > others) & (mag["n_live"].to_numpy() > 0)
    assert flip.any(), "the stack has no anti-phase-dominant foreign gauge"
    m = mag[flip]
    assert (m["purity"] > 0.5).all() and (m["home_phase"] == -1).all()
    assert (m["anti_phase_share"] > 0.5).all()


def test_analysis_change_reuses_the_probe_worlds(setup, monkeypatch):
    def no_sim(*a, **k):
        raise AssertionError("re-simulated a probe world for an analysis change")
    monkeypatch.setattr(excite, "simulate_probe", no_sim)
    new_classes = copy.deepcopy(setup["classes"])        # projection+magnitude
    st = setup["build"](new_classes, setup["root"])
    old = setup["old"]
    assert st.stack_hash != old.stack_hash
    assert canonical_hash(store.physics(st.spec)) == \
        canonical_hash(store.physics(old.spec))
    mf = store._read_json(st.path / "manifest.json")
    assert mf["probes_from"] == old.path.name and not (st.path / "worlds").exists()
    assert st.base["gate"] == "projection" and st.base["mixture_rule"] == "magnitude"
    # its mixture is exactly the re-analysis of the donor's worlds
    f = float(new_classes["null_factor"])
    again = mp.remix(old.base, null_factor=f, gate="projection",
                     mixture="magnitude")["mixture"]
    cols = [c for c in again.columns if c.startswith(("w_", "phase_"))]
    a = st.mixture.set_index("id").loc[again["id"], cols].to_numpy(float)
    assert np.allclose(a, again[cols].to_numpy(float))
    # and it reloads, probes included, through the pointer
    re = store.load_stack(st.path, with_probes=True)
    assert set(re.probes) == set(old.probes)
    # a stack that BORROWS worlds is never itself a donor
    assert store._physics_donor(st.path.parent, st.spec,
                                list(old.probes)) == old.path
