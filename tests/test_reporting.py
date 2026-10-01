"""The atlas on a REAL experiments tree: Graeme micro worlds built by the
project's own pipelines in the engine's layout (``_experiments_tree``)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

from fedwater.reporting import Atlas, build_index, world_catalog

import _experiments_tree as T

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tree(tmp_path_factory, built_partitions):
    import conftest
    conftest._write_partitions(T.NETWORK, ("manual", "spectral"))
    return T.build(tmp_path_factory.mktemp("experiments"))


@pytest.fixture()
def atlas(tree):
    return Atlas(root=tree, project=ROOT)


def test_index_is_one_row_per_study_and_world(atlas):
    idx = atlas.index
    assert set(idx["study"]) == {"demo_study", "other_study"}
    # 3 distinct worlds; the manual/District_A world belongs to both studies
    assert idx["sim_hash"].nunique() == 3 and len(idx) == 4
    shared = idx.groupby("sim_hash")["study"].nunique()
    assert (shared == 2).sum() == 1
    assert (idx["status"] == "ok").all()
    assert atlas.partitions("demo_study") == [(T.NETWORK, "manual"),
                                              (T.NETWORK, "spectral")]
    # the index reads the stack pointer of every world
    assert idx["selection_stack"].notna().all()


def test_worlds_on_one_partition_share_one_stack_object(atlas):
    manual = atlas.worlds("demo_study", T.NETWORK, "manual", ok_only=True)
    assert manual["selection_stack"].nunique() == 1 and len(manual) == 2
    a, b = (atlas.world(h) for h in manual["sim_hash"])
    s = a.selection(with_probes=True)
    assert b.selection() is s and b.selection(with_probes=True) is s
    assert len(s.probes) == len(s.names)
    spectral = atlas.worlds("demo_study", T.NETWORK, "spectral")["sim_hash"][0]
    assert atlas.world(spectral).selection() is not s
    assert set(atlas.study_stacks("demo_study")) == {
        f"manual/selection:{s.stack_hash}",
        f"spectral/selection:{atlas.world(spectral).row['selection_stack']}"}


def test_world_reads_its_clone_without_a_session(atlas):
    w = atlas.world(atlas.index["sim_hash"][0])
    pl = w.load("sensor_placement")
    assert {"district", "kind", "slot", "sensor"} <= set(pl.columns)
    assert w.load("sensor_placement") is pl                      # read once
    assert w.params["sensor_placement"]["source"] == "dynamic"
    assert w.effective["seed"] == 43
    # the catalog is anchored at the clone, whatever the working directory
    cat, _, g = world_catalog(w.clone)
    assert g["network"] == T.NETWORK
    assert Path(str(cat.get("sensor_placement")._filepath)).is_relative_to(
        w.clone.resolve())


def test_index_cache_is_reused_then_rebuilt_on_change(tree):
    a = Atlas(root=tree, project=ROOT)
    stamp = tree / "_atlas" / "signature.txt"
    before = stamp.stat().st_mtime_ns
    b = Atlas(root=tree, project=ROOT)
    assert stamp.stat().st_mtime_ns == before and b.index.equals(a.index)
    # a new study referencing an existing world: the index must pick it up
    h = a.index["sim_hash"][0]
    rdir = tree / "new_study" / "runs" / f"{h}__r"
    rdir.mkdir(parents=True)
    time.sleep(0.01)
    (rdir / "manifest.json").write_text(json.dumps({"sim_hash": h}))
    try:
        assert "new_study" in b.refresh()["study"].values
    finally:
        shutil.rmtree(tree / "new_study")
    assert "new_study" not in b.refresh()["study"].values


def test_partition_view_rebuilds_the_stored_partitions(atlas):
    P = atlas.partition(T.NETWORK)
    assert set(P.results) == {"manual", "spectral"}
    rep = P.reports
    assert set(rep["summary"]["method"]) == {"manual", "spectral"}
    assert rep["sanity"].groupby("method")["ok"].count().gt(0).all()
    st = P.store.set_index("method")
    assert st.loc["manual", "status"] == "current"


def test_selector_cascades_and_rerenders_bound_sections(atlas):
    sel = atlas.selector(study="demo_study")
    seen = {"world": [], "study": []}
    sel.show(lambda w: seen["world"].append(w.sim_hash))
    sel.show(lambda a, s: seen["study"].append(s), level="study")
    assert seen["study"] == ["demo_study"] and len(seen["world"]) == 1
    sel.select(partition=(T.NETWORK, "spectral"))
    spectral = atlas.worlds("demo_study", T.NETWORK, "spectral")["sim_hash"][0]
    assert sel.world.sim_hash == spectral and seen["world"][-1] == spectral
    sel.select(study="other_study")
    assert seen["study"][-1] == "other_study"
    assert [v for _, v in sel.w_part.options] == [(T.NETWORK, "manual")]
    # a failing section is contained, not raised
    sel.show(lambda w: 1 / 0)


def test_labels_say_what_each_world_is(atlas):
    idx = atlas.index
    row = idx[(idx["districting"] == "manual")
              & (idx["drift_district"] == "District_B")].iloc[0]
    init = row["initial_map"].split("_")
    final = row["final_map"].split("_")
    # only the drift district's token changes, to the drift target
    assert [i for i, (a, b) in enumerate(zip(init, final)) if a != b] == [1]
    assert final[1] == "LC" and row["drift_change"] == \
        "low residential -> low commercial"
    sel = atlas.selector(study="demo_study")
    labels = [l for l, _ in sel.w_world.options]
    assert any(f"{row['initial_map']} -> {row['final_map']}" in l
               and "drift B" in l for l in labels)
    parts = [l for l, _ in sel.w_part.options]
    assert all("k=5" in l and "partition" in l for l in parts)
    assert "drift" in sel.w_card.value and "&rarr;" in sel.w_card.value


def test_build_index_on_an_empty_root(tmp_path):
    assert build_index(tmp_path).empty
    assert Atlas(root=tmp_path / "missing", project=ROOT).index.empty
