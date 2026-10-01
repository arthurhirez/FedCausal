"""EPANET scratch handling in ``run_hydraulics`` on a real Graeme world:
cleanup that never fails a world nor masks EPANET's error, a fail-fast space
check sized by the real results file, and a sweep of debris from dead runs."""
from __future__ import annotations

import copy
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from fedwater.config import load_base_params
from fedwater.networks import partitions as pstore
from fedwater.networks.profile import resolve_params
from fedwater.pipelines.hydraulics import nodes as H

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world(built_partitions):
    """wn_variant + demand_series of a real Graeme world (4 x 7-day months)."""
    import wntr
    from fedwater.placement import excite
    net = "graeme"
    raw = ROOT / "data/01_raw" / net
    profile = yaml.safe_load((raw / "profile.yml").read_text())
    part = pstore.read_partition(ROOT, net, "manual")
    params, _ = resolve_params(load_base_params(ROOT), profile)
    params = copy.deepcopy(params)
    params["time"].update(n_months=4, days_per_month=7)
    params["scenario"]["n_months"] = 4
    params["scenario"]["drift"].update(tgt_district="District_A",
                                       warmup_months=1, to_income="low",
                                       to_land_use="commercial")
    data = {"network_inp": wntr.network.WaterNetworkModel(str(raw / "network.inp")),
            "network_profile": profile, "districts": part["districts"],
            "partition_manifest": part["manifest"]}
    for k in ("hydraulics", "scenario", "validation", "time", "coupling",
              "seed", "land_use", "buildings", "patterns"):
        data[f"params:{k}"] = params[k]
    out = excite._run_inprocess(
        excite.core_pipeline(targets=("demand_series", "wn_variant")), data)
    return out["wn_variant"], out["demand_series"]


@pytest.fixture()
def scratch(tmp_path, monkeypatch):
    monkeypatch.setenv(H.SCRATCH_ENV, str(tmp_path))
    return tmp_path


def _left(root):
    return sorted(root.glob(f"{H.SCRATCH_PREFIX}*"))


def test_run_leaves_no_scratch_behind(world, scratch):
    wn, demand = world
    p, f, d = H.run_hydraulics(wn, demand)
    assert len(p) == len(demand) and len(f) == len(demand)
    assert _left(scratch) == []


def test_size_estimate_matches_the_real_results_file(world, scratch):
    import wntr
    wn, demand = world
    keep = scratch / "keep"
    keep.mkdir()
    w = copy.deepcopy(wn)
    w.options.time.duration = 14 * 24 * 3600
    w.options.time.report_timestep = 3600
    wntr.sim.EpanetSimulator(w).run_sim(file_prefix=str(keep / "run"))
    real = (keep / "run.bin").stat().st_size
    assert 0.98 <= real / H.scratch_bytes(w, 14 * 24) <= 1.05


def test_full_disk_fails_fast_with_its_own_message(world, scratch, monkeypatch):
    wn, demand = world
    real = shutil.disk_usage
    monkeypatch.setattr(H.shutil, "disk_usage",
                        lambda p: real(p)._replace(free=1_000))
    with pytest.raises(OSError, match=H.SCRATCH_ENV):
        H.run_hydraulics(wn, demand)
    assert _left(scratch) == []                     # EPANET never started


def test_a_locked_scratch_warns_instead_of_failing(world, scratch, monkeypatch):
    wn, demand = world

    def locked(path, *a, **k):
        raise PermissionError(32, "in use by another process", str(path))
    monkeypatch.setattr(H.shutil, "rmtree", locked)
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    with pytest.warns(UserWarning, match="could not remove scratch"):
        p, _, _ = H.run_hydraulics(wn, demand)
    assert len(p) == len(demand) and len(_left(scratch)) == 1


def test_sweep_removes_only_stale_scratch(scratch):
    old = scratch / f"{H.SCRATCH_PREFIX}old"
    new = scratch / f"{H.SCRATCH_PREFIX}new"
    other = scratch / "someone_else_old"
    for d in (old, new, other):
        d.mkdir()
        (d / "run.bin").write_bytes(b"x")
    past = time.time() - (H.STALE_HOURS + 1) * 3600
    for d in (old, other):
        os.utime(d / "run.bin", (past, past))
        os.utime(d, (past, past))
    assert H.sweep_stale(scratch) == [old]
    assert new.exists() and other.exists() and not old.exists()
