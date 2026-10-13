"""The exact layer (placement.exact, placement.epanet_loop) on a toy network.

Everything here is physics with a known answer: a branch carries exactly the demand downstream of
it, so its composition and its response to a drift are known in closed form; the blocked twin on a
run's own trajectory must reproduce that run; the no-drift city spliced from probe worlds must be
the demand nobody drifted in. The last test builds a two-world probe stack on disk and runs
``ensure_exact`` end to end (gates, tables, cache).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fedwater.placement import epanet_loop as el
from fedwater.placement import exact as ex

from _toy_network import DAYS_PER_MONTH, DISTRICTS, DRIFT_MONTH, fake_stack, toy_demand, toy_model

H = 24 * 14


# --------------------------------------------------------------------------------------------- topology
def test_link_topology_classes_parallel_pairs_and_isolated():
    T, down = ex.link_topology(toy_model(parallel=False, isolated=True))
    t = T.set_index("element")["topo"]
    assert t["P0"] == t["P6"] == t["PU"] == "source_bridge"
    assert t["P1"] == t["P2"] == t["P3"] == "loop"
    assert t["P4"] == t["P5"] == "branch"
    assert t["P9"] == "isolated"
    assert down["P4"] == {"J4", "J5"} and down["P5"] == {"J5"}
    assert T.set_index("element").at["P4", "down_sign"] == 1.0          # .inp start J3 -> end J4 (downstream)

    T2, down2 = ex.link_topology(toy_model(parallel=True))
    t2 = T2.set_index("element")["topo"]
    assert t2["P4"] == t2["P4b"] == "loop"                              # a parallel pair is never a bridge
    assert t2["P5"] == "branch" and "P4" not in down2


# --------------------------------------------------------------------------------------------- splice
def test_splice_is_exact_and_reports_donor_mismatch():
    D0 = toy_demand(H)
    worlds = [("District_A", toy_demand(H, {"District_A": 0.8}, start_month=1)),
              ("District_B", toy_demand(H, {"District_B": 0.8}, start_month=1))]
    S, rep = ex.splice_nodrift(iter(worlds), DISTRICTS)
    assert np.array_equal(S[list(D0.columns)].to_numpy(), D0.to_numpy())
    assert (rep["max_abs_diff"] == 0).all()

    three = worlds + [("District_A", toy_demand(H, {"District_A": 0.8, "District_B": 0.1}, start_month=1))]
    _, rep = ex.splice_nodrift(iter(three), DISTRICTS)
    assert rep.set_index("district").at["District_B", "max_abs_diff"] > 0   # B's donors disagree

    with pytest.raises(ValueError, match="no world"):
        ex.splice_nodrift(iter(worlds[:1]), DISTRICTS)


# --------------------------------------------------------------------------------------------- E1
@pytest.fixture(scope="module")
def nodrift_run():
    wn = toy_model()
    D = toy_demand(H)
    r = el.simulate(wn, D, H)
    assert el.run_ok(r), r["warnings"]
    return wn, D, r


def test_toy_run_exercises_the_pump_and_the_tank(nodrift_run):
    _, _, r = nodrift_run
    assert r["switches"].sum() >= 2                                     # the level controls fire
    assert np.ptp(r["tank_head"][:, 0]) > 1.0


def test_e1_closes_and_branch_composition_is_downstream_demand(nodrift_run):
    wn, D, r = nodrift_run
    links = r["links"]
    jcols = [c for c in D.columns if c != "month"]
    e1 = ex.E1Net(wn, links, jcols, DISTRICTS)
    out = e1.run(r["snap"].astype(float), D[e1.jcols].to_numpy(float), range(H), 24)
    assert out["row_dev"] <= 1e-9 and out["close_dev"] <= 1e-6
    C = out["day"].astype(float).sum(axis=0)
    m = ex._renorm(C, 2)
    i4, i5 = links.index("P4"), links.index("P5")
    tot = D[["J4", "J5"]].to_numpy(float).sum(axis=0)
    assert m[i4, 0] == pytest.approx(tot[0] / tot.sum(), abs=1e-6)       # J4 is A, J5 is B
    assert m[i5, 1] == pytest.approx(1.0, abs=1e-9)
    # no storage / reservoir water on a branch; "unaccounted" holds the float32 imbalance only
    assert ex._raw(C)[i4, 2:].sum() == pytest.approx(0.0, abs=1e-6)
    # the tank link carries filling water part of the time
    assert ex._raw(C)[links.index("P6"), 2] > 0


def test_branch_response_equals_downstream_demand_change(nodrift_run):
    wn, D0, r0 = nodrift_run
    D1 = toy_demand(H, {"District_B": 0.8}, start_month=1)
    r1 = el.simulate(wn, D1, H)
    assert el.run_ok(r1)
    lo = 24 * DAYS_PER_MONTH
    dQ = (r1["snap"] - r0["snap"]).astype(float)[lo:]
    dJ5 = (D1["J5"] - D0["J5"]).to_numpy(float)[lo:]
    for ln in ("P4", "P5"):                                             # J5 is downstream of both
        assert np.abs(dQ[:, r0["links"].index(ln)] - dJ5).max() <= 1e-5
    dD = (D1[DISTRICTS["District_B"]].sum(axis=1) - D0[DISTRICTS["District_B"]].sum(axis=1)).to_numpy(float)[lo:]
    g = ex._gain_w(dQ, dD, 24, lo)
    proj = float(ex._profile(dJ5, 24, lo)[0] @ ex._profile(dD, 24, lo)[0]) / float(
        (ex._profile(dD, 24, lo)[0] ** 2).sum())
    assert g[r0["links"].index("P5")] == pytest.approx(proj, abs=1e-5)


# --------------------------------------------------------------------------------------------- E2 twin
def test_twin_on_its_own_trajectory_reproduces_the_run(nodrift_run):
    wn, D, r = nodrift_run
    wb, dem, info = el.blocked_model(wn, D, r, H)
    assert info["ill_posed_hours"] == 0
    assert "T" in wb.reservoir_name_list and not wb.tank_name_list
    b = el.simulate(wb, dem, H)
    assert el.run_ok(b)
    B = b["snap"][:, [b["links"].index(l) for l in r["links"]]].astype(float)
    err = np.abs(B - r["snap"].astype(float))[1:]                        # hour 0 is the cold start
    assert err.max() <= 1e-3 * np.abs(r["snap"]).max()


def test_paired_effect_is_zero_without_change_and_exact_for_a_proportional_response():
    rng = np.random.default_rng(0)
    T = 24 * 14
    Q0 = 1.0 + rng.random((T, 3))
    dD = np.sin(np.arange(T) * 2 * np.pi / 24) + 0.3
    ef = ex.paired_effect(Q0, Q0, dD, 24, 0, np.ones(3))
    assert np.allclose(ef["g_w"], 0) and np.allclose(ef["S"], 0) and np.allclose(ef["amp"], 0)
    a = np.array([0.5, -2.0, 0.0])
    ef = ex.paired_effect(Q0 + np.outer(dD, a), Q0, dD, 24, 0, np.array([1.0, -1.0, 1.0]))
    assert np.allclose(ef["g_w"], [0.5, 2.0, 0.0])                        # oriented by the no-drift sign


# --------------------------------------------------------------------------------------------- ensure_exact
def test_ensure_exact_end_to_end_gates_tables_and_cache(tmp_path):
    path = fake_stack(tmp_path)
    X = ex.ensure_exact(path, verbose=False)
    G = X["gates"].set_index("gate")
    assert G["passed"].all(), G
    L, B = X["links"].set_index("element"), X["labels"]
    assert L.at["P5", "topo"] == "branch" and L.at["P5", "m_District_B"] == pytest.approx(1.0)
    assert bool(L.at["PU", "is_pump"]) and L.at["P5", "down_demand"] == pytest.approx(
        toy_demand(24 * DAYS_PER_MONTH * DRIFT_MONTH)["J5"].mean(), rel=1e-9)
    assert len(B) == len(L) * len(DISTRICTS)
    p5 = B[(B["element"] == "P5")].set_index("k")
    assert p5.at["District_B", "own"] and p5.at["District_B", "Nr"] == pytest.approx(1.0)
    assert p5.at["District_A", "g_w"] == pytest.approx(0.0, abs=1e-6)     # A's drift never reaches J5
    assert abs(p5.at["District_B", "sigma"]) <= 1e-3                      # a branch needs no hub
    assert {"dm_k", "dm_null_se", "dm_null_half", "dm_t_paired", "g_de0", "g_ie0"} <= set(B.columns)
    assert (X["path"] / "nodrift_flows.parquet").exists()
    t_first = json.loads((X["path"] / "manifest.json").read_text())["created_utc"]
    X2 = ex.ensure_exact(path, verbose=False)                              # cached: not recomputed
    assert json.loads((X2["path"] / "manifest.json").read_text())["created_utc"] == t_first
    assert X2["path"] == X["path"] == ex.exact_dir(path)


def test_sigma_is_undefined_below_the_gain_floor_and_the_floor_never_recomputes(tmp_path):
    s = ex.sigma_share([0.5, 0.0, 2e-4, 0.0], [0.5, 0.3, 4e-4, 0.0], 1e-3)
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(1.0)
    assert np.isnan(s[2]) and np.isnan(s[3])                               # a ratio of near-zero gains
    path = fake_stack(tmp_path)
    X = ex.ensure_exact(path, verbose=False)
    B = X["labels"]
    den = B["g_de0"].abs() + B["g_ie0"].abs()
    assert B.loc[den < ex.DEFAULTS["sigma_floor"], "sigma"].isna().all()
    assert B.loc[den >= ex.DEFAULTS["sigma_floor"], "sigma"].notna().all()
    big = float(den[den > 0].median())
    assert ex.exact_dir(path, {"sigma_floor": big}) == X["path"]          # outside the cache hash
    Xb = ex.load_exact(X["path"], {"sigma_floor": big})
    assert Xb["labels"].loc[den < big, "sigma"].isna().all()
    assert Xb["manifest"]["sigma_floor"] == big
    np.testing.assert_allclose(Xb["labels"].loc[den >= big, "sigma"], B.loc[den >= big, "sigma"])


def test_trajectory_twins_periods_months_and_cards(tmp_path):
    path = fake_stack(tmp_path)
    X = ex.ensure_exact(path, verbose=False)
    G = X["gates"].set_index("gate")
    for g in ("G-X0c", "G-X6d", "G-X6e"):                                  # E2 on the drift trajectories
        assert g in G.index and bool(G.at[g, "passed"]), G
    B = X["labels"]
    assert {"g_de1", "g_ie1", "interaction", "S_de0"} <= set(B.columns)
    br = B[B["element"] == "P5"].set_index("k")                            # a branch: demand transit only
    assert br.at["District_B", "S_de0"] == pytest.approx(br.at["District_B", "S"], abs=1e-6)
    assert br.at["District_B", "interaction"] == pytest.approx(0.0, abs=1e-3)
    LP = X["labels_period"]
    fin = LP[LP["period"] == "final"].merge(B[["element", "k", "g_w", "S"]], on=["element", "k"], suffixes=("", "_lab"))
    np.testing.assert_allclose(fin["S"], fin["S_lab"], atol=1e-12)          # final period = the labels' window
    np.testing.assert_allclose(fin["g_w"].fillna(0), fin["g_w_lab"].fillna(0), atol=1e-12)
    init = LP[LP["period"] == "init"]
    assert len(init) and (init["g_w"] == 0).all() and (init["S"] == 0).all()   # before the drift: exactly zero
    M = X["labels_monthly"]
    n_months = X["manifest"]["hours"] // X["manifest"]["steps_month"]
    assert len(M) == n_months * len(X["links"]) * len(DISTRICTS)
    assert M.loc[M["element"] == "PU", "on_drift"].notna().all()           # pump on-fraction per month
    C = ex.load_card(X["path"], "District_B")
    L, P_ = len(C["links"]), 7 * int(C["steps_day"])
    for run in ex.CARD_RUNS:
        assert C[f"prof__{run}"].shape == (L, P_) and C[f"week__{run}"].shape == (2, P_, L)
    assert C["how__drift"].shape == (L, P_, len(C["dests"]))
    i5 = list(C["links"]).index("P5")
    d_how = C["how__drift"][i5].sum(axis=0)                                  # P5's water ends in District_B only
    assert d_how[list(C["dests"]).index("District_B")] / d_how.sum() == pytest.approx(1.0)
