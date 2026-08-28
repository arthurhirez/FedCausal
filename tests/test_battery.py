"""Tests for the dependence battery — every tier validated on synthetic
systems where the truth is known by construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedwater.pipelines.dependence_oracle import methods as M
from fedwater.pipelines.dependence_oracle.nodes import (
    minirocket_trajectories,
    structure_recovery,
)

RNG = np.random.default_rng(123)
N = 3000


def _ar1(n, phi=0.7, rng=RNG):
    x = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.fixture(scope="module")
def coupled_pair():
    """b driven by a with lag 1: a -> b, ground truth directional."""
    rng = np.random.default_rng(7)
    a = _ar1(N, rng=rng)
    b = np.zeros(N)
    e = rng.normal(size=N)
    for t in range(1, N):
        b[t] = 0.5 * b[t - 1] + 0.6 * a[t - 1] + 0.5 * e[t]
    return a, b


@pytest.fixture(scope="module")
def independent_pair():
    rng = np.random.default_rng(8)
    return _ar1(N, rng=rng), _ar1(N, rng=np.random.default_rng(9))


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------
def test_pearson_surrogates_discriminate(coupled_pair, independent_pair):
    rng = np.random.default_rng(0)
    _, p_dep = M.circular_shift_pvalue(M.pearson, *coupled_pair, 100, rng)
    _, p_ind = M.circular_shift_pvalue(M.pearson, *independent_pair, 100, rng)
    assert p_dep < 0.02
    assert p_ind > 0.05


def test_lead_lag_direction(coupled_pair):
    a, b = coupled_pair
    r, lag = M.max_lagged_xcorr(a, b, max_lag=5)
    assert lag == 1 and abs(r) > 0.4  # a leads b by exactly one step


def test_precision_partial_suppresses_mediated_path():
    """Chain a -> b -> c: partial(a,c | b) must vanish vs marginal corr."""
    rng = np.random.default_rng(11)
    a = _ar1(N, rng=rng)
    b = 0.8 * a + 0.4 * rng.normal(size=N)
    c = 0.8 * b + 0.4 * rng.normal(size=N)
    partial = M.precision_partial_corr(np.column_stack([a, b, c]))
    marginal_ac = M.pearson(a, c)
    assert abs(partial[0, 2]) < 0.15 < abs(marginal_ac)


# --------------------------------------------------------------------------
# Tier 2
# --------------------------------------------------------------------------
def test_dcor_catches_nonlinear_dependence():
    """y = x^2: Pearson blind, distance correlation must see it."""
    rng = np.random.default_rng(12)
    x = rng.normal(size=1000)
    y = x**2 + 0.1 * rng.normal(size=1000)
    assert abs(M.pearson(x, y)) < 0.15
    assert M.distance_correlation(x, y) > 0.4
    p_rng = np.random.default_rng(1)
    _, p = M.circular_shift_pvalue(M.distance_correlation, x, y, 50, p_rng)
    assert p < 0.05


def test_mutual_information_positive_for_coupled(coupled_pair, independent_pair):
    mi_dep = M.mutual_information(*(M.strided(s, 1500) for s in coupled_pair))
    mi_ind = M.mutual_information(*(M.strided(s, 1500) for s in independent_pair))
    assert mi_dep > mi_ind


# --------------------------------------------------------------------------
# Tier 3
# --------------------------------------------------------------------------
def test_granger_direction(coupled_pair):
    a, b = coupled_pair
    f_ab = M.granger_f(a, b, lags=4)   # a -> b: should be large
    f_ba = M.granger_f(b, a, lags=4)   # b -> a: should be small
    assert f_ab > 5 * max(f_ba, 1.0)


def test_ccm_detects_coupling(coupled_pair, independent_pair):
    """CCM convention: a->b coupling => a reconstructable from M_b."""
    rng = np.random.default_rng(3)
    kw = dict(embed_dim=3, tau=1, lib_sizes=[100, 800], n_neighbors=4)
    skill_dep, conv_dep = M.ccm_skill(*coupled_pair, rng=rng, **kw)
    skill_ind, _ = M.ccm_skill(*independent_pair, rng=rng, **kw)
    assert skill_dep > skill_ind + 0.2
    assert conv_dep > 0  # skill grows with library size: true coupling


# --------------------------------------------------------------------------
# Tier 4
# --------------------------------------------------------------------------
def test_rv_coefficient_bounds():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 8))
    assert M.rv_coefficient(X, X) == pytest.approx(1.0)
    Y = rng.normal(size=(300, 8))
    assert M.rv_coefficient(X, Y) < 0.2


def test_minirocket_trajectories_shared_space_and_determinism():
    """Trajectories live in ONE space per kind, deterministically."""
    rng = np.random.default_rng(5)
    steps = 14 * 24  # two weeks hourly
    rows = []
    base = np.sin(2 * np.pi * np.arange(steps) / 24)
    for d in ["District_A", "District_B"]:
        sig = base + 0.3 * rng.normal(size=steps)
        rows.append(pd.DataFrame({
            "step": np.arange(steps), "month": 0, "district": d,
            "sensor": f"p_{d}", "kind": "pressure", "value": sig,
            "observed": sig,
        }))
    ss = pd.concat(rows, ignore_index=True)
    oracle = {"minirocket": {"window_h": 24, "stride_h": 6,
                             "n_kernels": 500, "pca_dims": 4}}
    time = {"resolution_h": 1}
    t1 = minirocket_trajectories(ss, oracle, time, seed=42)
    t2 = minirocket_trajectories(ss, oracle, time, seed=42)
    pd.testing.assert_frame_equal(t1, t2)  # deterministic given seed
    assert set(t1["district"]) == {"District_A", "District_B"}
    assert t1.groupby("district")["window"].count().nunique() == 1


# --------------------------------------------------------------------------
# structure recovery
# --------------------------------------------------------------------------
def test_structure_recovery_perfect_and_inverted():
    topo = pd.DataFrame({
        "district_a": ["A", "A", "B"], "district_b": ["B", "C", "C"],
        "open_boundary_pipes": [3, 0, 1],
        "boundary_pipes": [3, 0, 1],
        "hydraulic_distance_m": [100.0, 900.0, 300.0],
    })
    perfect = pd.DataFrame({
        "kind": "flow", "district_a": ["A", "A", "B"],
        "district_b": ["B", "C", "C"], "tier": 1, "method": "perfect",
        "direction": "sym", "statistic": [0.9, 0.05, 0.5],
        "p_value": 0.01, "extra": np.nan,
    })
    scores = structure_recovery(perfect, topo)
    assert scores["auroc"].iloc[0] == pytest.approx(1.0)
    assert scores["spearman_vs_proximity"].iloc[0] == pytest.approx(1.0)

    inverted = perfect.assign(statistic=[0.05, 0.9, 0.1], method="inverted")
    scores = structure_recovery(inverted, topo)
    assert scores["auroc"].iloc[0] == pytest.approx(0.0)
