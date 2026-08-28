"""Tests for federated dependence detection — synthetic truths per property."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedwater.pipelines.dependence_detection import fed_methods as F
from fedwater.pipelines.dependence_detection.nodes import (
    federated_dependence,
    residualize_latents,
)

FL = {"dependence": {"max_lag_windows": 4, "n_surrogates": 79,
                     "n_surrogates_expensive": 20}}
TIME = {"resolution_h": 1, "days_per_month": 5, "n_months": 4}


def _traj_df(mats: dict, months=None):
    """Pack {client: (n x d)} into the latent_trajectories schema."""
    frames = []
    n = len(next(iter(mats.values())))
    windows = np.arange(n) * 4                      # stride-4h grid
    months = np.zeros(n, dtype=int) if months is None else months
    for c, X in mats.items():
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df.insert(0, "month", months)
        df.insert(0, "window", windows)
        df.insert(0, "kind", "aer_latent")
        df.insert(0, "district", c)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _proto_hist(disp_pairs: dict, dim=6, months=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    shared = rng.normal(size=(months, dim))
    for c, coupled in disp_pairs.items():
        base = rng.normal(size=dim)
        steps = shared if coupled else rng.normal(size=(months, dim))
        vecs = base + np.cumsum(steps, axis=0)
        for m in range(months):
            rows.append(dict(round=0, client=c, month=m,
                             **{f"f{i}": v for i, v in enumerate(vecs[m])}))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def coupled_world():
    """A-B share a latent factor; C independent. n=400, d=6."""
    rng = np.random.default_rng(1)
    n, d = 400, 6
    shared = rng.normal(size=(n, d))
    mats = {"A": shared + 0.4 * rng.normal(size=(n, d)),
            "B": shared + 0.4 * rng.normal(size=(n, d)),
            "C": rng.normal(size=(n, d))}
    return mats


def test_level_t_detects_coupling_and_nulls(coupled_world):
    traj = _traj_df(coupled_world)
    protos = _proto_hist({"A": True, "B": True, "C": False})
    out = federated_dependence(traj, protos, FL, seed=0)
    rv = out[(out["level"] == "T") & (out["method"] == "rv")] \
        .set_index(["district_a", "district_b"])
    assert rv.loc[("A", "B"), "statistic"] > 0.5
    assert rv.loc[("A", "B"), "q_value"] < 0.05
    assert rv.loc[("A", "C"), "statistic"] < 0.15
    assert rv.loc[("A", "C"), "p_value"] > 0.1
    # cost accounting present and level-consistent
    assert (out.groupby("level")["comm_floats_per_client"].nunique() == 1).all()


def test_partial_rv_suppresses_mediated_chain():
    """A -> B -> C: partial RV(A, C | B) collapses vs marginal RV."""
    rng = np.random.default_rng(2)
    n, d = 500, 5
    A = rng.normal(size=(n, d))
    B = A + 0.3 * rng.normal(size=(n, d))
    C = B + 0.3 * rng.normal(size=(n, d))
    from fedwater.pipelines.dependence_oracle.methods import rv_coefficient
    marginal = rv_coefficient(A, C)
    partial = F.partial_rv(A, C, B)
    assert partial < 0.2 < marginal


def test_residualization_kills_shared_seasonality():
    """Independent clients + common monthly offset: raw RV high, resid low."""
    rng = np.random.default_rng(3)
    n, d = 480, 6
    months = np.repeat(np.arange(4), n // 4)
    season = np.repeat(rng.normal(size=(4, d)) * 3, n // 4, axis=0)
    mats = {"A": season + rng.normal(size=(n, d)),
            "B": season + rng.normal(size=(n, d))}
    traj = _traj_df(mats, months=months)
    from fedwater.pipelines.dependence_oracle.methods import rv_coefficient
    raw = rv_coefficient(mats["A"], mats["B"])
    resid = residualize_latents(traj, FL, TIME)
    fcols = [c for c in resid.columns if c.startswith("f")]
    Ar = resid[resid["district"] == "A"][fcols].to_numpy()
    Br = resid[resid["district"] == "B"][fcols].to_numpy()
    assert rv_coefficient(Ar, Br) < 0.1 < raw


def test_level_p_displacement_comovement():
    protos = _proto_hist({"A": True, "B": True, "C": False}, months=20)
    traj = _traj_df({c: np.random.default_rng(9).normal(size=(60, 4))
                     for c in "ABC"})
    out = federated_dependence(traj, protos, FL, seed=1)
    p_rv = out[(out["level"] == "P") & (out["method"] == "rv")] \
        .set_index(["district_a", "district_b"])["statistic"]
    assert p_rv[("A", "B")] > p_rv[("A", "C")]
    assert p_rv[("A", "B")] > 0.5


def test_bh_fdr_known_vector():
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.8])
    q = F.bh_fdr(p)
    np.testing.assert_allclose(q, [0.005, 0.025, 0.05, 0.05, 0.8], atol=1e-12)
    assert (np.diff(q[np.argsort(p)]) >= -1e-15).all()   # monotone


def test_dependence_deterministic(coupled_world):
    traj = _traj_df(coupled_world)
    protos = _proto_hist({"A": True, "B": True, "C": False})
    o1 = federated_dependence(traj, protos, FL, seed=5)
    o2 = federated_dependence(traj, protos, FL, seed=5)
    pd.testing.assert_frame_equal(o1, o2)
