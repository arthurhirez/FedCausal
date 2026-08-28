"""FL backbone tests — every correctness fix is pinned by a test.

Covers: preprocessing (windowing equivalence, label majority, scaler
leakage guard), AER (shapes, latent == encoder bottleneck), the vectorized
hierarchical loss vs a literal reference implementation, prototype
extraction soundness, FINCH aggregation, FedAvg, drift metrics on a planted
shift, and an end-to-end smoke run with bit-reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fedwater.pipelines.drift_detection.nodes import (
    compute_drift_signals,
    evaluate_drift,
)
from fedwater.pipelines.fl_preprocessing.nodes import (
    _fit_reference_scaler,
    _windows_and_labels,
    preprocess_clients,
)
from fedwater.pipelines.fl_training.aer import AER, split_window_targets
from fedwater.pipelines.fl_training.federated import (
    FPLTrainer,
    aggregate_prototypes,
    extract_prototypes,
    hierarchical_proto_loss,
)

FL = {
    "preprocessing": {"interval_agg_h": 2, "window_size": 12, "step_size": 1,
                      "reference_months": 2, "label_threshold": 0.75,
                      "feature_range": [-1.0, 1.0]},
    "model": {"lstm_units": 8, "reg_ratio": 0.5},
    "training": {"rounds": 2, "local_epochs": 1, "learning_rate": 1e-3,
                 "batch_size": 32, "participation": 1.0, "averaging": "equal",
                 "proto_alpha": 0.2, "infonce_temperature": 0.02,
                 "device": "cpu"},
    "drift": {"reference_months": 2, "k_sigma": 3.0, "persistence": 2},
}
TIME = {"n_months": 4, "days_per_month": 5, "resolution_h": 1}


def _client_df(seed, drift_from_month=None, n_months=4, days=5):
    """Synthetic client CSV: diurnal sine + noise; optional level shift."""
    rng = np.random.default_rng(seed)
    steps = n_months * days * 24
    t = np.arange(steps)
    month = t // (days * 24)
    base = np.sin(2 * np.pi * t / 24)
    cols = {}
    for i, name in enumerate(["p_1", "p_2", "q_1"]):
        sig = 10 + base * (1 + 0.1 * i) + 0.15 * rng.normal(size=steps)
        if drift_from_month is not None:
            sig = sig + 5.0 * (month >= drift_from_month)
        cols[name] = sig
    return pd.DataFrame({"timestamp": t * 3600, "month": month, **cols})


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def test_windows_match_naive_loop():
    rng = np.random.default_rng(0)
    agg = rng.normal(size=(50, 3))
    months = np.repeat([0, 1], 25)
    X, labels, starts = _windows_and_labels(agg, months, window=8, stride=2,
                                            threshold=0.75)
    # naive reference
    for w_idx in range(len(X)):
        s = starts[w_idx]
        np.testing.assert_array_equal(X[w_idx], agg[s:s + 8].astype(np.float32))
    # boundary windows (mixed months below threshold) must be dropped
    frac = [(months[s:s + 8] == labels[i]).mean() for i, s in enumerate(starts)]
    assert min(frac) >= 0.75


def test_scaler_no_future_leakage():
    """THE guard: changing post-reference data must not move the scaler."""
    rng = np.random.default_rng(1)
    agg = rng.normal(size=(200, 2))
    months = np.repeat(np.arange(4), 50)
    p1 = _fit_reference_scaler(agg, months, reference_months=2)
    corrupted = agg.copy()
    corrupted[months >= 2] += 100.0            # massive future drift
    p2 = _fit_reference_scaler(corrupted, months, reference_months=2)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a, b)


def test_preprocess_clients_end_to_end():
    data = {"C_A": _client_df(0), "C_B": _client_df(1)}
    windows, scalers, report = preprocess_clients(data, FL, TIME)
    assert set(windows) == {"C_A", "C_B"}
    W = windows["C_A"]["windows"]
    assert W.dtype == np.float32 and W.shape[1] == 12 and W.shape[2] == 3
    # reference months scaled inside [-1, 1] exactly
    assert report.query("check == 'ref_scaled_in_range'")["value"].max() <= 1 + 1e-9
    assert (scalers.groupby("client").size() == 3).all()


def test_preprocess_rejects_straddling_aggregation():
    bad_fl = {**FL, "preprocessing": {**FL["preprocessing"],
                                      "interval_agg_h": 7}}
    with pytest.raises(ValueError, match="straddle"):
        preprocess_clients({"C_A": _client_df(0)}, bad_fl, TIME)


# --------------------------------------------------------------------------
# AER
# --------------------------------------------------------------------------
def test_aer_shapes_and_latent_identity():
    torch.manual_seed(0)
    model = AER(n_features=3, window_size=12, lstm_units=8)
    wb = torch.randn(5, 12, 3)
    x, ry_t, y_t, fy_t = split_window_targets(wb)
    ry, y, fy, z = model(x)
    assert ry.shape == (5, 3) and fy.shape == (5, 3)
    assert y.shape == (5, 10, 3) and z.shape == (5, 16)
    # the returned latent IS the encoder bottleneck (the legacy bug fix)
    torch.testing.assert_close(z, model.encode(x))
    assert z.requires_grad


# --------------------------------------------------------------------------
# hierarchical loss: vectorized == literal reference
# --------------------------------------------------------------------------
def _reference_proto_loss(z, labels, protos, alpha, T):
    """Literal per-instance implementation of the FPL loss."""
    total, count = 0.0, 0
    for i in range(len(z)):
        m = int(labels[i])
        if m not in protos:
            continue
        pos = torch.tensor(protos[m][0], dtype=torch.float32)
        neg = torch.cat([torch.tensor(protos[k][0], dtype=torch.float32)
                         for k in protos if k != m]) if len(protos) > 1 \
            else torch.zeros((0, z.shape[1]))
        allp = torch.cat([pos, neg])
        sims = torch.cosine_similarity(z[i:i + 1], allp) / T
        info = -torch.log(torch.exp(sims[:len(pos)]).sum()
                          / torch.exp(sims).sum())
        reg = torch.nn.functional.mse_loss(
            z[i], torch.tensor(protos[m][1], dtype=torch.float32))
        total = total + alpha * info + (1 - alpha) * reg
        count += 1
    return total / count


def test_vectorized_loss_matches_reference():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    protos = {m: (rng.normal(size=(2, 6)), rng.normal(size=6))
              for m in (0, 1, 2)}
    z = torch.randn(9, 6, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 5, 5])  # two unmatched
    fast = hierarchical_proto_loss(z, labels, protos, 0.2, 0.02,
                                   torch.device("cpu"))
    slow = _reference_proto_loss(z, labels, protos, 0.2, 0.02)
    torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-6)
    fast.backward()          # grad flows
    assert z.grad is not None


def test_loss_safe_when_no_label_matches():
    protos = {7: (np.ones((1, 4)), np.ones(4))}
    z = torch.randn(3, 4)
    out = hierarchical_proto_loss(z, torch.tensor([0, 1, 2]), protos,
                                  0.2, 0.02, torch.device("cpu"))
    assert float(out) == 0.0    # the legacy crash case


# --------------------------------------------------------------------------
# prototypes
# --------------------------------------------------------------------------
def test_prototype_extraction_is_clean_mean():
    torch.manual_seed(4)
    model = AER(3, 12, 8)
    windows = torch.randn(40, 12, 3)
    labels = np.repeat([0, 1], 20)
    protos = extract_prototypes(model, windows, labels, batch_size=16)
    with torch.no_grad():
        z = model.encode(windows[:, 1:-1]).numpy()
    np.testing.assert_allclose(protos[0], z[:20].mean(0), rtol=1e-5)
    np.testing.assert_allclose(protos[1], z[20:].mean(0), rtol=1e-5)


def test_aggregate_prototypes_shapes_and_singleton():
    rng = np.random.default_rng(5)
    local = {f"c{i}": {0: rng.normal(size=6), 1: rng.normal(size=6)}
             for i in range(4)}
    local["c9"] = {2: rng.normal(size=6)}          # singleton month
    out = aggregate_prototypes(local, seed=0)
    assert set(out) == {0, 1, 2}
    clusters, mean = out[0]
    assert clusters.ndim == 2 and clusters.shape[1] == 6
    np.testing.assert_allclose(mean, clusters.mean(0))
    assert out[2][0].shape == (1, 6)               # singleton passthrough


def test_fedavg_equal_average():
    data = {"a": {"windows": np.random.rand(8, 12, 3).astype(np.float32),
                  "labels": np.zeros(8, dtype=int),
                  "window_start_step": np.arange(8), "sensors": list("xyz")},
            "b": {"windows": np.random.rand(8, 12, 3).astype(np.float32),
                  "labels": np.zeros(8, dtype=int),
                  "window_start_step": np.arange(8), "sensors": list("xyz")}}
    tr = FPLTrainer(data, FL, seed=0)
    w1 = {k: torch.ones_like(v) for k, v in tr.models["a"].state_dict().items()}
    w3 = {k: 3 * torch.ones_like(v) for k, v in w1.items()}
    tr.models["a"].load_state_dict(w1)
    tr.models["b"].load_state_dict(w3)
    tr._fedavg(["a", "b"])
    for v in tr.global_model.state_dict().values():
        torch.testing.assert_close(v, 2 * torch.ones_like(v))


# --------------------------------------------------------------------------
# drift detection
# --------------------------------------------------------------------------
def _planted_history(shift_client="C_D", shift_month=4, n_months=10, dim=8):
    rng = np.random.default_rng(7)
    base = {c: rng.normal(size=dim) for c in ["C_A", "C_B", "C_D"]}
    rows = []
    for c, b in base.items():
        drifted = b + 4.0 * rng.normal(size=dim)   # a distant direction
        for m in range(n_months):
            v = b + 0.02 * rng.normal(size=dim)
            if c == shift_client and m >= shift_month:
                v = drifted + 0.02 * rng.normal(size=dim)
            rows.append(dict(round=0, client=c, month=m,
                             **{f"f{i}": x for i, x in enumerate(v)}))
    return pd.DataFrame(rows)


def test_drift_signals_and_detection_on_planted_shift():
    hist = _planted_history()
    signals = compute_drift_signals(hist, FL)
    gt = pd.DataFrame({"node": ["1"], "district": ["C_D"],
                       "drift_month": [4], "to_income": ["high"],
                       "to_density": ["low"]})
    report = evaluate_drift(signals, gt, FL)
    summary = report[report["client"] == "__summary__"].iloc[0]
    assert bool(summary["drifted_rank_is_1"])
    assert summary["detection_month"] == 4        # exact onset, no delay
    assert summary["false_alarms"] == 0
    # delta_roll spikes exactly at the shift month
    roll = signals[signals["client"] == "C_D"].set_index("month")["delta_roll"]
    assert roll.idxmax() == 4


# --------------------------------------------------------------------------
# end-to-end smoke + determinism
# --------------------------------------------------------------------------
def test_fl_end_to_end_deterministic():
    data = {"C_A": _client_df(0), "C_D": _client_df(1, drift_from_month=2)}
    windows, _, _ = preprocess_clients(data, FL, TIME)

    from fedwater.pipelines.fl_training.nodes import train_federated
    out1 = train_federated(windows, FL, seed=11)
    out2 = train_federated(windows, FL, seed=11)
    pd.testing.assert_frame_equal(out1[1], out2[1])   # prototype history
    pd.testing.assert_frame_equal(out1[3], out2[3])   # latent trajectories

    models, protos, gprotos, traj, log = out1
    assert set(traj.columns[:4]) == {"district", "kind", "window", "month"}
    assert log["loss"].notna().all()
    # prototype-history coverage: both clients, all months present
    cov = protos[protos["round"] == protos["round"].max()] \
        .groupby("client")["month"].nunique()
    assert (cov == TIME["n_months"]).all()
