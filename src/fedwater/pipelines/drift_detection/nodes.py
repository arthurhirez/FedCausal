"""Drift detection on prototype trajectories, evaluated against ground truth.

Signals (faithful to BEPE):
* ``delta_first`` — cosine distance of month-m's local prototype to the
  month-0 prototype: cumulative drift w.r.t. the client's original domain.
* ``delta_roll``  — cosine distance between consecutive months' prototypes:
  drift *rate*.

Detection rule: a client's reference distribution is its delta_first over
the reference months (pre-drift by construction of the commissioning
period); the detection month is the first month where delta_first exceeds
``mean_ref + k_sigma * std_ref`` and STAYS above for ``persistence``
consecutive months (one-off excursions are seasonal noise, not domain
change).

Evaluation consumes ``gt_drift_schedule``: the drifting district must rank
first by drift magnitude; the detection month is compared with the true
onset (drift-affected-share curve); stable clients count false alarms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(1.0 - (a @ b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def compute_drift_signals(prototype_history: pd.DataFrame,
                          fl: dict) -> pd.DataFrame:
    """delta_first / delta_roll per (client, month) from the FINAL round."""
    fcols = [c for c in prototype_history.columns if c.startswith("f")]
    final = prototype_history[prototype_history["round"]
                              == prototype_history["round"].max()]
    rows = []
    for client, grp in final.groupby("client"):
        grp = grp.sort_values("month")
        months = grp["month"].to_numpy()
        vecs = grp[fcols].to_numpy()
        if len(vecs) < 2:
            raise AssertionError(f"{client}: fewer than 2 monthly prototypes.")
        for i in range(len(vecs)):
            rows.append(dict(
                client=client, month=int(months[i]),
                delta_first=_cosine_dist(vecs[i], vecs[0]) if i else 0.0,
                delta_roll=_cosine_dist(vecs[i], vecs[i - 1]) if i else np.nan,
            ))
    return pd.DataFrame(rows)


def _detection_month(sig: pd.DataFrame, reference_months: int,
                     k_sigma: float, persistence: int,
                     sigma_floor: float = 1e-3):
    """First month where delta_first exceeds the reference threshold and
    persists. Returns (month or None, threshold).

    ``sigma_floor`` should be the sigma POOLED across clients' reference
    windows: with only a handful of per-client reference points, a fluky
    small local sigma makes the threshold hair-trigger; under H0 the
    reference deltas are exchangeable across clients, so the pooled
    estimate is the legitimate stabilizer."""
    ref = sig[(sig["month"] > 0) & (sig["month"] < reference_months)]
    mu = ref["delta_first"].mean() if len(ref) else 0.0
    sd = ref["delta_first"].std(ddof=0) if len(ref) > 1 else 0.0
    thr = mu + k_sigma * max(sd, sigma_floor, 1e-3)

    post = sig[sig["month"] >= reference_months].sort_values("month")
    above = post["delta_first"].to_numpy() > thr
    months = post["month"].to_numpy()
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= persistence:
            return int(months[i - persistence + 1]), thr
    return None, thr


def evaluate_drift(drift_signals: pd.DataFrame,
                   gt_drift_schedule: pd.DataFrame, fl: dict) -> pd.DataFrame:
    """Score the detector against the simulator's ground truth."""
    cfg = fl["drift"]
    gt_drift_schedule = gt_drift_schedule.copy()
    drifted_client = gt_drift_schedule["district"].iloc[0]
    true_onset = int(gt_drift_schedule["drift_month"].min())

    ref_all = drift_signals[(drift_signals["month"] > 0)
                            & (drift_signals["month"] < cfg["reference_months"])]
    pooled_sd = float(ref_all["delta_first"].std(ddof=0)) if len(ref_all) > 1 \
        else 1e-3

    rows = []
    magnitudes = {}
    for client, sig in drift_signals.groupby("client"):
        det, thr = _detection_month(sig, cfg["reference_months"],
                                    cfg["k_sigma"], cfg["persistence"],
                                    sigma_floor=pooled_sd)
        post = sig[sig["month"] >= cfg["reference_months"]]
        magnitudes[client] = post["delta_first"].mean()
        rows.append(dict(client=client, is_drifted=client == drifted_client,
                         mean_delta_post=magnitudes[client],
                         threshold=thr, detection_month=det))
    report = pd.DataFrame(rows).sort_values("mean_delta_post",
                                            ascending=False,
                                            ignore_index=True)
    report["rank"] = np.arange(1, len(report) + 1)

    det_row = report[report["is_drifted"]].iloc[0]
    summary = dict(
        client="__summary__", is_drifted=True,
        mean_delta_post=det_row["mean_delta_post"],
        threshold=np.nan,
        detection_month=det_row["detection_month"],
        rank=det_row["rank"],
    )
    summary["drifted_rank_is_1"] = bool(det_row["rank"] == 1)
    summary["true_onset_month"] = true_onset
    summary["detection_delay_months"] = (
        det_row["detection_month"] - true_onset
        if det_row["detection_month"] is not None else np.nan)
    summary["false_alarms"] = int(
        report.loc[~report["is_drifted"], "detection_month"].notna().sum())
    summary["separation_ratio"] = float(
        det_row["mean_delta_post"]
        / report.loc[~report["is_drifted"], "mean_delta_post"].mean())
    return pd.concat([report, pd.DataFrame([summary])], ignore_index=True)
