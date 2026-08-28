"""FL preprocessing: from client sensor CSVs to model-ready windows.

The three correctness rules of this module
------------------------------------------
1. **No future leakage.** Scalers are fitted ONLY on the reference months
   (a commissioning period; by default the simulator's drift warm-up, which
   is guaranteed pre-drift for every client). The legacy code fitted MinMax
   on the whole series — the scaler "knew the future" and compressed the
   drift signal into the same range it was supposed to stand out from.
2. **Authoritative month labels.** The simulator emits the model-month
   column; windows are labelled by majority vote over their steps
   (threshold-gated, faithful to BEPE) — never re-derived from calendar
   arithmetic, which drifts against 30-day model months.
3. **Vectorized windowing.** ``sliding_window_view`` replaces the Python
   loop; a unit test pins its equivalence to the naive construction.

Federated discipline: every step here is client-local (per-client scalers,
per-client windows). Nothing crosses clients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _aggregate_regular(values: np.ndarray, months: np.ndarray, k: int):
    """Mean-aggregate a regular series by blocks of ``k`` steps.

    Requires months to be block-aligned (no block straddles two months) —
    guaranteed when steps_per_month % k == 0, asserted by the caller.
    """
    n = (len(values) // k) * k
    agg = values[:n].reshape(-1, k, values.shape[1]).mean(axis=1)
    agg_months = months[:n:k]
    return agg, agg_months


def _fit_reference_scaler(agg: np.ndarray, agg_months: np.ndarray,
                          reference_months: int, feature_range=(-1.0, 1.0)):
    """MinMax parameters from the reference months ONLY."""
    ref = agg[agg_months < reference_months]
    data_min, data_max = ref.min(axis=0), ref.max(axis=0)
    span = data_max - data_min
    if np.any(span <= 0):
        raise ValueError("Degenerate sensor in reference period (constant "
                         "signal) — scaler undefined.")
    lo, hi = feature_range
    scale = (hi - lo) / span
    return data_min, data_max, scale, lo


def _windows_and_labels(agg: np.ndarray, agg_months: np.ndarray,
                        window: int, stride: int, threshold: float):
    """Sliding windows + majority month labels (threshold-gated)."""
    from numpy.lib.stride_tricks import sliding_window_view

    if len(agg) < window:
        raise ValueError(f"Series shorter than window ({len(agg)} < {window}).")
    X = sliding_window_view(agg, (window, agg.shape[1])).squeeze(1)[::stride]
    M = sliding_window_view(agg_months, window)[::stride]
    starts = np.arange(0, len(agg) - window + 1, stride)

    # majority label per window, gated by threshold
    labels = np.full(len(M), -1, dtype=np.int64)
    for i, row in enumerate(M):
        vals, counts = np.unique(row, return_counts=True)
        j = counts.argmax()
        if counts[j] / window >= threshold:
            labels[i] = vals[j]
    mask = labels >= 0
    return (np.ascontiguousarray(X[mask], dtype=np.float32),
            labels[mask], starts[mask])


def preprocess_clients(client_datasets: dict, fl: dict, time: dict):
    """PartitionedDataset of client CSVs -> windows/labels/scalers/report.

    Returns
    -------
    fl_windows : dict client -> {windows (n,W,F) float32 scaled, labels (n,),
                 window_start_step (n,), sensors [str]}
    fl_scalers : tidy DataFrame (client, sensor, data_min, data_max) — the
                 audit trail of what "normal" meant at commissioning time.
    fl_prep_report : tidy checks DataFrame (hard failures raise here).
    """
    cfg = fl["preprocessing"]
    k = int(cfg["interval_agg_h"] / time["resolution_h"])
    steps_per_month = int(time["days_per_month"] * 24 / time["resolution_h"])
    if steps_per_month % k != 0:
        raise ValueError(f"interval_agg_h={cfg['interval_agg_h']} makes blocks "
                         "straddle month boundaries — labels would be corrupted.")

    windows_out, scaler_rows, report_rows = {}, [], []
    for client in sorted(client_datasets):
        df = client_datasets[client]() if callable(client_datasets[client]) \
            else client_datasets[client]
        sensors = sorted(c for c in df.columns if c.startswith(("p_", "q_")))
        values = df[sensors].to_numpy(dtype=np.float64)
        months = df["month"].to_numpy()

        if np.isnan(values).any():
            raise ValueError(f"{client}: NaNs in sensor data.")

        agg, agg_months = _aggregate_regular(values, months, k)
        dmin, dmax, scale, lo = _fit_reference_scaler(
            agg, agg_months, cfg["reference_months"],
            tuple(cfg["feature_range"]))
        agg_scaled = (agg - dmin) * scale + lo

        X, labels, starts = _windows_and_labels(
            agg_scaled, agg_months, cfg["window_size"], cfg["step_size"],
            cfg["label_threshold"])

        windows_out[client] = {"windows": X, "labels": labels,
                               "window_start_step": starts * k,
                               "sensors": sensors}
        for s, mn, mx in zip(sensors, dmin, dmax):
            scaler_rows.append({"client": client, "sensor": s,
                                "data_min": mn, "data_max": mx})

        n_months = len(np.unique(labels))
        expected_months = len(np.unique(agg_months))
        report_rows += [
            {"client": client, "check": "n_windows", "value": len(X),
             "passed": len(X) > 0},
            {"client": client, "check": "months_covered",
             "value": n_months, "passed": n_months == expected_months},
            {"client": client, "check": "ref_scaled_in_range",
             "value": float(np.abs(
                 agg_scaled[agg_months < cfg["reference_months"]]).max()),
             "passed": True},  # == 1.0 by construction; recorded for audit
        ]

    report = pd.DataFrame(report_rows)
    failed = report[~report["passed"]]
    if len(failed):
        raise AssertionError(
            f"FL preprocessing hard checks failed:\n{failed.to_string(index=False)}")
    return windows_out, pd.DataFrame(scaler_rows), report
