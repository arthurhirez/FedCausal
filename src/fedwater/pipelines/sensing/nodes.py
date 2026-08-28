"""Sensing: what the federated clients actually observe.

Demand stochasticity (behaviour) and measurement noise (instrumentation) are
distinct physical phenomena, so they live in distinct layers: everything up to
``hydraulics`` is the true state of the world; this pipeline degrades it into
observations — additive Gaussian noise plus quantization, per sensor type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def extract_sensor_series(pressures: pd.DataFrame, flows: pd.DataFrame,
                          sensors: dict) -> pd.DataFrame:
    """Long tidy frame: step, month, district, sensor, kind, value (true)."""
    frames = []
    for district, cfg in sensors.items():
        for node in cfg["pressure"]:
            frames.append(pd.DataFrame({
                "step": pressures.index, "month": pressures["month"],
                "district": district, "sensor": f"p_{node}",
                "kind": "pressure", "value": pressures[str(node)].to_numpy(),
            }))
        for link in cfg["flow"]:
            frames.append(pd.DataFrame({
                "step": flows.index, "month": flows["month"],
                "district": district, "sensor": f"q_{link}",
                "kind": "flow", "value": flows[str(link)].to_numpy(),
            }))
    return pd.concat(frames, ignore_index=True)


def add_measurement_noise(sensor_series: pd.DataFrame, noise: dict,
                          seed: int) -> pd.DataFrame:
    """observed = quantize(true + eps), eps keyed per sensor for auditability."""
    df = sensor_series.copy()
    observed = np.empty(len(df))
    for (sensor, kind), grp in df.groupby(["sensor", "kind"]):
        rng = np.random.default_rng([seed, hash(sensor) % 2**31])
        sigma = noise[f"{kind}_sigma"]
        q = noise[f"{kind}_quantization"]
        vals = grp["value"].to_numpy() + rng.normal(0.0, sigma, len(grp))
        observed[grp.index] = np.round(vals / q) * q
    df["observed"] = observed
    return df


def package_client_datasets(sensor_series: pd.DataFrame, time: dict,
                            start_date: str) -> dict:
    """One BEPE-compatible CSV per district: timestamp + one column per sensor.

    Timestamps are synthetic (30-day months) — the authoritative month label
    is carried alongside, so downstream labeling never re-derives months from
    calendar arithmetic.
    """
    step_s = int(time["resolution_h"] * 3600)
    t0 = pd.Timestamp(start_date)

    out = {}
    for district, grp in sensor_series.groupby("district"):
        wide = grp.pivot(index="step", columns="sensor", values="observed")
        wide.insert(0, "timestamp",
                    ((t0.value // 10**9) + wide.index * step_s).astype("int64"))
        month = grp.drop_duplicates("step").set_index("step")["month"]
        wide.insert(1, "month", month)
        out[district] = wide.reset_index(drop=True)
    return out
