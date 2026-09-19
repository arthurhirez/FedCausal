"""Sensing: what the federated clients actually observe.

The sensors themselves are chosen upstream (``sensor_placement``).

Demand stochasticity (behaviour) and measurement noise (instrumentation) are
distinct physical phenomena, so they live in distinct layers: everything up to
``hydraulics`` is the true state of the world; this pipeline degrades it into
observations — additive Gaussian noise plus quantization, per sensor type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedwater.hashing import stable_hash


def extract_sensor_series(pressures: pd.DataFrame, flows: pd.DataFrame,
                          sensor_placement: pd.DataFrame) -> pd.DataFrame:
    """Long tidy frame: step, month, district, sensor, slot, kind, value (true).

    Which element carries a gauge comes from ``sensor_placement`` -- chosen per
    world by the ``sensor_placement`` pipeline, or the bundle profile's
    hand-placed block when ``sensor_placement.source`` is ``manual``.

    ``sensor`` is the ELEMENT id (``q_P-13``) and keys the measurement-noise
    RNG, so an element's noise realisation does not depend on which slot it
    was chosen for. ``slot`` (``q_pure_0``) is the client-CSV column: slots are
    identical across districts, which is what gives every federated client the
    same feature geometry. For a manual placement ``slot == sensor``, so the
    client CSVs keep their historical column names.
    """
    src = {"pressure": pressures, "flow": flows}
    frames = []
    for r in sensor_placement.itertuples(index=False):
        table = src[r.kind]
        frames.append(pd.DataFrame({
            "step": table.index, "month": table["month"],
            "district": r.district, "sensor": r.sensor, "slot": r.slot,
            "kind": r.kind, "value": table[str(r.element)].to_numpy(),
        }))
    if not frames:
        raise ValueError("sensor_placement is empty -- no client would observe "
                         "anything")
    return pd.concat(frames, ignore_index=True)


def add_measurement_noise(sensor_series: pd.DataFrame, noise: dict,
                          seed: int) -> pd.DataFrame:
    """observed = quantize(true + eps), eps keyed per sensor for auditability.

    The key goes through ``fedwater.hashing.stable_hash``, NOT the builtin
    ``hash``. ``hash()`` on a str is salted per process (PEP 456), so the
    previous version drew a DIFFERENT noise realisation every time the same
    world was rerun. The true ``value`` column was identical and ``observed``
    was not, which is the worst shape for the bug to take: the world cache is
    content-addressed, so "same sim_hash" was taken to mean "same client
    data", and it did not -- ``client_datasets`` is downstream of
    ``observed``, so every replicate silently varied the instrumentation layer
    alongside whatever it meant to vary.
    """
    df = sensor_series.copy()
    observed = np.empty(len(df))
    for (sensor, kind), grp in df.groupby(["sensor", "kind"]):
        rng = np.random.default_rng([seed, stable_hash(sensor)])
        sigma = noise[f"{kind}_sigma"]
        q = noise[f"{kind}_quantization"]
        vals = grp["value"].to_numpy() + rng.normal(0.0, sigma, len(grp))
        observed[grp.index] = np.round(vals / q) * q
    df["observed"] = observed
    return df


def package_client_datasets(sensor_series: pd.DataFrame, time: dict,
                            start_date: str) -> dict:
    """One BEPE-compatible CSV per district: timestamp + one column per SLOT.

    Timestamps are synthetic (30-day months) — the authoritative month label
    is carried alongside, so downstream labeling never re-derives months from
    calendar arithmetic.

    Columns are slot names (``p_mixed_0 ... q_pure_2``), identical for every
    district, so ``preprocess_clients``' sorted column order means the same
    thing in every client. The element behind each slot is recorded in
    ``sensor_placement.csv``.
    """
    step_s = int(time["resolution_h"] * 3600)
    t0 = pd.Timestamp(start_date)

    out = {}
    for district, grp in sensor_series.groupby("district"):
        wide = grp.pivot(index="step", columns="slot", values="observed")
        wide.insert(0, "timestamp",
                    ((t0.value // 10**9) + wide.index * step_s).astype("int64"))
        month = grp.drop_duplicates("step").set_index("step")["month"]
        wide.insert(1, "month", month)
        out[district] = wide.reset_index(drop=True)
    return out
