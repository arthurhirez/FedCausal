"""Demand synthesis: from cohort portfolios to hourly node demands (L/s).

The behavioural model, in one paragraph
---------------------------------------
One household unit consumes on a daily profile made of three Gaussian use
peaks (morning / afternoon / evening) over a night base. Units differ in
*when* they hit their peaks: peak times are jittered across units with a
density-specific dispersion ``sigma_c`` (low-density suburbs are synchronised,
high-density blocks are staggered). Summing N such units has a closed form —
the convolution of a Gaussian bump of width ``w0`` with a Gaussian jitter of
spread ``sigma_c`` is a bump of width ``sqrt(w0^2 + sigma_c^2)``, and the
residual randomness of the finite crowd shrinks as ``1/sqrt(N)``. Crowd
smoothing is therefore *derived*, not injected: big cohorts are flat because
they are big, small cohorts are peaky because they are small.

On top of the daily shape: weekday/weekend structure, annual seasonality
(sinusoid over the 12-month cycle), a common-mode day-to-day multiplier
(weather affects everyone at once, so it does NOT shrink with N), and a small
month-level lognormal wobble. Each node-month series is then normalised so it
integrates EXACTLY to the node's target volume — volume correctness holds by
construction, and a validation node re-checks it downstream.
"""
from __future__ import annotations

import numpy as np

from fedwater.pipelines.urban_scenario.nodes import ANCHOR_DAYS_PER_MONTH
import pandas as pd

L_PER_M3 = 1000.0


def _gauss(t: np.ndarray, mu: float, w: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / w) ** 2)


def _cohort_day_shape(t, peaks, night, sigma_c, units, rng, weekend, patterns):
    """One cohort's daily shape (arbitrary scale, >=0)."""
    w0 = patterns["peak_width_h"]
    w_eff = np.sqrt(w0**2 + sigma_c**2)
    shift = patterns["weekend_morning_shift_h"] if weekend else 0.0
    residual = 1.0 / np.sqrt(max(units, 1.0))

    shape = np.full_like(t, night, dtype=float)
    for name, (mu, amp) in peaks.items():
        mu_eff = mu + (shift if name == "morning" else 0.0)
        mu_eff += rng.normal(0.0, sigma_c * residual)          # crowd wobble
        amp_eff = amp * (1.0 + rng.normal(0.0, patterns["amp_sigma"] * residual))
        if weekend and name == "morning":
            amp_eff *= patterns["weekend_morning_damp"]
        shape += max(amp_eff, 0.0) * _gauss(t, mu_eff, w_eff)
    return np.clip(shape, 1e-6, None)


def synthesize_demands(
    assignments_timeline: pd.DataFrame, patterns: dict, time: dict, seed: int,
) -> pd.DataFrame:
    """Hourly demand series per node, in L/s. Wide frame: index=hour, cols=nodes.

    Determinism: one child RNG per (node, month), spawned from the master seed,
    so any single node-month can be regenerated in isolation for audit.
    """
    res_h = time["resolution_h"]
    steps_day = int(round(24 / res_h))
    days = time["days_per_month"]
    n_months = time["n_months"]
    t = (np.arange(steps_day) + 0.5) * res_h  # bin centers, hours

    peaks_by_density = {
        d: {k: tuple(v) for k, v in cfg.items()}
        for d, cfg in patterns["peaks_by_density"].items()
    }
    nodes = sorted(assignments_timeline["node"].unique(), key=int)
    out = np.zeros((n_months * days * steps_day, len(nodes)))
    grouped = assignments_timeline.groupby(["month", "node"])

    def rng_for(node: str, month: int) -> np.random.Generator:
        # Keyed RNG: any single (node, month) is regenerable in isolation.
        return np.random.default_rng([seed, month, int(node)])

    seasonal_amp = patterns["seasonal_amplitude"]
    peak_month = patterns["seasonal_peak_month"]

    for (month, node), cohorts in grouped:
        rng = rng_for(node, int(month))
        density = cohorts["density"].iloc[0]
        cfg = peaks_by_density[density]
        night = patterns["night_by_density"][density]
        sigma_c = patterns["peak_dispersion_h_by_density"][density]
        sigma_day = patterns["day_sigma_by_density"][density]

        seasonal = 1.0 + seasonal_amp * np.cos(
            2 * np.pi * (int(month) % 12 - peak_month) / 12.0
        )
        v_month = (
            cohorts.drop_duplicates("template")["volume_m3_month"].iloc[0]
            * seasonal
            * rng.lognormal(0.0, patterns["month_sigma"])
        )

        units = cohorts.set_index("template")["units"]
        month_series = np.empty(days * steps_day)
        day0 = int(month) * days
        for d in range(days):
            weekend = ((day0 + d) % 7) >= 5
            shape = np.zeros(steps_day)
            for tpl, u in units.items():
                shape += (u / units.sum()) * _cohort_day_shape(
                    t, cfg, night, sigma_c, u, rng, weekend, patterns
                )
            shape *= 1.0 + rng.normal(0.0, sigma_day)  # common-mode day factor
            month_series[d * steps_day:(d + 1) * steps_day] = np.clip(shape, 1e-6, None)

        # Exact volume normalisation: sum(demand_lps * step_seconds) == liters.
        # v_month is a monthly volume on the 30-day anchor basis; the simulated
        # month integrates the corresponding daily rate over its actual days.
        liters = v_month * L_PER_M3 * (days / ANCHOR_DAYS_PER_MONTH)
        step_s = res_h * 3600.0
        month_series *= liters / (month_series.sum() * step_s)

        col = nodes.index(node)
        h0 = int(month) * days * steps_day
        out[h0:h0 + days * steps_day, col] = month_series

    idx = pd.RangeIndex(out.shape[0], name="step")
    df = pd.DataFrame(out, index=idx, columns=nodes)
    df.insert(0, "month", np.repeat(np.arange(n_months), days * steps_day))
    return df


def apply_drift_ramp(demand_series: pd.DataFrame, gt_drift_schedule: pd.DataFrame,
                     patterns: dict, time: dict) -> pd.DataFrame:
    """Gradual onset: blend each drifted node's old->new regime linearly over
    ``ramp_days`` at the start of its drift month (concept drift is a process,
    not a step)."""
    df = demand_series.copy()
    steps_day = int(round(24 / time["resolution_h"]))
    ramp_steps = int(patterns["drift_ramp_days"] * steps_day)
    days = time["days_per_month"]

    for _, row in gt_drift_schedule.iterrows():
        node, m = str(row["node"]), int(row["drift_month"])
        if node not in demand_series.columns:
            continue  # zero-demand trunk junctions carry no portfolio, hence
                      # no demand column: their "drift" has no observable.
        if m == 0 or ramp_steps == 0:
            continue
        h0 = m * days * steps_day
        prev_month = df[df["month"] == m - 1][node].to_numpy()
        # Previous regime proxy for the ramp window: last cycle of month m-1.
        old = np.resize(prev_month[-ramp_steps:], ramp_steps)
        w = np.linspace(0.0, 1.0, ramp_steps)
        seg = df.loc[h0:h0 + ramp_steps - 1, node].to_numpy()
        df.loc[h0:h0 + ramp_steps - 1, node] = (1 - w) * old + w * seg
    return df
