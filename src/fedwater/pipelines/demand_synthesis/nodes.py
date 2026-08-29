"""Demand synthesis: from sector cohorts to hourly node demands (L/s).

The behavioural model, in one paragraph
---------------------------------------
A block is a mix of residential, commercial and industrial plots. Each sector
has its own daily signature: residential is three Gaussian use peaks over a
night floor; commercial and industrial are plateaus — a boxcar over a night
floor — differing in window, night level and day lift. Plots differ in *when*
they hit their peak or open their doors: start times are jittered across plots
with a sector-specific dispersion. Summing N such plots has a closed form. The
convolution of a Gaussian bump of width ``w0`` with a Gaussian jitter of spread
``sigma`` is a bump of width ``sqrt(w0^2 + sigma^2)``; the convolution of a
boxcar with the same jitter is a difference of two erfs of that width. In both
cases the residual randomness of the finite crowd shrinks as ``1/sqrt(N)``, so
crowd smoothing is *derived*, not injected — and because establishments are few
per block, commercial and industrial cohorts come out naturally noisier than
residential ones with no extra parameter.

Each sector also has a weekly signature: residential shifts and damps its
weekend morning, commercial nearly closes, industrial dips mildly. This is the
part that matters most downstream. The FL scaler is a per-client MinMax fitted
on the commissioning months, which is affine and therefore destroys demand
LEVEL but preserves a weekday/weekend gate and a night-floor change. The FL
window is 84 x 2 h = exactly 7 days, so a weekly gate is fully resolved inside
one window.

On top of the daily and weekly shape: annual seasonality (sinusoid over the
12-month cycle) and a common-mode day-to-day multiplier (weather affects
everyone at once, so it does NOT shrink with N) — both VOLUME-WEIGHTED across
the node's sectors, because industrial process water is weather-insensitive
while residential is not — plus a small month-level lognormal wobble. Each
node-month series is then normalised so it integrates EXACTLY to the node's
target volume; volume correctness holds by construction and a validation node
re-checks it downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import erf

from fedwater.pipelines.urban_scenario.nodes import ANCHOR_DAYS_PER_MONTH

L_PER_M3 = 1000.0


def _gauss(t: np.ndarray, mu: float, w: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / w) ** 2)


def _boxcar(t: np.ndarray, a: float, b: float, w: float) -> np.ndarray:
    """A plateau over [a, b], convolved with Gaussian start-time jitter of
    width ``w`` — i.e. a difference of two erfs. Assumes ``a < b``: windows
    that wrap past midnight are not supported (deferred)."""
    return 0.5 * (erf((t - a) / (np.sqrt(2) * w)) - erf((t - b) / (np.sqrt(2) * w)))


def sector_day_shape(t: np.ndarray, sector_cfg: dict, plots: float,
                     rng: np.random.Generator, weekend: bool,
                     patterns: dict) -> np.ndarray:
    """One sector cohort's daily shape (arbitrary scale, > 0).

    The same jitter/amplitude machinery drives both ``kind`` values, so the
    plateau sectors inherit the sqrt-N crowd smoothing rather than needing a
    smoothing parameter of their own.
    """
    w_eff = np.sqrt(patterns["peak_width_h"] ** 2 + sector_cfg["dispersion_h"] ** 2)
    residual = 1.0 / np.sqrt(max(plots, 1.0))
    jitter = rng.normal(0.0, sector_cfg["dispersion_h"] * residual)

    def amp_noise(a: float) -> float:
        return max(a * (1.0 + rng.normal(0.0, patterns["amp_sigma"] * residual)), 0.0)

    if sector_cfg["kind"] == "peaks":
        shape = np.full_like(t, sector_cfg["night"], dtype=float)
        shift = patterns["weekend_morning_shift_h"] if weekend else 0.0
        for name, (mu, amp) in sector_cfg["peaks"].items():
            amp_eff = amp_noise(amp) * (patterns["weekend_morning_damp"]
                                        if (weekend and name == "morning") else 1.0)
            mu_eff = mu + jitter + (shift if name == "morning" else 0.0)
            shape += amp_eff * _gauss(t, mu_eff, w_eff)
    elif sector_cfg["kind"] == "plateau":
        a, b = sector_cfg["window"]
        shape = sector_cfg["night"] + amp_noise(sector_cfg["amp"]) * _boxcar(
            t, a + jitter, b + jitter, w_eff)
    else:
        raise ValueError(f"Unknown sector kind '{sector_cfg['kind']}' "
                         "(expected 'peaks' or 'plateau').")

    if weekend:
        shape = shape * sector_cfg["weekend_factor"]
    return np.clip(shape, 1e-6, None)


def synthesize_demands(assignments_timeline: pd.DataFrame, land_use: dict,
                       patterns: dict, time: dict, seed: int) -> pd.DataFrame:
    """Hourly demand series per node, in L/s. Wide frame: index=step, cols=nodes.

    Determinism: one child RNG per (node, month), spawned from the master seed,
    so any single node-month can be regenerated in isolation for audit. Sector
    cohorts are consumed in sorted order so the RNG stream does not depend on
    the row order of the timeline frame.

    Two mixing rules, each found by a failure:

    * **Cohorts mix by VOLUME share, not plot count.** Sectors differ by ~10x
      in intensity, so plot-weighting drowns the commercial signal under the
      residential majority even in a commercial-coded block.
    * **Each cohort is normalised over the WHOLE MONTH, not per day.** Per-day
      normalisation rescales every day to the same total and so erases
      commercial's weekend collapse before it ever reaches the mix.
    """
    sectors = land_use["sectors"]
    res_h = time["resolution_h"]
    steps_day = int(round(24 / res_h))
    days = int(time["days_per_month"])
    n_months = int(time["n_months"])
    t = (np.arange(steps_day) + 0.5) * res_h  # bin centers, hours

    seasonality_scale = float(patterns.get("seasonality_scale", 1.0))
    peak_month = patterns["seasonal_peak_month"]

    nodes = sorted(assignments_timeline["node"].unique(), key=int)
    col_of = {n: j for j, n in enumerate(nodes)}
    out = np.zeros((n_months * days * steps_day, len(nodes)))

    for (month, node), cohorts in assignments_timeline.groupby(["month", "node"]):
        month = int(month)
        # Keyed RNG: any single (node, month) is regenerable in isolation.
        rng = np.random.default_rng([seed, month, int(node)])

        sector_volume = cohorts.set_index("sector")["sector_volume_m3_month"].sort_index()
        if not np.isfinite(sector_volume.sum()) or sector_volume.sum() <= 0:
            # Zero-volume node (zero-base-demand trunk junction). Skipping it
            # here as well as in the cohort builder keeps NaN out of the EPANET
            # pattern — a NaN multiplier fails the solver with "Error 200".
            continue
        vol_w = sector_volume / sector_volume.sum()
        plots = cohorts.set_index("sector")["plots"]

        # Weekends come from a GLOBAL day counter, so the weekly cycle stays
        # continuous across month boundaries at any days_per_month.
        weekend = np.array([((month * days + d) % 7) >= 5 for d in range(days)])

        series = np.zeros(days * steps_day)
        for sector, w in vol_w.items():
            cohort = np.concatenate([
                sector_day_shape(t, sectors[sector], float(plots[sector]), rng,
                                 bool(wk), patterns)
                for wk in weekend
            ])
            series += w * (cohort / cohort.mean())

        # Weather and seasonality are volume-weighted across the node's mix.
        seasonal_amp = seasonality_scale * sum(
            w * sectors[s]["seasonal_amplitude"] for s, w in vol_w.items())
        day_sigma = sum(w * sectors[s]["day_sigma"] for s, w in vol_w.items())
        series = series * np.repeat(1.0 + rng.normal(0.0, day_sigma, days), steps_day)

        v_month = (sector_volume.sum()
                   * (1.0 + seasonal_amp * np.cos(
                       2 * np.pi * (month % 12 - peak_month) / 12.0))
                   * rng.lognormal(0.0, patterns["month_sigma"]))

        # Exact volume normalisation: sum(demand_lps * step_seconds) == liters.
        # v_month is a monthly volume on the 30-day anchor basis; the simulated
        # month integrates the corresponding daily rate over its actual days.
        liters = v_month * L_PER_M3 * (days / ANCHOR_DAYS_PER_MONTH)
        series = np.clip(series, 1e-9, None)
        series = series * (liters / (series.sum() * res_h * 3600.0))

        h0 = month * days * steps_day
        out[h0:h0 + days * steps_day, col_of[node]] = series

    idx = pd.RangeIndex(out.shape[0], name="step")
    df = pd.DataFrame(out, index=idx, columns=nodes)
    df.insert(0, "month", np.repeat(np.arange(n_months), days * steps_day))
    return df


def apply_drift_ramp(demand_series: pd.DataFrame, gt_drift_schedule: pd.DataFrame,
                     patterns: dict, time: dict) -> pd.DataFrame:
    """Gradual onset: blend each drifted node's old->new regime linearly over
    ``drift_ramp_days`` at the start of its drift month (a land-use conversion
    is construction-scale, not a step change).

    The ramp is CLAMPED against the remaining horizon:

        ramp_days = min(drift_ramp_days,
                        (n_months - last_switch_month) * days_per_month - 1)

    Without this, a ramp starting near the end of the horizon writes past the
    frame and raises a broadcast shape mismatch. ``growth_chance`` and
    ``max_neighbors_per_month`` decide how late the diffusion front finishes,
    so any horizon or drift-rate change can silently re-break it — hence the
    clamp lives here, where both the schedule and the horizon are in scope,
    rather than at the call site.
    """
    df = demand_series.copy()
    if not len(gt_drift_schedule):
        return df

    steps_day = int(round(24 / time["resolution_h"]))
    days = int(time["days_per_month"])
    n_months = int(time["n_months"])

    last_switch = int(gt_drift_schedule["drift_month"].max())
    available_days = (n_months - last_switch) * days
    ramp_days = min(int(patterns["drift_ramp_days"]), max(available_days - 1, 1))
    ramp_steps = int(ramp_days * steps_day)
    if ramp_steps == 0:
        return df

    for _, row in gt_drift_schedule.iterrows():
        node, m = str(row["node"]), int(row["drift_month"])
        if node not in demand_series.columns:
            continue  # zero-demand trunk junctions carry no portfolio, hence
                      # no demand column: their "drift" has no observable.
        if m == 0:
            continue  # nothing to blend from; build_drift_schedule enforces
                      # warmup_months >= 1, so this is defensive only.
        h0 = m * days * steps_day
        prev_month = df[df["month"] == m - 1][node].to_numpy()
        # Previous regime proxy for the ramp window: the tail of month m-1,
        # tiled if the ramp is longer than one month.
        old = np.resize(prev_month[-ramp_steps:], ramp_steps)
        w = np.linspace(0.0, 1.0, ramp_steps)
        seg = df.loc[h0:h0 + ramp_steps - 1, node].to_numpy()
        df.loc[h0:h0 + ramp_steps - 1, node] = (1 - w) * old + w * seg
    return df
