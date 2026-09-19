"""verify.py -- does the chosen set see THIS world's drift as its labels say?

Selection never reads the target world. This runs afterwards, on the world's
own series, and asks one question per sensor: when the world's drift district
changed, how much of that change reached the gauge? It is the POC's
``world_response`` estimator applied to the real world -- one column of the
gain matrix -- next to the label the probe stack predicted for that column
(``w_<drift district>``).

It is a REPORT, never a gate: a world whose drift has not settled inside its
horizon gets ``status = unsettled`` rows and the pipeline continues.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mixture_probe as mp
from . import signal_probe as sp

__all__ = ["verify_placement"]


def verify_placement(placement: pd.DataFrame, pressures: pd.DataFrame,
                     flows: pd.DataFrame, demand: pd.DataFrame,
                     schedule: pd.DataFrame, districts: dict, time: dict,
                     patterns: dict, settle: dict,
                     null_factor: float) -> pd.DataFrame:
    base_cols = ["district", "kind", "slot", "slot_class", "filled_as",
                 "sensor", "element"]
    out = placement[base_cols].copy()
    if schedule is None or not len(schedule):
        out["status"] = "no_drift"
        return out
    tgt = str(schedule.sort_values("drift_month")["district"].iloc[0])
    out.insert(0, "drift_district", tgt)
    label = f"w_{tgt}"
    out["expected_w_drift"] = (placement[label].to_numpy()
                               if label in placement.columns else np.nan)

    P = sp.Probe(districts=districts["districts"],
                 assets=districts.get("assets") or {}, demand=demand,
                 pressures=pressures, flows=flows, schedule=schedule,
                 time=time, params={"patterns": patterns},
                 drift={"tgt_district": tgt}, label="world")
    cand = pd.DataFrame({
        "id": placement["sensor"].to_numpy(),
        "kind": placement["kind"].to_numpy(),
        "district": placement["district"].to_numpy(),
        "role": placement["role"].fillna("").to_numpy(),
        "element": placement["element"].astype(str).to_numpy()})
    n, sm = int(time["n_months"]), P.steps_month
    final0 = int(P.phases()["final"][0])
    need = int(settle.get("min_months", 3))
    if float((patterns or {}).get("seasonality_scale", 0.0)) > 0:
        # seasonality on: whole-year windows on both sides, as the stacks use
        try:
            base, window = mp.season_windows(P, pad=int(settle["pad"]))
        except ValueError as exc:
            out["status"] = f"unsettled: {exc}"[:300]
            return out
    else:
        # The settle scan follows the drifting district's AGGREGATE profile,
        # which can flip early (the seed is the district's largest consumer)
        # while the front is still converting nodes. The window therefore
        # starts at the later of the settle month and the drift's completion.
        try:
            s_lo, _ = mp.settled_window(
                P, ref_months=int(settle["ref_months"]),
                slack=float(settle["slack"]), pad=int(settle["pad"]),
                min_months=0)
            start = max(s_lo // sm, final0)
        except ValueError:
            start = final0
        if n - start < need:
            out["status"] = (f"unsettled: {n - start} settled month(s) after "
                             f"the drift completes (month {final0}), {need} "
                             "needed; the world horizon is too short to verify")
            return out
        base, window = None, (start * sm, n * sm)
    r = mp.world_response(P, cand, slack=float(settle["slack"]),
                          pad=int(settle["pad"]),
                          ref_months=int(settle["ref_months"]),
                          window=window, baseline=base)

    r = r.set_index("id").reindex(out["sensor"])
    out["observed_g"] = r["g"].to_numpy()
    out["observed_proj"] = r["proj"].to_numpy()
    out["noise_proj"] = r["nu"].to_numpy()
    out["dz_norm"] = r["dz_norm"].to_numpy()
    out["null"] = r["null"].to_numpy()
    out["responded"] = (r["dz_norm"] > null_factor * r["null"]).to_numpy()
    out["clears_noise"] = (r["proj"].abs() > r["nu"]).to_numpy()
    out["settled_from_month"] = r["settle_start_month"].to_numpy()

    # agreement between prediction and observation, across the chosen set
    ok = out[["expected_w_drift", "observed_g"]].dropna()
    rho = (float(ok.rank().corr().iloc[0, 1]) if len(ok) > 2
           and ok["expected_w_drift"].nunique() > 1 else np.nan)
    out["spearman_expected_vs_observed"] = rho
    out["status"] = "ok"
    return out
