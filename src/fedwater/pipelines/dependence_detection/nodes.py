"""Federated dependence detection & quantification.

The experimental axis is the communication budget:

* **Level P — prototypes only.** The server already holds per-month
  prototypes (FPL communicates them anyway; zero marginal cost). Signal:
  co-movement of month-to-month prototype *displacements* (differencing is
  the only detrending available at 24 points/client).
* **Level T — latent trajectories.** Clients send per-window latents. The
  encoder is the FedAvg'd global model, so all clients' latents live in ONE
  shared space by construction — alignment is free, exactly like
  MiniRocket's fixed kernels in the oracle's Tier 4.
* **Level R — centralized raw.** The oracle battery (already computed);
  Levels P/T are scored against it as the upper bound.

Quantification: the RV coefficient is the headline number — bounded [0, 1],
the multivariate generalization of r^2, read as "fraction of shared
co-inertia between the two clients' dynamics". ``partial RV`` conditions on
all other clients (direct vs mediated coupling). Every statistic carries a
circular-shift surrogate p-value and a BH-FDR q-value across the pair
family, plus the communication cost (floats sent per client) of its level.

Mandatory preprocessing (the Act-V lesson): month-aware residualization of
the trajectories — per client, per latent dimension, remove the mean per
(month x window-phase-of-day). Without it, shared seasonality masquerades as
coupling. Uses each client's OWN means only: client-local, federated-legit.
(Windows span 7 days, so weekday phase is balanced by construction.)

Deployment note: in-process we compute pairwise statistics directly; in a
real deployment Level T means trajectories go to the server — that is the
budget statement. Secure pairwise computation / DP noise are future knobs.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from fedwater.pipelines.dependence_oracle import methods as M

from . import fed_methods as F


def residualize_latents(latent_trajectories: pd.DataFrame,
                        fl: dict, time: dict) -> pd.DataFrame:
    """Client-local month-aware residualization of latent trajectories."""
    steps_day = int(24 / time["resolution_h"])
    fcols = [c for c in latent_trajectories.columns if c.startswith("f")]
    df = latent_trajectories.copy()
    df["phase"] = df["window"] % steps_day
    df["dow"] = (df["window"] // steps_day) % 7      # start day-of-week
    out = []
    for _, grp in df.groupby(["district", "kind"]):   # own data only
        resid = grp.copy()
        resid[fcols] = grp[fcols] - grp.groupby(["month", "phase", "dow"])[
            fcols].transform("mean")
        out.append(resid)
    result = pd.concat(out, ignore_index=True).drop(columns=["phase", "dow"])
    if result[fcols].isna().any().any():
        raise AssertionError("Residualization produced NaNs.")
    return result


def _aligned_matrices(traj: pd.DataFrame) -> dict[str, np.ndarray]:
    """{client: (n x D)} on the common window grid (asserted synchronized)."""
    fcols = [c for c in traj.columns if c.startswith("f")]
    grids = {c: g.sort_values("window") for c, g in traj.groupby("district")}
    windows = None
    for c, g in grids.items():
        w = g["window"].to_numpy()
        if windows is None:
            windows = w
        elif not np.array_equal(windows, w):
            raise AssertionError(f"Window grids differ for {c} — clients must "
                                 "be preprocessed identically.")
    return {c: g[fcols].to_numpy() for c, g in grids.items()}


def federated_dependence(latents_resid: pd.DataFrame,
                         prototype_history: pd.DataFrame,
                         fl: dict, seed: int) -> pd.DataFrame:
    cfg = fl["dependence"]
    rows = []

    # ---------------- Level T: latent trajectories -----------------------
    for kind, traj in latents_resid.groupby("kind"):
        mats = _aligned_matrices(traj)
        clients = sorted(mats)
        n, d = next(iter(mats.values())).shape
        cost_t = n * d
        for i, a in enumerate(clients):
            for b in clients[i + 1:]:
                X, Y = mats[a], mats[b]
                Z = np.hstack([mats[c] for c in clients if c not in (a, b)])
                rng = np.random.default_rng(
                    [seed, zlib.crc32(f"{a}|{b}|{kind}".encode())])
                base = dict(level="T", kind=kind, district_a=a, district_b=b,
                            comm_floats_per_client=cost_t)

                s, p = F.roll_pvalue(M.rv_coefficient, X, Y,
                                     cfg["n_surrogates"], rng)
                rows.append(dict(**base, method="rv", statistic=s, p_value=p,
                                 extra=np.nan))
                s, p = F.roll_pvalue(M.trajectory_dcor, X, Y,
                                     cfg["n_surrogates_expensive"], rng)
                rows.append(dict(**base, method="trajectory_dcor",
                                 statistic=s, p_value=p, extra=np.nan))
                s, p = F.roll_pvalue(lambda u, v: F.partial_rv(u, v, Z),
                                     X, Y, cfg["n_surrogates_expensive"], rng)
                rows.append(dict(**base, method="partial_rv", statistic=s,
                                 p_value=p, extra=np.nan))
                (r, lag), p = F.roll_pvalue(
                    lambda u, v: M.max_lagged_xcorr(
                        F.first_pc(u), F.first_pc(v),
                        cfg["max_lag_windows"]),
                    X, Y, cfg["n_surrogates"], rng)
                rows.append(dict(**base, method="pc1_lagged_xcorr",
                                 statistic=r, p_value=p, extra=float(lag)))

    # ---------------- Level P: prototype displacements -------------------
    fcols = [c for c in prototype_history.columns if c.startswith("f")]
    final = prototype_history[prototype_history["round"]
                              == prototype_history["round"].max()]
    disp = {}
    for c, g in final.groupby("client"):
        vecs = g.sort_values("month")[fcols].to_numpy()
        disp[c] = np.diff(vecs, axis=0)          # (months-1, D)
    clients = sorted(disp)
    n_p, d_p = next(iter(disp.values())).shape
    n_rounds = prototype_history["round"].nunique()
    cost_p = n_rounds * (n_p + 1) * d_p          # what FPL already sends
    for i, a in enumerate(clients):
        for b in clients[i + 1:]:
            rng = np.random.default_rng(
                [seed, 99, zlib.crc32(f"{a}|{b}".encode())])
            base = dict(level="P", kind="prototype_displacement",
                        district_a=a, district_b=b,
                        comm_floats_per_client=cost_p)
            s, p = F.roll_pvalue(M.rv_coefficient, disp[a], disp[b],
                                 cfg["n_surrogates"], rng)
            rows.append(dict(**base, method="rv", statistic=s, p_value=p,
                             extra=np.nan))
            s, p = F.roll_pvalue(M.trajectory_dcor, disp[a], disp[b],
                                 cfg["n_surrogates_expensive"], rng)
            rows.append(dict(**base, method="trajectory_dcor", statistic=s,
                             p_value=p, extra=np.nan))

    out = pd.DataFrame(rows)
    out["q_value"] = np.nan
    for (_, _, _), idx in out.groupby(["level", "kind", "method"]).groups.items():
        out.loc[idx, "q_value"] = F.bh_fdr(out.loc[idx, "p_value"].to_numpy())
    return out


def evaluate_dependence(client_dependence: pd.DataFrame,
                        gt_topology: pd.DataFrame,
                        gt_dependence_battery: pd.DataFrame) -> pd.DataFrame:
    """Score federated levels against structure and against the oracle."""
    from scipy.stats import spearmanr

    from fedwater.pipelines.dependence_oracle.nodes import structure_recovery

    # structure recovery reuses the oracle scorer (tier codes: P=5, T=6)
    fed = client_dependence.copy()
    fed["tier"] = fed["level"].map({"P": 5, "T": 6})
    fed["direction"] = "sym"
    scores = structure_recovery(
        fed[["kind", "tier", "method", "district_a", "district_b",
             "statistic", "direction", "p_value"]], gt_topology)
    scores.insert(0, "evaluation", "structure_recovery")

    # oracle rank agreement: federated pair ranking vs each centralized method
    rows = []
    fed_t = fed[(fed["level"] == "T")]
    for method, g in fed_t.groupby("method"):
        f = g.set_index(["district_a", "district_b"])["statistic"].abs()
        for (kind, cm), og in gt_dependence_battery.groupby(["kind", "method"]):
            o = (og.assign(abs_stat=og["statistic"].abs())
                 .groupby(["district_a", "district_b"])["abs_stat"].max())
            joined = pd.concat([f, o], axis=1, join="inner")
            if len(joined) < 3:
                continue
            rho = float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)
            rows.append(dict(evaluation="oracle_rank_agreement", kind=kind,
                             tier=6, method=f"{method}~vs~{cm}", auroc=np.nan,
                             spearman_vs_proximity=rho, n_pairs=len(joined)))
    return pd.concat([scores, pd.DataFrame(rows)], ignore_index=True)
