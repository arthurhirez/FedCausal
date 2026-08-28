"""Dependence oracle: ground truth for the dissertation's core question.

Four artifacts, all catalog outputs:

* ``gt_topology``            — structural truth from the network object.
* ``feature_trajectories``   — MiniRocket feature trajectories per district
                               (the FL-deployable representation, Tier 4).
* ``gt_dependence_battery``  — tidy table: every (pair, kind, method) with a
                               statistic and a circular-shift surrogate
                               p-value. Tiers: 1 linear, 2 nonlinear,
                               3 directional, 4 representation-space.
* ``gt_structure_recovery``  — how well each method's pair ranking recovers
                               the physical topology (AUROC vs open
                               boundaries, Spearman vs hydraulic proximity).
                               The centralized upper bound the federated
                               method will be benchmarked against.

All statistics operate on month-aware deseasonalized residuals: removing the
per-(month x hour x weekday) profile strips the diurnal cycle AND shared slow
structure (seasonality, drift ramps). What remains is fluctuation dependence
— the physically transmitted signal — rather than exogenous-driver
correlation (all clients share the weather).
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import methods as M


# --------------------------------------------------------------------------
# shared signal preparation
# --------------------------------------------------------------------------
def deseasonalize(wide: pd.DataFrame, steps_day: int, month: pd.Series,
                  month_aware: bool = True) -> pd.DataFrame:
    """Remove per-sensor mean profile. Month-aware by default (see module
    docstring); ``month_aware=False`` gives the naive hour-of-day version,
    kept for the exogenous-confound demonstration."""
    hour = wide.index % steps_day
    weekend = ((wide.index // steps_day) % 7) >= 5
    arrays = ([month.reindex(wide.index), hour, weekend] if month_aware
              else [hour, weekend])
    profile = wide.groupby(pd.MultiIndex.from_arrays(arrays)).transform("mean")
    return wide - profile


def district_signals(sensor_series: pd.DataFrame, steps_day: int,
                     month_aware: bool = True) -> dict:
    """{(district, kind): residual signal} — mean of the district's sensor
    residuals per kind. Shared by the battery and the trajectory node."""
    wide = sensor_series.pivot_table(index="step", columns=["district", "kind"],
                                     values="observed")
    month = (sensor_series.drop_duplicates("step").set_index("step")["month"]
             .sort_index())
    resid = deseasonalize(wide, steps_day, month, month_aware)
    grouped = resid.T.groupby(level=[0, 1]).mean().T
    return {key: grouped[key].to_numpy() for key in grouped.columns}


# --------------------------------------------------------------------------
# topology (unchanged behaviour)
# --------------------------------------------------------------------------
def topology_features(wn, districts: dict, gt_boundaries: pd.DataFrame,
                      sensors: dict) -> pd.DataFrame:
    import networkx as nx

    closed = set(gt_boundaries.loc[gt_boundaries["closed"], "pipe"])
    G = nx.Graph()
    for name in wn.pipe_name_list:
        pipe = wn.get_link(name)
        if name not in closed:
            G.add_edge(pipe.start_node_name, pipe.end_node_name,
                       weight=pipe.length)

    names = list(districts["districts"].keys())
    rows = []
    for i, da in enumerate(names):
        for db in names[i + 1:]:
            pair = gt_boundaries[(gt_boundaries["district_a"] == min(da, db)) &
                                 (gt_boundaries["district_b"] == max(da, db))]
            dists = []
            for na in sensors[da]["pressure"]:
                for nb in sensors[db]["pressure"]:
                    try:
                        dists.append(nx.shortest_path_length(
                            G, str(na), str(nb), weight="weight"))
                    except nx.NetworkXNoPath:
                        pass
            rows.append({
                "district_a": da, "district_b": db,
                "boundary_pipes": int(len(pair)),
                "open_boundary_pipes": int((~pair["closed"]).sum()),
                "hydraulic_distance_m": float(np.mean(dists)) if dists else np.inf,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tier 4 input: MiniRocket feature trajectories
# --------------------------------------------------------------------------
def minirocket_trajectories(sensor_series: pd.DataFrame, oracle: dict,
                            time: dict, seed: int) -> pd.DataFrame:
    """Fixed-kernel features on synchronized sliding windows of the residual
    signals -> one feature trajectory per (district, kind).

    FL rationale: MiniRocket kernels are fixed and deterministic given the
    seed, so every client shares an identical representation space with zero
    training and zero communication. Two deployment caveats, recorded here
    for the dissertation: bias quantiles are fitted from data (a federated
    deployment fixes them from a reference series — emulated below by
    fitting on the first district only), and the joint PCA would become a
    federated PCA or a fixed random projection.
    """
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.decomposition import PCA

    cfg = oracle["minirocket"]
    steps_day = int(round(24 / time["resolution_h"]))
    win = int(cfg["window_h"] / time["resolution_h"])
    stride = int(cfg["stride_h"] / time["resolution_h"])

    signals = district_signals(sensor_series, steps_day)
    frames = []
    for kind in sorted({k for _, k in signals}):
        keys = sorted([d for d, k in signals if k == kind])
        n = min(len(signals[(d, kind)]) for d in keys)
        starts = np.arange(0, n - win + 1, stride)
        stacked = {
            d: np.stack([signals[(d, kind)][s:s + win] for s in starts])[:, None, :]
            for d in keys
        }
        transformer = MiniRocket(n_kernels=cfg["n_kernels"], random_state=seed)
        transformer.fit(stacked[keys[0]])  # reference-client bias quantiles
        feats = {d: transformer.transform(X) for d, X in stacked.items()}

        # Joint standardization + joint PCA => one shared space per kind.
        all_f = np.vstack([feats[d] for d in keys])
        mu, sd = all_f.mean(0), all_f.std(0) + 1e-12
        pca = PCA(n_components=cfg["pca_dims"], random_state=seed)
        pca.fit((all_f - mu) / sd)
        for d in keys:
            Z = pca.transform((feats[d] - mu) / sd)
            df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
            df.insert(0, "window", np.arange(len(Z)))
            df.insert(0, "kind", kind)
            df.insert(0, "district", d)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# the battery
# --------------------------------------------------------------------------
def _roll_matrix_pvalue(stat_fn, X, Y, n_sur, rng):
    """Surrogate p-value for trajectory statistics: circularly shift X's
    window order, preserving each trajectory's autocorrelation."""
    obs = stat_fn(X, Y)
    lo = max(1, len(X) // 10)
    null = np.empty(n_sur)
    for i in range(n_sur):
        shift = int(rng.integers(lo, len(X) - lo))
        null[i] = stat_fn(np.roll(X, shift, axis=0), Y)
    p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_sur + 1.0)
    return float(obs), float(p)


def dependence_battery(sensor_series: pd.DataFrame,
                       feature_trajectories: pd.DataFrame,
                       oracle: dict, time: dict, seed: int) -> pd.DataFrame:
    steps_day = int(round(24 / time["resolution_h"]))
    max_lag = int(oracle["max_lag_h"] / time["resolution_h"])
    tiers = set(oracle["tiers"])
    n_sur = oracle["n_surrogates"]
    n_sur_exp = oracle["n_surrogates_expensive"]
    sub, sub_dcor = oracle["subsample"], oracle["dcor_subsample"]

    signals = district_signals(sensor_series, steps_day)
    kinds = sorted({k for _, k in signals})
    rows = []

    for kind in kinds:
        keys = sorted([d for d, k in signals if k == kind])
        sig = {d: signals[(d, kind)] for d in keys}

        # ---- joint: precision-matrix partial correlation (Tier 1) --------
        if 1 in tiers:
            partial = M.precision_partial_corr(
                np.column_stack([sig[d] for d in keys]))
            for i, da in enumerate(keys):
                for j in range(i + 1, len(keys)):
                    rows.append(dict(kind=kind, district_a=da,
                                     district_b=keys[j], tier=1,
                                     method="partial_corr_precision",
                                     direction="sym",
                                     statistic=float(partial[i, j]),
                                     p_value=np.nan, extra=np.nan))

        for i, da in enumerate(keys):
            for db in keys[i + 1:]:
                a, b = sig[da], sig[db]
                rng = np.random.default_rng(
                    [seed, zlib.crc32(f"{da}|{db}|{kind}".encode())])
                base = dict(kind=kind, district_a=da, district_b=db)

                if 1 in tiers:
                    for name, fn in [("pearson", M.pearson),
                                     ("spearman", M.spearman)]:
                        s, p = M.circular_shift_pvalue(fn, a, b, n_sur, rng)
                        rows.append(dict(**base, tier=1, method=name,
                                         direction="sym", statistic=s,
                                         p_value=p, extra=np.nan))
                    (r, lag), p = M.circular_shift_pvalue(
                        lambda x, y: M.max_lagged_xcorr(x, y, max_lag),
                        a, b, n_sur, rng)
                    rows.append(dict(**base, tier=1, method="max_lagged_xcorr",
                                     direction="sym", statistic=r, p_value=p,
                                     extra=float(lag)))

                if 2 in tiers:
                    ad, bd = M.strided(a, sub_dcor), M.strided(b, sub_dcor)
                    s, p = M.circular_shift_pvalue(
                        M.distance_correlation, ad, bd, n_sur_exp, rng)
                    rows.append(dict(**base, tier=2, method="distance_corr",
                                     direction="sym", statistic=s, p_value=p,
                                     extra=np.nan))
                    am, bm = M.strided(a, sub), M.strided(b, sub)
                    s, p = M.circular_shift_pvalue(
                        M.mutual_information, am, bm, n_sur_exp, rng)
                    rows.append(dict(**base, tier=2, method="mutual_info",
                                     direction="sym", statistic=s, p_value=p,
                                     extra=np.nan))

                if 3 in tiers:
                    ac, bc = M.contiguous(a, sub), M.contiguous(b, sub)
                    lags = oracle["granger_lags"]
                    ccm = oracle["ccm"]
                    for direction, (x, y) in [(f"{da}->{db}", (ac, bc)),
                                              (f"{db}->{da}", (bc, ac))]:
                        s, p = M.circular_shift_pvalue(
                            lambda u, v: M.granger_f(u, v, lags),
                            x, y, n_sur_exp, rng)
                        rows.append(dict(**base, tier=3, method="granger_f",
                                         direction=direction, statistic=s,
                                         p_value=p, extra=np.nan))

                        def ccm_fn(u, v):
                            return M.ccm_skill(u, v, ccm["embed_dim"],
                                               ccm["tau"], ccm["lib_sizes"],
                                               ccm["n_neighbors"],
                                               np.random.default_rng(seed))
                        (skill, conv), p = M.circular_shift_pvalue(
                            ccm_fn, x, y, n_sur_exp, rng)
                        rows.append(dict(**base, tier=3, method="ccm_skill",
                                         direction=direction, statistic=skill,
                                         p_value=p, extra=conv))

        if 4 in tiers and len(feature_trajectories):
            traj = feature_trajectories[feature_trajectories["kind"] == kind]
            fcols = [c for c in traj.columns if c.startswith("f")]
            mats = {d: g.sort_values("window")[fcols].to_numpy()
                    for d, g in traj.groupby("district")}
            for i, da in enumerate(keys):
                for db in keys[i + 1:]:
                    if da not in mats or db not in mats:
                        continue
                    X, Y = mats[da], mats[db]
                    rng = np.random.default_rng(
                        [seed, 4, zlib.crc32(f"{da}|{db}|{kind}".encode())])
                    base = dict(kind=kind, district_a=da, district_b=db)

                    s, p = _roll_matrix_pvalue(M.rv_coefficient, X, Y,
                                               n_sur_exp, rng)
                    rows.append(dict(**base, tier=4, method="minirocket_rv",
                                     direction="sym", statistic=s, p_value=p,
                                     extra=np.nan))
                    s, p = _roll_matrix_pvalue(M.trajectory_dcor, X, Y,
                                               max(10, n_sur_exp // 2), rng)
                    rows.append(dict(**base, tier=4, method="minirocket_dcor",
                                     direction="sym", statistic=s, p_value=p,
                                     extra=np.nan))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# structure recovery: score every method against physical truth
# --------------------------------------------------------------------------
def structure_recovery(battery: pd.DataFrame,
                       gt_topology: pd.DataFrame) -> pd.DataFrame:
    """Per (kind, method): AUROC of |statistic| against 'pair shares an open
    boundary', and Spearman of |statistic| against hydraulic proximity.
    Directional methods are aggregated per pair by max |statistic|."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    topo = gt_topology.copy()
    topo["positive"] = topo["open_boundary_pipes"] > 0
    finite = np.isfinite(topo["hydraulic_distance_m"])
    cap = topo.loc[finite, "hydraulic_distance_m"].max() * 2 if finite.any() else 1.0
    topo["proximity"] = -topo["hydraulic_distance_m"].clip(upper=cap)

    agg = (battery.assign(abs_stat=battery["statistic"].abs())
           .groupby(["kind", "tier", "method", "district_a", "district_b"],
                    as_index=False)["abs_stat"].max()
           .merge(topo, on=["district_a", "district_b"]))

    rows = []
    for (kind, tier, method), grp in agg.groupby(["kind", "tier", "method"]):
        y = grp["positive"].to_numpy()
        auroc = (float(roc_auc_score(y, grp["abs_stat"]))
                 if 0 < y.sum() < len(y) else np.nan)
        rho = float(spearmanr(grp["abs_stat"], grp["proximity"]).statistic)
        rows.append(dict(kind=kind, tier=int(tier), method=method, auroc=auroc,
                         spearman_vs_proximity=rho, n_pairs=len(grp)))
    return pd.DataFrame(rows).sort_values(["kind", "tier", "method"],
                                          ignore_index=True)
