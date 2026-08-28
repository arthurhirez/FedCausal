"""Dependence-aware drift attribution — the corrector ladder (C0..C3).

Every client's prototype trajectory is local domain + network state (the
spillover). Each corrector separates them at increasing sophistication and
communication budget; all are scored by the SAME harness on both the coupled
baseline (must fix the contamination) and the isolated world (must be a
near-no-op: a correction that manufactures artifacts when clients are truly
independent is worse than none).

C0  uncorrected      cosine delta_first on raw prototypes (the status quo).
C1  peer z-score     per month, robust z of a client's delta against the
                     peer population: drift = moving DIFFERENTLY, not moving.
C2  median removal   subtract the per-month MEDIAN prototype (the robust
                     common mode: with a drifted minority, a mean reference
                     absorbs 1/K of the drift signal and leaks -1/K into
                     every stable client; the median ignores the minority).
                     Deltas on residuals use EUCLIDEAN distance — for
                     residual vectors the displacement magnitude IS the
                     signal, whereas cosine is unstable near the origin.
C3  peer prediction  ridge-predict each client's residualized latent
                     trajectory from its peers, weights fitted on the
                     REFERENCE MONTHS ONLY (the weights encode baseline
                     hydraulic coupling, never the drift echo); detect on the
                     prediction residual's monthly prototypes. Two variants:
                     all peers, and peers masked by the partial-RV dependence
                     graph — if the masked variant matches the full one,
                     dependence DETECTION is functionally necessary.

Attribution (C2) is a first-class output: per (client, month),
``delta_total = delta_common (network) + delta_local (yours)``.

Caveat (learned from the planted-world tests): ``delta_first`` on
autocorrelated residuals climbs toward its stationary plateau over roughly
the noise decorrelation time — the reference window used to calibrate the
threshold must cover that time, or the plateau reads as sustained drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedwater.pipelines.drift_detection.nodes import _cosine_dist


def _final_prototypes(prototype_history: pd.DataFrame):
    fcols = [c for c in prototype_history.columns if c.startswith("f")]
    final = prototype_history[prototype_history["round"]
                              == prototype_history["round"].max()]
    protos = {c: g.sort_values("month")[fcols].to_numpy()
              for c, g in final.groupby("client")}
    months = np.sort(final["month"].unique())
    return protos, months


def _signal_rows(corrector, client, months, delta_first, delta_roll):
    return [dict(corrector=corrector, client=client, month=int(m),
                 delta_first=float(df_), delta_roll=float(dr))
            for m, df_, dr in zip(months, delta_first, delta_roll)]


def _euclid_deltas(vecs: np.ndarray):
    d_first = np.linalg.norm(vecs - vecs[0], axis=1)
    d_roll = np.r_[np.nan, np.linalg.norm(np.diff(vecs, axis=0), axis=1)]
    return d_first, d_roll


def apply_correctors(prototype_history: pd.DataFrame,
                     latents_resid: pd.DataFrame,
                     client_dependence: pd.DataFrame,
                     fl: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = fl["attribution"]
    protos, months = _final_prototypes(prototype_history)
    clients = sorted(protos)
    rows: list[dict] = []

    # ---- C0: uncorrected (cosine, as the base detector) ------------------
    c0 = {}
    for c in clients:
        v = protos[c]
        d_first = np.array([_cosine_dist(v[i], v[0]) if i else 0.0
                            for i in range(len(v))])
        d_roll = np.r_[np.nan, [_cosine_dist(v[i], v[i - 1])
                                for i in range(1, len(v))]]
        c0[c] = d_first
        rows += _signal_rows("C0_uncorrected", c, months, d_first, d_roll)

    # ---- C1: robust peer z-score of the C0 deltas -------------------------
    mat = np.stack([c0[c] for c in clients])            # (K, M)
    med = np.median(mat, axis=0)
    mad = np.median(np.abs(mat - med), axis=0) * 1.4826 + 1e-9
    z = (mat - med) / mad
    for i, c in enumerate(clients):
        rows += _signal_rows("C1_peer_zscore", c, months, z[i],
                             np.r_[np.nan, np.diff(z[i])])

    # ---- C2: median common-mode removal + attribution ---------------------
    stack = np.stack([protos[c] for c in clients])      # (K, M, D)
    common = np.median(stack, axis=0)                   # (M, D)
    attribution = []
    delta_common, _ = _euclid_deltas(common)
    scale = {}
    for i, c in enumerate(clients):
        resid = stack[i] - common
        d_first, d_roll = _euclid_deltas(resid)
        # scale-normalize by the client's reference-month roll magnitude
        ref = d_roll[1:cfg["reference_months"]]
        scale[c] = max(np.nanmedian(ref), 1e-9)
        rows += _signal_rows("C2_median_removal", c, months,
                             d_first / scale[c], d_roll / scale[c])
        total, _ = _euclid_deltas(stack[i])
        for m, t_, cm_, lo_ in zip(months, total, delta_common, d_first):
            attribution.append(dict(client=c, month=int(m), delta_total=t_,
                                    delta_common=cm_, delta_local=lo_))

    # ---- C2b: leave-one-out median (asymmetric reference) -----------------
    for i, c in enumerate(clients):
        others = np.median(np.delete(stack, i, axis=0), axis=0)
        resid = stack[i] - others
        d_first, d_roll = _euclid_deltas(resid)
        sc = max(np.nanmedian(d_roll[1:cfg["reference_months"]]), 1e-9)
        rows += _signal_rows("C2b_loo_median", c, months,
                             d_first / sc, d_roll / sc)

    # ---- C4: Design-B loop — rank-1-with-gains + client-sparse ------------
    # Model: Y_c,m := p_c,m - p_c,0  ~  lambda_c * g_m + S_c,m,  S sparse in c.
    # The median (C2) assumes lambda_c == 1 for everyone; the real spillover
    # has heterogeneous hydraulic gains, so the loop ESTIMATES them,
    # alternating (g, lambda, weights). The per-client weight trajectory is
    # itself the drifter identification: the drifted client's weight
    # collapses as its sparse component grows.
    Y = stack - stack[:, :1, :]                        # (K, M, D)
    K = len(clients)
    lam = np.ones(K)
    w = np.ones(K)
    diag_rows = []
    for it in range(cfg["loop_iterations"]):
        wl = w * lam
        g = np.einsum("k,kmd->md", wl, Y) / max((wl * lam).sum(), 1e-9)
        S = Y - lam[:, None, None] * g[None]
        score = np.linalg.norm(S, axis=2).mean(axis=1)          # (K,)
        med = max(np.median(score), 1e-9)
        w = 1.0 / (1.0 + (score / med) ** 2)                    # soft-sparse
        gg = np.einsum("md,md->", g, g)
        lam = np.einsum("kmd,md->k", Y, g) / max(gg, 1e-9)
        lam = np.clip(lam, 0.0, None)                           # gains >= 0
        for c_, wv, lv, sv in zip(clients, w, lam, score):
            diag_rows.append(dict(iteration=it, client=c_, weight=wv,
                                  gain=lv, sparse_score=sv))
    for i, c in enumerate(clients):
        d_first, d_roll = _euclid_deltas(S[i])
        sc = max(np.nanmedian(d_roll[1:cfg["reference_months"]]), 1e-9)
        rows += _signal_rows("C4_sparse_loop", c, months,
                             d_first / sc, d_roll / sc)
    loop_diagnostics = pd.DataFrame(diag_rows)

    # ---- C3: ridge peer prediction on latent trajectories -----------------
    fcols = [c for c in latents_resid.columns if c.startswith("f")]
    mats = {c: g.sort_values("window") for c, g in
            latents_resid.groupby("district")}
    X = {c: g[fcols].to_numpy() for c, g in mats.items()}
    month_of = {c: g["month"].to_numpy() for c, g in mats.items()}
    lam = cfg["ridge_lambda"]

    # dependence mask from partial RV (q-threshold; fallback = all peers)
    prv = client_dependence[(client_dependence["level"] == "T")
                            & (client_dependence["method"] == "partial_rv")]
    edges = {(r.district_a, r.district_b) for r in prv.itertuples()
             if r.q_value < cfg["mask_q_threshold"]}
    def peers(c, masked):
        ps = [p for p in clients if p != c]
        if not masked:
            return ps
        sel = [p for p in ps if (min(c, p), max(c, p)) in edges
               or (c, p) in edges or (p, c) in edges]
        return sel or ps                                # documented fallback

    for masked, name in [(False, "C3_peer_pred_full"),
                         (True, "C3_peer_pred_masked")]:
        for c in clients:
            Z = np.hstack([X[p] for p in peers(c, masked)])
            ref = month_of[c] < cfg["reference_months"]
            Zr, Yr = Z[ref], X[c][ref]
            W = np.linalg.solve(Zr.T @ Zr + lam * np.eye(Z.shape[1]),
                                Zr.T @ Yr)
            resid = X[c] - Z @ W
            proto = np.stack([resid[month_of[c] == m].mean(axis=0)
                              for m in months])
            d_first, d_roll = _euclid_deltas(proto)
            ref_scale = max(np.nanmedian(d_roll[1:cfg["reference_months"]]),
                            1e-9)
            rows += _signal_rows(name, c, months, d_first / ref_scale,
                                 d_roll / ref_scale)

    return (pd.DataFrame(rows), pd.DataFrame(attribution),
            loop_diagnostics)


def evaluate_correctors(corrected_drift_signals: pd.DataFrame,
                        gt_drift_schedule: pd.DataFrame,
                        fl: dict) -> pd.DataFrame:
    """Run the SAME drift evaluation per corrector; one comparable table."""
    from fedwater.pipelines.drift_detection.nodes import evaluate_drift

    out = []
    for corrector, sig in corrected_drift_signals.groupby("corrector"):
        rep = evaluate_drift(sig.drop(columns="corrector"),
                             gt_drift_schedule, fl)
        rep.insert(0, "corrector", corrector)
        out.append(rep)
    report = pd.concat(out, ignore_index=True)
    if report[report["client"] == "__summary__"]["rank"].isna().any():
        raise AssertionError("Corrector evaluation produced NaN ranks.")
    return report
