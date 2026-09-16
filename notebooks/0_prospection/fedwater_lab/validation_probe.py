"""validation_probe.py -- does the dependence estimator measure what it claims?

Three checks, none of which needs the drift engine touched.

**Excitation space** (in `mixture_probe`, exercised here). The shape-space
elasticity divides by the norm of the district's z-scored demand change. On
Graeme that left pressure tracking the volume level factor 9.19**beta almost
exactly, so the "per unit of excitation" normalisation was not removing the
excitation. The native-space form divides by the demand change in L/s, which
carries level and shape together and is the linearised operator itself.

**Placebo.** The same estimator run on data where nothing drifted: the real
drift directions u_k are kept, but the sensor delta is taken between two
halves of the SETTLED window, where the regime is stationary. Every entry of
the resulting matrix should be ~0, diagonal included. If it is not, the
pipeline manufactures dependence and no amount of interpretation saves the
real matrix.

**Superposition.** One world where two districts drift together. If
`d_s^(AB) ~= d_s^(A) + d_s^(B)`, the single-district design generalises to the
real case where several clients move at once; if not, the dependence matrix is
only valid one drift at a time and that has to be said out loud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import landuse_world as lw
import signal_probe as sp
import mixture_probe as mp

__all__ = ["delta_vectors", "build_placebo", "placebo_report",
           "superposed_cfg", "load_or_build", "superposition",
           "excitation_report", "plot_excitation", "plot_placebo",
           "plot_superposition", "verdict", "level_ratios"]

EPS = 1e-12


# ==========================================================================
# shared: the response vector of every gauge, in either space
# ==========================================================================
def delta_vectors(P: sp.Probe, cand: pd.DataFrame, space: str = "native",
                  windows: tuple | None = None):
    """(sensor deltas, district demand deltas) between two windows.

    Defaults to init -> settled. `windows` overrides both, which is how the
    placebo swaps in two halves of a stationary stretch.
    """
    if windows is None:
        w0, w1 = mp.baseline_window(P), mp.settled_window(P)
    else:
        w0, w1 = windows
    S0, D0, _ = mp._profile(P, cand, *w0)
    S1, D1, _ = mp._profile(P, cand, *w1)
    if space == "shape":
        return sp.zscore(S1) - sp.zscore(S0), sp.zscore(D1) - sp.zscore(D0)
    return S1 - S0, D1 - D0


def _halves(lo: int, hi: int, steps_day: int):
    """Split a window into two whole-day halves."""
    mid = lo + ((hi - lo) // (2 * steps_day)) * steps_day
    return (lo, mid), (mid, hi)


def _quarters(lo: int, hi: int, steps_day: int):
    q = ((hi - lo) // (4 * steps_day)) * steps_day
    return (lo, lo + q), (lo + q, lo + 2 * q), (lo + 2 * q, lo + 3 * q), \
        (lo + 3 * q, hi)


# ==========================================================================
# 1. PLACEBO
# ==========================================================================
def build_placebo(probes: dict, verbose: bool = True) -> dict:
    """A `base` dict with the same schema as `build_mixture`, from no-drift data.

    For each world k the real excitation direction u_k is kept -- the question
    is whether the estimator reports a response ALONG A REAL DIRECTION when the
    sensor did not actually change. The sensor delta comes from the first half
    of the settled window against the second, where the regime is stationary by
    construction (that is what `settled_window` certified). The noise floor
    comes from the two quarters of the first half, so it is independent of the
    delta it gates -- using the same split for both would make the soft
    threshold return exactly zero and the test vacuous.

    Because the schema matches, every read-out in `mixture_probe` --
    `elasticity`, `dependence`, `response_matrix` -- runs on the result
    unchanged.
    """
    districts = sorted(probes)
    P0 = probes[districts[0]]
    ref = sp.candidates(P0)
    common = set(ref["id"])
    for d in districts[1:]:
        common &= set(sp.candidates(probes[d])["id"])
    cand = ref[ref["id"].isin(common)].reset_index(drop=True)

    frames = []
    for d in districts:
        P = probes[d]
        sd = P.steps_day
        lo1, hi1 = mp.settled_window(P)
        hA, hB = _halves(lo1, hi1, sd)
        q1, q2, _, _ = _quarters(lo1, hi1, sd)

        # real direction, from the real drift
        _, dD = delta_vectors(P, cand, "shape")
        _, dDn = delta_vectors(P, cand, "native")
        k = list(P.district_demand.columns).index(P.drift["tgt_district"])
        rows = {}
        for space, DD in (("shape", dD), ("native", dDn)):
            u = DD[k] / max(float(np.linalg.norm(DD[k])), EPS)
            exc = float(np.linalg.norm(DD[k]))
            dS, _ = delta_vectors(P, cand, space, windows=(hA, hB))
            dN, _ = delta_vectors(P, cand, space, windows=(q1, q2))
            # the noise window is half the length of the signal window, so
            # rescale it the same way `split_half_null` does
            n_s = (hA[1] - hA[0]) / sd
            n_n = (q1[1] - q1[0]) / sd
            scale = float(np.sqrt((2 / n_s) / (2 / n_n))) if n_n > 0 else 1.0
            rows[space] = (dS @ u, np.abs((dN * scale) @ u), exc,
                           np.linalg.norm(dS, axis=1))
        frames.append(pd.DataFrame({
            "id": cand["id"].to_numpy(), "kind": cand["kind"].to_numpy(),
            "district": cand["district"].to_numpy(),
            "role": cand["role"].to_numpy(), "element": cand["element"].to_numpy(),
            "drift_district": d,
            "proj": rows["shape"][0], "nu": rows["shape"][1],
            "excite": rows["shape"][2], "dz_norm": rows["shape"][3],
            "proj_native": rows["native"][0], "nu_native": rows["native"][1],
            "excite_native": rows["native"][2],
            "g": rows["shape"][0] / max(rows["shape"][2], EPS),
            "resid": np.nan, "null": rows["shape"][1],
            "d_level": np.nan, "d_level_district": np.nan,
            "settle_start_month": lo1 // P.steps_month,
        }))
        if verbose:
            print(f"  placebo {d}: settled halves "
                  f"{hA[0]//P.steps_month}..{hB[1]//P.steps_month} months",
                  flush=True)

    gains = pd.concat(frames, ignore_index=True)
    piv = lambda c: gains.pivot(index="id", columns="drift_district",
                                values=c).reindex(columns=districts)
    NULL = gains.groupby("id")["null"].max()
    return {"gains": gains, "cand": cand, "districts": districts,
            "PROJ": piv("proj"), "NU": piv("nu"),
            "EXC": gains.groupby("drift_district")["excite"].first().reindex(districts),
            "PROJ_native": piv("proj_native"), "NU_native": piv("nu_native"),
            "EXC_native": gains.groupby("drift_district")["excite_native"]
            .first().reindex(districts),
            "G": piv("g"), "DZ": piv("dz_norm"), "null": NULL,
            "degenerate": pd.Series(False, index=NULL.index),
            "placebo": True}


def placebo_report(real: dict, placebo: dict, kind: str = "flow",
                   space: str = "shape") -> dict:
    """Real against placebo, elementwise. The pass criterion, in numbers.

    `diag_ratio` is the real diagonal over the placebo diagonal -- the signal
    to pipeline-artefact ratio, and the number that licenses the whole method.
    `off_ratio` is the same for the off-diagonal: if the real off-diagonal is
    not clearly above its placebo, the coupling being reported is not real.
    `placebo_survivors` is the share of placebo cells that survive the soft
    threshold at all; it should be small.
    """
    Dr, _, _ = mp.dependence(real, kind=kind, space=space)
    Dp, _, _ = mp.dependence(placebo, kind=kind, space=space)
    Er = mp.elasticity(real, space=space)
    Ep = mp.elasticity(placebo, space=space)
    ck = (real["cand"].set_index("id")["kind"].reindex(Ep.index) == kind).to_numpy()

    mask = ~np.eye(len(Dr), dtype=bool)
    stats = pd.DataFrame([{
        "kind": kind, "space": space,
        "real_diag": float(np.nanmean(np.diag(Dr.to_numpy()))),
        "placebo_diag": float(np.nanmean(np.diag(Dp.to_numpy()))),
        "real_off": float(np.nanmean(Dr.to_numpy()[mask])),
        "placebo_off": float(np.nanmean(Dp.to_numpy()[mask])),
        "placebo_survivors": float((np.abs(Ep.to_numpy()[ck]) > 0).mean()),
        "real_survivors": float((np.abs(Er.to_numpy()[ck]) > 0).mean()),
    }])
    stats["diag_ratio"] = stats["real_diag"] / stats["placebo_diag"].replace(0, np.nan)
    stats["off_ratio"] = stats["real_off"] / stats["placebo_off"].replace(0, np.nan)
    stats["verdict"] = np.where(
        (stats["diag_ratio"] > 10) & (stats["off_ratio"] > 3), "pass",
        np.where(stats["off_ratio"] > 3, "diagonal weak", "FAIL"))
    return {"stats": stats.round(4), "D_real": Dr, "D_placebo": Dp,
            "E_real": Er, "E_placebo": Ep, "kind_mask": ck}


# ==========================================================================
# 2. SUPERPOSITION
# ==========================================================================
def superposed_cfg(base_cfg: lw.WorldCfg, a: str, b: str) -> lw.WorldCfg:
    """The world where districts `a` and `b` drift together."""
    from dataclasses import replace
    return replace(base_cfg, tgt_district=a, co_targets=(b,))


def load_or_build(cfg: lw.WorldCfg, root, out_root, verbose: bool = True):
    """Cache-first: read the world if its hash is on disk, otherwise solve it."""
    import pathlib
    d = pathlib.Path(out_root) / lw.sim_hash(cfg)
    if (d / "manifest.json").exists():
        from worlds import load_world
        if verbose:
            print(f"  cache {d.name}")
        return sp.from_lab_world(load_world(d))
    if verbose:
        print(f"  solving {d.name} ...", flush=True)
    world = lw.build(cfg, root, verbose=False)
    lw.persist(world, out_root)
    return sp.from_world(world)


def superposition(P_a: sp.Probe, P_b: sp.Probe, P_ab: sp.Probe,
                  space: str = "native", label: str = "") -> dict:
    """Is the response to two drifts the sum of the two single responses?

    Tested at VECTOR level, which is the strongest form and needs no
    projection: for every gauge, compare the weekly-profile change in the
    two-district world against the sum of the two single-district changes.

    `slope` is the regression of the observed joint response on the predicted
    sum -- 1.0 is exact superposition, below 1 means the second drift is partly
    absorbed (saturation), above 1 means it is amplified. `rel_err` is the
    residual as a share of the observed response. Both are reported pooled and
    per gauge, because an average near 1 can hide a minority of gauges that
    behave completely differently.

    Native space is the default: L/s and mca add, z-scores do not.

    RUN IT ACROSS BETA, not once. Superposition is a small-signal property: any
    network saturates if the drift is large enough, so a single "not additive"
    at one beta says nothing about whether the estimator is valid. The
    informative object is the trend -- if `pooled_slope` approaches 1 as beta
    falls, the non-additivity is a magnitude effect and the dependence matrix
    is sound in the regime it is quoted for. If the slope is flat and far from
    1 at every beta, the single-district design does not compose and that has
    to be stated.
    """
    cand = sp.candidates(P_ab)
    ids = set(cand["id"])
    for P in (P_a, P_b):
        ids &= set(sp.candidates(P)["id"])
    cand = cand[cand["id"].isin(ids)].reset_index(drop=True)

    dA, _ = delta_vectors(P_a, cand, space)
    dB, _ = delta_vectors(P_b, cand, space)
    dAB, _ = delta_vectors(P_ab, cand, space)
    pred = dA + dB

    num = (dAB * pred).sum(axis=1)
    den = (pred * pred).sum(axis=1)
    slope = np.divide(num, den, out=np.full(len(num), np.nan), where=den > EPS)
    res = np.linalg.norm(dAB - pred, axis=1)
    obs = np.linalg.norm(dAB, axis=1)
    rel = np.divide(res, obs, out=np.full(len(res), np.nan), where=obs > EPS)
    cosang = np.divide(num, np.sqrt(den) * obs,
                       out=np.full(len(num), np.nan),
                       where=(den > EPS) & (obs > EPS))

    per = cand[["id", "kind", "district", "role"]].copy()
    per["slope"] = slope
    per["rel_err"] = rel
    per["cos"] = cosang
    per["obs_norm"] = obs
    per["pred_norm"] = np.linalg.norm(pred, axis=1)
    per["a_norm"] = np.linalg.norm(dA, axis=1)
    per["b_norm"] = np.linalg.norm(dB, axis=1)

    rows = []
    for kind, g in per.groupby("kind"):
        # pooled slope: one regression over every gauge's whole profile, so
        # large responders carry the weight they should
        m = (cand["kind"] == kind).to_numpy()
        x, y = pred[m].ravel(), dAB[m].ravel()
        rows.append({
            "label": label, "kind": kind, "space": space, "n": int(m.sum()),
            "pooled_slope": float(x @ y / (x @ x)) if (x @ x) > EPS else np.nan,
            "pooled_r": float(np.corrcoef(x, y)[0, 1]),
            "median_slope": float(g["slope"].median()),
            "median_rel_err": float(g["rel_err"].median()),
            "p90_rel_err": float(g["rel_err"].quantile(0.9)),
            "share_within_10%": float((g["slope"].sub(1).abs() < 0.1).mean()),
        })
    summary = pd.DataFrame(rows).round(4)
    summary["verdict"] = np.where(
        (summary["pooled_slope"].between(0.9, 1.1))
        & (summary["median_rel_err"] < 0.25), "additive",
        np.where(summary["pooled_slope"].between(0.75, 1.25),
                 "weakly additive", "NOT additive"))
    return {"summary": summary, "per_gauge": per.round(4),
            "vectors": {"A": dA, "B": dB, "AB": dAB, "pred": pred},
            "cand": cand}


# ==========================================================================
# 3. excitation space
# ==========================================================================
def excitation_report(bases: dict, ref_beta: float, kind: str = "flow",
                      level_ratio: dict | None = None) -> pd.DataFrame:
    """Beta-invariance of the elasticity, in both spaces, against prediction.

    `bases` is {beta: base}. `level_ratio` is the predicted slope if the
    response were purely volume-driven -- `(intensity_ratio ** beta) /
    (intensity_ratio ** ref_beta)`. An observed slope that tracks it means the
    normalisation has not removed the excitation; a slope flat at 1.0 means it
    has, and the matrix can be quoted at any operating point.
    """
    rows = []
    ref = bases[ref_beta]
    for space in ("shape", "native"):
        for beta, b in sorted(bases.items()):
            if beta == ref_beta:
                continue
            r = mp.beta_linearity(ref, b, kind, f"b={ref_beta}", f"b={beta}",
                                  space=space).iloc[0].to_dict()
            r["beta"] = beta
            r["predicted_if_volume"] = (level_ratio or {}).get(beta, np.nan)
            rows.append(r)
    out = pd.DataFrame(rows)
    if not len(out):
        return pd.DataFrame(columns=["space", "kind", "beta", "n", "r", "slope",
                                     "predicted_if_volume", "slope_err_vs_1",
                                     "slope_err_vs_volume"])
    out["slope_err_vs_1"] = (out["slope"] - 1).abs()
    out["slope_err_vs_volume"] = (out["slope"] - out["predicted_if_volume"]).abs()
    return out[["space", "kind", "beta", "n", "r", "slope",
                "predicted_if_volume", "slope_err_vs_1",
                "slope_err_vs_volume"]].round(3)


def level_ratios(betas, ref_beta: float, intensity_ratio: float = 9.19) -> dict:
    """`(ratio**beta)/(ratio**ref)` -- the slope a purely volume-driven
    response would show. 9.19 is low-income residential -> industrial."""
    return {b: float(intensity_ratio ** b / intensity_ratio ** ref_beta)
            for b in betas}


# ==========================================================================
# figures
# ==========================================================================
def plot_excitation(rep: pd.DataFrame, kind: str = "flow", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.6))
    d = rep[rep["kind"] == kind]
    for space, m in (("shape", "o"), ("native", "s")):
        g = d[d["space"] == space].sort_values("beta")
        ax.plot(g["beta"], g["slope"], marker=m, label=f"{space} space")
    g = d[d["space"] == "shape"].sort_values("beta")
    ax.plot(g["beta"], g["predicted_if_volume"], ls="--", color="crimson",
            label="if purely volume-driven")
    ax.axhline(1.0, color="k", lw=.7)
    ax.set(xlabel="beta", ylabel="slope vs reference beta",
           title=f"{kind}: is E excitation-invariant?")
    ax.legend(fontsize=7)
    return ax


def plot_placebo(rep: dict, ax=None, title: str = ""):
    """Real and placebo elasticity distributions on the same axis."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.4))
    ck = rep["kind_mask"]
    r = np.abs(rep["E_real"].to_numpy()[ck]).ravel()
    p = np.abs(rep["E_placebo"].to_numpy()[ck]).ravel()
    r, p = r[np.isfinite(r)], p[np.isfinite(p)]
    lo = max(min(r[r > 0].min() if (r > 0).any() else 1e-6,
                 p[p > 0].min() if (p > 0).any() else 1e-6), 1e-9)
    bins = np.logspace(np.log10(lo), np.log10(max(r.max(), p.max(), lo * 10)), 45)
    ax.hist(r[r > 0], bins=bins, alpha=.6, label="real drift")
    ax.hist(p[p > 0], bins=bins, alpha=.6, label="placebo (no drift)")
    ax.set(xscale="log", yscale="log", xlabel="|E|", ylabel="gauge-district cells",
           title=title or "real vs placebo elasticity")
    ax.legend(fontsize=7)
    return ax


def plot_superposition(res: dict, kind: str = "flow", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))
    m = (res["cand"]["kind"] == kind).to_numpy()
    x = res["vectors"]["pred"][m].ravel()
    y = res["vectors"]["AB"][m].ravel()
    ax.scatter(x, y, s=3, alpha=.15)
    lim = np.nanpercentile(np.abs(np.concatenate([x, y])), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color="crimson", lw=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
           xlabel="predicted  dA + dB", ylabel="observed  dAB",
           title=f"superposition — {kind}")
    return ax


def verdict(exc: pd.DataFrame, plac: list, sup: pd.DataFrame) -> pd.DataFrame:
    """One table: does the estimator survive all three checks?"""
    rows = []
    for kind in sorted(exc["kind"].unique()):
        e = exc[exc["kind"] == kind]
        best = e.groupby("space")["slope_err_vs_1"].mean()
        rows.append({
            "kind": kind,
            "best_space": best.idxmin(),
            "shape_slope_err": round(float(best.get("shape", np.nan)), 3),
            "native_slope_err": round(float(best.get("native", np.nan)), 3),
            "placebo": "; ".join(
                f"{r['space']}:{r['verdict']}"
                for r in [x.iloc[0].to_dict() for x in plac]
                if r["kind"] == kind),
            "superposition": "; ".join(
                sup.loc[sup["kind"] == kind, "verdict"].astype(str)),
        })
    return pd.DataFrame(rows)
