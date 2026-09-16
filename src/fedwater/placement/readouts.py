"""readouts.py -- cross-world read-outs over a probe stack's battery scores.

Ported from the POC's ``landuse_sweep.py``; only the read-outs survive. The
grid itself (one world per drifting district) is now built by
:mod:`fedwater.placement.store`, and `beta` is a single operating point per
stack rather than an axis -- which is why every function below still groups by
`beta` (a stack has exactly one value) and by `network`.

`scores` is ``signal_probe.battery`` output for every probe world of one stack,
stacked, with ``network``, ``drift_district``, ``beta`` and ``sim_hash``
columns added.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import signal_probe as sp

__all__ = ["cell_row", "leak_matrix", "signal_carriers", "drift_carriers",
           "stability", "plot_leak", "plot_carriers"]


def cell_row(P: sp.Probe, sc: pd.DataFrame, dl: pd.DataFrame,
              phase: str) -> dict:
    """One line per cell: did the drift land, and did anything see it."""
    tgt = P.drift["tgt_district"]
    t = dl[dl["district"] == tgt].iloc[0]
    g = sp.discriminative_gap(sc)
    g = g[g["phase"] == phase].set_index("kind")

    own = sc[(sc["phase"] == phase) & (sc["district"] == tgt)]
    top = {}
    for kind in ("flow", "pressure"):
        k = own[own["kind"] == kind].nlargest(3, "r_res_own")
        top[kind] = float(k["response"].mean()) if len(k) else np.nan
    # The leak: the strongest response anywhere OUTSIDE the drifting district.
    # A closed network guarantees it is non-zero; how big it is relative to
    # the target's own response is the confounding this whole thesis is about.
    other = sc[(sc["phase"] == phase) & (sc["district"] != tgt)
               & (sc["kind"] == "flow")]
    return {
        "d_demand_%": float(t["d_demand_%"]), "shape_shift": float(t["shape_shift"]),
        "d_pressure": float(t["d_pressure"]), "d_boundary_%": float(t["d_boundary_%"]),
        "gap_flow": float(g.loc["flow", "gap_regime"]) if "flow" in g.index else np.nan,
        "gap_pressure": (float(g.loc["pressure", "gap_regime"])
                         if "pressure" in g.index else np.nan),
        "hit_flow": float(g.loc["flow", "regime_hit_rate"]) if "flow" in g.index else np.nan,
        "hit_pressure": (float(g.loc["pressure", "regime_hit_rate"])
                         if "pressure" in g.index else np.nan),
        "resp_flow_top": top["flow"], "resp_pressure_top": top["pressure"],
        "resp_leak_max": float(other["response"].abs().max()) if len(other) else np.nan,
        "resp_leak_p95": (float(other["response"].abs().quantile(0.95))
                          if len(other) else np.nan),
    }


def leak_matrix(S: pd.DataFrame, phase: str = "final", kind: str = "flow",
                top: int = 3, agg: str = "mean") -> dict:
    """Rows = which district drifted, cols = where the response was measured.

    Each entry is the mean |response| of that column-district's `top`
    best-resembling gauges. The diagonal is the signal; everything off it is
    hydraulic contamination arriving through shared pipes. The ratio of the
    two is the de-confounding problem stated in numbers, per network.
    """
    out = {}
    d = S[(S["phase"] == phase) & (S["kind"] == kind)]
    for network, gn in d.groupby("network"):
        rows = {}
        for drifted, gd in gn.groupby("drift_district"):
            rows[drifted] = {}
            for district, gg in gd.groupby("district"):
                # top-`top` WITHIN each beta, then pooled -- taking the
                # globally largest rows would silently weight whichever beta
                # happened to produce the cleanest resemblance.
                best = (gg.groupby("beta", group_keys=False)
                        .apply(lambda x: x.nlargest(top, "r_res_own"),
                               include_groups=False))
                v = best["response"].abs()
                rows[drifted][district] = float(getattr(v, agg)())
        M = pd.DataFrame(rows).T
        out[network] = M.reindex(index=sorted(M.index), columns=sorted(M.columns))
    return out


def signal_carriers(S: pd.DataFrame, phase: str = "init", top: int = 5) -> pd.DataFrame:
    """Gauges that RESEMBLE their district, averaged over the whole grid.

    Scored in the drift-free `init` phase and averaged across every cell, so
    this is a property of the network and the placement, not of any one drift.
    `std` is the stability: a gauge with a high mean and a low std is a
    placement you can commit to across scenarios.
    """
    g = (S[S["phase"] == phase]
         .groupby(["network", "district", "kind", "id", "role"])["r_res_own"]
         .agg(["mean", "std", "min", "count"]).reset_index()
         .rename(columns={"mean": "r_mean", "std": "r_std", "min": "r_min"}))
    g["rank"] = g.groupby(["network", "district", "kind"])["r_mean"] \
        .rank(ascending=False, method="first")
    return (g[g["rank"] <= top].sort_values(["network", "district", "kind", "rank"])
            .reset_index(drop=True).round(4))


def drift_carriers(S: pd.DataFrame, phase: str = "final", top: int = 5) -> pd.DataFrame:
    """Gauges that RESPOND when their own district is the one that drifts.

    The complement of `signal_carriers`, and the stricter test: restricted to
    the cells where this gauge's district is the drifting one, and ranked by
    response rather than resemblance. `beta_min` is the smallest strength at
    which that gauge already responded above 0.5.
    """
    d = S[(S["phase"] == phase) & (S["district"] == S["drift_district"])]
    g = (d.groupby(["network", "district", "kind", "id", "role"])
         .agg(resp_mean=("response", "mean"), resp_std=("response", "std"),
              r_mean=("r_res_own", "mean"), n=("response", "size")).reset_index())
    strong = d[d["response"] >= 0.5].groupby(["network", "district", "kind", "id"])["beta"] \
        .min().rename("beta_min").reset_index()
    g = g.merge(strong, on=["network", "district", "kind", "id"], how="left")
    g["rank"] = g.groupby(["network", "district", "kind"])["resp_mean"] \
        .rank(ascending=False, method="first")
    return (g[g["rank"] <= top].sort_values(["network", "district", "kind", "rank"])
            .reset_index(drop=True).round(4))


def stability(S: pd.DataFrame, phase: str = "init", top: int = 5) -> pd.DataFrame:
    """How often a gauge lands in its district's top-`top` across all cells.

    `share` near 1.0 means the ranking does not depend on which district
    drifted or how hard -- i.e. placement is a property of the network and can
    be decided once. A low share everywhere means placement has to be
    re-derived per scenario, which would be a finding in its own right.
    """
    d = S[S["phase"] == phase].copy()
    d["rk"] = d.groupby(["network", "sim_hash", "district", "kind"])["r_res_own"] \
        .rank(ascending=False, method="first")
    n = d.groupby(["network", "district", "kind"])["sim_hash"].nunique() \
        .rename("cells")
    hits = (d[d["rk"] <= top].groupby(["network", "district", "kind", "id"])
            .size().rename("in_top"))
    out = hits.reset_index().merge(n.reset_index(),
                                   on=["network", "district", "kind"])
    out["share"] = out["in_top"] / out["cells"]
    return out.sort_values(["network", "district", "kind", "share"],
                           ascending=[True, True, True, False]).reset_index(drop=True)


def plot_leak(M: dict, axes=None, vmax: float | None = None):
    """The leak matrices. Diagonal = signal, off-diagonal = contamination."""
    import matplotlib.pyplot as plt
    nets = sorted(M)
    if axes is None:
        _, axes = plt.subplots(1, len(nets), figsize=(4.6 * len(nets), 4.0),
                               squeeze=False)
        axes = axes[0]
    for ax, net in zip(np.atleast_1d(axes), nets):
        A = M[net]
        im = ax.imshow(A.to_numpy(), cmap="magma", vmin=0,
                       vmax=vmax or float(np.nanmax(A.to_numpy())))
        ax.set(xticks=range(A.shape[1]), yticks=range(A.shape[0]),
               xlabel="measured in", ylabel="drifted",
               title=f"{net}: |response| leak")
        ax.set_xticklabels([c.replace("District_", "") for c in A.columns])
        ax.set_yticklabels([c.replace("District_", "") for c in A.index])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A.to_numpy()[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v < 0.6 * im.get_clim()[1] else "k")
    return axes


def plot_carriers(car: pd.DataFrame, network: str, kind: str = "flow", ax=None):
    """Mean resemblance with its spread, per district's top gauges."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 3.6))
    d = car[(car["network"] == network) & (car["kind"] == kind)]
    x = np.arange(len(d))
    ax.errorbar(x, d["r_mean"], yerr=d["r_std"].fillna(0), fmt="o", ms=4,
                capsize=2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.district.replace('District_','')}:{r.id}"
                        for r in d.itertuples()], rotation=75, fontsize=6)
    ax.axhline(0, color="k", lw=.6)
    ax.set(ylabel="r_res_own (mean +/- sd over cells)",
           title=f"{network} -- {kind} signal carriers")
    return ax

