"""Multi-dimensional views of monthly prototype trajectories.

Meant to sit beside `plot_prototype_trajectory`; relies on the same module's
`fcols` and `colour_map`, and on `world.phases()` -> {"init": (lo, hi), "final": (lo, hi)}.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

_INIT_C, _FINAL_C = "#55A868", "#4C72B0"


__all__ = ["fcols", "project", "plot_latent", "plot_similarity",
           "plot_metrics_by_round", "plot_client_series", "plot_window",
           "plot_scaling_compare", "plot_drift_progress",
           "plot_prototype_trajectory", "plot_update_gram", "PALETTE",
           "SOURCE_STYLE", "plot_reconstruction", "plot_finch_month",
           "plot_retrieval_summary"]

# Colour-blind-safe qualitative palette, stable across figures so a client
# keeps its colour from the series plot to the latent plot.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C"]


def fcols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def colour_map(keys) -> dict:
    ks = sorted(keys)
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(ks)}


# ---------------------------------------------------------------- helpers
def _round_slice(protos, scope, round_idx):
    d = protos[protos["scope"] == scope]
    r = int(d["round"].max()) if round_idx is None else int(round_idx)
    return d[d["round"] == r], r


def _in_phase(month, bounds):
    """Months inside a phase band; bounds read as inclusive [lo, hi]."""
    lo, hi = bounds
    return (month >= lo) & (month <= hi)


def _phase_centroids(d, F, world, by_client):
    """Mean prototype per phase.

    by_client=True  -> DataFrame (client x F) per phase.
    by_client=False -> Series over F: mean of the per-client centroids, so every
                       client weighs the same regardless of row count.
    """
    ph = world.phases()
    out = {}
    for name in ("init", "final"):
        per_client = d[_in_phase(d["month"], ph[name])].groupby("client")[F].mean()
        out[name] = per_client if by_client else per_client.mean()
    return out


def _shade_phases(ax, world):
    ph = world.phases()
    ax.axvspan(*ph["init"], color=_INIT_C, alpha=.10, lw=0)
    ax.axvspan(*ph["final"], color=_FINAL_C, alpha=.10, lw=0)


def _client_centroid(mu, c, phase):
    if c not in mu[phase].index:
        warnings.warn(f"client {c!r} has no months in the '{phase}' phase")
        return None
    return mu[phase].loc[c].to_numpy(float)


# ------------------------------------------- 1. transition-axis projection
def transition_coordinates(protos, world, scope="local", round_idx=None,
                           axis="shared", eps=1e-12):
    """Position along the init->final axis and distance off it.

    With v = mu_final - mu_init and x = p_c(m) - mu_init:
        t     = <x, v> / ||v||^2          (0 = init regime, 1 = final regime)
        resid = ||x - t v||               (movement not explained by the shift)

    axis="shared": mu from the mean of per-client centroids (one axis for all).
    axis="client": mu from each client's own months (t is NaN when ||v||^2 < eps).
    """
    if axis not in ("shared", "client"):
        raise ValueError("axis must be 'shared' or 'client'")
    d, r = _round_slice(protos, scope, round_idx)
    F = fcols(d)
    mu = _phase_centroids(d, F, world, by_client=(axis == "client"))

    rows = []
    for c, g in d.groupby("client"):
        g = g.sort_values("month")
        if axis == "client":
            mi = _client_centroid(mu, c, "init")
            mf = _client_centroid(mu, c, "final")
        else:
            mi, mf = mu["init"].to_numpy(float), mu["final"].to_numpy(float)

        t = resid = np.full(len(g), np.nan)
        if mi is not None and mf is not None:
            v = mf - mi
            vv = float(v @ v)
            if vv < eps:
                warnings.warn(f"client {c!r}: init and final centroids coincide; "
                              f"t undefined on axis={axis!r}")
            else:
                X = g[F].to_numpy(float) - mi
                t = X @ v / vv
                resid = np.linalg.norm(X - np.outer(t, v), axis=1)
        rows.append(pd.DataFrame({"client": c, "month": g["month"].to_numpy(),
                                  "t": t, "resid": resid}))
    out = pd.concat(rows, ignore_index=True)
    out.attrs.update(round=r, scope=scope, axis=axis)
    return out


def plot_transition_axis(ax_t, ax_r, protos, world, scope="local",
                         round_idx=None, axis="shared"):
    """Two panels: t over months (top) and the orthogonal residual (bottom)."""
    tc = transition_coordinates(protos, world, scope, round_idx, axis)
    r = tc.attrs["round"]
    cmap = colour_map(tc["client"].dropna().unique())
    for c, g in tc.groupby("client"):
        kw = dict(marker="o", ms=3, lw=1.2, color=cmap[c], label=str(c))
        ax_t.plot(g["month"], g["t"], **kw)
        ax_r.plot(g["month"], g["resid"], **kw)
    for ax in (ax_t, ax_r):
        _shade_phases(ax, world)
    for y in (0, 1):
        ax_t.axhline(y, color="0.5", lw=.6, ls=":")
    ax_t.set_ylabel(f"t  (axis={axis}, {scope}, round {r})")
    ax_r.set_ylabel("off-axis residual")
    ax_r.set_xlabel("month")
    ax_t.legend(fontsize=7, frameon=False, ncol=2)
    return ax_t, ax_r


# ---------------------------------------- 2. distance to own init centroid
def reference_distance(protos, world, scope="local", round_idx=None):
    """d_c(m) = 1 - cos(p_c(m), mu_init,c), mu_init,c = client's own init centroid."""
    d, r = _round_slice(protos, scope, round_idx)
    F = fcols(d)
    mu = _phase_centroids(d, F, world, by_client=True)

    rows = []
    for c, g in d.groupby("client"):
        g = g.sort_values("month")
        m = _client_centroid(mu, c, "init")
        dist = np.full(len(g), np.nan)
        if m is not None:
            P = g[F].to_numpy(float)
            den = np.linalg.norm(P, axis=1) * np.linalg.norm(m)
            with np.errstate(invalid="ignore", divide="ignore"):
                dist = 1.0 - (P @ m) / den
        rows.append(pd.DataFrame({"client": c, "month": g["month"].to_numpy(),
                                  "d": dist}))
    out = pd.concat(rows, ignore_index=True)
    out.attrs.update(round=r, scope=scope)
    return out


def plot_reference_distance(ax, protos, world, scope="local", round_idx=None):
    rd = reference_distance(protos, world, scope, round_idx)
    cmap = colour_map(rd["client"].dropna().unique())
    for c, g in rd.groupby("client"):
        ax.plot(g["month"], g["d"], marker="o", ms=3, lw=1.2,
                color=cmap[c], label=str(c))
    _shade_phases(ax, world)
    ax.set_xlabel("month")
    ax.set_ylabel(f"1 - cos(p, μ_init)  ({scope}, round {rd.attrs['round']})")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return ax


# ------------------------------------------------ 4. per-client heatmaps
def prototype_deviation(protos, world, scope="local", round_idx=None):
    """z_c,k(m) = (p_c,k(m) - mu_init,c,k) / sigma_k.

    sigma_k is the pooled std of dimension k over all rows of the round, so one
    colour scale reads the same across dimensions and clients. Constant
    dimensions (sigma_k = 0) are dropped.
    """
    d, r = _round_slice(protos, scope, round_idx)
    F = fcols(d)
    sigma = d[F].std(ddof=0)
    F = [f for f in F if sigma[f] > 0]
    mu = _phase_centroids(d, F, world, by_client=True)["init"]

    parts = []
    for c, g in d.groupby("client"):
        if c not in mu.index:
            warnings.warn(f"client {c!r} has no months in the 'init' phase")
            continue
        z = (g[F] - mu.loc[c]) / sigma[F]
        z.insert(0, "month", g["month"].to_numpy())
        z.insert(0, "client", c)
        parts.append(z)
    out = pd.concat(parts, ignore_index=True)
    out.attrs.update(round=r, scope=scope, features=F)
    return out


def plot_prototype_heatmaps(axes, protos, world, scope="local", round_idx=None,
                            vlim_q=0.99, cmap="RdBu_r", feature_labels=True):
    """One (dimension x month) heatmap per client, shared row order and colour scale.

    Rows are ordered by average-linkage clustering on correlation distance
    (1 - corr) of z, pooled over clients and months, so dimensions that move
    together sit together. Returns (image, row_order) for a shared colorbar.
    """
    z = prototype_deviation(protos, world, scope, round_idx)
    F, r = z.attrs["features"], z.attrs["round"]
    order = [F[i] for i in leaves_list(
        linkage(z[F].to_numpy(float).T, method="average", metric="correlation"))]

    months = np.arange(int(z["month"].min()), int(z["month"].max()) + 1)
    vmax = float(np.nanquantile(np.abs(z[F].to_numpy(float)), vlim_q))
    clients = sorted(z["client"].unique())
    axes = np.ravel(axes)
    if len(axes) < len(clients):
        raise ValueError(f"need {len(clients)} axes, got {len(axes)}")

    ph = world.phases()
    im = None
    for ax, c in zip(axes, clients):
        M = (z[z["client"] == c].set_index("month")[order]
             .reindex(months).to_numpy(float).T)
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap,
                       vmin=-vmax, vmax=vmax,
                       extent=[months[0] - .5, months[-1] + .5, len(order) - .5, -.5])
        for name, col in (("init", _INIT_C), ("final", _FINAL_C)):
            lo, hi = ph[name]
            for x in (lo - .5, hi + .5):
                ax.axvline(x, color=col, lw=1.0, ls="--")
        ax.set_title(str(c), fontsize=8)
        ax.set_xlabel("month")
        if feature_labels:
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=4)
        else:
            ax.set_yticks([])
    for ax in axes[len(clients):]:
        ax.set_visible(False)
    axes[0].set_ylabel(f"dimension ({scope}, round {r})")
    return im, order
