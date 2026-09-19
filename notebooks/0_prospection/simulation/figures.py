"""Pure figures: `(tables) -> Axes`. No recomputation, no widgets.

Every function here takes tables that already exist -- either the persisted
artifacts or the objects `run_cell` returned -- and draws them. Nothing
rebuilds windows or re-encodes latents, so a figure can never disagree with
the run it claims to show. The one place that legitimately has to recompute
is the preprocessing browser, and it lives in `widgets` with an explicit
cache.

`project` is carried over verbatim in behaviour from the AER sandbox,
including the `fit_on` toggle, because the subtlety it encodes is easy to
lose: prototypes are *means* of latents, so a map fitted on the window cloud
collapses them near its centroid and their mutual structure -- the thing the
clustering question is about -- becomes unreadable. Anchoring the same kind
of linear map on the prototypes instead sends the windows through it and
opens the prototype spread up. Both are one map applied to both sets; only
the variance the axes are chosen to explain differs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = ["fcols", "project", "plot_latent", "plot_similarity",
           "plot_metrics_by_round", "plot_client_series", "plot_window",
           "plot_scaling_compare", "plot_drift_progress",
           "plot_prototype_trajectory", "plot_update_gram", "PALETTE"]

# Colour-blind-safe qualitative palette, stable across figures so a client
# keeps its colour from the series plot to the latent plot.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C"]


def fcols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def colour_map(keys) -> dict:
    ks = sorted(keys)
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(ks)}


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def project(Z_pts: np.ndarray, Z_pro: np.ndarray, method: str = "PCA",
            seed: int = 0, fit_on: str = "windows"):
    """One space for both sets. -> (E_points, E_prototypes, subtitle).

    `fit_on="windows"` reads the prototypes into the window cloud;
    `fit_on="prototypes"` anchors on the prototypes and sends the windows
    through. t-SNE and UMAP have no out-of-sample transform, so they are
    fitted on the concatenation and split afterwards.
    """
    Z_pts = np.asarray(Z_pts, float)
    Z_pro = (np.asarray(Z_pro, float) if Z_pro is not None and len(Z_pro)
             else np.empty((0, Z_pts.shape[1])))

    anchor = Z_pro if (fit_on == "prototypes" and len(Z_pro) >= 3) else Z_pts
    note = "" if anchor is Z_pts else " · proto-fitted"
    scaler = StandardScaler().fit(anchor)
    P = scaler.transform(Z_pts)
    Q = scaler.transform(Z_pro) if len(Z_pro) else np.empty((0, P.shape[1]))

    if method == "PCA":
        A = scaler.transform(anchor)
        p = PCA(n_components=min(10, A.shape[1]), random_state=seed).fit(A)
        sub = "  ".join(f"PC{i + 1} {v:.1%}"
                        for i, v in enumerate(p.explained_variance_ratio_[:4]))
        return p.transform(P), (p.transform(Q) if len(Q) else Q), sub + note

    stack = np.vstack([P, Q])
    if method == "t-SNE":
        from sklearn.manifold import TSNE
        E = TSNE(n_components=2, init="pca", random_state=seed,
                 perplexity=min(30, max(5, len(stack) // 4))
                 ).fit_transform(stack)
        sub = "t-SNE (joint fit)" + note
    elif method == "UMAP":
        import umap
        E = umap.UMAP(n_components=2, random_state=seed).fit_transform(stack)
        sub = "UMAP (joint fit)" + note
    else:
        raise KeyError(f"method {method!r} must be PCA|t-SNE|UMAP")
    return E[:len(P)], E[len(P):], sub


# --------------------------------------------------------------------------
# latent space
# --------------------------------------------------------------------------
_MARKER = {"local": "o", "post_fedavg": "s", "post_fedavg_full": "P",
           "global": "D"}


def plot_latent(ax, points: pd.DataFrame, E_pts: np.ndarray,
                protos: pd.DataFrame | None = None,
                E_pro: np.ndarray | None = None, colour_by: str = "district",
                subtitle: str = "", trajectory: bool = False,
                point_size: float = 6, alpha: float = 0.35,
                cmap: dict | None = None):
    """The window cloud with prototypes overlaid in the same space.

    Marker key: `o` local (pre-FedAvg), `s` post-FedAvg, `P` post-FedAvg over
    all windows, `D` global FINCH cluster. `trajectory=True` joins each
    client's prototypes in month order -- its drift path.
    """
    # `dtype == object` no longer identifies strings in pandas 3 (they get a
    # dedicated `str` dtype), so branch on numeric-ness instead.
    categorical = (colour_by in points.columns
                   and not pd.api.types.is_numeric_dtype(points[colour_by]))
    if categorical:
        cmap = cmap or colour_map(points[colour_by].unique())
        for k, grp in points.groupby(colour_by):
            i = points.index.get_indexer(grp.index)
            ax.scatter(E_pts[i, 0], E_pts[i, 1], s=point_size, alpha=alpha,
                       lw=0, color=cmap[k], label=str(k))
    else:
        v = points[colour_by].to_numpy() if colour_by in points else None
        sc = ax.scatter(E_pts[:, 0], E_pts[:, 1], s=point_size, alpha=alpha,
                        lw=0, c=v, cmap="viridis")
        if v is not None:
            ax.figure.colorbar(sc, ax=ax, fraction=.035, pad=.02,
                               label=colour_by)

    if protos is not None and E_pro is not None and len(protos):
        pc = cmap or colour_map(protos["client"].dropna().unique())
        for scope, grp in protos.groupby("scope"):
            i = protos.index.get_indexer(grp.index)
            if scope == "global":
                face = "none"
            else:
                keys = (grp["client"] if "client" in grp.columns
                        else pd.Series(["?"] * len(grp), index=grp.index))
                face = [pc.get(c, "#333333") for c in keys]
            ax.scatter(E_pro[i, 0], E_pro[i, 1], s=70,
                       marker=_MARKER.get(scope, "*"), facecolor=face,
                       edgecolor="black", lw=.8, zorder=3)
        if trajectory and "client" in protos.columns:
            for c, grp in protos[protos["scope"] == "local"].groupby("client"):
                g = grp.sort_values("month")
                i = protos.index.get_indexer(g.index)
                ax.plot(E_pro[i, 0], E_pro[i, 1], lw=.9, alpha=.7,
                        color=pc.get(c, "#333333"), zorder=2)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_alpha(.25)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, loc="left", color="#444444")
    return ax


def plot_prototype_trajectory(ax, protos: pd.DataFrame, world,
                              scope: str = "local", round_idx: int | None = None,
                              component: int = 0):
    """One latent component of each client's monthly prototype, over months.

    Phase bands are shaded from the world's own schedule, so the transition
    stretch is visible rather than implied.
    """
    d = protos[protos["scope"] == scope]
    r = int(d["round"].max()) if round_idx is None else int(round_idx)
    d = d[d["round"] == r]
    f = fcols(d)[component]
    cmap = colour_map(d["client"].dropna().unique())
    for c, grp in d.groupby("client"):
        g = grp.sort_values("month")
        ax.plot(g["month"], g[f], marker="o", ms=3, lw=1.2,
                color=cmap[c], label=str(c))
    ph = world.phases()
    ax.axvspan(*ph["init"], color="#55A868", alpha=.10, lw=0)
    ax.axvspan(*ph["final"], color="#4C72B0", alpha=.10, lw=0)
    ax.set_xlabel("month")
    ax.set_ylabel(f"{f} ({scope}, round {r})")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return ax


# --------------------------------------------------------------------------
# similarity / clustering
# --------------------------------------------------------------------------
def plot_similarity(ax, S: pd.DataFrame, labels: dict | None = None,
                    title: str = "", vlim: tuple | None = None,
                    annotate: bool = True):
    """Similarity heatmap. Tick labels carry the regime token when given."""
    A = np.asarray(S, float)
    off = A[~np.eye(len(A), dtype=bool)]
    if vlim is not None:
        lo, hi = vlim
    else:
        # Scale to the OFF-DIAGONAL range. Raw prototype cosines routinely sit
        # near 0.97 for every pair (the common mode), and a fixed symmetric
        # scale renders that as a uniform block -- the structure worth seeing
        # is the spread, not the level.
        lo, hi = float(np.nanmin(off)), float(np.nanmax(off))
        if lo < 0 < hi:                        # straddles zero -> keep it centred
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        else:
            pad = max(1e-6, 0.05 * (hi - lo))
            lo, hi = lo - pad, hi + pad
    im = ax.imshow(A, cmap="RdBu_r", vmin=lo, vmax=hi)
    mid = (lo + hi) / 2
    names = [f"{c.split('_')[-1]}" + (f" ({labels[c]})" if labels and c in labels
                                      else "") for c in S.index]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    if annotate:
        for i in range(len(A)):
            for j in range(len(A)):
                far = abs(A[i, j] - mid) > .4 * (hi - lo)
                ax.text(j, i, f"{A[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7, color="white" if far else "#222222")
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.03)
    return ax


def plot_update_gram(ax, update_gram: pd.DataFrame, round_idx: int,
                     labels: dict | None = None):
    """The FedAvg update-delta cosine matrix for one round."""
    g = update_gram[update_gram["round"] == int(round_idx)]
    clients = sorted(set(g["client_a"]) | set(g["client_b"]))
    S = pd.DataFrame(np.eye(len(clients)), index=clients, columns=clients)
    for r in g.itertuples():
        S.loc[r.client_a, r.client_b] = r.cos
        S.loc[r.client_b, r.client_a] = r.cos
    return plot_similarity(ax, S, labels,
                           title=f"update deltas · round {round_idx}")


def plot_metrics_by_round(ax, by_round: pd.DataFrame,
                          cols=("sil_district", "sil_month", "eff_dim"),
                          normalise: bool = False):
    """Latent metrics against the round axis; round -1 is the control."""
    d = by_round.sort_values("round")
    for i, c in enumerate(cols):
        if c not in d.columns:
            continue
        y = d[c].to_numpy(float)
        if normalise and np.nanmax(np.abs(y)) > 0:
            y = y / np.nanmax(np.abs(y))
        ax.plot(d["round"], y, marker="o", ms=3, lw=1.3,
                color=PALETTE[i % len(PALETTE)], label=c)
    ax.axvline(-0.5, color="#888888", lw=.8, ls=":")
    ax.text(-0.45, ax.get_ylim()[1], " untrained control", fontsize=7,
            va="top", color="#666666")
    ax.set_xlabel("round")
    ax.legend(fontsize=7, frameon=False)
    return ax


# --------------------------------------------------------------------------
# preprocessing / EDA
# --------------------------------------------------------------------------
def plot_client_series(ax, frames: dict, channel: int = 0, clients=None,
                       span: tuple | None = None, month: pd.Series | None = None):
    """Transformed client series, one line per client, with an optional shade."""
    clients = list(clients or sorted(frames))
    cmap = colour_map(clients)
    for c in clients:
        f = frames[c]
        cols = [x for x in f.columns if x not in ("timestamp", "month")]
        j = min(channel, len(cols) - 1)
        ax.plot(f[cols[j]].to_numpy(), lw=.4, alpha=.8, color=cmap[c],
                label=f"{c} · {cols[j]}")
    if span:
        ax.axvspan(span[0], span[1], color="#C44E52", alpha=.18, lw=0)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_xlabel("step")
    del month
    return ax


def plot_window(ax, fl_windows: dict, index: int, channel: int = 0,
                clients=None):
    """The window the model actually sees, in whatever units scaling left."""
    clients = list(clients or sorted(fl_windows))
    cmap = colour_map(clients)
    for c in clients:
        d = fl_windows[c]
        w = d["windows"]
        i = min(index, len(w) - 1)
        j = min(channel, w.shape[2] - 1)
        ax.plot(w[i, :, j], lw=1.1, color=cmap[c],
                label=f"{c} · {d['sensors'][j]}")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_xlabel("step within window")
    return ax


def plot_scaling_compare(ax, variants: dict, index: int = 0, channel: int = 0,
                         client: str | None = None):
    """The SAME window under several scaling modes -- the `shared` sanity check.

    `variants` is {mode: fl_windows}. Per-client `minmax_ref` pins every
    client to the feature range and so deletes cross-client level; a pooled
    `shared` scaler does not. Seeing them on one axis is the fastest way to
    confirm an override did something.
    """
    for i, (mode, fw) in enumerate(sorted(variants.items())):
        c = client or sorted(fw)[0]
        w = fw[c]["windows"]
        k = min(index, len(w) - 1)
        j = min(channel, w.shape[2] - 1)
        ax.plot(w[k, :, j], lw=1.2, color=PALETTE[i % len(PALETTE)],
                label=f"{mode}")
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel("step within window")
    ax.set_title("same window, different scaling", fontsize=9, loc="left")
    return ax


def plot_drift_progress(ax, world):
    """Demand-weighted switched fraction, with the phase bands shaded.

    Makes the cost of the schedule visible: the label-clean stretches are
    short and the transition band in between is most of the horizon.
    """
    p = world.drift_progress()
    ax.plot(p["month"], p["progress"], marker="o", ms=3, lw=1.4,
            color=PALETTE[3])
    ph = world.phases()
    ax.axvspan(*ph["init"], color="#55A868", alpha=.15, lw=0, label="init")
    ax.axvspan(*ph["final"], color="#4C72B0", alpha=.15, lw=0, label="final")
    ax.set_xlabel("month")
    ax.set_ylabel(f"switched share of {world.drift_district}")
    ax.set_ylim(-.02, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title(f"{world.tag} · warm-up {world.warmup_months} · "
                 f"ramp {world.ramp_days:.0f}d", fontsize=9, loc="left")
    return ax
