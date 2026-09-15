"""placement_figures.py -- static figures. Nothing is computed here.

Every panel is a view of a table `district_sweep` already persisted, or a thin
call into the plotters that ship with `signal_probe` / `mixture_probe`. The
one thing this module adds is the NETWORK view, because the existing
`plot_tier_map` needs a live `wntr` model and the browsers have to work from
persisted tables alone. Coordinates and link endpoints come from
`bundles.topology`, straight out of the .inp.

District identity on a network plot
-----------------------------------
Drawn as faint district-coloured node halos plus one labelled centroid per
district, NOT as a convex hull or an alpha shape. A hull is the prettier
option right up until a partition has one district covering most of the
network -- D-Town's walktrap cut puts 265 of 399 junctions in one district --
and then the hull covers the canvas and hides everything underneath. Halos
degrade gracefully: the district is as visible as it is compact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mixture_probe as mp
import signal_probe as sp

__all__ = ["TIER_COLOUR", "district_colours", "draw_base", "district_backdrop",
           "tier_network", "metric_network", "selection_network",
           "mixture_bars", "matrix", "weekly_shapes", "selection_vs_demand",
           "settle_panel", "cluster_heatmap", "coverage_bars", "channel_bars"]

TIER_COLOUR = {"core": "#2e8b57", "transition": "#e08214",
               "foreign": "#c0392b", "unusable": "#c8ccd0"}


def district_colours(names) -> dict:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    return {d: cmap(i % 10) for i, d in enumerate(sorted(names))}


def _short(d) -> str:
    return str(d).replace("District_", "")


# ==========================================================================
# network canvas
# ==========================================================================
def draw_base(ax, topo: dict, colour="0.90", width=0.6):
    """Every pipe, in grey. The substrate every network panel sits on."""
    pos = topo["coords"]
    segs = [(pos[u], pos[v]) for u, v, _ in topo["links"].values()
            if u in pos and v in pos]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=colour, linewidths=width,
                                     zorder=0))
    xy = np.array(list(pos.values()))
    pad = 0.03 * (xy.max(axis=0) - xy.min(axis=0))
    ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
    ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
    ax.set(xticks=[], yticks=[], aspect="equal")
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def district_backdrop(ax, topo: dict, districts: dict, alpha=0.20, size=9,
                      label=True, colours=None):
    """Faint district-coloured nodes plus one haloed centroid label each."""
    import matplotlib.patheffects as pe
    pos = topo["coords"]
    colours = colours or district_colours(districts)
    for d, nodes in districts.items():
        xy = np.array([pos[n] for n in nodes if n in pos])
        if not len(xy):
            continue
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=[colours[d]], alpha=alpha,
                   linewidths=0, zorder=1)
        if label:
            # the medoid, not the mean: a district with an outlying arm would
            # otherwise get its label in empty space
            c = xy.mean(axis=0)
            m = xy[np.argmin(((xy - c) ** 2).sum(axis=1))]
            t = ax.text(m[0], m[1], _short(d), fontsize=11, fontweight="bold",
                        ha="center", va="center", color=colours[d], zorder=6)
            t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    return ax


def _element_xy(topo, element, kind):
    pos = topo["coords"]
    if kind == "pressure":
        return pos.get(element)
    u, v, _ = topo["links"].get(element, (None, None, None))
    if u in pos and v in pos:
        return tuple(np.mean([pos[u], pos[v]], axis=0))
    return None


def _element_seg(topo, element):
    pos = topo["coords"]
    u, v, _ = topo["links"].get(element, (None, None, None))
    if u in pos and v in pos:
        return (pos[u], pos[v])
    return None


# ==========================================================================
# tier map
# ==========================================================================
def tier_network(ax, topo: dict, table: pd.DataFrame, districts: dict,
                 kind: str = "flow", show_unusable: bool = True,
                 title: str = ""):
    """The configuration drawn with every element coloured by its tier.

    Answers "are a district's core gauges contiguous" by eye; the number is in
    `mixture_probe.core_connectivity`.
    """
    from matplotlib.collections import LineCollection
    draw_base(ax, topo)
    district_backdrop(ax, topo, districts)
    m = table[table["kind"] == kind]
    tiers = ("unusable", "foreign", "transition", "core") if show_unusable \
        else ("foreign", "transition", "core")
    for tier in tiers:                     # drawn last = drawn on top
        sel = m[m["tier"] == tier]
        if not len(sel):
            continue
        c = TIER_COLOUR[tier]
        if kind == "pressure":
            xy = [p for p in (_element_xy(topo, e, kind) for e in sel["element"])
                  if p is not None]
            if xy:
                xy = np.array(xy)
                ax.scatter(xy[:, 0], xy[:, 1], s=16, c=c, zorder=3,
                           linewidths=0, label=f"{tier} ({len(sel)})")
        else:
            segs = [s for s in (_element_seg(topo, e) for e in sel["element"])
                    if s is not None]
            if segs:
                ax.add_collection(LineCollection(
                    segs, colors=c, linewidths=1.8, zorder=3,
                    capstyle="round", label=f"{tier} ({len(sel)})"))
    ax.set_title(title or f"{kind} tiers", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    return ax


# ==========================================================================
# metric map -- the network replacement for the resemblance scatter
# ==========================================================================
def metric_network(ax, topo: dict, table: pd.DataFrame, districts: dict,
                   kind: str = "flow", value: str = "r_mean", top_n: int = 5,
                   group: str = "district", cmap: str = "viridis",
                   title: str = "", vmin=None, vmax=None):
    """The top-`top_n` gauges per district, with `value` on colour AND size.

    The scatter this replaces plotted resemblance against response on abstract
    axes. On the network the same numbers answer the question that actually
    follows -- WHERE the readable gauges are, and whether they sit together --
    which is the thing a placement decision needs and a scatter cannot show.
    """
    from matplotlib.collections import LineCollection
    draw_base(ax, topo)
    district_backdrop(ax, topo, districts, alpha=0.13, label=True)

    d = table[table["kind"] == kind].dropna(subset=[value])
    if group in d.columns and top_n:
        d = (d.sort_values(value, ascending=False)
             .groupby(group, group_keys=False).head(int(top_n)))
    if not len(d):
        ax.set_title(title or f"{kind}: no gauge with {value}", fontsize=10)
        return ax

    v = d[value].to_numpy(float)
    lo = float(np.nanmin(v)) if vmin is None else vmin
    hi = float(np.nanmax(v)) if vmax is None else vmax
    norm = np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    size = 22 + 150 * norm
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)

    if kind == "pressure":
        xy, s, col = [], [], []
        for (_, row), sz, nv in zip(d.iterrows(), size, norm):
            p = _element_xy(topo, row["element"], kind)
            if p is not None:
                xy.append(p); s.append(sz); col.append(cm(nv))
        if xy:
            xy = np.array(xy)
            sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=col, zorder=4,
                            edgecolors="white", linewidths=0.6)
    else:
        segs, lw, col = [], [], []
        for (_, row), sz, nv in zip(d.iterrows(), size, norm):
            seg = _element_seg(topo, row["element"])
            if seg is not None:
                segs.append(seg); lw.append(1.0 + 4.5 * nv); col.append(cm(nv))
        if segs:
            ax.add_collection(LineCollection(segs, colors=col, linewidths=lw,
                                             zorder=4, capstyle="round"))
        sc = ax.scatter([], [], c=[], cmap=cmap, vmin=lo, vmax=hi)
    sc.set_clim(lo, hi)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(value, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title or f"{kind}: top {top_n}/district by {value}", fontsize=10)
    return ax


def selection_network(ax, topo: dict, sel_table: pd.DataFrame, districts: dict,
                      arm: str, kind: str = "flow", title: str = ""):
    """Where one arm's chosen gauges actually sit. The localisation check."""
    from matplotlib.collections import LineCollection
    draw_base(ax, topo)
    colours = district_colours(districts)
    district_backdrop(ax, topo, districts, alpha=0.13, colours=colours)
    d = sel_table[(sel_table["arm"] == arm) & (sel_table["kind"] == kind)]
    for district, g in d.groupby("district"):
        c = colours.get(district, "k")
        if kind == "pressure":
            xy = [p for p in (_element_xy(topo, e, kind) for e in g["element"])
                  if p is not None]
            if xy:
                xy = np.array(xy)
                ax.scatter(xy[:, 0], xy[:, 1], s=70, c=[c], zorder=5,
                           marker="o", edgecolors="k", linewidths=0.8)
        else:
            segs = [s for s in (_element_seg(topo, e) for e in g["element"])
                    if s is not None]
            if segs:
                ax.add_collection(LineCollection(segs, colors=[c] * len(segs),
                                                 linewidths=3.4, zorder=5,
                                                 capstyle="round"))
    ax.set_title(title or f"{arm} -- {kind} ({len(d)} gauges)", fontsize=10)
    return ax


# ==========================================================================
# thin adapters over the shipped plotters
# ==========================================================================
def _as_res(result) -> dict:
    """A `ConfigResult` in the shape `mixture_probe`'s plotters expect."""
    return {"districts": result.names, "mixture": result.mixture}


def mixture_bars(ax, result, district: str, kind: str = "flow", top: int = 20):
    return mp.plot_mixture_bars(_as_res(result), district, kind=kind, top=top,
                                ax=ax)


def matrix(ax, A: pd.DataFrame, title="", vmax=1.0, xlabel="", ylabel=""):
    return mp.plot_matrix(A, ax=ax, title=title, vmax=vmax, xlabel=xlabel,
                          ylabel=ylabel)


def settle_panel(axes, probes: dict, ref_months: int = 3, slack: float = 0.01):
    for ax, d in zip(np.atleast_1d(axes), sorted(probes)):
        mp.plot_settle(probes[d], ax=ax, ref_months=ref_months, slack=slack)
        ax.set_title(_short(d), fontsize=10)
    return axes


def weekly_shapes(axes, P, district: str, scores: pd.DataFrame,
                  kind: str = "flow", n: int = 2):
    return sp.plot_weekly(P, district, scores, kind=kind, n=n, axes=axes)


def cluster_heatmap(ax, P, cand, phase: str = "init", kind: str = "flow"):
    return sp.plot_cluster(sp.cluster(P, cand, phase=phase, kind=kind), ax=ax)


def selection_vs_demand(ax, P, ids, district: str, phase: str = "init",
                        cand: pd.DataFrame | None = None):
    """Selected gauges' weekly shapes against their district's demand.

    The sanity check the read-out asks for: a gauge chosen for its mixture
    should still be recognisably reading water. A flat trace here against a
    strongly shaped demand week means the arm picked something the district's
    behaviour does not reach.
    """
    cand = sp.candidates(P) if cand is None else cand
    pr = sp.profiles(P, cand, phase)
    j = pr["districts"].index(district)
    x = np.arange(pr["demand"].shape[1]) / P.steps_day
    ax.plot(x, sp.zscore(pr["demand"][j]), lw=2.6, color="k", label="demand")
    ax.plot(x, sp.zscore(pr["total"]), lw=1.0, color="grey", ls="--",
            label="network total")
    order = list(cand["id"])
    for gid in ids:
        if gid not in order:
            continue
        ax.plot(x, sp.zscore(pr["sensor"][order.index(gid)]), lw=1.2, alpha=.85,
                label=gid)
    for day in (5, 6):
        ax.axvspan(day, day + 1, color="grey", alpha=.10, zorder=0)
    ax.set(xlabel="day of week", ylabel="z-scored weekly profile",
           title=f"{_short(district)} -- {phase}")
    ax.legend(fontsize=6, ncol=2)
    return ax


def coverage_bars(ax, cov: pd.DataFrame, arm: str, n_max: int = 5):
    """How full each (district, kind) set came out. Short sets are the point."""
    d = cov[cov["arm"] == arm]
    labels = [f"{_short(r.district)}/{r.kind[0]}" for r in d.itertuples()]
    x = np.arange(len(d))
    colours = ["#2e8b57" if r.complete else ("#c0392b" if r.empty else "#e08214")
               for r in d.itertuples()]
    ax.bar(x, d["n_selected"], color=colours)
    ax.axhline(n_max, color="k", lw=.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, fontsize=6)
    ax.set(ylabel="gauges selected", ylim=(0, n_max + 0.6),
           title=f"{arm}: set completeness")
    return ax


def channel_bars(ax, channels: pd.DataFrame, min_tv: float = 0.25):
    """Mean pairwise mixture distance per channel, against the verdict line.

    The one panel that says whether a channel carries a placement at all. A
    bar below the line is a channel whose gauges all report the same blend --
    KY7 pressure sits at 0.05-0.12 against everything else at 0.66-0.78 --
    so `n_max` gauges from it are one gauge repeated.
    """
    d = channels.reset_index(drop=True)
    labels = [f"{r['kind']}" + (f"\n{r['config_id']}" if "config_id" in d else "")
              for _, r in d.iterrows()]
    x = np.arange(len(d))
    colours = ["#2e8b57" if v == "ok" else "#c0392b" for v in d["verdict"]]
    ax.bar(x, d["mean_pairwise_tv"], color=colours)
    ax.axhline(min_tv, color="k", lw=.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=6)
    ax.set(ylabel="mean pairwise mixture distance", ylim=(0, 1),
           title="channel diversity — below the line is one reading repeated")
    return ax
