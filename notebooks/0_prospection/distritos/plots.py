"""Seeing the partition, and seeing what the network has in it.

Three views:

* :func:`plot_districts` -- one partition, nodes coloured by district, with the
  hydraulic components (reservoirs, tanks, pumps, valves by subtype) drawn as
  distinct markers and, where an optimisation has run, boundary pipes coloured
  by whether they take a flow meter or a gate valve.
* :func:`plot_comparison` -- two partitions side by side over the same asset
  overlay, with the second recoloured to match the first so the eye compares
  *boundaries* rather than chasing an arbitrary colour permutation.
* :func:`interactive_districting` -- the factor dials, live.

Why the assets are drawn at all
-------------------------------
A district that looks tidy on a node plot can be a bad district because of what
it contains or excludes: a district with no source, a tank stranded on a
boundary, a pump sitting exactly on a cut. Those are the decisions the next
stage (sensor placement) inherits, and they are invisible unless the components
are on the plot.

All plotting uses ``plt.show()`` explicitly and never ``%matplotlib`` magic, so
the figures render identically in VS Code, Jupyter Lab and a headless export.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .graph import hydraulic_assets
from .weights import DistrictingWeights

#: Marker, colour and size per component class. Reservoirs and tanks are the
#: sources a district may or may not own; pumps and regulating valves are the
#: places where coupling becomes directional.
ASSET_STYLE = {
    "reservoir": {"marker": "s", "color": "#1b3a6b", "size": 95, "label": "reservoir"},
    "tank":      {"marker": "^", "color": "#0f8b8d", "size": 95, "label": "tank"},
    "pump":      {"marker": "D", "color": "#c1121f", "size": 55, "label": "pump"},
    "valve":     {"marker": "v", "color": "#e08e00", "size": 55, "label": "valve"},
}

DISTRICT_CMAP = "tab10"


# ==========================================================================
# helpers
# ==========================================================================

def _positions(wn) -> dict:
    return {n: wn.get_node(n).coordinates for n in wn.node_name_list
            if wn.get_node(n).coordinates}


def _draw_pipes(wn, pos, ax, color="0.86", lw=0.35):
    for name in wn.link_name_list:
        link = wn.get_link(name)
        a, b = link.start_node_name, link.end_node_name
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=color, lw=lw, zorder=1)


def _draw_assets(wn, pos, ax, kinds=("reservoir", "tank", "pump", "valve"),
                 assets: pd.DataFrame | None = None):
    """Overlay the hydraulic components; returns legend handles actually drawn."""
    assets = hydraulic_assets(wn) if assets is None else assets
    handles = []
    for kind in kinds:
        sub = assets[(assets.kind == kind) & assets.x.notna()]
        if sub.empty:
            continue
        st = ASSET_STYLE[kind]
        ax.scatter(sub.x, sub.y, marker=st["marker"], s=st["size"],
                   facecolor=st["color"], edgecolor="white", linewidths=0.8, zorder=6)
        handles.append(Line2D([], [], marker=st["marker"], linestyle="none",
                              markerfacecolor=st["color"], markeredgecolor="white",
                              markersize=8, label="%s (%d)" % (st["label"], len(sub))))
    return handles


def _draw_nodes(wn, pos, ax, labels: pd.Series, order=None, size=9):
    order = order if order is not None else sorted(labels.unique())
    cmap = plt.get_cmap(DISTRICT_CMAP)
    for i, d in enumerate(order):
        pts = [pos[n] for n in labels.index[labels == d] if n in pos]
        if pts:
            ax.scatter(*zip(*pts), s=size, color=cmap(i % 10), linewidths=0, zorder=3)
    return order


def _draw_status(wn, pos, ax, status: pd.DataFrame):
    """Boundary links coloured by the device they need: meter (open) or valve (closed)."""
    for _, r in status.iterrows():
        link = wn.get_link(r.link)
        a, b = link.start_node_name, link.end_node_name
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], lw=2.0, zorder=4,
                    color="tab:green" if r.status == "open" else "tab:red")


def _finish(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def match_labels(reference: pd.Series, other: pd.Series) -> dict:
    """Relabel ``other``'s districts to line up with ``reference``'s colours.

    Two partitions of the same network name their districts independently, so
    ``District_A`` in one has nothing to do with ``District_A`` in the other.
    Comparing them visually without matching means reading a colour permutation
    instead of a difference. Maximum-overlap assignment on the contingency table
    fixes the colours; it changes nothing about either partition, and it is used
    for display only -- never written to a ``districts.yml``.
    """
    from scipy.optimize import linear_sum_assignment

    common = reference.index.intersection(other.index)
    a, b = reference.loc[common], other.loc[common]
    ra, rb = sorted(a.unique()), sorted(b.unique())
    table = np.zeros((len(rb), len(ra)))
    for i, db in enumerate(rb):
        for j, da in enumerate(ra):
            table[i, j] = int(((b == db) & (a == da)).sum())
    rows, cols = linear_sum_assignment(-table)
    return {rb[i]: ra[j] for i, j in zip(rows, cols)}


# ==========================================================================
# one partition
# ==========================================================================

def plot_districts(wn, labels: pd.Series, ax=None, title: str = "",
                   status: pd.DataFrame | None = None, show_assets: bool = True,
                   node_size: int = 9, order=None, legend: bool = True):
    """One partition with the full hydraulic inventory overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    pos = _positions(wn)
    _draw_pipes(wn, pos, ax)
    order = _draw_nodes(wn, pos, ax, labels, order=order, size=node_size)
    if status is not None:
        _draw_status(wn, pos, ax, status)
    handles = _draw_assets(wn, pos, ax) if show_assets else []
    if legend and handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False,
                  handletextpad=0.2, borderaxespad=0.1)
    _finish(ax, title)
    return ax, order


# ==========================================================================
# two partitions
# ==========================================================================

def plot_comparison(wn, labels_a: pd.Series, labels_b: pd.Series,
                    name_a: str = "A", name_b: str = "B",
                    status_a: pd.DataFrame | None = None,
                    status_b: pd.DataFrame | None = None,
                    show_disagreement: bool = True, figsize=(15, 5.4),
                    node_size: int = 9):
    """Two methods side by side, colours matched, plus where they disagree.

    The third panel is the one that answers the question actually being asked --
    *where* do these two partitions differ -- which neither of the first two can
    show on its own. A node is grey when both methods put it in the same
    (matched) district and highlighted when they do not.
    """
    remap = match_labels(labels_a, labels_b)
    labels_b_matched = labels_b.map(lambda d: remap.get(d, d))
    order = sorted(labels_a.unique())

    n_panels = 3 if show_disagreement else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    pos = _positions(wn)

    plot_districts(wn, labels_a, axes[0], "%s — %d districts" % (name_a, labels_a.nunique()),
                   status=status_a, node_size=node_size, order=order, legend=True)
    plot_districts(wn, labels_b_matched, axes[1],
                   "%s — %d districts (colours matched to %s)"
                   % (name_b, labels_b.nunique(), name_a),
                   status=status_b, node_size=node_size, order=order, legend=False)

    if show_disagreement:
        ax = axes[2]
        _draw_pipes(wn, pos, ax)
        common = labels_a.index.intersection(labels_b_matched.index)
        same = [n for n in common if labels_a[n] == labels_b_matched[n] and n in pos]
        diff = [n for n in common if labels_a[n] != labels_b_matched[n] and n in pos]
        if same:
            ax.scatter(*zip(*[pos[n] for n in same]), s=node_size, color="0.78",
                       linewidths=0, zorder=3)
        if diff:
            ax.scatter(*zip(*[pos[n] for n in diff]), s=node_size * 2.2, color="#111111",
                       linewidths=0, zorder=4)
        _draw_assets(wn, pos, ax)
        pct = 100 * len(diff) / max(len(common), 1)
        _finish(ax, "disagreement — %d of %d nodes (%.1f%%)" % (len(diff), len(common), pct))

    fig.tight_layout()
    return fig, axes


def disagreement_table(labels_a: pd.Series, labels_b: pd.Series,
                       name_a: str = "A", name_b: str = "B") -> pd.DataFrame:
    """Contingency table of two partitions after colour matching.

    Reads as: how many of ``name_a``'s District_X nodes each of ``name_b``'s
    districts claimed. A near-diagonal table means the two methods agree on the
    structure and differ only at the edges; a smeared one means they disagree
    about what the districts *are*.
    """
    remap = match_labels(labels_a, labels_b)
    b = labels_b.map(lambda d: remap.get(d, d))
    common = labels_a.index.intersection(b.index)
    return pd.crosstab(labels_a.loc[common].rename(name_a), b.loc[common].rename(name_b))


# ==========================================================================
# interactive
# ==========================================================================

def interactive_comparison(wn, results: dict, default=None, show_table: bool = True):
    """Two dropdowns, two partitions, one figure -- pick any pair of methods.

    ``results`` is the ``{method: DistrictingResult}`` mapping from
    :func:`districting.runner.run_all`. Boundary status is drawn whenever the
    selected run actually optimised, so meters and gate valves appear on the
    methods that have them and are simply absent on the ones that did not.
    """
    import ipywidgets as widgets

    names = list(results)
    a0, b0 = (default or (names[0], names[1] if len(names) > 1 else names[0]))

    def render(method_a=a0, method_b=b0, show_disagreement=True, node_size=9):
        ra, rb = results[method_a], results[method_b]
        plot_comparison(wn, ra.labels, rb.labels, method_a, method_b,
                        status_a=ra.status if ra.best else None,
                        status_b=rb.status if rb.best else None,
                        show_disagreement=show_disagreement, node_size=node_size)
        plt.show()
        if show_table and method_a != method_b:
            print(disagreement_table(ra.labels, rb.labels, method_a, method_b))

    return widgets.interact(
        render,
        method_a=widgets.Dropdown(options=names, value=a0),
        method_b=widgets.Dropdown(options=names, value=b0),
        show_disagreement=widgets.Checkbox(value=True),
        node_size=widgets.IntSlider(min=4, max=24, value=9, continuous_update=False),
    )


def interactive_districting(wn, network: str = "network", reference: dict | None = None,
                            coupling: pd.DataFrame | None = None,
                            mapping: dict | None = None, k: int = 5):
    """The factor dials, live: move a slider, see the partition and its metrics.

    Sliders cover ``k``, the four factor weights, the balance target and
    tolerance, and the seed. The graph is built once and reused across every
    slider move, so only the partition itself is recomputed.

    The coupling dial is disabled unless a coupling matrix was supplied, rather
    than silently doing nothing -- which is how a dead factor reads as a bug.
    """
    import ipywidgets as widgets

    from .runner import DistrictingConfig, run
    from .graph import build_graph
    from .metrics import agreement

    g = build_graph(wn)

    def render(method="spectral", k=k, w_resistance=1.0, w_elevation=1.0,
               w_valve_cut=1.0, w_coupling=0.0, balance_target="default",
               balance_tol=0.15, seed=0, repair="default"):
        cfg = DistrictingConfig(
            k=k, method=method, seed=seed,
            weights=DistrictingWeights(w_resistance, w_elevation, w_valve_cut, w_coupling),
            balance_target=None if balance_target == "none" else balance_target,
            balance_tol=balance_tol,
            repair=None if repair == "default" else (repair == "on"))
        res = run(wn, cfg, network=network, coupling=coupling, mapping=mapping,
                  reference=reference, graph=g)
        rep = res.report

        print("%s / %s: requested k=%d, achieved %d, %d fragment(s) repaired"
              % (network, method, k, res.k, res.n_fragments_repaired))
        if method != "spectral":
            print("note: the factor dials apply to 'spectral' only -- %s uses bare topology"
                  % method)
        if not res.elevation_check["informative"] and w_elevation > 0:
            print("WARNING: elevation span is %.1f m here -- that dial is inert regardless "
                  "of its weight" % res.elevation_check["span_m"])
        print("balance: count %.2fx, demand %.2fx   |   contiguity: %s"
              % (rep["balance"]["count_imbalance"],
                 rep["balance"].get("demand_imbalance", np.nan),
                 "ALL CONTIGUOUS" if rep["contiguity"].contiguous.all()
                 else "FRAGMENTED — investigate"))
        print("structural coupling ratio %.4f (%s)   |   modularity %.4f   |   elevation F %.2f"
              % (rep["structural"]["structural_coupling_ratio"],
                 rep["structural"]["weight_used"], rep["modularity"],
                 rep["elevation"]["f_ratio"]))
        if reference:
            ag = agreement(res.labels, reference)
            print("vs published partition: AMI %.4f, ARI %.4f (n=%d)"
                  % (ag["ami"], ag["ari"], ag["n_common"]))
        print("boundary links: %d (%d not closable)"
              % (len(res.boundary), int((~res.boundary.closable).sum())))

        plot_districts(wn, res.labels, title="%s — %s, k=%d" % (network, method, res.k))
        plt.show()
        return res

    controls = dict(
        method=widgets.Dropdown(options=[m for m in ("spectral", "girvan_newman",
                                                     "fast_greedy", "walktrap")],
                                value="spectral"),
        k=widgets.IntSlider(min=2, max=10, value=k, continuous_update=False),
        w_resistance=widgets.FloatSlider(min=0, max=2, step=0.1, value=1.0, continuous_update=False),
        w_elevation=widgets.FloatSlider(min=0, max=2, step=0.1, value=1.0, continuous_update=False),
        w_valve_cut=widgets.FloatSlider(min=0, max=2, step=0.1, value=1.0, continuous_update=False),
        w_coupling=widgets.FloatSlider(min=0, max=2, step=0.1, value=0.0,
                                       continuous_update=False, disabled=coupling is None),
        balance_target=widgets.Dropdown(options=["default", "count", "demand", "none"],
                                        value="default"),
        balance_tol=widgets.FloatSlider(min=0.0, max=2.0, step=0.05, value=0.15,
                                        continuous_update=False),
        seed=widgets.IntSlider(min=0, max=20, value=0, continuous_update=False),
        repair=widgets.Dropdown(options=["default", "on", "off"], value="default"),
    )
    return widgets.interact(render, **controls)
