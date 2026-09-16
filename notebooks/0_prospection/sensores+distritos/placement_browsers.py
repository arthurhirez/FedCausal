"""placement_browsers.py -- the three interactive composites.

Wiring only: every number shown comes from a table `district_sweep` persisted
or from a `Probe` the caller already holds. Nothing is estimated here.

    sweep_browser     the cross-configuration view: response/leak matrix, and
                      the resemblance metric drawn ON THE NETWORK with the
                      gauge count on a slider.
    mixture_browser   one configuration, four panels: tiers on the network
                      (top-left), the per-district mixture simplex (top-right),
                      dependence and response heatmaps (second row).
    lab_browser       one world: weekly shapes, the selection map, and the
                      shape-distance clustering.

`sweep_browser` and `mixture_browser` read PERSISTED TABLES ONLY, so what you
see is what the store holds and they work on a `district_sweep.load`ed result
with no world on disk. `lab_browser` needs a live `Probe`, because weekly
shapes and the clustering are functions of the series rather than of any
summary table.

VS Code: `plt.ioff()` plus an explicit `show()` inside the `Output` widget.
Without the first, every redraw leaks a second copy of the figure into the
notebook; without the second, nothing renders at all.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

import placement_figures as F
import placement as pl
import signal_probe as sp

__all__ = ["sweep_browser", "mixture_browser", "lab_browser"]


def _require():
    try:
        import ipywidgets as W
        from IPython.display import display
        import matplotlib.pyplot as plt
    except ImportError as exc:            # pragma: no cover
        raise ImportError("the browsers need ipywidgets, IPython and "
                          "matplotlib; `placement_figures` works without "
                          "them") from exc
    return W, display, plt


class _Memo:
    """Small bounded LRU. Agglomerative clustering on a few hundred gauges is
    seconds, and a kind toggle would otherwise recompute it on every click."""

    def __init__(self, maxsize=16):
        self.maxsize, self._s = int(maxsize), OrderedDict()

    def __call__(self, key, fn):
        if key in self._s:
            self._s.move_to_end(key)
            return self._s[key]
        v = self._s[key] = fn()
        while len(self._s) > self.maxsize:
            self._s.popitem(last=False)
        return v


# ==========================================================================
# 1. the sweep composite
# ==========================================================================
def sweep_browser(results: dict, topo: dict, figsize=(14.5, 6.0)):
    """Response matrix + the resemblance metric on the network.

    Left panel is the leak/response matrix: rows the district that drifted,
    columns where the response was measured. The diagonal is signal, the rest
    is hydraulic contamination, and the ratio of the two is the de-confounding
    problem stated in numbers for that partition.

    Right panel replaces the resemblance/response scatter. `# sensors` keeps
    the top N per district by the chosen metric; the metric drives colour and
    size together, so a district whose best gauges are faint and thin is one
    whose signal no placement recovers.
    """
    W, display, plt = _require()
    plt.ioff()
    ok = {k: r for k, r in results.items() if len(r.mixture) or len(r.carriers)}
    if not ok:
        raise ValueError("no configuration in `results` has any analysis")

    w_cfg = W.Dropdown(options=sorted(ok), description="config",
                       layout=W.Layout(width="330px"))
    w_kind = W.ToggleButtons(options=["flow", "pressure"], value="flow",
                             description="channel")
    w_src = W.Dropdown(options=[("response R[j,k]", "R"),
                                ("leak (sweep §4)", "leak")],
                       value="R", description="matrix",
                       layout=W.Layout(width="300px"))
    w_met = W.Dropdown(options=[("resemblance r_mean", "r_mean"),
                                ("purity", "purity"),
                                ("response SNR", "snr_home"),
                                ("second-district weight", "w_second")],
                       value="r_mean", description="metric",
                       layout=W.Layout(width="330px"))
    w_n = W.IntSlider(value=5, min=1, max=25, step=1, description="# sensors",
                      continuous_update=False, layout=W.Layout(width="420px"))
    out = W.Output()

    def _metric_table(r, metric):
        """`carriers` carries r_mean, `mixture` carries the rest."""
        if metric == "r_mean":
            if not len(r.carriers):
                return pd.DataFrame()
            d = r.carriers.rename(columns={"district": "home"}).copy()
            d["element"] = [i[len(sp._PREFIX[k]):]
                            for i, k in zip(d["id"], d["kind"])]
            return d
        return r.mixture if len(r.mixture) else pd.DataFrame()

    def _draw(_=None):
        with out:
            out.clear_output(wait=True)
            r = ok[w_cfg.value]
            kind, metric = w_kind.value, w_met.value
            fig, axes = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={"width_ratios": [1, 1.45]})
            A = (r.response.get((kind, "all")) if w_src.value == "R"
                 else r.leak.get(kind))
            if A is not None and len(A):
                V = A.to_numpy(dtype=float)
                off = V[~np.eye(len(V), dtype=bool)]
                F.matrix(axes[0], A, vmax=float(np.nanmax(V)) or 1.0,
                         title=(f"{r.method} -- {kind} {w_src.value}\n"
                                f"diag {np.nanmean(np.diag(V)):.3f} | off "
                                f"{np.nanmean(off):.3f} | ratio "
                                f"{np.nanmean(np.diag(V))/max(np.nanmean(off),1e-9):.1f}x"),
                         xlabel="measured at", ylabel="drifted")
            else:
                axes[0].text(.5, .5, f"no {w_src.value} matrix\n({r.status})",
                             ha="center", va="center", fontsize=9)
                axes[0].set_axis_off()

            tab = _metric_table(r, metric)
            if len(tab) and metric in tab.columns:
                F.metric_network(axes[1], topo, tab, r.districts, kind=kind,
                                 value=metric, top_n=w_n.value,
                                 title=f"{r.method} -- top {w_n.value}/district "
                                       f"by {metric} ({kind})")
            else:
                axes[1].text(.5, .5, f"{metric} unavailable", ha="center",
                             va="center", fontsize=9)
                axes[1].set_axis_off()
            fig.suptitle(f"{r.config_id}  [{r.status}]", fontsize=10)
            plt.tight_layout()
            plt.show()

    for w in (w_cfg, w_kind, w_src, w_met, w_n):
        w.observe(_draw, names="value")
    _draw()
    display(W.VBox([W.HBox([w_cfg, w_kind]), W.HBox([w_src, w_met]), w_n, out]))
    return {"config": w_cfg, "kind": w_kind, "metric": w_met, "n": w_n}


# ==========================================================================
# 2. the mixture composite
# ==========================================================================
def mixture_browser(result, topo: dict, figsize=(14.5, 11.0), top_bars=20):
    """One configuration: tiers, simplex, dependence, response.

    Layout is fixed by what the panels are for. The network is top-left
    because every other panel is a summary OF it. The simplex bars are
    top-right, one district at a time -- five stacked at once is unreadable at
    this width, and the district selector costs one click.

    Second row is the two matrices, and they are not the same object.
    `dependence R[j,k]` reads "j's meters respond R as strongly to k's drift as
    to j's own" and uses EVERY gauge homed in j, no threshold and no tier.
    `response` is the transposed view restricted to a tier -- `core` should be
    near-diagonal by construction, and if `transition` is not markedly more
    coupled then the tiering is not separating anything.
    """
    W, display, plt = _require()
    plt.ioff()
    if not len(result.mixture):
        raise ValueError(f"{result.config_id}: no mixture "
                         f"(status={result.status!r}: {result.error})")
    table = pl.classification(result)

    w_kind = W.ToggleButtons(options=["flow", "pressure"], value="flow",
                             description="channel")
    w_dis = W.Dropdown(options=result.names, value=result.names[0],
                       description="district", layout=W.Layout(width="300px"))
    uses = sorted({u for (k, u) in result.response}) or ["all"]
    w_use = W.ToggleButtons(options=uses,
                            value="all" if "all" in uses else uses[0],
                            description="gauges")
    w_top = W.IntSlider(value=top_bars, min=5, max=60, step=5,
                        description="bars", continuous_update=False,
                        layout=W.Layout(width="380px"))
    w_unu = W.Checkbox(value=True, description="show unusable")
    out = W.Output()

    def _draw(_=None):
        with out:
            out.clear_output(wait=True)
            kind = w_kind.value
            fig, axes = plt.subplots(2, 2, figsize=figsize,
                                     gridspec_kw={"height_ratios": [1.25, 1]})
            F.tier_network(axes[0][0], topo, table, result.districts, kind=kind,
                           show_unusable=w_unu.value,
                           title=f"{result.method} -- {kind} tiers")
            try:
                F.mixture_bars(axes[0][1], result, w_dis.value, kind=kind,
                               top=w_top.value)
            except Exception as exc:                      # noqa: BLE001
                axes[0][1].text(.5, .5, f"no live gauge\n{type(exc).__name__}",
                                ha="center", va="center", fontsize=9)
                axes[0][1].set_axis_off()

            dep = result.dependence.get(kind)
            if dep is not None and len(dep["R"]):
                F.matrix(axes[1][0], dep["R"], vmax=1.0,
                         title=f"dependence  R[j,k] -- {kind}",
                         xlabel="whose drift (k)", ylabel="observed at (j)")
            else:
                axes[1][0].set_axis_off()
            A = result.response.get((kind, w_use.value))
            if A is not None and len(A):
                V = A.to_numpy(dtype=float)
                off = V[~np.eye(len(V), dtype=bool)]
                F.matrix(axes[1][1], A, vmax=1.0,
                         title=(f"response -- {kind}, {w_use.value}  "
                                f"(diag {np.nanmean(np.diag(V)):.2f} / off "
                                f"{np.nanmean(off):.2f})"),
                         xlabel="measured at", ylabel="drifted")
            else:
                axes[1][1].text(.5, .5, f"no `{w_use.value}` gauges of this kind",
                                ha="center", va="center", fontsize=9)
                axes[1][1].set_axis_off()
            ch = pl.channel_diversity(table)
            v = ch.loc[ch["kind"] == kind, "verdict"]
            flag = ("  [CHANNEL DEGENERATE — every gauge reports the same blend]"
                    if len(v) and v.iat[0] != "ok" else "")
            fig.suptitle(f"{result.config_id}{flag}", fontsize=11,
                         color=("#c0392b" if flag else "black"))
            plt.tight_layout()
            plt.show()

    for w in (w_kind, w_dis, w_use, w_top, w_unu):
        w.observe(_draw, names="value")
    _draw()
    display(W.VBox([W.HBox([w_kind, w_dis]), W.HBox([w_use, w_top, w_unu]), out]))
    return {"kind": w_kind, "district": w_dis, "use": w_use}


# ==========================================================================
# 3. the one-world composite
# ==========================================================================
def lab_browser(P, scores: pd.DataFrame, cand: pd.DataFrame | None = None,
                figsize=(14.5, 9.0)):
    """Weekly shapes, the selection map, and the shape-distance clustering.

    The selection map's upper right is where a prototype comes from: a gauge
    that tracks the district's shape in both phases and sits at response ~ 0
    is dominated by transit flow -- it looks like the district because the
    district's water passes through it, and does not move when the district's
    behaviour changes.

    The clustering carries two ARIs. Against DISTRICTS is the placement
    question (can a gauge be traced to its client); against REGIMES is the
    thesis question (is the land-use signature legible at all). Districts
    sharing a regime token cannot be separated by shape, so a low district-ARI
    with a high regime-ARI is the demand model working, not failing.
    """
    W, display, plt = _require()
    plt.ioff()
    cand = sp.candidates(P) if cand is None else cand
    memo = _Memo()

    districts = sorted(set(cand["district"].dropna()))
    tgt = P.drift.get("tgt_district")
    w_kind = W.ToggleButtons(options=["flow", "pressure"], value="flow",
                             description="channel")
    w_dis = W.Dropdown(options=districts,
                       value=tgt if tgt in districts else districts[0],
                       description="district", layout=W.Layout(width="300px"))
    w_phase = W.ToggleButtons(options=["init", "final"], value="init",
                              description="phase")
    w_n = W.IntSlider(value=2, min=1, max=6, description="# best",
                      continuous_update=False, layout=W.Layout(width="340px"))
    out = W.Output()

    def _draw(_=None):
        with out:
            out.clear_output(wait=True)
            kind = w_kind.value
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
            ax_i = fig.add_subplot(gs[0, 0])
            ax_f = fig.add_subplot(gs[0, 1], sharey=ax_i)
            F.weekly_shapes([ax_i, ax_f], P, w_dis.value, scores, kind=kind,
                            n=w_n.value)
            ax_s = fig.add_subplot(gs[1, 0])
            sp.plot_selection_map(scores, w_phase.value, ax=ax_s)
            ax_c = fig.add_subplot(gs[1, 1])
            cl = memo((w_phase.value, kind),
                      lambda: sp.cluster(P, cand, phase=w_phase.value, kind=kind))
            sp.plot_cluster(cl, ax=ax_c)
            fig.suptitle(f"{P.label} -- {kind}", fontsize=10)
            plt.tight_layout()
            plt.show()

    for w in (w_kind, w_dis, w_phase, w_n):
        w.observe(_draw, names="value")
    _draw()
    display(W.VBox([W.HBox([w_kind, w_dis]), W.HBox([w_phase, w_n]), out]))
    return {"kind": w_kind, "district": w_dis, "phase": w_phase}
