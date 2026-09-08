"""ipywidgets browsers: thin wiring over `figures`, nothing computed here.

Two browsers, matching the two questions the notebooks asked interactively:

* `preprocessing_browser` -- "what am I feeding the model": the client series
  after the transform with the selected window shaded, the window the model
  actually sees, and the same window under every scaling mode side by side.
  This is the one place that legitimately has to rebuild windows on a control
  change, so it takes an explicit bounded `WindowCache` and has a
  **Materialize** button: nothing is rebuilt until you ask. The notebooks
  rebuilt on every slider event against an unbounded module-level dict.

* `latent_browser` -- the latent cloud with prototypes overlaid, per round,
  per scope, with the `fit on` toggle. Reads **persisted tables only**, so
  what you see is what the run saved.

Embeddings are memoised per control combination: t-SNE on a few thousand
points is seconds, and a round slider would otherwise recompute it on every
step.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from . import figures as F
from .data import CACHE, WindowCache, apply_transform, build_windows, client_frames
from .specs import SCALINGS, TRANSFORMS
from .worlds import World

__all__ = ["preprocessing_browser", "latent_browser", "EmbeddingCache"]


class EmbeddingCache:
    """Bounded LRU for projections, keyed by the control combination."""

    def __init__(self, maxsize: int = 24):
        self.maxsize = int(maxsize)
        self._store: "OrderedDict[tuple, tuple]" = OrderedDict()

    def get_or_compute(self, key, fn):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        val = fn()
        self._store[key] = val
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return val

    def clear(self):
        self._store.clear()


def _require_widgets():
    try:
        import ipywidgets as W
        from IPython.display import display
        import matplotlib.pyplot as plt
    except ImportError as exc:            # pragma: no cover
        raise ImportError("the browsers need ipywidgets, IPython and "
                          "matplotlib; `figures` works without them") from exc
    return W, display, plt


# --------------------------------------------------------------------------
# preprocessing / EDA
# --------------------------------------------------------------------------
def preprocessing_browser(world: World, spec, cache: WindowCache | None = None,
                          max_clients: int = 3):
    """Browse placement x transform x scaling x geometry on the raw series.

    Rebuilds only on **Materialize**, and only for the selected clients, into
    the bounded cache you pass (default the package-level one).
    """
    W, display, plt = _require_widgets()
    cache = cache or CACHE
    plt.ioff()

    all_clients = world.client_names
    from .specs import PLACEMENTS
    w_cl = W.SelectMultiple(options=all_clients,
                            value=tuple(all_clients[:max_clients]),
                            rows=min(6, len(all_clients)), description="clients")
    w_pl = W.Dropdown(options=list(PLACEMENTS), value=spec.placement,
                      description="placement")
    w_tr = W.Dropdown(options=list(TRANSFORMS), value=spec.transform,
                      description="transform")
    w_sc = W.Dropdown(options=list(SCALINGS), value=spec.scaling,
                      description="scaling")
    w_ag = W.Dropdown(options=[1, 2, 3, 4, 6, 8],
                      value=spec.geometry.interval_agg_h, description="agg h")
    w_ws = W.Dropdown(options=[12, 24, 48, 84, 168],
                      value=spec.geometry.window_size, description="window")
    w_st = W.Dropdown(options=[1, 3, 6, 12, 24],
                      value=spec.geometry.step_size, description="stride")
    w_ch = W.IntSlider(value=0, min=0, max=8, description="channel")
    w_ix = W.IntSlider(value=0, min=0, max=100, description="window #",
                       continuous_update=False, layout=W.Layout(width="95%"))
    w_go = W.Button(description="Materialize", button_style="primary",
                    tooltip="rebuild windows for the current settings")
    w_cmp = W.Checkbox(value=True, description="compare scalings")
    out = W.Output()
    state: dict = {}

    def _spec_now():
        return spec.with_(placement=w_pl.value, transform=w_tr.value,
                          scaling=w_sc.value,
                          **{"geometry.interval_agg_h": w_ag.value,
                             "geometry.window_size": w_ws.value,
                             "geometry.step_size": w_st.value})

    def _materialize(_=None):
        with out:
            out.clear_output(wait=True)
            clients = list(w_cl.value)[:max_clients]
            if not clients:
                print("select at least one client")
                return
            s = _spec_now()
            try:
                fr = client_frames(world, s.placement, s.kinds, s.noise)
                fr = {c: fr[c] for c in clients}
                src = apply_transform(fr, s.transform, world.steps_day)
                fw, sc_tab, fl = build_windows(world, s, clients=tuple(clients),
                                               cache=cache)
                variants = {s.scaling: fw}
                if w_cmp.value:
                    for mode in ("minmax_ref", "shared", "zscore"):
                        if mode != s.scaling:
                            v, _, _ = build_windows(world, s.with_(scaling=mode),
                                                    clients=tuple(clients),
                                                    cache=cache)
                            variants[mode] = v
            except Exception as err:
                print(f"{type(err).__name__}: {err}")
                return
            state.update(src=src, fw=fw, fl=fl, clients=clients,
                         variants=variants, spec=s)
            n = min(len(fw[c]["windows"]) for c in clients)
            w_ix.max = max(0, n - 1)
            w_ch.max = max(0, fw[clients[0]]["windows"].shape[2] - 1)
            _draw()

    def _draw(_=None):
        if "fw" not in state:
            with out:
                out.clear_output(wait=True)
                print("press Materialize")
            return
        with out:
            out.clear_output(wait=True)
            s, src, fw = state["spec"], state["src"], state["fw"]
            clients, fl = state["clients"], state["fl"]
            idx = min(w_ix.value, min(len(fw[c]["windows"]) for c in clients) - 1)
            nrow = 3 if w_cmp.value else 2
            fig, ax = plt.subplots(nrow, 1, figsize=(12, 3 + 2.2 * nrow),
                                   gridspec_kw={"height_ratios":
                                                [2, 1, 1][:nrow]})
            start = int(fw[clients[0]]["window_start_step"][idx])
            span_len = int(s.geometry.window_size * s.geometry.interval_agg_h
                           / float(world.time["resolution_h"]))
            F.plot_client_series(ax[0], src, w_ch.value, clients,
                                 span=(start, start + span_len))
            ax[0].set_title(f"{world.tag} · {s.label()}", fontsize=9, loc="left")
            F.plot_window(ax[1], fw, idx, w_ch.value, clients)
            ax[1].set_title(f"window {idx} · month "
                            f"{int(fw[clients[0]]['labels'][idx])} · "
                            f"scaling={s.scaling}", fontsize=9, loc="left")
            if w_cmp.value:
                F.plot_scaling_compare(ax[2], state["variants"], idx,
                                       w_ch.value, clients[0])
            fig.tight_layout()
            display(fig)
            plt.close(fig)

            rr = _ref_table(fw, fl)
            print(f"\nreference-window bounds "
                  f"(identical across clients = per-client minmax): "
                  f"{rr.attrs['identical_bounds']}")
            print(rr.to_string(index=False))

    def _ref_table(fw, fl):
        from .data import ref_range_per_client
        return ref_range_per_client(fw, fl["preprocessing"]["reference_months"])

    w_go.on_click(_materialize)
    for w in (w_ix, w_ch):
        w.observe(_draw, names="value")
    for w in (w_cl, w_pl, w_tr, w_sc, w_ag, w_ws, w_st, w_cmp):
        w.observe(lambda *_: None, names="value")

    ui = W.VBox([W.HBox([w_cl, W.VBox([w_pl, w_tr, w_sc]),
                         W.VBox([w_ag, w_ws, w_st])]),
                 W.HBox([w_ch, w_cmp, w_go]), w_ix, out])
    display(ui)
    _materialize()
    return ui


# --------------------------------------------------------------------------
# latent space
# --------------------------------------------------------------------------
def latent_browser(world: World, res: dict, embeddings: EmbeddingCache | None = None,
                   max_points: int = 3000):
    """Browse the latent cloud + prototypes, per round, from saved tables.

    `res` is either a `run_cell` result or a `store.load_run` dict; the
    snapshot table is accepted under either name.
    """
    W, display, plt = _require_widgets()
    plt.ioff()
    emb = embeddings or EmbeddingCache()

    snaps = res.get("snapshots")
    if snaps is None:
        snaps = res.get("latents_by_round")
    lat = res.get("latent_trajectories")
    protos = res.get("prototypes")
    if snaps is None and lat is None:
        raise ValueError("no latent tables in this run")

    rounds = sorted(snaps["round"].unique()) if snaps is not None else []
    scopes = sorted(protos["scope"].unique()) if protos is not None else []
    clients = sorted((snaps if snaps is not None else lat)["district"].unique())

    w_src = W.Dropdown(options=([("per-round snapshots", "snaps")] if snaps is not None
                                else []) + ([("final, all windows", "final")]
                                            if lat is not None else []),
                       description="source")
    w_rd = W.SelectionSlider(options=rounds or [0], value=(rounds or [0])[-1],
                             description="round", continuous_update=False)
    w_stg = W.Dropdown(options=["pre", "post"], value="pre", description="stage")
    w_me = W.Dropdown(options=["PCA", "t-SNE", "UMAP"], value="PCA",
                      description="method")
    w_fit = W.Dropdown(options=[("fit on windows", "windows"),
                                ("fit on prototypes", "prototypes")],
                       value="windows", description="fit on")
    w_col = W.Dropdown(options=["district", "month", "phase", "drift_status"],
                       value="district", description="colour")
    w_sco = W.SelectMultiple(options=scopes or ["local"],
                             value=tuple(s for s in scopes if s == "local")
                             or tuple(scopes[:1]),
                             rows=4, description="scopes")
    w_cl = W.SelectMultiple(options=clients, value=tuple(clients),
                            rows=min(6, len(clients)), description="clients")
    w_tr = W.Checkbox(value=True, description="proto trajectory")
    w_ph = W.Dropdown(options=["all", "init", "final"], value="all",
                      description="phase")
    out = W.Output()

    def _draw(_=None):
        with out:
            out.clear_output(wait=True)
            use_snaps = w_src.value == "snaps" and snaps is not None
            d = snaps if use_snaps else lat
            d = d[d["district"].isin(list(w_cl.value))]
            if use_snaps:
                d = d[(d["round"] == w_rd.value)]
                if "stage" in d.columns:
                    d = d[d["stage"] == w_stg.value]
            if w_ph.value != "all":
                months = world.phase_months(w_ph.value)
                d = d[d["month"].isin(months)]
            if not len(d):
                print("no rows for this filter")
                return
            if len(d) > max_points:
                d = d.sample(max_points, random_state=0)
            d = d.sort_values(["district", "window"]).reset_index(drop=True)

            pr = None
            if protos is not None and len(w_sco.value):
                pr = protos[protos["scope"].isin(list(w_sco.value))]
                r = w_rd.value if use_snaps else int(pr["round"].max())
                pr = pr[pr["round"] == r]
                pr = pr[pr["scope"].eq("global")
                        | pr["client"].isin(list(w_cl.value))]
                if w_ph.value != "all":
                    pr = pr[pr["month"].isin(world.phase_months(w_ph.value))]
                pr = pr.reset_index(drop=True)

            key = (w_src.value, int(w_rd.value), w_stg.value, w_me.value,
                   w_fit.value, tuple(sorted(w_cl.value)),
                   tuple(sorted(w_sco.value)), w_ph.value, len(d))

            def _compute():
                Zp = d[F.fcols(d)].to_numpy()
                Zq = (pr[F.fcols(pr)].to_numpy() if pr is not None and len(pr)
                      else np.empty((0, Zp.shape[1])))
                return F.project(Zp, Zq, w_me.value, 0, w_fit.value)

            Ep, Eq, sub = emb.get_or_compute(key, _compute)

            fig, ax = plt.subplots(figsize=(8.5, 7))
            stage = f" · {w_stg.value}-FedAvg" if w_src.value == "snaps" else ""
            F.plot_latent(ax, d, Ep, pr, Eq, colour_by=w_col.value,
                          subtitle=f"{w_me.value}  {sub}{stage}  ·  n={len(d)}",
                          trajectory=w_tr.value)
            if w_col.value == "district":
                ax.legend(fontsize=7, frameon=False, markerscale=2, ncol=2)
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    for w in (w_src, w_rd, w_stg, w_me, w_fit, w_col, w_sco, w_cl, w_tr, w_ph):
        w.observe(_draw, names="value")

    ui = W.VBox([W.HBox([W.VBox([w_src, w_rd, w_stg]),
                         W.VBox([w_me, w_fit, w_col]),
                         W.VBox([w_ph, w_tr])]),
                 W.HBox([w_cl, w_sco]), out])
    display(ui)
    _draw()
    return ui
