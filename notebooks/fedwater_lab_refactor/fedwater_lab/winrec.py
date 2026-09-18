"""Window reconstruction: what the AER does to single windows.

Prototypes are means; this module looks at the thing they average. It reads
an `aligned.AlignedRun` (its window table `wt`, row-aligned with the
decoded windows `decoded`) and the windows the run was trained on.

Objects
-------
For client c, window w (start t_w hours), channel i:

* ``x[w]``   the window the model saw (scaled, W steps x F channels);
* ``d[w]``   its reconstruction ``dec(enc(x[w]))`` (ry, y, fy rows stitched
  back into one W-step window, as `train.decode` returns it).

The AER encodes ALL channels of a window into one latent and decodes all of
them from it, so a channel's reconstruction is a slice of one joint output;
pure / mixed is a property of the slice, read from `recon.channel_table`.

Scores (per window, per channel)
--------------------------------
``rmse = sqrt(mean_t (d - x)^2)``;
``r    = corr_t(d, x)``                 shape agreement, level-free;
``amp  = std_t d / std_t x``            0 = flat reconstruction;
``lvl  = mean_t d - mean_t x``          level error.

Position profile
----------------
For each position t inside the window, over a set of windows w:
``r_t = corr_w(d[w][t], x[w][t])`` (does the reconstruction at position t
track the value there, window to window?) and
``rmse_t = sqrt(mean_w (d[w][t] - x[w][t])^2)``. A decoder that reproduces
the first steps and then settles to a constant shows r_t falling to ~0
after a few steps. The overlap-add timeline HIDES that (it averages the good
start of many windows), so read this profile before trusting the timeline.

Timeline (overlap-add)
----------------------
Consecutive windows overlap (stride < W), so every aggregated step s is
covered by several windows. The timeline is, per step,
``xhat[s] = mean over windows w covering s of d[w][s - start_w]``, drawn next
to the true aggregated series (which every covering window agrees on). It
shows the reconstruction as one continuous signal instead of 84-step
fragments; `count[s]` is the number of covering windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import recon as RC

__all__ = ["window_scores", "score_summary", "timeline", "step_profile",
           "plot_step_profile", "plot_window",
           "plot_timeline", "plot_score_grid", "window_browser", "WEEKDAYS"]

WEEKDAYS = ("d0", "d1", "d2", "d3", "d4", "d5", "d6")


def _client_arrays(run, fl_windows: dict, client: str):
    mask = (run.wt["client"] == client).to_numpy()
    sub = run.wt.loc[mask]
    X = np.asarray(fl_windows[client]["windows"], float)[sub["row"].to_numpy()]
    return sub.reset_index(drop=True), X, run.decoded[mask]


def window_scores(run, fl_windows: dict, channels: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """One row per (client, window, channel) with the four scores and the
    calendar keys (month, phase, weekday, hour)."""
    slot = ({} if channels is None else
            {(r.client, int(r.channel)): (r.slot_class, r.slot)
             for r in channels.itertuples(index=False)})
    rows = []
    for c in sorted(fl_windows):
        sub, X, D = _client_arrays(run, fl_windows, c)
        sx, sd = X.std(axis=1), D.std(axis=1)
        mx, md = X.mean(axis=1), D.mean(axis=1)
        err = np.sqrt(((D - X) ** 2).mean(axis=1))
        for k, w in enumerate(sub.itertuples(index=False)):
            for i in range(X.shape[2]):
                cls, sl = slot.get((c, i), ("all", str(i)))
                rows.append({
                    "client": c, "row": int(w.row), "start": int(w.start),
                    "month": int(w.month), "phase": w.phase,
                    "weekday": int(w.start_h // 24 % 7),
                    "hour": int(w.start_h % 24), "channel": i,
                    "slot_class": cls, "slot": sl,
                    "rmse": float(err[k, i]),
                    "r": RC._r(D[k, :, i], X[k, :, i]),
                    "amp": float(sd[k, i] / sx[k, i]) if sx[k, i] > 1e-12
                    else np.nan,
                    "lvl": float(md[k, i] - mx[k, i])})
    return pd.DataFrame(rows)


def score_summary(ws: pd.DataFrame, by=("phase", "slot_class")) -> pd.DataFrame:
    """Median scores per group (medians: r and amp are ratio-like)."""
    by = list(by)
    out = ws.groupby(by)[["rmse", "r", "amp", "lvl"]].median()
    out["n"] = ws.groupby(by).size()
    out = out.reset_index()
    if "phase" in by:
        out["phase"] = pd.Categorical(out["phase"], RC.PERIODS, ordered=True)
        out = out.sort_values(by)
    return out.reset_index(drop=True)


def timeline(run, fl_windows: dict, client: str, lo: int, hi: int
             ) -> pd.DataFrame:
    """Overlap-add reconstruction on aggregated steps [lo, hi).

    -> long frame: step, channel, true, recon, count. `step` is the
    aggregated step index (window_start_step / k)."""
    sub, X, D = _client_arrays(run, fl_windows, client)
    W, F = X.shape[1], X.shape[2]
    agg_starts = (sub["start"].to_numpy() // run.agg_k).astype(int)
    n = hi - lo
    acc = np.zeros((n, F))
    tru = np.full((n, F), np.nan)
    cnt = np.zeros(n)
    for k, s0 in enumerate(agg_starts):
        a, b = max(lo, s0), min(hi, s0 + W)
        if a >= b:
            continue
        acc[a - lo:b - lo] += D[k, a - s0:b - s0]
        tru[a - lo:b - lo] = X[k, a - s0:b - s0]
        cnt[a - lo:b - lo] += 1
    with np.errstate(invalid="ignore"):
        rec = acc / cnt[:, None]
    rec[cnt == 0] = np.nan
    steps = np.arange(lo, hi)
    return pd.DataFrame({"step": np.repeat(steps, F),
                         "channel": np.tile(np.arange(F), n),
                         "true": tru.ravel(), "recon": rec.ravel(),
                         "count": np.repeat(cnt, F)})


def step_profile(run, fl_windows: dict, client: str, rows=None
                 ) -> pd.DataFrame:
    """r_t and rmse_t per (position t, channel), over `rows` (window rows of
    `client`; default all)."""
    sub, X, D = _client_arrays(run, fl_windows, client)
    if rows is not None:
        keep = sub["row"].isin(list(rows)).to_numpy()
        X, D = X[keep], D[keep]
    out = []
    for i in range(X.shape[2]):
        x, d = X[:, :, i], D[:, :, i]
        xc, dc = x - x.mean(axis=0), d - d.mean(axis=0)
        den = np.sqrt((xc ** 2).sum(axis=0) * (dc ** 2).sum(axis=0))
        r = np.where(den > 1e-12, (xc * dc).sum(axis=0) / np.maximum(den, 1e-12),
                     np.nan)
        rmse = np.sqrt(((d - x) ** 2).mean(axis=0))
        out.append(pd.DataFrame({"step": np.arange(X.shape[1]), "channel": i,
                                 "r": r, "rmse": rmse}))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_step_profile(ax, prof: pd.DataFrame, agg_h: float, title: str = ""):
    """r_t per channel against position in the window (days)."""
    from .figures import PALETTE
    for i, d in prof.groupby("channel"):
        ax.plot(d["step"] * agg_h / 24, d["r"], color=PALETTE[int(i) % 8],
                label=f"channel {i}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="position in window (days)",
           ylabel="corr over windows at that position", ylim=(-0.2, 1.02))
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7)
    return ax



def plot_window(ax, run, fl_windows: dict, client: str, row: int,
                channel: int, ctrl=None, title: str = "",
                ctrl_label: str = "untrained model"):
    """One window: true vs reconstruction (and the untrained control's
    reconstruction when `ctrl`, an AlignedRun of the control, is given)."""
    sub, X, D = _client_arrays(run, fl_windows, client)
    k = int(np.flatnonzero(sub["row"].to_numpy() == row)[0])
    ax.plot(X[k, :, channel], color="k", lw=1.8, label="true window")
    ax.plot(D[k, :, channel], color="#DD8452", lw=1.5,
            label=getattr(run, "label", None) or "reconstruction")
    if ctrl is not None:
        _, _, Dc = _client_arrays(ctrl, fl_windows, client)
        kc = int(np.flatnonzero(_client_arrays(ctrl, fl_windows, client)[0]
                                ["row"].to_numpy() == row)[0])
        ax.plot(Dc[kc, :, channel], color="#8C8C8C", lw=1.0, ls="--",
                label=ctrl_label)
    steps_day = int(round(24 / run.agg_h))
    for s in range(0, X.shape[1], steps_day):
        ax.axvline(s, color="0.85", lw=0.6, zorder=0)
    ax.set(xlabel="step (window resolution; grid = 1 day from start)",
           ylabel="scaled value")
    ax.set_title(title, fontsize=9, loc="left")
    return ax


def plot_timeline(ax, tl: pd.DataFrame, channel: int, agg_h: float,
                  title: str = ""):
    """Overlap-add timeline for one channel, weekend days shaded
    (weekday 5 and 6, day mod 7 from step 0)."""
    d = tl[tl["channel"] == channel]
    hours = d["step"].to_numpy() * agg_h
    ax.plot(hours / 24, d["true"], color="k", lw=1.2, label="true series")
    ax.plot(hours / 24, d["recon"], color="#DD8452", lw=1.2,
            label="overlap-add reconstruction")
    if len(d):
        d0, d1 = int(hours.min() // 24), int(np.ceil(hours.max() / 24))
        for day in range(d0, d1 + 1):
            if day % 7 >= 5:
                ax.axvspan(day, day + 1, color="#8172B3", alpha=0.08, lw=0)
    ax.set(xlabel="day (from step 0; shaded = weekday 5, 6)",
           ylabel="scaled value")
    ax.set_title(title, fontsize=9, loc="left")
    return ax


def plot_score_grid(ax, ws: pd.DataFrame, client: str, metric: str = "r",
                    slot_class: str | None = None, vmin=-1, vmax=1,
                    cmap="RdBu_r"):
    """month x start class (weekday·hour) grid of the median `metric`."""
    d = ws[ws["client"] == client]
    if slot_class is not None:
        d = d[d["slot_class"] == slot_class]
    d = d.assign(cls=[f"d{a}·{b:02d}h" for a, b in zip(d["weekday"], d["hour"])])
    piv = d.pivot_table(index="cls", columns="month", values=metric,
                        aggfunc="median")
    piv = piv.reindex(sorted(piv.index))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin,
                   vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    xt = list(range(0, len(piv.columns), max(1, len(piv.columns) // 13)))
    ax.set_xticks(xt)
    ax.set_xticklabels([piv.columns[i] for i in xt], fontsize=7)
    ax.set(xlabel="month", ylabel="start class")
    ax.set_title(f"{client} · median {metric}"
                 + (f" · {slot_class}" if slot_class else ""),
                 fontsize=9, loc="left")
    return im


# --------------------------------------------------------------------------
# browser
# --------------------------------------------------------------------------
def window_browser(run, fl_windows: dict, ws: pd.DataFrame | None = None,
                   channels: pd.DataFrame | None = None, ctrl=None,
                   ctrl_label: str = "untrained control",
                   run_label: str = "reconstruction"):
    """client / channel / month / window-in-month.

    Top: the selected window, true vs reconstruction (+ untrained control).
    Bottom: the overlap-add timeline over `span` days centred on it, with
    the window's own span marked. The table below lists the scores of that
    window on every channel."""
    from .widgets import _channel_label, _require_widgets
    W, display, plt = _require_widgets()
    plt.ioff()
    clients = sorted(fl_windows)
    months = sorted(int(m) for m in run.wt["month"].unique())
    Wlen = next(iter(fl_windows.values()))["windows"].shape[1]
    n_ch = next(iter(fl_windows.values()))["windows"].shape[2]
    spd = int(round(24 / run.agg_h))

    w_cl = W.Dropdown(options=clients, description="client")
    w_ch = W.Dropdown(description="channel", layout=W.Layout(width="520px"))
    w_mo = W.SelectionSlider(options=months, description="month",
                             continuous_update=False)
    w_wi = W.Dropdown(description="window")
    w_sp = W.IntSlider(value=21, min=7, max=70, step=7, description="span (d)",
                       continuous_update=False)
    w_ct = W.Checkbox(value=ctrl is not None, description=ctrl_label,
                      disabled=ctrl is None)
    out = W.Output()

    def _set_channels(*_):
        keep = w_ch.value
        if channels is not None and len(channels):
            sub = channels[channels["client"] == w_cl.value]
            opts = [(_channel_label(r), int(r["channel"]))
                    for _, r in sub.iterrows()]
        else:
            opts = [(str(i), i) for i in range(n_ch)]
        w_ch.options = opts
        valid = [v for _, v in opts]
        w_ch.value = keep if keep in valid else valid[0]

    def _set_windows(*_):
        sub = run.wt[(run.wt["client"] == w_cl.value)
                     & (run.wt["month"] == w_mo.value)].sort_values("start")
        opts = [(f"row {int(r.row)} · d{int(r.start_h // 24 % 7)}·"
                 f"{int(r.start_h % 24):02d}h · day {r.start_h / 24:.1f}",
                 int(r.row)) for r in sub.itertuples()]
        w_wi.options = opts
        if opts:
            w_wi.value = opts[0][1]

    def _draw(*_):
        with out:
            out.clear_output(wait=True)
            if w_wi.value is None or w_ch.value is None:
                return
            c, row, ch = w_cl.value, int(w_wi.value), int(w_ch.value)
            r = run.wt[(run.wt["client"] == c) & (run.wt["row"] == row)].iloc[0]
            s0 = int(r["start"] // run.agg_k)
            half = w_sp.value * spd // 2
            lo = max(0, s0 + Wlen // 2 - half)
            tl = timeline(run, fl_windows, c, lo, lo + 2 * half)
            fig, ax = plt.subplots(2, 1, figsize=(13, 7))
            plot_window(ax[0], run, fl_windows, c, row, ch,
                        ctrl if w_ct.value else None, ctrl_label=ctrl_label,
                        title=f"{c} · month {int(r['month'])} ({r['phase']}) · "
                              f"start d{int(r['start_h'] // 24 % 7)}·"
                              f"{int(r['start_h'] % 24):02d}h · channel {ch}")
            ax[0].legend(fontsize=7)
            plot_timeline(ax[1], tl, ch, run.agg_h,
                          title=f"overlap-add, {w_sp.value} days around it")
            a, b = s0 * run.agg_h / 24, (s0 + Wlen) * run.agg_h / 24
            ax[1].axvspan(a, b, color="#DD8452", alpha=0.08, lw=0)
            ax[1].legend(fontsize=7)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            if ws is not None and len(ws):
                t = ws[(ws["client"] == c) & (ws["row"] == row)]
                print(t[["channel", "slot", "slot_class", "rmse", "r", "amp",
                         "lvl"]].round(3).to_string(index=False))

    w_cl.observe(lambda *_: (_set_channels(), _set_windows(), _draw()),
                 names="value")
    w_mo.observe(lambda *_: (_set_windows(), _draw()), names="value")
    for x in (w_ch, w_wi, w_sp, w_ct):
        x.observe(_draw, names="value")
    _set_channels()
    _set_windows()
    run.label = run_label
    ui = W.VBox([W.HBox([w_cl, w_ch]), W.HBox([w_mo, w_wi]),
                 W.HBox([w_sp, w_ct]), out])
    display(ui)
    _draw()
    return ui
