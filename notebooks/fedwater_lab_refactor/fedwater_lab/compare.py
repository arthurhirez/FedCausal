"""Comparing schedules: how visible is the drift, and to whom?

Every section of POC_02 describes ONE trained cell. This module puts several
side by side and asks the thesis question directly: does a regime change in
one client show up in the artifacts, and do the others stay put?

Per (client, month) it builds two signals from an `aligned.AlignedRun`:

* **reconstruction** -- median over that month's windows and channels of
  `r` (corr with the truth), `rmse` and `amp`. A model that no longer fits a
  client after a change shows `r` falling. This signal only works when the
  model fits in the first place, which POC_02 v4 finally achieved.
* **latent displacement** -- the month's mean latent against the client's
  own reference-phase mean:
  `e_lat = ||z_m - z_ref|| / ||z_ref||` and `cos_lat = 1 - cos(z_m, z_ref)`.
  This needs no reconstruction quality at all, which is why it worked in
  earlier POCs when decoding did not.

`visibility` turns a signal into one number per client, using only months
the model can be compared across:

    z = (median over post months - median over reference months)
        / (IQR over reference months)

with `post` = months from `world.first_switch` on. A drift is visible when
the drifting client's `z` is large AND the others' are not; `contrast` is
the drifting client's `z` minus the largest of the rest. Sign convention:
for `r` the signal is negated first, so positive `z` always means "moved".

The comparison is only fair when the cells being compared share the world,
the sensor selection, the geometry and the scaler, which is what running
them from one notebook section guarantees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recon import _r as RC_r
from .worlds import World

__all__ = ["drift_profile", "visibility", "compare_visibility",
           "plot_profiles", "plot_prequential", "prototype_similarity",
           "similarity_fidelity", "plot_similarity", "SIGNALS"]

SIGNALS = ("r", "rmse", "e_lat", "cos_lat")
_LOWER_IS_MOVED = ("r", "amp")            # these fall when the fit breaks


def drift_profile(world: World, run, fl_windows: dict,
                  channels: pd.DataFrame | None = None,
                  slot_class: str | None = None,
                  ref_phase: str = "init") -> pd.DataFrame:
    """One row per (client, month): reconstruction medians and latent
    displacement from the client's `ref_phase` mean latent."""
    from . import winrec as WR
    ws = WR.window_scores(run, fl_windows, channels)
    if slot_class is not None:
        ws = ws[ws["slot_class"] == slot_class]
    rec = (ws.groupby(["client", "month", "phase"])[["r", "rmse", "amp"]]
           .median().reset_index())

    fc = [c for c in run.wt.columns if c.startswith("f") and c[1:].isdigit()]
    rows = []
    for c, d in run.wt.groupby("client"):
        ref = d[d["phase"] == ref_phase]
        if not len(ref):
            continue
        zr = ref[fc].to_numpy(float).mean(axis=0)
        nr = max(float(np.linalg.norm(zr)), 1e-12)
        for m, dm in d.groupby("month"):
            z = dm[fc].to_numpy(float).mean(axis=0)
            rows.append({"client": c, "month": int(m), "n": len(dm),
                         "e_lat": float(np.linalg.norm(z - zr)) / nr,
                         "cos_lat": 1.0 - float(
                             z @ zr / max(float(np.linalg.norm(z)) * nr, 1e-12))})
    return rec.merge(pd.DataFrame(rows), on=["client", "month"], how="outer")


def visibility(prof: pd.DataFrame, world: World, signals=SIGNALS,
               ref_phase: str = "init") -> pd.DataFrame:
    """One z per (client, signal), plus the contrast against the other
    clients (see the module doc). `post` months start at
    `world.first_switch`."""
    switch = int(world.first_switch)
    ref = prof[prof["phase"] == ref_phase]
    post = prof[prof["month"] >= switch]
    rows = []
    for c in sorted(prof["client"].unique()):
        r0, r1 = ref[ref["client"] == c], post[post["client"] == c]
        for s in signals:
            if s not in prof.columns or not len(r0) or not len(r1):
                continue
            sign = -1.0 if s in _LOWER_IS_MOVED else 1.0
            a, b = sign * r0[s].to_numpy(), sign * r1[s].to_numpy()
            iqr = float(np.subtract(*np.percentile(a[~np.isnan(a)], [75, 25]))) \
                if np.isfinite(a).any() else np.nan
            iqr = abs(iqr)
            rows.append({"client": c, "signal": s,
                         "ref_median": float(np.nanmedian(a)) * sign,
                         "post_median": float(np.nanmedian(b)) * sign,
                         "ref_iqr": iqr,
                         "z": (float(np.nanmedian(b) - np.nanmedian(a))
                               / iqr if iqr > 1e-12 else np.nan)})
    out = pd.DataFrame(rows)
    out["contrast"] = np.nan
    drift = world.drift_district
    for _, g in out.groupby("signal"):                 # only on the drift row
        d = g[g["client"] == drift]
        others = g[g["client"] != drift]["z"]
        if len(d) and len(others):
            out.loc[d.index, "contrast"] = float(d["z"].iloc[0]) - float(others.max())
    return out


def compare_visibility(vis: dict, world: World, signal: str = "e_lat"
                       ) -> pd.DataFrame:
    """{cell name: visibility frame} -> clients x cells for one signal, with
    the contrast row (drifting client minus the loudest other)."""
    cols = {}
    for name, v in vis.items():
        g = v[v["signal"] == signal].set_index("client")
        cols[name] = g["z"]
    t = pd.DataFrame(cols)
    d = world.drift_district
    if d in t.index:
        t.loc["contrast"] = t.loc[d] - t.drop(index=d).max()
    return t


def plot_profiles(axes, profs: dict, world: World, signal: str = "e_lat",
                  slot_note: str = ""):
    """One panel per cell: `signal` over months, one line per client, with
    the drift district bold and the transition shaded."""
    from .figures import colour_map
    clients = sorted(next(iter(profs.values()))["client"].unique())
    cm = colour_map(clients)
    ph = world.phases()
    for ax, (name, p) in zip(np.ravel(axes), profs.items()):
        for c in clients:
            d = p[p["client"] == c].sort_values("month")
            drift = c == world.drift_district
            ax.plot(d["month"], d[signal], color=cm[c], label=c,
                    lw=2.2 if drift else 1.0, zorder=3 if drift else 2)
        ax.axvspan(ph["init"][1], ph["final"][0], color="0.85", alpha=0.5,
                   lw=0, zorder=0)
        ax.set(xlabel="month", ylabel=signal)
        ax.set_title(f"{name}{slot_note}", fontsize=9, loc="left")
    np.ravel(axes)[0].legend(fontsize=7, ncol=2)
    return axes


def plot_prequential(ax, preq: pd.DataFrame, world: World, metric: str = "r",
                     model: str = "global"):
    """Streaming only: the score of each block BEFORE it was trained on,
    one line per client. A block whose regime the carried-over model has not
    seen shows up as a dip."""
    from .figures import colour_map
    d = preq[preq["model"] == model]
    cm = colour_map(sorted(d["client"].unique()))
    for c, g in d.groupby("client"):
        g = g.sort_values("block")
        x = (g["m_lo"] + g["m_hi"]) / 2
        ax.plot(x, g[metric], marker="o", color=cm[c], label=c,
                lw=2.2 if c == world.drift_district else 1.0)
    ph = world.phases()
    ax.axvspan(ph["init"][1], ph["final"][0], color="0.85", alpha=0.5, lw=0,
               zorder=0)
    ax.set(xlabel="block (month range midpoint)",
           ylabel=f"{metric} on the UNSEEN block ({model} model)")
    ax.legend(fontsize=7, ncol=2)
    return ax


# --------------------------------------------------------------------------
# are the prototypes good enough for the NEXT step?
# --------------------------------------------------------------------------
def prototype_similarity(run, channels: pd.DataFrame | None = None,
                         slot_class: str | None = "pure") -> pd.DataFrame:
    """Client-to-client similarity per calendar class, in three spaces.

    For an `aligned.AlignedRun` built with a calendar key (`ALIGN168`), and
    for each group g = (block, cycle_pos) and each ordered pair c < d:

    s_true = mean over the selected channels i of corr_t(T[c,g,i], T[d,g,i])
             -- the data's own answer, the ceiling;
    s_dec  = the same on the DECODED prototypes dec(p[c,g]);
    s_lat  = cos(p[c,g], p[d,g]) -- the raw latent prototype, the artifact a
             federated dependence step would actually exchange.

    Channels are compared index by index, which is what makes sense here
    because `data.client_frames` orders them by slot in every client;
    `slot_class` restricts to pure or mixed slots.

    The point of the three columns is that `s_true` is computable ONLY with
    the raw data, so it is the yardstick the other two are scored against
    (`similarity_fidelity`)."""
    g = run.groups.set_index("gid")
    true, dec = run.arrays["true"], run.arrays.get("aligned", {})
    fc = [c for c in run.protos.columns if c.startswith("f") and c[1:].isdigit()]
    lat = {(r.client, int(r.gid)): np.array([getattr(r, f) for f in fc], float)
           for r in run.protos.itertuples(index=False)}
    keep = None
    if channels is not None and slot_class is not None:
        keep = {c: set(sub.loc[sub["slot_class"] == slot_class, "channel"]
                       .astype(int))
                for c, sub in channels.groupby("client")}

    by_gid: dict = {}
    for (c, gid) in true:
        by_gid.setdefault(int(gid), []).append(c)
    rows = []
    for gid, cs in by_gid.items():
        cs = sorted(cs)
        info = g.loc[gid]
        for i, c in enumerate(cs):
            for d in cs[i + 1:]:
                Tc, Td = true[(c, gid)], true[(d, gid)]
                idx = range(Tc.shape[1]) if keep is None else sorted(
                    keep.get(c, set()) & keep.get(d, set()))
                if not len(list(idx)):
                    continue
                idx = list(idx)
                st = float(np.mean([RC_r(Tc[:, k], Td[:, k]) for k in idx]))
                row = {"gid": gid, "phase": info["phase"], "cycle": info["cycle"],
                       "block_label": info["block_label"], "pair": f"{c[-1]}{d[-1]}",
                       "client_a": c, "client_b": d, "s_true": st}
                if (c, gid) in dec and (d, gid) in dec:
                    Dc, Dd = dec[(c, gid)], dec[(d, gid)]
                    row["s_dec"] = float(np.mean([RC_r(Dc[:, k], Dd[:, k])
                                                  for k in idx]))
                if (c, gid) in lat and (d, gid) in lat:
                    a, b = lat[(c, gid)], lat[(d, gid)]
                    row["s_lat"] = float(a @ b / max(np.linalg.norm(a)
                                                     * np.linalg.norm(b), 1e-12))
                rows.append(row)
    return pd.DataFrame(rows)


def similarity_fidelity(sim: pd.DataFrame, by=("phase",)) -> pd.DataFrame:
    """How well each space reproduces the data's similarity structure.

    Per group of `by`, for space in {dec, lat}: Pearson and Spearman of
    (s_space, s_true) over the client pairs, the mean signed error, and the
    share of PAIR ORDERINGS preserved (the fraction of pair couples ranked
    the same way in both) -- ordering is what a dependence step consumes."""
    out = []
    for key, d in sim.groupby(list(by)):
        base = d["s_true"].to_numpy(float)
        for space in ("s_dec", "s_lat"):
            if space not in d or d[space].isna().all():
                continue
            v = d[space].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(base)
            if ok.sum() < 3:
                continue
            a, b = base[ok], v[ok]
            pear = float(np.corrcoef(a, b)[0, 1])
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            spear = float(np.corrcoef(ra, rb)[0, 1])
            conc = np.mean([(a[i] - a[j]) * (b[i] - b[j]) > 0
                            for i in range(len(a)) for j in range(i + 1, len(a))
                            if a[i] != a[j]])
            row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            row.update(space=space[2:], n_pairs=int(ok.sum()), pearson=pear,
                       spearman=spear, concordant=float(conc),
                       mean_err=float(np.mean(b - a)))
            out.append(row)
    return pd.DataFrame(out)


def plot_similarity(ax, sim: pd.DataFrame, space: str = "s_dec",
                    phase: str | None = None):
    """s_space against s_true, one point per (pair, class); the diagonal is
    perfect recovery of the data's similarity."""
    from .figures import colour_map
    d = sim if phase is None else sim[sim["phase"] == phase]
    cm = colour_map(sorted(d["pair"].unique()))
    for p, g in d.groupby("pair"):
        ax.scatter(g["s_true"], g[space], s=12, alpha=0.7, color=cm[p], label=p)
    lo = float(min(d["s_true"].min(), d[space].min()))
    ax.plot([lo, 1], [lo, 1], "k--", lw=0.8)
    ax.set(xlabel="similarity from the true class means",
           ylabel=f"similarity from {space[2:]} prototypes")
    ax.set_title(phase or "all phases", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=3)
    return ax
