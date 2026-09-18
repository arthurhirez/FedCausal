"""Aligned prototypes: what the month prototype averages away, and one that
does not.

Why the month prototype decodes to a line
-----------------------------------------
`extract_prototypes` averages every window labelled month m. Those windows
are shifted views of the same stretch of series, starting at
``t_w = window_start_step * resolution_h``. Averaging views shifted by
``t_w`` multiplies a component of period P by the comb gain

    H(P) = | mean_w exp(2 pi i t_w / P) |

With `a3w84s12` (starts 36 h apart, ~6 per month) the starts alternate
between two times of day 12 h apart, so H(24h) = H(8h) = 0 and only the
12 h harmonic passes (H = 1). The mean latent also sits between the latent
modes those start positions occupy, and the decoder is fed that one vector.

The fix: key the average by where the window starts in a calendar cycle
------------------------------------------------------------------------
    cycle_pos(w) = floor((t_w mod period_h) / bin_h) * bin_h

    p[c, g] = mean of z_w over client c's windows in group g,
    g = (block, cycle_pos)

A block is a run of `block` months inside one world phase (never across
init / transition / final), or the whole phase (`block="phase"`). The
calendar is public and synchronised, so every client computes the same
keys and the server can aggregate per g with the unchanged protocol
(`aggregate_prototypes` accepts any hashable key). The size-weighted mean
of p[c, g] over the cycle positions of a block is the pooled block
prototype -- for `block=1` exactly today's `post_fedavg_full` prototype
(asserted by `check`).

Objects, per (client c, group g), all in model units
-----------------------------------------------------
* ``true``       T[c,g]   mean window over g's windows (the reference)
* ``aligned``    g(p[c,g])                      the keyed prototype, decoded
* ``ceiling``    mean_{w in g} g(z_w)           decode-then-average; NOT a
  federated artifact -- what the latent space carries for g when the
  average is taken after the (nonlinear) decoder. A reference, not a
  strict bound: averaging decoded windows smooths too, so `aligned` can
  exceed it
* ``pooled``     g(p[c,block])                  today's prototype, decoded,
  compared against every g of its block
* ``ref_pooled`` T[c,block]                     today's reference (itself
  comb-filtered)
* ``finch_aligned`` the decoded FINCH centroid p[c,g] falls in, server side
  (`add_server`)

`run.window_rmse` is the model's own per-window quality, before any
averaging: ``window_rmse = sqrt(mean_{w,t,i} (g(z_w) - x_w)^2)``,
``window_amp = median_{w,i} std_t g(z_w)_i / std_t x_wi``,
``window_r = median_{w,i} corr_t(g(z_w)_i, x_wi)``. Read it first: if
single windows already decode flat, no averaging key can help -- the
bottleneck is the AER, not the prototype.

Metrics (per source, client, group, channel i; X = the source's window)
-----------------------------------------------------------------------
``rmse     = sqrt(mean_t (X_ti - T_ti)^2)``  distance to the reference.
``r_shape  = corr_t(X_i, T_i)``  waveform agreement, level-free.
``amp      = std_t(X_i) / std_t(T_i)``  amplitude retention: 0 is a flat
line, 1 the reference's swing. The number that says "it is a line".
``r2_dem   = corr_t(X_i, D_ci)^2`` with D_c = client c's own district
demand cut at g's starts and averaged: does the window look like the
demand that drives it (sign-agnostic, as in `recon`).

``eta2 = sum_k n_k ||zbar_k - zbar||^2 / sum_w ||z_w - zbar||^2`` per
(client, block): share of the block's latent spread explained by
cycle_pos. Under random relabelling with fixed class sizes its
expectation is ``chance = (K-1)/(n-1)``; `lift = eta2 - chance`; `p` is a
seeded permutation p-value (labels shuffled within the block).
High lift -> the latent's dominant within-block factor is calendar
position, and keying by it is the fix. Low lift with a gap between
`aligned` and `ceiling` -> the latent has modes beyond position; cluster
windows client-side instead.

Torch is only touched by `build` (decoding, via `train.decode`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import recon as RC
from .worlds import World

__all__ = ["Alignment", "AlignedRun", "window_table", "build", "check",
           "analyse", "headline", "assign_keys", "loss_by_round",
           "loss_baselines", "decoder_levels", "eta2_by_round",
           "spread_by_round", "encode_windows", "control_run",
           "control_check", "invariance", "invariance_summary",
           "plot_invariance", "sanity", "plot_losses", "calendar_columns",
           "comb_table", "eta2_table", "fidelity", "summary", "coverage",
           "add_server", "phase_arrays", "plot_ladder", "plot_ladder_summary",
           "plot_eta2", "plot_comb", "aligned_browser", "cycle_label",
           "SOURCES"]

SOURCES = ("true", "aligned", "ceiling", "pooled", "ref_pooled")
PERIODS = RC.PERIODS


# --------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Alignment:
    """How windows are grouped before averaging.

    period_h  calendar cycle in hours (24 time-of-day, 168 time-of-week);
              None = no key, every window of a block in one group (today).
    bin_h     width of a cycle_pos class in hours; None = every distinct
              start position is its own class. `period_h=168, bin_h=24`
              is a weekday key.
    block     months pooled per block (1 = month), or "phase" to pool a
              whole init / transition / final stretch.
    """

    period_h: int | None = 24
    bin_h: int | None = None
    block: int | str = 1

    def __post_init__(self):
        if self.block != "phase" and int(self.block) < 1:
            raise ValueError("block must be >= 1 or 'phase'")
        if self.bin_h is not None and self.period_h is None:
            raise ValueError("bin_h needs a period_h")
        if self.bin_h is not None and self.period_h % self.bin_h:
            raise ValueError("bin_h must divide period_h")

    def cycle_pos(self, start_h: np.ndarray) -> np.ndarray:
        t = np.asarray(start_h, float)
        if self.period_h is None:
            return np.zeros(len(t), int)
        pos = np.mod(t, self.period_h)
        if self.bin_h:
            pos = np.floor(pos / self.bin_h) * self.bin_h
        return np.round(pos).astype(int)

    def label(self) -> str:
        p = "none" if self.period_h is None else f"P{self.period_h}"
        b = "" if not self.bin_h else f"b{self.bin_h}"
        return f"{p}{b}_B{self.block}"


def cycle_label(pos: int, period_h: int | None) -> str:
    """Display label of a cycle position. Weekday numbers are ``day mod 7``
    counted from step 0, as `data.window_metadata` counts them."""
    if period_h is None:
        return "all"
    pos = int(pos)
    if period_h > 24:
        return f"d{pos // 24}·{pos % 24:02d}h"
    return f"{pos:02d}h"


def _phase_lo(world: World, phase: str) -> int:
    ph = world.phases()
    return int(ph["init"][1]) if phase == "transition" else int(ph[phase][0])


def _fc(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


# --------------------------------------------------------------------------
# windows -> groups
# --------------------------------------------------------------------------
def assign_keys(world: World, df: pd.DataFrame, align: Alignment
                ) -> pd.DataFrame:
    """Add start_h, phase, block, bid, cycle_pos, gid to any frame with
    `client, start, month` (windows, or a snapshot subset of them)."""
    df = df.copy()
    df["start_h"] = df["start"] * float(world.time["resolution_h"])
    df["phase"] = [world.phase_of(int(m)) for m in df["month"]]
    lo = {p: _phase_lo(world, p) for p in df["phase"].unique()}
    if align.block == "phase":
        df["block"] = 0
    else:
        df["block"] = [(m - lo[p]) // int(align.block)
                       for m, p in zip(df["month"], df["phase"])]
    df["cycle_pos"] = align.cycle_pos(df["start_h"])

    order = {p: i for i, p in enumerate(PERIODS)}
    blocks = (df.groupby(["phase", "block"])["month"].min().reset_index()
              .assign(o=lambda d: d["phase"].map(order))
              .sort_values(["month", "o"]).reset_index(drop=True))
    blocks["bid"] = np.arange(len(blocks))
    df = df.merge(blocks[["phase", "block", "bid"]], on=["phase", "block"])
    groups = (df[["bid", "cycle_pos"]].drop_duplicates()
              .sort_values(["bid", "cycle_pos"]).reset_index(drop=True))
    groups["gid"] = np.arange(len(groups))
    return df.merge(groups, on=["bid", "cycle_pos"])


def window_table(world: World, fl_windows: dict, latents: pd.DataFrame,
                 align: Alignment) -> pd.DataFrame:
    """One row per window: calendar keys + its final-model latent.

    `latents` is `res["latent_trajectories"]` (final global weights, every
    window). It is joined on (client, start step) and asserted to cover
    exactly the windows of `fl_windows` with the same month labels, so the
    windows rebuilt from the spec are provably the ones the run encoded.

    Columns: client, row (index into fl_windows[client]), start (native
    steps), start_h, month, phase, block (index within the phase), bid
    (block id), cycle_pos, gid (group id, same across clients), f0..
    """
    fc = _fc(latents)
    frames = []
    for c in sorted(fl_windows):
        d = fl_windows[c]
        start = np.asarray(d["window_start_step"]).astype(int)
        month = np.asarray(d["labels"]).astype(int)
        frames.append(pd.DataFrame({"client": c, "row": np.arange(len(start)),
                                    "start": start, "month": month}))
    wt = pd.concat(frames, ignore_index=True)

    lt = (latents.rename(columns={"district": "client", "window": "start",
                                  "month": "month_lt"})
          [["client", "start", "month_lt"] + fc])
    if lt.duplicated(["client", "start"]).any():
        raise AssertionError("latent table has duplicate (client, start) rows")
    wt = wt.merge(lt, on=["client", "start"], how="left", validate="1:1")
    if wt["month_lt"].isna().any():
        miss = wt[wt["month_lt"].isna()].groupby("client").size().to_dict()
        raise AssertionError(f"windows without a latent (spec/run mismatch?): {miss}")
    if len(lt) != len(wt):
        raise AssertionError(f"latent table has {len(lt)} rows, windows "
                             f"{len(wt)}: the run encoded a different window set")
    if not (wt["month_lt"].astype(int) == wt["month"]).all():
        raise AssertionError("month labels differ between windows and latents")
    wt = wt.drop(columns="month_lt")

    wt = assign_keys(world, wt, align)
    lead = ["client", "row", "start", "start_h", "month", "phase", "block",
            "bid", "cycle_pos", "gid"]
    return (wt[lead + fc].sort_values(["client", "row"])
            .reset_index(drop=True))


def group_table(wt: pd.DataFrame, align: Alignment) -> pd.DataFrame:
    """gid -> bid, phase, months, cycle_pos, label."""
    g = (wt.groupby("gid")
         .agg(bid=("bid", "first"), phase=("phase", "first"),
              cycle_pos=("cycle_pos", "first"),
              m_lo=("month", "min"), m_hi=("month", "max"))
         .reset_index())
    g["cycle"] = [cycle_label(p, align.period_h) for p in g["cycle_pos"]]
    g["block_label"] = [f"{p} m{a}" if a == b else f"{p} m{a}-{b}"
                        for p, a, b in zip(g["phase"], g["m_lo"], g["m_hi"])]
    return g


def coverage(wt: pd.DataFrame, align: Alignment) -> pd.DataFrame:
    """Windows per (group, client): whether the key leaves anything to
    average. `n_min == 1` means the group prototype IS a single window."""
    n = wt.groupby(["gid", "client"]).size().unstack("client")
    g = group_table(wt, align).set_index("gid")
    out = g[["block_label", "phase", "cycle"]].join(
        pd.DataFrame({"n_min": n.min(axis=1), "n_max": n.max(axis=1),
                      "n_clients": n.notna().sum(axis=1)}))
    return out.reset_index()


def _relabel(fl_windows: dict, wt: pd.DataFrame, key: str) -> dict:
    """fl_windows with `labels` replaced by `wt[key]` (row-aligned), so
    `recon.true_windows` / `demand_windows` group by that key unchanged."""
    out = {}
    for c, d in fl_windows.items():
        sub = wt[wt["client"] == c].sort_values("row")
        if len(sub) != len(d["labels"]):
            raise AssertionError(f"{c}: window table and windows disagree")
        out[c] = {**d, "labels": sub[key].to_numpy()}
    return out


def _means(wt: pd.DataFrame, key: str) -> pd.DataFrame:
    fc = _fc(wt)
    g = wt.groupby(["client", key])
    return (g[fc].mean().join(g.size().rename("n")).reset_index())


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
@dataclass
class AlignedRun:
    align: Alignment
    wt: pd.DataFrame
    groups: pd.DataFrame
    protos: pd.DataFrame                  # client, gid, n, f.. (the artifact)
    pooled: pd.DataFrame                  # client, bid, n, f..
    arrays: dict = field(default_factory=dict)   # source -> {(c, gid): (W, F)}
    demand: dict = field(default_factory=dict)   # (c, gid) -> {district: (W,)}
    window_rmse: pd.DataFrame | None = None
    server: dict = field(default_factory=dict)
    decoded: np.ndarray | None = None     # dec(z_w), row-aligned with `wt`
    agg_h: float = 1.0                    # hours per model step
    agg_k: int = 1                        # native steps per model step
    label: str | None = None              # legend name in window plots

    @property
    def gid_bid(self) -> dict:
        return dict(zip(self.groups["gid"], self.groups["bid"]))


def _decode(model, Z: np.ndarray, batch: int = 512) -> np.ndarray:
    from . import train as T                      # deferred: torch
    Z = np.array(Z, np.float32)                   # writable copy for torch
    if not len(Z):
        return np.zeros((0, model.window_size, model.head.out_features))
    return np.concatenate([T.decode(model, Z[i:i + batch])
                           for i in range(0, len(Z), batch)])


def build(world: World, res: dict, fl_windows: dict, fl: dict, model,
          align: Alignment) -> AlignedRun:
    """Group, average, decode. Everything downstream reads the result."""
    lt = res["latent_trajectories"]
    wt = window_table(world, fl_windows, lt, align)
    groups = group_table(wt, align)
    fc = _fc(wt)
    agg_h = float(fl["preprocessing"]["interval_agg_h"])
    run = AlignedRun(align=align, wt=wt, groups=groups,
                     protos=_means(wt, "gid"), pooled=_means(wt, "bid"),
                     agg_h=agg_h,
                     agg_k=int(round(agg_h / float(world.time["resolution_h"]))))
    gb = run.gid_bid

    # references
    fw_g = _relabel(fl_windows, wt, "gid")
    true = RC.true_windows(fw_g)
    true_b = RC.true_windows(_relabel(fl_windows, wt, "bid"))
    run.demand = RC.demand_windows(world, fw_g, fl)

    # decoded prototypes
    dec = _decode(model, run.protos[fc].to_numpy())
    aligned = {(c, int(g)): dec[i] for i, (c, g)
               in enumerate(zip(run.protos["client"], run.protos["gid"]))}
    decb = _decode(model, run.pooled[fc].to_numpy())
    pooled_b = {(c, int(b)): decb[i] for i, (c, b)
                in enumerate(zip(run.pooled["client"], run.pooled["bid"]))}

    # decode-then-average, and the model's own per-window error
    decw = _decode(model, wt[fc].to_numpy())
    run.decoded = decw
    ceiling, err = {}, []
    for (c, g), idx in wt.groupby(["client", "gid"]).indices.items():
        ceiling[(c, int(g))] = decw[idx].mean(axis=0)
        X = np.asarray(fl_windows[c]["windows"])[wt["row"].to_numpy()[idx]]
        R = decw[idx]
        sx, sr = X.std(axis=1), R.std(axis=1)             # (n, F)
        ok = sx > 1e-12
        err.append({"client": c, "gid": int(g), "n": len(idx),
                    "window_rmse": float(np.sqrt(np.mean((R - X) ** 2))),
                    "window_amp": float(np.median(sr[ok] / sx[ok])),
                    "window_r": float(np.median([
                        RC._r(R[w, :, i], X[w, :, i])
                        for w in range(len(idx)) for i in range(X.shape[2])
                        if ok[w, i]]))})
    run.window_rmse = pd.DataFrame(err).merge(
        groups[["gid", "phase", "cycle"]], on="gid")

    keys = list(true)
    run.arrays = {
        "true": true,
        "aligned": aligned,
        "ceiling": ceiling,
        "pooled": {k: pooled_b[(k[0], gb[k[1]])] for k in keys},
        "ref_pooled": {k: true_b[(k[0], gb[k[1]])] for k in keys},
    }
    return run


def check(run: AlignedRun, res: dict, atol: float = 1e-5) -> pd.DataFrame:
    """Contract checks. The pooled prototype must be the size-weighted mean
    of its aligned prototypes, and with `block=1` it must equal the run's
    own `post_fedavg_full` prototype."""
    fc = _fc(run.protos)
    rows = []
    p = run.protos.merge(run.groups[["gid", "bid"]], on="gid")
    w = p[fc].to_numpy() * p[["n"]].to_numpy()
    rec = (pd.DataFrame(w, columns=fc).assign(client=p["client"], bid=p["bid"],
                                               n=p["n"])
           .groupby(["client", "bid"]).sum())
    rec = rec[fc].div(rec["n"], axis=0)
    ref = run.pooled.set_index(["client", "bid"])[fc].loc[rec.index]
    d = float(np.abs(rec.to_numpy() - ref.to_numpy()).max())
    rows.append({"check": "pooled == weighted mean of aligned", "value": d,
                 "passed": d < atol})

    n_win = len(run.wt)
    rows.append({"check": "every window keyed", "value": n_win,
                 "passed": n_win == len(res["latent_trajectories"])})

    if run.align.block == 1:
        pr = res["prototypes"]
        pf = pr[pr["scope"] == "post_fedavg_full"]
        bm = (run.wt.groupby("bid")["month"].first())
        q = run.pooled.assign(month=run.pooled["bid"].map(bm))
        m = q.merge(pf, on=["client", "month"], suffixes=("", "_r"))
        d = float(np.abs(m[fc].to_numpy()
                         - m[[f + "_r" for f in fc]].to_numpy()).max()) \
            if len(m) else np.nan
        rows.append({"check": "block=1 pooled == post_fedavg_full",
                     "value": d, "passed": len(m) == len(q) and d < atol})

    k = run.groups.groupby("bid")["gid"].nunique()
    rows.append({"check": "cycle positions per block (min..max)",
                 "value": f"{k.min()}..{k.max()}", "passed": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def comb_table(wt: pd.DataFrame, periods=(168, 84, 24, 12, 8)) -> pd.DataFrame:
    """H(P) = |mean_w exp(2 pi i t_w / P)| over a set of window starts.

    `pooled`: the set a block prototype averages. `aligned`: the set a
    group prototype averages, size-weighted over the block's groups.
    Rows per (client, bid, scope, period)."""
    rows = []
    for (c, b), blk in wt.groupby(["client", "bid"]):
        t = blk["start_h"].to_numpy(float)
        for P in periods:
            e = np.exp(2j * np.pi * t / P)
            gp = abs(e.mean())
            ga = sum(abs(e[idx].mean()) * len(idx) for idx in
                     blk.groupby("gid").indices.values()) / len(t)
            base = {"client": c, "bid": int(b), "phase": blk["phase"].iloc[0],
                    "period_h": P, "n": len(t)}
            rows += [{**base, "scope": "pooled", "gain": gp},
                     {**base, "scope": "aligned", "gain": ga}]
    return pd.DataFrame(rows)


def _eta2(Z: np.ndarray, lab: np.ndarray) -> float:
    zc = Z - Z.mean(axis=0)
    sst = float((zc ** 2).sum())
    if sst <= 0:
        return np.nan
    ssb = 0.0
    for k in np.unique(lab):
        m = lab == k
        ssb += m.sum() * float((zc[m].mean(axis=0) ** 2).sum())
    return ssb / sst


def eta2_table(wt: pd.DataFrame, n_perm: int = 199, seed: int = 0
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (client, block): eta2, chance, lift, permutation p.

    Second frame: the pooled test per (client, phase) and overall -- the
    mean eta2 over blocks against the same statistic with labels shuffled
    within every block at once (one shuffle index shared by all blocks),
    so power accumulates across months instead of staying at one block's
    p-floor. Blocks with n <= K (nothing to separate) are skipped."""
    fc = _fc(wt)
    rng = np.random.default_rng(seed)
    rows, nulls = [], []
    for (c, b), blk in wt.groupby(["client", "bid"]):
        lab = blk["cycle_pos"].to_numpy()
        K, n = len(np.unique(lab)), len(lab)
        if K < 2 or n <= K:
            continue
        Z = blk[fc].to_numpy(float)
        obs = _eta2(Z, lab)
        null = np.array([_eta2(Z, rng.permutation(lab)) for _ in range(n_perm)])
        rows.append({"client": c, "bid": int(b), "phase": blk["phase"].iloc[0],
                     "n": n, "K": K, "eta2": obs,
                     "chance": (K - 1) / (n - 1),
                     "lift": obs - (K - 1) / (n - 1),
                     "p": (1 + int((null >= obs - 1e-12).sum())) / (n_perm + 1)})
        nulls.append(null)
    per = pd.DataFrame(rows)
    if not len(per):
        return per, pd.DataFrame()
    N = np.vstack(nulls)                                   # blocks x perms
    pooled = []
    sets = ([(("all", "all"), per.index)]
            + [(("all", p), g.index) for p, g in per.groupby("phase")]
            + [((c, p), g.index) for (c, p), g in per.groupby(["client", "phase"])])
    for (client, phase), idx in sets:
        obs = per.loc[idx, "eta2"].mean()
        null = N[idx].mean(axis=0)
        pooled.append({"client": client, "phase": phase, "blocks": len(idx),
                       "eta2": obs, "chance": per.loc[idx, "chance"].mean(),
                       "lift": obs - per.loc[idx, "chance"].mean(),
                       "p": (1 + int((null >= obs - 1e-12).sum())) / (n_perm + 1)})
    return per, pd.DataFrame(pooled)


def fidelity(run: AlignedRun, sources=None) -> pd.DataFrame:
    """rmse / r_shape / amp / r2_dem per (source, client, gid, channel)."""
    sources = sources or ([s for s in SOURCES if s in run.arrays]
                          + [s for s in run.arrays if s not in SOURCES])
    g = run.groups.set_index("gid")
    true = run.arrays["true"]
    rows = []
    for s in sources:
        for (c, gid), X in run.arrays[s].items():
            T = true.get((c, gid))
            if T is None:
                continue
            dm = run.demand.get((c, gid))
            own = None if dm is None else dm.get(c)
            info = g.loc[gid]
            for i in range(T.shape[1]):
                x, t = X[:, i], T[:, i]
                st = t.std()
                rows.append({
                    "source": s, "client": c, "gid": int(gid),
                    "bid": int(info["bid"]), "phase": info["phase"],
                    "cycle": info["cycle"], "channel": i,
                    "rmse": float(np.sqrt(np.mean((x - t) ** 2))),
                    "r_shape": RC._r(x, t),
                    "amp": float(x.std() / st) if st > 1e-12 else np.nan,
                    "r2_dem": np.nan if own is None else RC._r(x, own) ** 2})
    return pd.DataFrame(rows)


def summary(fid: pd.DataFrame, metrics=("rmse", "r_shape", "amp", "r2_dem")
            ) -> pd.DataFrame:
    """Mean per (source, phase); phases in init/transition/final order."""
    out = (fid.groupby(["source", "phase"])[list(metrics)].mean()
           .join(fid.groupby(["source", "phase"]).size().rename("n"))
           .reset_index())
    out["phase"] = pd.Categorical(out["phase"], PERIODS, ordered=True)
    order = {s: i for i, s in enumerate(fid["source"].drop_duplicates())}
    return (out.assign(o=out["source"].map(order))
            .sort_values(["o", "phase"]).drop(columns="o")
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------
def add_server(run: AlignedRun, model, seed: int = 0) -> AlignedRun:
    """FINCH per group key with the upstream aggregator, unchanged: the key
    is the group id instead of the month. Adds `finch_aligned` to
    `run.arrays` and `run.server = {centroids, membership}`. Membership is
    recovered as the nearest centroid by cosine (the aggregator returns
    centroids only), as in `recon.finch_membership`."""
    from . import bridge
    fc = _fc(run.protos)
    local = {}
    for r in run.protos.itertuples(index=False):
        local.setdefault(r.client, {})[int(r.gid)] = np.asarray(
            [getattr(r, f) for f in fc], float)
    agg = bridge.aggregate_prototypes(local, seed)

    cents, mem = [], []
    for gid, (clusters, _) in agg.items():
        C = np.asarray(clusters, float)
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
        for j, v in enumerate(C):
            cents.append({"gid": int(gid), "cluster": j,
                          **{f: v[i] for i, f in enumerate(fc)}})
        for c, protos in local.items():
            if gid not in protos:
                continue
            v = protos[gid] / max(np.linalg.norm(protos[gid]), 1e-12)
            s = Cn @ v
            j = int(np.argmax(s))
            mem.append({"gid": int(gid), "client": c, "cluster": j,
                        "cos": float(s[j]), "n_clusters": len(C)})
    cents, mem = pd.DataFrame(cents), pd.DataFrame(mem)
    mem["n_members"] = mem.groupby(["gid", "cluster"])["client"].transform("count")
    dec = _decode(model, cents[fc].to_numpy())
    lut = {(int(g), int(k)): dec[i]
           for i, (g, k) in enumerate(zip(cents["gid"], cents["cluster"]))}
    run.arrays["finch_aligned"] = {(r.client, int(r.gid)): lut[(int(r.gid),
                                                                int(r.cluster))]
                                   for r in mem.itertuples(index=False)}
    run.server = {"centroids": cents, "membership": mem.merge(
        run.groups[["gid", "phase", "cycle", "block_label"]], on="gid")}
    return run


# --------------------------------------------------------------------------
# period view
# --------------------------------------------------------------------------
def phase_arrays(run: AlignedRun, phase: str) -> tuple[dict, dict]:
    """({source: {(client, cycle_pos): mean over the phase's groups}},
    {(client, cycle_pos): {district: mean demand}}) -- the init / final view
    at each calendar position."""
    g = run.groups[run.groups["phase"] == phase].set_index("gid")["cycle_pos"]
    arr: dict = {}
    for s, d in run.arrays.items():
        acc: dict = {}
        for (c, gid), a in d.items():
            if gid in g.index:
                acc.setdefault((c, int(g[gid])), []).append(a)
        arr[s] = {k: np.mean(v, axis=0) for k, v in acc.items()}
    dem: dict = {}
    for (c, gid), dm in run.demand.items():
        if gid in g.index:
            for k, v in dm.items():
                dem.setdefault((c, int(g[gid])), {}).setdefault(k, []).append(v)
    dem = {key: {k: np.mean(v, axis=0) for k, v in d.items()}
           for key, d in dem.items()}
    return arr, dem


# --------------------------------------------------------------------------
# model sanity: is the AER reconstructing at all?
# --------------------------------------------------------------------------
def _aer_loss_np(pred: np.ndarray, X: np.ndarray, r: float) -> float:
    """`aer.aer_loss` on arrays; `pred`, `X` are full (n, W, F) windows."""
    def m(a, b):
        return float(np.mean((a - b) ** 2))
    return ((r / 2) * m(pred[:, 0], X[:, 0])
            + (1 - r) * m(pred[:, 1:-1], X[:, 1:-1])
            + (r / 2) * m(pred[:, -1], X[:, -1]))


def loss_by_round(res: dict) -> pd.DataFrame:
    """`training_log` per round, mean over clients and local epochs.

    `d_mse` = relative change of `loss_mse` from the previous round;
    `aux_share` = 1 - loss_mse / loss: the share of the objective that is
    NOT reconstruction (prototype term and any lab terms, weighted as
    trained; the raw `loss_proto` column is unweighted)."""
    log = res.get("training_log")
    if log is None or not len(log):
        return pd.DataFrame()
    t = log.groupby("round")[["loss", "loss_mse", "loss_proto"]].mean()
    t["d_mse"] = t["loss_mse"].pct_change()
    t["aux_share"] = 1.0 - t["loss_mse"] / t["loss"]
    return t.reset_index()


def loss_baselines(run: AlignedRun, fl_windows: dict, fl: dict) -> pd.DataFrame:
    """The final model's AER loss on every window, against two flat
    predictors scored with the same weighting (`reg_ratio`).

    base_const  each channel's client-wide mean, at every step: needs no
                latent at all.
    base_wmean  each window's own input mean (steps 1..W-2), at every step,
                including ry / fy: the best flat line a latent could encode.
    final / base_wmean ~ 1 means the model tracks window level and nothing
    of the within-window shape."""
    r = float(fl["model"]["reg_ratio"])
    rows = []
    for c in sorted(fl_windows):
        mask = (run.wt["client"] == c).to_numpy()
        X = np.asarray(fl_windows[c]["windows"], float)[
            run.wt.loc[mask, "row"].to_numpy()]
        D = run.decoded[mask]
        const = np.broadcast_to(X[:, 1:-1].mean(axis=(0, 1)), X.shape)
        wmean = np.broadcast_to(X[:, 1:-1].mean(axis=1, keepdims=True), X.shape)
        final = _aer_loss_np(D, X, r)
        bc, bw = _aer_loss_np(const, X, r), _aer_loss_np(wmean, X, r)
        rows.append({"client": c, "n": len(X), "loss_final": final,
                     "base_const": bc, "base_wmean": bw,
                     "final_over_const": final / bc,
                     "final_over_wmean": final / bw})
    return pd.DataFrame(rows)


def decoder_levels(run: AlignedRun, fl_windows: dict) -> pd.DataFrame:
    """Does the decoder follow window LEVEL, window SHAPE, both or neither?

    Per (client, phase, channel), over windows w:
    level_r      corr_w(mean_t dec_w, mean_t x_w)
    level_ratio  std_w(mean_t dec_w) / std_w(mean_t x_w)
    shape_amp    median_w std_t(dec_w) / std_t(x_w)
    level_r ~ 1 with shape_amp ~ 0 is a level-only decoder."""
    rows = []
    for c in sorted(fl_windows):
        mask = (run.wt["client"] == c).to_numpy()
        sub = run.wt.loc[mask]
        X = np.asarray(fl_windows[c]["windows"], float)[sub["row"].to_numpy()]
        D = run.decoded[mask]
        for ph, idx in sub.reset_index(drop=True).groupby("phase").indices.items():
            x, d = X[idx], D[idx]
            for i in range(X.shape[2]):
                lx, ld = x[:, :, i].mean(axis=1), d[:, :, i].mean(axis=1)
                sx = x[:, :, i].std(axis=1)
                ok = sx > 1e-12
                rows.append({
                    "client": c, "phase": ph, "channel": i, "n": len(idx),
                    "level_r": RC._r(ld, lx) if len(idx) > 2 else np.nan,
                    "level_ratio": float(ld.std() / lx.std())
                    if lx.std() > 1e-12 else np.nan,
                    "shape_amp": float(np.median(
                        d[ok, :, i].std(axis=1) / sx[ok])) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def _snapshot_slices(world: World, res: dict, align: Alignment,
                     run: AlignedRun | None = None):
    """[(round, stage, keyed frame)] from `latents_by_round`, plus the
    final all-window latents as round "final_all" when `run` is given."""
    snaps = res.get("snapshots")
    if snaps is None:
        snaps = res.get("latents_by_round")
    out = []
    if snaps is not None and len(snaps):
        fc = _fc(snaps)
        base = snaps.rename(columns={"district": "client", "window": "start"})
        for (r, st), d in base.groupby(["round", "stage"]):
            d = d[["client", "start", "month"] + fc].reset_index(drop=True)
            out.append((int(r), st, assign_keys(world, d, align)))
    if run is not None:
        out.append(("final_all", "post", run.wt))
    return out


def eta2_by_round(world: World, res: dict, align: Alignment,
                  run: AlignedRun | None = None, n_perm: int = 99,
                  seed: int = 0) -> pd.DataFrame:
    """The pooled eta2 test (client = all) per (round, stage, phase).

    Round -1 is the untrained initialisation, shared by every client: the
    control. Snapshot rounds use the month-stratified subset (~5 windows per
    month), so their chance level is higher than the all-window row's."""
    rows = []
    for r, st, d in _snapshot_slices(world, res, align, run):
        _, pooled = eta2_table(d, n_perm=n_perm, seed=seed)
        if not len(pooled):
            continue
        pooled = pooled[pooled["client"] == "all"]
        rows.append(pooled.assign(round=str(r), stage=st))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["round", "stage", "phase", "blocks", "eta2", "chance",
                "lift", "p"]]


def spread_by_round(world: World, res: dict, align: Alignment,
                    run: AlignedRun | None = None) -> pd.DataFrame:
    """Absolute latent spread per (round, stage), RMS per dimension:

    norm      sqrt(mean_w ||z_w||^2 / d)
    s_client  sqrt(mean_w ||z_w - zbar_client||^2 / d)
    s_block   same around the (client, block) mean   -- what a block
              prototype averages over
    s_group   same around the (client, group) mean   -- what is left after
              the calendar key
    rel_block = s_block / norm. Shrinking over rounds while `norm` holds
    means training compresses within-month spread (the prototype term's
    MSE-to-mean pull does exactly that)."""
    rows = []
    for r, st, d in _snapshot_slices(world, res, align, run):
        fc = _fc(d)
        Z = d[fc].to_numpy(float)
        dim = Z.shape[1]

        def rms_around(keys):
            mu = d.groupby(keys)[fc].transform("mean").to_numpy(float)
            return float(np.sqrt(((Z - mu) ** 2).sum(axis=1).mean() / dim))

        norm = float(np.sqrt((Z ** 2).sum(axis=1).mean() / dim))
        sb = rms_around(["client", "bid"])
        rows.append({"round": str(r), "stage": st, "n": len(d), "norm": norm,
                     "s_client": rms_around(["client"]), "s_block": sb,
                     "s_group": rms_around(["client", "gid"]),
                     "rel_block": sb / norm if norm > 0 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# controls: the untrained encoder on every window
# --------------------------------------------------------------------------
def calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """weekday (day mod 7 from step 0, the demand model's own convention)
    and hour of each window start, from `start_h`."""
    out = df.copy()
    out["weekday"] = (out["start_h"] // 24 % 7).astype(int)
    out["hour"] = (out["start_h"] % 24).astype(int)
    return out


def encode_windows(model, fl_windows: dict, batch: int = 512) -> pd.DataFrame:
    """Every window through `model.encode`, in the `latent_trajectories`
    schema (district, kind, window, month, f0..)."""
    import torch
    from .train import model_device
    dev = model_device(model)
    frames = []
    model.eval()
    with torch.no_grad():
        for c in sorted(fl_windows):
            d = fl_windows[c]
            X = torch.as_tensor(np.array(d["windows"], np.float32), device=dev)
            z = torch.cat([model.encode(X[i:i + batch, 1:-1])
                           for i in range(0, len(X), batch)]).cpu().numpy()
            df = pd.DataFrame(z, columns=[f"f{i}" for i in range(z.shape[1])])
            df.insert(0, "month", np.asarray(d["labels"]))
            df.insert(0, "window", np.asarray(d["window_start_step"]))
            df.insert(0, "kind", "aer_latent")
            df.insert(0, "district", c)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def control_run(world: World, res: dict, fl_windows: dict, fl: dict,
                align: Alignment) -> AlignedRun:
    """`build` with the round -1 model (the shared untrained init) in place
    of the trained one: the same tables, for a random encoder/decoder."""
    from . import train as T
    m0 = T.untrained_model(res)
    fake = {"latent_trajectories": encode_windows(m0, fl_windows)}
    return build(world, fake, fl_windows, fl, m0, align)


def control_check(res: dict, ctrl: AlignedRun, atol: float = 1e-5
                  ) -> pd.DataFrame:
    """The rebuilt untrained model must reproduce the saved round -1
    snapshot latents (same windows, same values)."""
    snaps = res.get("snapshots")
    if snaps is None:
        snaps = res.get("latents_by_round")
    if snaps is None or not len(snaps) or -1 not in set(snaps["round"]):
        return pd.DataFrame([{"check": "untrained == round -1 snapshot",
                              "value": "no round -1 snapshot",
                              "passed": True}])
    s = snaps[snaps["round"] == -1].rename(columns={"district": "client",
                                                    "window": "start"})
    fc = _fc(s)
    m = s[["client", "start"] + fc].merge(
        ctrl.wt[["client", "start"] + fc], on=["client", "start"],
        suffixes=("", "_c"))
    d = float(np.abs(m[fc].to_numpy()
                     - m[[f + "_c" for f in fc]].to_numpy()).max())
    return pd.DataFrame([{"check": "untrained == round -1 snapshot",
                          "value": d, "passed": len(m) == len(s) and d < atol}])


# --------------------------------------------------------------------------
# invariance across blocks, per calendar class
# --------------------------------------------------------------------------
def invariance(world: World, wt: pd.DataFrame, fl_windows: dict, model=None,
               channels: pd.DataFrame | None = None, block: int = 6,
               period_h: int = 168, bin_h: int | None = None,
               ref_phase: str = "init") -> pd.DataFrame:
    """Does a calendar class look the same in every block as in the
    reference phase?

    Class g = start position mod `period_h` (168: weekday x hour). For
    client c, block b, class g, channel i:

    T[b,g]  mean true window; T[ref,g] the same over `ref_phase`, LEAVING
            OUT block b's own months when b lies inside `ref_phase` (else
            the comparison would include the block itself);
    z[b,g]  mean latent; dec = decoded mean latent (if `model`).

    r_true   = corr_t(T[b,g,i], T[ref,g,i])      shape kept by the data
    amp_true = std_t T[b,g,i] / std_t T[ref,g,i]
    lvl_true = mean_t T[b,g,i] - mean_t T[ref,g,i]   (MinMax is fitted on the
               reference months and fixed, so level is comparable in-client)
    d_lat    = 1 - cos(z[b,g], z[ref,g])         representation change
    e_lat    = ||z[b,g] - z[ref,g]|| / ||z[ref,g]||  the same, Euclidean
               (latents share a large common direction, so d_lat is small in
               absolute terms; compare it across clients / against control)
    r_dec    = corr_t(dec[b,g,i], dec[ref,g,i])   decoded change
    r_fid    = corr_t(dec[b,g,i], T[b,g,i])       decoded vs its own truth

    Expected on a drift world: pure channels of non-drifting clients stay
    near r_true = 1; the drifting client, and mixed channels listening to
    it, depart."""
    al = Alignment(period_h, bin_h, block)
    fc = _fc(wt)
    d = assign_keys(world, wt[["client", "row", "start", "month"] + fc], al)
    slot = ({} if channels is None else
            {(r.client, int(r.channel)): r.slot_class
             for r in channels.itertuples(index=False)})
    bl = d.groupby("bid").agg(phase=("phase", "first"),
                              lo=("month", "min"), hi=("month", "max"))
    labels = pd.Series([f"{p} m{a}" if a == b else f"{p} m{a}-{b}"
                        for p, a, b in zip(bl["phase"], bl["lo"], bl["hi"])],
                       index=bl.index)

    cells, Z = [], []
    for c, dc in d.groupby("client"):
        X = np.asarray(fl_windows[c]["windows"], float)
        ref_all = dc[dc["phase"] == ref_phase]
        for (b, g), blk in dc.groupby(["bid", "cycle_pos"]):
            ref = ref_all[(ref_all["cycle_pos"] == g)
                          & ~ref_all["month"].isin(blk["month"].unique())]
            if not len(ref):
                continue
            cells.append({"client": c, "bid": int(b), "cycle_pos": int(g),
                          "phase": blk["phase"].iloc[0],
                          "n": len(blk), "n_ref": len(ref),
                          "T_b": X[blk["row"].to_numpy()].mean(axis=0),
                          "T_r": X[ref["row"].to_numpy()].mean(axis=0)})
            Z += [blk[fc].to_numpy(float).mean(axis=0),
                  ref[fc].to_numpy(float).mean(axis=0)]
    if not cells:
        return pd.DataFrame()
    Z = np.asarray(Z)
    dec = _decode(model, Z) if model is not None else None

    def r(a, b):
        return RC._r(a, b)

    rows = []
    for k, cell in enumerate(cells):
        zb, zr = Z[2 * k], Z[2 * k + 1]
        nr = max(float(np.linalg.norm(zr)), 1e-12)
        cos = float(zb @ zr / max(float(np.linalg.norm(zb)) * nr, 1e-12))
        e_lat = float(np.linalg.norm(zb - zr)) / nr
        Tb, Tr = cell["T_b"], cell["T_r"]
        for i in range(Tb.shape[1]):
            sr = Tr[:, i].std()
            row = {k2: cell[k2] for k2 in ("client", "bid", "cycle_pos",
                                            "phase", "n", "n_ref")}
            row.update(channel=i,
                       slot_class=slot.get((cell["client"], i), "all"),
                       r_true=r(Tb[:, i], Tr[:, i]),
                       amp_true=float(Tb[:, i].std() / sr) if sr > 1e-12
                       else np.nan,
                       lvl_true=float(Tb[:, i].mean() - Tr[:, i].mean()),
                       d_lat=1.0 - cos, e_lat=e_lat)
            if dec is not None:
                Db, Dr = dec[2 * k][:, i], dec[2 * k + 1][:, i]
                row.update(r_dec=r(Db, Dr), r_fid=r(Db, Tb[:, i]))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["block_label"] = out["bid"].map(labels)
    out["cycle"] = [cycle_label(p, period_h) for p in out["cycle_pos"]]
    return out


def invariance_summary(inv: pd.DataFrame, metrics=("r_true", "lvl_true",
                                                   "d_lat", "e_lat", "r_dec",
                                                   "r_fid")
                       ) -> pd.DataFrame:
    """Mean per (client, slot_class, block) over classes and channels."""
    metrics = [m for m in metrics if m in inv.columns]
    g = inv.groupby(["client", "slot_class", "bid", "block_label", "phase"])
    return (g[metrics].mean().join(g["n"].mean().rename("n_mean"))
            .reset_index().sort_values(["client", "slot_class", "bid"]))


def plot_invariance(ax, summ: pd.DataFrame, metric: str, slot_class: str,
                    vmin=None, vmax=None, cmap="RdBu_r"):
    """Heatmap client x block of `metric` for one slot class."""
    d = summ[summ["slot_class"] == slot_class]
    piv = d.pivot_table(index="client", columns="bid", values=metric)
    lab = d.drop_duplicates("bid").set_index("bid")["block_label"]
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([lab[b] for b in piv.columns], rotation=40, ha="right",
                       fontsize=7)
    ax.set_title(f"{metric} · {slot_class}", fontsize=9, loc="left")
    return im


# --------------------------------------------------------------------------
# one sanity row per trained cell
# --------------------------------------------------------------------------
def sanity(world: World, out: dict, n_perm: int = 99) -> dict:
    """Every §8 check plus the untrained control, for one `analyse` output.
    -> {"row": Series, "loss": ..., "baselines": ..., "levels": ...,
        "eta": ..., "eta_ctrl": ..., "spread": ..., "control": AlignedRun}"""
    run, res = out["run"], out["res"]
    ctrl = control_run(world, res, out["fw"], out["fl"], out["align"])
    loss = loss_by_round(res)
    base = loss_baselines(run, out["fw"], out["fl"])
    lev = decoder_levels(run, out["fw"])
    eta = eta2_by_round(world, res, out["align"], run=run, n_perm=n_perm)
    _, eta_c = eta2_table(ctrl.wt, n_perm=n_perm)
    spr = spread_by_round(world, res, out["align"], run=run)
    wq = run.window_rmse

    def pick(t, rnd):
        x = t[(t["round"] == rnd) & (t["phase"] == "all")]
        return float(x["lift"].iloc[0]) if len(x) else np.nan

    row = {
        "label": out["spec"].label(),
        "run_key": res.get("run_key"),
        "schedule": (out["spec"].schedule.label()
                     if out["spec"].schedule is not None else "all"),
        "steps/client": np.nan,
        "loss_mse_last": float(loss["loss_mse"].iloc[-1]) if len(loss) else np.nan,
        "d_mse_last": float(loss["d_mse"].iloc[-1]) if len(loss) else np.nan,
        "aux_share": float(loss["aux_share"].iloc[-1]) if len(loss) else np.nan,
        "final_over_wmean": float(base["final_over_wmean"].mean()),
        "final_over_const": float(base["final_over_const"].mean()),
        "window_amp": float(wq["window_amp"].mean()),
        "window_r": float(wq["window_r"].mean()),
        "level_r": float(lev["level_r"].mean()),
        "eta_lift_final": pick(eta, "final_all"),
        "eta_lift_ctrl": (float(eta_c.loc[(eta_c["client"] == "all")
                                          & (eta_c["phase"] == "all"),
                                          "lift"].iloc[0])
                          if len(eta_c) else np.nan),
        "rel_block_first": float(spr["rel_block"].iloc[0]) if len(spr) else np.nan,
        "rel_block_last": float(spr["rel_block"].iloc[-1]) if len(spr) else np.nan,
        "ctrl_window_amp": float(ctrl.window_rmse["window_amp"].mean()),
    }
    from .train import fl_mismatches
    tr = out["fl"]["training"]
    n_win = np.mean([len(d["windows"]) for d in out["fw"].values()])
    rounds_done = int(res.get("meta", {}).get("rounds", tr["rounds"]))
    n_train = res.get("meta", {}).get("train_windows", n_win)
    row["steps/client"] = int(rounds_done * tr["local_epochs"]
                              * np.ceil(n_train / tr["batch_size"]))
    # what actually trained (saved fl + log) must be what the spec says
    saved_fl = res.get("fl") or out["fl"]
    bad = fl_mismatches(out["spec"], saved_fl)
    if bad:
        raise AssertionError(f"{row['label']}: the saved run trained with "
                             f"{bad} -- purge it (store.audit / purge)")
    # a schedule (streaming) numbers rounds cumulatively, so the contract is
    # against what the run RECORDS having trained, not the per-block setting
    want = int(res.get("meta", {}).get("rounds", out["spec"].training.rounds))
    if len(loss) and len(loss) != want:
        raise AssertionError(f"{row['label']}: training_log has {len(loss)} "
                             f"rounds, meta says {want}")
    return {"row": pd.Series(row), "loss": loss, "baselines": base,
            "levels": lev, "eta": eta, "eta_ctrl": eta_c, "spread": spr,
            "control": ctrl, "control_check": control_check(res, ctrl)}


def plot_losses(axes, logs: dict):
    """`logs` = {cell name: training_log}. Panel 0: loss_mse per round;
    panel 1: every non-zero term of the objective per round (mean over
    clients and epochs), one line style per term."""
    from .figures import PALETTE
    styles = {"loss_proto": "--", "loss_diff": ":", "loss_spec": "-.",
              "loss_var": (0, (1, 3))}
    for j, (name, log) in enumerate(logs.items()):
        col = PALETTE[j % len(PALETTE)]
        t = log.groupby("round").mean(numeric_only=True)
        axes[0].plot(t.index, t["loss_mse"], color=col, label=name)
        for k, ls in styles.items():
            if k in t and t[k].abs().sum() > 0:
                axes[1].plot(t.index, t[k], color=col, ls=ls,
                             label=f"{name} · {k[5:]}")
    axes[0].set(xlabel="round", ylabel="loss_mse (reconstruction)")
    axes[1].set(xlabel="round", ylabel="other terms")
    for a in axes:
        a.legend(fontsize=7)
    return axes


# --------------------------------------------------------------------------
# one call per (spec, alignment)
# --------------------------------------------------------------------------
def analyse(world: World, spec, align: Alignment, store=None, cache=None,
            server: bool = True, n_perm: int = 199, overwrite: bool = False
            ) -> dict:
    """run_cell (a cache read when the key exists) -> windows -> model ->
    `build` -> every table above. A spec whose geometry was never trained
    TRAINS here."""
    from . import data as D, train as T
    from .runner import run_cell
    cache = D.CACHE if cache is None else cache
    res = run_cell(world, spec, store=store, cache=cache, overwrite=overwrite)
    res.setdefault("snapshots", res.get("latents_by_round"))
    fw, _, fl = D.build_windows(world, spec, cache=cache)
    model = T.reconstruction_model(res, world, store=store)
    run = build(world, res, fw, fl, model, align)
    if server:
        add_server(run, model)
    per, pooled = eta2_table(run.wt, n_perm=n_perm)
    fid = fidelity(run)
    return {"spec": spec, "align": align, "res": res, "fw": fw, "fl": fl,
            "model": model, "run": run, "checks": check(run, res),
            "coverage": coverage(run.wt, align), "comb": comb_table(run.wt),
            "eta": per, "eta_pooled": pooled, "fid": fid,
            "summary": summary(fid)}


def headline(out: dict, phase: str | None = None) -> pd.Series:
    """One row per analysis, for comparing geometries / keys side by side.
    `phase` restricts the fidelity and window-quality means (None = all)."""
    run, fid, comb = out["run"], out["fid"], out["comb"]
    if phase is not None:
        fid = fid[fid["phase"] == phase]
        comb = comb[comb["phase"] == phase]
    wq = run.window_rmse if phase is None else \
        run.window_rmse[run.window_rmse["phase"] == phase]
    m = fid.groupby("source")[["amp", "r_shape", "r2_dem"]].mean()
    cg = comb.groupby(["scope", "period_h"])["gain"].median()
    ep = out["eta_pooled"]
    ep = ep[(ep["client"] == "all") & (ep["phase"] == (phase or "all"))] \
        if len(ep) else ep
    geo = getattr(out["spec"], "geometry", None)
    row = {"geometry": geo.label() if geo is not None else "?",
           "align": out["align"].label(),
           "win/group (median)": float(out["coverage"]["n_min"].median()),
           "window_amp": float(wq["window_amp"].mean()),
           "window_r": float(wq["window_r"].mean()),
           "eta2_lift": float(ep["lift"].iloc[0]) if len(ep) else np.nan,
           "eta2_p": float(ep["p"].iloc[0]) if len(ep) else np.nan}
    for P in (24, 168):
        for sc in ("pooled", "aligned"):
            row[f"H{P}_{sc}"] = float(cg.get((sc, P), np.nan))
    for met in ("amp", "r_shape", "r2_dem"):
        for s in ("pooled", "aligned", "ceiling", "true"):
            if s in m.index and not (s == "true" and met != "r2_dem"):
                row[f"{met}_{s}"] = float(m.loc[s, met])
    return pd.Series(row)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
STYLE = {
    "true": dict(color="k", lw=2.0, label="true (aligned mean window)"),
    "aligned": dict(color="#DD8452", lw=1.6, label="decoded aligned prototype"),
    "ceiling": dict(color="#55A868", lw=1.2, ls="--",
                    label="decode-then-average (ceiling)"),
    "pooled": dict(color="#4C72B0", lw=1.4, label="decoded month/block prototype"),
    "ref_pooled": dict(color="#8C8C8C", lw=1.0, ls="-.",
                       label="true pooled window (today's reference)"),
    "finch_aligned": dict(color="#8172B3", lw=1.2, ls=":",
                          label="decoded FINCH centroid (aligned key)"),
    "own": dict(color="#C44E52", lw=1.1, ls=":", label="own district demand"),
}


def plot_ladder(ax, arrays: dict, channel: int, own=None, title: str = ""):
    """One channel: every source in `arrays` ({source: (W, F)}) and, if
    given, the own-district demand shape drawn on the true window's scale
    (sign matched to its correlation, display only)."""
    from .figures import _on_scale
    for name, a in arrays.items():
        if a is not None:
            ax.plot(a[:, channel], **STYLE.get(name, dict(label=name)))
    ref = arrays.get("true")
    if own is not None and ref is not None:
        r = ref[:, channel]
        c = RC._r(r, own)
        ax.plot(_on_scale(own, r, np.sign(c) or 1), **STYLE["own"])
    ax.set(xlabel="step (window resolution)", ylabel="scaled value")
    ax.set_title(title, fontsize=9, loc="left")
    return ax


def plot_ladder_summary(axes, summ: pd.DataFrame,
                        metrics=("amp", "r_shape", "r2_dem")):
    """One panel per metric: bars per source, grouped by phase."""
    phases = [p for p in PERIODS if p in set(summ["phase"].astype(str))]
    srcs = list(summ["source"].drop_duplicates())
    width = 0.8 / max(len(srcs), 1)
    for ax, m in zip(np.ravel(axes), metrics):
        for j, s in enumerate(srcs):
            d = summ[summ["source"] == s].set_index(
                summ.loc[summ["source"] == s, "phase"].astype(str))
            ax.bar(np.arange(len(phases)) + j * width,
                   [d[m].get(p, np.nan) for p in phases], width,
                   color=STYLE.get(s, {}).get("color"), label=s)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(len(phases)) + width * (len(srcs) - 1) / 2)
        ax.set_xticklabels(phases)
        ax.set_title(m, fontsize=9, loc="left")
    np.ravel(axes)[0].legend(fontsize=7)
    return axes


def plot_eta2(ax, per: pd.DataFrame):
    """eta2 per block over months, one line per client, chance dashed."""
    from .figures import colour_map
    cm = colour_map(per["client"].unique())
    for c, d in per.groupby("client"):
        d = d.sort_values("bid")
        ax.plot(d["bid"], d["eta2"], marker=".", color=cm[c], label=c)
    ch = per.groupby("bid")["chance"].mean()
    ax.plot(ch.index, ch.values, "k--", lw=1, label="chance (K-1)/(n-1)")
    ax.set(xlabel="block", ylabel="eta2 (cycle position)", ylim=(0, 1))
    ax.legend(fontsize=7, ncol=3)
    return ax


def plot_comb(ax, comb: pd.DataFrame):
    """Median comb gain per period, pooled vs aligned."""
    m = comb.groupby(["period_h", "scope"])["gain"].median().unstack("scope")
    m = m.sort_index(ascending=False)
    x = np.arange(len(m))
    for j, s in enumerate([s for s in ("pooled", "aligned") if s in m]):
        ax.bar(x + (j - 0.5) * 0.4, m[s], 0.4, label=s,
               color="#4C72B0" if s == "pooled" else "#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} h" for p in m.index])
    ax.set(ylabel="|H(P)| (median)", ylim=(0, 1.05),
           title="what averaging keeps, by period")
    ax.legend(fontsize=7)
    return ax


# --------------------------------------------------------------------------
# browser
# --------------------------------------------------------------------------
def aligned_browser(run: AlignedRun, fid: pd.DataFrame | None = None,
                    channels: pd.DataFrame | None = None):
    """client / channel / block-or-phase / cycle position, every source.

    `period = block`: one block, one cycle position (or all positions as
    small multiples). `period = init|transition|final`: each source
    averaged over that phase's groups at each cycle position. The table
    under the plot is `fidelity` for exactly that selection."""
    from .widgets import _channel_label, _require_widgets
    W, display, plt = _require_widgets()
    plt.ioff()
    clients = sorted(run.wt["client"].unique())
    G = run.groups
    blocks = (G.drop_duplicates("bid").set_index("bid")["block_label"])
    n_ch = next(iter(run.arrays["true"].values())).shape[1]

    w_cl = W.Dropdown(options=clients, description="client")
    w_ch = W.Dropdown(description="channel", layout=W.Layout(width="520px"))
    w_pe = W.ToggleButtons(options=["block", *PERIODS], value="block",
                           description="period")
    w_bl = W.SelectionSlider(options=[(v, int(k)) for k, v in blocks.items()],
                             description="block", continuous_update=False,
                             layout=W.Layout(width="520px"))
    w_cy = W.Dropdown(description="cycle")
    w_all = W.Checkbox(value=False, description="all cycle positions")
    w_src = W.SelectMultiple(options=list(run.arrays),
                             value=tuple(s for s in run.arrays
                                         if s != "ref_pooled"),
                             rows=min(6, len(run.arrays)), description="sources")
    w_dem = W.Checkbox(value=True, description="demand overlay")
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

    def _positions():
        if w_pe.value == "block":
            g = G[G["bid"] == w_bl.value]
        else:
            g = G[G["phase"] == w_pe.value].drop_duplicates("cycle_pos")
        return [(r.cycle, int(r.cycle_pos), int(r.gid))
                for r in g.sort_values("cycle_pos").itertuples()]

    def _set_cycles(*_):
        keep = w_cy.value
        opts = [(lab, pos) for lab, pos, _ in _positions()]
        w_cy.options = opts
        valid = [v for _, v in opts]
        if valid:
            w_cy.value = keep if keep in valid else valid[0]

    def _panel(pos, gid):
        c = w_cl.value
        if w_pe.value == "block":
            arr = {s: run.arrays[s].get((c, gid)) for s in w_src.value}
            arr["true"] = run.arrays["true"].get((c, gid))
            dm = run.demand.get((c, gid))
        else:
            pa, pd_ = phase_arrays(run, w_pe.value)
            arr = {s: pa[s].get((c, pos)) for s in w_src.value}
            arr["true"] = pa["true"].get((c, pos))
            dm = pd_.get((c, pos))
        arr = {"true": arr.pop("true"), **arr}
        own = dm.get(c) if (w_dem.value and dm) else None
        return arr, own

    def _draw(*_):
        with out:
            out.clear_output(wait=True)
            if w_ch.value is None or w_cy.value is None:
                return
            pos_all = _positions()
            sel = pos_all if w_all.value else \
                [p for p in pos_all if p[1] == w_cy.value]
            if not sel:
                print("nothing in this selection")
                return
            fig, axes = plt.subplots(len(sel), 1,
                                     figsize=(13, 3.0 * len(sel)),
                                     squeeze=False)
            where = (blocks[w_bl.value] if w_pe.value == "block"
                     else f"{w_pe.value} mean")
            for ax, (lab, pos, gid) in zip(axes[:, 0], sel):
                arr, own = _panel(pos, gid)
                if arr["true"] is None:
                    ax.set_title(f"{w_cl.value}: no windows at {lab}")
                    continue
                plot_ladder(ax, arr, w_ch.value, own,
                            title=f"{w_cl.value} · {where} · start {lab} · "
                                  f"channel {w_ch.value}")
            axes[0, 0].legend(fontsize=7, ncol=3)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            if fid is not None and len(fid):
                t = fid[(fid["client"] == w_cl.value)
                        & (fid["channel"] == w_ch.value)]
                t = (t[t["bid"] == w_bl.value] if w_pe.value == "block"
                     else t[t["phase"] == w_pe.value])
                if not w_all.value:
                    t = t[t["cycle"].isin([s[0] for s in sel])]
                print(t.groupby(["source", "cycle"])
                      [["rmse", "r_shape", "amp", "r2_dem"]].mean()
                      .round(3).to_string())

    def _on_period(ch):
        w_bl.disabled = ch["new"] != "block"
        _set_cycles()
        _draw()

    w_cl.observe(lambda *_: (_set_channels(), _draw()), names="value")
    w_bl.observe(lambda *_: (_set_cycles(), _draw()), names="value")
    w_pe.observe(_on_period, names="value")
    for x in (w_ch, w_cy, w_all, w_src, w_dem):
        x.observe(_draw, names="value")
    _set_cycles()
    _set_channels()
    ui = W.VBox([W.HBox([w_cl, w_ch]), w_pe, W.HBox([w_bl, w_cy]),
                 W.HBox([w_src, W.VBox([w_all, w_dem])]), out])
    display(ui)
    _draw()
    return ui
