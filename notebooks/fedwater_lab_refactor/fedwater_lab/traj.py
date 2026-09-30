"""Stages 1-2 -- one frozen ruler, and the trajectory it measures.

Everything downstream consumes ``phi[c, b]``: a small, fixed feature vector
per client per block. This module builds it, and builds the two things it
has to be read against -- the data-side counterpart (the centralized
ceiling) and the moving-ruler version (the artifact you get if you forget to
freeze the encoder).

Why a frozen ruler
------------------
`prototype_history` stores the prototype of month m as it was at the round it
was produced, so a trajectory read straight off it mixes WORLD change with
TRAINING progress -- POC_04's prequential curves are dominated by the latter
(median r climbing monotonically from 0 to 0.79). The trajectory here is
instead built from ONE model applied to every window (`aligned.build`, whose
`arrays["aligned"]` are `dec(p[c,g])` under the final weights). The
commissioning cell is the principled choice of ruler: it never trained on
the drift, so it cannot adapt the signal away. `history_features` builds the
moving-ruler version on purpose, so `ruler_stability` can measure how much
of the naive trajectory was the model moving rather than the world.

The features, per decoded window ``X`` (W steps of ``agg_h`` hours, F channels)
--------------------------------------------------------------------------
Channel-wise, then averaged over the kept channels (`slot_class` selects
pure / mixed / all):

``lvl = mean_t X``
    The window's level. Under `minmax_ref` the scaler is per client, so this
    is comparable WITHIN a client over time and NOT across clients. Idea 1
    only ever regresses a client's own series on another's, so a
    non-comparable level is not a problem there; `chain`'s codebook is
    cross-client and it IS, which is why `codebook_space` exists there.

``amp = std_t X``
    Within-window swing. On a demand ladder the diurnal amplitude is the
    regime (few consumers -> peaky, many -> crowd-smoothed), and it survives
    per-client MinMax where the level does not.

``p24 = band_share(X, 24 h)``
    Share of the window's variance in the frequency bins nearest one cycle
    per 24 h:  ``sum_{j in B} |F_j|^2 / sum_{j>0} |F_j|^2`` with
    ``B = {j : |j - W*agg_h/24| <= width}`` on the mean-removed window,
    orthonormal FFT. NOTE the target bin ``W*agg_h/24`` need not be an
    integer (a3w84 gives 10.5), so the band is a genuine band of `width`
    bins on either side of the nearest one, and the same band is used for
    every source -- it is a relative measure, compared like with like.

``p168``
    The same at one cycle per week, when the window is long enough to carry
    it (``W*agg_h >= 168``); NaN otherwise.

Per block, features are averaged over the block's cycle classes, and -- when
the alignment key is weekly -- a weekend contrast is added:

``amp_we = mean_{weekend classes} amp - mean_{weekday classes} amp``

which is the cheapest shape statement that a level-destroying scaler cannot
remove.

Standardisation
---------------
`standardize` z-scores each (client, feature) using the INIT blocks only and
applies those statistics everywhere. Fitting the scaler on the whole
trajectory would leak the drift into the units the drift is measured in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["FeatureSpec", "KEYS", "band_share", "window_features", "class_features",
           "demand_class_features", "block_features", "block_index",
           "standardize", "history_features", "ruler_stability",
           "recoverability", "scaling_discrimination", "trajectory",
           "feature_cols", "resolution_h", "latent_windows", "latent_phi",
           "FEATS"]

FEATS = ("lvl", "amp", "p24")


@dataclass(frozen=True)
class FeatureSpec:
    """How a decoded window becomes numbers."""

    feats: tuple[str, ...] = FEATS
    band_width: int = 1                 # bins either side of the target bin
    weekend_contrast: bool = True       # adds `amp_we`/`lvl_we` on a weekly key

    def label(self) -> str:
        return "+".join(self.feats) + (f"_w{self.band_width}")


# --------------------------------------------------------------------------
# one window -> features
# --------------------------------------------------------------------------
def band_share(x: np.ndarray, agg_h: float, period_h: float = 24.0,
               width: int = 1) -> float:
    """Share of variance in the bins nearest one cycle per `period_h`."""
    x = np.asarray(x, float)
    W = len(x)
    if W < 4:
        return np.nan
    span_h = W * float(agg_h)
    if span_h < period_h:
        return np.nan
    # scipy.fft, NOT np.fft: a conda MKL NumPy routes np.fft through MKL, whose
    # FFT initialises a second copy of Intel's OpenMP runtime on first use.
    # With torch's own copy already running (every notebook here calls the
    # decoder first) that aborts the kernel on Windows with OMP Error #15.
    # scipy.fft is pocketfft, runs on one worker, and loads no OpenMP runtime.
    from scipy import fft as _sfft
    F = np.abs(_sfft.rfft(x - x.mean(), norm="ortho")) ** 2
    tot = float(F[1:].sum())
    if tot <= 1e-24:
        return np.nan
    j = span_h / float(period_h)
    lo = max(1, int(np.floor(j)) - int(width))
    hi = min(len(F) - 1, int(np.ceil(j)) + int(width))
    return float(F[lo:hi + 1].sum() / tot)


def _nanmean(vals) -> float:
    """Mean of the finite entries; NaN (quietly) when there are none."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return float(v.mean()) if len(v) else np.nan


def window_features(X: np.ndarray, agg_h: float, channels=None,
                    fs: FeatureSpec = FeatureSpec()) -> dict:
    """Features of one decoded window `(W, F)` (or `(W,)`), channel-averaged."""
    A = np.asarray(X, float)
    if A.ndim == 1:
        A = A[:, None]
    idx = range(A.shape[1]) if channels is None else [i for i in channels
                                                      if i < A.shape[1]]
    idx = list(idx)
    if not idx:
        return {f: np.nan for f in ("lvl", "amp", "p24", "p168")}
    lvl = float(np.mean([A[:, i].mean() for i in idx]))
    amp = float(np.mean([A[:, i].std() for i in idx]))
    p24 = _nanmean([band_share(A[:, i], agg_h, 24.0, fs.band_width)
                    for i in idx])
    p168 = _nanmean([band_share(A[:, i], agg_h, 168.0, fs.band_width)
                     for i in idx])
    return {"lvl": lvl, "amp": amp, "p24": p24, "p168": p168}


# --------------------------------------------------------------------------
# an AlignedRun -> per-class features
# --------------------------------------------------------------------------
def _keep(channels: pd.DataFrame | None, slot_class: str | None
          ) -> dict[str, list[int]] | None:
    if channels is None or slot_class in (None, "all"):
        return None
    return {c: sorted(sub.loc[sub["slot_class"] == slot_class, "channel"]
                      .astype(int))
            for c, sub in channels.groupby("client")}


def block_index(run) -> pd.DataFrame:
    """bid -> phase, first/last month, ordered by time."""
    g = (run.groups.groupby("bid")
         .agg(phase=("phase", "first"), m_lo=("m_lo", "min"),
              m_hi=("m_hi", "max"), n_classes=("gid", "nunique"))
         .reset_index().sort_values("m_lo").reset_index(drop=True))
    return g


def class_features(run, channels: pd.DataFrame | None = None,
                   slot_class: str | None = "pure", source: str = "aligned",
                   fs: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    """One row per (client, gid): features of `run.arrays[source]`.

    `source="aligned"` is the decoded prototype (the federated artifact);
    `"true"` is the mean window it summarises (the in-model-units reference);
    `"ceiling"` is decode-then-average.
    """
    arrays = run.arrays[source]
    keep = _keep(channels, slot_class)
    g = run.groups.set_index("gid")
    rows = []
    for (c, gid), X in arrays.items():
        info = g.loc[int(gid)]
        f = window_features(X, run.agg_h,
                            None if keep is None else keep.get(c), fs)
        rows.append({"client": c, "gid": int(gid), "bid": int(info["bid"]),
                     "phase": info["phase"], "cycle_pos": int(info["cycle_pos"]),
                     "source": source, **f})
    return pd.DataFrame(rows).sort_values(["client", "bid", "cycle_pos"]) \
        .reset_index(drop=True)


def demand_class_features(run, fs: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    """The centralized ceiling: the same features on the client's OWN district
    demand, cut at exactly the same window starts (`run.demand`)."""
    g = run.groups.set_index("gid")
    rows = []
    for (c, gid), d in run.demand.items():
        if c not in d:
            continue
        info = g.loc[int(gid)]
        f = window_features(np.asarray(d[c]), run.agg_h, None, fs)
        rows.append({"client": c, "gid": int(gid), "bid": int(info["bid"]),
                     "phase": info["phase"], "cycle_pos": int(info["cycle_pos"]),
                     "source": "demand", **f})
    return pd.DataFrame(rows).sort_values(["client", "bid", "cycle_pos"]) \
        .reset_index(drop=True)


# --------------------------------------------------------------------------
# per-class -> per-block phi
# --------------------------------------------------------------------------
def block_features(cls: pd.DataFrame, period_h: int | None = 168,
                   fs: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    """Average the cycle classes of a block; add the weekend contrast."""
    feats = [f for f in fs.feats if f in cls.columns]
    base = (cls.groupby(["client", "bid", "phase", "source"])[feats]
            .mean().reset_index())
    if fs.weekend_contrast and period_h is not None and period_h > 24:
        d = cls.assign(weekend=(cls["cycle_pos"] // 24) >= 5)
        if d["weekend"].nunique() == 2:
            con = (d.groupby(["client", "bid", "source", "weekend"])[feats]
                   .mean().unstack("weekend"))
            for f in feats:
                if f in ("lvl", "amp"):
                    base[f + "_we"] = (con[(f, True)] - con[(f, False)]).reindex(
                        pd.MultiIndex.from_frame(
                            base[["client", "bid", "source"]])).to_numpy()
    return base.sort_values(["client", "bid"]).reset_index(drop=True)


#: columns that are keys or bookkeeping, never features
KEYS = ("client", "bid", "phase", "source", "b", "cycle_pos", "gid", "seq",
        "n", "month", "round", "resolution_h", "start", "start_h", "block",
        "row", "kind", "window", "hour", "weekday", "drift_status")


def feature_cols(phi: pd.DataFrame) -> list[str]:
    return [c for c in phi.columns if c not in KEYS and
            pd.api.types.is_numeric_dtype(phi[c])]


def standardize(phi: pd.DataFrame, ref_phase: str = "init") -> pd.DataFrame:
    """z per (client, feature) using `ref_phase` blocks only."""
    out = phi.copy()
    cols = feature_cols(out)
    for f in cols:
        out[f] = out[f].astype(float)
    for c, sub in out.groupby("client"):
        ref = sub[sub["phase"] == ref_phase]
        if not len(ref):
            ref = sub
        for f in cols:
            mu = float(ref[f].mean())
            sd = float(ref[f].std(ddof=0))
            sd = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
            out.loc[sub.index, f] = (sub[f] - mu) / sd
    return out


def trajectory(run, channels: pd.DataFrame | None = None,
               slot_class: str | None = "pure",
               fs: FeatureSpec = FeatureSpec(),
               standardise: bool = True, ref_phase: str = "init") -> dict:
    """The whole stage: class features -> phi, model side and demand side.

    Two resolutions, and they are for different stages:

    ``phi_class``  one row per (client, block, cycle class). The calendar
                   classes are the REPLICATE axis that Stages 3-5 pool over
                   (`transfer.panel`, `chain.symbol_table`,
                   `partialdep.client_matrix` with `seq_key="cycle_pos"`).
                   This is what those stages must consume.
    ``phi``        one row per (client, block): classes averaged, plus the
                   weekend contrasts, which only exist across classes. This
                   is for block-level reading (`recoverability`,
                   `ruler_stability`). Feeding it to Stages 3-5 silently
                   collapses the replicate axis to S = 1.
    """
    cls = class_features(run, channels, slot_class, "aligned", fs)
    dem = demand_class_features(run, fs)
    period = run.align.period_h
    phi = block_features(cls, period, fs)
    phi_d = block_features(dem, period, fs)
    keys = ["client", "bid", "phase", "cycle_pos", "source"]
    feats = [f for f in fs.feats if f in cls.columns]
    phi_c, phi_cd = cls[keys + feats].copy(), dem[keys + feats].copy()
    if standardise:
        phi, phi_d = standardize(phi, ref_phase), standardize(phi_d, ref_phase)
        phi_c, phi_cd = standardize(phi_c, ref_phase), standardize(phi_cd, ref_phase)
    return {"class": cls, "class_demand": dem,
            "phi": phi, "phi_demand": phi_d,
            "phi_class": phi_c, "phi_class_demand": phi_cd,
            "blocks": block_index(run), "slot_class": slot_class or "all",
            "fs": fs}


# --------------------------------------------------------------------------
# the moving ruler, and the checks that compare the two
# --------------------------------------------------------------------------
def _decode(model, Z: np.ndarray, batch: int = 512) -> np.ndarray:
    from . import train as T
    Z = np.array(Z, np.float32)
    if not len(Z):
        return np.zeros((0, 1, 1))
    return np.concatenate([T.decode(model, Z[i:i + batch])
                           for i in range(0, len(Z), batch)])


def history_features(res: dict, model, world, channels=None,
                     slot_class: str | None = "pure",
                     fs: FeatureSpec = FeatureSpec(), mode: str = "auto",
                     *, agg_h: float) -> pd.DataFrame:
    """The MOVING ruler: `prototype_history` decoded with the frozen decoder.

    One row per (client, month). `mode="first"` takes each month's prototype
    at the round it was first trained (the streaming schedule's own answer,
    available when the history carries a `block` column); `"last"` takes the
    final round; `"auto"` picks `first` when a block column exists.

    Decoding every round's latent with ONE decoder is what makes the two
    trajectories comparable at all: the feature definition is then identical
    and the only difference is which weights produced the latent.

    `agg_h` is REQUIRED and must be the run's own hours-per-step
    (`AlignedRun.agg_h`). `world.fl` is the world's DEFAULT preprocessing,
    not the geometry the run was trained with; reading it would compute the
    24 h / 168 h bands at the wrong frequency whenever the spec overrides the
    geometry, and `ruler_stability` would then compare two different
    features without complaint.
    """
    ph = res.get("prototype_history")
    if ph is None or not len(ph):
        return pd.DataFrame()
    fc = [c for c in ph.columns if c.startswith("f") and c[1:].isdigit()]
    if mode == "auto":
        mode = "first" if "block" in ph.columns else "last"
    pick = (ph.groupby(["client", "month"])["round"]
            .min() if mode == "first" else
            ph.groupby(["client", "month"])["round"].max()).reset_index()
    sel = ph.merge(pick, on=["client", "month", "round"], how="inner")
    D = _decode(model, sel[fc].to_numpy())
    agg = float(agg_h)
    keep = _keep(channels, slot_class)
    rows = []
    for i, r in enumerate(sel.itertuples(index=False)):
        f = window_features(D[i], agg, None if keep is None
                            else keep.get(r.client), fs)
        rows.append({"client": r.client, "month": int(r.month),
                     "round": int(r.round), "phase": world.phase_of(int(r.month)),
                     "source": f"history_{mode}", **f})
    return pd.DataFrame(rows).sort_values(["client", "month"]) \
        .reset_index(drop=True)


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def ruler_stability(phi_frozen: pd.DataFrame, phi_moving: pd.DataFrame,
                    blocks: pd.DataFrame) -> pd.DataFrame:
    """Frozen-encoder trajectory against the moving-ruler one, per feature.

    Low correlation means the naive trajectory is largely a training curve
    and the frozen ruler is doing real work; high correlation means the two
    agree and the freezing is cheap insurance. Either way it is a number.
    """
    if not len(phi_moving):
        return pd.DataFrame()
    b = blocks[["bid", "m_lo", "m_hi"]]
    mv = phi_moving.copy()
    bid = []
    for m in mv["month"]:
        hit = b[(b["m_lo"] <= m) & (m <= b["m_hi"])]
        bid.append(int(hit["bid"].iloc[0]) if len(hit) else -1)
    mv["bid"] = bid
    feats = [f for f in feature_cols(phi_frozen) if f in mv.columns]
    mv = mv.groupby(["client", "bid"])[feats].mean().reset_index()
    j = phi_frozen.merge(mv, on=["client", "bid"], suffixes=("", "_mv"))
    rows = []
    for c, sub in j.groupby("client"):
        for f in feats:
            rows.append({"client": c, "feature": f, "n": len(sub),
                         "r_frozen_moving": _corr(sub[f], sub[f + "_mv"])})
    return pd.DataFrame(rows)


def recoverability(phi: pd.DataFrame, phi_demand: pd.DataFrame) -> pd.DataFrame:
    """Does phi track the data at all? Per (client, feature), the correlation
    of the model-side trajectory with its own district-demand counterpart.

    This is the gate on every later number: a feature that does not track the
    demand it claims to summarise cannot support a dependence claim."""
    feats = [f for f in feature_cols(phi) if f in phi_demand.columns]
    j = phi.merge(phi_demand, on=["client", "bid"], suffixes=("", "_dem"))
    rows = []
    for c, sub in j.groupby("client"):
        for f in feats:
            rows.append({"client": c, "feature": f, "n": len(sub),
                         "r_model_demand": _corr(sub[f], sub[f + "_dem"])})
    return pd.DataFrame(rows)


def scaling_discrimination(phi_a: pd.DataFrame, phi_b: pd.DataFrame,
                           label_a: str = "minmax_ref",
                           label_b: str = "shared") -> pd.DataFrame:
    """Two scalings, same world: which features actually changed?

    Expectation, and the check: the LEVEL features must differ between a
    per-client and a pooled scaler (if `lvl` survives per-client MinMax it is
    not a level), and the SHAPE features must not (if `amp`/`p24` move with
    the scaler they are carrying level after all).
    """
    feats = [f for f in feature_cols(phi_a) if f in phi_b.columns]
    j = phi_a.merge(phi_b, on=["client", "bid"], suffixes=("_a", "_b"))
    rows = []
    for f in feats:
        kind = "level" if f.startswith("lvl") else "shape"
        r = _corr(j[f + "_a"], j[f + "_b"])
        rows.append({"feature": f, "kind": kind, "n": len(j),
                     f"r({label_a},{label_b})": r,
                     "expected": "differs" if kind == "level" else "unchanged",
                     "ok": (r < 0.99) if kind == "level" else (r > 0.9)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the world-free path: latents only
# --------------------------------------------------------------------------
# `aligned.build` needs the world (its demand, its sensor table) and the
# decoder. When only a saved run is at hand -- `latent_trajectories` plus
# `fl.json` -- the same trajectory can still be built in LATENT space, which
# is the raw-latent arm of the comparison rather than a substitute for the
# decoded one. Everything here is derived from the run's own artifacts; no
# world is consulted and nothing is assumed about one.


def resolution_h(latents: pd.DataFrame, preprocessing: dict) -> float:
    """Native hours per step, recovered from the window starts.

    Consecutive windows are `step_size` MODEL steps apart, i.e.
    `step_size * interval_agg_h` hours, and `window` counts NATIVE steps, so

        resolution_h = step_size * interval_agg_h / median diff(window).
    """
    w = np.sort(np.asarray(latents["window"].unique(), float))
    if len(w) < 2:
        raise ValueError("need at least two distinct window starts")
    d = float(np.median(np.diff(w)))
    hop_h = float(preprocessing["step_size"]) * float(preprocessing["interval_agg_h"])
    return hop_h / d


def latent_windows(latents: pd.DataFrame, preprocessing: dict,
                   phases: dict, block_months: int = 3,
                   period_h: int | None = 168, bin_h: int | None = 24
                   ) -> pd.DataFrame:
    """Per-window latents + calendar keys, from a saved run alone.

    `phases` is the run's own `meta["phases"]` (half-open month ranges);
    anything outside them is `transition`, exactly as `World.phase_of` says.
    Blocks are cut inside a phase, never across one.
    """
    res = resolution_h(latents, preprocessing)
    wt = latents.rename(columns={"district": "client", "window": "start"}).copy()
    wt["start_h"] = wt["start"].astype(float) * res
    pos = np.mod(wt["start_h"], period_h) if period_h else np.zeros(len(wt))
    if bin_h:
        pos = np.floor(pos / bin_h) * bin_h
    wt["cycle_pos"] = np.round(pos).astype(int)

    def phase_of(m):
        for name, (lo, hi) in phases.items():
            if lo <= m < hi:
                return name
        return "transition"

    wt["phase"] = [phase_of(int(m)) for m in wt["month"]]
    lo = {p: (int(phases["init"][1]) if p == "transition"
              else int(phases[p][0])) for p in wt["phase"].unique()}
    wt["block"] = [(int(m) - lo[p]) // int(block_months)
                   for m, p in zip(wt["month"], wt["phase"])]
    order = {"init": 0, "transition": 1, "final": 2}
    blocks = (wt.groupby(["phase", "block"])["month"].min().reset_index()
              .assign(o=lambda d: d["phase"].map(order))
              .sort_values(["month", "o"]).reset_index(drop=True))
    blocks["bid"] = np.arange(len(blocks))
    wt = wt.merge(blocks[["phase", "block", "bid"]], on=["phase", "block"])
    wt["resolution_h"] = res
    return wt


def latent_phi(wt: pd.DataFrame, n_comp: int = 2, ref_phase: str = "init",
               basis: str = "per_client", standardise: bool = True,
               seed: int = 0) -> dict:
    """Latent prototypes per (client, block, cycle class), projected to `n_comp`.

    The prototype itself is the group mean latent -- the artifact the
    protocol already exchanges (`aligned._means` computes the same thing).
    The projection is fitted on `ref_phase` ONLY and then frozen, so the
    drift is never inside the basis that measures it.

    `basis="per_client"` fits each client its own PCA: the components are
    then NOT comparable across clients, which is fine for `transfer` (the
    regression learns the map) and wrong for `chain` (whose codebook is
    cross-client) -- use `basis="pooled"` there.
    """
    from sklearn.decomposition import PCA
    fc = [c for c in wt.columns if c.startswith("f") and c[1:].isdigit()]
    protos = (wt.groupby(["client", "bid", "cycle_pos", "phase"])[fc]
              .mean().join(wt.groupby(["client", "bid", "cycle_pos", "phase"])
                           .size().rename("n")).reset_index())
    rows = []
    if basis == "pooled":
        fit = protos[protos["phase"] == ref_phase]
        pca = PCA(n_components=n_comp, random_state=seed).fit(
            (fit if len(fit) > n_comp else protos)[fc].to_numpy(float))
        Z = pca.transform(protos[fc].to_numpy(float))
        out = protos[["client", "bid", "cycle_pos", "phase", "n"]].copy()
        for j in range(n_comp):
            out[f"z{j}"] = Z[:, j]
        evr = {"pooled": pca.explained_variance_ratio_.tolist()}
    else:
        parts, evr = [], {}
        for c, sub in protos.groupby("client"):
            fit = sub[sub["phase"] == ref_phase]
            pca = PCA(n_components=n_comp, random_state=seed).fit(
                (fit if len(fit) > n_comp else sub)[fc].to_numpy(float))
            Z = pca.transform(sub[fc].to_numpy(float))
            q = sub[["client", "bid", "cycle_pos", "phase", "n"]].copy()
            for j in range(n_comp):
                q[f"z{j}"] = Z[:, j]
            parts.append(q)
            evr[c] = pca.explained_variance_ratio_.tolist()
        out = pd.concat(parts, ignore_index=True)
    del rows
    out = out.sort_values(["client", "bid", "cycle_pos"]).reset_index(drop=True)
    if standardise:
        out = standardize(out, ref_phase)
    return {"protos": protos, "phi": out, "basis": basis,
            "explained_variance": evr,
            "occupancy": protos.groupby(["client", "bid"])["n"].min()}
