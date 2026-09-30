"""Stage 3 -- does client c's past say anything about client d's next block?

Idea 1, as a prequential out-of-sample test rather than a Granger F. Nothing
here imports torch or the rest of the package: it takes `phi` (one row per
client and block, from `traj`) and returns numbers, so it can be smoke-tested
against a synthetic system with a known direction.

The estimand
------------
For target client d, source client c, lag order p, and block b:

    restricted:  phi_d[b] ~ phi_d[b-1..b-p]  +  ctrl[b]  +  ctrl[b-1..b-p]
    full:        the same, plus phi_c[b-1..b-p]

where ``ctrl`` is built from the clients OTHER than c and d -- their mean
(`control="mean"`) or each of them separately (`control="each"`).

The control enters at the CONCURRENT block as well as at lags, and that is
the design decision the whole test rests on. A shared driver (season,
population growth, a network-wide operating change) makes every client a
noisy measurement of the same latent f[b]. With lagged controls only, f[b]
must be extrapolated, so ANY additional client improves the forecast and the
gain becomes a statement about measurement count rather than mechanism --
measured on the stand-in, that inverts the known direction (G(A->B) 0.34
against G(B->A) 0.64 with lagged controls; 0.32 against -0.07 once the
controls are concurrent). Conditioning on the others' PRESENT absorbs f[b],
so only the source's PAST can still carry anything.

Subtracting the cross-client mean instead -- the obvious alternative, and
the operation `fedwater`'s invariant 5 prescribes for dependence statistics
-- is the wrong tool here: with four clients the sum-to-zero constraint
forces a spurious negative dependence between residuals and the known
coupling disappears entirely (G(A->B) -0.15). Control, do not subtract.

Because the control set excludes c, an INSTANTANEOUS mechanism from c to d
is not absorbed and can reach G through c's own autocorrelation. G is
therefore a PREDICTIVE dependence; the instantaneous part is Stage 5's
business (`partialdep`), which is exactly the split the physics asks for on
this data -- hydraulic coupling is at lag 0, drift propagation is not.

Both models are fitted on blocks strictly before b (expanding window, or a
sliding window of `win` for the rolling version), so every error is
out-of-sample and the drift can never be fitted before it is predicted.
Ridge with penalty `lam` on design columns standardised by TRAINING-fold
statistics only.

    G_{c->d} = 1 - SSE_full / SSE_restricted

summed over the evaluated blocks and over the features of phi. G <= 0 means
c bought nothing. The asymmetry G_{c->d} - G_{d->c} is the directional
statement; G itself is a predictability statement.

The three controls, and why each exists
---------------------------------------
``r2_restricted``
    ``1 - SSE_restricted / SSE_mean``, where SSE_mean predicts the training
    mean. This is the AR-R^2 diagnostic: if G tracks r2_restricted across
    targets, the asymmetry is a self-predictability artifact ON THIS WORLD,
    which is a measurement rather than an inherited assumption. Plot one
    against the other before reading any direction.

``capacity_null``
    The full model always has more parameters than the restricted one, so a
    positive G can be capacity alone. The null refits the full model with a
    SURROGATE source of identical dimension (`permute` shuffles the source's
    blocks, `shift` rotates it, `client` substitutes a third client), giving
    the distribution of G under "a regressor of this shape that cannot
    carry the mechanism". Read `z` and the empirical `p`, not G alone. Out of
    sample the null should centre at or below 0; a null centred clearly
    ABOVE 0 means the protocol is leaking (a scaler fitted on the full
    series, or a training fold that is not strictly earlier).

``reverse_blocks``
    Time reversal. A propagation signal weakens or inverts; a variance
    artifact is symmetric.

Rolling and onset
-----------------
``rolling_gain`` refits on a SLIDING window of `win` blocks, predicts the
next block, and pools the errors over `smooth` consecutive blocks:

    G(b) = 1 - sum_{b' in (b-smooth, b]} e_full(b')^2
             / sum_{b' in (b-smooth, b]} e_restricted(b')^2

``onset_test`` asks where G(b) peaks. Under the null that the peak is
uniform over the evaluated blocks, ``p = n_ramp_blocks / n_evaluated`` --
exact, cheap, and hard to hit by accident, which is what makes "the gain
peaks inside the ramp" a stronger claim than any static G.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TransferCfg", "panel", "transfer_gain", "gain_matrix",
           "capacity_null", "null_matrix", "rolling_gain", "onset_test",
           "reverse_blocks", "asymmetry", "arm_table"]


@dataclass(frozen=True)
class TransferCfg:
    p: int = 1                     # lag order
    min_train: int = 8             # blocks before the first prediction
    lam: float = 1e-2              # ridge penalty (standardised design)
    control: str = "mean"          # none | mean | each  (clients not in {c,d})
    control_time: str = "both"     # lag | now | both -- `now` is what de-confounds
    feats: tuple[str, ...] | None = None   # None = every numeric column
    eval_phase: str | None = None  # score only blocks of this phase

    def __post_init__(self):
        if self.control not in ("none", "mean", "each"):
            raise KeyError(f"control {self.control!r}")
        if self.control_time not in ("lag", "now", "both"):
            raise KeyError(f"control_time {self.control_time!r}")

    def label(self) -> str:
        return (f"p{self.p}_mt{self.min_train}_l{self.lam:g}"
                f"_{self.control}-{self.control_time}"
                + (f"_{self.eval_phase}" if self.eval_phase else ""))


# --------------------------------------------------------------------------
# phi -> aligned panel
# --------------------------------------------------------------------------
def panel(phi: pd.DataFrame, feats: tuple[str, ...] | None = None,
          seq_key: str | None = "cycle_pos") -> dict:
    """{'Y': {client: (S, B, k)}, 'b': (B,), 'phase': (B,), 'feats': [...]}.

    `S` is the REPLICATE axis: the calendar classes of a block, when `phi`
    still carries them (`seq_key`), else 1. One model is fitted pooled over
    the replicates and scored on each, which multiplies both the training
    rows and the error terms per block by S -- the same "classes as
    replicates" move `chain` uses, and what makes the rolling gain
    estimable at all. Replicates are NOT independent months, so they buy
    estimation and not inference: the nulls permute whole blocks, jointly
    across replicates.

    The grid is RAGGED in general: a coarse window hop (e.g. `s12`, 36 h)
    visits the weekday classes unevenly, so some (class, block) cells hold
    no window for some client. Missing cells are NaN in `Y`, and `_errors`
    drops every design row that touches one -- both models on the same
    rows, so the SSE ratio stays a like-for-like comparison. `coverage`
    reports the share of the grid each client actually fills.
    """
    if feats is None:
        from .traj import feature_cols
        feats = feature_cols(phi)
    feats = [f for f in feats if f in phi.columns]
    d = phi.copy()
    d["_seq"] = (0 if seq_key is None or seq_key not in d.columns
                 else d[seq_key].astype(int))
    clients = sorted(d["client"].unique())
    bids = sorted(set.intersection(*[set(d.loc[d["client"] == c, "bid"])
                                     for c in clients])) if clients else []
    seqs = sorted(set.intersection(*[set(d.loc[d["client"] == c, "_seq"])
                                     for c in clients])) if clients else [0]
    d = d[d["bid"].isin(bids) & d["_seq"].isin(seqs)]

    # a feature undefined for every row of some client carries nothing
    feats = [f for f in feats
             if not d.groupby("client")[f].apply(lambda v: v.notna().sum() == 0).any()]
    grid = pd.MultiIndex.from_product([seqs, bids], names=["_seq", "bid"])
    Y, cover = {}, {}
    for c in clients:
        sub = (d[d["client"] == c].set_index(["_seq", "bid"])[feats])
        sub = sub[~sub.index.duplicated(keep="first")]
        A = sub.reindex(grid).to_numpy(float)
        cover[c] = float(np.isfinite(A).all(axis=1).mean())
        Y[c] = A.reshape(len(seqs), len(bids), len(feats))
    phase = (d.groupby("bid")["phase"].first().reindex(bids).to_numpy()
             if "phase" in d.columns else None)
    return {"Y": Y, "b": np.asarray(bids, int), "phase": phase,
            "feats": list(feats), "clients": clients,
            "seqs": np.asarray(seqs, int), "coverage": cover}


def reverse_blocks(phi: pd.DataFrame) -> pd.DataFrame:
    """Same rows, reversed time order (the `bid` axis is flipped)."""
    out = phi.copy()
    hi = int(out["bid"].max())
    out["bid"] = hi - out["bid"]
    return out.sort_values(["client", "bid"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# ridge
# --------------------------------------------------------------------------
def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                 lam: float) -> np.ndarray:
    """Ridge with training-fold standardisation; one fit, many forecasts.

    `Xte` may be a single row or a stack of them (the replicates of one
    block share a fit -- solving per replicate was the same computation
    done S times).
    """
    Xte = np.atleast_2d(Xte)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xtr - mu) / sd
    Zt = (Xte - mu) / sd
    ym = float(ytr.mean())
    A = Z.T @ Z + float(lam) * len(Z) * np.eye(Z.shape[1])
    try:
        w = np.linalg.solve(A, Z.T @ (ytr - ym))
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Z.T @ (ytr - ym), rcond=None)[0]
    return Zt @ w + ym


def _lagmat(A: np.ndarray, s: int, b: int, p: int) -> np.ndarray:
    """Rows b-1 .. b-p of replicate s, flattened."""
    return np.concatenate([A[s, b - l] for l in range(1, p + 1)])


def _errors(P: dict, target: str, source: str | None, cfg: TransferCfg,
            src_override: np.ndarray | None = None,
            win: int | None = None) -> pd.DataFrame:
    """Per-block out-of-sample squared errors of both models.

    `win` None = expanding window (train on every earlier block); an integer
    = sliding window of that many blocks. Every fit pools the replicates;
    every block contributes S x k squared errors.
    """
    Y, bids, phase = P["Y"], P["b"], P["phase"]
    S, B, k = Y[target].shape
    p = int(cfg.p)
    others = [c for c in P["clients"] if c not in (target, source)]
    if cfg.control == "mean" and others:
        ctrls = [np.mean([Y[c] for c in others], axis=0)]
    elif cfg.control == "each" and others:
        ctrls = [Y[c] for c in others]
    else:
        ctrls = []
    Src = Y[source] if source is not None else None
    if src_override is not None:
        Src = src_override

    def row(s: int, b: int, with_src: bool) -> np.ndarray:
        parts = [_lagmat(Y[target], s, b, p)]
        for M in ctrls:
            if cfg.control_time in ("lag", "both"):
                parts.append(_lagmat(M, s, b, p))
            if cfg.control_time in ("now", "both"):
                parts.append(M[s, b])
        if with_src and Src is not None:
            parts.append(_lagmat(Src, s, b, p))
        return np.concatenate(parts)

    rows = []
    for b in range(max(p, cfg.min_train), B):
        if cfg.eval_phase is not None and phase is not None \
                and phase[b] != cfg.eval_phase:
            continue
        lo = p if win is None else max(p, b - int(win))
        blocks = list(range(lo, b))
        if len(blocks) < cfg.min_train:
            continue
        tr = [(s, t) for s in range(S) for t in blocks]
        Xr = np.array([row(s, t, False) for s, t in tr])
        Xf = np.array([row(s, t, True) for s, t in tr])
        Xr_te = np.array([row(s, b, False) for s in range(S)])
        Xf_te = np.array([row(s, b, True) for s in range(S)])
        # ragged grids: a row is usable only if BOTH designs are finite, so
        # the restricted and full models are always fitted on the same rows
        okX = np.isfinite(Xr).all(axis=1) & np.isfinite(Xf).all(axis=1)
        okT = np.isfinite(Xr_te).all(axis=1) & np.isfinite(Xf_te).all(axis=1)
        for j in range(k):
            ytr = np.array([Y[target][s, t, j] for s, t in tr])
            yte = Y[target][:, b, j]
            m = okX & np.isfinite(ytr)
            te = okT & np.isfinite(yte)
            if m.sum() < Xf.shape[1] + 2 or not te.any():
                continue
            er = _fit_predict(Xr[m], ytr[m], Xr_te[te], cfg.lam) - yte[te]
            ef = _fit_predict(Xf[m], ytr[m], Xf_te[te], cfg.lam) - yte[te]
            em = float(ytr[m].mean()) - yte[te]
            for i, s in enumerate(np.flatnonzero(te)):
                rows.append({"bid": int(bids[b]), "b_index": b, "seq": int(s),
                             "feature": j,
                             "phase": None if phase is None else phase[b],
                             "n_train": int(m.sum()), "e_res": float(er[i]) ** 2,
                             "e_full": float(ef[i]) ** 2,
                             "e_mean": float(em[i]) ** 2})
    return pd.DataFrame(rows)


def _gain(err: pd.DataFrame) -> dict:
    if not len(err):
        return {"G": np.nan, "r2_restricted": np.nan, "n_eval": 0}
    sr, sf, sm = err["e_res"].sum(), err["e_full"].sum(), err["e_mean"].sum()
    return {"G": float(1 - sf / sr) if sr > 1e-24 else np.nan,
            "r2_restricted": float(1 - sr / sm) if sm > 1e-24 else np.nan,
            "n_eval": int(err["bid"].nunique()),
            "sse_res": float(sr), "sse_full": float(sf)}


# --------------------------------------------------------------------------
# the public estimators
# --------------------------------------------------------------------------
def transfer_gain(phi: pd.DataFrame, target: str, source: str,
                  cfg: TransferCfg = TransferCfg()) -> dict:
    """G_{source->target} with its AR-R^2 diagnostic."""
    P = panel(phi, cfg.feats)
    err = _errors(P, target, source, cfg)
    out = _gain(err)
    out.update(target=target, source=source, n_feats=len(P["feats"]),
               cfg=cfg.label())
    return out


def gain_matrix(phi: pd.DataFrame, cfg: TransferCfg = TransferCfg()
                ) -> pd.DataFrame:
    """Every ordered pair. One row per (source, target)."""
    P = panel(phi, cfg.feats)
    rows = []
    for d in P["clients"]:
        for c in P["clients"]:
            if c == d:
                continue
            err = _errors(P, d, c, cfg)
            rows.append({"source": c, "target": d, **_gain(err)})
    return pd.DataFrame(rows)


def asymmetry(gm: pd.DataFrame) -> pd.DataFrame:
    """Per unordered pair: both directions and their difference."""
    rows = []
    seen = set()
    idx = gm.set_index(["source", "target"])
    for (c, d) in idx.index:
        key = tuple(sorted((c, d)))
        if key in seen:
            continue
        seen.add(key)
        a, b = key
        try:
            g_ab = float(idx.loc[(a, b), "G"])
            g_ba = float(idx.loc[(b, a), "G"])
        except KeyError:
            continue
        rows.append({"client_a": a, "client_b": b, "G_a_to_b": g_ab,
                     "G_b_to_a": g_ba, "asym": g_ab - g_ba,
                     "direction": a + "->" + b if g_ab > g_ba else b + "->" + a})
    return pd.DataFrame(rows)


def capacity_null(phi: pd.DataFrame, target: str, source: str,
                  cfg: TransferCfg = TransferCfg(), n_surr: int = 200,
                  kind: str = "permute", seed: int = 0) -> dict:
    """G against a dimension-matched surrogate source.

    `kind`: `permute` (shuffle the source's blocks), `shift` (circular
    rotation, which preserves the source's own autocorrelation), `client`
    (substitute each other client in turn, cycling).
    """
    P = panel(phi, cfg.feats)
    obs = _gain(_errors(P, target, source, cfg))
    rng = np.random.default_rng(seed)
    S = P["Y"][source]
    pool = [c for c in P["clients"] if c not in (target, source)]
    vals = []
    for i in range(int(n_surr)):
        nb = S.shape[1]
        if kind == "permute":
            Sx = S[:, rng.permutation(nb)]          # whole blocks, replicates attached
        elif kind == "shift":
            Sx = np.roll(S, int(rng.integers(1, max(2, nb))), axis=1)
        elif kind == "client":
            if not pool:
                break
            Sx = P["Y"][pool[i % len(pool)]]
            Sx = np.roll(Sx, int(rng.integers(1, max(2, nb))), axis=1)
        else:
            raise KeyError(f"unknown surrogate kind {kind!r}")
        vals.append(_gain(_errors(P, target, source, cfg,
                                  src_override=Sx))["G"])
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    G = obs["G"]
    sd = float(v.std(ddof=1)) if len(v) > 2 else np.nan
    return {"source": source, "target": target, "G": G,
            "r2_restricted": obs["r2_restricted"], "kind": kind,
            "n_surr": int(len(v)), "null_mean": float(v.mean()) if len(v) else np.nan,
            "null_sd": sd, "null_q95": float(np.quantile(v, 0.95)) if len(v) else np.nan,
            "z": float((G - v.mean()) / sd) if len(v) > 2 and sd > 1e-12 else np.nan,
            "p_emp": float((1 + np.sum(v >= G)) / (1 + len(v))) if len(v) else np.nan}


def null_matrix(phi: pd.DataFrame, cfg: TransferCfg = TransferCfg(),
                n_surr: int = 100, kind: str = "permute", seed: int = 0
                ) -> pd.DataFrame:
    """`capacity_null` for every ordered pair."""
    P = panel(phi, cfg.feats)
    rows = []
    for i, d in enumerate(P["clients"]):
        for j, c in enumerate(P["clients"]):
            if c == d:
                continue
            rows.append(capacity_null(phi, d, c, cfg, n_surr, kind,
                                      seed + 97 * i + j))
    return pd.DataFrame(rows)


def rolling_gain(phi: pd.DataFrame, target: str, source: str,
                 cfg: TransferCfg = TransferCfg(), win: int = 8,
                 smooth: int = 3) -> pd.DataFrame:
    """G(b) on a sliding training window, pooled over `smooth` blocks."""
    P = panel(phi, cfg.feats)
    err = _errors(P, target, source, cfg, win=win)
    if not len(err):
        return pd.DataFrame()
    agg = (err.groupby(["bid", "b_index", "phase"], dropna=False)
           [["e_res", "e_full"]].sum().reset_index().sort_values("b_index"))
    # pooled over replicates and features: S x k error terms per block
    r = agg["e_res"].rolling(int(smooth), min_periods=1).sum()
    f = agg["e_full"].rolling(int(smooth), min_periods=1).sum()
    agg["G"] = 1 - (f / r.where(r > 1e-24))
    agg["source"], agg["target"], agg["win"] = source, target, int(win)
    return agg


def onset_test(gb: pd.DataFrame, ramp_blocks) -> dict:
    """Where does G(b) peak, and how surprising is that under a uniform null?"""
    d = gb.dropna(subset=["G"])
    if not len(d):
        return {"peak_bid": None, "in_ramp": None, "p_uniform": np.nan,
                "n_eval": 0, "n_ramp": 0}
    ramp = set(int(b) for b in (ramp_blocks or []))
    peak = int(d.loc[d["G"].idxmax(), "bid"])
    n_eval = int(d["bid"].nunique())
    n_ramp = int(sum(1 for b in d["bid"].unique() if int(b) in ramp))
    return {"peak_bid": peak, "in_ramp": peak in ramp,
            "G_peak": float(d["G"].max()), "n_eval": n_eval, "n_ramp": n_ramp,
            "p_uniform": float(n_ramp / n_eval) if n_eval else np.nan}


def arm_table(phis: dict, cfg: TransferCfg = TransferCfg()) -> pd.DataFrame:
    """The pure / mixed / all arms side by side.

    `phis` is `{"pure": phi_pure, "mixed": phi_mixed, "all": phi_all}`.
    `G_mixed - G_pure` is the hydraulic-exposure claim; `G_all` against
    `max(G_pure, G_mixed)` says whether the joint encoding is mixing the two
    signals destructively, which is a statement about the AER and not about
    dependence.
    """
    frames = []
    for arm, phi in phis.items():
        gm = gain_matrix(phi, cfg)
        gm.insert(0, "arm", arm)
        frames.append(gm)
    out = pd.concat(frames, ignore_index=True)
    wide = out.pivot_table(index=["source", "target"], columns="arm",
                           values="G")
    if {"pure", "mixed"} <= set(wide.columns):
        wide["exposure(mixed-pure)"] = wide["mixed"] - wide["pure"]
    if "all" in wide.columns and {"pure", "mixed"} <= set(wide.columns):
        wide["all_minus_best"] = wide["all"] - wide[["pure", "mixed"]].max(axis=1)
    return wide.reset_index()
