"""Prototype-only dependence / attribution / causality battery over `fed_sandbox`.

Scope: everything here reads **only** the prototype tables written by the federated
sandbox (`prototype_history`, optionally `global_prototype_history`). No latents, no
raw sensors, no model weights. Runs over every `(world_id, run_id)` found in the cache.

Design commitments (see the notebook §0 for the reasoning):

* **Rotation invariance.** The encoder is defined only up to rotation and the sandbox
  plots use a per-round PCA fit, so raw coordinates are not comparable across rounds
  or runs. Every cross-round / cross-run statistic here is computed either on the
  5x5 relational (distance / Gram) structure or on within-round residuals, both of
  which are invariant to a common orthogonal change of basis. `basis_drift` measures
  how much rotation is actually present, so the choice can be justified rather than
  asserted.
* **Common seasonal driver.** All districts share the month index and the seasonal
  multiplier, so two independent clients are trivially mutually "causal". The
  cross-client mean (common mode) is removed before any lead-lag or Granger test. For
  an ordered pair (i, j) the common mode is estimated from the OTHER clients only
  (`common_mode_exclude`), so the removal cannot manufacture or destroy coupling
  between exactly the two clients being tested.
* **Federated coupling.** FedAvg + the global prototype anchor couple all clients
  every round through a complete graph. `representation="dev_global"` expresses each
  local prototype as a deviation from the same-month global prototype; comparing the
  two representations bounds how much of any detected dependence is protocol rather
  than hydraulics. A `proto_alpha=0` run in the cache is the stronger control.
* **Nulls.** Circular-shift surrogates (reusing `fed_methods.roll_pvalue`) for
  everything time-indexed: they preserve each series' autocorrelation and destroy only
  the cross-relation. Exact enumeration over the 5! label assignments for the regime
  tests. In-sample statistics are kept in-sample and calibrated by the surrogate
  distribution, which carries the same optimism.

Reuses the repo's estimators (`dependence_detection.fed_methods` as F,
`dependence_oracle.methods` as M) rather than reimplementing them.
"""
from __future__ import annotations

import itertools
import json
import re
import warnings
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from fedwater.pipelines.dependence_detection import fed_methods as F
from fedwater.pipelines.dependence_oracle import methods as M

# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------
CFG = dict(
    representation="resid",      # raw | centered | resid | dev_global[_centered]
    baseline_months=6,           # fallback pre-drift window when onset is unknown
    metric="cosine",             # relational distance metric
    common_mode="geomedian",     # geomedian | mean | median  (see geometric_median)
    n_components=2,              # dim reduction for lead-lag / Granger / transfer
    difference=True,             # first-difference the series used by modules C/D/E
    lags=2,                      # VAR order
    max_lag=4,                   # lead-lag search range (months)
    ridge=1e-2,
    train_frac=0.7,              # out-of-sample split for predictive transfer
    n_surrogates=299,
    n_surrogates_expensive=99,
    alpha=0.05,
    seed=0,
)

def geometric_median(X: np.ndarray, iters=64, tol=1e-9) -> np.ndarray:
    """Weiszfeld geometric median of the rows of X.

    Why not the plain mean: subtracting the cross-client MEAN injects -1/N of the
    epicenter's own drift into every other client's residual, so every client appears
    to change at the epicenter's onset month and the onset ORDERING — the statistic
    that carries the propagation delay — is destroyed. Why not the per-coordinate
    median: it is robust but basis-dependent, which would forfeit the rotation
    invariance the rest of the module relies on. The geometric median minimizes the
    sum of Euclidean distances, so it is both robust to one outlying client and
    equivariant under rotation.
    """
    y = X.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(X - y, axis=1)
        if (d < tol).any():
            return X[np.argmin(d)]
        w = 1.0 / d
        y_new = (w[:, None] * X).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


LEVELS = {"L": 0, "M": 1, "H": 2}
SECTORS = {"R": "residential", "M": "mixed", "C": "commercial", "I": "industrial"}


# ==========================================================================
# 1. discovery
# ==========================================================================
def parse_world_id(world_id: str) -> dict:
    """`<letter>_<target_token>__<tokA>_<tokB>_<tokC>_<tokD>_<tokE>[_<hash>]`.

    Tolerant: anything it cannot parse comes back as None instead of raising, so a
    differently-named world still flows through the battery (minus the regime tests).
    """
    out = dict(world_id=world_id, drift_letter=None, drift_token=None,
               consumption_map=None, tokens=None)
    head, _, tail = world_id.partition("__")
    toks = [t for t in tail.split("_") if re.fullmatch(r"[LMH][RMCI]", t)]
    if toks:
        out["tokens"] = toks[:5]
        out["consumption_map"] = "_".join(toks[:5])
    hm = re.match(r"^([A-E])_([LMH][RMCI])$", head)
    if hm:
        out["drift_letter"], out["drift_token"] = hm.group(1), hm.group(2)
    return out


def world_manifest(world_id: str, repo_root: Path | None) -> dict:
    if repo_root is None:
        return {}
    p = repo_root / "data" / "09_experiments" / "worlds" / world_id / "manifest.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _read_table(base: Path) -> pd.DataFrame | None:
    for suf in (".parquet", ".csv.gz", ".csv"):
        p = base.with_suffix(suf)
        if p.exists():
            return pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p)
    return None


def list_fed_runs(sandbox: Path, repo_root: Path | None = None,
                  onset_default: int | None = None) -> pd.DataFrame:
    """Every `(world_id, run_id)` under `sandbox` that carries a prototype history.

    Ground truth (drift district, onset month) is resolved in this order: the world's
    `manifest.json` -> the world_id token -> `onset_default`. `onset_source` records
    which one won, because a silently-defaulted onset changes the baseline window and
    therefore every downstream statistic.
    """
    sandbox = Path(sandbox)
    rows = []
    for wdir in sorted(p for p in sandbox.iterdir() if p.is_dir()):
        meta = parse_world_id(wdir.name)
        man = world_manifest(wdir.name, repo_root)
        drift = (((man.get("effective") or {}).get("scenario") or {}).get("drift") or {})
        district = drift.get("tgt_district")
        warm = drift.get("warmup_months")
        onset = int(warm) + 1 if warm is not None else None
        src = "manifest"
        if district is None and meta["drift_letter"]:
            district, src = f"District_{meta['drift_letter']}", "world_id"
        if onset is None:
            onset, src = onset_default, f"{src}+default_onset"
        for rdir in sorted(p for p in wdir.iterdir() if p.is_dir()):
            ph = _read_table(rdir / "prototype_history")
            if ph is None:
                continue
            fcols = _fcols(ph)
            rows.append(dict(
                world_id=wdir.name, run_id=rdir.name, path=str(rdir),
                drift_district=district, drift_onset=onset, onset_source=src,
                drift_token=meta["drift_token"], consumption_map=meta["consumption_map"],
                n_clients=ph["client"].nunique(), n_months=ph["month"].nunique(),
                n_rounds=ph["round"].nunique(), latent_dim=len(fcols),
                has_global=(_read_table(rdir / "global_prototype_history") is not None),
                fl_signature=re.sub(r"^\d{8}-\d{6}_", "", rdir.name)))
    return pd.DataFrame(rows)


def _fcols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"f\d+", c)]
    return sorted(cols, key=lambda c: int(c[1:]))


# ==========================================================================
# 2. representation
# ==========================================================================
class ProtoRun:
    """Prototype tensor for one run plus the four representations.

    `P[r, c, m, :]` is client `c`'s month-`m` prototype at round `r`.
    """

    def __init__(self, path, world_id, run_id, drift_district=None, drift_onset=None,
                 tokens=None, cfg=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.path, self.world_id, self.run_id = Path(path), world_id, run_id
        self.drift_district, self.drift_onset = drift_district, drift_onset
        ph = _read_table(self.path / "prototype_history")
        if ph is None:
            raise FileNotFoundError(f"no prototype_history in {path}")
        self.fcols = _fcols(ph)
        self.clients = sorted(ph["client"].unique())
        self.rounds = sorted(ph["round"].unique())
        # months present for every client at every round — a ragged grid would make
        # "trajectory" ill-defined, so intersect rather than pad
        cnt = ph.groupby("month")[["client", "round"]].nunique()
        full = (cnt["client"] == len(self.clients)) & (cnt["round"] == len(self.rounds))
        self.months = sorted(cnt.index[full])
        dropped = sorted(set(ph["month"].unique()) - set(self.months))
        if dropped:
            warnings.warn(f"{world_id}/{run_id}: dropped ragged months {dropped}")
        idx = {("round", r): i for i, r in enumerate(self.rounds)}
        self.P = np.full((len(self.rounds), len(self.clients), len(self.months),
                          len(self.fcols)), np.nan)

        cmap = {c: i for i, c in enumerate(self.clients)}
        mmap = {m: i for i, m in enumerate(self.months)}
        sub = ph[ph["month"].isin(self.months)].copy()
        sub["ri"] = sub["round"].map({r: i for i, r in enumerate(self.rounds)})
        sub["ci"] = sub["client"].map(cmap)
        sub["mi"] = sub["month"].map(mmap)
        self.P[sub["ri"].to_numpy(), sub["ci"].to_numpy(), sub["mi"].to_numpy(), :] = \
            sub[self.fcols].to_numpy()
        if not np.isfinite(self.P).all():
            raise AssertionError(f"{world_id}/{run_id}: incomplete prototype grid")

        gp = _read_table(self.path / "global_prototype_history")
        self.G = self._global_tensor(gp)

        # tokens: explicit argument wins, else parsed from the world_id
        self.tokens = tokens or parse_world_id(world_id)["tokens"]
        self.labels = (dict(zip(self.clients, self.tokens))
                       if self.tokens and len(self.tokens) == len(self.clients) else None)
        b = self.cfg["baseline_months"]
        onset_idx = (mmap.get(self.drift_onset) if self.drift_onset is not None else None)
        self.baseline = list(range(onset_idx if onset_idx else min(b, len(self.months) - 2)))
        if len(self.baseline) < 2:
            self.baseline = list(range(max(2, len(self.months) // 4)))

    # ---- global prototype -------------------------------------------------
    def _global_tensor(self, gp):
        """(R, M, D). Uses `global_prototype_history` when present (mean over clusters
        if the table is per-cluster), else the cross-client mean of local prototypes —
        which is what the server's `mean_proto` reduces to for a single cluster."""
        if gp is None:
            self.global_source = "cross_client_mean"
            return self.P.mean(axis=1)
        fc = _fcols(gp)
        self.global_source = "global_prototype_history"
        Gm = np.full((len(self.rounds), len(self.months), len(fc)), np.nan)
        rmap = {r: i for i, r in enumerate(self.rounds)}
        mmap = {m: i for i, m in enumerate(self.months)}
        agg = gp[gp["month"].isin(self.months)].groupby(["round", "month"])[fc].mean()
        ri = agg.index.get_level_values("round").map(rmap)
        mi = agg.index.get_level_values("month").map(mmap)
        valid = ri.notna() & mi.notna()
        Gm[ri[valid].to_numpy(int), mi[valid].to_numpy(int), :] = agg.to_numpy()[valid]

        if not np.isfinite(Gm).all():
            self.global_source = "cross_client_mean(fallback)"
            return self.P.mean(axis=1)
        return Gm

    # ---- representations --------------------------------------------------
    def round_index(self, which="last") -> int:
        return {"last": len(self.rounds) - 1, "first": 0}.get(which, which)

    def matrix(self, which="last", representation=None, exclude=()) -> np.ndarray:
        """(C, M, D) in the requested representation.

        raw        — as stored (carries the district fingerprint offset)
        centered   — minus each client's own baseline-month mean (kills the
                     fingerprint; also kills static level/similarity information)
        resid      — centered minus the common mode across clients (kills the shared
                     seasonal driver); `exclude` holds out clients from the common-mode
                     estimate so a pairwise test is not contaminated by its own members
        dev_global — raw minus the same-month global prototype (removes the FedAvg /
                     anchor common component instead of the empirical mean). Note it
                     does NOT baseline-center, so the district fingerprint survives and
                     displacement magnitudes are dominated by distance-from-global
                     rather than by drift: use it to check whether a pair statistic is
                     protocol-driven, not as an attribution representation.
        dev_global_centered — the same, then baseline-centered: the FL-coupling control
                     that still supports onset detection and attribution.
        """
        rep = representation or self.cfg["representation"]
        r = self.round_index(which)
        X = self.P[r].copy()
        if rep == "raw":
            return X
        if rep == "dev_global":
            return X - self.G[r][None, :, :]
        if rep == "dev_global_centered":
            X = X - self.G[r][None, :, :]
            return X - X[:, self.baseline, :].mean(axis=1, keepdims=True)
        X = X - X[:, self.baseline, :].mean(axis=1, keepdims=True)
        if rep == "centered":
            return X
        if rep == "resid":
            keep = [i for i, c in enumerate(self.clients) if c not in set(exclude)]
            cm = self.cfg["common_mode"]
            if cm == "mean":
                mode = X[keep].mean(axis=0)
            elif cm == "median":
                mode = np.median(X[keep], axis=0)
            else:
                mode = np.stack([geometric_median(X[keep, m, :])
                                 for m in range(X.shape[1])])
            return X - mode[None, :, :]
        raise ValueError(rep)

    def scalar(self, which="last", exclude=(), mode="norm",
               difference=None) -> np.ndarray:
        """(C, M) drift signal.

        `norm` — displacement magnitude, kept in LEVELS: onset detection needs the
        step to stay a step, and differencing would turn the change point into a
        single spike that a two-segment mean fit cannot see.
        `pc1`  — signed leading component per client, DIFFERENCED by default: a drift
        response is a monotone ramp, and two ramps correlate at essentially every lag,
        so lead-lag on levels returns an arbitrary argmax (observed: wrong-signed lags
        even with a correct ground truth). The derivative of a ramp is a bump localized
        at the onset, which is what actually identifies the delay.
        """
        X = self.matrix(which, exclude=exclude)
        if mode == "norm":
            return np.linalg.norm(X, axis=2)
        Z = np.stack([F.first_pc(X[i]) for i in range(X.shape[0])])
        d = self.cfg["difference"] if difference is None else difference
        return np.diff(Z, axis=1) if d else Z

    def reduced(self, which="last", exclude=(), k=None,
                difference=None) -> np.ndarray:
        """(C, M', k) per-client SVD reduction, first-differenced by default.

        Differencing is also what keeps the VAR out of spurious-regression territory:
        on levels, two independent ramps are mutually "Granger causal" because each is
        predictable from its own trend and the trends overlap.
        """
        k = k or self.cfg["n_components"]
        X = self.matrix(which, exclude=exclude)
        d = self.cfg["difference"] if difference is None else difference
        if d:
            X = np.diff(X, axis=1)
        out = []
        for i in range(X.shape[0]):
            A = X[i] - X[i].mean(0)
            _, _, vt = np.linalg.svd(A, full_matrices=False)
            out.append(A @ vt[:k].T)
        return np.stack(out)

    def key(self) -> dict:
        return dict(world_id=self.world_id, run_id=self.run_id)


# ==========================================================================
# 3. diagnostics — must pass before any causal number is read
# ==========================================================================
def basis_drift(run: ProtoRun) -> pd.DataFrame:
    """Procrustes disparity between consecutive rounds' prototype configurations.

    Large values mean the latent basis rotates between rounds, i.e. raw-coordinate
    statistics compared across rounds are partly measuring the basis, not the data.
    """
    rows = []
    for a, b in zip(run.rounds[:-1], run.rounds[1:]):
        Xa = run.P[run.rounds.index(a)].reshape(-1, len(run.fcols))
        Xb = run.P[run.rounds.index(b)].reshape(-1, len(run.fcols))
        _, _, disp = procrustes(Xa, Xb)
        rows.append(dict(**run.key(), round_from=a, round_to=b,
                         procrustes_disparity=float(disp)))
    return pd.DataFrame(rows)


def diagnostics(run: ProtoRun) -> pd.DataFrame:
    C = run.matrix(representation="centered")
    R = run.matrix(representation="resid")
    common = 1 - float(R.var()) / float(C.var()) if C.var() > 0 else np.nan
    bd = basis_drift(run)
    fin = run.matrix("last", "raw")
    return pd.DataFrame([dict(
        **run.key(), n_clients=len(run.clients), n_months=len(run.months),
        n_rounds=len(run.rounds), latent_dim=len(run.fcols),
        drift_district=run.drift_district, drift_onset=run.drift_onset,
        baseline_months=len(run.baseline), global_source=run.global_source,
        common_mode_var_share=common,
        procrustes_mean=float(bd["procrustes_disparity"].mean()),
        procrustes_last=float(bd["procrustes_disparity"].iloc[-1]),
        fingerprint_ratio=float(
            np.linalg.norm(fin.mean(1) - fin.mean((0, 1))) / (np.linalg.norm(
                fin - fin.mean(1, keepdims=True)) + 1e-12)),
        obs_per_pair=len(run.months) - run.cfg["lags"])])


def add_q(df: pd.DataFrame) -> pd.DataFrame:
    """BH q-values within each family — one run's pair table is one family.

    Applied by every module that returns a pair table, so a single-run inspection and
    the pooled batch carry the same column. Idempotent.
    """
    if df is None or not len(df) or "p_value" not in df:
        return df
    keys = [c for c in ("world_id", "run_id", "module", "conditional", "test",
                        "grouping", "round") if c in df]
    df = df.copy()
    df["q_value"] = np.nan
    for _, idx in (df.groupby(keys, dropna=False).groups.items() if keys
                   else [(None, df.index)]):
        sub = df.loc[idx, "p_value"]
        ok = sub.notna()
        if ok.any():
            df.loc[sub[ok].index, "q_value"] = F.bh_fdr(sub[ok].to_numpy())
    return df


# ==========================================================================
# 4. module A — relational geometry and regime similarity
# ==========================================================================
def distance_matrix(X: np.ndarray, months, metric="cosine") -> np.ndarray:
    """(C, C) mean pairwise distance between client prototypes over `months`."""
    V = X[:, months, :].reshape(X.shape[0], -1)
    if metric == "cosine":
        n = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        return 1 - n @ n.T
    return np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)


def _n_within(labels) -> int:
    lab = np.asarray(labels)
    iu = np.triu_indices(len(lab), 1)
    return int((lab[iu[0]] == lab[iu[1]]).sum())


def _group_contrast(D: np.ndarray, labels: list) -> float:
    """mean(between) - mean(within); >0 means same-label clients sit closer."""
    lab = np.asarray(labels)
    iu = np.triu_indices(len(lab), 1)
    same = lab[iu[0]] == lab[iu[1]]
    if same.all() or (~same).all():
        return np.nan
    return float(D[iu][~same].mean() - D[iu][same].mean())


def _partition_canon(labels):
    """Canonical form of a label vector for a statistic that sees only the grouping.

    A clustering contrast cannot distinguish "B is commercial, C is industrial" from
    the reverse: both make B and C singletons. Enumerating label assignments therefore
    double-counts, and the reference distribution has to be over distinct PARTITIONS.
    """
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    return frozenset(frozenset(v) for v in groups.values())


def _exact_label_null(D, labels, stat_fn, canon=_partition_canon):
    """Exact p over the distinct reference assignments, enumerated exhaustively.

    `canon` collapses assignments the statistic cannot tell apart (see
    `_partition_canon`); the observed assignment is itself in the reference set, so
    p >= 1 / n_null. That floor is the binding constraint at N = 5 districts: a 3/1/1
    regime split yields only C(5,3) = 10 distinct partitions, so the clustering
    contrast cannot return p < 0.10 however clean the geometry is. The returned
    `p_floor` makes that visible, and it is the main argument for reading the
    statistic's sign and magnitude, for the ordinal test (finer null), and for pooling
    across runs rather than chasing significance within one.
    """
    obs = stat_fn(D, list(labels))
    if not np.isfinite(obs):
        return obs, np.nan, 0, np.nan
    seen, null = set(), []
    for a in itertools.permutations(labels):
        key = canon(a)
        if key in seen:
            continue
        seen.add(key)
        v = stat_fn(D, list(a))
        if np.isfinite(v):
            null.append(v)
    null = np.asarray(null)
    if not len(null):
        return float(obs), np.nan, 0, np.nan
    return float(obs), float(np.mean(null >= obs)), len(null), float(1 / len(null))


def _ordinal_stat(D, levels):
    """Spearman between prototype distance and |level difference| — tests whether the
    geometry is consistent with an ordinal L < M < H layout rather than mere clustering.
    """
    lv = np.asarray([LEVELS[x] for x in levels], float)
    iu = np.triu_indices(len(lv), 1)
    gap = np.abs(lv[iu[0]] - lv[iu[1]])
    if len(np.unique(gap)) < 2:
        return np.nan
    return float(spearmanr(D[iu], gap).statistic)


def regime_geometry(run: ProtoRun, months=None, representation="raw") -> pd.DataFrame:
    """Static similarity: do same-regime districts co-locate, pre-drift?

    Run at the first and last round: if the contrast is no better at the end, that is
    the "regime legibility does not improve with training" result stated as a number
    rather than read off a PCA scatter.
    """
    if run.labels is None:
        return pd.DataFrame()
    months = list(months if months is not None else run.baseline)
    toks = [run.labels[c] for c in run.clients]
    rows = []
    for which in ("first", "last"):
        D = distance_matrix(run.matrix(which, representation), months,
                            run.cfg["metric"])
        for name, labels in (("token", toks), ("sector", [t[1] for t in toks]),
                             ("level", [t[0] for t in toks])):
            s, p, n, floor = _exact_label_null(D, labels, _group_contrast)
            rows.append(dict(**run.key(), module="regime_geometry", round=which,
                             representation=representation, grouping=name,
                             test="group_contrast", statistic=s, p_value=p, n_null=n,
                             n_within_pairs=_n_within(labels), p_floor=floor))
        lv = [t[0] for t in toks]
        if len(set(lv)) > 1:
            # the ordinal statistic uses the level GAPS, so swapping two distinct
            # levels changes it: its reference set is the distinct label vectors, a
            # finer null than the partition one and hence a lower p floor
            s, p, n, floor = _exact_label_null(D, lv, _ordinal_stat, canon=tuple)
            rows.append(dict(**run.key(), module="regime_geometry", round=which,
                             representation=representation, grouping="level",
                             test="ordinal_spearman", statistic=s, p_value=p, n_null=n,
                             n_within_pairs=_n_within(lv), p_floor=floor))
    return add_q(pd.DataFrame(rows))


def regime_pair_table(run: ProtoRun, months=None, representation="raw") -> pd.DataFrame:
    """Per-pair distances with a same/different-regime flag — the table behind the
    "E should be near A and D" claim."""
    months = list(months if months is not None else run.baseline)
    D = distance_matrix(run.matrix("last", representation), months, run.cfg["metric"])
    rows = []
    for i, a in enumerate(run.clients):
        for j, b in enumerate(run.clients):
            if j <= i:
                continue
            ta = run.labels.get(a) if run.labels else None
            tb = run.labels.get(b) if run.labels else None
            rows.append(dict(**run.key(), district_a=a, district_b=b,
                             token_a=ta, token_b=tb, distance=float(D[i, j]),
                             same_token=(ta == tb) if ta and tb else None))
    return pd.DataFrame(rows)


# ==========================================================================
# 5. module B — onset ordering and drift attribution
# ==========================================================================
def _two_segment(s: np.ndarray, tau_min=2):
    """Exhaustive single change-point mean-shift fit. -> (tau, delta, r2)."""
    n = len(s)
    hi = n - tau_min
    if hi <= tau_min:
        return np.nan, np.nan, np.nan
    best = (np.inf, None)
    for tau in range(tau_min, hi):
        a, b = s[:tau], s[tau:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if sse < best[0]:
            best = (sse, tau)
    sse, tau = best
    tot = float(((s - s.mean()) ** 2).sum())
    return tau, float(s[tau:].mean() - s[:tau].mean()), (1 - sse / tot) if tot > 0 else np.nan


def onset_table(run: ProtoRun, which="last") -> pd.DataFrame:
    """Per-client change point in the common-mode-removed displacement magnitude.

    p-value: permute the month order (999 draws) and keep the best r2 — tests "there
    is a change point at all" against a no-structure null, which is the right question
    for a statistic that is itself an argmin over tau.
    """
    rng = np.random.default_rng([run.cfg["seed"], 7])
    S = run.scalar(which, mode="norm")
    rows = []
    for i, c in enumerate(run.clients):
        s = S[i]
        tau, delta, r2 = _two_segment(s)
        null = np.empty(999)
        for k in range(999):
            null[k] = _two_segment(rng.permutation(s))[2]
        p = float((1 + np.sum(null >= r2)) / (len(null) + 1)) if np.isfinite(r2) else np.nan
        rows.append(dict(**run.key(), client=c,
                         tau_index=tau,
                         tau_month=(run.months[int(tau)] if tau is not None
                                    and np.isfinite(tau) else np.nan),
                         delta=delta, r2=r2, p_value=p,
                         is_epicenter=(c == run.drift_district)))
    return pd.DataFrame(rows)


def attribution(run: ProtoRun, onsets: pd.DataFrame) -> pd.DataFrame:
    """Which client is the drift source? Two estimators, both scored against truth.

    `argmax_delta`  — largest post-change level shift (magnitude of response)
    `argmin_tau`    — earliest change point among clients with a significant one
                      (timing; this is the estimator that tracks the BFS delay)

    `rank_of_truth` / `p_exact` = rank of the true epicenter under the estimator's
    ordering; p = rank / n_clients is exact under a uniform-guess null.
    """
    if onsets.empty:
        return pd.DataFrame()
    o = onsets.set_index("client")
    truth = run.drift_district
    rows = []
    # rank ALL clients under both estimators: restricting argmin_tau to the
    # significant subset would make rank_of_truth = 1 whenever the epicenter is the
    # only client with a detectable change point, which is not evidence of ordering.
    # Non-significant clients are pushed to the back and tie-broken on |delta|.
    tau_eff = o[["tau_index", "delta", "p_value"]].copy()
    tau_eff["blocked"] = (tau_eff["p_value"] >= run.cfg["alpha"]).astype(int)
    tau_eff["tau_index"] = tau_eff["tau_index"].fillna(np.inf)
    tau_order = tau_eff.assign(neg_delta=-tau_eff["delta"].abs()).sort_values(
        ["blocked", "tau_index", "neg_delta"], kind="mergesort")["tau_index"]
    for name, series, ascending in (
            ("argmax_delta", o["delta"].abs(), False),
            ("argmin_tau", tau_order, True)):
        order = series if name == "argmin_tau" else series.sort_values(ascending=ascending)
        guess = order.index[0] if len(order) else None
        rank = (list(order.index).index(truth) + 1
                if truth in list(order.index) else np.nan)
        rows.append(dict(**run.key(), estimator=name, guess=guess, truth=truth,
                         hit=(guess == truth) if truth else None,
                         rank_of_truth=rank,
                         p_exact=(rank / len(run.clients) if np.isfinite(rank) else np.nan),
                         order="|".join(map(str, order.index))))
    return pd.DataFrame(rows)


def delay_structure(run: ProtoRun, onsets: pd.DataFrame,
                    hops: dict | None = None) -> pd.DataFrame:
    """Onset delay vs hop distance from the epicenter.

    The BFS diffusion ramp in the simulator *is* a monotone delay structure, so this is
    the statistic most directly aligned with the generative mechanism. Without a
    topology table, `hops` can be passed explicitly; otherwise only the epicenter-first
    check is available (see `attribution`).
    """
    if not hops or onsets.empty:
        return pd.DataFrame()
    o = onsets.set_index("client")
    ok = [c for c in run.clients if c in hops and np.isfinite(o.loc[c, "tau_index"])]
    if len(ok) < 4:
        return pd.DataFrame()
    tau = np.array([o.loc[c, "tau_index"] for c in ok], float)
    h = np.array([hops[c] for c in ok], float)
    rho = float(spearmanr(tau, h).statistic)
    perms = [float(spearmanr(tau, np.array(p, float)).statistic)
             for p in itertools.permutations(h)]
    p = float((1 + np.sum(np.asarray(perms) >= rho)) / (len(perms) + 1))
    return pd.DataFrame([dict(**run.key(), module="delay_structure",
                              test="tau_vs_hops_spearman", statistic=rho,
                              p_value=p, n_clients=len(ok))])


# ==========================================================================
# 6. module C — lead-lag on scalar drift signals
# ==========================================================================
def leadlag_matrix(run: ProtoRun, which="last") -> pd.DataFrame:
    """Max lagged cross-correlation per ordered pair, common mode from the other
    clients only. `lag > 0` means A leads B. Cheap and by far the best-powered
    directional instrument at ~30 months, but it cannot separate a direct path from
    one mediated by a third client.
    """
    rows = []
    for a, b in itertools.permutations(run.clients, 2):
        S = run.scalar(which, exclude=(a, b), mode="pc1")
        x = S[run.clients.index(a)]
        y = S[run.clients.index(b)]
        rng = np.random.default_rng([run.cfg["seed"], 11,
                                     zlib.crc32(f"{a}|{b}".encode())])
        (r, lag), p = F.roll_pvalue(
            lambda u, v: M.max_lagged_xcorr(u, v, run.cfg["max_lag"]),
            x, y, run.cfg["n_surrogates"], rng)
        rows.append(dict(**run.key(), module="leadlag", source=a, target=b,
                         statistic=float(r), lag=int(lag), p_value=p,
                         source_is_epicenter=(a == run.drift_district)))
    return add_q(pd.DataFrame(rows))


# ==========================================================================
# 7. module D — Granger causality on prototype trajectories
# ==========================================================================
def _ridge_sse(Xd: np.ndarray, Y: np.ndarray, lam: float) -> float:
    Xd = np.column_stack([Xd, np.ones(len(Xd))])
    A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
    W = np.linalg.solve(A, Xd.T @ Y)
    return float(((Y - Xd @ W) ** 2).sum())


def _lag_design(series_list, lags, n):
    """Stack lagged copies of each (n, k) array -> (n - lags, sum_k * lags)."""
    cols = []
    for S in series_list:
        for L in range(1, lags + 1):
            cols.append(S[lags - L: n - L])
    return np.column_stack(cols) if cols else np.zeros((n - lags, 0))


def granger_matrix(run: ProtoRun, which="last", conditional=False) -> pd.DataFrame:
    """Does adding the source's lags improve prediction of the target's next step
    *beyond the target's own lags*?

    The own-lag baseline is what makes this directional — a model that only checks
    whether the source predicts the target measures a shared driver, not causality.

    Statistic: in-sample proportional SSE reduction, `1 - SSE_full / SSE_own`.
    Kept in-sample deliberately: the null is generated by circularly shifting the
    source series, so the surrogate models carry exactly the same number of extra
    parameters and the same optimism. `conditional=True` also includes every other
    client's lags in both models, which removes transitive false positives (E->D->C
    showing up as E->C) at a serious cost in degrees of freedom: with ~30 months this
    arm is expected to be underpowered and should be read as a check on the pairwise
    arm, not a replacement.
    """
    cfg = run.cfg
    lags, k = cfg["lags"], cfg["n_components"]
    rows = []
    for a, b in itertools.permutations(run.clients, 2):
        Z = run.reduced(which, exclude=(a, b), k=k)
        ia, ib = run.clients.index(a), run.clients.index(b)
        n = Z.shape[1]
        Y = Z[ib][lags:]
        others = ([Z[i] for i in range(len(run.clients)) if i not in (ia, ib)]
                  if conditional else [])
        own = _lag_design([Z[ib]] + others, lags, n)
        n_obs, n_par = len(Y), own.shape[1] + k * lags + 1
        # 3 observations per parameter is already generous for a ridge VAR; below that
        # the SSE ratio is fitting noise and the surrogate null cannot rescue it
        if n_obs < 3 * n_par:
            rows.append(dict(**run.key(), module="granger", source=a, target=b,
                             conditional=conditional, statistic=np.nan, p_value=np.nan,
                             n_obs=n_obs, n_params=n_par,
                             dof_ratio=n_obs / n_par, underpowered=True,
                             source_is_epicenter=(a == run.drift_district)))
            continue

        def stat(Xsrc, _):
            full = np.column_stack([own, _lag_design([Xsrc], lags, n)])
            sse_own = _ridge_sse(own, Y, cfg["ridge"])
            sse_full = _ridge_sse(full, Y, cfg["ridge"])
            return 1 - sse_full / sse_own if sse_own > 0 else 0.0

        rng = np.random.default_rng([cfg["seed"], 13, zlib.crc32(f"{a}|{b}".encode())])
        s, p = F.roll_pvalue(stat, Z[ia], Y, cfg["n_surrogates_expensive"], rng)
        rows.append(dict(**run.key(), module="granger", source=a, target=b,
                         conditional=conditional, statistic=float(s), p_value=p,
                         n_obs=n_obs, n_params=n_par,
                         dof_ratio=n_obs / n_par, underpowered=False,
                         source_is_epicenter=(a == run.drift_district)))
    return add_q(pd.DataFrame(rows))


# ==========================================================================
# 8. module E — predictive transfer (out-of-sample form of the original idea)
# ==========================================================================
def predictive_transfer(run: ProtoRun, which="last") -> pd.DataFrame:
    """"Use the last L months of district A to predict district B's next position."

    Scored as the *increment* over a model using B's own last L months, on a held-out
    tail of the series (`train_frac`). Raw predictive accuracy would be high for every
    pair simply because prototypes are autocorrelated and seasonally driven; only
    delta_r2 > 0 says A carries information about B that B does not already carry
    about itself. Expect this to be noisy — the test split is ~8 months.
    """
    cfg = run.cfg
    lags, k = cfg["lags"], cfg["n_components"]
    rows = []
    for a, b in itertools.permutations(run.clients, 2):
        Z = run.reduced(which, exclude=(a, b), k=k)
        ia, ib = run.clients.index(a), run.clients.index(b)
        n = Z.shape[1]
        Y = Z[ib][lags:]
        own = _lag_design([Z[ib]], lags, n)
        split = int(cfg["train_frac"] * len(Y))
        if split < lags + 2 or len(Y) - split < 3:
            continue

        def r2(Xd):
            Xtr = np.column_stack([Xd[:split], np.ones(split)])
            Xte = np.column_stack([Xd[split:], np.ones(len(Y) - split)])
            A = Xtr.T @ Xtr + cfg["ridge"] * np.eye(Xtr.shape[1])
            W = np.linalg.solve(A, Xtr.T @ Y[:split])
            sse = float(((Y[split:] - Xte @ W) ** 2).sum())
            tot = float(((Y[split:] - Y[:split].mean(0)) ** 2).sum())
            return 1 - sse / tot if tot > 0 else np.nan

        def stat(Xsrc, _):
            return r2(np.column_stack([own, _lag_design([Xsrc], lags, n)])) - r2(own)

        rng = np.random.default_rng([cfg["seed"], 17, zlib.crc32(f"{a}|{b}".encode())])
        s, p = F.roll_pvalue(stat, Z[ia], Y, cfg["n_surrogates_expensive"], rng)
        rows.append(dict(**run.key(), module="predictive_transfer", source=a, target=b,
                         statistic=float(s), r2_own=float(r2(own)), p_value=p,
                         n_test=len(Y) - split,
                         source_is_epicenter=(a == run.drift_district)))
    return add_q(pd.DataFrame(rows))


# ==========================================================================
# 9. module F — matched-pair / quasi-interventional contrast
# ==========================================================================
def matched_pairs(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs sharing a consumption map and FL config but differing in drift target.

    Same demand structure, same hyperparameters, different intervention: the closest
    thing to an interventional contrast available in the existing cache. A true ATE
    needs matched drift-on / drift-off worlds at the same seed, which has to be
    simulated — `matched_displacement` is written so those slot straight in.
    """
    r = runs.dropna(subset=["consumption_map"])
    rows = []
    for (cmap, sig), g in r.groupby(["consumption_map", "fl_signature"]):
        if g["drift_district"].nunique() < 2:
            continue
        for i, j in itertools.combinations(range(len(g)), 2):
            A, B = g.iloc[i], g.iloc[j]
            if A["drift_district"] == B["drift_district"]:
                continue
            rows.append(dict(consumption_map=cmap, fl_signature=sig,
                             world_a=A["world_id"], run_a=A["run_id"],
                             world_b=B["world_id"], run_b=B["run_id"],
                             drift_a=A["drift_district"], drift_b=B["drift_district"]))
    return pd.DataFrame(rows)


def matched_displacement(pairs: pd.DataFrame, load) -> pd.DataFrame:
    """Per-client post-baseline displacement in each arm of a matched pair.

    `spillover` is the displacement of a client that is NOT the epicenter in either
    arm: it moved because someone else did, which is dependence measured against an
    external contrast rather than inferred from a single series.
    """
    rows = []
    for _, p in pairs.iterrows():
        try:
            A, B = load(p["world_a"], p["run_a"]), load(p["world_b"], p["run_b"])
        except Exception as exc:                                # noqa: BLE001
            warnings.warn(f"matched pair skipped: {exc}")
            continue
        for arm, run in (("a", A), ("b", B)):
            S = run.scalar("last", mode="norm")
            post = [i for i in range(len(run.months)) if i not in run.baseline]
            for ci, c in enumerate(run.clients):
                rows.append(dict(consumption_map=p["consumption_map"], arm=arm,
                                 world_id=run.world_id, run_id=run.run_id, client=c,
                                 displacement=float(S[ci, post].mean()),
                                 is_epicenter=(c == run.drift_district),
                                 spillover=(c != p["drift_a"] and c != p["drift_b"])))
    return pd.DataFrame(rows)


# ==========================================================================
# 10. runner
# ==========================================================================
def analyse_run(run: ProtoRun, hops: dict | None = None,
                modules=("A", "B", "C", "D", "E")) -> dict:
    out = {"diagnostics": diagnostics(run), "basis_drift": basis_drift(run)}
    if "A" in modules:
        out["regime_geometry"] = regime_geometry(run)
        out["regime_pairs"] = regime_pair_table(run)
    if "B" in modules:
        out["onsets"] = onset_table(run)
        out["attribution"] = attribution(run, out["onsets"])
        out["delay_structure"] = delay_structure(run, out["onsets"], hops)
    if "C" in modules:
        out["leadlag"] = leadlag_matrix(run)
    if "D" in modules:
        out["granger"] = pd.concat([granger_matrix(run, conditional=False),
                                    granger_matrix(run, conditional=True)],
                                   ignore_index=True)
    if "E" in modules:
        out["predictive_transfer"] = predictive_transfer(run)
    return out


def analyse_all(runs: pd.DataFrame, cfg=None, hops=None, modules=("A", "B", "C", "D", "E"),
                progress=True) -> dict:
    """Run the battery over every row of `runs`; concatenate per table, then FDR."""
    acc: dict[str, list] = {}
    for _, row in runs.iterrows():
        if progress:
            print(f"  {row['world_id']} / {row['run_id']}", flush=True)
        try:
            run = ProtoRun(row["path"], row["world_id"], row["run_id"],
                           row.get("drift_district"), row.get("drift_onset"), cfg=cfg)
        except Exception as exc:                                 # noqa: BLE001
            warnings.warn(f"{row['world_id']}/{row['run_id']} skipped: {exc}")
            continue
        for name, df in analyse_run(run, hops=(hops or {}).get(row["world_id"]),
                                    modules=modules).items():
            if df is not None and len(df):
                acc.setdefault(name, []).append(df)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    for name in ("leadlag", "granger", "predictive_transfer", "regime_geometry"):
        if name in tables:
            tables[name] = add_q(tables[name])
    return tables


# ==========================================================================
# 11. cross-run scoreboard
# ==========================================================================
def _source_scores(r: pd.DataFrame, module: str) -> pd.Series:
    """Per-source epicenter score for one run's pair table.

    For `leadlag` the score is the net LEADERSHIP, mean over targets of
    sign(lag) * |r|: a client that merely co-moves with everyone scores near zero,
    while one that consistently moves first scores high. Scoring by mean |r| alone
    (the obvious choice) throws away the lag sign, which is the only directional
    information the statistic carries — every client carries a delayed copy of the
    epicenter's bump, so mean |r| is high for all of them and ranks the epicenter at
    chance. For the regression modules the statistic is already directional.
    """
    if module == "leadlag":
        v = np.sign(r["lag"]) * r["statistic"].abs()
        return r.assign(v=v).groupby("source")["v"].mean()
    return r.assign(v=r["statistic"].abs()).groupby("source")["v"].mean()


def scoreboard(tables: dict, alpha=0.05) -> pd.DataFrame:
    """Does each method find the known epicenter, aggregated over all runs?

    Every world in the cache carries a known drift district, so "top-1 epicenter
    accuracy over N runs" is the one honest end-to-end score available without new
    simulations; `chance` is 1 / n_clients. `epicenter_lift` is the mean statistic for
    pairs sourced at the true epicenter minus the mean over all other pairs: positive
    means the method ranks the real source above the rest even when no single pair
    clears significance, which is the more sensitive read at this sample size.
    """
    rows = []
    att = tables.get("attribution")
    if att is not None and len(att):
        for est, g in att.groupby("estimator"):
            n_cl = int(g["order"].str.count(r"\|").max()) + 1
            rows.append(dict(module="attribution", method=est, n_runs=len(g),
                             top1_accuracy=float(g["hit"].mean()),
                             mean_rank_of_truth=float(g["rank_of_truth"].mean()),
                             n_runs_scored=len(g),
                             chance=1 / n_cl, epicenter_lift=np.nan,
                             frac_significant=np.nan, frac_underpowered=np.nan))
    for name in ("leadlag", "granger", "predictive_transfer"):
        t = tables.get(name)
        if t is None or not len(t):
            continue
        groups = (list(t.groupby("conditional")) if "conditional" in t
                  else [(None, t)])
        for key, g in groups:
            key = key[0] if isinstance(key, tuple) else key
            hits, ranks = [], []
            for _, r in g.groupby(["world_id", "run_id"]):
                r = r[r["statistic"].notna()]
                # an all-NaN run (every cell refused for want of degrees of freedom)
                # would otherwise be "ranked" from an arbitrary sort order
                if not len(r):
                    continue
                truth = r.loc[r["source_is_epicenter"], "source"].unique()
                agg = _source_scores(r, name).sort_values(ascending=False)
                if not len(truth) or truth[0] not in agg.index:
                    continue
                hits.append(agg.index[0] == truth[0])
                ranks.append(list(agg.index).index(truth[0]) + 1)
            ep = g.loc[g["source_is_epicenter"], "statistic"].abs()
            other = g.loc[~g["source_is_epicenter"], "statistic"].abs()
            rows.append(dict(
                module=name,
                method=name + ("|conditional" if key is True else ""),
                n_runs=g[["world_id", "run_id"]].drop_duplicates().shape[0],
                n_runs_scored=len(hits),
                top1_accuracy=float(np.mean(hits)) if hits else np.nan,
                mean_rank_of_truth=float(np.mean(ranks)) if ranks else np.nan,
                chance=1 / g["source"].nunique(),
                epicenter_lift=float(np.nanmean(ep) - np.nanmean(other)),
                frac_significant=(float((g["q_value"] < alpha).mean())
                                  if "q_value" in g else np.nan),
                frac_underpowered=(float(g["underpowered"].mean())
                                   if "underpowered" in g else np.nan)))
    return pd.DataFrame(rows)


def save_tables(tables: dict, out_dir) -> list[Path]:
    """Write every table to `out_dir` (parquet, csv.gz fallback)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        if df is None or not len(df):
            continue
        try:
            p = out / f"{name}.parquet"
            df.to_parquet(p, index=False)
        except Exception:                                        # noqa: BLE001
            p = out / f"{name}.csv.gz"
            df.to_csv(p, index=False)
        written.append(p)
    return written


def stress_grid(runs: pd.DataFrame, grid: list[dict], base=None,
                modules=("B", "C", "D", "E"), hops=None) -> pd.DataFrame:
    """Re-score the whole battery under each config in `grid`.

    A method that identifies the epicenter at one `(lags, n_components, difference,
    representation, common_mode)` setting and loses it at the next is not a finding,
    it is a coincidence. This is the stress test: the scoreboard is only worth reading
    where it is stable across the grid.
    """
    out = []
    for i, ov in enumerate(grid):
        cfg = {**CFG, **(base or {}), **ov}
        tables = analyse_all(runs, cfg=cfg, hops=hops, modules=modules, progress=False)
        sb = scoreboard(tables, alpha=cfg["alpha"])
        if len(sb):
            sb.insert(0, "config", "|".join(f"{k}={v}" for k, v in ov.items()) or "base")
            sb.insert(1, "config_idx", i)
            out.append(sb)
        print(f"  [{i + 1}/{len(grid)}] {ov or 'base'}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ==========================================================================
# 12. plots
# ==========================================================================
def plot_distance_heatmaps(run: ProtoRun, representation="raw", figsize=(11, 4)):
    pre, post = run.baseline, [i for i in range(len(run.months)) if i not in run.baseline]
    X = run.matrix("last", representation)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    Dp = distance_matrix(X, pre, run.cfg["metric"])
    Dq = distance_matrix(X, post, run.cfg["metric"])
    tick = [f"{c.split('_')[-1]}\n{run.labels[c] if run.labels else ''}"
            for c in run.clients]
    for ax, D, t in ((axes[0], Dp, "pre-drift"), (axes[1], Dq, "post-drift"),
                     (axes[2], Dq - Dp, "delta")):
        vm = np.abs(D).max()
        im = ax.imshow(D, cmap="RdBu_r" if t == "delta" else "viridis",
                       vmin=-vm if t == "delta" else None,
                       vmax=vm if t == "delta" else None)
        ax.set_xticks(range(len(tick)), tick, fontsize=7)
        ax.set_yticks(range(len(tick)), tick, fontsize=7)
        ax.set_title(f"{t}  ({run.cfg['metric']})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f"{run.world_id} / {run.run_id}  ·  drift={run.drift_district}",
                 fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_drift_signals(run: ProtoRun, onsets: pd.DataFrame | None = None,
                       figsize=(11, 4)):
    S = run.scalar("last", mode="norm")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, c in enumerate(run.clients):
        ep = c == run.drift_district
        axes[0].plot(run.months, S[i], lw=2.2 if ep else 1.1,
                     label=f"{c}{' *' if ep else ''}", alpha=1 if ep else .75)
    axes[0].axvline(run.months[len(run.baseline)], color="k", ls=":", lw=1,
                    label="baseline end")
    if onsets is not None and len(onsets):
        for _, r in onsets.iterrows():
            if np.isfinite(r["tau_index"]):
                axes[1].scatter(r["tau_month"], r["delta"],
                                s=140 if r["is_epicenter"] else 60,
                                marker="*" if r["is_epicenter"] else "o")
                axes[1].annotate(r["client"].split("_")[-1],
                                 (r["tau_month"], r["delta"]), fontsize=8,
                                 xytext=(4, 3), textcoords="offset points")
        axes[1].set_xlabel("estimated onset month  (tau)")
        axes[1].set_ylabel("level shift  (delta)")
        axes[1].axhline(0, color="k", lw=.6)
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("||residual displacement||")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run.world_id} / {run.run_id}", fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_matrix(df: pd.DataFrame, value="statistic", title="", alpha=0.05,
                figsize=(5.2, 4.4)):
    """Source x target heatmap; significant cells (q < alpha) are boxed."""
    piv = df.pivot_table(index="source", columns="target", values=value)
    fig, ax = plt.subplots(figsize=figsize)
    vm = np.nanmax(np.abs(piv.to_numpy())) or 1
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-vm, vmax=vm)
    ax.set_xticks(range(len(piv.columns)), [c.split("_")[-1] for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [c.split("_")[-1] for c in piv.index])
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    if "q_value" in df:
        q = df.pivot_table(index="source", columns="target", values="q_value")
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = q.to_numpy()[i, j]
                if np.isfinite(v) and v < alpha:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               ec="k", lw=2))
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    plt.show()


def plot_scoreboard(sb: pd.DataFrame, figsize=(8, 3.6)):
    if not len(sb):
        return
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sb))
    ax.bar(x, sb["top1_accuracy"], color="steelblue")
    for i, ch in enumerate(sb["chance"]):
        if np.isfinite(ch):
            ax.hlines(ch, i - .42, i + .42, color="crimson", ls="--", lw=1.4)
    ax.set_xticks(x, sb["method"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("top-1 epicenter accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("epicenter recovery across all runs (dashed = chance)", fontsize=9)
    fig.tight_layout()
    plt.show()
