"""fed_regimes — is the demand regime recoverable from FEDERATED artifacts?

Successor to `FLC_033_regimes_and_transitions.ipynb` Part IV. Two things moved:

* **land use replaces density.** A consumption-map token is now two independent
  alphabets — income ``{L,M,H}`` and land use ``{R,M,C,I}`` — so the old single
  ``CODE_ORDER`` ladder cannot score it. Scoring runs on TWO axes: an income
  LEVEL rank and a sector SHAPE rank (night/day ordering). Either axis may be
  degenerate in a given world; that is reported as ``no contrast``, not as a
  failure of the signal.
* **the signals come from the FL run, not from a per-client state-space fit.**
  Everything is read out of what `aer_federated_sandbox` already saved in
  ``fed_sandbox/<world_id>/<run_id>/``, so the whole notebook runs over every
  federated case in ``runs.csv`` without retraining anything.

What is kept from Part IV: the signal / control / ceiling triad, non-cosine
similarity (z-score per feature -> Euclidean -> ``1 - d/max d``; cosine divides
out the norm and on a level ladder the norm IS the label), the exact ``S(N,k)``
partition null, the 1-D MDS order test, the migration test, and the pairwise
confound regression.

What is dropped: the Granger objects (``Â_mn``, the ``ΔÂ`` transition map, the
centralized oracle). FPL produces no off-diagonal block. The cohesion negative
control survives as a latent cross-correlation, and its directed variant keeps a
``ΔW`` object.

Basis caveat, and it is load-bearing: ``latents_by_round`` / ``prototype_history``
are captured **pre-FedAvg**, so each client sits in its own diverged encoder and
raw features are NOT comparable across clients. Signals 0-2 therefore read the
post-FedAvg tables (one shared basis); the round-resolved signal 4 uses only
rotation-invariant summaries.
"""
from __future__ import annotations

import json
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# ---------------------------------------------------------------------------
# 0. the two alphabets  (spec.py's codec, kept in sync by hand)
# ---------------------------------------------------------------------------
INCOME = {"L": "low", "M": "medium", "H": "high"}
LAND_USE = {"R": "residential", "M": "mixed", "C": "commercial", "I": "industrial"}

#: income LEVEL rank. Ordinal is enough — every order test here is rank-based.
INCOME_RANK = {"low": 0, "medium": 1, "high": 2}

#: sector SHAPE rank, ordered by night/day ratio (the minimum-night-flow
#: signature). Derivation: `sector_day_shape` on `params["land_use"]["sectors"]`
#: gives commercial < residential < industrial (test_sector_night_day_ordering);
#: `mixed` is a residential/commercial plot mix, so it lands between the two.
#: Re-derive if the sector signatures or the mixed plot mix change.
SECTOR_RANK = {"commercial": 0, "mixed": 1, "residential": 2, "industrial": 3}

DISTRICTS_DEFAULT = [f"District_{c}" for c in "ABCDE"]


def decode_token(tok: str) -> tuple[str, str]:
    """``'LR' -> ('low', 'residential')``. Position disambiguates ``M``."""
    tok = str(tok).strip().upper()
    if len(tok) != 2 or tok[0] not in INCOME or tok[1] not in LAND_USE:
        raise ValueError(f"bad consumption-map token {tok!r}")
    return INCOME[tok[0]], LAND_USE[tok[1]]


def encode_token(income: str, land_use: str) -> str:
    inv_i = {v: k for k, v in INCOME.items()}
    inv_l = {v: k for k, v in LAND_USE.items()}
    return inv_i[income] + inv_l[land_use]


# ---------------------------------------------------------------------------
# 1. discovery — worlds, saved runs, registry
# ---------------------------------------------------------------------------
def find_repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "conf" / "base" / "parameters.yml").exists() and (cand / "src" / "fedwater").is_dir():
            return cand
    return None


def list_worlds(root: Path) -> dict:
    """world_id -> {client_dir, label, meta}. Same convention as the sandboxes."""
    worlds: dict[str, dict] = {}
    local = root / "data" / "07_model_output" / "clients"
    if local.exists():
        worlds["local"] = {"client_dir": local, "meta": {},
                           "label": "local (data/07_model_output)"}
    for mpath in sorted((root / "data" / "09_experiments" / "worlds").glob("*/manifest.json")):
        try:
            man = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if man.get("status") != "ok":
            continue
        sim_hash = man.get("sim_hash", mpath.parent.name)
        cdir = mpath.parent / "clone" / "data" / "07_model_output" / "clients"
        if not cdir.exists():
            continue
        meta = man.get("world", {})
        worlds[sim_hash] = {"client_dir": cdir, "meta": meta,
                            "label": f"{sim_hash[:10]} · {meta.get('variant','?')} · "
                                     f"{meta.get('consumption_map','?')}"}
    return worlds


def data_root(world: dict) -> Path:
    """`.../data/` for this world — two levels above its client_dir."""
    return world["client_dir"].parents[1]


def list_runs(sandbox: Path, world_id: str | None = None) -> list[tuple[str, str]]:
    out = []
    for wdir in sorted(p for p in sandbox.iterdir() if p.is_dir()):
        if world_id and wdir.name != world_id:
            continue
        for rdir in sorted(p for p in wdir.iterdir() if p.is_dir()):
            if (rdir / "fed_model.pt").exists():
                out.append((wdir.name, rdir.name))
    return out


def read_table(path_base: Path) -> pd.DataFrame | None:
    for suf in (".parquet", ".csv.gz", ".csv"):
        p = path_base.with_suffix(suf)
        if p.exists():
            return pd.read_parquet(p) if suf == ".parquet" else pd.read_csv(p)
    return None


def write_table(df: pd.DataFrame, path_base: Path) -> None:
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(path_base.with_suffix(".csv.gz"), index=False)


def load_run(sandbox: Path, world_id: str, run_id: str, weights: bool = False) -> dict:
    """The saved tables for one run. `weights=True` also loads `fed_model.pt`."""
    d = sandbox / world_id / run_id
    out: dict = {"world_id": world_id, "run_id": run_id, "dir": d}
    for name in ("prototype_history", "global_prototype_history", "latent_trajectories",
                 "latents_by_round", "prototypes", "drift_signals", "checks",
                 "latent_by_round_metrics", "probes_by_round"):
        out[name] = read_table(d / name)
    cfg = d / "config.json"
    out["config"] = json.loads(cfg.read_text()) if cfg.exists() else {}
    if weights:
        import torch
        try:
            out["state"] = torch.load(d / "fed_model.pt", map_location="cpu", weights_only=False)
        except TypeError:
            out["state"] = torch.load(d / "fed_model.pt", map_location="cpu")
    return out


def registry(sandbox: Path) -> pd.DataFrame:
    p = sandbox / "runs.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def patch_registry(sandbox: Path, world_id: str, run_id: str, cols: dict) -> None:
    """Write scalar summaries onto this run's row in runs.csv (save_analysis's half)."""
    reg = registry(sandbox)
    if reg.empty:
        return
    row = (reg["world_id"] == world_id) & (reg["run_id"] == run_id)
    if not row.any():
        return
    for k, v in cols.items():
        if k not in reg.columns:
            reg[k] = pd.Series(pd.NA, index=reg.index, dtype="object")
        try:
            reg.loc[row, k] = v
        except (TypeError, ValueError):
            reg[k] = reg[k].astype("object")
            reg.loc[row, k] = v
    reg.to_csv(sandbox / "runs.csv", index=False)


def world_meta(worlds: dict, world_id: str, reg: pd.DataFrame | None = None) -> dict:
    """World meta from the manifest, falling back to runs.csv's `w_*` columns."""
    meta = dict(worlds.get(world_id, {}).get("meta") or {})
    if meta or reg is None or reg.empty:
        return meta
    sub = reg[reg["world_id"] == world_id]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    return {c[2:]: r[c] for c in reg.columns if c.startswith("w_") and pd.notna(r[c])}


# ---------------------------------------------------------------------------
# 2. ground truth — derived from the world, never hardcoded
# ---------------------------------------------------------------------------
def regime_states(meta: dict, districts: list[str] | None = None) -> pd.DataFrame:
    """district -> its token before and after the drift.

    ``consumption_map`` is the commissioning state; the drift district switches to
    ``drift_to_income`` / ``drift_to_land_use``. Returns an empty frame when the
    world has no meta (e.g. ``"local"``), which callers treat as unlabelled.
    """
    cmap = meta.get("consumption_map")
    if not cmap:
        return pd.DataFrame(columns=["district", "token_init", "token_final",
                                     "income_init", "land_use_init",
                                     "income_final", "land_use_final", "mover"])
    toks = str(cmap).split("_")
    names = districts or [f"District_{chr(ord('A') + i)}" for i in range(len(toks))]
    rows = []
    for d, t in zip(names, toks):
        inc, lu = decode_token(t)
        rows.append(dict(district=d, token_init=t.upper(), income_init=inc, land_use_init=lu,
                         token_final=t.upper(), income_final=inc, land_use_final=lu))
    st = pd.DataFrame(rows)

    drifter = meta.get("drift_district")
    to_i, to_l = meta.get("drift_to_income"), meta.get("drift_to_land_use")
    k = st["district"] == drifter
    if k.any():
        inc = to_i if to_i in INCOME_RANK else st.loc[k, "income_init"].iloc[0]
        lu = to_l if to_l in SECTOR_RANK else st.loc[k, "land_use_init"].iloc[0]
        st.loc[k, ["income_final", "land_use_final"]] = [inc, lu]
        st.loc[k, "token_final"] = encode_token(inc, lu)
    st["mover"] = st["token_init"] != st["token_final"]
    return st


def phase_months(world: dict, n_months: int | None = None) -> dict[str, list[int]]:
    """``{'init': pre-drift months, 'final': post-ramp months}``.

    The diffusion span itself is EXCLUDED from both: during the ramp neither token
    is true of the mover, so any window drawn from it is mislabelled by
    construction. Falls back to a halves split when the world has no schedule.
    """
    try:
        sch = pd.read_csv(data_root(world) / "03_primary/gt_drift_schedule.csv")
        m_min, m_max = int(sch["drift_month"].min()), int(sch["drift_month"].max())
        n = int(n_months or (m_max + 12))
        return {"init": list(range(0, m_min)), "final": list(range(m_max + 1, n)),
                "ramp": list(range(m_min, m_max + 1))}
    except Exception:
        n = int(n_months or 24)
        return {"init": list(range(0, n // 2)), "final": list(range(n // 2, n)), "ramp": []}


def axis_ranks(states: pd.DataFrame, phase: str,
               income_rank: dict | None = None,
               sector_rank: dict | None = None) -> dict[str, np.ndarray]:
    """The two ordinal axes for one phase, aligned to ``states.district`` order."""
    ir = income_rank or INCOME_RANK
    sr = sector_rank or SECTOR_RANK
    suf = "init" if phase == "init" else "final"
    return {"income": np.array([ir[v] for v in states[f"income_{suf}"]], float),
            "sector": np.array([sr[v] for v in states[f"land_use_{suf}"]], float)}


# ---------------------------------------------------------------------------
# 3. similarity — non-cosine, magnitude-preserving
# ---------------------------------------------------------------------------
def sim_from_features(F: np.ndarray) -> np.ndarray:
    """(N, f) features -> (N, N) similarity. z-score per feature, Euclidean,
    ``1 - d/max d``. NOT cosine: cosine divides out the norm, and on an L/M/H
    ladder the norm is the label."""
    F = np.nan_to_num(np.asarray(F, float), nan=0.0, posinf=0.0, neginf=0.0)
    sd = F.std(0)
    sd[sd == 0] = 1.0
    Z = (F - F.mean(0)) / sd
    D = squareform(pdist(Z)) if len(Z) > 1 else np.zeros((len(Z), len(Z)))
    mx = D.max()
    S = np.ones_like(D) if mx <= 0 else 1.0 - D / mx
    np.fill_diagonal(S, 1.0)
    return (S + S.T) / 2


def _fcols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _features_to_sim(feat: dict[str, np.ndarray], districts: list[str]) -> np.ndarray | None:
    if not feat or any(d not in feat for d in districts):
        return None
    n = max(len(v) for v in feat.values())
    F = np.vstack([np.resize(np.asarray(feat[d], float), n) for d in districts])
    return sim_from_features(F)


# ---------------------------------------------------------------------------
# 4. the signals
# ---------------------------------------------------------------------------
def sig_prototype_ladder(run: dict, months: list[int], districts: list[str]):
    """FPL prototypes, ``scope=post_fedavg`` — the server already holds these, so
    this signal costs ZERO extra communication. Post-FedAvg means one shared
    basis, so the raw latent means are comparable across clients."""
    p = run.get("prototypes")
    if p is None or "scope" not in p:
        return None
    p = p[(p["scope"] == "post_fedavg") & p["month"].isin(months)]
    if p.empty:
        return None
    fc = _fcols(p)
    g = p.groupby("client")[fc].mean()
    return _features_to_sim({c: g.loc[c].to_numpy() for c in g.index}, districts)


def sig_latent_centroid(run: dict, months: list[int], districts: list[str]):
    """Per-district mean and spread of the post-FedAvg window latents."""
    lt = run.get("latent_trajectories")
    if lt is None:
        return None
    lt = lt[lt["month"].isin(months)]
    if lt.empty:
        return None
    fc = _fcols(lt)
    feat = {}
    for d, g in lt.groupby("district"):
        Z = g[fc].to_numpy(float)
        feat[d] = np.concatenate([Z.mean(0), Z.std(0)])
    return _features_to_sim(feat, districts)


def sig_diurnal_norm(run: dict, months: list[int], districts: list[str],
                     resolution_h: float = 1.0):
    """‖latent‖ averaged by hour of day of the window start.

    The federated analogue of a demand curve: the norm is basis-invariant, and
    the diurnal amplitude is what the sector signatures actually differ in.
    """
    lt = run.get("latent_trajectories")
    if lt is None or "window" not in lt.columns:
        return None
    lt = lt[lt["month"].isin(months)]
    if lt.empty:
        return None
    fc = _fcols(lt)
    hour = ((lt["window"].to_numpy() * resolution_h) % 24).astype(int)
    nrm = np.linalg.norm(lt[fc].to_numpy(float), axis=1)
    tmp = pd.DataFrame({"district": lt["district"].to_numpy(), "hour": hour, "n": nrm})
    piv = tmp.pivot_table(index="district", columns="hour", values="n", aggfunc="mean")
    piv = piv.reindex(columns=range(24)).interpolate(axis=1, limit_direction="both")
    return _features_to_sim({d: piv.loc[d].to_numpy() for d in piv.index}, districts)


def sig_weight_divergence(run: dict, districts: list[str]):
    """How far each client's final weights drifted from the global model, per
    tensor. Personalization is the client's answer to its own data, so its
    geometry is a regime candidate. Phase-independent (one fit per run)."""
    state = run.get("state")
    if not state:
        return None
    glob = state.get("global")
    if glob is None:
        return None
    feat = {}
    for c in districts:
        sd = state.get(c)
        if sd is None:
            return None
        v = []
        for k, g in glob.items():
            if k not in sd or not hasattr(g, "numpy"):
                continue
            a = np.asarray(sd[k].detach().cpu().numpy(), float).ravel()
            b = np.asarray(g.detach().cpu().numpy(), float).ravel()
            den = np.linalg.norm(b) or 1.0
            v.append(np.linalg.norm(a - b) / den)
        feat[c] = np.array(v)
    return _features_to_sim(feat, districts)


def sig_round_spectrum(snaps_round: pd.DataFrame, months: list[int], districts: list[str]):
    """Rotation-invariant summary of one round's PRE-FedAvg latents.

    Each client sits in its own diverged encoder at snapshot time, so raw features
    are not comparable. Singular values of the month-centroid matrix and the mean
    month-to-month step size are invariant to that rotation.
    """
    d = snaps_round[snaps_round["month"].isin(months)]
    if d.empty:
        return None
    fc = _fcols(d)
    feat = {}
    for c, g in d.groupby("district"):
        M = g.groupby("month")[fc].mean().to_numpy(float)
        if len(M) < 2:
            return None
        sv = np.linalg.svd(M - M.mean(0), compute_uv=False)
        step = np.linalg.norm(np.diff(M, axis=0), axis=1)
        feat[c] = np.concatenate([np.resize(sv, 5) / (np.linalg.norm(sv) or 1.0),
                                  [np.linalg.norm(M, axis=1).mean(), step.mean()]])
    return _features_to_sim(feat, districts)


def sig_cohesion(run: dict, months: list[int], districts: list[str], lag: int = 0):
    """NEGATIVE CONTROL — cross-correlation of the districts' latent-norm series.

    This is cohesion, not structural equivalence: a big value is evidence of a
    PIPE, not of a shared regime. Reusing it as a regime signal is the central
    mistake the FLC_033 refactor exists to avoid, so it is carried explicitly.
    ``lag > 0`` gives the directed (asymmetric) variant used by ``delta_w``.
    """
    lt = run.get("latent_trajectories")
    if lt is None or "window" not in lt.columns:
        return None
    lt = lt[lt["month"].isin(months)]
    if lt.empty:
        return None
    fc = _fcols(lt)
    ser = {}
    for d, g in lt.groupby("district"):
        g = g.sort_values("window")
        ser[d] = pd.Series(np.linalg.norm(g[fc].to_numpy(float), axis=1),
                           index=g["window"].to_numpy())
    if any(d not in ser for d in districts):
        return None
    X = pd.DataFrame({d: ser[d] for d in districts}).dropna()
    if len(X) < 8 + lag:
        return None
    A = X.to_numpy(float)                     # numpy, NOT pandas: Series.corr
    n = A.shape[1]                            # re-aligns on the index and would
    if lag == 0:                              # silently undo the shift
        with np.errstate(invalid="ignore"):
            M = np.corrcoef(A, rowvar=False)
        return np.nan_to_num(M, nan=0.0)
    M = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue                      # row i leads col j by `lag`
            a, b = A[:-lag, i], A[lag:, j]
            sa, sb = a.std(), b.std()
            M[i, j] = 0.0 if sa == 0 or sb == 0 else float(np.corrcoef(a, b)[0, 1])
    return M


def sig_demand_profile(client_dir: Path, months: list[int], districts: list[str],
                       resolution_h: float = 1.0):
    """CEILING — mean daily demand profile from the raw client CSVs, pooled
    centrally. If this cannot recover the regime, no federated signal can, and a
    weak federated score is a statement about the DATA, not about the method."""
    feat = {}
    for d in districts:
        p = client_dir / f"{d}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        q = [c for c in df.columns if c.startswith("q_")]
        if not q or "month" not in df.columns:
            return None
        df = df[df["month"].isin(months)]
        if df.empty:
            return None
        hour = ((np.arange(len(df)) * resolution_h) % 24).astype(int)
        prof = pd.Series(df[q].sum(axis=1).to_numpy(), index=hour).groupby(level=0).mean()
        feat[d] = prof.reindex(range(24)).interpolate(limit_direction="both").to_numpy()
    return _features_to_sim(feat, districts)


SIGNAL_KIND = {
    "0. prototype ladder": "federated",
    "1. latent centroid": "federated",
    "2. diurnal norm profile": "federated",
    "3. weight divergence": "federated",
    "4. round spectrum": "federated",
    "5. latent cohesion": "control",
    "6. demand profile": "ceiling",
}


def phase_signals(run: dict, world: dict, months: list[int], districts: list[str],
                  resolution_h: float = 1.0) -> dict[str, np.ndarray]:
    """Every phase-level signal for one run, skipping the ones this run cannot
    support (missing table, empty phase) rather than faking them."""
    out = {
        "0. prototype ladder": sig_prototype_ladder(run, months, districts),
        "1. latent centroid": sig_latent_centroid(run, months, districts),
        "2. diurnal norm profile": sig_diurnal_norm(run, months, districts, resolution_h),
        "3. weight divergence": sig_weight_divergence(run, districts),
        "5. latent cohesion": sig_cohesion(run, months, districts),
    }
    if world.get("client_dir") is not None:
        out["6. demand profile"] = sig_demand_profile(world["client_dir"], months,
                                                      districts, resolution_h)
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# 5. scoring — partition + two ordinal axes, exact nulls
# ---------------------------------------------------------------------------
def cluster_from_similarity(S: np.ndarray, k: int) -> np.ndarray:
    n = len(S)
    if k <= 1:
        return np.zeros(n, int)
    if k >= n:
        return np.arange(n)
    D = 1.0 - np.asarray(S, float)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    D = np.nan_to_num(D, nan=float(np.nanmax(D)) if np.isfinite(D).any() else 1.0)
    try:
        km = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:                                   # scikit-learn < 1.2
        km = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    return km.fit_predict(D)


def partitions_into_k(n: int, k: int) -> list[tuple[int, ...]]:
    """Every partition of n labelled items into EXACTLY k non-empty blocks
    (restricted growth strings). ``len(...) == S(n, k)``: S(5,3) = 25."""
    out: list[tuple[int, ...]] = []

    def rec(pref: tuple[int, ...], mx: int):
        i = len(pref)
        if i == n:
            if mx + 1 == k:
                out.append(pref)
            return
        for b in range(mx + 2):
            if b > k - 1 or (k - max(mx + 1, b) - 1) > (n - i - 1):
                continue
            rec(pref + (b,), max(mx, b))

    rec((), -1)
    return out


def _mds1(S: np.ndarray) -> np.ndarray:
    """Classical-MDS first coordinate of ``1 - S``. Sign is arbitrary — every
    order test below is therefore two-sided."""
    D = 1.0 - np.asarray(S, float)
    np.fill_diagonal(D, 0.0)
    D = np.nan_to_num(D)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    i = int(np.argmax(w))
    return V[:, i] * np.sqrt(max(float(w[i]), 0.0))


def order_test(coord: np.ndarray, ranks: np.ndarray) -> tuple[float, float]:
    """Two-sided Spearman with an EXACT null over all ``N!`` orderings."""
    if len(set(ranks.tolist())) < 2 or np.allclose(coord, coord[0]):
        return np.nan, np.nan
    rho = float(spearmanr(coord, ranks).statistic)
    null = np.array([abs(float(spearmanr(coord, np.array(p)).statistic))
                     for p in permutations(ranks.tolist())])
    return rho, float((null >= abs(rho) - 1e-12).mean())


def score_similarity(S: np.ndarray, tokens: list[str],
                     ranks: dict[str, np.ndarray]) -> dict:
    """One similarity matrix vs the ground truth of one phase.

    * partition recovery: ARI against the joint token, with the exact ``S(N,k)``
      null (best attainable p at N=5, k=3 is 1/25 = 0.04);
    * order recovery: does the 1-D MDS axis ORDER the districts along the income
      LEVEL ladder, and along the sector SHAPE ladder? An axis with fewer than
      two distinct values is reported as ``no contrast``.
    """
    y = np.unique(np.asarray(tokens, dtype=object), return_inverse=True)[1]
    n, k = len(y), len(set(y.tolist()))
    out: dict = {"k": k, "partition": "", "ARI": np.nan, "p_partition": np.nan,
                 "exact_match": np.nan, "note": ""}
    # Two degeneracies that would otherwise read as perfect recovery:
    #   k == 1  -> every district shares a token; nothing to separate.
    #   k == N  -> the truth is the all-singletons partition, S(N,N) = 1, so any
    #              clustering scores ARI 1.0 at p = 1.0 by construction.
    if k <= 1:
        out["note"] = "no contrast (single token)"
    elif k >= n:
        out["note"] = f"k == N ({n}) — partition test vacuous, read the order tests"
    else:
        lab = cluster_from_similarity(S, k)
        ari = float(adjusted_rand_score(y, lab))
        cand = partitions_into_k(len(y), k)
        null = np.array([adjusted_rand_score(y, np.array(c)) for c in cand])
        out.update(ARI=ari, p_partition=float((null >= ari - 1e-12).mean()),
                   exact_match=float(ari == 1.0),
                   partition="|".join("".join(str(i) for i in range(len(lab)) if lab[i] == b)
                                      for b in sorted(set(lab.tolist()))))
    coord = _mds1(S)
    for name, r in ranks.items():
        rho, p = order_test(coord, r)
        out[f"rho_{name}"] = rho
        out[f"p_{name}"] = p
        out[f"n_{name}"] = len(set(r.tolist()))
    return out


def migration(S_init: np.ndarray, S_final: np.ndarray, states: pd.DataFrame,
              districts: list[str]) -> list[dict]:
    """Does the SAME signal put the mover with its initial peers AND with its
    final peers? That conjunction is a far smaller target than either half."""
    def company(S, k, who):
        lab = cluster_from_similarity(S, k)
        i = districts.index(who)
        return {districts[j][-1] for j in range(len(districts))
                if lab[j] == lab[i] and j != i}

    st = states.set_index("district")
    rows = []
    for mover in states.loc[states["mover"], "district"]:
        if mover not in districts:
            continue
        tgt_i = {d[-1] for d in districts
                 if d != mover and st.at[d, "token_init"] == st.at[mover, "token_init"]}
        tgt_f = {d[-1] for d in districts
                 if d != mover and st.at[d, "token_final"] == st.at[mover, "token_final"]}
        k_i = st["token_init"].reindex(districts).nunique()
        k_f = st["token_final"].reindex(districts).nunique()
        got_i, got_f = company(S_init, k_i, mover), company(S_final, k_f, mover)
        rows.append(dict(mover=mover[-1],
                         token=f"{st.at[mover,'token_init']}->{st.at[mover,'token_final']}",
                         k_init=int(k_i), k_final=int(k_f),
                         with_init="{" + ",".join(sorted(got_i)) + "}",
                         with_final="{" + ",".join(sorted(got_f)) + "}",
                         target_init="{" + ",".join(sorted(tgt_i)) + "}",
                         target_final="{" + ",".join(sorted(tgt_f)) + "}",
                         right_init=got_i == tgt_i, right_final=got_f == tgt_f,
                         moved=got_i != got_f,
                         moved_correctly=(got_i == tgt_i) and (got_f == tgt_f)))
    return rows


def confound_regression(S: np.ndarray, states: pd.DataFrame, districts: list[str],
                        phase: str, topology: pd.DataFrame | None = None,
                        amplitude: pd.Series | None = None,
                        n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Is the similarity reading the REGIME, or adjacency / size?

    One row per district pair, similarity regressed on standardised
    ``same token + income gap + sector gap + adjacency + size gap`` at once.
    With N=5 there are 10 rows: read the coefficients as direction, not estimate.
    p-values come from a permutation test on the pair rows.
    """
    suf = "init" if phase == "init" else "final"
    st = states.set_index("district")
    gt = {}
    if topology is not None and {"district_a", "district_b"} <= set(topology.columns):
        for _, r in topology.iterrows():
            gt[tuple(sorted((r["district_a"], r["district_b"])))] = \
                float(r.get("open_boundary_pipes", 0) or 0)

    rows, yv = [], []
    for i, j in combinations(range(len(districts)), 2):
        a, b = districts[i], districts[j]
        rows.append({
            "pair": a[-1] + b[-1],
            "same token": float(st.at[a, f"token_{suf}"] == st.at[b, f"token_{suf}"]),
            "income gap": abs(INCOME_RANK[st.at[a, f"income_{suf}"]]
                              - INCOME_RANK[st.at[b, f"income_{suf}"]]),
            "sector gap": abs(SECTOR_RANK[st.at[a, f"land_use_{suf}"]]
                              - SECTOR_RANK[st.at[b, f"land_use_{suf}"]]),
            "adjacent": float(gt.get(tuple(sorted((a, b))), 0.0) > 0),
            "size gap": (abs(float(np.log(amplitude[a])) - float(np.log(amplitude[b])))
                         if amplitude is not None else 0.0),
        })
        yv.append(float(S[i, j]))

    X = pd.DataFrame(rows).set_index("pair")
    keep = [c for c in X.columns if X[c].std() > 0]
    if not keep:
        return pd.DataFrame(columns=["predictor", "beta", "p_perm"])
    A = ((X[keep] - X[keep].mean()) / X[keep].std()).to_numpy()
    A = np.column_stack([np.ones(len(A)), A])
    y = np.array(yv)
    beta = np.linalg.lstsq(A, y, rcond=None)[0]

    rng = np.random.default_rng(seed)
    null = np.abs(np.array([np.linalg.lstsq(A, rng.permutation(y), rcond=None)[0]
                            for _ in range(n_perm)]))
    p = (null >= np.abs(beta) - 1e-12).mean(axis=0)
    return pd.DataFrame({"predictor": ["intercept"] + keep, "beta": beta, "p_perm": p})


# ---------------------------------------------------------------------------
# 6. run the whole thing over every federated case
# ---------------------------------------------------------------------------
def score_run(sandbox: Path, worlds: dict, world_id: str, run_id: str,
              resolution_h: float = 1.0, rounds: str | list[int] = "all",
              reg: pd.DataFrame | None = None) -> dict:
    """All signals x both phases for one saved run.

    -> {scores, migration, similarity, states, note}. ``note`` is non-empty when
    the run could not be scored at all (unlabelled world, no tables).
    """
    world = worlds.get(world_id, {"client_dir": None, "meta": {}})
    meta = world_meta(worlds, world_id, reg)
    states = regime_states(meta)
    empty = {"scores": pd.DataFrame(), "migration": pd.DataFrame(),
             "similarity": pd.DataFrame(), "states": states}
    if states.empty:
        return {**empty, "note": "world has no consumption_map (unlabelled)"}

    run = load_run(sandbox, world_id, run_id, weights=True)
    lt = run.get("latent_trajectories")
    if lt is None or lt.empty:
        return {**empty, "note": "no latent_trajectories saved"}

    districts = [d for d in states["district"] if d in set(lt["district"])]
    if len(districts) < 3:
        return {**empty, "note": f"only {len(districts)} labelled clients in the run"}
    states = states[states["district"].isin(districts)].reset_index(drop=True)

    n_months = int(meta.get("n_months") or (lt["month"].max() + 1))
    ph = phase_months(world, n_months)
    snaps = run.get("latents_by_round")

    srows, simrows, keep_S = [], [], {}
    for phase in ("init", "final"):
        months = [m for m in ph[phase] if m in set(lt["month"])]
        if not months:
            continue
        ranks = axis_ranks(states, phase, )
        tokens = list(states[f"token_{'init' if phase == 'init' else 'final'}"])
        sigs = phase_signals(run, world, months, districts, resolution_h)

        if snaps is not None and "round" in snaps.columns:
            rs = (sorted(snaps["round"].unique()) if rounds == "all"
                  else list(rounds) if rounds != "last" else [snaps["round"].max()])
            for r in rs:
                S = sig_round_spectrum(snaps[snaps["round"] == r], months, districts)
                if S is None:
                    continue
                srows.append({"signal": "4. round spectrum", "round": int(r), "phase": phase,
                              **score_similarity(S, tokens, ranks)})

        for name, S in sigs.items():
            srows.append({"signal": name, "round": np.nan, "phase": phase,
                          **score_similarity(S, tokens, ranks)})
            keep_S[(phase, name)] = S
            for i, a in enumerate(districts):
                for j, b in enumerate(districts):
                    simrows.append({"phase": phase, "signal": name,
                                    "district_a": a, "district_b": b, "value": float(S[i, j])})

    scores = pd.DataFrame(srows)
    if not scores.empty:
        scores.insert(0, "run_id", run_id)
        scores.insert(0, "world_id", world_id)
        scores["kind"] = scores["signal"].map(SIGNAL_KIND)
        scores["n_months_used"] = n_months

    mrows = []
    for name in {n for _, n in keep_S}:
        if ("init", name) in keep_S and ("final", name) in keep_S:
            for m in migration(keep_S[("init", name)], keep_S[("final", name)],
                               states, districts):
                mrows.append({"world_id": world_id, "run_id": run_id, "signal": name,
                              "kind": SIGNAL_KIND.get(name), **m})

    sim = pd.DataFrame(simrows)
    if not sim.empty:
        sim.insert(0, "run_id", run_id)
        sim.insert(0, "world_id", world_id)
    return {"scores": scores, "migration": pd.DataFrame(mrows), "similarity": sim,
            "states": states, "note": "", "phases": ph, "districts": districts,
            "S": keep_S, "run": run, "world": world}


def score_all(sandbox: Path, worlds: dict, resolution_h: float = 1.0,
              rounds: str | list[int] = "last", runs: list | None = None,
              verbose: bool = True) -> dict:
    """Every saved run in the sandbox. Scores are pooled by CONCATENATION;
    aggregate them with `aggregate` — never pool the features themselves, since
    each run's latent basis is arbitrarily rotated relative to every other."""
    reg = registry(sandbox)
    pairs = runs or list_runs(sandbox)
    S, M, X, notes = [], [], [], []
    for wid, rid in pairs:
        try:
            r = score_run(sandbox, worlds, wid, rid, resolution_h, rounds, reg)
        except Exception as e:                      # one bad run must not kill the sweep
            notes.append({"world_id": wid, "run_id": rid, "note": f"{type(e).__name__}: {e}"})
            continue
        if r["note"]:
            notes.append({"world_id": wid, "run_id": rid, "note": r["note"]})
            continue
        S.append(r["scores"]); M.append(r["migration"]); X.append(r["similarity"])
        if verbose:
            f = r["scores"]
            f = f[f["kind"] == "federated"]
            print(f"{wid[:10]:10s} {rid[:44]:44s} "
                  f"best ARI {f['ARI'].max():+.3f}  |ρ| {f[['rho_income','rho_sector']].abs().max().max():.3f}")
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return {"scores": cat(S), "migration": cat(M), "similarity": cat(X),
            "notes": pd.DataFrame(notes)}


def aggregate(scores: pd.DataFrame, by: tuple[str, ...] = ("kind", "signal")) -> pd.DataFrame:
    """Mean score per signal across runs/worlds/phases. Averaging SCORES is the
    only legitimate way to pool separately trained runs."""
    if scores.empty:
        return scores
    cols = [c for c in ("ARI", "p_partition", "rho_income", "rho_sector",
                        "p_income", "p_sector") if c in scores.columns]
    g = scores.assign(abs_rho_income=scores["rho_income"].abs(),
                      abs_rho_sector=scores["rho_sector"].abs())
    out = g.groupby(list(by)).agg(n=("ARI", "size"), **{c: (c, "mean") for c in cols},
                                  abs_rho_income=("abs_rho_income", "mean"),
                                  abs_rho_sector=("abs_rho_sector", "mean"))
    return out.sort_values("ARI", ascending=False)


def delta_w(run: dict, world: dict, phases: dict, districts: list[str],
            lag: int = 1) -> pd.DataFrame:
    """[EXTENSION] The directed transition object that survives the move to FPL.

    ``ΔW_mn`` = lagged latent cross-correlation (final) − (init). Out-strength is
    how much m explains the others, in-strength the reverse; a district whose
    demand regime rises should draw more through its boundary pipes and become a
    net EXPORTER. Cohesion, not equivalence — read it alongside the control.
    """
    W = {}
    for phase in ("init", "final"):
        months = phases.get(phase) or []
        W[phase] = sig_cohesion(run, months, districts, lag=lag)
    if W["init"] is None or W["final"] is None:
        return pd.DataFrame()
    d = np.abs(W["final"]) - np.abs(W["init"])
    np.fill_diagonal(d, 0.0)
    out = pd.DataFrame({"district": [x[-1] for x in districts],
                        "d_out": np.nansum(d, axis=1), "d_in": np.nansum(d, axis=0)})
    out["d_net"] = out["d_out"] - out["d_in"]
    return out.sort_values("d_net", ascending=False, ignore_index=True)
