"""Client similarity and partition -- **one (scope, round, phase) slice at a time**.

Why the API is shaped this way: in `prototype_history` the *round* explains
the large majority of the variance, because all clients' prototypes migrate
together as training proceeds. Compute a similarity matrix over the stacked
table and the dominant axis is training progress, not client identity. So
every function here takes a single slice and asserts it -- pooling across
rounds is impossible by accident, and round becomes a loop axis you watch
rather than a confound you forget.

Scope of this module in the current pass: the channels that already existed,
unified. Nothing new is invented.

| method        | ported from | what it compares |
|---------------|-------------|------------------|
| `proto_cos`   | AER sandbox `client_prototype_similarity` | per-month prototype cosine, averaged over months |
| `proto_cos_dc`| `personalization._deconfounded_prototypes` | same, per-month median (common mode) removed first |
| `drift_sig`   | `personalization._drift_signatures` | cosine of mean-centered `delta_first` trajectories |
| `update_cos`  | reads `update_gram` | cosine of the clients' FedAvg update deltas |

`update_cos` is the one method the refactor *enables* rather than ports: the
Gram is computed at train time (it needs weights that are otherwise not
retained) and this just reads the stored table. It is registered so the
persisted column is usable, not because the channel is being expanded.

Readouts are deliberately thin. `regime_gap` (within- minus between-regime
mean similarity) against its own exhaustive label-shuffle null is the
primary number; ARI is a secondary column because with 5 clients and 2
tokens it takes ~8 discrete values and mangles real movement. Every p-value
has a floor of `1/n_assignments`: 0.10 for a 3-2 split, 0.20 for 4-1. Rank
by effect size; a 4-1 world can never be individually significant.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import bridge
from metrics import fcols
from worlds import World

__all__ = ["SCOPES", "METHODS", "slice_prototypes", "similarity",
           "regime_gap", "shuffle_null", "partition", "ari", "score_run",
           "long_form"]

SCOPES = ("local", "global", "post_fedavg", "post_fedavg_full")


# --------------------------------------------------------------------------
# slicing
# --------------------------------------------------------------------------
def slice_prototypes(prototypes: pd.DataFrame, world: World, scope: str,
                     round_idx: int, phase: str | None = None,
                     clients=None) -> dict[str, np.ndarray]:
    """{client: (n_months, D)} for exactly one (scope, round, phase).

    Months are the intersection across clients and are sorted, so every
    client's matrix is row-aligned by month -- required by anything that
    compares month-for-month.
    """
    if scope not in SCOPES:
        raise KeyError(f"scope {scope!r} not in {SCOPES}")
    d = prototypes[prototypes["scope"] == scope]
    if not len(d):
        raise KeyError(f"no rows for scope {scope!r}")

    d = d[d["round"] == int(round_idx)]
    if not len(d):
        have = sorted(prototypes.loc[prototypes["scope"] == scope, "round"]
                      .unique())
        raise KeyError(f"scope {scope!r} has no round {round_idx} (have {have})")
    if d["round"].nunique() != 1:                    # belt and braces
        raise AssertionError("slice spans more than one round")

    if phase is not None:
        months = world.phase_months(phase)
        d = d[d["month"].isin(months)]
        if not len(d):
            raise KeyError(f"no prototypes in phase {phase!r} "
                           f"(months {months[:3]}...)")

    f = fcols(d)
    if scope == "global":                            # server side, no client
        return {f"cluster_{int(c)}": g.sort_values("month")[f].to_numpy()
                for c, g in d.groupby("cluster")}

    protos = {c: g.sort_values("month")[f].to_numpy(np.float64)
              for c, g in d.groupby("client")}
    if clients:
        protos = {c: protos[c] for c in clients}
    common = sorted(set.intersection(*(
        set(g["month"]) for _, g in d.groupby("client"))))
    if not common:
        raise AssertionError("clients share no month in this slice")
    return {c: g[g["month"].isin(common)].sort_values("month")[f]
            .to_numpy(np.float64)
            for c, g in d.groupby("client") if not clients or c in clients}


# --------------------------------------------------------------------------
# similarity channels
# --------------------------------------------------------------------------
def _cos_matrix(P: dict[str, np.ndarray]) -> pd.DataFrame:
    """Row-wise cosine, averaged over rows, for equal-shaped client stacks."""
    ks = sorted(P)
    M = np.zeros((len(ks), len(ks)))
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            A, B = np.atleast_2d(P[a]), np.atleast_2d(P[b])
            n = min(len(A), len(B))
            A, B = A[:n], B[:n]
            den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
            M[i, j] = float(np.nanmean((A * B).sum(1)
                                       / np.where(den > 0, den, np.nan)))
    return pd.DataFrame(M, index=ks, columns=ks)


def _flat_cos_matrix(rows: dict[str, np.ndarray]) -> pd.DataFrame:
    """Cosine between flattened per-client vectors (the personalization form)."""
    ks = sorted(rows)
    M = np.vstack([np.asarray(rows[k], float).ravel() for k in ks])
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    S = (M / nrm) @ (M / nrm).T
    return pd.DataFrame(S, index=ks, columns=ks)


def _m_proto_cos(prototypes, world, scope, round_idx, phase, **kw):
    return _cos_matrix(slice_prototypes(prototypes, world, scope, round_idx,
                                        phase))


def _m_proto_cos_dc(prototypes, world, scope, round_idx, phase, **kw):
    """Per-month median (common mode) removed, then flattened per client.

    Same robust operator `personalization` uses: the median ignores the
    drifted minority, so what remains is each client's idiosyncratic domain
    rather than the shared hydraulic spillover. Note the consequence the
    notebooks documented -- after centring, the mean off-diagonal cosine is
    pinned near `-1/(N-1)`, so read *spread*, never the mean.
    """
    P = slice_prototypes(prototypes, world, scope, round_idx, phase)
    clients = sorted(P)
    stack = np.stack([P[c] for c in clients])        # (K, M, D)
    common = np.median(stack, axis=0)
    return _flat_cos_matrix({c: (P[c] - common).ravel() for c in clients})


def _m_drift_sig(prototypes, world, scope, round_idx, phase,
                 drift_signals=None, **kw):
    """Cosine of the clients' mean-centered `delta_first` trajectories.

    Restricted to the phase's months when a phase is given. Note this channel
    is known to saturate near 1.0 for every pair when clients share a
    common-mode drift shape -- that is a property of the coupled world, not
    of the implementation.
    """
    if drift_signals is None:
        raise KeyError("drift_sig needs `drift_signals`")
    d = drift_signals
    if phase is not None:
        d = d[d["month"].isin(world.phase_months(phase))]
    piv = d.pivot_table(index="client", columns="month",
                        values="delta_first").fillna(0.0)
    arr = piv.to_numpy()
    arr = arr - arr.mean(axis=1, keepdims=True)
    return _flat_cos_matrix({c: arr[i] for i, c in enumerate(piv.index)})


def _m_update_cos(prototypes, world, scope, round_idx, phase,
                  update_gram=None, **kw):
    """Cosine of the clients' FedAvg update deltas, from the stored Gram.

    Phase-independent by construction (an update is a whole-round object), so
    a phase argument is accepted and ignored.
    """
    if update_gram is None or not len(update_gram):
        raise KeyError("update_cos needs `update_gram`")
    g = update_gram[update_gram["round"] == int(round_idx)]
    if not len(g):
        raise KeyError(f"update_gram has no round {round_idx}")
    clients = sorted(set(g["client_a"]) | set(g["client_b"]))
    S = pd.DataFrame(np.eye(len(clients)), index=clients, columns=clients)
    for r in g.itertuples():
        S.loc[r.client_a, r.client_b] = r.cos
        S.loc[r.client_b, r.client_a] = r.cos
    return S


METHODS = {"proto_cos": _m_proto_cos,
           "proto_cos_dc": _m_proto_cos_dc,
           "drift_sig": _m_drift_sig,
           "update_cos": _m_update_cos}


def similarity(prototypes: pd.DataFrame, world: World, method: str = "proto_cos",
               scope: str = "local", round_idx: int = 0,
               phase: str | None = "init", **extra) -> pd.DataFrame:
    """Client x client similarity for ONE (scope, round, phase) slice."""
    if method not in METHODS:
        raise KeyError(f"method {method!r} not in {sorted(METHODS)}")
    S = METHODS[method](prototypes, world, scope, round_idx, phase, **extra)
    # Mutate the numpy array, never `DataFrame.values`: under pandas
    # copy-on-write that attribute is a read-only view.
    A = np.array(S, dtype=float, copy=True)
    np.fill_diagonal(A, 1.0)
    return pd.DataFrame(A, index=list(S.index), columns=list(S.columns))


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------
def regime_gap(S: pd.DataFrame, labels: dict[str, str]) -> float:
    """Within-regime minus between-regime mean off-diagonal similarity."""
    clients = [c for c in S.index if c in labels]
    within, between = [], []
    for a, b in itertools.combinations(clients, 2):
        (within if labels[a] == labels[b] else between).append(
            float(S.loc[a, b]))
    if not within or not between:
        return np.nan
    return float(np.mean(within) - np.mean(between))


def shuffle_null(S: pd.DataFrame, labels: dict[str, str],
                 stat=regime_gap, n_max: int = 5000) -> dict:
    """Exhaustive label-shuffle null: holds the DATA fixed, permutes labels.

    The only honest floor for a 5-client world. Returns the real statistic,
    its p-value, the null mean (so `lift = real - null_mean` is readable),
    and the number of DISTINCT assignments -- which is the p-value floor.
    """
    clients = sorted(c for c in S.index if c in labels)
    if len(clients) < 3:
        return {"real": np.nan, "p": np.nan, "null_mean": np.nan,
                "n_assign": 0, "p_floor": np.nan}
    real = stat(S, {c: labels[c] for c in clients})
    toks = [labels[c] for c in clients]
    seen, null = set(), []
    for perm in itertools.permutations(toks):
        if perm in seen:
            continue
        seen.add(perm)
        v = stat(S, dict(zip(clients, perm)))
        if np.isfinite(v):
            null.append(v)
        if len(seen) >= n_max:
            break
    null = np.asarray(null, float)
    n = len(seen)
    return {"real": real,
            "p": float((null >= real).mean()) if len(null) else np.nan,
            "null_mean": float(null.mean()) if len(null) else np.nan,
            "n_assign": n, "p_floor": 1.0 / n if n else np.nan}


def partition(S: pd.DataFrame, k: int | None = None,
              seed: int = 0) -> dict[str, int]:
    """Cluster on a similarity matrix. FINCH when available, else linkage.

    Same routine `personalization._cluster` uses, so partitions here and in
    the pipeline are produced the same way.
    """
    clients = list(S.index)
    if len(clients) <= 2:
        return {c: 0 for c in clients}
    if k is None:
        try:
            FINCH = bridge.FINCH()
            c, _, _ = FINCH(S.to_numpy(), distance="cosine",
                            ensure_early_exit=False, verbose=False)
            return {cl: int(lab) for cl, lab in zip(clients, c[:, 0])}
        except Exception:
            k = max(2, len(clients) // 2)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = 1.0 - S.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(Z, t=int(k), criterion="maxclust")
    return {cl: int(x) for cl, x in zip(clients, lab)}


def ari(assign: dict[str, int], labels: dict[str, str]) -> float:
    """Adjusted Rand Index -- secondary only (coarse at 5 clients)."""
    from sklearn.metrics import adjusted_rand_score
    clients = sorted(set(assign) & set(labels))
    if len(clients) < 3:
        return np.nan
    return float(adjusted_rand_score([labels[c] for c in clients],
                                     [assign[c] for c in clients]))


# --------------------------------------------------------------------------
# long-form scoring
# --------------------------------------------------------------------------
def long_form(S: pd.DataFrame, **cols) -> pd.DataFrame:
    """A similarity matrix as tidy pair rows, with provenance columns."""
    rows = []
    ks = list(S.index)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            rows.append({**cols, "client_a": a, "client_b": b,
                         "similarity": float(S.loc[a, b])})
    return pd.DataFrame(rows)


def score_run(res: dict, world: World, run_key: str | None = None,
              methods=("proto_cos", "proto_cos_dc"),
              scopes=("local", "post_fedavg"),
              rounds=None, phases=("init", "final"),
              basis: str = "token") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one saved run -> (similarity_rows, cluster_rows), both long-form.

    Reads persisted tables only. `rounds=None` means every round present in
    the requested scope, one slice at a time.
    """
    key = run_key or res.get("run_key") or (res.get("meta") or {}).get("run_key")
    protos = res.get("prototypes")
    if protos is None or not len(protos):
        return pd.DataFrame(), pd.DataFrame()

    extra = {"drift_signals": res.get("drift_signals"),
             "update_gram": res.get("update_gram")}
    sim_rows, clu_rows = [], []

    for scope in scopes:
        avail = protos[protos["scope"] == scope]
        if not len(avail):
            continue
        rs = sorted(avail["round"].unique()) if rounds is None else list(rounds)
        for r in rs:
            for phase in phases:
                labels = world.regimes(phase=phase, basis=basis)
                for method in methods:
                    try:
                        S = similarity(protos, world, method=method,
                                       scope=scope, round_idx=int(r),
                                       phase=phase, **extra)
                    except (KeyError, AssertionError):
                        continue
                    null = shuffle_null(S, labels)
                    assign = partition(S)
                    meta = {"world_hash": world.sim_hash, "run_key": key,
                            "scope": scope, "round": int(r), "phase": phase,
                            "method": method, "basis": basis}
                    sim_rows.append(long_form(S, **meta))
                    clu_rows.append(pd.DataFrame([{
                        **meta, "client": c, "cluster": assign.get(c),
                        "regime": labels.get(c),
                        "regime_gap": null["real"], "gap_p": null["p"],
                        "gap_null": null["null_mean"],
                        "gap_lift": (null["real"] - null["null_mean"]
                                     if np.isfinite(null["null_mean"])
                                     else np.nan),
                        "p_floor": null["p_floor"],
                        "ari": ari(assign, labels),
                        "sd_offdiag": float(np.std(
                            S.to_numpy()[~np.eye(len(S), dtype=bool)])),
                    } for c in S.index]))

    sim = pd.concat(sim_rows, ignore_index=True) if sim_rows else pd.DataFrame()
    clu = pd.concat(clu_rows, ignore_index=True) if clu_rows else pd.DataFrame()
    return sim, clu
