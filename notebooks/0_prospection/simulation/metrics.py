"""Latent-space metrics, controlled probes, structural checks, drift.

One definition each. The two notebooks had two different `latent_metrics`
(PP_01 returned five fields, the AER sandbox added `probe_month`, chance
baselines and `pca90`), which silently made their numbers incomparable; this
is the superset, so every run lands in one table.

The distinction that matters here is **clustered** vs **linearly decodable**.
`sil_*` asks whether the space is grouped by a label; `probe_*` asks whether
a linear map can read the label off. They disagree constantly -- a random
projection already separates clients (`probe_district` near 1.0 at round -1),
which is why the controlled probes exist:

* `month_within_district` decodes month *inside* each client, so district
  identity is constant within a fold and carries no signal.
* `regime_lodo` decodes regime on a HELD-OUT district, so the probe never
  saw that client's fingerprint and can only succeed if the latent encodes
  something the regime shares across districts. That is the actual claim.

Drift metrics are computed and stored but never rank or gate anything: the
drift facet is parked, and `compute_drift_signals` is imported verbatim so
the numbers stay comparable with the pipeline's own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bridge
from worlds import World

__all__ = ["fcols", "latent_metrics", "probe_controlled", "regime_lodo",
           "dispersion", "metrics_by_round", "drift_metrics", "check_run",
           "score_run"]


def fcols(df: pd.DataFrame) -> list[str]:
    """The latent feature columns, `f0..fD` (never `phase` or other f-words)."""
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def _probe(Z: np.ndarray, y: np.ndarray, seed: int = 0):
    """(held-out accuracy, majority-class share) for a linear probe."""
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return np.nan, np.nan
    strat = y if counts.min() >= 2 else None
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.3,
                                          random_state=seed, stratify=strat)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Ztr, ytr)
    chance = float(pd.Series(yte).value_counts(normalize=True).max())
    return float(clf.score(Zte, yte)), chance


def latent_metrics(df: pd.DataFrame, seed: int = 0, sil_n: int = 2000,
                   probe_n: int = 12000) -> dict:
    """Cosine silhouettes, linear probes with chance, and spectrum width."""
    f = fcols(df)
    Z = df[f].to_numpy(np.float64)
    d, m = df["district"].to_numpy(), df["month"].to_numpy()
    rng = np.random.default_rng(seed)
    i = rng.choice(len(Z), min(sil_n, len(Z)), replace=False)
    p = rng.choice(len(Z), min(probe_n, len(Z)), replace=False)
    ev = PCA().fit(Z).explained_variance_ratio_
    acc_d, ch_d = _probe(Z[p], d[p], seed)
    acc_m, ch_m = _probe(Z[p], m[p], seed)
    return {
        "n": int(len(Z)),
        "sil_district": float(silhouette_score(Z[i], d[i], metric="cosine")),
        "sil_month": float(silhouette_score(Z[i], m[i], metric="cosine")),
        "probe_district": acc_d, "chance_district": ch_d,
        "probe_month": acc_m, "chance_month": ch_m,
        "pca90": int(np.searchsorted(np.cumsum(ev), 0.90) + 1),
        "eff_dim": float(1.0 / (ev ** 2).sum()),
    }


def probe_controlled(df: pd.DataFrame, world: World, seed: int = 0,
                     probe_n: int = 12000, phase: str = "init",
                     n_perm: int = 100) -> dict:
    """The two probes that cannot ride on the district fingerprint."""
    f = fcols(df)
    idx = np.random.default_rng(seed).choice(len(df), min(probe_n, len(df)),
                                            replace=False)
    d = df.iloc[idx]
    Z = d[f].to_numpy(np.float64)
    dist, month = d["district"].to_numpy(), d["month"].to_numpy()

    accs, chs = [], []
    for c in np.unique(dist):
        k = dist == c
        a, ch = _probe(Z[k], month[k], seed)
        if not np.isnan(a):
            accs.append(a)
            chs.append(ch)
    out = {"n": int(len(Z)),
           "month_within_district": float(np.mean(accs)) if accs else np.nan,
           "chance_month_within": float(np.mean(chs)) if chs else np.nan}
    out.update(regime_lodo(df, world, phase=phase, seed=seed,
                           probe_n=probe_n, n_perm=n_perm))
    return out


def regime_lodo(df: pd.DataFrame, world: World, phase: str = "init",
                seed: int = 0, probe_n: int = 8000,
                n_perm: int = 100) -> dict:
    """Leave-one-district-out regime probe with a label-permutation null.

    Restricted to the phase's label-clean months. The null shuffles the
    district->regime ASSIGNMENT, not per-window labels: per-window shuffling
    would break the district blocks and give a null that is far too
    optimistic.
    """
    reg = world.regimes(phase=phase)
    months = world.phase_months(phase)
    d = df[df["month"].isin(months)]
    if not len(d) or not reg:
        return {"regime_lodo": np.nan, "regime_lodo_p": np.nan,
                "regime_lodo_null": np.nan, "regime_lodo_folds": 0}

    idx = np.random.default_rng(seed).choice(len(d), min(probe_n, len(d)),
                                             replace=False)
    d = d.iloc[idx]
    Z = d[fcols(d)].to_numpy(np.float64)
    dist = d["district"].to_numpy()
    clients = sorted(set(dist) & set(reg))

    def score(mapping):
        accs, y = [], np.array([mapping[c] for c in dist])
        for c in clients:
            tr, te = dist != c, dist == c
            others = {mapping[k] for k in clients if k != c}
            if len(np.unique(y[tr])) < 2 or mapping[c] not in others:
                continue                    # its regime has no other example
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            clf.fit(Z[tr], y[tr])
            accs.append(float(clf.score(Z[te], y[te])))
        return (float(np.mean(accs)) if accs else np.nan), len(accs)

    real, folds = score({c: reg[c] for c in clients})
    toks = [reg[c] for c in clients]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_perm)):
        perm = rng.permutation(toks)
        s, k = score({c: perm[i] for i, c in enumerate(clients)})
        if k and not np.isnan(s):
            null.append(s)
    null = np.asarray(null, float)
    return {"regime_lodo": real,
            "regime_lodo_p": (float((null >= real).mean())
                              if len(null) and np.isfinite(real) else np.nan),
            "regime_lodo_null": float(null.mean()) if len(null) else np.nan,
            "regime_lodo_folds": int(folds)}


def dispersion(df: pd.DataFrame, by: str = "district") -> pd.DataFrame:
    """Between- vs within-group scatter -- a scale-free separability ratio."""
    f = fcols(df)
    Z = df[f].to_numpy(np.float64)
    g = df[by].to_numpy()
    grand = Z.mean(0)
    rows = []
    for u in np.unique(g):
        Zi = Z[g == u]
        rows.append({by: u, "n": len(Zi),
                     "between": float(np.linalg.norm(Zi.mean(0) - grand)),
                     "within": float(np.linalg.norm(Zi - Zi.mean(0),
                                                    axis=1).mean())})
    out = pd.DataFrame(rows).set_index(by)
    out["ratio"] = out["between"] / out["within"]
    return out


def metrics_by_round(snapshots: pd.DataFrame | None,
                     stage: str = "pre") -> pd.DataFrame | None:
    """`latent_metrics` per snapshot round. Round -1 is the untrained control."""
    if snapshots is None or not len(snapshots):
        return None
    d = snapshots
    if "stage" in d.columns:
        d = d[d["stage"] == stage]
    if not len(d):
        return None
    return pd.DataFrame([{"round": int(r), "stage": stage, **latent_metrics(g)}
                         for r, g in d.groupby("round")])


def drift_metrics(prototype_history: pd.DataFrame, fl: dict,
                  world: World) -> dict:
    """Prototype drift, computed and stored -- never used to rank cells.

    `drift_rank` 1 means the true drifted district moved most; `drift_sep` is
    in robust units of the other clients' spread (median/MAD, so one
    collapsed-variance client cannot inflate it); `others_sd` is reported so
    a near-zero denominator is visible rather than silently amplifying.
    """
    truth = world.drift_district
    start = world.first_switch
    nan = {"drift_rank": np.nan, "drift_sep": np.nan, "drift_gap": np.nan,
           "others_sd": np.nan, "drift_auroc": np.nan}
    if truth is None or start is None:
        return nan
    ds = bridge.compute_drift_signals(prototype_history, fl)
    if truth not in set(ds["client"]):
        return nan

    post = ds[ds["month"] >= start]
    m = post.groupby("client")["delta_first"].mean().abs()
    others = m.drop(truth)
    mad = float(np.median(np.abs(others - others.median()))) \
        or float(others.mean()) or 1.0
    tgt = ds[ds["client"] == truth]
    return {
        "drift_rank": float(m.rank(ascending=False)[truth]),
        "drift_sep": float((m[truth] - others.median()) / mad),
        "drift_gap": float(m[truth] / (others.max() or np.nan)),
        "others_sd": float(others.std(ddof=0)),
        "drift_auroc": _auroc(tgt["month"].to_numpy() >= start,
                              tgt["delta_first"].abs().to_numpy()),
    }


def _auroc(y, s) -> float:
    from scipy import stats as sps
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return np.nan
    r = sps.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------
# structural checks
# --------------------------------------------------------------------------
def check_run(res: dict, world: World) -> pd.DataFrame:
    """Are the artifacts well-formed and mutually consistent?

    The check that matters: `extract_prototypes` on a client's final model
    must reproduce the per-month mean of `latent_trajectories` -- prototypes
    and the latent table are the same object computed the same way, in eval
    mode. It also pins the protocol asymmetry: `prototype_history`'s final
    round is pre-FedAvg, `latent_trajectories` is post-FedAvg.
    """
    fl = res["fl"]
    D = 2 * int(fl["model"]["lstm_units"])
    lt, ph = res["latent_trajectories"], res["prototype_history"]
    gph, snaps = res["global_prototype_history"], res.get("snapshots")
    lf, pf = fcols(lt), fcols(ph)
    fw = res.get("fl_windows") or {}
    n_win = sum(len(d["windows"]) for d in fw.values()) if fw else len(lt)
    rounds = int(fl["training"]["rounds"])
    n_months = {c: len(np.unique(d["labels"])) for c, d in fw.items()}
    exp_proto = (rounds * sum(n_months.values())
                 if fw and fl["training"]["participation"] == 1.0 else None)

    checks = [("latent_dim", len(lf), len(lf) == D),
              ("prototype_dim", len(pf), len(pf) == D),
              ("latent_rows == windows", len(lt), len(lt) == n_win),
              ("latents_finite", True,
               bool(np.isfinite(lt[lf].to_numpy()).all())),
              ("prototypes_finite", True,
               bool(np.isfinite(ph[pf].to_numpy()).all())),
              ("proto_rows", len(ph),
               exp_proto is None or len(ph) == exp_proto),
              ("global_proto_rounds", int(gph["round"].nunique()) if len(gph) else 0,
               (not len(gph)) or int(gph["round"].nunique()) == rounds)]

    tr = res.get("trainer")
    if tr is not None and fw:
        gw = tr.global_model.state_dict()
        import torch
        synced = all(torch.equal(gw[k], tr.models[c].state_dict()[k])
                     for c in tr.clients for k in gw)
        checks.append(("clients_synced_after_fedavg", synced, synced))

        c0 = tr.clients[0]
        w0, l0 = tr.data[c0]
        p0 = bridge.extract_prototypes(tr.models[c0], w0, l0.cpu().numpy(),
                                       fl["training"]["batch_size"])
        g0 = lt[lt["district"] == c0].groupby("month")[lf].mean()
        err = max(float(np.abs(p0[m] - g0.loc[m].to_numpy()).max())
                  for m in p0 if m in g0.index)
        checks.append(("prototypes == eval-mode mean latent",
                       round(err, 8), err < 1e-4))

    if snaps is not None and "stage" in snaps.columns:
        stages = set(snaps["stage"].unique())
        checks.append(("snapshots have pre and post", sorted(stages),
                       {"pre", "post"} <= stages))
        init_n = int((snaps["month"].isin(world.phase_months("init"))).sum())
        checks.append(("init-phase snapshot rows", init_n, init_n > 0))

    out = pd.DataFrame(checks, columns=["check", "value", "passed"])
    return out


# --------------------------------------------------------------------------
# the scoring entry point
# --------------------------------------------------------------------------
def score_run(res: dict, world: World, run_key: str | None = None,
              stage: str = "pre") -> pd.DataFrame:
    """Long-form latent metrics for one run: one row per (round, stage).

    Reads persisted tables only, so it can be re-run over the whole grid
    without retraining. `round = 9999` is the final model on ALL windows
    (`latent_trajectories`), which is not a snapshot round.
    """
    key = run_key or res.get("run_key") or (res.get("meta") or {}).get("run_key")
    snaps = res.get("snapshots")
    if snaps is None:                    # a cache hit names it by artifact
        snaps = res.get("latents_by_round")
    rows = []

    br = metrics_by_round(snaps, stage=stage)
    if br is not None:
        for r in br.to_dict("records"):
            rows.append(r)

    lt = res.get("latent_trajectories")
    if lt is not None and len(lt):
        rows.append({"round": 9999, "stage": "final_all_windows",
                     **latent_metrics(lt)})

    if not rows:
        return pd.DataFrame()

    ph, fl = res.get("prototype_history"), res.get("fl")
    drift = drift_metrics(ph, fl, world) if (ph is not None and fl) else {}

    out = pd.DataFrame(rows)
    out.insert(0, "run_key", key)
    out.insert(0, "world_hash", world.sim_hash)
    for k, v in drift.items():
        out[k] = v
    return out
