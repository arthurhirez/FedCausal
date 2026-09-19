"""What a prototype remembers: decoded prototypes against the data.

Torch-free. `train.reconstruct_prototypes` does the one step that needs the
model (latent -> window); everything here compares those decoded windows to
what the clients actually saw, month by month, and asks whether the
district MIXTURE a sensor is labelled with can be read back out of them.

Objects (all in the model's own units: aggregated, scaled windows)
------------------------------------------------------------------
For client c, month m, channel i (a slot; see `data.client_frames`):

* ``T[c,m][:, i]`` -- TRUE mean window: the mean of client c's windows
  labelled m. The reference every decoding is scored against.
* ``R_s[c,m][:, i]`` -- a DECODED prototype, for source s in
  {local, post_fedavg, post_fedavg_full, finch}. `finch` is the FINCH
  cluster centroid client c's local prototype falls in (see
  `finch_membership`).
* ``D_k[c,m]`` -- DEMAND window of district k: district k's total demand,
  mean-aggregated to the window resolution and cut at exactly client c's
  window start steps for month m, then averaged. Same alignment as T, so a
  window that starts at 03:00 is compared with demand that starts at 03:00.

Metrics
-------
All shape metrics use z-scored windows, ``z(x) = (x - mean x) / std x``
over the W steps: the scaler already destroys level, and a sensor responds
to demand with a sign (flow up, pressure down) and a gain, so only shape is
comparable.

``fidelity``
    ``rmse(R, T) = sqrt(mean_{t,i} (R - T)^2)`` -- how well the decoded
    prototype reproduces the mean window it summarises. Low = the
    prototype still carries the month's waveform.

``r2_own``, ``r2_mix``
    ``r2_X = corr(z(Y_i), z(X))^2`` for ``X = D_c`` (own district) and
    ``X = M_i = sum_k w_ik z(D_k)`` (the sensor's mixture-weighted demand
    SHAPE; ``w_ik`` = the label stack's share of district k at sensor i,
    normalised to sum to 1). Shapes are mixed, not raw demands: the labels
    are shares of each district's shape change per unit excitation, so a
    316-junction district must not outweigh a 30-junction one by volume.
    The squared correlation is the share of the channel's variance a linear
    map of X explains, sign-agnostic.

``gain = r2_mix - r2_own``
    Does the mixture explain the channel better than the own district
    alone? The hypothesis: gain > 0 for `mixed` slots, ~0 for `pure` slots
    (their mixture IS the own district), and strongest in the `final` phase
    of a drifted world, when the drifted district's shape differs from the
    others. For ``Y = T`` this is the ceiling the data allows; for
    ``Y = R_s`` it says whether the prototype kept that signal.

``room = 1 - corr(z(D_c), z(M_i))^2``
    How much of the mixture's shape the own district does NOT already
    carry -- the most any gain could be. room ~ 0 when every district in
    the sensor's mixture shares the own district's regime (e.g. a mixed
    sensor whose weights sit on districts that never drifted): gain is then
    ~0 by construction and says nothing. Read `gain` only where `room` is
    clearly above 0.

``dtw_own``, ``dtw_mix``, ``dtw_gain = dtw_own - dtw_mix``
    The same question with a DTW distance (Sakoe-Chiba band of ``band``
    steps, squared cost, normalised by W), after flipping X's sign to match
    the channel's correlation sign. DTW forgives small timing shifts
    (hydraulic lag, the 3-h aggregation, a decoder that smears a peak) that
    penalise correlation; ``dtw_gain > 0`` means the mixture is closer.

``separability``
    ``1 - mean_{j<k} corr(z(D_j), z(D_k))`` per (client, month): how
    different the districts' demand SHAPES are. When every district has the
    same regime (an all-residential map before drift), the D_k are nearly
    identical, M_i ~ D_c for every weighting, and gain is ~0 BY
    CONSTRUCTION. A network-level view; `room` is the per-sensor one.

Limits (this is a first POC)
----------------------------
Linear, instantaneous, one lag-free mixture; the mixture labels were
measured on weekly shape profiles, not on 252-h windows; DTW band fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .worlds import World

__all__ = ["true_windows", "demand_windows", "channel_table", "as_arrays",
           "channel_weights", "mixture_demand", "phase_demand",
           "finch_membership", "finch_arrays", "phase_mean", "zs", "dtw",
           "fidelity_table", "retrieval_table", "separability",
           "phase_summary", "PERIODS"]

PERIODS = ("init", "transition", "final")


# --------------------------------------------------------------------------
# references: sensors and demand, in window space
# --------------------------------------------------------------------------
def true_windows(fl_windows: dict) -> dict[tuple, np.ndarray]:
    """{(client, month): (W, F)} mean scaled window per month label."""
    out = {}
    for c, d in fl_windows.items():
        X, lab = np.asarray(d["windows"]), np.asarray(d["labels"])
        for m in np.unique(lab):
            out[(c, int(m))] = X[lab == m].mean(axis=0)
    return out


def demand_windows(world: World, fl_windows: dict, fl: dict
                   ) -> dict[tuple, dict[str, np.ndarray]]:
    """{(client, month): {district: (W,)}} -- district demand cut at the
    client's own window starts for that month, mean-aggregated like the
    sensors (`interval_agg_h`), then averaged over those windows."""
    pre = fl["preprocessing"]
    k = int(round(pre["interval_agg_h"] / float(world.time["resolution_h"])))
    W = int(pre["window_size"])
    dem = world.district_demand
    names = [c for c in dem.columns if c != "month"]
    raw = dem[names].to_numpy(float)
    n_agg = len(raw) // k
    agg = raw[:n_agg * k].reshape(n_agg, k, len(names)).mean(axis=1)
    out = {}
    for c, d in fl_windows.items():
        starts = np.asarray(d["window_start_step"]) // k
        lab = np.asarray(d["labels"])
        for m in np.unique(lab):
            s = starts[lab == m]
            s = s[s + W <= n_agg]
            if not len(s):
                continue
            stack = np.stack([agg[i:i + W] for i in s])    # (n, W, K)
            mean = stack.mean(axis=0)
            out[(c, int(m))] = {nm: mean[:, j] for j, nm in enumerate(names)}
    return out


def channel_table(world: World, fl_windows: dict) -> pd.DataFrame:
    """One row per (client, channel): slot, sensor, kind, slot_class and the
    label mixture ``w_<district>`` (NaN on a `manual` world, which has
    none)."""
    sp = world.sensor_placement
    wcols = [f"w_{d}" for d in world.districts]
    rows = []
    for c, d in fl_windows.items():
        sub = sp[sp["district"] == c].set_index("slot")
        for i, slot in enumerate(d["sensors"]):
            r = sub.loc[slot] if slot in sub.index else None
            row = {"client": c, "channel": i, "slot": slot,
                   "sensor": None if r is None else r["sensor"],
                   "kind": None if r is None else r["kind"],
                   "slot_class": None if r is None else r["slot_class"]}
            for w in wcols:
                row[w] = (float(r[w]) if r is not None and w in sub.columns
                          and pd.notna(r[w]) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# decoded prototypes -> arrays
# --------------------------------------------------------------------------
def as_arrays(recon: pd.DataFrame, scope: str, round_: int | None = None
              ) -> dict[tuple, np.ndarray]:
    """`train.reconstruct_prototypes` long output -> {(client, month): (W, F)}
    for one per-client scope (local / post_fedavg / post_fedavg_full), at
    `round_` (default: that scope's last round)."""
    g = recon[recon["scope"] == scope]
    if not len(g):
        return {}
    r = int(g["round"].max()) if round_ is None else int(round_)
    g = g[g["round"] == r]
    out = {}
    for (c, m), h in g.groupby(["client", "month"]):
        out[(c, int(m))] = (h.pivot_table(index="step", columns="channel",
                                          values="value").to_numpy())
    return out


def _fcols(df):
    return [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]


def finch_membership(prototypes: pd.DataFrame, round_: int | None = None
                     ) -> pd.DataFrame:
    """Which FINCH cluster each client's local prototype belongs to.

    FINCH's own assignment is not persisted (only the centroids are), so
    membership is recovered as the nearest centroid by cosine -- the metric
    FINCH clustered with. ``cos`` is that similarity; ``n_members`` how many
    clients share the cluster that month (1 = the client kept its own).
    """
    p = prototypes
    r = int(p.loc[p["scope"] == "global", "round"].max()) if round_ is None \
        else int(round_)
    loc = p[(p["scope"] == "local") & (p["round"] == r)]
    glo = p[(p["scope"] == "global") & (p["round"] == r)]
    f = _fcols(p)
    rows = []
    for m, lm in loc.groupby("month"):
        gm = glo[glo["month"] == m]
        if not len(gm):
            continue
        G = gm[f].to_numpy(float)
        G = G / np.linalg.norm(G, axis=1, keepdims=True).clip(1e-12)
        for _, row in lm.iterrows():
            v = row[f].to_numpy(float)
            v = v / max(np.linalg.norm(v), 1e-12)
            sims = G @ v
            j = int(np.argmax(sims))
            rows.append({"round": r, "month": int(m), "client": row["client"],
                         "cluster": int(gm["cluster"].iloc[j]),
                         "cos": float(sims[j]), "n_clusters": len(gm)})
    out = pd.DataFrame(rows)
    if len(out):
        out["n_members"] = out.groupby(["month", "cluster"])["client"] \
            .transform("count")
    return out


def finch_arrays(recon: pd.DataFrame, membership: pd.DataFrame
                 ) -> dict[tuple, np.ndarray]:
    """{(client, month): decoded centroid of the client's FINCH cluster}."""
    g = recon[recon["scope"] == "global"]
    out = {}
    for r in membership.itertuples(index=False):
        h = g[(g["round"] == r.round) & (g["month"] == r.month)
              & (g["cluster"] == r.cluster)]
        if len(h):
            out[(r.client, int(r.month))] = h.pivot_table(
                index="step", columns="channel", values="value").to_numpy()
    return out


def phase_mean(arrays: dict[tuple, np.ndarray], world: World, phase: str
               ) -> dict[str, np.ndarray]:
    """{client: mean over the months of `phase`} -- the period view."""
    out: dict[str, list] = {}
    for (c, m), a in arrays.items():
        if world.phase_of(int(m)) == phase:
            out.setdefault(c, []).append(a)
    return {c: np.mean(v, axis=0) for c, v in out.items()}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def zs(x) -> np.ndarray:
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def _r(a, b) -> float:
    a, b = zs(a), zs(b)
    return float(np.mean(a * b)) if a.any() and b.any() else 0.0


def channel_weights(info, wcols) -> dict[str, float] | None:
    """{district: w} normalised to 1, or None when the channel has no
    usable mixture (manual world, or an unresponsive fill with all-zero
    weights)."""
    w = np.array([info[x] for x in wcols], float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    w = w / w.sum()
    return {x[2:]: float(v) for x, v in zip(wcols, w)}


def mixture_demand(dm: dict[str, np.ndarray], weights: dict[str, float]
                   ) -> np.ndarray:
    """M = sum_k w_k z(D_k): the mixture-weighted demand SHAPE (see the
    module docstring for why shapes, not volumes)."""
    return sum(w * zs(dm[k]) for k, w in weights.items())


def phase_demand(demand: dict, world: World, phase: str
                 ) -> dict[str, dict[str, np.ndarray]]:
    """{client: {district: mean demand window over the phase's months}}."""
    acc: dict[str, dict[str, list]] = {}
    for (c, m), dm in demand.items():
        if world.phase_of(int(m)) != phase:
            continue
        for k, v in dm.items():
            acc.setdefault(c, {}).setdefault(k, []).append(v)
    return {c: {k: np.mean(v, axis=0) for k, v in d.items()}
            for c, d in acc.items()}


def dtw(a, b, band: int = 2) -> float:
    """DTW distance, squared cost, Sakoe-Chiba band, normalised by len(a):
    sqrt(D[n, n] / n)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(
                D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, n] / n))


def fidelity_table(sources: dict[str, dict], true: dict, world: World
                   ) -> pd.DataFrame:
    """rmse(R_s, T) per (source, client, month, channel)."""
    rows = []
    for s, arrs in sources.items():
        for (c, m), R in arrs.items():
            T = true.get((c, m))
            if T is None:
                continue
            for i in range(min(R.shape[1], T.shape[1])):
                rows.append({"source": s, "client": c, "month": m,
                             "phase": world.phase_of(m), "channel": i,
                             "rmse": float(np.sqrt(np.mean(
                                 (R[:, i] - T[:, i]) ** 2)))})
    return pd.DataFrame(rows)


def retrieval_table(sources: dict[str, dict], demand: dict,
                    channels: pd.DataFrame, world: World,
                    band: int = 2) -> pd.DataFrame:
    """Own vs mixture demand, per (source, client, month, channel).

    `sources` maps a name to {(client, month): (W, F)} -- pass the true
    windows as `"true"` to get the ceiling alongside the decoded ones.
    Channels without a mixture (manual world) are skipped.
    """
    wcols = [c for c in channels.columns if c.startswith("w_")]
    ch = channels.set_index(["client", "channel"])
    rows = []
    for s, arrs in sources.items():
        for (c, m), Y in arrs.items():
            dm = demand.get((c, m))
            if dm is None:
                continue
            own = dm[c]
            for i in range(Y.shape[1]):
                info = ch.loc[(c, i)]
                weights = channel_weights(info, wcols)
                if weights is None:
                    continue
                mix = mixture_demand(dm, weights)
                y = Y[:, i]
                r_own, r_mix = _r(y, own), _r(y, mix)
                d_own = dtw(zs(y), np.sign(r_own or 1) * zs(own), band)
                d_mix = dtw(zs(y), np.sign(r_mix or 1) * zs(mix), band)
                rows.append({
                    "source": s, "client": c, "month": m,
                    "phase": world.phase_of(m), "channel": i,
                    "slot": info["slot"], "kind": info["kind"],
                    "slot_class": info["slot_class"],
                    "w_own": float(info.get(f"w_{c}", np.nan)),
                    "r2_own": r_own ** 2, "r2_mix": r_mix ** 2,
                    "gain": r_mix ** 2 - r_own ** 2,
                    "room": 1.0 - _r(own, mix) ** 2,
                    "dtw_own": d_own, "dtw_mix": d_mix,
                    "dtw_gain": d_own - d_mix})
    return pd.DataFrame(rows)


def separability(demand: dict, world: World) -> pd.DataFrame:
    """1 - mean pairwise shape correlation of the district demand windows."""
    rows = []
    for (c, m), dm in demand.items():
        names = list(dm)
        cors = [_r(dm[a], dm[b]) for i, a in enumerate(names)
                for b in names[i + 1:]]
        rows.append({"client": c, "month": m, "phase": world.phase_of(m),
                     "separability": 1.0 - float(np.mean(cors))})
    return pd.DataFrame(rows)


def phase_summary(tbl: pd.DataFrame, by=("source", "phase", "kind",
                                          "slot_class")) -> pd.DataFrame:
    """Mean of every metric per group, phases in init/transition/final
    order, with the row count (`n`)."""
    if not len(tbl):
        return tbl
    by = [b for b in by if b in tbl.columns]
    num = [c for c in tbl.columns if c not in by and c not in (
        "client", "month", "channel", "slot") and
        pd.api.types.is_numeric_dtype(tbl[c])]
    out = tbl.groupby(by, dropna=False)[num].mean()
    out["n"] = tbl.groupby(by, dropna=False).size()
    out = out.reset_index()
    if "phase" in out.columns:
        out["phase"] = pd.Categorical(out["phase"], PERIODS, ordered=True)
        out = out.sort_values([b for b in by if b != "phase"] + ["phase"])
    return out.reset_index(drop=True)
