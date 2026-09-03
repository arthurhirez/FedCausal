"""Sensor-placement POC: does WHERE we measure change WHAT dependence we detect?

Placement is a column selection over `pressures` / `flows`, so every strategy is
scored on the SAME simulated world -- no re-simulation, no new pipeline.

Three rules this POC is built around:

1. **No ground truth in the loop.** A strategy sees network topology and each
   client's own residual series. `gt_boundaries` enters only at scoring time.
   (Corollary: run this on `coupling.variant: baseline`, where every boundary
   is open. Under `partial`, closed pipes are both a topological fact and the
   evaluation label, so a topology-driven strategy leaks.)
2. **Federated template.** Every district gets the same channel roles in the
   same order, and a physical element is metered by at most one district --
   two clients sharing a pipe would make their coupling trivially perfect.
3. **Full horizon for estimation.** The 84-step window is an FL-representation
   constraint, not an estimation one. `sc_sample_size` shows the cost of the
   alternative rather than asserting it.

Placement is scored with the district's *sensor matrix* (RV, partial RV), not
only the district-mean signal: averaging a district's sensors into one series
is exactly the operation that erases the placement effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from fedwater.pipelines.dependence_detection import fed_methods as F
from fedwater.pipelines.dependence_oracle import methods as M
from fedwater.pipelines.dependence_oracle.nodes import deseasonalize
from fedwater.pipelines.sensing.nodes import (add_measurement_noise,
                                              extract_sensor_series)

TEMPLATE = {"pressure": 2, "flow": 3}      # BEPE-compatible; equal per client


# ==========================================================================
# candidate inventory
# ==========================================================================
def node_to_district(districts: dict) -> dict:
    return {n: d for d, nodes in districts["districts"].items() for n in nodes}


def candidate_table(wn, districts: dict) -> pd.DataFrame:
    """One row per (district, candidate sensor).

    A boundary pipe is a candidate for BOTH adjacent districts -- either side
    may meter the interconnection -- but `select` grants it to only one.
    """
    n2d = node_to_district(districts)
    rows = []
    for n in wn.junction_name_list:
        rows.append(dict(kind="pressure", element=n, district=n2d[n],
                         scope="node", nodes=(n,)))
    for name in wn.pipe_name_list:
        p = wn.get_link(name)
        u, v = p.start_node_name, p.end_node_name
        da, db = n2d.get(u), n2d.get(v)
        if da is None or db is None:          # source / tank links
            continue
        scope = "internal" if da == db else "boundary"
        for d in ({da} if da == db else {da, db}):
            rows.append(dict(kind="flow", element=name, district=d,
                             scope=scope, nodes=(u, v)))
    cands = pd.DataFrame(rows)
    cands["cid"] = cands["kind"].str[0].map({"p": "p", "f": "q"}) \
        + "_" + cands["element"]
    return cands.reset_index(drop=True)


def is_closed(link) -> bool:
    return "closed" in str(getattr(link, "initial_status", "")).lower()


# ==========================================================================
# topology features (structural only -- no simulated data)
# ==========================================================================
def network_graph(wn):
    import networkx as nx
    G = nx.Graph()
    for name in wn.pipe_name_list:
        p = wn.get_link(name)
        if is_closed(p):
            continue
        G.add_edge(p.start_node_name, p.end_node_name,
                   weight=max(float(p.length), 1.0), pipe=name)
    return G


def topology_features(wn, districts: dict, cands: pd.DataFrame):
    """(feats, dist) -- per-candidate structural scores + candidate distances.

    d_boundary : weighted shortest path to the nearest inter-district pipe.
    carry      : source->junction edge betweenness (proxy for the share of
                 network demand that transits the element).
    dist       : candidate x candidate graph distance, used as the
                 non-redundancy term in place of an estimated entropy.
    """
    import networkx as nx

    G = network_graph(wn)
    n2d = node_to_district(districts)
    sources = [n for n in (wn.reservoir_name_list + wn.tank_name_list)
               if n in G]
    targets = [n for n in wn.junction_name_list if n in G]

    bet = nx.edge_betweenness_centrality_subset(
        G, sources=sources, targets=targets, weight="weight", normalized=True)
    carry_edge = {}
    for (a, b), val in bet.items():
        carry_edge[G[a][b]["pipe"]] = val

    bnodes = {n for n in G for m in G[n]
              if n2d.get(n) and n2d.get(m) and n2d[n] != n2d[m]}
    dbound = (nx.multi_source_dijkstra_path_length(G, bnodes, weight="weight")
              if bnodes else {})

    apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    big = max((max(v.values()) for v in apsp.values()), default=1.0)

    def pos(row):
        return [n for n in row["nodes"] if n in G]

    feats = []
    for _, r in cands.iterrows():
        ns = pos(r)
        db = min((dbound.get(n, np.inf) for n in ns), default=np.inf)
        if r["kind"] == "flow":
            carry = carry_edge.get(r["element"], 0.0)
        else:
            carry = max((carry_edge[G[r["element"]][m]["pipe"]]
                         for m in G.adj.get(r["element"], {})), default=0.0)
        feats.append(dict(d_boundary=min(db, big), carry=carry,
                          reachable=len(ns) > 0))
    feats = pd.DataFrame(feats, index=cands.index)

    idx = cands.index.to_numpy()
    node_lists = [pos(cands.loc[i]) for i in idx]
    D = np.full((len(idx), len(idx)), big, float)
    for a in range(len(idx)):
        for b in range(a, len(idx)):
            vals = [apsp[x].get(y, big)
                    for x in node_lists[a] for y in node_lists[b]]
            D[a, b] = D[b, a] = min(vals) if vals else big
    dist = pd.DataFrame(D, index=idx, columns=idx)
    return feats, dist


def residual_stats(pressures: pd.DataFrame, flows: pd.DataFrame,
                   cands: pd.DataFrame, steps_day: int) -> pd.DataFrame:
    """Client-local screen: std of each candidate's month-aware residual.

    Legitimate for a strategy to use (each district owns its own elements) and
    the cheapest way to drop dead channels -- a closed pipe has ~zero variance.
    """
    month = pressures["month"]
    out = {}
    for src, kind in ((pressures, "pressure"), (flows, "flow")):
        cols = [c for c in cands.loc[cands["kind"] == kind, "element"].unique()
                if c in src.columns]
        resid = deseasonalize(src[cols], steps_day, month, month_aware=True)
        out.update({(kind, c): float(resid[c].std()) for c in cols})
    std = cands.apply(lambda r: out.get((r["kind"], r["element"]), np.nan),
                      axis=1)
    return pd.DataFrame({"resid_std": std}, index=cands.index)


# ==========================================================================
# selection: one greedy, five strategies
# ==========================================================================
def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def select(cands: pd.DataFrame, gain: pd.Series, dist: pd.DataFrame,
           lam: float, rng, template: dict = TEMPLATE,
           mask: pd.Series | None = None) -> dict:
    """Round-robin greedy under the federated template.

    gain : per-candidate score (higher = better), indexed like `cands`.
    lam  : weight on the redundancy penalty exp(-d_graph / scale).
    mask : optional per-candidate eligibility (e.g. a variance floor).
    """
    scale = float(np.median(dist.to_numpy())) or 1.0
    eligible = cands.index if mask is None else cands.index[mask.astype(bool)]
    pool_by = {(d, k): [i for i in eligible
                        if cands.at[i, "district"] == d
                        and cands.at[i, "kind"] == k]
               for d in sorted(cands["district"].unique()) for k in template}

    placement = {d: {k: [] for k in template}
                 for d in sorted(cands["district"].unique())}
    chosen = {d: {k: [] for k in template} for d in placement}
    taken: set = set()

    for kind, k in template.items():
        for _ in range(k):
            for d in placement:
                pool = [i for i in pool_by[(d, kind)]
                        if (kind, cands.at[i, "element"]) not in taken]
                if not pool:
                    raise ValueError(f"{d}/{kind}: candidate pool exhausted")
                sel = chosen[d][kind]
                pen = [max((np.exp(-dist.at[i, j] / scale) for j in sel),
                           default=0.0) for i in pool]
                score = (gain.reindex(pool).fillna(0.0).to_numpy()
                         - lam * np.asarray(pen)
                         + rng.normal(0, 1e-9, len(pool)))
                best = pool[int(np.argmax(score))]
                chosen[d][kind].append(best)
                taken.add((kind, cands.at[best, "element"]))
                placement[d][kind].append(cands.at[best, "element"])
    return placement


def strategies(cands, feats, stats, dist, params, seed=0) -> dict:
    """{name: placement}. Every strategy is topology + client-local only."""
    rng = lambda: np.random.default_rng(seed)          # noqa: E731
    floor = stats["resid_std"] > 1e-9                  # drop dead channels
    zero = pd.Series(0.0, index=cands.index)

    out = {"manual": {d: {k: [str(x) for x in v[k]] for k in TEMPLATE}
                      for d, v in params["sensors"].items()}}
    out["random"] = select(cands, pd.Series(rng().random(len(cands)),
                                            index=cands.index),
                           dist, 0.0, rng(), mask=floor)
    out["variance"] = select(cands, _z(stats["resid_std"].fillna(0.0)),
                             dist, 0.0, rng(), mask=floor)
    out["spread"] = select(cands, zero, dist, 1.0, rng(), mask=floor)
    out["boundary"] = select(
        cands,
        0.5 * _z(feats["carry"]) - 0.5 * _z(feats["d_boundary"]),
        dist, 0.5, rng(), mask=floor)
    return out


# ==========================================================================
# evaluation
# ==========================================================================
def build_series(pressures, flows, placement, noise=None, seed=42):
    ss = extract_sensor_series(pressures, flows, placement)
    if noise is None:
        ss = ss.assign(observed=ss["value"])           # reproducible headline
    else:
        ss = add_measurement_noise(ss, noise, seed)
    return ss


def district_data(sensor_series: pd.DataFrame, steps_day: int,
                  month_aware: bool = True):
    """(mats, sigs): per (district, kind) the sensor MATRIX and its mean signal.

    `month_aware=False` fits ONE profile across the whole window instead of
    per-month. On a short single-regime window that is the better choice:
    per-month profiles there are estimated from ~30 days each and start
    absorbing the very signal being measured.
    """
    dup = sensor_series.duplicated(["step", "sensor"]).any()
    if dup:
        raise AssertionError("a sensor is shared by two districts")
    wide = sensor_series.pivot(index="step", columns="sensor",
                               values="observed")
    month = (sensor_series.drop_duplicates("step")
             .set_index("step")["month"].sort_index())
    resid = deseasonalize(wide, steps_day, month, month_aware=month_aware)
    key = (sensor_series.drop_duplicates("sensor")
           .set_index("sensor")[["district", "kind"]])
    mats, sigs = {}, {}
    for (d, k), grp in key.groupby(["district", "kind"]):
        cols = sorted(grp.index)
        mats[(d, k)] = resid[cols].to_numpy(float)
        sigs[(d, k)] = mats[(d, k)].mean(axis=1)
    return mats, sigs


def pair_scores(mats: dict, sigs: dict, max_lag=24, lags=24, n_sur=99,
                seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kind in sorted({k for _, k in sigs}):
        ds = sorted(d for d, k in sigs if k == kind)
        P = M.precision_partial_corr(np.column_stack([sigs[(d, kind)]
                                                      for d in ds]))
        for i, a in enumerate(ds):
            for j in range(i + 1, len(ds)):
                b = ds[j]
                X, Y = mats[(a, kind)], mats[(b, kind)]
                Z = np.column_stack([mats[(c, kind)] for c in ds
                                     if c not in (a, b)])
                r, p_r = M.circular_shift_pvalue(
                    M.pearson, sigs[(a, kind)], sigs[(b, kind)], n_sur, rng)
                rv, p_rv = F.roll_pvalue(M.rv_coefficient, X, Y, n_sur, rng)
                prv, p_prv = F.roll_pvalue(
                    lambda x, y, Z=Z: F.partial_rv(x, y, Z), X, Y, n_sur, rng)
                f_ab = M.granger_f(sigs[(a, kind)], sigs[(b, kind)], lags)
                f_ba = M.granger_f(sigs[(b, kind)], sigs[(a, kind)], lags)
                _, lag = M.max_lagged_xcorr(sigs[(a, kind)], sigs[(b, kind)],
                                            max_lag)
                base = dict(kind=kind, district_a=a, district_b=b)
                rows += [
                    {**base, "method": "partial_corr", "statistic": P[i, j],
                     "p_value": np.nan},
                    {**base, "method": "pearson", "statistic": r,
                     "p_value": p_r},
                    {**base, "method": "rv", "statistic": rv, "p_value": p_rv},
                    {**base, "method": "partial_rv", "statistic": prv,
                     "p_value": p_prv},
                    {**base, "method": "granger_max",
                     "statistic": max(f_ab, f_ba), "p_value": np.nan},
                    {**base, "method": "lead_lag_h", "statistic": float(lag),
                     "p_value": np.nan},
                ]
    return pd.DataFrame(rows)


def adjacency(gt_boundaries: pd.DataFrame, districts: dict) -> pd.DataFrame:
    ds = sorted(districts["districts"])
    open_gt = gt_boundaries[~gt_boundaries["closed"]] \
        if "closed" in gt_boundaries.columns else gt_boundaries
    pairs = set(zip(open_gt["district_a"], open_gt["district_b"]))
    return pd.DataFrame([
        dict(district_a=a, district_b=b, adjacent=(min(a, b), max(a, b)) in pairs)
        for i, a in enumerate(ds) for b in ds[i + 1:]])


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return np.nan
    r = sps.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recovery(scores: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    df = scores.merge(adj, on=["district_a", "district_b"], how="left")
    df = df[df["method"] != "lead_lag_h"]
    out = []
    for (kind, method), g in df.groupby(["kind", "method"]):
        s = g["statistic"].abs().to_numpy()
        y = g["adjacent"].to_numpy(bool)
        out.append(dict(
            kind=kind, method=method, n_pairs=len(g),
            auroc=_auroc(y, s),
            margin=float(np.nanmean(s[y]) - np.nanmean(s[~y])),
            mean_adjacent=float(np.nanmean(s[y])),
            mean_other=float(np.nanmean(s[~y]))))
    return pd.DataFrame(out)


def evaluate(pressures, flows, placement, districts, gt_boundaries, steps_day,
             noise=None, seed=42, n_sur=99):
    ss = build_series(pressures, flows, placement, noise, seed)
    mats, sigs = district_data(ss, steps_day)
    scores = pair_scores(mats, sigs, n_sur=n_sur, seed=seed)
    return scores, recovery(scores, adjacency(gt_boundaries, districts))


# ==========================================================================
# sanity checks
# ==========================================================================
def sc_inventory(wn, districts, cands, placements) -> pd.DataFrame:
    """Partition integrity, pool sizes, template compliance, exclusivity."""
    n2d = node_to_district(districts)
    rows = [
        dict(check="junctions_partitioned",
             value=len(set(wn.junction_name_list) ^ set(n2d)),
             passed=set(wn.junction_name_list) == set(n2d)),
        dict(check="min_pool_pressure",
             value=int(cands[cands.kind == "pressure"]
                       .groupby("district").size().min()),
             passed=True),
        dict(check="min_pool_flow",
             value=int(cands[cands.kind == "flow"]
                       .groupby("district").size().min()),
             passed=True),
    ]
    for name, pl in placements.items():
        shape_ok = all(len(v[k]) == TEMPLATE[k] for v in pl.values()
                       for k in TEMPLATE)
        elems = [(k, e) for v in pl.values() for k in TEMPLATE for e in v[k]]
        rows += [
            dict(check=f"{name}:template", value=len(pl), passed=shape_ok),
            dict(check=f"{name}:exclusive", value=len(elems) - len(set(elems)),
                 passed=len(elems) == len(set(elems))),
        ]
    df = pd.DataFrame(rows)
    df.loc[df.check.str.startswith("min_pool"), "passed"] = \
        df.loc[df.check.str.startswith("min_pool"), "value"] >= 3
    return df


def sc_pressure_vs_flow(pressures, flows, cands, steps_day) -> pd.DataFrame:
    """Quantify 'pressure is nearly identical across clients'.

    within  : mean |corr| between residuals of two elements in the SAME district
    across  : same, for elements in DIFFERENT districts
    A kind whose across == within carries no district-discriminative
    information: every client sees the same signal.
    """
    month = pressures["month"]
    rows = []
    for src, kind in ((pressures, "pressure"), (flows, "flow")):
        sub = cands[cands.kind == kind].drop_duplicates("element")
        cols = [c for c in sub["element"] if c in src.columns]
        d = sub.set_index("element")["district"]
        resid = deseasonalize(src[cols], steps_day, month, month_aware=True)
        keep = [c for c in cols if resid[c].std() > 1e-9]
        C = resid[keep].corr().to_numpy()
        same = np.equal.outer(d[keep].to_numpy(), d[keep].to_numpy())
        iu = np.triu_indices(len(keep), 1)
        w = np.abs(C[iu][same[iu]])
        a = np.abs(C[iu][~same[iu]])
        lvl = resid[keep].std()
        rows.append(dict(kind=kind, n_elements=len(keep),
                         within_abs_corr=float(np.nanmean(w)),
                         across_abs_corr=float(np.nanmean(a)),
                         discriminative_gap=float(np.nanmean(w) - np.nanmean(a)),
                         resid_std_cv=float(lvl.std() / lvl.mean())))
    return pd.DataFrame(rows)


def _ar1(n: int, rng, phi=0.7) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _synthetic_ar_system(n: int, rng):
    """a -> b at lag 1 (coupled), c independent of a, and a mediated chain
    a -> mid -> far. Shared fixture behind every known-truth check below."""
    a, c = _ar1(n, rng), _ar1(n, rng)
    b = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        b[t] = 0.5 * b[t - 1] + 0.6 * a[t - 1] + 0.5 * e[t]
    mid = 0.8 * a + 0.4 * rng.normal(size=n)
    far = 0.8 * mid + 0.4 * rng.normal(size=n)
    return a, b, c, mid, far


def sc_estimators_known_truth(n: int, seed=0) -> pd.DataFrame:
    """Do the estimators recover a KNOWN answer at this sample size?

    a -> b at lag 1 (coupled), c independent, and a mediated chain. Run at the
    same n as the study so the answer is about this design, not about n=3000.
    """
    rng = np.random.default_rng(seed)
    a, b, c, mid, far = _synthetic_ar_system(n, rng)
    P = M.precision_partial_corr(np.column_stack([a, mid, far]))
    prng = np.random.default_rng(seed + 1)
    _, p_dep = M.circular_shift_pvalue(M.pearson, a, b, 99, prng)
    _, p_ind = M.circular_shift_pvalue(M.pearson, a, c, 99, prng)
    f_ab, f_ba = M.granger_f(a, b, 24), M.granger_f(b, a, 24)
    _, lag = M.max_lagged_xcorr(a, b, 5)
    rv_dep = M.rv_coefficient(np.column_stack([a, b]),
                              np.column_stack([b, a]))
    rv_ind = M.rv_coefficient(np.column_stack([a, _ar1(n, rng)]),
                              np.column_stack([c, _ar1(n, rng)]))
    return pd.DataFrame([
        dict(check="coupled_detected", value=p_dep, passed=p_dep < 0.05),
        dict(check="independent_not_detected", value=p_ind, passed=p_ind > 0.05),
        dict(check="granger_direction", value=f_ab / max(f_ba, 1.0),
             passed=f_ab > 5 * max(f_ba, 1.0)),
        dict(check="lead_lag_is_1", value=float(lag), passed=lag == 1),
        dict(check="mediated_path_suppressed", value=abs(P[0, 2]),
             passed=abs(P[0, 2]) < 0.15),
        dict(check="rv_separates", value=rv_dep - rv_ind,
             passed=rv_dep - rv_ind > 0.2),
    ])


def sc_sample_size(mats, sigs, ns, pair, kind, n_sur=99, seed=0) -> pd.DataFrame:
    """Statistic vs series length, on both halves -- estimator stability.

    The spread between halves at a given n is the honest error bar. This is
    why selection uses the full horizon, not 84-step windows.
    """
    a, b = pair
    rows = []
    for n in ns:
        for half, sl in (("first", slice(0, None)), ("second", slice(None, None))):
            T = len(sigs[(a, kind)])
            off = 0 if half == "first" else max(T - n, 0)
            sa, sb = sigs[(a, kind)][off:off + n], sigs[(b, kind)][off:off + n]
            Xa, Xb = mats[(a, kind)][off:off + n], mats[(b, kind)][off:off + n]
            if len(sa) < 3 * 24:
                continue
            rng = np.random.default_rng(seed)
            _, p = M.circular_shift_pvalue(M.pearson, sa, sb, n_sur, rng)
            rows.append(dict(n=n, half=half, pearson=M.pearson(sa, sb),
                             p_value=p, rv=M.rv_coefficient(Xa, Xb)))
    df = pd.DataFrame(rows)
    return (df.pivot_table(index="n", columns="half",
                           values=["pearson", "rv", "p_value"])
            .round(4))


def sc_topology_vs_physics(feats, cands, flows, steps_day, pressures):
    """Does the structural `carry` proxy predict the simulated flow it stands
    for? If not, the boundary strategy is optimizing a fiction."""
    sub = cands[cands.kind == "flow"].drop_duplicates("element")
    cols = [c for c in sub["element"] if c in flows.columns]
    mean_abs = flows[cols].abs().mean()
    carry = feats.loc[sub.index].set_index(sub["element"])["carry"].loc[cols]
    rho = sps.spearmanr(carry.to_numpy(), mean_abs.to_numpy())
    return pd.DataFrame([dict(check="carry_vs_mean_abs_flow",
                              value=float(rho.statistic),
                              passed=rho.statistic > 0.4)])


def sc_null_calibration(mats, kind, n_sur=99, seed=0) -> pd.DataFrame:
    """Shift one district's series by half the horizon and re-score: a
    correctly calibrated statistic must collapse toward its null."""
    rng = np.random.default_rng(seed)
    ds = sorted(d for d, k in mats if k == kind)
    a, b = ds[0], ds[1]
    X, Y = mats[(a, kind)], mats[(b, kind)]
    obs, p_obs = F.roll_pvalue(M.rv_coefficient, X, Y, n_sur, rng)
    Xs = np.roll(X, len(X) // 2, axis=0)
    sur, p_sur = F.roll_pvalue(M.rv_coefficient, Xs, Y, n_sur, rng)
    return pd.DataFrame([
        dict(check="observed_rv", value=obs, p_value=p_obs, passed=p_obs < 0.05),
        dict(check="shifted_rv", value=sur, p_value=p_sur, passed=sur < obs),
    ])


# ==========================================================================
# multi-world: read cached worlds straight off the experiments root
# ==========================================================================
WORLD_ARTIFACTS = {
    "districts":    "data/01_raw/districts_graeme.yml",
    "wn":           "data/02_intermediate/wn_variant.pkl",
    "pressures":    "data/02_intermediate/pressures.parquet",
    "flows":        "data/02_intermediate/flows.parquet",
    "gt_bounds":    "data/03_primary/gt_boundaries.csv",
}


def discover_worlds(project_root, experiments_root="data/09_experiments"
                    ) -> pd.DataFrame:
    """Index every cached world: manifest status, `flat` spec, artifact check.

    Reads the clone directly -- no Kedro session, no re-simulation. `usable`
    means status ok AND every artifact this POC needs is on disk.
    """
    import json
    from pathlib import Path

    root = Path(project_root) / experiments_root / "worlds"
    rows = []
    for wdir in sorted(root.glob("*")) if root.exists() else []:
        mf = wdir / "manifest.json"
        if not mf.is_file():
            continue
        m = json.loads(mf.read_text())
        clone = wdir / "clone"
        missing = [k for k, rel in WORLD_ARTIFACTS.items()
                   if not (clone / rel).exists()]
        flat = m.get("world") or {}                 # engine key is "world"
        rows.append({**flat,
                     "sim_hash": m.get("sim_hash", wdir.name),
                     "status": m.get("status"),
                     "usable": m.get("status") == "ok" and not missing,
                     "missing": ",".join(missing),
                     "dir": str(wdir)})
    df = pd.DataFrame(rows)
    need = {"drift_district", "drift_to_income", "drift_to_land_use",
            "consumption_map", "variant"}
    if len(df) and need <= set(df.columns):
        df["tag"] = (df["drift_district"].str[-1]
                     + df["drift_to_income"].str[0]
                     + df["drift_to_land_use"].str[0]
                     + "__" + df["consumption_map"] + "__" + df["variant"])
    elif len(df):
        missing_cols = need - set(df.columns)
        print(f"discover_worlds: {missing_cols} absent from manifest 'world' "
              f"block on some/all rows -- tag left unset.")
        df["tag"] = df.get("sim_hash")
    return df


def load_world(world_dir) -> dict:
    """Artifacts + the world's OWN effective parameters (n_months, sensors,
    noise and resolution differ between worlds)."""
    import json
    import pickle
    from pathlib import Path

    import yaml

    wdir = Path(world_dir)
    clone = wdir / "clone"
    eff = json.loads((wdir / "manifest.json").read_text())["effective"]
    p = lambda k: clone / WORLD_ARTIFACTS[k]                       # noqa: E731
    return {
        "districts": yaml.safe_load(p("districts").read_text()),
        "wn": pickle.loads(p("wn").read_bytes()),
        "pressures": pd.read_parquet(p("pressures")),
        "flows": pd.read_parquet(p("flows")),
        "gt_bounds": pd.read_csv(p("gt_bounds"), dtype={"pipe": str}),
        "params": eff,
        "steps_day": int(24 / eff["time"]["resolution_h"]),
    }


def label_degeneracy(art) -> str | None:
    """Why a world cannot be scored, or None. AUROC needs both classes."""
    adj = adjacency(art["gt_bounds"], art["districts"])
    n1, n0 = int(adj["adjacent"].sum()), int((~adj["adjacent"]).sum())
    if n1 < 2 or n0 < 2:
        return f"degenerate_labels(adjacent={n1}, other={n0})"
    return None


def run_world(art, kinds=("flow",), methods=None, n_sur=49, seed=0,
              noise=False) -> tuple:
    """All strategies on one world -> (long per-pair scores, diagnostics)."""
    cands = candidate_table(art["wn"], art["districts"])
    feats, dist = topology_features(art["wn"], art["districts"], cands)
    stats = residual_stats(art["pressures"], art["flows"], cands,
                           art["steps_day"])
    pls = strategies(cands, feats, stats, dist, art["params"], seed=seed)

    diag = pd.concat([
        sc_pressure_vs_flow(art["pressures"], art["flows"], cands,
                            art["steps_day"]).assign(check="pressure_vs_flow"),
        sc_topology_vs_physics(feats, cands, art["flows"], art["steps_day"],
                               art["pressures"]).assign(kind="flow"),
    ], ignore_index=True)
    inv = sc_inventory(art["wn"], art["districts"], cands, pls)
    if not inv["passed"].all():
        raise AssertionError(inv[~inv.passed].to_string(index=False))

    adj = adjacency(art["gt_bounds"], art["districts"])
    out = []
    for name, pl in pls.items():
        ss = build_series(art["pressures"], art["flows"], pl,
                          art["params"]["noise"] if noise else None, seed)
        ss = ss[ss["kind"].isin(kinds)]
        mats, sigs = district_data(ss, art["steps_day"])
        sc = pair_scores(mats, sigs, n_sur=n_sur, seed=seed)
        if methods:
            sc = sc[sc["method"].isin(methods)]
        out.append(sc.assign(strategy=name))
    scores = pd.concat(out, ignore_index=True).merge(
        adj, on=["district_a", "district_b"], how="left")
    return scores, diag, pls


def sweep_worlds(index: pd.DataFrame, verbose=True, **kw) -> tuple:
    """Score every usable world. Failures are recorded, never fatal."""
    import time
    scores, diags, notes = [], [], []
    todo = index[index["usable"]]
    for n, (_, w) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        try:
            art = load_world(w["dir"])
            why = label_degeneracy(art)
            if why:
                notes.append({**w[["sim_hash", "tag"]], "note": why})
                continue
            sc, dg, _ = run_world(art, **kw)
            meta = dict(sim_hash=w["sim_hash"], tag=w["tag"],
                        variant=w["variant"], sim_seed=w.get("sim_seed"),
                        consumption_map=w.get("consumption_map"),
                        beta=w.get("beta"), n_months=w.get("n_months"))
            scores.append(sc.assign(**meta))
            diags.append(dg.assign(**meta))
        except Exception as exc:                      # noqa: BLE001
            notes.append({"sim_hash": w["sim_hash"], "tag": w.get("tag"),
                          "note": f"{type(exc).__name__}: {exc}"})
        if verbose:
            print(f"[{n}/{len(todo)}] {w.get('tag', w['sim_hash'])} "
                  f"{time.time() - t0:.0f}s")
    empty = pd.DataFrame()
    return (pd.concat(scores, ignore_index=True) if scores else empty,
            pd.concat(diags, ignore_index=True) if diags else empty,
            pd.DataFrame(notes))


# ==========================================================================
# cross-world analysis
# ==========================================================================
def recovery_by_world(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["sim_hash", "tag", "variant", "strategy", "kind", "method"]
    out = []
    for k, g in scores.groupby(keys):
        s, y = g["statistic"].abs().to_numpy(), g["adjacent"].to_numpy(bool)
        out.append({**dict(zip(keys, k)), "n_pairs": len(g),
                    "auroc": _auroc(y, s),
                    "margin": float(np.nanmean(s[y]) - np.nanmean(s[~y]))})
    return pd.DataFrame(out)


def pooled_auroc(scores: pd.DataFrame, by=("strategy", "kind", "method")
                 ) -> pd.DataFrame:
    """AUROC over ALL (world x pair) observations.

    Statistics are not comparable in scale across worlds, so each world's
    |statistic| is converted to a within-world percentile rank first. This is
    the whole point of going multi-world: 10 pairs becomes 10 x n_worlds.
    """
    df = scores.copy()
    df["r"] = (df.groupby(list(by) + ["sim_hash"])["statistic"]
               .transform(lambda s: sps.rankdata(s.abs()) / len(s)))
    out = []
    for k, g in df.groupby(list(by)):
        out.append({**dict(zip(by, k)),
                    "n_worlds": g["sim_hash"].nunique(), "n_obs": len(g),
                    "pooled_auroc": _auroc(g["adjacent"].to_numpy(bool),
                                           g["r"].to_numpy())})
    return pd.DataFrame(out).sort_values("pooled_auroc", ascending=False)


def paired_vs_reference(rec: pd.DataFrame, ref="manual", n_boot=2000,
                        seed=0) -> pd.DataFrame:
    """Per-world paired comparison against a reference placement.

    Paired is the right unit: world-to-world variation is large and shared,
    so the delta has far less variance than either AUROC alone. Sign test on
    wins, percentile bootstrap over WORLDS for the mean delta.
    """
    rng = np.random.default_rng(seed)
    piv = rec.pivot_table(index=["sim_hash", "kind", "method"],
                          columns="strategy", values="auroc")
    if ref not in piv.columns:
        raise ValueError(f"reference '{ref}' not in {list(piv.columns)}")
    out = []
    for (kind, method), g in piv.groupby(level=["kind", "method"]):
        for strat in [c for c in piv.columns if c != ref]:
            d = (g[strat] - g[ref]).dropna().to_numpy()
            if not len(d):
                continue
            boot = np.array([rng.choice(d, len(d), replace=True).mean()
                             for _ in range(n_boot)])
            wins, losses = int((d > 0).sum()), int((d < 0).sum())
            p = (sps.binomtest(wins, wins + losses).pvalue
                 if wins + losses else np.nan)
            out.append(dict(kind=kind, method=method, strategy=strat,
                            n_worlds=len(d), mean_delta=float(d.mean()),
                            ci_lo=float(np.percentile(boot, 2.5)),
                            ci_hi=float(np.percentile(boot, 97.5)),
                            wins=wins, losses=losses,
                            ties=int((d == 0).sum()), sign_p=float(p)))
    return pd.DataFrame(out).sort_values(["method", "mean_delta"],
                                         ascending=[True, False])


def variance_share(rec: pd.DataFrame, method="partial_rv", kind="flow"
                   ) -> pd.DataFrame:
    """How much AUROC spread is strategy vs world? If world dominates,
    a single-world ranking is noise."""
    g = rec[(rec.method == method) & (rec.kind == kind)]
    if g.empty:
        return pd.DataFrame()
    grand = g["auroc"].mean()
    ss = lambda col: float(g.groupby(col)["auroc"].transform("mean")   # noqa
                           .sub(grand).pow(2).sum())
    tot = float(g["auroc"].sub(grand).pow(2).sum())
    return pd.DataFrame([
        dict(source="strategy", ss=ss("strategy"), share=ss("strategy") / tot),
        dict(source="world", ss=ss("sim_hash"), share=ss("sim_hash") / tot),
        dict(source="residual", ss=tot - ss("strategy") - ss("sim_hash"),
             share=(tot - ss("strategy") - ss("sim_hash")) / tot),
    ])


def load_live_world(project_root) -> dict:
    """Fallback: the working project's own `data/` as a single world.

    Use when no cached worlds exist yet. Same shape as `load_world`, so every
    downstream function is unchanged -- but n=1, so read it as a smoke test.
    """
    import pickle
    from pathlib import Path

    import yaml

    root = Path(project_root)
    eff = yaml.safe_load((root / "conf/base/parameters.yml").read_text())
    p = lambda k: root / WORLD_ARTIFACTS[k]                        # noqa: E731
    return {
        "districts": yaml.safe_load(p("districts").read_text()),
        "wn": pickle.loads(p("wn").read_bytes()),
        "pressures": pd.read_parquet(p("pressures")),
        "flows": pd.read_parquet(p("flows")),
        "gt_bounds": pd.read_csv(p("gt_bounds"), dtype={"pipe": str}),
        "params": eff,
        "steps_day": int(24 / eff["time"]["resolution_h"]),
    }


# ==========================================================================
# dependence/causality module: method catalog, worked example, power curves
# ==========================================================================
METHOD_CATALOG = pd.DataFrame([
    dict(method="pearson", scope="signal (district mean)", directional=False,
        null="circular time-shift surrogate",
        measures="raw co-movement of the two district MEANS"),
    dict(method="partial_corr",
        scope="signal, other districts partialled out", directional=False,
        null="none (point estimate)",
        measures="pearson with every OTHER district's mean regressed out "
                 "first -- the shared-driver / confound check"),
    dict(method="rv", scope="matrix (all of a district's sensors)",
        directional=False, null="circular time-shift surrogate (block roll)",
        measures="pearson generalized to two SENSOR MATRICES; doesn't "
                 "collapse a district's internal structure to one number"),
    dict(method="partial_rv",
        scope="matrix, other districts partialled out", directional=False,
        null="circular time-shift surrogate (block roll)",
        measures="rv with the other districts' matrices projected out -- "
                 "matrix analogue of partial_corr"),
    dict(method="granger_max", scope="signal, directional", directional=True,
        null="none here (F-statistic only)",
        measures="max(F(a->b), F(b->a)) -- does one district's past predict "
                 "the other's future beyond its own past"),
    dict(method="lead_lag_h", scope="signal, directional", directional=True,
        null="none (descriptive)",
        measures="lag (steps) where cross-correlation peaks -- who leads"),
])


def explain_pair(mats: dict, sigs: dict, a: str, b: str, kind: str,
                 lags=24, max_lag=24, n_sur=99, seed=0) -> pd.DataFrame:
    """Every method on ONE concrete pair, with what each number means.

    This is exactly what `pair_scores` runs for all C(n_districts, 2) pairs
    at once -- unrolled here for one pair so the mechanics are legible:
    `pearson`/`granger`/`lead_lag_h` reduce each district to its mean signal;
    `rv`/`partial_rv` keep the full sensor matrix; the `partial_*` methods
    additionally regress out every OTHER district before comparing a and b.
    """
    ds = sorted({d for d, k in sigs if k == kind})
    others = [d for d in ds if d not in (a, b)]
    X, Y = mats[(a, kind)], mats[(b, kind)]
    Z = (np.column_stack([mats[(c, kind)] for c in others])
         if others else np.zeros((len(X), 0)))
    rng = np.random.default_rng(seed)

    r, p_r = M.circular_shift_pvalue(M.pearson, sigs[(a, kind)],
                                     sigs[(b, kind)], n_sur, rng)
    P = M.precision_partial_corr(np.column_stack([sigs[(d, kind)]
                                                  for d in ds]))
    ia, ib = ds.index(a), ds.index(b)
    rv, p_rv = F.roll_pvalue(M.rv_coefficient, X, Y, n_sur, rng)
    if Z.shape[1]:
        prv, p_prv = F.roll_pvalue(
            lambda x, y: F.partial_rv(x, y, Z), X, Y, n_sur, rng)
    else:
        prv, p_prv = rv, p_rv
    f_ab = M.granger_f(sigs[(a, kind)], sigs[(b, kind)], lags)
    f_ba = M.granger_f(sigs[(b, kind)], sigs[(a, kind)], lags)
    _, lag = M.max_lagged_xcorr(sigs[(a, kind)], sigs[(b, kind)], max_lag)

    return pd.DataFrame([
        dict(method="pearson", statistic=r, p_value=p_r,
            measures="raw co-movement of district MEANS"),
        dict(method="partial_corr", statistic=P[ia, ib], p_value=np.nan,
            measures="pearson with other districts regressed out"),
        dict(method="rv", statistic=rv, p_value=p_rv,
            measures="pearson generalized to the full sensor MATRIX"),
        dict(method="partial_rv", statistic=prv, p_value=p_prv,
            measures="rv with other districts' matrices projected out"),
        dict(method=f"granger_{a}->{b}", statistic=f_ab, p_value=np.nan,
            measures=f"does {a}'s past predict {b}'s future beyond its own?"),
        dict(method=f"granger_{b}->{a}", statistic=f_ba, p_value=np.nan,
            measures=f"does {b}'s past predict {a}'s future beyond its own?"),
        dict(method="lead_lag_h", statistic=float(lag), p_value=np.nan,
            measures="lag (steps) where cross-corr peaks; sign = who leads"),
    ])


def known_truth_power(ns=(84, 336, 1008, 2016, 4320, 8640), n_rep=20,
                      n_sur=49, seed=0) -> pd.DataFrame:
    """Detection power of each known-truth check, across `n_rep` independent
    draws per horizon `n`. Answers: how much data before the toolkit
    reliably tells coupled from independent, gets direction and lag right,
    and rejects a spuriously-correlated mediated pair?

    One synthetic draw per (n, rep); the rate over reps is a Monte-Carlo
    power/false-positive estimate, not a single lucky (or unlucky) seed.
    """
    rows = []
    for n in ns:
        lags = max(2, min(24, n // 6))
        for rep in range(n_rep):
            rng = np.random.default_rng((seed, n, rep))
            a, b, c, mid, far = _synthetic_ar_system(n, rng)
            _, p_dep = M.circular_shift_pvalue(M.pearson, a, b, n_sur, rng)
            _, p_ind = M.circular_shift_pvalue(M.pearson, a, c, n_sur, rng)
            f_ab, f_ba = M.granger_f(a, b, lags), M.granger_f(b, a, lags)
            _, lag = M.max_lagged_xcorr(a, b, 5)
            Pm = M.precision_partial_corr(np.column_stack([a, mid, far]))
            rv_dep = M.rv_coefficient(np.column_stack([a, b]),
                                      np.column_stack([b, a]))
            rv_ind = M.rv_coefficient(
                np.column_stack([a, _ar1(n, rng)]),
                np.column_stack([c, _ar1(n, rng)]))
            rows.append(dict(
                n=n, rep=rep,
                power_coupled=p_dep < 0.05,
                fpr_independent=p_ind < 0.05,
                direction_correct=f_ab > 5 * max(f_ba, 1.0),
                lag_correct=(lag == 1),
                mediation_suppressed=abs(Pm[0, 2]) < 0.15,
                rv_separates=(rv_dep - rv_ind) > 0.2))
    df = pd.DataFrame(rows)
    return (df.groupby("n")[["power_coupled", "fpr_independent",
                             "direction_correct", "lag_correct",
                             "mediation_suppressed", "rv_separates"]]
            .mean().reset_index())


# ==========================================================================
# horizon x AUROC on REAL worlds: does more data make placement matter more?
# ==========================================================================
def truncate(mats: dict, sigs: dict, n: int) -> tuple:
    return ({k: v[:n] for k, v in mats.items()},
            {k: v[:n] for k, v in sigs.items()})


def prepare_world_for_horizon(art: dict, placement: dict, kind="flow"
                              ) -> tuple:
    """One placement, scored once at full horizon; `horizon_scores` then
    truncates the SAME residuals rather than re-deseasonalizing per `n`."""
    ss = build_series(art["pressures"], art["flows"], placement)
    ss = ss[ss["kind"] == kind]
    mats, sigs = district_data(ss, art["steps_day"])
    adj = adjacency(art["gt_bounds"], art["districts"])
    return mats, sigs, adj


def horizon_scores(prepared: dict, ns, methods=None, n_sur=49, seed=0
                   ) -> pd.DataFrame:
    """`prepared`: {tag: (mats, sigs, adj)} from `prepare_world_for_horizon`.

    Truncating to the first `n` steps asks the operational question -- "if
    only `n` steps were available so far, would this statistic already be
    useful?" -- not a random subsample, which would answer a different
    question about estimator variance alone.
    """
    rows = []
    for tag, (mats, sigs, adj) in prepared.items():
        T = next(iter(sigs.values())).shape[0]
        for n in ns:
            nn = min(n, T)
            tm, ts = truncate(mats, sigs, nn)
            sc = pair_scores(tm, ts, n_sur=n_sur, seed=seed)
            if methods:
                sc = sc[sc["method"].isin(methods)]
            sc = sc.merge(adj, on=["district_a", "district_b"], how="left")
            rows.append(sc.assign(sim_hash=tag, n=nn))
    return pd.concat(rows, ignore_index=True)


def auroc_vs_horizon(df: pd.DataFrame, by=("kind", "method")) -> pd.DataFrame:
    """Pooled AUROC (rank-normalized within world, see `pooled_auroc`) at
    each horizon `n` -- one curve per method, answering 'how much data
    before this statistic reliably separates adjacent from non-adjacent'."""
    out = []
    for n, g in df.groupby("n"):
        p = pooled_auroc(g, by=by)
        p.insert(0, "n", n)
        out.append(p)
    return pd.concat(out, ignore_index=True)


# ==========================================================================
# directed pairwise measure: "B -> C", not just B~C
# ==========================================================================
def directed_scores(mats: dict, sigs: dict, kind: str, lags=24, max_lag=24
                    ) -> pd.DataFrame:
    """The actual directed measure per district pair -- what `pair_scores`
    discards by returning `max(f_ab, f_ba)` under a single `granger_max` row.

    Conventions (from `dependence_oracle.methods`, unchanged here):
    `granger_f(a, b)` = "a Granger-causes b" -- does a's lagged values
    improve the autoregression of b, beyond b's own past. `max_lagged_xcorr`
    returns a signed lag with `lag > 0` meaning the FIRST argument leads.

    One row per unordered pair, with both directions side by side so the
    asymmetry itself is visible -- a genuinely one-way effect has
    `granger_a_to_b >> granger_b_to_a`; a symmetric/confounded pair has
    both large and comparable, which `dominant` and `direction_ratio` flag
    rather than silently picking a winner.
    """
    ds = sorted({d for d, k in sigs if k == kind})
    rows = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            sa, sb = sigs[(a, kind)], sigs[(b, kind)]
            f_ab = M.granger_f(sa, sb, lags)     # a -> b
            f_ba = M.granger_f(sb, sa, lags)     # b -> a
            r, lag = M.max_lagged_xcorr(sa, sb, max_lag)   # lag>0: a leads b
            dominant = (f"{a}->{b}" if f_ab >= f_ba else f"{b}->{a}")
            leader = a if lag > 0 else (b if lag < 0 else "simultaneous")
            rows.append(dict(
                district_a=a, district_b=b,
                granger_a_to_b=f_ab, granger_b_to_a=f_ba,
                dominant_direction=dominant,
                direction_ratio=f_ab / max(f_ba, 1e-9),
                lead_lag_steps=abs(lag), lead_lag_leader=leader,
                lead_lag_r=r))
    return pd.DataFrame(rows)


# ==========================================================================
# directional ground truth + causality sanity checks
# ==========================================================================
def boundary_flow_direction(art: dict) -> pd.DataFrame:
    """PHYSICAL directed ground truth: net signed flow through boundary pipes.

    wntr link flowrate is signed relative to start_node -> end_node, so the
    mean sign says which district supplies which. This is the directional
    label the symmetric adjacency AUROC could never provide.

    `reversal_frac` is the important caveat column: if flow through the
    interconnection reverses sign over the day (tank filling/draining), there
    is no stable physical direction for that pair and any directional claim
    about it -- Granger or otherwise -- is meaningless.
    """
    n2d = node_to_district(art["districts"])
    wn, flows = art["wn"], art["flows"]
    rows = []
    for name in wn.pipe_name_list:
        p = wn.get_link(name)
        da, db = n2d.get(p.start_node_name), n2d.get(p.end_node_name)
        if da is None or db is None or da == db or name not in flows.columns:
            continue
        q = flows[name].to_numpy(float)
        rows.append(dict(pipe=name, d_from=da, d_to=db, mean_q=float(q.mean()),
                         mean_abs_q=float(np.abs(q).mean()),
                         frac_positive=float((q > 0).mean())))
    if not rows:
        return pd.DataFrame()
    pipes = pd.DataFrame(rows)

    out = []
    for (a, b), g in pipes.groupby(
            [pipes[["d_from", "d_to"]].min(axis=1),
             pipes[["d_from", "d_to"]].max(axis=1)]):
        # orient every pipe's mean flow as "a -> b"
        signed = np.where(g["d_from"].to_numpy() == a,
                          g["mean_q"].to_numpy(), -g["mean_q"].to_numpy())
        net = float(signed.sum())
        rev = float(np.mean([min(f, 1 - f) for f in g["frac_positive"]]))
        out.append(dict(district_a=a, district_b=b, n_boundary_pipes=len(g),
                        net_q_a_to_b=net,
                        physical_direction=f"{a}->{b}" if net > 0
                        else f"{b}->{a}",
                        transfer_strength=float(np.abs(signed).sum()),
                        reversal_frac=rev,
                        stable_direction=rev < 0.05))
    return pd.DataFrame(out)


def granger_with_null(sigs: dict, kind: str, lags=24, n_sur=99, seed=0
                      ) -> pd.DataFrame:
    """`directed_scores` with a surrogate null on BOTH directions.

    Circularly shifting the putative source destroys cross-dependence while
    preserving each series' own autocorrelation -- so an F that survives the
    null reflects genuine cross-prediction, not one district simply being
    more self-predictable than the other. Without this, raw F values across
    pairs are not comparable.
    """
    rng = np.random.default_rng(seed)
    ds = sorted({d for d, k in sigs if k == kind})
    gf = lambda x, y: M.granger_f(x, y, lags)                     # noqa: E731
    rows = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            sa, sb = sigs[(a, kind)], sigs[(b, kind)]
            f_ab, p_ab = M.circular_shift_pvalue(gf, sa, sb, n_sur, rng)
            f_ba, p_ba = M.circular_shift_pvalue(gf, sb, sa, n_sur, rng)
            sig_ab, sig_ba = p_ab < 0.05, p_ba < 0.05
            verdict = ("bidirectional" if sig_ab and sig_ba else
                       f"{a}->{b}" if sig_ab else
                       f"{b}->{a}" if sig_ba else "none")
            rows.append(dict(district_a=a, district_b=b,
                             f_a_to_b=f_ab, p_a_to_b=p_ab,
                             f_b_to_a=f_ba, p_b_to_a=p_ba,
                             verdict_at_05=verdict))
    return pd.DataFrame(rows)


def lag_profile(sigs: dict, a: str, b: str, kind: str, max_lag=48
                ) -> pd.DataFrame:
    """Cross-correlation at every lag -- the plot that reveals whether the
    coupling has ANY temporal structure. A sharp peak at lag 0 means
    instantaneous coupling, and Granger direction should not be trusted."""
    sa, sb = sigs[(a, kind)], sigs[(b, kind)]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.corrcoef(sa[:len(sa) - lag or None], sb[lag:])[0, 1]
        else:
            r = np.corrcoef(sa[-lag:], sb[:lag])[0, 1]
        rows.append(dict(lag=lag, r=float(r)))
    return pd.DataFrame(rows)


def reverse_time_control(sigs: dict, kind: str, lags=24) -> pd.DataFrame:
    """Time-reversal control for Granger asymmetry.

    Genuine temporal causation should weaken or flip when time is reversed.
    An asymmetry that SURVIVES reversal is driven by static properties of
    the series (variance, autocorrelation), not by direction of influence --
    a standard and rather unforgiving diagnostic.
    """
    ds = sorted({d for d, k in sigs if k == kind})
    rows = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            sa, sb = sigs[(a, kind)], sigs[(b, kind)]
            fwd = M.granger_f(sa, sb, lags) / max(M.granger_f(sb, sa, lags), 1e-9)
            rev = (M.granger_f(sa[::-1], sb[::-1], lags)
                   / max(M.granger_f(sb[::-1], sa[::-1], lags), 1e-9))
            rows.append(dict(district_a=a, district_b=b,
                             ratio_forward=fwd, ratio_reversed=rev,
                             survives_reversal=(fwd > 1) == (rev > 1)))
    return pd.DataFrame(rows)


def autocorr_confound(sigs: dict, kind: str, lags=24) -> pd.DataFrame:
    """Is the Granger asymmetry just a self-predictability difference?

    Per district: R^2 of its own AR(`lags`) model. If `direction_ratio`
    tracks the DIFFERENCE in these, the 'causality' is a confound.
    """
    ds = sorted({d for d, k in sigs if k == kind})
    out = {}
    for d in ds:
        x = sigs[(d, kind)]
        Xd = np.column_stack([x[lags - k - 1:len(x) - k - 1]
                              for k in range(lags)])
        y = x[lags:]
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xd)), Xd]),
                                   y, rcond=None)
        pred = np.column_stack([np.ones(len(Xd)), Xd]) @ beta
        ss = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        out[d] = ss
    return pd.DataFrame([dict(district=d, ar_r2=v) for d, v in out.items()])


# ==========================================================================
# regime invariance: is the measured dependence PHYSICAL?
# ==========================================================================
def run_world_directed(art: dict, placement="variance", kind="flow", lags=24,
                       max_lag=48, n_sur=49, seed=0, subsample=None
                       ) -> pd.DataFrame:
    """Directed measures + physical flow direction for one world, merged.

    `placement` names a strategy (or pass a placement dict directly), so the
    sensor set is EXPLICIT rather than inherited from whatever `mats`/`sigs`
    happened to be in scope.

    `subsample`: contiguous chunk length for the lag-structure methods, via
    `methods.contiguous` -- the framework's own convention for Granger (the
    oracle uses 1500). Surrogate Granger is ~1000 OLS fits per world, so the
    full 21600-step horizon makes a 40-world sweep impractical. Set it to
    None for the full series on a single world, and check the result against
    the horizon analysis in Part 2 before trusting a subsampled value.
    """
    cands = candidate_table(art["wn"], art["districts"])
    if isinstance(placement, str):
        feats, dist = topology_features(art["wn"], art["districts"], cands)
        stats = residual_stats(art["pressures"], art["flows"], cands,
                               art["steps_day"])
        placement = strategies(cands, feats, stats, dist, art["params"],
                               seed=seed)[placement]
    ss = build_series(art["pressures"], art["flows"], placement)
    ss = ss[ss["kind"] == kind]
    mats, sigs = district_data(ss, art["steps_day"])
    if subsample:
        sigs = {k: M.contiguous(v, subsample) for k, v in sigs.items()}
        mats = {k: M.contiguous(v, subsample) for k, v in mats.items()}

    dsc = directed_scores(mats, sigs, kind, lags=lags, max_lag=max_lag)
    gwn = granger_with_null(sigs, kind, lags=lags, n_sur=n_sur, seed=seed)
    phys = boundary_flow_direction(art)
    rev = reverse_time_control(sigs, kind, lags=lags)
    df = (dsc.merge(gwn, on=["district_a", "district_b"], how="left")
          .merge(rev, on=["district_a", "district_b"], how="left"))
    if len(phys):
        df = df.merge(phys, on=["district_a", "district_b"], how="left")
        df["granger_matches_physical"] = (
            df["dominant_direction"] == df["physical_direction"])
        df["adjacent"] = df["n_boundary_pipes"].notna()
    return df


def sweep_directed(index: pd.DataFrame, placement="variance", verbose=True,
                   **kw) -> tuple:
    """Directed measures across all usable worlds."""
    import time
    frames, notes = [], []
    todo = index[index["usable"]]
    for n, (_, w) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        try:
            art = load_world(w["dir"])
            df = run_world_directed(art, placement=placement, **kw)
            frames.append(df.assign(
                sim_hash=w["sim_hash"], tag=w.get("tag"),
                consumption_map=w.get("consumption_map"),
                drift_district=w.get("drift_district"),
                variant=w.get("variant")))
        except Exception as exc:                            # noqa: BLE001
            notes.append({"sim_hash": w["sim_hash"], "tag": w.get("tag"),
                          "note": f"{type(exc).__name__}: {exc}"})
        if verbose:
            print(f"[{n}/{len(todo)}] {w.get('tag')} {time.time() - t0:.0f}s")
    empty = pd.DataFrame()
    return (pd.concat(frames, ignore_index=True) if frames else empty,
            pd.DataFrame(notes))


def directed_consistency(df: pd.DataFrame, group=None) -> pd.DataFrame:
    """Per pair: how often does the measured direction agree across worlds?

    A PHYSICAL dependence should give the same answer regardless of the
    consumption regime. Pass `group="consumption_map"` to test exactly that:
    if `modal_agreement` is high within each regime but the modal direction
    DIFFERS between regimes, the measure is tracking demand structure, not
    hydraulic transfer.
    """
    keys = ["district_a", "district_b"] + ([group] if group else [])
    out = []
    for k, g in df.groupby(keys):
        mode = g["dominant_direction"].mode()
        modal = mode.iloc[0] if len(mode) else None
        row = {**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
               "n_worlds": len(g), "modal_direction": modal,
               "modal_agreement": float((g["dominant_direction"] == modal).mean()),
               "median_ratio": float(g["direction_ratio"].median())}
        if "granger_matches_physical" in g:
            row["match_physical_rate"] = float(
                g["granger_matches_physical"].mean())
        if "survives_reversal" in g:
            row["survives_reversal_rate"] = float(g["survives_reversal"].mean())
        if "physical_direction" in g:
            pm = g["physical_direction"].mode()
            row["physical_direction"] = pm.iloc[0] if len(pm) else None
            row["physical_stable_rate"] = float(
                g["stable_direction"].mean()) if "stable_direction" in g else np.nan
        out.append(row)
    return pd.DataFrame(out)


def regime_invariance(df: pd.DataFrame, by="consumption_map") -> pd.DataFrame:
    """Does each pair's measured direction survive a change of regime?

    `n_regimes_agreeing` / `n_regimes` == 1.0 means the direction is
    invariant to consumption structure -- the signature of a physical
    coupling. Anything less means the regime is driving the answer.
    """
    per = directed_consistency(df, group=by)
    out = []
    for (a, b), g in per.groupby(["district_a", "district_b"]):
        mode = g["modal_direction"].mode()
        top = mode.iloc[0] if len(mode) else None
        row = dict(district_a=a, district_b=b, n_regimes=len(g),
                   overall_direction=top,
                   n_regimes_agreeing=int((g["modal_direction"] == top).sum()),
                   invariant=bool((g["modal_direction"] == top).all()),
                   min_within_regime_agreement=float(g["modal_agreement"].min()))
        if "physical_direction" in g:
            pm = g["physical_direction"].mode()
            row["physical_direction"] = pm.iloc[0] if len(pm) else None
            row["matches_physical"] = (row["overall_direction"]
                                       == row["physical_direction"])
        out.append(row)
    return pd.DataFrame(out)


# ==========================================================================
# month windows: isolate a single consumption regime (pre-drift)
# ==========================================================================
def month_window(art: dict, n_months: int = 3, start: int = 0) -> dict:
    """Restrict a world to a contiguous month window.

    The row index is deliberately NOT reset: `deseasonalize` derives hour and
    weekday from the absolute step index, so preserving it keeps the weekly
    phase correct for windows that don't start at month 0.

    `fl.reference_months` is the simulator's drift warm-up, so
    `start=0, n_months<=2` is guaranteed pre-drift for every client -- a
    single, stationary consumption regime.
    """
    months = sorted(art["pressures"]["month"].unique())[start:start + n_months]
    keep = set(months)
    out = dict(art)
    out["pressures"] = art["pressures"][art["pressures"]["month"].isin(keep)]
    out["flows"] = art["flows"][art["flows"]["month"].isin(keep)]
    out["window_months"] = months
    return out


def arr2_ordering_agreement(sigs: dict, kind: str, lags=24) -> dict:
    """Is the Granger 'causal graph' just the self-predictability ordering?

    Rank districts by their own AR(`lags`) R^2 and predict that the more
    self-predictable district Granger-causes the less. `agreement` is the
    fraction of pairs where that prediction matches the observed dominant
    direction. Near 1.0 means the directional result carries no information
    beyond which series is smoother.
    """
    ar = autocorr_confound(sigs, kind, lags=lags).set_index("district")["ar_r2"]
    ds = sorted({d for d, k in sigs if k == kind})
    hit, tot = 0, 0
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            f_ab = M.granger_f(sigs[(a, kind)], sigs[(b, kind)], lags)
            f_ba = M.granger_f(sigs[(b, kind)], sigs[(a, kind)], lags)
            observed = f"{a}->{b}" if f_ab >= f_ba else f"{b}->{a}"
            predicted = f"{a}->{b}" if ar[a] >= ar[b] else f"{b}->{a}"
            hit += observed == predicted
            tot += 1
    return dict(arr2_ordering_agreement=hit / tot, n_pairs=tot,
                ar_r2_min=float(ar.min()), ar_r2_max=float(ar.max()))


def directional_diagnostics(art: dict, placement="variance", kind="flow",
                            lags=24, max_lag=48, n_sur=99, seed=0,
                            month_aware=True, label="") -> pd.DataFrame:
    """The whole Part-4 battery collapsed to ONE summary row.

    Built so windows / deseasonalization settings can be compared on equal
    terms: every column is a scalar verdict, and the ones that matter are
    `peak_lag_zero_frac` (is there any precedence to exploit),
    `arr2_agreement` + `spearman_ratio_arr2` (is direction the smoothness
    confound), `survives_reversal_frac` (artifact), and
    `match_physical_stable` (agreement with real flow direction).
    """
    cands = candidate_table(art["wn"], art["districts"])
    if isinstance(placement, str):
        feats, dist = topology_features(art["wn"], art["districts"], cands)
        stats = residual_stats(art["pressures"], art["flows"], cands,
                               art["steps_day"])
        placement = strategies(cands, feats, stats, dist, art["params"],
                               seed=seed)[placement]
    ss = build_series(art["pressures"], art["flows"], placement)
    ss = ss[ss["kind"] == kind]
    mats, sigs = district_data(ss, art["steps_day"], month_aware=month_aware)
    ds = sorted({d for d, k in sigs if k == kind})
    T = len(next(iter(sigs.values())))
    ml = min(max_lag, max(2, T // 8))
    lg = min(lags, max(2, T // 20))

    peaks, r0 = [], []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            lp = lag_profile(sigs, a, b, kind, max_lag=ml)
            top = lp.loc[lp["r"].abs().idxmax()]
            peaks.append(int(top["lag"]))
            r0.append(float(lp.loc[lp.lag == 0, "r"].iloc[0]))

    dsc = directed_scores(mats, sigs, kind, lags=lg, max_lag=ml)
    gwn = granger_with_null(sigs, kind, lags=lg, n_sur=n_sur, seed=seed)
    rev = reverse_time_control(sigs, kind, lags=lg)
    ordr = arr2_ordering_agreement(sigs, kind, lags=lg)
    ar = autocorr_confound(sigs, kind, lags=lg).set_index("district")["ar_r2"]

    chk = dsc.assign(
        d=lambda d: [ar[x] - ar[y] for x, y in zip(d.district_a, d.district_b)],
        lr=lambda d: np.log(d["direction_ratio"].clip(1e-9)))
    rho = float(chk[["d", "lr"]].corr(method="spearman").iloc[0, 1])

    phys = boundary_flow_direction(art)
    match_stable = np.nan
    n_stable = 0
    if len(phys):
        cmp = dsc.merge(phys, on=["district_a", "district_b"], how="inner")
        st = cmp[cmp["stable_direction"]]
        n_stable = len(st)
        if n_stable:
            match_stable = float((st["dominant_direction"]
                                  == st["physical_direction"]).mean())

    return pd.DataFrame([dict(
        label=label or f"{len(art.get('window_months', [])) or 'all'}mo",
        months=len(art.get("window_months", [])) or None,
        month_aware=month_aware, n_steps=T, lags_used=lg,
        peak_lag_zero_frac=float(np.mean([p == 0 for p in peaks])),
        mean_r_at_lag0=float(np.mean(r0)),
        ar_r2_min=ordr["ar_r2_min"], ar_r2_max=ordr["ar_r2_max"],
        arr2_agreement=ordr["arr2_ordering_agreement"],
        spearman_ratio_arr2=rho,
        survives_reversal_frac=float(rev["survives_reversal"].mean()),
        bidirectional_frac=float((gwn["verdict_at_05"]
                                  == "bidirectional").mean()),
        none_frac=float((gwn["verdict_at_05"] == "none").mean()),
        n_stable_physical=n_stable, match_physical_stable=match_stable)])


def compare_windows(art: dict, specs=((2, True), (2, False), (3, True),
                                      (3, False), (6, True), (None, True)),
                    **kw) -> pd.DataFrame:
    """Run the battery across (n_months, month_aware) settings.

    `n_months=None` means the full horizon. Read `n_steps` alongside every
    verdict: a short window buys regime purity at the cost of sample size,
    and Part 2 showed the matrix statistics need thousands of steps.
    """
    rows = []
    for n_months, ma in specs:
        sub = art if n_months is None else month_window(art, n_months)
        lbl = f"{'all' if n_months is None else n_months}mo" \
              f"{'' if ma else '-flat'}"
        rows.append(directional_diagnostics(sub, month_aware=ma, label=lbl,
                                            **kw))
    return pd.concat(rows, ignore_index=True)


def sweep_windows(index: pd.DataFrame, specs=None, verbose=True, **kw
                  ) -> pd.DataFrame:
    """`compare_windows` across worlds -- one row per (world, setting)."""
    import time
    frames = []
    todo = index[index["usable"]]
    for n, (_, w) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        try:
            art = load_world(w["dir"])
            df = (compare_windows(art, **kw) if specs is None
                  else compare_windows(art, specs=specs, **kw))
            frames.append(df.assign(sim_hash=w["sim_hash"], tag=w.get("tag"),
                                    consumption_map=w.get("consumption_map")))
        except Exception as exc:                            # noqa: BLE001
            print(f"  skip {w.get('tag')}: {type(exc).__name__}: {exc}")
        if verbose:
            print(f"[{n}/{len(todo)}] {w.get('tag')} {time.time() - t0:.0f}s")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
