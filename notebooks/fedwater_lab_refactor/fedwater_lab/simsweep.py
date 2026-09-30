"""Client similarity and clustering from prototypes, on every stored run.

Three questions, asked of the RAW latent prototypes and of the RECONSTRUCTED
(decoded) ones, and scored against two centralized ceilings.

Spaces, per calendar group g = (block, cycle class) and client pair (c, d)
--------------------------------------------------------------------------
``s_lat``      cos(p[c,g], p[d,g]) -- the raw latent prototype, the artifact
               the protocol exchanges (POC_04's definition).
``s_dec``      mean over the arm's channels i of corr_t(dec(p[c,g])_i,
               dec(p[d,g])_i), channels matched by slot index
               (`data.client_frames` orders them by slot in every client).
``s_dec_abs``  the same with |corr| per channel. On an UNORIENTED run a flow
               channel's sign is arbitrary, so the signed value can flip for
               reasons unrelated to demand; the absolute value cannot.
``s_true``     / ``s_true_abs``  the same on the mean TRUE windows (model
               units) -- the sensor-level ceiling, which includes hydraulic
               mixing on the mixed channels.
``s_dem``      corr_t of the two districts' OWN demand, cut at exactly the
               same window starts -- the demand-level ceiling, free of
               mixing. It is the cleaner yardstick for "are these two CLIENTS
               similar".
All correlations are `recon._r` (population Pearson, 0 for a constant
series), exactly as POC_04. The ceilings need the raw data and are
evaluation-only.

Arms: ``pure`` / ``mixed`` / ``all`` restrict the channels of s_dec and
s_true; s_lat and s_dem do not depend on the arm and carry `latent` and
`demand`.

Q1 -- fidelity (`fidelity`)
---------------------------
Does a federated space reproduce a ceiling's ORDERING of pairs? Per phase,
at two levels: ``rows`` (every class-level group x pair, POC_04's
`similarity_fidelity`) and ``pairs`` (the six phase-averaged pair values, the
thing a clustering step consumes). Pearson, Spearman, and

    concordant = share of pair couples (i, j), a_i != a_j, ranked the same
                 way: (a_i - a_j)(b_i - b_j) > 0.

Q2 -- clustering against regimes (`clusters`)
---------------------------------------------
Only where a clean label exists: `World.regimes(phase)` is defined for init
and final and raises for transition months. From the phase-averaged pair
similarity S, average-linkage clustering on D = 1 - S, cut at the TRUE number
of regime groups k:

    ari        adjusted Rand index against the regime partition;
    top_ok     the most similar pair is a same-regime pair;
    auc_same   P(s_same-regime pair > s_different-regime pair), ties 1/2.

A partition with k = 1 or k = n_clients is trivial (nothing to recover) and
is recorded as such, not scored. With four clients there are six pairs, so a
single run's exact p cannot go below 1/6: read these across runs.

Q3 -- tracking the drift (`tracking`)
-------------------------------------
Per block b, the pair's block-mean similarity s_b against the drift
district's demand-weighted switched fraction pi_b (mean of
`World.drift_progress` over the block's months):

    rho = Spearman(s_b, pi_b) over blocks.

Expected sign, from the regimes: for a pair (A, X) with A the drift
district, -1 when X shares A's INIT regime and not its final one (A moves
away), +1 when X shares A's FINAL regime and not its init one (A converges),
0 otherwise -- and 0 for every pair without A (nothing changes there).
`sign_ok` is scored only where the expected sign is non-zero. The p-value of
a Spearman over ~24 autocorrelated blocks is optimistic; it is stored, the
sign test across runs is what should be read.

Nothing here trains: the decoded path goes through `grid.assert_cached` and
refuses a non-cache-hit. Checkpointing and the bundle mirror `sweep`.
"""
from __future__ import annotations

import gc
import json
import pathlib
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import grid as GD
from . import traj as TJ
from .sweep import _Stage, _add, _done, _jsonable, _tag

__all__ = ["SimParams", "run_config", "sweep", "collect", "DICTIONARY",
           "similarity_long", "latent_similarity_long", "fidelity", "clusters",
           "tracking"]

FED = (("s_lat", "latent"),
       ("s_dec", "pure"), ("s_dec", "mixed"), ("s_dec", "all"),
       ("s_dec_abs", "pure"), ("s_dec_abs", "mixed"), ("s_dec_abs", "all"))


@dataclass(frozen=True)
class SimParams:
    block_months: int = 3
    bin_h: int | None = 24
    period_h: int = 168
    arms: tuple = ("pure", "mixed", "all")
    cluster_phases: tuple = ("init", "final")
    n_perm: int = 19                  # eta2 permutations inside analyse (unused)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


# --------------------------------------------------------------------------
# similarity tables (class level)
# --------------------------------------------------------------------------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def similarity_long(run, channels: pd.DataFrame, arms=("pure", "mixed", "all")
                    ) -> pd.DataFrame:
    """Every space, every class-level group, every pair, from an AlignedRun."""
    from .recon import _r
    g = run.groups.set_index("gid")
    true, dec, dem = run.arrays["true"], run.arrays.get("aligned", {}), run.demand
    fc = [c for c in run.protos.columns if c.startswith("f") and c[1:].isdigit()]
    lat = {(r.client, int(r.gid)): np.array([getattr(r, f) for f in fc], float)
           for r in run.protos.itertuples(index=False)}
    keep = {}
    for arm in arms:
        if arm == "all":
            keep[arm] = None
        else:
            keep[arm] = {c: set(sub.loc[sub["slot_class"] == arm, "channel"].astype(int))
                         for c, sub in channels.groupby("client")}
    by_gid: dict = {}
    for (c, gid) in true:
        by_gid.setdefault(int(gid), []).append(c)
    rows = []
    for gid, cs in by_gid.items():
        cs = sorted(cs)
        info = g.loc[gid]
        base = {"gid": gid, "bid": int(info["bid"]), "phase": info["phase"],
                "cycle_pos": int(info["cycle_pos"])}
        for i, c in enumerate(cs):
            for d in cs[i + 1:]:
                pair = {**base, "client_a": c, "client_b": d}
                if (c, gid) in lat and (d, gid) in lat:
                    rows.append({**pair, "space": "s_lat", "arm": "latent",
                                 "value": _cos(lat[(c, gid)], lat[(d, gid)])})
                if (c, gid) in dem and (d, gid) in dem and c in dem[(c, gid)] \
                        and d in dem[(d, gid)]:
                    rows.append({**pair, "space": "s_dem", "arm": "demand",
                                 "value": _r(np.asarray(dem[(c, gid)][c]),
                                             np.asarray(dem[(d, gid)][d]))})
                Tc, Td = true[(c, gid)], true[(d, gid)]
                for arm in arms:
                    idx = (list(range(Tc.shape[1])) if keep[arm] is None else
                           sorted(keep[arm].get(c, set()) & keep[arm].get(d, set())))
                    if not idx:
                        continue
                    rt = [_r(Tc[:, k], Td[:, k]) for k in idx]
                    rows.append({**pair, "space": "s_true", "arm": arm,
                                 "value": float(np.mean(rt))})
                    rows.append({**pair, "space": "s_true_abs", "arm": arm,
                                 "value": float(np.mean(np.abs(rt)))})
                    if (c, gid) in dec and (d, gid) in dec:
                        Dc, Dd = dec[(c, gid)], dec[(d, gid)]
                        rd = [_r(Dc[:, k], Dd[:, k]) for k in idx]
                        rows.append({**pair, "space": "s_dec", "arm": arm,
                                     "value": float(np.mean(rd))})
                        rows.append({**pair, "space": "s_dec_abs", "arm": arm,
                                     "value": float(np.mean(np.abs(rd)))})
    return pd.DataFrame(rows)


def latent_similarity_long(wt: pd.DataFrame) -> pd.DataFrame:
    """World-free: s_lat per (block, cycle class, pair) from the saved latents
    alone (the group-mean latent is the prototype the protocol exchanges)."""
    fc = [c for c in wt.columns if c.startswith("f") and c[1:].isdigit()]
    protos = wt.groupby(["client", "bid", "cycle_pos", "phase"])[fc].mean()
    rows = []
    for (bid, cp, ph), sub in protos.groupby(level=["bid", "cycle_pos", "phase"]):
        cs = sorted(sub.index.get_level_values("client"))
        vec = {c: sub.xs(c, level="client").to_numpy(float).ravel() for c in cs}
        for i, c in enumerate(cs):
            for d in cs[i + 1:]:
                rows.append({"gid": -1, "bid": int(bid), "phase": ph,
                             "cycle_pos": int(cp), "client_a": c, "client_b": d,
                             "space": "s_lat", "arm": "latent",
                             "value": _cos(vec[c], vec[d])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Q1 fidelity
# --------------------------------------------------------------------------
def _fid(a: np.ndarray, b: np.ndarray) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return {"n": int(len(a)), "pearson": np.nan, "spearman": np.nan,
                "concordant": np.nan, "mean_err": np.nan}
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    pairs = [(a[i] - a[j]) * (b[i] - b[j]) > 0
             for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] != a[j]]
    return {"n": int(len(a)), "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(np.corrcoef(ra, rb)[0, 1]),
            "concordant": float(np.mean(pairs)) if pairs else np.nan,
            "mean_err": float(np.mean(b - a))}


def _refs_for(space: str, arm: str) -> list[tuple[str, str]]:
    """Which ceilings a federated (space, arm) is scored against."""
    if space == "s_lat":
        return [("s_true", "pure"), ("s_dem", "demand")]
    if space == "s_dec":
        return [("s_true", arm), ("s_dem", "demand")]
    if space == "s_dec_abs":
        return [("s_true_abs", arm), ("s_dem", "demand")]
    return []


def fidelity(long: pd.DataFrame) -> pd.DataFrame:
    """Federated space against ceiling, per phase, rows- and pairs-level.
    Plus the ceiling against itself (s_true vs s_dem): how much of the
    sensor-level similarity is demand similarity -- the mixing reference."""
    key = ["phase", "gid", "bid", "cycle_pos", "client_a", "client_b"]
    wide = long.pivot_table(index=key, columns=["space", "arm"], values="value")
    out = []
    combos = [(s, a, r, ra) for s, a in FED for r, ra in _refs_for(s, a)]
    combos += [("s_true", arm, "s_dem", "demand") for arm in ("pure", "mixed", "all")]
    for s, a, r, ra in combos:
        if (s, a) not in wide.columns or (r, ra) not in wide.columns:
            continue
        for ph, d in wide.groupby(level="phase"):
            f = _fid(d[(r, ra)].to_numpy(float), d[(s, a)].to_numpy(float))
            out.append({"phase": ph, "level": "rows", "space": s, "arm": a,
                        "ref_space": r, "ref_arm": ra, **f})
            pm = d[[(s, a), (r, ra)]].groupby(level=["client_a", "client_b"]).mean()
            f = _fid(pm[(r, ra)].to_numpy(float), pm[(s, a)].to_numpy(float))
            out.append({"phase": ph, "level": "pairs", "space": s, "arm": a,
                        "ref_space": r, "ref_arm": ra, **f})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Q2 clustering against regimes
# --------------------------------------------------------------------------
def _regime_partition(world, phase: str) -> dict | None:
    try:
        return dict(world.regimes(basis="token", phase=phase))
    except Exception:                                            # noqa: BLE001
        return None


def clusters(long: pd.DataFrame, regimes: dict[str, dict]) -> pd.DataFrame:
    """Per (space, arm, phase): average-linkage on 1 - S cut at the true k."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score
    out = []
    for (s, a, ph), d in long.groupby(["space", "arm", "phase"]):
        truth = regimes.get(ph)
        if not truth:
            continue
        pm = d.groupby(["client_a", "client_b"])["value"].mean()
        cl = sorted(set(pm.index.get_level_values(0)) | set(pm.index.get_level_values(1)))
        cl = [c for c in cl if c in truth]
        labels = [truth[c] for c in cl]
        k = len(set(labels))
        rec = {"space": s, "arm": a, "phase": ph, "k_true": k,
               "n_blocks": int(d["bid"].nunique()), "partition": "|".join(
                   f"{c[-1]}={truth[c]}" for c in cl)}
        if k in (1, len(cl)) or len(cl) < 3:
            out.append({**rec, "trivial": True})
            continue
        S = np.eye(len(cl))
        for (ca, cb), v in pm.items():
            if ca in cl and cb in cl:
                i, j = cl.index(ca), cl.index(cb)
                S[i, j] = S[j, i] = v
        D = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        pred = fcluster(Z, t=k, criterion="maxclust")
        same, diff = [], []
        best, best_v = None, -np.inf
        for (ca, cb), v in pm.items():
            if ca not in cl or cb not in cl or not np.isfinite(v):
                continue
            (same if truth[ca] == truth[cb] else diff).append(v)
            if v > best_v:
                best, best_v = (ca, cb), v
        auc = (np.mean([(x > y) + 0.5 * (x == y) for x in same for y in diff])
               if same and diff else np.nan)
        out.append({**rec, "trivial": False,
                    "ari": float(adjusted_rand_score(labels, pred)),
                    "pred": "|".join(f"{c[-1]}={p}" for c, p in zip(cl, pred)),
                    "top_pair": f"{best[0][-1]}{best[1][-1]}" if best else None,
                    "top_ok": bool(best and truth[best[0]] == truth[best[1]]),
                    "auc_same": float(auc), "n_same": len(same), "n_diff": len(diff)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Q3 tracking the drift
# --------------------------------------------------------------------------
def _block_progress(world, blocks: pd.DataFrame) -> dict[int, float]:
    prog = world.drift_progress().set_index("month")["progress"]
    return {int(r.bid): float(prog.loc[int(r.m_lo):int(r.m_hi)].mean())
            for r in blocks.itertuples(index=False)}


def _expected_sign(a: str, b: str, drift: str, r_init: dict, r_final: dict) -> int:
    if drift not in (a, b):
        return 0
    x = b if a == drift else a
    ai, af, rx = r_init.get(drift), r_final.get(drift), r_init.get(x)
    if rx is None or ai is None or af is None or ai == af:
        return 0
    if rx == ai:
        return -1
    if rx == af:
        return +1
    return 0


def tracking(long: pd.DataFrame, progress: dict[int, float], drift: str,
             regimes: dict[str, dict]) -> pd.DataFrame:
    from scipy.stats import spearmanr
    r0, r1 = regimes.get("init") or {}, regimes.get("final") or {}
    blk = (long.groupby(["space", "arm", "client_a", "client_b", "bid"])["value"]
           .mean().reset_index())
    blk["progress"] = blk["bid"].map(progress)
    out = []
    for (s, a, ca, cb), d in blk.groupby(["space", "arm", "client_a", "client_b"]):
        d = d.dropna(subset=["value", "progress"])
        exp = _expected_sign(ca, cb, drift, r0, r1)
        rec = {"space": s, "arm": a, "client_a": ca, "client_b": cb,
               "pair": f"{ca[-1]}{cb[-1]}", "has_drift": drift in (ca, cb),
               "expected": exp, "n_blocks": len(d)}
        if len(d) < 4 or d["progress"].nunique() < 2 or d["value"].std() < 1e-12:
            out.append({**rec, "rho": np.nan, "p_spearman": np.nan, "sign_ok": np.nan})
            continue
        rho, p = spearmanr(d["value"], d["progress"])
        out.append({**rec, "rho": float(rho), "p_spearman": float(p),
                    "sign_ok": (bool(np.sign(rho) == exp) if exp != 0 else np.nan)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# per run
# --------------------------------------------------------------------------
def run_config(row, p: SimParams, store=None, world=None, decoded: bool = True
               ) -> tuple[dict, dict]:
    tables: dict[str, list] = {}
    status: dict = {}
    key = row["run_key"]
    long = None
    blocks = None

    if decoded and world is not None and store is not None:
        with _Stage(status, "decoded:analyse"):
            from . import aligned as AL
            from . import recon as RC
            from .store import _spec_from_dict
            spec = _spec_from_dict(json.loads(
                (pathlib.Path(row["dir"]) / "spec.json").read_text()))
            GD.assert_cached(store, world, spec, expected_key=key)   # never trains
            align = AL.Alignment(period_h=p.period_h, bin_h=p.bin_h,
                                 block=p.block_months)
            out = AL.analyse(world, spec, align, store=store, server=False,
                             n_perm=p.n_perm)
            if not out["res"].get("cache_hit"):
                raise RuntimeError("analyse() trained -- refusing to use it")
            run = out["run"]
            channels = RC.channel_table(world, out["fw"])
            long = similarity_long(run, channels, p.arms)
            blocks = TJ.block_index(run)
            del out, run
    else:
        status["decoded:analyse"] = {"ok": False, "skipped": True,
                                     "error": "no world / store / repo"}

    if long is None:                    # world-free: raw latent only
        with _Stage(status, "latent:load"):
            t = GD.load_tables(row, tables=("latent_trajectories",))
            if t["latent_trajectories"] is None:
                raise FileNotFoundError("run has no latent_trajectories")
            wt = TJ.latent_windows(t["latent_trajectories"], t["fl"]["preprocessing"],
                                   t["meta"]["phases"], p.block_months,
                                   p.period_h, p.bin_h)
            long = latent_similarity_long(wt)
            blocks = (wt.groupby("bid")["month"].agg(m_lo="min", m_hi="max")
                      .reset_index())
    if long is None or not len(long):
        _add(tables, "status_note", pd.DataFrame([{"note": "no similarity computed"}]))
        return {k: _tag(pd.concat(v, ignore_index=True), run_key=key)
                for k, v in tables.items()}, status

    with _Stage(status, "similarity_block"):
        blk = (long.groupby(["space", "arm", "phase", "bid", "client_a", "client_b"])
               ["value"].agg(["mean", "std", "size"]).reset_index()
               .rename(columns={"mean": "value", "std": "sd_over_classes",
                                "size": "n_classes"}))
        _add(tables, "similarity_block", blk)
    with _Stage(status, "q1_fidelity"):
        _add(tables, "fidelity", fidelity(long))

    regimes = {}
    if world is not None:
        with _Stage(status, "regimes"):
            for ph in ("init", "final"):
                regimes[ph] = _regime_partition(world, ph)
            _add(tables, "regimes", pd.DataFrame(
                [{"phase": ph, "client": c, "regime": r}
                 for ph, m in regimes.items() if m for c, r in m.items()]))
        with _Stage(status, "q2_clusters"):
            sub = long[long["phase"].isin(p.cluster_phases)]
            _add(tables, "clusters", clusters(sub, regimes))
        with _Stage(status, "q3_tracking"):
            prog = _block_progress(world, blocks)
            _add(tables, "block_progress",
                 pd.DataFrame([{"bid": b, "progress": v} for b, v in prog.items()]))
            _add(tables, "tracking", tracking(long, prog, world.drift_district, regimes))
    else:
        for st in ("q2_clusters", "q3_tracking"):
            status[st] = {"ok": False, "skipped": True,
                          "error": "needs the world (regimes, drift progress)"}
    gc.collect()
    return {k: _tag(pd.concat(v, ignore_index=True), run_key=key)
            for k, v in tables.items()}, status


# --------------------------------------------------------------------------
# the loop and the bundle
# --------------------------------------------------------------------------
def sweep(idx: pd.DataFrame, p: SimParams, out_dir, store=None, load_world=None,
          decoded: bool = True, overwrite: bool = False, progress=None
          ) -> pd.DataFrame:
    """Every row of `idx`; checkpoints per run under `out_dir/parts/`."""
    out_dir = pathlib.Path(out_dir)
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    done = set() if overwrite else _done(out_dir)
    worlds: dict = {}
    rows = []
    t_start = time.time()
    todo = idx.reset_index(drop=True)
    for i, r in todo.iterrows():
        key = r["run_key"]
        if progress:
            progress(i, len(todo), r.get("label", key), time.time() - t_start)
        if key in done:
            rows.append({"run_key": key, "skipped": "already done"})
            continue
        world = None
        wp, wh = r.get("world_path"), r.get("world_hash")
        if decoded and load_world is not None and wp and pathlib.Path(str(wp)).exists():
            if wh not in worlds:
                try:
                    worlds[wh] = load_world(wp)
                except Exception:                                # noqa: BLE001
                    worlds[wh] = None
            world = worlds[wh]
        t0 = time.time()
        tables, status = run_config(r, p, store=store, world=world, decoded=decoded)
        part = out_dir / "parts" / key
        part.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(part / f"{name}.parquet", index=False)
        stages = {k: v for k, v in status.items() if isinstance(v, dict)}
        n_fail = sum(1 for v in stages.values()
                     if not v.get("ok") and not v.get("skipped"))
        summary = {"run_key": key, "label": r.get("label"),
                   "seconds": round(time.time() - t0, 1), "stages": stages,
                   "n_failed_stages": n_fail}
        (part / "status.json").write_text(json.dumps(_jsonable(summary), indent=1,
                                                     default=str))
        rows.append({"run_key": key, "seconds": summary["seconds"],
                     "n_failed_stages": n_fail})
        gc.collect()
    if progress:
        progress(len(todo), len(todo), "done", time.time() - t_start)
    return pd.DataFrame(rows)


def _headline(t: dict, axes: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, space, arm): the numbers the three questions read."""
    recs: dict = {}

    def rec(k, s, a):
        return recs.setdefault((k, s, a), {"run_key": k, "space": s, "arm": a})

    fd = t.get("fidelity")
    if fd is not None and len(fd):
        for r in fd[fd["level"] == "pairs"].itertuples(index=False):
            ref = "dem" if r.ref_space == "s_dem" else "true"
            rec(r.run_key, r.space, r.arm)[f"fid_{ref}_{r.phase}"] = r.spearman
    cl = t.get("clusters")
    if cl is not None and len(cl):
        for r in cl[~cl["trivial"].astype(bool)].itertuples(index=False):
            x = rec(r.run_key, r.space, r.arm)
            x[f"ari_{r.phase}"] = r.ari
            x[f"top_ok_{r.phase}"] = r.top_ok
            x[f"auc_same_{r.phase}"] = r.auc_same
    tr = t.get("tracking")
    if tr is not None and len(tr):
        for (k, s, a), d in tr.groupby(["run_key", "space", "arm"]):
            x = rec(k, s, a)
            e = d[d["expected"] != 0].dropna(subset=["rho"])
            n = d[(~d["has_drift"].astype(bool))].dropna(subset=["rho"])
            x["track_sign_ok"] = float(e["sign_ok"].astype(float).mean()) if len(e) else np.nan
            x["track_rho_signed_median"] = (float((e["rho"] * e["expected"]).median())
                                            if len(e) else np.nan)
            x["track_rho_nondrift_abs_median"] = (float(n["rho"].abs().median())
                                                  if len(n) else np.nan)
    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(list(recs.values())).merge(axes, on="run_key", how="left")


DICTIONARY = {
    "configs": "one row per run: identity, experiment axes, comparability block, status",
    "similarity_block": "run x space x arm x phase x block x pair: class-mean similarity "
                        "(sd over classes, n classes)",
    "fidelity": "run x phase x level(rows|pairs) x federated (space, arm) x ceiling "
                "(ref_space, ref_arm): pearson, spearman, concordant, mean_err",
    "regimes": "run x phase(init|final) x client: World.regimes token",
    "clusters": "run x space x arm x phase: k_true, trivial, ari, top_pair, top_ok, "
                "auc_same (average linkage on 1-S cut at k_true)",
    "block_progress": "run x block: mean drift progress of the drift district",
    "tracking": "run x space x arm x pair: Spearman rho(similarity, drift progress) "
                "over blocks, expected sign, sign_ok",
    "headline": "run x space x arm: fid_{dem,true}_{phase}, ari/top_ok/auc_same_{phase}, "
                "track_sign_ok, track_rho_signed_median, track_rho_nondrift_abs_median",
}


def collect(out_dir, idx: pd.DataFrame, p: SimParams, extra: dict | None = None,
            zip_name: str = "POC_05c_similarity_bundle.zip") -> dict:
    import platform
    import zipfile
    out_dir = pathlib.Path(out_dir)
    tdir = out_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    names = sorted({f.stem for d in (out_dir / "parts").glob("*")
                    for f in d.glob("*.parquet")})
    t = {}
    for n in names:
        fr = [pd.read_parquet(f) for f in (out_dir / "parts").glob(f"*/{n}.parquet")]
        t[n] = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
        t[n].to_parquet(tdir / f"{n}.parquet", index=False)
    stat = []
    for d in sorted((out_dir / "parts").glob("*")):
        f = d / "status.json"
        if f.exists():
            s = json.loads(f.read_text())
            row = {"run_key": s["run_key"], "seconds": s["seconds"],
                   "n_failed_stages": s["n_failed_stages"]}
            for st, v in s["stages"].items():
                row[f"ok:{st}"] = v.get("ok")
                if not v.get("ok") and v.get("error"):
                    tag = "skip" if v.get("skipped") else "err"      # skipped != failed
                    row[f"{tag}:{st}"] = str(v["error"])[:300]
            stat.append(row)
    status = pd.DataFrame(stat)
    keep = [c for c in idx.columns if c not in ("phases", "clients")]
    configs = idx[keep].copy()
    for c in configs.columns:
        if configs[c].dtype == object:
            configs[c] = configs[c].map(lambda v: v if v is None or isinstance(v, str)
                                        else json.dumps(v, default=str))
    configs = configs.merge(status, on="run_key", how="left")
    configs.to_parquet(tdir / "configs.parquet", index=False)
    axes = [c for c in ("run_key", "label", "world_hash", "drift_district", "schedule",
                        "proto_weight", "spec_term", "geometry", "rounds", "orient",
                        "kinds", "cmp_block") if c in idx.columns]
    head = _headline(t, idx[axes])
    head.to_parquet(tdir / "headline.parquet", index=False)
    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("sklearn", "scipy", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:                                        # noqa: BLE001
            pass
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": p.as_dict(),
                "method_hash": GD.method_hash("POC_05c_similarity", p.as_dict()),
                "n_runs_indexed": int(len(idx)), "n_runs_done": int(len(status)),
                "n_runs_with_failed_stages":
                    int((status["n_failed_stages"] > 0).sum()) if len(status) else 0,
                "tables": {n: int(len(df)) for n, df in t.items()},
                "dictionary": DICTIONARY, "versions": versions,
                "platform": platform.platform(), **(extra or {})}
    (out_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1,
                                                      default=str))
    zp = out_dir / zip_name
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "manifest.json", "manifest.json")
        for f in sorted(tdir.glob("*.parquet")):
            z.write(f, f"tables/{f.name}")
    return {"tables": t, "configs": configs, "headline": head,
            "manifest": manifest, "zip": zp}
