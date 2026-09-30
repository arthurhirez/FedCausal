"""The grid sweep: every POC_05 estimator, on every stored configuration.

POC_05 ran the full estimator stack on ONE cell and only raw transfer gain on
the rest. This module runs the stack on every stored run, individually, and
writes the results as long-form tables keyed by `run_key`, so configurations
can be compared without re-running anything.

Nothing here trains, and nothing here is new estimation: every number comes
from the same `traj` / `transfer` / `chain` / `partialdep` / `precond` calls
POC_05 executed. What this module adds is the loop, the per-stage error
isolation, the checkpointing and the bundle.

Arms
----
``pure`` / ``mixed`` / ``all``
    DECODED arm: `aligned.analyse` under the run's own final model, features
    of the decoded class-level prototypes (`traj.trajectory(...)["phi_class"]`)
    restricted to that slot class. Needs the world and the repo.
``latent``
    RAW-LATENT arm: the group-mean latent, per-client PCA fitted on init
    (`traj.latent_phi`); the chain uses the pooled basis, because its
    codebook is cross-client. Needs only the saved run.

Which stages run on which arms is a parameter (`SweepParams`): transfer on
every arm, the chain / partial / onset stages on `pure` and `latent` by
default -- they are the expensive ones and the pure arm is the demand claim.

Two choices that differ from POC_05's single-cell notebook
----------------------------------------------------------
* The ramp for the onset test is the set of blocks the ANALYSED phi labels
  `transition`. POC_05's decoded branch took `precond.drift_ramp`, which
  numbers blocks globally (`month // block_months`); the pipeline cuts blocks
  inside each phase, so the two numberings drift apart after the first
  partial block. Taking the ramp from phi keeps both arms on one numbering.
* The MI nulls are computed on the two clients of the pair only. Identical
  results (checked on saved runs), ~5x faster for the month-block null.

Checkpointing
-------------
Each run writes `parts/<run_key>/<table>.parquet` plus `status.json` as soon
as it finishes; a restarted sweep skips runs whose `status.json` exists
(`overwrite=False`). `collect` concatenates the parts into `tables/`, writes
`headline.parquet` and `manifest.json`, and zips the lot.
"""
from __future__ import annotations

import gc
import json
import pathlib
import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import chain as CH
from . import grid as GD
from . import partialdep as PD
from . import precond as PC
from . import traj as TJ
from . import transfer as TR
from .checks import Register

__all__ = ["SweepParams", "world_stage", "run_config", "sweep", "collect",
           "DICTIONARY"]

DECODED_ARMS = ("pure", "mixed", "all")


@dataclass(frozen=True)
class SweepParams:
    block_months: int = 3
    bin_h: int | None = 24
    period_h: int = 168
    k_states: int = 3
    n_surr: int = 100                 # capacity-null surrogates per ordered pair
    n_rep: int = 100                  # MI-null draws per pair and null
    n_stab: int = 10                  # codebook half-sample refits
    transfer_arms: tuple = ("pure", "mixed", "all", "latent")
    chain_arms: tuple = ("pure", "latent")
    partial_arms: tuple = ("pure", "latent")
    onset_arms: tuple = ("pure", "latent")
    onset_top: int = 3                # pairs with the largest |asymmetry|
    onset_win: int = 8
    onset_smooth: int = 3
    feats: tuple = ("lvl", "amp", "p24")
    n_comp: int = 2
    partial_feat_decoded: str = "amp"
    partial_feat_latent: str = "z0"
    lams: tuple = (0.01, 0.05, 0.1, 0.2, 0.5)
    partial_lam: float = 0.1
    p: int = 1
    min_train: int = 4
    lam: float = 1e-2
    control: str = "mean"
    control_time: str = "both"
    n_perm: int = 19                  # eta2 permutations inside analyse (unused here)
    seed: int = 0

    def cfg(self, feats) -> TR.TransferCfg:
        return TR.TransferCfg(p=self.p, min_train=self.min_train, lam=self.lam,
                              control=self.control,
                              control_time=self.control_time, feats=tuple(feats))

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _tag(df: pd.DataFrame | None, **kv) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    out = df.copy()
    for i, (k, v) in enumerate(kv.items()):
        out.insert(i, k, v)
    return out


def _add(tables: dict, name: str, df: pd.DataFrame) -> None:
    if df is not None and len(df):
        tables.setdefault(name, []).append(df)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


class _Stage:
    """`with _Stage(status, "chain:pure"):` -- errors are recorded, not raised."""

    def __init__(self, status: dict, name: str):
        self.status, self.name, self.t0 = status, name, None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, et, ev, tb):
        dt = round(time.time() - self.t0, 2)
        if et is None:
            self.status[self.name] = {"ok": True, "s": dt}
        else:
            self.status[self.name] = {
                "ok": False, "s": dt, "error": f"{et.__name__}: {ev}",
                "where": "".join(traceback.format_tb(tb, limit=-2))[-600:]}
        return True                                     # swallow: next stage runs


# --------------------------------------------------------------------------
# per world (once)
# --------------------------------------------------------------------------
def world_stage(world, p: SweepParams) -> dict:
    """Phases, blocks, storage exchange per tank, adjacency -- once per world."""
    wh = world.sim_hash
    out: dict[str, list] = {}
    reg = Register(f"world {wh}")
    st: dict = {}
    s0 = None
    with _Stage(st, "stage0"):
        s0 = PC.stage0(world, None, block_months=p.block_months,
                       n_classes=int(168 // (p.bin_h or 168)), k=p.k_states,
                       min_blocks_transition=8, reg=reg)
        _add(out, "world_phases", _tag(s0["phases"], world_hash=wh))
        if s0["storage"]["available"]:
            _add(out, "world_tanks", _tag(s0["storage"]["per_tank"], world_hash=wh))
        _add(out, "world_register", _tag(reg.frame().astype({"value": str}),
                                         world_hash=wh))
    with _Stage(st, "adjacency"):
        adj, src = PC.district_adjacency(world)
        _add(out, "world_adjacency", _tag(adj, world_hash=wh, source=src))
    summ = world.summary()
    info = {"world_hash": wh, "tag": world.tag, "network": world.network,
            "districting": world.districting_method,
            "placement_source": world.placement_source,
            "drift_district": summ.get("drift_district"),
            "regime_init": summ.get("regime_init"),
            "regime_final": summ.get("regime_final"),
            "lag_arm": bool(s0 is not None and s0["lag_arm"]),
            "stages": st}
    return {"tables": out, "info": info,
            "adjacency": out.get("world_adjacency", [pd.DataFrame()])[0]}


# --------------------------------------------------------------------------
# per arm
# --------------------------------------------------------------------------
def _transfer(phi, arm, p, tables, reg):
    feats = TJ.feature_cols(phi)
    cfg = p.cfg(feats)
    gm = TR.gain_matrix(phi, cfg)
    nl = TR.null_matrix(phi, cfg, n_surr=p.n_surr, kind="permute", seed=p.seed)
    rv = TR.gain_matrix(TR.reverse_blocks(phi), cfg)
    t = (gm[["source", "target", "G", "r2_restricted", "n_eval"]]
         .merge(nl[["source", "target", "null_mean", "null_sd", "null_q95",
                    "z", "p_emp", "n_surr"]], on=["source", "target"], how="left")
         .merge(rv[["source", "target", "G"]].rename(columns={"G": "G_reversed"}),
                on=["source", "target"], how="left"))
    t["p_floor"] = 1.0 / (t["n_surr"] + 1)
    t["feats"] = "+".join(feats)
    _add(tables, "transfer", _tag(t, arm=arm))
    reg.report("transfer", f"pairs clearing capacity null [{arm}]",
               f"{int((t['p_emp'] < 0.05).sum())}/{len(t)}",
               f"p floor {float(t['p_floor'].iloc[0]):.4f}")
    return t


def _onset(phi, arm, t, p, tables):
    ramp = sorted(phi.loc[phi["phase"] == "transition", "bid"].unique().tolist())
    asym = TR.asymmetry(t.rename(columns={}))
    asym = asym.reindex(asym["asym"].abs().sort_values(ascending=False).index)
    rows = []
    feats = TJ.feature_cols(phi)
    for r in asym.head(p.onset_top).itertuples(index=False):
        src, tgt = ((r.client_a, r.client_b) if r.G_a_to_b >= r.G_b_to_a
                    else (r.client_b, r.client_a))
        gb = TR.rolling_gain(phi, tgt, src, p.cfg(feats), win=p.onset_win,
                             smooth=p.onset_smooth)
        on = TR.onset_test(gb, ramp)
        on.update(source=src, target=tgt)
        rows.append(on)
        _add(tables, "rolling_gain",
             _tag(gb[["bid", "phase", "G"]] if len(gb) else gb, arm=arm,
                  source=src, target=tgt))
    _add(tables, "onset", _tag(pd.DataFrame(rows), arm=arm))


def _chain(phi_pool, arm, p, tables, reg):
    k = p.k_states
    stab = CH.codebook_stability(phi_pool, k=k, space="shape", n_rep=p.n_stab,
                                 seed=p.seed)
    cb = CH.fit_codebook(phi_pool, k=k, space="shape", fit_phase="init",
                         seed=p.seed)
    sym = CH.symbol_table(phi_pool, cb, seq_key="cycle_pos")
    clients = sorted(sym["client"].unique())
    o = CH.markov_order_test(CH.sequences(sym, clients[0]), k)
    p01 = float(o.loc[o["test"] == "order 0 vs 1", "p"].iloc[0])
    p12 = float(o.loc[o["test"] == "order 1 vs 2", "p"].iloc[0])
    g01 = float(o.loc[o["test"] == "order 0 vs 1", "G2"].iloc[0])
    n_tr = {c: int(sum(len(s) - 1 for s in CH.sequences(sym, c))) for c in clients}
    gates = pd.DataFrame([{
        "k": k, "space": "shape", "ari_mean": stab["ari_mean"],
        "ari_min": stab["ari_min"], "distortion": stab["distortion"],
        "stable": stab["stable"], "order01_G2": g01, "order01_p": p01,
        "order12_p": p12, "dynamics": p01 < 0.01,
        "n_transitions_min": min(n_tr.values())}])
    _add(tables, "chain_gates", _tag(gates, arm=arm))
    reg.gate("chain", f"codebook stability [{arm}]", round(stab["ari_mean"], 3),
             stab["stable"])
    reg.gate("chain", f"order 0 rejected [{arm}]", f"p={p01:.2e}", p01 < 0.01)

    occ = (pd.crosstab([sym["client"], sym["phase"]], sym["s"], normalize="index")
           .stack().rename("share").reset_index())
    _add(tables, "occupancy", _tag(occ, arm=arm))
    sims = [CH.occupancy_matrix(sym, k, ph) for ph in ("init", "transition", "final")]
    _add(tables, "occupancy_sim",
         _tag(pd.concat(sims, ignore_index=True), arm=arm))
    spl = pd.DataFrame([CH.stationarity_split(sym, k, c, ph)
                        for c in clients for ph in ("init", "transition", "final")])
    if len(spl):
        spl["n_first"] = [d.get("first", np.nan) for d in spl["n_transitions"]]
        spl["n_second"] = [d.get("second", np.nan) for d in spl["n_transitions"]]
        spl = spl.drop(columns=["n_transitions"])
    _add(tables, "stationarity", _tag(spl, arm=arm))

    pt = CH.pair_table(sym, k)
    _add(tables, "mi_te", _tag(pt.drop(columns=["phase"], errors="ignore"), arm=arm))
    nul = []
    for r in pt.itertuples(index=False):
        ab = sym[sym["client"].isin([r.client_a, r.client_b])]
        nul.append(CH.month_block_null(ab, k, r.client_a, r.client_b,
                                       n_rep=p.n_rep, seed=p.seed, stat="mi"))
        nul.append(CH.chain_null(ab, k, r.client_a, r.client_b,
                                 n_rep=p.n_rep, seed=p.seed, stat="mi"))
    nd = pd.DataFrame(nul)
    if len(nd):
        nd["degenerate"] = nd["null"].str.contains("degenerate")
        nd["null"] = nd["null"].str.replace(r" \(degenerate.*\)", "", regex=True)
        nd["p_floor"] = 1.0 / (nd["n_rep"] + 1)
    _add(tables, "mi_nulls", _tag(nd, arm=arm))


def _partial(phi, arm, feat, adjacency, p, tables, reg):
    X, cl = PD.client_matrix(phi, feat, seq_key="cycle_pos")
    cond = PD.condition_report(X, cl)
    c = pd.DataFrame([cond.to_dict()])
    c["feature"] = feat
    _add(tables, "partial_condition", _tag(c, arm=arm))
    reg.gate("partial", f"precision matrix estimable [{arm}]",
             round(float(cond["cond_number"]), 1), bool(cond["estimable"]))
    path = PD.shrinkage_path(X, cl, lams=p.lams)
    _add(tables, "partial", _tag(path.assign(feature=feat), arm=arm))
    mvp = PD.marginal_vs_partial(X, cl, lam=p.partial_lam)
    if adjacency is not None and len(adjacency):
        sc = PD.structural_contrast(mvp, adjacency[["client_a", "client_b",
                                                    "adjacent"]])
        _add(tables, "partial_structure",
             _tag(sc.assign(feature=feat, lam=p.partial_lam), arm=arm))


# --------------------------------------------------------------------------
# per run
# --------------------------------------------------------------------------
def run_config(row, p: SweepParams, store=None, world=None, world_info=None,
               adjacency=None, decoded: bool = True) -> tuple[dict, dict]:
    """Every stage for one stored run. -> (tables, status). Never trains."""
    tables: dict[str, list] = {}
    status: dict = {}
    key = row["run_key"]
    reg = Register(key)

    # ------------------------------------------------ latent arm (world-free)
    lat = {}
    with _Stage(status, "latent:load"):
        t = GD.load_tables(row, tables=("latent_trajectories", "prototypes"))
        lt, protos = t["latent_trajectories"], t["prototypes"]
        if lt is None:
            raise FileNotFoundError("run has no latent_trajectories")
        fc = [c for c in lt.columns if c.startswith("f") and c[1:].isdigit()]
        if protos is not None and "post_fedavg_full" in set(protos["scope"]):
            m = (lt.groupby(["district", "month"])[fc].mean().reset_index()
                 .rename(columns={"district": "client"}))
            j = m.merge(protos[protos["scope"] == "post_fedavg_full"],
                        on=["client", "month"], suffixes=("", "_r"))
            d = float(np.abs(j[fc].to_numpy()
                             - j[[f + "_r" for f in fc]].to_numpy()).max()) \
                if len(j) else np.nan
            reg.expect("contract", "month-mean latents == post_fedavg_full",
                       d, len(j) > 0 and d < 1e-6)
        wt = TJ.latent_windows(lt, t["fl"]["preprocessing"], t["meta"]["phases"],
                               p.block_months, p.period_h, p.bin_h)
        occ = wt.groupby(["client", "bid", "cycle_pos"]).size()
        reg.report("calendar", "occupancy min (windows per prototype)",
                   int(occ.min()))
        reg.report("calendar", "blocks per phase",
                   wt.groupby("phase")["bid"].nunique().to_dict())
        lat["phi"] = TJ.latent_phi(wt, n_comp=p.n_comp, basis="per_client")["phi"]
        lat["phi_pool"] = TJ.latent_phi(wt, n_comp=p.n_comp, basis="pooled")["phi"]
        status["n_windows"] = int(len(lt))
    if status.get("latent:load", {}).get("ok"):
        if "latent" in p.transfer_arms:
            with _Stage(status, "transfer:latent"):
                tl = _transfer(lat["phi"], "latent", p, tables, reg)
            if "latent" in p.onset_arms and status["transfer:latent"]["ok"]:
                with _Stage(status, "onset:latent"):
                    _onset(lat["phi"], "latent", tl, p, tables)
        if "latent" in p.chain_arms:
            with _Stage(status, "chain:latent"):
                _chain(lat["phi_pool"], "latent", p, tables, reg)
        if "latent" in p.partial_arms:
            with _Stage(status, "partial:latent"):
                _partial(lat["phi"], "latent", p.partial_feat_latent, adjacency,
                         p, tables, reg)

    # ----------------------------------------------------- decoded arms
    arms_needed = [a for a in DECODED_ARMS
                   if a in set(p.transfer_arms) | set(p.chain_arms)
                   | set(p.partial_arms) | set(p.onset_arms)]
    if not decoded or world is None or store is None or not arms_needed:
        status["decoded"] = {"ok": False, "skipped": True,
                             "error": "no world / store / repo"}
    else:
        dec = {}
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
            cb = PC.channel_balance(channels)
            reg.expect("stage0", "channels per client equal",
                       cb["total"].to_dict(), cb["total"].nunique() == 1)
            fs = TJ.FeatureSpec(feats=tuple(p.feats), band_width=1)
            for arm in arms_needed:
                dec[arm] = TJ.trajectory(run, channels,
                                         None if arm == "all" else arm, fs)
            rec = TJ.recoverability(dec[arms_needed[0]]["phi"],
                                    dec[arms_needed[0]]["phi_demand"])
            _add(tables, "recoverability", _tag(rec, arm=arms_needed[0]))
            if "pure" in dec:
                hist = TJ.history_features(out["res"], out["model"], world,
                                           channels, "pure", fs, agg_h=run.agg_h)
                rs = TJ.ruler_stability(dec["pure"]["phi"], hist,
                                        dec["pure"]["blocks"])
                _add(tables, "ruler_stability", _tag(rs, arm="pure"))
            del out, run
        if status["decoded:analyse"]["ok"]:
            for arm in arms_needed:
                phi = dec[arm]["phi_class"]
                if arm in p.transfer_arms:
                    with _Stage(status, f"transfer:{arm}"):
                        ta = _transfer(phi, arm, p, tables, reg)
                    if arm in p.onset_arms and status[f"transfer:{arm}"]["ok"]:
                        with _Stage(status, f"onset:{arm}"):
                            _onset(phi, arm, ta, p, tables)
                if arm in p.chain_arms:
                    with _Stage(status, f"chain:{arm}"):
                        _chain(phi, arm, p, tables, reg)
                if arm in p.partial_arms:
                    with _Stage(status, f"partial:{arm}"):
                        _partial(phi, arm, p.partial_feat_decoded, adjacency,
                                 p, tables, reg)
        del dec
        gc.collect()

    _add(tables, "register", reg.frame().astype({"value": str}))
    tables = {k: _tag(pd.concat(v, ignore_index=True), run_key=key)
              for k, v in tables.items()}
    return tables, status


# --------------------------------------------------------------------------
# the loop, and the bundle
# --------------------------------------------------------------------------
def _done(out_dir: pathlib.Path) -> set:
    return {d.name for d in (out_dir / "parts").glob("*")
            if (d / "status.json").exists()}


def sweep(idx: pd.DataFrame, p: SweepParams, out_dir, store=None,
          load_world=None, decoded: bool = True, overwrite: bool = False,
          progress=None) -> pd.DataFrame:
    """Run every row of `idx` (a `grid.scan` frame). Checkpoints per run.

    `load_world(path) -> World` is `fedwater_lab.worlds.load_world` when the
    repo is on the path; `None` runs the latent arm only. `progress` is a
    callable `(i, n, label, seconds_so_far)` for the notebook's bar.
    """
    out_dir = pathlib.Path(out_dir)
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    (out_dir / "worlds").mkdir(parents=True, exist_ok=True)
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
        # ---- the run's world, once
        world = winfo = adj = None
        wp, wh = r.get("world_path"), r.get("world_hash")
        if decoded and load_world is not None and wp and pathlib.Path(str(wp)).exists():
            if wh not in worlds:
                try:
                    w = load_world(wp)
                    ws = world_stage(w, p)
                    for name, parts in ws["tables"].items():
                        pd.concat(parts, ignore_index=True).to_parquet(
                            out_dir / "worlds" / f"{wh}__{name}.parquet", index=False)
                    (out_dir / "worlds" / f"{wh}__info.json").write_text(
                        json.dumps(_jsonable(ws["info"]), indent=1, default=str))
                    worlds[wh] = (w, ws)
                except Exception as exc:                          # noqa: BLE001
                    worlds[wh] = (None, {"info": {"error": f"{type(exc).__name__}: {exc}"},
                                         "adjacency": pd.DataFrame()})
            world, ws = worlds[wh]
            winfo, adj = ws["info"], ws.get("adjacency")
        t0 = time.time()
        tables, status = run_config(r, p, store=store, world=world,
                                    world_info=winfo, adjacency=adj,
                                    decoded=decoded)
        part = out_dir / "parts" / key
        part.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(part / f"{name}.parquet", index=False)
        stages = {k: v for k, v in status.items() if isinstance(v, dict)}
        n_fail = sum(1 for v in stages.values()
                     if not v.get("ok") and not v.get("skipped"))
        summary = {"run_key": key, "label": r.get("label"),
                   "seconds": round(time.time() - t0, 1),
                   "n_windows": status.get("n_windows"),
                   "stages": stages, "n_failed_stages": n_fail}
        (part / "status.json").write_text(json.dumps(_jsonable(summary), indent=1,
                                                     default=str))
        rows.append({"run_key": key, "seconds": summary["seconds"],
                     "n_failed_stages": n_fail})
        gc.collect()
    if progress:
        progress(len(todo), len(todo), "done", time.time() - t_start)
    return pd.DataFrame(rows)


def _headline(t: dict, configs: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, arm): the numbers a verdict is read from.

    `drift_separation` scores idea 2 against the ground truth (evaluation
    only): with d = sim_init - sim_final per pair, the mean d of pairs that
    contain the drift district minus the mean d of the other pairs. Positive
    = the drifted client's pairs lost more similarity than everyone else's.
    """
    drift = dict(zip(configs["run_key"], configs.get("drift_district",
                                                     pd.Series(dtype=object))))
    recs: dict = {}

    def rec(key, arm):
        return recs.setdefault((key, arm), {"run_key": key, "arm": arm})

    tr = t.get("transfer")
    if tr is not None and len(tr):
        for (key, arm), d in tr.groupby(["run_key", "arm"]):
            r = rec(key, arm)
            b = d.loc[d["G"].idxmax()]
            r.update(pairs_clear_null=int((d["p_emp"] < 0.05).sum()),
                     n_pairs=len(d), G_max=float(d["G"].max()),
                     G_median=float(d["G"].median()),
                     null_mean_median=float(d["null_mean"].median()),
                     best_pair=f"{b['source']}->{b['target']}",
                     best_pair_p=float(b["p_emp"]))
    cg = t.get("chain_gates")
    if cg is not None and len(cg):
        for r0 in cg.itertuples(index=False):
            rec(r0.run_key, r0.arm).update(
                codebook_stable=bool(r0.stable), ari_mean=float(r0.ari_mean),
                order01_p=float(r0.order01_p), dynamics=bool(r0.dynamics))
    mn = t.get("mi_nulls")
    if mn is not None and len(mn):
        for (key, arm, nul), d in mn.groupby(["run_key", "arm", "null"]):
            tag = "monthblock" if nul.startswith("month") else "chain"
            rec(key, arm)[f"mi_pairs_sig_{tag}"] = int((d["p_emp"] < 0.05).sum())
    mt = t.get("mi_te")
    if mt is not None and len(mt):
        for (key, arm), d in mt.groupby(["run_key", "arm"]):
            rec(key, arm).update(
                mi_pairs_above_bias=int((d["mi"] > d["mi_bias"]).sum()),
                te_dirs_above_bias=int((d["te_a_to_b"] > d["te_bias"]).sum()
                                       + (d["te_b_to_a"] > d["te_bias"]).sum()))
    os_ = t.get("occupancy_sim")
    if os_ is not None and len(os_):
        for (key, arm), d in os_.groupby(["run_key", "arm"]):
            dd = drift.get(key)
            w = d.pivot_table(index=["client_a", "client_b"], columns="phase",
                              values="sim")
            val = np.nan
            if dd is not None and {"init", "final"} <= set(w.columns):
                delta = (w["init"] - w["final"]).reset_index(name="d")
                has = (delta["client_a"] == dd) | (delta["client_b"] == dd)
                if has.any() and (~has).any():
                    val = float(delta.loc[has, "d"].mean()
                                - delta.loc[~has, "d"].mean())
            rec(key, arm)["drift_separation"] = val
    pc = t.get("partial_condition")
    if pc is not None and len(pc):
        for r0 in pc.itertuples(index=False):
            rec(r0.run_key, r0.arm).update(partial_cond=float(r0.cond_number),
                                           partial_estimable=bool(r0.estimable))
    ps = t.get("partial_structure")
    if ps is not None and len(ps):
        for r0 in ps.itertuples(index=False):
            rec(r0.run_key, r0.arm)[f"auc_{r0.signal}"] = float(r0.auc)
    if not recs:
        return pd.DataFrame()
    h = pd.DataFrame(list(recs.values()))
    return h.merge(configs, on="run_key", how="left")


DICTIONARY = {
    "configs": "one row per run: identity, experiment axes, comparability block, status",
    "transfer": "run x arm x ordered pair: G, r2_restricted, n_eval, capacity null "
                "(null_mean, null_sd, null_q95, z, p_emp, p_floor), G_reversed",
    "onset": "run x arm x top-|asym| pair: peak block of G(b), in_ramp, p_uniform",
    "rolling_gain": "run x arm x pair x block: G(b)",
    "chain_gates": "run x arm: codebook ARI / distortion, Markov order tests",
    "occupancy": "run x arm x client x phase x state: share of blocks in state",
    "occupancy_sim": "run x arm x pair x phase: 1 - JS(pi_a, pi_b) / log 2",
    "stationarity": "run x arm x client x phase: homogeneity G2 of the two halves",
    "mi_te": "run x arm x pair: MI, TE both directions, plug-in biases, n",
    "mi_nulls": "run x arm x pair x null (month blocks | fitted chains): p_emp, degenerate",
    "partial_condition": "run x arm: rows/parameter, condition number, verdict",
    "partial": "run x arm x pair x lambda: marginal_r, partial_r, sign stability",
    "partial_structure": "run x arm x signal: AUC of |r| for adjacent vs other pairs",
    "recoverability": "run x arm x client x feature: r(model phi, own-demand phi) over blocks",
    "ruler_stability": "run x client x feature: r(frozen, moving ruler) and its n",
    "register": "run x every check / gate / report the stages filed",
    "headline": "run x arm: the summary numbers a verdict is read from, with the axes",
    "worlds/*": "per world: phases and blocks, tanks (storage exchange), adjacency, stage-0 register",
}


def collect(out_dir, idx: pd.DataFrame, p: SweepParams, extra: dict | None = None,
            zip_name: str = "POC_05_grid_bundle.zip") -> dict:
    """Concatenate the parts into `tables/`, write headline + manifest, zip."""
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
                   "n_windows": s.get("n_windows"),
                   "n_failed_stages": s["n_failed_stages"]}
            for st, v in s["stages"].items():
                row[f"ok:{st}"] = v.get("ok")
                if not v.get("ok") and v.get("error"):
                    row[f"err:{st}"] = v["error"][:300]
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
    axes = [c for c in ("run_key", "label", "drift_district", "schedule",
                        "schedule_months", "proto_weight", "spec_term", "diff_term",
                        "var_term", "geometry", "rounds", "orient", "kinds",
                        "cmp_block", "world_hash") if c in idx.columns]
    head = _headline(t, idx[axes])
    head.to_parquet(tdir / "headline.parquet", index=False)

    try:
        import sklearn, scipy                                    # noqa: E401
        versions = {"numpy": np.__version__, "pandas": pd.__version__,
                    "sklearn": sklearn.__version__, "scipy": scipy.__version__}
    except Exception:                                            # noqa: BLE001
        versions = {"numpy": np.__version__, "pandas": pd.__version__}
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:                                            # noqa: BLE001
        pass
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "params": p.as_dict(),
                "method_hash": GD.method_hash("POC_05_sweep", p.as_dict()),
                "n_runs_indexed": int(len(idx)),
                "n_runs_done": int(len(status)),
                "n_runs_with_failed_stages": int((status["n_failed_stages"] > 0).sum())
                if len(status) else 0,
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
        for f in sorted((out_dir / "worlds").glob("*")):
            z.write(f, f"worlds/{f.name}")
    return {"tables": t, "configs": configs, "headline": head,
            "manifest": manifest, "zip": zp}
