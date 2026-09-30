"""Self-check against SAVED RUNS -- no synthetic data anywhere.

Point it at a directory of stored runs (`runs/<world>/<key>/`, or a flat
export of run folders) and it exercises every estimator on the artifacts
those runs actually contain, filing each verdict in a `checks.Register`.

What it can check without the world
-----------------------------------
A saved run carries `latent_trajectories` (every window's latent under the
final weights), `prototypes`, `prototype_history`, `latents_by_round` and
its own `fl.json` / `meta.json`. That is enough for the RAW-LATENT arm of
the trajectory (`traj.latent_windows` + `traj.latent_phi`) and therefore for
every estimator in Stages 3-5. It is NOT enough for the decoded arm, which
needs the decoder (`fed_model.pt` plus the repo's AER class) and the world
(district demand, the sensor table). Those checks are reported as
`unavailable` rather than skipped silently, and the notebook runs them when
the repo is importable.

The checks that can fail
------------------------
``protocol contract``
    The per-month mean of `latent_trajectories` must equal the run's own
    `post_fedavg_full` prototype. This is what makes the trajectory THE
    artifact the protocol exchanges rather than something adjacent to it --
    if it fails, the run's latents and prototypes came from different
    weights and nothing downstream is about the federated object.

``occupancy floor``
    Windows per (client, block, cycle class) at the chosen key. Below the
    floor the prototype is one window: the privacy floor and the averaging
    floor are the same number.

``capacity null behaviour``
    Refit with a dimension-matched surrogate source. The null is REPORTED,
    not assumed: a null mean far from zero means the gain has to be read
    against it rather than against zero, which is the whole reason it
    exists.

``markov order``
    Order 0 vs 1 must be rejected for the chain to be worth anything over
    an occupancy histogram.

``codebook stability``
    ARI of half-sample refits against the full fit.

``partial-correlation precondition``
    Condition number of the client correlation matrix. Above 100 the
    partial correlations are not interpretable and the stage says so
    instead of returning numbers.

Plus a set of reported measurements (phase spans, block budget, plug-in
bias, transitions per half) that never fail but that no later claim should
be read without.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import chain as CH
from . import grid as G
from . import partialdep as PD
from . import traj as TJ
from . import transfer as TR
from .checks import Register

__all__ = ["check_run", "check_store"]


def _read(d, name):
    import pathlib
    d = pathlib.Path(d)
    for suf in (".parquet", ".csv.gz"):
        f = (d / name).with_suffix(suf)
        if f.exists():
            return pd.read_parquet(f) if suf == ".parquet" else pd.read_csv(f)
    return None


def check_run(row, block_months: int = 3, bin_h: int | None = 24,
              period_h: int = 168, n_comp: int = 2, k: int = 3,
              n_min: int = 4, n_surr: int = 60, n_rep: int = 60,
              reg: Register | None = None) -> dict:
    """Every estimator, on one saved run. `row` is a `grid.scan` row."""
    reg = Register("selfcheck") if reg is None else reg
    t = G.load_tables(row, tables=("latent_trajectories", "prototypes",
                                   "prototype_history"))
    meta, fl = t["meta"], t["fl"]
    pp = fl["preprocessing"]
    lt, protos = t["latent_trajectories"], t["prototypes"]
    label = meta.get("label", meta.get("run_key"))
    fc = [c for c in lt.columns if c.startswith("f") and c[1:].isdigit()]

    reg.report("run", "cell", label,
               f"{meta.get('network')} · {meta.get('districting')} · "
               f"drift {meta.get('drift_district')}")

    # ---------------------------------------------------- calendar and keys
    res = TJ.resolution_h(lt, pp)
    reg.expect("calendar", "native resolution recovered (h)", res,
               res > 0 and abs(res - round(res)) < 1e-6,
               "step_size x interval_agg_h / median diff(window start)")
    wt = TJ.latent_windows(lt, pp, meta["phases"], block_months, period_h, bin_h)
    n_cls = int(wt["cycle_pos"].nunique())
    blocks = wt.groupby("phase")["bid"].nunique().to_dict()
    reg.report("calendar", "cycle classes", n_cls,
               f"period {period_h} h, bin {bin_h}")
    reg.report("calendar", "blocks per phase", blocks,
               f"block = {block_months} months")

    occ = wt.groupby(["client", "bid", "cycle_pos"]).size()
    reg.expect("calendar", "occupancy floor (windows per prototype)",
               int(occ.min()), int(occ.min()) >= n_min,
               "below the floor a prototype IS one window -- the privacy "
               "floor and the averaging floor are the same number")

    # ----------------------------------------------- the protocol contract
    if protos is not None and "post_fedavg_full" in set(protos["scope"]):
        m = (lt.groupby(["district", "month"])[fc].mean().reset_index()
             .rename(columns={"district": "client"}))
        pf = protos[protos["scope"] == "post_fedavg_full"]
        j = m.merge(pf, on=["client", "month"], suffixes=("", "_r"))
        d = float(np.abs(j[fc].to_numpy()
                         - j[[f + "_r" for f in fc]].to_numpy()).max()) \
            if len(j) else np.nan
        reg.expect("contract", "month-mean latents == post_fedavg_full", d,
                   len(j) > 0 and d < 1e-6,
                   "the trajectory is the artifact the protocol exchanges")
    else:
        reg.report("contract", "month-mean latents == post_fedavg_full",
                   "unavailable", "no post_fedavg_full scope in this run")

    # --------------------------------------------------------- Stage 3
    phi = TJ.latent_phi(wt, n_comp=n_comp, basis="per_client")["phi"]
    feats = tuple(f"z{j}" for j in range(n_comp))
    cfg = TR.TransferCfg(p=1, min_train=4, lam=1e-2, control="mean",
                         control_time="both", feats=feats)
    gm = TR.gain_matrix(phi, cfg)
    reg.report("transfer", "restricted AR-R^2 per target",
               gm.groupby("target")["r2_restricted"].mean().round(3).to_dict(),
               "if G tracks this across targets, the asymmetry is a "
               "self-predictability artifact ON THIS WORLD")
    reg.report("transfer", "G range", (round(float(gm["G"].min()), 3),
                                       round(float(gm["G"].max()), 3)),
               "negative = the source bought nothing out of sample")

    best = gm.sort_values("G", ascending=False).iloc[0]
    nl = TR.capacity_null(phi, best["target"], best["source"], cfg,
                          n_surr=n_surr, seed=0)
    reg.expect("transfer", "capacity null is usable",
               {k2: round(v, 3) for k2, v in nl.items()
                if k2 in ("G", "null_mean", "null_q95", "p_emp")},
               abs(nl["null_mean"]) < 0.25,
               "a null far from 0 means G must be read against it, never "
               "against 0")
    rev = TR.transfer_gain(TR.reverse_blocks(phi), best["target"],
                           best["source"], cfg)
    reg.report("transfer", "time reversal (best pair)",
               {"forward": round(float(best["G"]), 3),
                "reversed": round(float(rev["G"]), 3)},
               "propagation weakens or inverts; a variance artifact is "
               "symmetric")

    # --------------------------------------------------------- Stage 4
    phip = TJ.latent_phi(wt, n_comp=n_comp, basis="pooled")["phi"]
    stab = CH.codebook_stability(phip, k=k, space="shape", seed=0)
    reg.gate("chain", f"codebook stability at k={k}",
               {"ari_mean": round(stab["ari_mean"], 3),
                "distortion": round(stab["distortion"], 3)},
               stab["stable"],
               "low ARI = k too large or the space is not clustered")
    cb = CH.fit_codebook(phip, k=k, space="shape", fit_phase="init", seed=0)
    sym = CH.symbol_table(phip, cb, seq_key="cycle_pos")
    n_tr = {c: int(sum(len(s) - 1 for s in CH.sequences(sym, c)))
            for c in sorted(sym["client"].unique())}
    reg.report("chain", "transitions per client (classes pooled)", n_tr,
               "classes buy ESTIMATION, not inference -- the null permutes "
               "whole months")
    clients = sorted(sym["client"].unique())
    order = CH.markov_order_test(CH.sequences(sym, clients[0]), k)
    p01 = float(order.loc[order["test"] == "order 0 vs 1", "p"].iloc[0])
    p12 = float(order.loc[order["test"] == "order 1 vs 2", "p"].iloc[0])
    reg.gate("chain", "order 0 rejected (there are dynamics)",
               f"p={p01:.1e}", p01 < 0.01,
               "not rejected = the chain adds nothing over the histogram")
    reg.report("chain", "order 1 vs 2", f"p={p12:.3f}",
               "not rejected = order 1 is memory enough")

    occ_m = CH.occupancy_matrix(sym, k, phase="final")
    occ_i = CH.occupancy_matrix(sym, k, phase="init")
    reg.report("chain", "occupancy similarity, init -> final",
               {f"{a[-1]}{b[-1]}": (round(float(i), 2), round(float(f2), 2))
                for a, b, i, f2 in zip(occ_i["client_a"], occ_i["client_b"],
                                       occ_i["sim"], occ_m["sim"])},
               "pairs the drifted client belongs to should fall")

    splits = {c: CH.stationarity_split(sym, k, c, "init") for c in clients}
    n_half = {c: v["n_transitions"] for c, v in splits.items()}
    reg.report("chain", "stationarity split: transitions per half", n_half,
               "read the p-values only if these are large enough to have "
               "power")

    pt = CH.pair_table(sym, k)
    above = pt["mi"] > pt["mi_bias"]
    reg.report("chain", "MI above plug-in bias", f"{int(above.sum())}/{len(pt)}",
               f"bias = {float(pt['mi_bias'].iloc[0]):.4f} nats at "
               f"n={int(pt['n'].iloc[0])}")
    te_ok = (pt[["te_a_to_b", "te_b_to_a"]].max(axis=1) > pt["te_bias"])
    reg.report("chain", "TE above plug-in bias", f"{int(te_ok.sum())}/{len(pt)}",
               f"bias = {float(pt['te_bias'].iloc[0]):.4f} nats -- nothing "
               "below this is a finding")
    mb = CH.month_block_null(sym, k, clients[0], clients[1], n_rep=n_rep,
                             seed=0, stat="mi")
    reg.report("chain", "MI against the month-block null",
               {k2: round(v, 3) for k2, v in mb.items()
                if k2 in ("value", "null_mean", "null_q95", "p_emp")},
               "a high null mean means most of MI is the shared calendar")

    # --------------------------------------------------------- Stage 5
    X, cl = PD.client_matrix(phi, feats[0], seq_key="cycle_pos")
    cond = PD.condition_report(X, cl)
    reg.gate("partial", "precision matrix is estimable",
               {"rows/par": round(float(cond["rows_per_parameter"]), 1),
                "cond": round(float(cond["cond_number"]), 1),
                "verdict": cond["verdict"]}, bool(cond["estimable"]),
               "above 100 the partial correlations are not interpretable")
    mvp = PD.marginal_vs_partial(X, cl, lam=0.1)

    # --------------------------------------------- untrained control cover
    lb = _read(row["dir"] if isinstance(row, dict) else row.dir,
               "latents_by_round")
    if lb is not None and -1 in set(lb["round"]):
        c0 = lb[(lb["round"] == -1) & (lb.get("stage", "pre") == "pre")]
        ph = {TJ.latent_windows(c0, pp, meta["phases"], block_months,
                                period_h, bin_h)["phase"].nunique()}
        reg.report("control", "round -1 snapshot coverage",
                   {"rows": int(len(c0)), "months": int(c0["month"].nunique()),
                    "phases": int(list(ph)[0])},
                   "a full-horizon untrained control needs "
                   "`aligned.control_run` (model required)")
    else:
        reg.report("control", "round -1 snapshot", "unavailable",
                   "no round -1 in latents_by_round")

    reg.honest_n("run", worlds=1, seeds=1, pairs=len(pt))
    return {"reg": reg, "wt": wt, "phi": phi, "phi_pooled": phip, "gm": gm,
            "sym": sym, "pairs": pt, "occupancy_init": occ_i,
            "occupancy_final": occ_m, "partial": mvp, "condition": cond,
            "codebook": cb, "meta": meta}


def check_store(root, run_key: str | None = None, **kw) -> dict:
    """`grid.scan` the directory, then `check_run` one cell (default: the
    first commissioning cell, which is the frozen ruler)."""
    idx = G.scan(root)
    if not len(idx):
        raise FileNotFoundError(f"no saved runs under {root}")
    if run_key is not None:
        row = idx[idx["run_key"] == run_key].iloc[0]
    else:
        comm = idx[idx.get("schedule") == "commissioning"]
        row = (comm if len(comm) else idx).iloc[0]
    out = check_run(row, **kw)
    out["index"] = idx
    return out
