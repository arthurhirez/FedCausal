"""Chronological schedules: commissioning and streaming.

`train.run_cell` trains on every month at once and every round, so a model
describing month 20 has already seen month 60. These two schedules do not.

Commissioning (`Schedule(mode="commissioning", months=(lo, hi))`)
----------------------------------------------------------------
Federated training on those months only, then the model is frozen. It is
handled inside `run_cell` by filtering the windows before the trainer sees
them; only the ENCODING afterwards covers the whole horizon. The frozen
model is a fixed ruler: reconstruction error and latent displacement from
its own prototypes are then measurements against a state of the world it
was fitted to, which is what a commissioning period gives you in practice.

Streaming (`Schedule(mode="streaming", block_months=B, rounds_per_block=R)`)
---------------------------------------------------------------------------
Prequential ("test-then-train"): the horizon is cut into blocks of B months.
For block b, in order:

1. **score first** -- every client's windows of block b are reconstructed
   with the CURRENT global model and with that client's current local model.
   Block b is unseen at that moment, so these rows (`prequential`) are the
   honest ones, and their latents are kept as `latent_trajectories_preq`.
2. **then train** -- R rounds of the ordinary FL loop on block b's windows
   only, carrying forward: the global weights, each client's local weights,
   and the global prototypes of the previous blocks (so FPL's contrastive
   term has a memory). Adam moments restart with each block, because the
   upstream trainer builds its optimisers with its data.

What carries and what does not is a real design choice, not plumbing:
prototypes accumulate across blocks (a month key appears once, in its own
block), and the contrastive term of block b therefore pulls towards
prototypes computed on earlier regimes. That is intended: it is what makes a
regime change visible as a pull AWAY from them.

Rounds are numbered cumulatively across blocks, so `training_log`,
`prototype_history` and the snapshots keep the schema every other module
expects; a `block` column is added next to it.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from . import bridge
from .aligned import encode_windows
from .specs import CellSpec
from .worlds import World

__all__ = ["month_blocks", "filter_windows", "run_stream", "step_plan",
           "rounds_for_steps"]


def _batches(n_windows: int, batch_size: int) -> int:
    return int(np.ceil(n_windows / max(batch_size, 1)))


def step_plan(spec: CellSpec, fl_windows: dict, fl: dict) -> dict:
    """How many optimiser steps per client this spec will actually run.

    Schedules train on slices, so equal `rounds` is NOT an equal budget:
    commissioning sees only its months, streaming sees one block at a time.
    -> {mode, windows_per_pass, batches, rounds, steps}."""
    tr = fl["training"]
    bs, ep = int(tr["batch_size"]), int(tr["local_epochs"])
    n_all = float(np.mean([len(d["labels"]) for d in fl_windows.values()]))
    sch = spec.schedule
    if sch is None:
        n, rounds = n_all, int(tr["rounds"])
        steps = rounds * ep * _batches(n, bs)
        return {"mode": "all", "windows_per_pass": n, "batches": _batches(n, bs),
                "rounds": rounds, "steps": steps}
    if sch.mode == "commissioning":
        sub = filter_windows(fl_windows, *sch.months)
        n = float(np.mean([len(d["labels"]) for d in sub.values()]))
        rounds = int(tr["rounds"])
        return {"mode": sch.label(), "windows_per_pass": n,
                "batches": _batches(n, bs), "rounds": rounds,
                "steps": rounds * ep * _batches(n, bs)}
    months = sorted({int(m) for d in fl_windows.values() for m in d["labels"]})
    blocks = month_blocks(months, sch.block_months)
    steps, ns = 0, []
    for bi, (lo, hi) in enumerate(blocks):
        sub = filter_windows(fl_windows, lo, hi)
        if not sub:
            continue
        n = float(np.mean([len(d["labels"]) for d in sub.values()]))
        r = int(sch.warm_rounds or sch.rounds_per_block) if bi == 0 \
            else int(sch.rounds_per_block)
        steps += r * ep * _batches(n, bs)
        ns.append(n)
    return {"mode": sch.label(), "windows_per_pass": float(np.mean(ns)),
            "batches": _batches(float(np.mean(ns)), bs),
            "rounds": len(ns) * int(sch.rounds_per_block), "steps": steps}


def rounds_for_steps(spec: CellSpec, fl_windows: dict, fl: dict,
                     target_steps: int) -> int:
    """Rounds (per block, for streaming) that come closest to
    `target_steps` per client with this spec's windows and epochs."""
    plan = step_plan(spec, fl_windows, fl)
    per_round = plan["steps"] / max(plan["rounds"], 1)
    return max(1, int(round(target_steps / max(per_round, 1e-9))))


def month_blocks(months, block_months: int) -> list[tuple[int, int]]:
    """[(lo, hi)] inclusive, covering `months` in chronological blocks."""
    lo, hi = int(min(months)), int(max(months))
    return [(a, min(a + block_months - 1, hi))
            for a in range(lo, hi + 1, block_months)]


def filter_windows(fl_windows: dict, lo: int, hi: int) -> dict:
    """The windows whose month label is in [lo, hi], client by client."""
    out = {}
    for c, d in fl_windows.items():
        lab = np.asarray(d["labels"])
        m = (lab >= lo) & (lab <= hi)
        if not m.any():
            continue
        out[c] = {**d, "windows": np.asarray(d["windows"])[m],
                  "labels": lab[m],
                  "window_start_step": np.asarray(d["window_start_step"])[m]}
    return out


def _scores(model, X, reg_ratio: float) -> dict:
    from .localeval import _aer, _med_amp_r, reconstruct
    D = reconstruct(model, X)
    wmean = np.broadcast_to(X[:, 1:-1].mean(axis=1, keepdims=True), X.shape)
    loss = _aer(D, X, reg_ratio)
    base = _aer(wmean, X, reg_ratio)
    amp, r = _med_amp_r(D, X)
    return {"loss": loss, "ratio": loss / base, "amp": amp, "r": r}


def run_stream(world: World, spec: CellSpec, fl_windows: dict, fl: dict,
               seed: int, verbose: bool = False) -> dict:
    """The streaming schedule. Returns what `run_cell` returns, plus
    `prequential` and `latent_trajectories_preq`."""
    from . import train as T

    sch = spec.schedule
    months = sorted({int(m) for d in fl_windows.values() for m in d["labels"]})
    blocks = month_blocks(months, sch.block_months)
    reg = float(fl["model"]["reg_ratio"])

    g_state, l_state, g_protos = None, {}, {}
    ph, gph, logs, preq, snaps, lat_preq = [], [], [], [], [], []
    gram = []
    cum, t0 = 0, time.time()

    for bi, (lo, hi) in enumerate(blocks):
        fw_b = filter_windows(fl_windows, lo, hi)
        if not fw_b:
            continue
        rounds = int(sch.warm_rounds or sch.rounds_per_block) if bi == 0 \
            else int(sch.rounds_per_block)
        fl_b = copy.deepcopy(fl)
        fl_b["training"]["rounds"] = rounds

        tr = T.SnapshotFPLTrainer(fw_b, fl_b, seed + bi, rounds=rounds,
                                  snapshot_rounds=(-1,) if bi == 0 else (),
                                  points=spec.snapshot_points, verbose=False,
                                  loss_terms=spec.loss_terms)
        if g_state is not None:                       # carry the model over
            tr.global_model.load_state_dict(g_state)
            for c in tr.clients:
                tr.models[c].load_state_dict(l_state.get(c, g_state))
            tr.global_protos = copy.deepcopy(g_protos)
        T.flatten_rnn_weights(tr.global_model, *tr.models.values())

        # ---- 1. score the unseen block, with the model as it stands now
        for c in tr.clients:
            X = np.asarray(fw_b[c]["windows"], float)
            row = {"block": bi, "m_lo": lo, "m_hi": hi, "client": c,
                   "n": len(X), "round_before": cum, "trained_on": False}
            preq.append({**row, "model": "global",
                         **_scores(tr.global_model, X, reg)})
            preq.append({**row, "model": "local",
                         **_scores(tr.models[c], X, reg)})
        lat_preq.append(encode_windows(tr.global_model, fw_b).assign(block=bi))

        # ---- 2. train on it
        tr.train(rounds)
        cum_map = {r: cum + r for r in range(rounds)}
        for rows, sink in ((tr.local_proto_rows, ph),
                           (tr.global_proto_rows, gph), (tr.log_rows, logs)):
            for r in rows:
                sink.append({**r, "block": bi,
                             "round": cum_map.get(int(r["round"]),
                                                  cum + int(r["round"]))})
        for r in tr.gram_rows:
            gram.append({**r, "block": bi, "round": cum + int(r["round"])})
        sn = tr.snapshots()
        if sn is not None and len(sn):
            sn = sn.copy()
            sn["round"] = [cum + int(r) if r >= 0 else -1 for r in sn["round"]]
            snaps.append(sn.assign(block=bi))

        g_state = {k: v.detach().cpu().clone()
                   for k, v in tr.global_model.state_dict().items()}
        l_state = {c: {k: v.clone() for k, v in st.items()}
                   for c, st in tr.local_states.items()}
        g_protos = copy.deepcopy(tr.global_protos)
        cum += rounds
        if verbose:
            p = [x for x in preq if x["block"] == bi and x["model"] == "global"]
            print(f"  block {bi} m{lo}-{hi}: {sum(x['n'] for x in p)} windows, "
                  f"{rounds} rounds, unseen r={np.mean([x['r'] for x in p]):.3f}, "
                  f"{time.time() - t0:.0f}s")

    # ---- final model over every window, as `run_cell` does
    tr.global_model.load_state_dict(g_state)
    lt = encode_windows(tr.global_model, fl_windows)
    preq = pd.DataFrame(preq)
    lat_preq = pd.concat(lat_preq, ignore_index=True)
    ph, gph = pd.DataFrame(ph), pd.DataFrame(gph)
    snaps = pd.concat(snaps, ignore_index=True) if snaps else None
    return {"prototype_history": ph, "global_prototype_history": gph,
            "latent_trajectories": T._tag(lt, world),
            "latent_trajectories_preq": T._tag(lat_preq, world),
            "prequential": preq,
            "training_log": pd.DataFrame(logs),
            "snapshots": None if snaps is None else T._tag(snaps, world),
            "update_gram": pd.DataFrame(gram),
            "prototypes": T.compile_prototypes(ph, gph, snaps, lt),
            "drift_signals": bridge.compute_drift_signals(ph, fl),
            "models": {"global": g_state, **l_state},
            "blocks": pd.DataFrame(blocks, columns=["m_lo", "m_hi"])
                        .rename_axis("block").reset_index(),
            "trainer": tr, "seconds": round(time.time() - t0, 1),
            "rounds_total": cum}
