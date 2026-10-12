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

Block-end snapshots (trajectory views)
-------------------------------------
The trainer only ever holds one block's windows, so its own snapshots could
never show the whole horizon under one encoder. Instead, once before block 0
(round -1, the untrained init) and after every block (round = that block's
last cumulative round), the run encodes windows of ALL months on detached
COPIES of the weights -- training is never touched, so a run with these
snapshots trains bit-identically to one without:

* `latents_by_round`: one month-stratified sample of every client's windows
  over the whole horizon (the SAME windows at every snapshot round, so a
  window's path across rounds is readable). `stage="post"` = global weights;
  `stage="pre"` = the client's local weights after its last local update of
  the block (pre-FedAvg). Round -1 has `pre` only, as before. Extra columns:
  `block` and `seen` (month <= the block's last month: in-sample by then).
* `prototypes`: a `block_full` scope -- per-client per-month mean over ALL
  windows, global weights -- at round -1 and at every block end, so a round
  selector reads "the whole-horizon trajectory as the model stood after
  block b". Its own scope, NOT more `post_fedavg_full` rows: `aligned.check`,
  `selfcheck` and `sweep` merge `post_fedavg_full` on (client, month) and
  rely on it holding exactly one round (the final model). The last block's
  `block_full` rows therefore equal `post_fedavg_full`.
* `meta.blocks` records gain `r_lo` / `r_hi` (cumulative round span), which
  is how a reader maps a snapshot round back to its block.
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


def stratified_index(fl_windows: dict, points: int, seed: int) -> dict:
    """Per-client window indices over ALL months, quota-balanced by month.

    The same rule as `SnapshotFPLTrainer._stratified_index`, applied to the
    full `fl_windows` rather than the trainer's slice."""
    rng = np.random.default_rng(seed)
    clients = sorted(fl_windows)
    per_client = max(1, points // max(1, len(clients)))
    out = {}
    for c in clients:
        labels = np.asarray(fl_windows[c]["labels"])
        months = np.unique(labels)
        quota = max(1, per_client // max(1, len(months)))
        picked, deficit = [], 0
        for m in months:
            idx = np.flatnonzero(labels == m)
            take = min(len(idx), quota + deficit)
            deficit = max(0, quota + deficit - len(idx))
            picked.append(rng.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(picked)) if picked else np.array([], int)
        out[c] = sel.astype(int)
    return out


def _weights_copy(model, state: dict | None = None):
    """A detached copy of `model` (same device), optionally loaded with
    `state`. Encoding through the copy leaves the trainer's model, its mode
    and its RNG untouched."""    
    m = copy.deepcopy(model)
    if state is not None:
        m.load_state_dict(state)
    from . import train as T
    T.flatten_rnn_weights(m)
    m.eval()
    return m


def _encode_sample(model, fl_windows: dict, idx: dict, clients,
                   batch: int = 512) -> pd.DataFrame:
    """`idx`-selected windows of `clients` through `model.encode`, in the
    trainer's snapshot schema (district, kind, round, stage, window, month,
    f0..) minus round/stage, which the caller adds."""
    sub = {c: {"windows": np.asarray(fl_windows[c]["windows"])[idx[c]],
               "labels": np.asarray(fl_windows[c]["labels"])[idx[c]],
               "window_start_step":
                   np.asarray(fl_windows[c]["window_start_step"])[idx[c]]}
           for c in clients if len(idx.get(c, ()))}
    return encode_windows(model, sub, batch=batch) if sub else pd.DataFrame()


def _block_end_snapshot(global_model, local_states: dict, fl_windows: dict,
                        idx: dict, round_idx: int, block: int,
                        m_hi: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(snapshot rows, `block_full` prototype rows) for one snapshot round.

    `local_states` empty -> only the `pre` stage from the global weights
    (round -1, where every client model IS the global init)."""
    clients = sorted(fl_windows)
    g = _weights_copy(global_model)
    parts = []
    if local_states:
        parts.append(_encode_sample(g, fl_windows, idx, clients)
                     .assign(stage="post"))
        for c in clients:
            lm = _weights_copy(global_model, local_states.get(c))
            parts.append(_encode_sample(lm, fl_windows, idx, [c])
                         .assign(stage="pre"))
            del lm
    else:
        parts.append(_encode_sample(g, fl_windows, idx, clients)
                     .assign(stage="pre"))
    snap = pd.concat([p for p in parts if len(p)], ignore_index=True)
    fc = [c for c in snap.columns if c.startswith("f") and c[1:].isdigit()]
    snap["round"] = int(round_idx)
    snap["block"] = int(block)
    snap["seen"] = (snap["month"] <= m_hi) if m_hi is not None else False
    snap = snap[["district", "kind", "round", "stage", "window", "month"]
                + fc + ["block", "seen"]]

    lt = encode_windows(g, fl_windows)
    full = (lt.groupby(["district", "month"])[fc].mean().reset_index()
            .rename(columns={"district": "client"})
            .assign(scope="block_full", round=int(round_idx),
                    cluster=np.nan))
    full = full[["scope", "round", "client", "month", "cluster"] + fc]
    del g
    return snap, full


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
    gram, block_full, spans = [], [], {}
    cum, t0 = 0, time.time()
    # one window sample over the WHOLE horizon, reused at every snapshot
    # round, so each sampled window has a path across rounds
    snap_idx = stratified_index(fl_windows, int(spec.snapshot_points), seed)

    for bi, (lo, hi) in enumerate(blocks):
        fw_b = filter_windows(fl_windows, lo, hi)
        if not fw_b:
            continue
        rounds = int(sch.warm_rounds or sch.rounds_per_block) if bi == 0 \
            else int(sch.rounds_per_block)
        fl_b = copy.deepcopy(fl)
        fl_b["training"]["rounds"] = rounds

        # snapshots are taken here, on copies, over every month -- the
        # trainer's own (one block's windows only) are switched off
        tr = T.SnapshotFPLTrainer(fw_b, fl_b, seed + bi, rounds=rounds,
                                  snapshot_rounds=(),
                                  points=spec.snapshot_points, verbose=False,
                                  loss_terms=spec.loss_terms)
        if g_state is not None:                       # carry the model over
            tr.global_model.load_state_dict(g_state)
            for c in tr.clients:
                tr.models[c].load_state_dict(l_state.get(c, g_state))
            tr.global_protos = copy.deepcopy(g_protos)
        T.flatten_rnn_weights(tr.global_model, *tr.models.values())

        if g_state is None:                # round -1: the shared untrained init
            sn, bf = _block_end_snapshot(tr.global_model, {}, fl_windows,
                                         snap_idx, -1, -1, None)
            snaps.append(sn)
            block_full.append(bf)

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

        # ---- 3. block-end snapshot: global + each client's local weights,
        #         every month, on copies (training above is untouched)
        r_end = cum + rounds - 1
        # a client offline for the whole block (participation < 1) keeps
        # the local weights it carried in
        sn, bf = _block_end_snapshot(tr.global_model,
                                     {**l_state, **tr.local_states},
                                     fl_windows, snap_idx, r_end, bi, hi)
        snaps.append(sn)
        block_full.append(bf)
        spans[bi] = (cum, r_end)

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
    prototypes = pd.concat([T.compile_prototypes(ph, gph, snaps, lt),
                            *block_full], ignore_index=True)
    blocks_df = (pd.DataFrame(blocks, columns=["m_lo", "m_hi"])
                 .rename_axis("block").reset_index())
    blocks_df["r_lo"] = [spans.get(b, (np.nan, np.nan))[0]
                         for b in blocks_df["block"]]
    blocks_df["r_hi"] = [spans.get(b, (np.nan, np.nan))[1]
                         for b in blocks_df["block"]]
    # `train.run_cell` records `tr.snap_rounds` as meta.snapshot_rounds; the
    # last trainer had none of its own, so report the rounds taken here
    tr.snap_rounds = set(int(r) for r in snaps["round"].unique()) \
        if snaps is not None else set()
    return {"prototype_history": ph, "global_prototype_history": gph,
            "latent_trajectories": T._tag(lt, world),
            "latent_trajectories_preq": T._tag(lat_preq, world),
            "prequential": preq,
            "training_log": pd.DataFrame(logs),
            "snapshots": None if snaps is None else T._tag(snaps, world),
            "update_gram": pd.DataFrame(gram),
            "prototypes": prototypes,
            "drift_signals": bridge.compute_drift_signals(ph, fl),
            "models": {"global": g_state, **l_state},
            "blocks": blocks_df,
            "trainer": tr, "seconds": round(time.time() - t0, 1),
            "rounds_total": cum}
