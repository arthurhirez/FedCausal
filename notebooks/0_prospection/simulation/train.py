"""Federated training: one `SnapshotFPLTrainer`, one `run_cell`.

`FPLTrainer`'s protocol, seeding and FedAvg are untouched -- this subclass
only adds observation. Three additions over what the two notebooks had, all
mechanical:

1. **Month-stratified snapshots.** Both notebooks sampled the snapshot subset
   with `rng.choice` over all of a client's windows, so the label-clean init
   months got roughly their share of the timeline -- a handful of points in
   the phase the init-state question is asked about. Quotas are per month
   now, so every month contributes.

2. **A `stage` column.** `_local_update` gives the personalized view
   (post-local, pre-FedAvg, which is what `prototype_history` is); the new
   `_fedavg` hook gives the global view (same weights everywhere, different
   data). Both land in one `latents_by_round` table, so `fedavg_gap` is a
   groupby and post-FedAvg prototypes exist for every snapshot round instead
   of only the last.

3. **An update Gram.** Cosine similarity of the client update deltas
   `w_i^r - w_global^(r-1)` per round, computed while the weights are still
   in memory. 5x5 per round is kilobytes, which is why per-round weight
   retention is never needed.

Snapshot rounds default to sparse (`{-1, 0, mid, final}`): the full latent
cloud is the only artifact here that is large, while `prototype_history`
stays every round because that is what the similarity channels consume.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import pandas as pd
import torch

import bridge
from specs import CellSpec
from worlds import World

__all__ = ["resolve_device", "snapshot_rounds_for", "SnapshotFPLTrainer",
           "run_cell", "compile_prototypes", "fedavg_gap",
           "assert_determinism"]


def resolve_device(name: str = "auto") -> str:
    """`FPLTrainer` does `torch.device(cfg['device'])`, which rejects 'auto'."""
    if name in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(name)


def snapshot_rounds_for(rounds: int, mode="sparse") -> set[int]:
    """Which rounds get a full latent snapshot. -1 is the untrained control."""
    if mode in (None, "none", ()):
        return set()
    if mode == "all":
        return {-1, *range(rounds)}
    if mode == "sparse":
        mid = max(0, (rounds - 1) // 2)
        return {-1, 0, mid, rounds - 1}
    return {int(r) for r in mode}


def assert_determinism(noise: bool) -> None:
    """Guard the one known reproducibility hole.

    `sensing.add_measurement_noise` keys its RNG on `hash(sensor)`, and
    Python salts string hashing per process unless `PYTHONHASHSEED` is set.
    Clean series (`noise=False`) never touch it; noisy ones do.
    """
    if noise and os.environ.get("PYTHONHASHSEED") in (None, "", "random"):
        raise AssertionError(
            "noise=True routes through add_measurement_noise, whose RNG is "
            "keyed on hash(sensor): set PYTHONHASHSEED (e.g. 0) before the "
            "interpreter starts, or the run key is not reproducible.")


def flatten_rnn_weights(*models) -> None:
    """Restore each LSTM's flat contiguous weight buffer.

    `FPLTrainer` broadcasts the global init with `copy.deepcopy`, and FedAvg
    reloads state dicts every round; both leave an `nn.LSTM`'s `_flat_weights`
    non-contiguous. On CPU that is invisible, but cuDNN then has to compact
    the weights on every forward and warns about it ("RNN module weights are
    not part of single contiguous chunk of memory"). Harmless numerically,
    wasteful per batch -- so re-flatten after any weight surgery.
    """
    for m in models:
        for mod in m.modules():
            if isinstance(mod, torch.nn.RNNBase):
                mod.flatten_parameters()


def _flat_delta(local: dict, ref: dict) -> np.ndarray:
    """Flattened `w_local - w_ref` over the state dict, in key order."""
    return np.concatenate([(local[k] - ref[k]).detach().cpu()
                           .to(torch.float64).reshape(-1).numpy()
                           for k in sorted(ref)])


def _make_trainer_class():
    """Build the subclass at call time, so importing this module is cheap
    and does not require the fedwater package to be resolvable yet."""
    base = bridge.FPLTrainer()

    class SnapshotFPLTrainer(base):                       # type: ignore[misc]
        """`FPLTrainer` + latent snapshots (pre/post FedAvg) + update Gram."""

        def __init__(self, client_windows, fl, seed, *, rounds: int,
                     snapshot_rounds="sparse", points: int = 1500,
                     verbose: bool = False):
            super().__init__(client_windows, fl, seed)
            flatten_rnn_weights(self.global_model, *self.models.values())
            self.starts = {c: np.asarray(client_windows[c]["window_start_step"])
                           for c in self.clients}
            self.n_rounds = int(rounds)
            self.snap_mode = snapshot_rounds
            self.snap_rounds = snapshot_rounds_for(self.n_rounds, snapshot_rounds)
            self.verbose = bool(verbose)
            self.snap_rows: list[pd.DataFrame] = []
            self.gram_rows: list[dict] = []
            self._round_start_global: dict | None = None

            self.snap_idx = self._stratified_index(int(points), seed)
            if self.snap_rounds:
                for c in self.clients:
                    self._snapshot(c, -1, "pre")          # shared untrained init

        # ------------------------------------------------------- selection
        def _stratified_index(self, points: int, seed: int) -> dict:
            """Per-client window indices, quota-balanced across months.

            A month with few windows contributes all of them; the leftover
            quota is redistributed over the remaining months, so the total
            stays near `points // n_clients` without starving short phases.
            """
            rng = np.random.default_rng(seed)
            per_client = max(1, points // max(1, len(self.clients)))
            out = {}
            for c in self.clients:
                labels = self.data[c][1].cpu().numpy()
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

        # -------------------------------------------------------- snapshots
        @torch.no_grad()
        def _snapshot(self, client: str, round_idx: int, stage: str):
            idx = self.snap_idx[client]
            if not len(idx):
                return
            model, bs = self.models[client], self.cfg_t["batch_size"]
            mode = model.training
            model.eval()
            w = self.data[client][0][idx]
            z = torch.cat([model.encode(w[i:i + bs, 1:-1])
                           for i in range(0, len(w), bs)]).cpu().numpy()
            model.train(mode)

            df = pd.DataFrame(z, columns=[f"f{i}" for i in range(z.shape[1])])
            df.insert(0, "month", self.data[client][1].cpu().numpy()[idx])
            df.insert(0, "window", self.starts[client][idx])
            df.insert(0, "stage", stage)
            df.insert(0, "round", int(round_idx))
            df.insert(0, "kind", "aer_latent")
            df.insert(0, "district", client)
            self.snap_rows.append(df)

        # ------------------------------------------------------------ hooks
        def _local_update(self, client: str, round_idx: int):
            protos = super()._local_update(client, round_idx)
            if round_idx in self.snap_rounds:
                self._snapshot(client, round_idx, "pre")
            return protos

        def _fedavg(self, online: list[str]):
            ref = self._round_start_global
            if ref is not None:
                deltas = {c: _flat_delta(self.models[c].state_dict(), ref)
                          for c in online}
                norms = {c: float(np.linalg.norm(v)) or 1.0
                         for c, v in deltas.items()}
                r = self._current_round
                for i, a in enumerate(online):
                    for b in online[i:]:
                        self.gram_rows.append(dict(
                            round=int(r), client_a=a, client_b=b,
                            cos=float(deltas[a] @ deltas[b]
                                      / (norms[a] * norms[b])),
                            norm_a=norms[a], norm_b=norms[b]))
            super()._fedavg(online)
            flatten_rnn_weights(self.global_model, *self.models.values())
            if self._current_round in self.snap_rounds:
                for c in self.clients:                    # global view
                    self._snapshot(c, self._current_round, "post")

        # ------------------------------------------------------------ train
        def train(self, rounds: int | None = None):
            rounds = self.n_rounds if rounds is None else int(rounds)
            if rounds != self.n_rounds:      # `rounds` decides which are final
                self.n_rounds = rounds
                self.snap_rounds = snapshot_rounds_for(rounds, self.snap_mode)
            n_online = max(1, int(round(self.cfg_t["participation"]
                                        * len(self.clients))))
            for r in range(rounds):
                self._current_round = r
                self._round_start_global = copy.deepcopy(
                    self.global_model.state_dict())
                online = sorted(self.rng.choice(self.clients, size=n_online,
                                                replace=False).tolist())
                local_protos = {c: self._local_update(c, r) for c in online}
                self.global_protos = bridge.aggregate_prototypes(local_protos,
                                                                 self.seed)
                for m, (clusters, _) in self.global_protos.items():
                    for ci, p in enumerate(clusters):
                        self.global_proto_rows.append(
                            dict(round=r, month=m, cluster=ci,
                                 **{f"f{i}": v for i, v in enumerate(p)}))
                self._fedavg(online)
                if self.verbose:
                    loss = np.mean([row["loss"] for row in self.log_rows
                                    if row["round"] == r])
                    print(f"  round {r:>3}  loss {loss:.5f}  "
                          f"online {len(online)}")
            return self

        # ---------------------------------------------------------- exports
        def snapshots(self) -> pd.DataFrame | None:
            return (pd.concat(self.snap_rows, ignore_index=True)
                    if self.snap_rows else None)

        def update_gram(self) -> pd.DataFrame:
            return pd.DataFrame(self.gram_rows)

    return SnapshotFPLTrainer


_TRAINER_CLASS = None


def SnapshotFPLTrainer(*args, **kw):     # noqa: N802 - factory, reads as a class
    global _TRAINER_CLASS
    if _TRAINER_CLASS is None:
        _TRAINER_CLASS = _make_trainer_class()
    return _TRAINER_CLASS(*args, **kw)


# --------------------------------------------------------------------------
# prototype scopes
# --------------------------------------------------------------------------
def compile_prototypes(prototype_history: pd.DataFrame,
                       global_prototype_history: pd.DataFrame,
                       snapshots: pd.DataFrame | None = None,
                       latent_trajectories: pd.DataFrame | None = None
                       ) -> pd.DataFrame:
    """The prototype family in one tidy table.

    | scope         | what                                        | source |
    |---------------|---------------------------------------------|--------|
    | `local`       | per-client per-month mean latent, pre-FedAvg | prototype_history |
    | `global`      | FINCH cluster centroids per month, server    | global_prototype_history |
    | `post_fedavg` | per-client per-month mean latent, global weights | snapshots(stage=post) |

    `post_fedavg` used to exist only at the final round (derived from
    `latent_trajectories`). With post-FedAvg snapshots it exists at every
    snapshot round; the final round is still filled from
    `latent_trajectories` when available, because that uses ALL windows
    rather than the snapshot subset.
    """
    fcols = [c for c in prototype_history.columns
             if c.startswith("f") and c[1:].isdigit()]
    last = int(prototype_history["round"].max())

    local = (prototype_history.assign(scope="local", cluster=np.nan)
             [["scope", "round", "client", "month", "cluster"] + fcols])

    if global_prototype_history is not None and len(global_prototype_history):
        glob = (global_prototype_history.assign(scope="global", client=pd.NA)
                [["scope", "round", "client", "month", "cluster"] + fcols])
    else:
        glob = local.iloc[0:0]

    parts = [local, glob]

    if snapshots is not None and "stage" in snapshots.columns:
        post = snapshots[snapshots["stage"] == "post"]
        if len(post):
            agg = (post.groupby(["round", "district", "month"])[fcols].mean()
                   .reset_index().rename(columns={"district": "client"}))
            parts.append(agg.assign(scope="post_fedavg", cluster=np.nan)
                         [["scope", "round", "client", "month", "cluster"] + fcols])

    if latent_trajectories is not None and len(latent_trajectories):
        lt_f = [c for c in latent_trajectories.columns
                if c.startswith("f") and c[1:].isdigit()]
        agg = (latent_trajectories.groupby(["district", "month"])[lt_f].mean()
               .reset_index().rename(columns={"district": "client"}))
        agg = agg.assign(scope="post_fedavg_full", round=last, cluster=np.nan)
        parts.append(agg[["scope", "round", "client", "month", "cluster"] + lt_f])

    return pd.concat(parts, ignore_index=True)


def fedavg_gap(prototypes: pd.DataFrame, round_idx: int | None = None) -> float:
    """Mean cosine distance between `local` and `post_fedavg` prototypes.

    How much of a client's monthly prototype is personalization rather than
    the shared model -- the asymmetry the protocol creates, quantified.
    """
    p = prototypes[prototypes["scope"].isin(["local", "post_fedavg"])]
    if not len(p):
        return float("nan")
    r = int(p["round"].max()) if round_idx is None else int(round_idx)
    p = p[p["round"] == r]
    fcols = [c for c in p.columns if c.startswith("f") and c[1:].isdigit()]
    piv = p.set_index(["scope", "client", "month"])[fcols]
    if "local" not in piv.index.get_level_values(0) or \
            "post_fedavg" not in piv.index.get_level_values(0):
        return float("nan")
    a = piv.loc["local"].sort_index()
    b = piv.loc["post_fedavg"].sort_index()
    common = a.index.intersection(b.index)
    A, B = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return float(np.mean(1 - (A * B).sum(1) / np.where(den > 0, den, np.nan)))


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------
def run_cell(world: World, spec: CellSpec, fl_windows=None, scalers=None,
             fl=None, cache=None, verbose: bool = False) -> dict:
    """Train one cell. Pure compute -- persistence lives in `store`.

    Returns the five `train_federated` artifacts plus `snapshots`,
    `update_gram`, `prototypes`, the resolved `fl`, and timings. Windows can
    be passed in when they were already built (the widget path); otherwise
    they are built here.
    """
    import data as D

    assert_determinism(spec.noise)
    if fl_windows is None:
        fl_windows, scalers, fl = D.build_windows(world, spec, cache=cache)

    fl = copy.deepcopy(fl)
    fl["training"]["device"] = resolve_device(fl["training"].get("device"))
    rounds = int(fl["training"]["rounds"])
    seed = bridge.effective_fl_seed(fl, int(world.params.get("seed", 42)))

    t0 = time.time()
    tr = SnapshotFPLTrainer(fl_windows, fl, seed, rounds=rounds,
                            snapshot_rounds=spec.snapshot_rounds,
                            points=spec.snapshot_points, verbose=verbose)
    tr.train(rounds)
    seconds = round(time.time() - t0, 1)

    ph = pd.DataFrame(tr.local_proto_rows)
    gph = pd.DataFrame(tr.global_proto_rows)
    log = pd.DataFrame(tr.log_rows)
    lt = tr.latent_trajectories(fl_windows)
    snaps = tr.snapshots()

    for name, df in (("prototype_history", ph), ("latent_trajectories", lt)):
        fc = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
        if not np.isfinite(df[fc].to_numpy()).all():
            raise AssertionError(f"non-finite values in {name}")

    lt = _tag(lt, world)
    snaps = None if snaps is None else _tag(snaps, world)

    return {"prototype_history": ph, "global_prototype_history": gph,
            "latent_trajectories": lt, "training_log": log,
            "snapshots": snaps, "update_gram": tr.update_gram(),
            "prototypes": compile_prototypes(ph, gph, snaps, lt),
            "drift_signals": bridge.compute_drift_signals(ph, fl),
            "models": {"global": tr.global_model.state_dict(),
                       **{c: tr.models[c].state_dict() for c in tr.clients}},
            "meta": {"lstm_units": fl["model"]["lstm_units"],
                     # The spec keeps the literal "auto" so run keys stay
                     # portable across machines; the RESOLVED device is
                     # recorded here, because CPU and GPU do not agree to the
                     # last digit and a cache hit should say which produced it.
                     "device": fl["training"]["device"],
                     "torch": torch.__version__,
                     "window_size": tr.global_model.window_size,
                     "n_features": tr.global_model.head.out_features,
                     "latent_dim": tr.global_model.latent_dim,
                     "sensors": fl_windows[next(iter(fl_windows))]["sensors"],
                     "clients": list(tr.clients), "seed": seed,
                     "rounds": rounds,
                     "snapshot_rounds": sorted(tr.snap_rounds)},
            "fl": fl, "fl_windows": fl_windows, "scalers": scalers,
            "trainer": tr, "seconds": seconds}


def _tag(df: pd.DataFrame, world: World) -> pd.DataFrame:
    """Add the calendar/phase/drift columns every figure and score reads."""
    res_h = float(world.time["resolution_h"])
    out = df.copy()
    out["hour"] = (out["window"] * res_h % 24).astype(int)
    out["phase"] = [world.phase_of(int(m)) for m in out["month"]]
    tgt = world.drift_district
    first = world.first_switch
    out["drift_status"] = np.where(
        (out["district"] == tgt) & (out["month"] >= (first if first is not None
                                                    else math.inf)),
        "drifted", "other")
    return out
