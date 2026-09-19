"""Federated training: one `SnapshotFPLTrainer`, one `run_cell`.

`FPLTrainer`'s protocol, seeding and FedAvg are untouched -- this subclass
only adds observation. Three additions over what the two notebooks had, all
mechanical:

0. **An optional lab-side local objective** (`spec.loss_terms`, see
   `specs.LossTerms`). With it, `_local_update` is RE-IMPLEMENTED here
   (`_local_update_lab`): same seeding, loader, epochs, prototype extraction
   and log schema as upstream, plus the extra terms. This is the one
   deliberate exception to "nothing upstream is re-implemented", and
   `LossTerms()` must reproduce the upstream loop bit for bit
   (`equivalence_check`). Without `loss_terms` the upstream loop runs
   untouched; the upstream knob `fl.training.proto_weight` then applies
   (needs the patched `FPLTrainer`; refused otherwise).

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

from . import bridge
from .specs import CellSpec
from .worlds import World

__all__ = ["resolve_device", "snapshot_rounds_for", "SnapshotFPLTrainer",
           "run_cell", "compile_prototypes", "fedavg_gap",
           "assert_determinism", "reconstruction_model", "decode",
           "reconstruct_prototypes", "untrained_model", "equivalence_check",
           "fl_mismatches", "assert_fl_matches", "local_model"]


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
                     verbose: bool = False, loss_terms=None):
            super().__init__(client_windows, fl, seed)
            pw = (fl.get("training") or {}).get("proto_weight")
            if pw is not None and loss_terms is not None:
                raise ValueError("set the prototype weight in ONE place: "
                                 "training.proto_weight (upstream) or "
                                 "loss_terms.proto_weight (lab), not both")
            if pw is not None and not hasattr(self, "proto_weight"):
                raise RuntimeError(
                    "fl.training.proto_weight is set but the installed "
                    "FPLTrainer does not read it -- apply the fl_training/"
                    "federated.py patch, or use spec.loss_terms instead")
            self.loss_terms = loss_terms
            # each client's weights right after its latest local update,
            # BEFORE FedAvg overwrites them (the model its shared prototypes
            # came from)
            self.local_states: dict = {}
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
        def _local_update_lab(self, client: str, round_idx: int):
            """`FPLTrainer._local_update` with `self.loss_terms` applied.
            Kept line-for-line parallel to upstream; see the module note."""
            from torch.utils.data import DataLoader, TensorDataset
            lt, cfg = self.loss_terms, self.cfg_t
            model, opt = self.models[client], self.optimizers[client]
            windows, labels = self.data[client]
            gen = torch.Generator().manual_seed(
                int(np.random.default_rng([self.seed, round_idx,
                                           self.clients.index(client)])
                    .integers(2**31)))
            loader = DataLoader(TensorDataset(windows, labels),
                                batch_size=cfg["batch_size"],
                                shuffle=True, generator=gen)
            use_proto = (bool(self.global_protos) and lt.proto_weight != 0
                         and round_idx >= lt.proto_start)
            zero = torch.zeros((), device=self.device)
            mse = torch.nn.functional.mse_loss
            model.train()
            for epoch in range(cfg["local_epochs"]):
                sums = dict.fromkeys(("loss", "loss_mse", "loss_proto",
                                      "loss_diff", "loss_spec", "loss_var"),
                                     0.0)
                for wb, lb in loader:
                    opt.zero_grad()
                    x, ry_t, y_t, fy_t = bridge.split_window_targets(wb)
                    ry, y, fy, z = model(x)
                    loss_mse = bridge.aer_loss(ry, y, fy, ry_t, y_t, fy_t,
                                               self.cfg_m["reg_ratio"])
                    loss_proto = (bridge.hierarchical_proto_loss(
                        z, lb, self.global_protos, cfg["proto_alpha"],
                        cfg["infonce_temperature"], self.device)
                        if use_proto else zero)
                    loss = loss_mse + lt.proto_weight * loss_proto
                    l_diff = l_spec = l_var = zero
                    if lt.diff:
                        dec = torch.cat([ry.unsqueeze(1), y, fy.unsqueeze(1)], 1)
                        l_diff = mse(torch.diff(dec, dim=1),
                                     torch.diff(wb, dim=1))
                        loss = loss + lt.diff * l_diff
                    if lt.spec:
                        def mag(a):
                            a = a - a.mean(dim=1, keepdim=True)
                            return torch.fft.rfft(a, dim=1, norm="ortho").abs()
                        l_spec = mse(mag(y), mag(y_t))
                        loss = loss + lt.spec * l_spec
                    if lt.var:
                        sd = torch.sqrt(z.var(dim=0) + 1e-4)
                        l_var = torch.relu(lt.var_gamma - sd).mean()
                        loss = loss + lt.var * l_var
                    loss.backward()
                    opt.step()
                    sums["loss"] += loss.item()
                    sums["loss_mse"] += loss_mse.item()
                    sums["loss_proto"] += float(loss_proto.detach())
                    sums["loss_diff"] += float(l_diff.detach())
                    sums["loss_spec"] += float(l_spec.detach())
                    sums["loss_var"] += float(l_var.detach())
                n = len(loader)
                self.log_rows.append(dict(round=round_idx, client=client,
                                          epoch=epoch,
                                          **{k: v / n for k, v in sums.items()}))

            protos = bridge.extract_prototypes(model, windows,
                                               labels.cpu().numpy(),
                                               cfg["batch_size"])
            for m, p in protos.items():
                if not np.isfinite(p).all():
                    raise AssertionError(f"Non-finite prototype: {client} month {m}")
                self.local_proto_rows.append(
                    dict(round=round_idx, client=client, month=m,
                         **{f"f{i}": v for i, v in enumerate(p)}))
            return protos

        def _local_update(self, client: str, round_idx: int):
            if self.loss_terms is None:
                protos = super()._local_update(client, round_idx)
            else:
                protos = self._local_update_lab(client, round_idx)
            self.local_states[client] = {
                k: v.detach().cpu().clone()
                for k, v in self.models[client].state_dict().items()}
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
_CONTRACT_TRAINING = ("rounds", "local_epochs", "learning_rate", "batch_size",
                      "participation", "averaging", "proto_alpha",
                      "infonce_temperature", "seed", "proto_weight")
_CONTRACT_MODEL = ("lstm_units", "reg_ratio")


def fl_mismatches(spec: CellSpec, fl: dict) -> dict:
    """{key: (spec value, fl value)} for every training / model / geometry
    field where a resolved `fl` disagrees with the spec it should come from."""
    want_t, want_m = spec.training.as_fl(), spec.model.as_fl()
    got_t, got_m = fl.get("training") or {}, fl.get("model") or {}
    out = {}
    for k in _CONTRACT_TRAINING:
        a, b = want_t.get(k), got_t.get(k)
        if a != b and not (a is None and b is None):
            out[f"training.{k}"] = (a, b)
    for k in _CONTRACT_MODEL:
        if want_m.get(k) != got_m.get(k):
            out[f"model.{k}"] = (want_m.get(k), got_m.get(k))
    g, pre = spec.geometry, fl.get("preprocessing") or {}
    for k in ("interval_agg_h", "window_size", "step_size"):
        if int(getattr(g, k)) != int(pre.get(k, -1)):
            out[f"preprocessing.{k}"] = (getattr(g, k), pre.get(k))
    return out


def assert_fl_matches(spec: CellSpec, fl: dict) -> None:
    bad = fl_mismatches(spec, fl)
    if bad:
        raise AssertionError(f"resolved fl disagrees with the spec: {bad}")


def _run_meta(fl, tr, fl_windows, spec, seed, rounds, snapshot_rounds):
    """The `meta` block of a run: enough to rebuild the model and to say
    what produced it. Shared by every schedule."""
    return {"lstm_units": fl["model"]["lstm_units"],
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
            "snapshot_rounds": snapshot_rounds,
            "client_weights": "local_pre_fedavg",
            # windows the trainer actually saw (a commissioning or streaming
            # schedule trains on a slice); `tr.data` is {client: (X, labels)}
            "train_windows": (float(np.mean([len(v[1]) for v in tr.data.values()]))
                              if getattr(tr, "data", None) else None),
            "orient": bool(spec.orient),
            "schedule": (None if spec.schedule is None
                         else spec.schedule.as_dict()),
            "proto_weight": getattr(tr, "proto_weight", None),
            "loss_terms": (None if spec.loss_terms is None
                           else spec.loss_terms.as_dict())}


def run_cell(world: World, spec: CellSpec, fl_windows=None, scalers=None,
             fl=None, cache=None, verbose: bool = False) -> dict:
    """Train one cell. Pure compute -- persistence lives in `store`.

    Returns the five `train_federated` artifacts plus `snapshots`,
    `update_gram`, `prototypes`, the resolved `fl`, and timings. Windows can
    be passed in when they were already built (the widget path); otherwise
    they are built here.
    """
    from . import data as D

    assert_determinism(spec.noise)
    if fl_windows is None:
        fl_windows, scalers, fl = D.build_windows(world, spec, cache=cache)

    fl = copy.deepcopy(fl)
    assert_fl_matches(spec, fl)
    fl["training"]["device"] = resolve_device(fl["training"].get("device"))
    rounds = int(fl["training"]["rounds"])
    seed = bridge.effective_fl_seed(fl, int(world.params.get("seed", 42)))

    sch = spec.schedule
    if sch is not None and sch.mode == "streaming":
        from . import streaming as ST
        out = ST.run_stream(world, spec, fl_windows, fl, seed, verbose=verbose)
        tr = out.pop("trainer")
        out["meta"] = _run_meta(fl, tr, fl_windows, spec, seed,
                               out.pop("rounds_total"),
                               sorted(getattr(tr, "snap_rounds", ()) or ()))
        out["meta"]["blocks"] = out["blocks"].to_dict("records")
        out["fl"], out["seconds"] = fl, out.get("seconds")
        return out

    train_windows = fl_windows
    if sch is not None and sch.mode == "commissioning":
        from . import streaming as ST
        train_windows = ST.filter_windows(fl_windows, *sch.months)
        missing = set(fl_windows) - set(train_windows)
        if missing:
            raise AssertionError(f"no windows in months {sch.months} for "
                                 f"{sorted(missing)}: FedAvg needs every client")
        if verbose:
            n = {c: len(d["labels"]) for c, d in train_windows.items()}
            print(f"commissioning on months {sch.months[0]}..{sch.months[1]}: "
                  f"{n} windows per client")

    t0 = time.time()
    tr = SnapshotFPLTrainer(train_windows, fl, seed, rounds=rounds,
                            snapshot_rounds=spec.snapshot_rounds,
                            points=spec.snapshot_points, verbose=verbose,
                            loss_terms=spec.loss_terms)
    tr.train(rounds)
    seconds = round(time.time() - t0, 1)

    ph = pd.DataFrame(tr.local_proto_rows)
    gph = pd.DataFrame(tr.global_proto_rows)
    log = pd.DataFrame(tr.log_rows)
    # ALWAYS every window, even when training saw only some of them: the
    # analysis encodes the whole horizon through the trained model.
    # `FPLTrainer.latent_trajectories` can only encode the windows it was
    # built with, so a commissioning run goes through `encode_windows`
    # (same schema, global weights -- after FedAvg every client model is
    # the global one).
    if train_windows is fl_windows:
        lt = tr.latent_trajectories(fl_windows)
    else:
        from .aligned import encode_windows
        lt = encode_windows(tr.global_model, fl_windows)
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
            # client entries are LOCAL (pre-final-FedAvg) weights; after
            # FedAvg `tr.models[c]` would just be the global model again
            "models": {"global": tr.global_model.state_dict(),
                       **{c: tr.local_states[c] for c in tr.clients
                          if c in tr.local_states}},
            "meta": _run_meta(fl, tr, fl_windows, spec, seed, rounds,
                              sorted(tr.snap_rounds)),
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


# --------------------------------------------------------------------------
# prototype reconstruction -- decoding a prototype back to window space
# --------------------------------------------------------------------------
# FedAvg syncs EVERY client's weights to the aggregated global model at the
# end of every round (`fl_training.federated.FPLTrainer._fedavg`:
# `for c in self.clients: self.models[c].load_state_dict(...global_w...)`),
# so after training, every client's model and the global model are
# bit-identical -- this is exactly what `check_run`'s
# `"clients_synced_after_fedavg"` row asserts. There is therefore no
# separate "local decoder" to load: `local` and `post_fedavg` prototypes
# differ only as LATENT VECTORS (the local one is the mean latent measured
# mid-round, under that round's just-updated personal weights, BEFORE the
# sync that overwrites them; the post-FedAvg one is measured after). What
# follows decodes both through the ONE decoder every client ends up
# sharing, which is the only decoding that is actually well-defined here.
@torch.no_grad()
def reconstruction_model(res: dict, world: World, store=None):
    """The trained GLOBAL model to decode prototypes through
    (`res["models"]["global"]`). Client entries are local weights (see
    `local_model`).

    Present in `res` after a FRESH `run_cell` (not a cache hit, which does
    not reload weights); on a cache hit, pass `store=` to load them from
    `fed_model.pt` instead (requires the run to have been saved with
    `spec.keep_weights != "none"`, the default).
    """
    m = res["meta"]
    if res.get("models"):
        mdl = bridge.aer_class()(m["n_features"], m["window_size"],
                                 m["lstm_units"])
        mdl.load_state_dict(res["models"]["global"])
        mdl.eval()
        return mdl
    if store is not None:
        key = res.get("run_key") or m.get("run_key")
        if not key:
            raise ValueError("res carries no run_key to load weights for")
        return store.load_model(world.sim_hash, key, client="global")
    raise ValueError(
        "no weights in `res` (this looks like a cache hit) and no `store` "
        "given to load `fed_model.pt` from -- pass store=, or call "
        "run_cell(..., overwrite=True) for weights in memory")


@torch.no_grad()
def local_model(res: dict, world: World, client: str, store=None):
    """`client`'s LOCAL model: its weights after the last local update,
    before the final FedAvg. Needs a run saved with
    `meta.client_weights == "local_pre_fedavg"`."""
    m = res["meta"]
    if m.get("client_weights") != "local_pre_fedavg":
        raise ValueError(f"run {res.get('run_key')} has no local weights -- "
                         f"retrain it with run_cell(..., overwrite=True)")
    if res.get("models") and client in res["models"]:
        mdl = bridge.aer_class()(m["n_features"], m["window_size"],
                                 m["lstm_units"])
        mdl.load_state_dict(res["models"][client])
        mdl.eval()
        return mdl
    if store is not None:
        return store.load_model(world.sim_hash, res["run_key"], client=client)
    raise ValueError("no local weights in `res` and no `store` to load them")


def model_device(model) -> "torch.device":
    """The device a model's parameters live on. Inputs must be built there:
    during a run the trainer's model is on the GPU, while a model rebuilt
    from a saved state dict is on the CPU, and mixing the two raises
    "Input and parameter tensors are not at the same device"."""
    return next(model.parameters()).device


@torch.no_grad()
def decode(model, z) -> np.ndarray:
    """(B, latent_dim) latent vectors -> (B, window_size, n_features)
    reconstructions -- the decoder half of `AER.forward`, standalone.

    A prototype is a MEAN of encoder outputs, never itself the output of
    `encode`, so there is no `x` to run the full `forward` on; this is
    `forward`'s tail (`z -> repeated -> decoder -> head`) applied directly,
    byte-for-byte the same operations. Row 0 of the window axis is `ry` (the
    step the model predicts BEFORE what it encoded), rows `1..-2` are `y`
    (the reconstruction proper, same span the encoder consumed), row -1 is
    `fy` (the step predicted AFTER) -- see `aer.AER.forward`.
    """
    model.eval()
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32,
                        device=model_device(model))
    repeated = z.unsqueeze(1).repeat(1, model.window_size, 1)
    seq, _ = model.decoder(repeated)
    return model.head(seq).cpu().numpy()          # (B, window_size, F)


def reconstruct_prototypes(model, protos: pd.DataFrame) -> pd.DataFrame:
    """Every row of `protos` (e.g. `res["prototypes"]`, filtered to the
    scopes/round/clients/months of interest), decoded.

    -> long-form: whichever of `scope, round, client, month, cluster` are
    present in `protos`, plus `step` (0..window_size-1) and `channel` (the
    model's own feature index -- see `res["meta"]["sensors"]`, or
    `fl_windows[client]["sensors"]`, for the channel -> sensor id order) and
    `value` (the decoded, SCALED reconstruction -- the model's own output
    units, i.e. whatever `spec.scaling` the windows were built with).
    `recon.as_arrays` / `recon.finch_arrays` turn this into the per
    (client, month) arrays the comparisons use.
    """
    fcols = [c for c in protos.columns if c.startswith("f") and c[1:].isdigit()]
    if not len(protos):
        return pd.DataFrame(columns=["step", "channel", "value"])
    dec = decode(model, protos[fcols].to_numpy(dtype=np.float32))
    n, w, f = dec.shape
    meta_cols = [c for c in ("scope", "round", "client", "month", "cluster")
                if c in protos.columns]
    idx = np.repeat(np.arange(n), w * f)
    out = protos.iloc[idx][meta_cols].reset_index(drop=True)
    out["step"] = np.tile(np.repeat(np.arange(w), f), n)
    out["channel"] = np.tile(np.arange(f), n * w)
    out["value"] = dec.reshape(-1)
    return out


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
@torch.no_grad()
def untrained_model(res: dict):
    """The round -1 model: `FPLTrainer`'s global init, rebuilt.

    `FPLTrainer.__init__` calls `torch.manual_seed(seed)` and builds the AER
    with no other RNG draw in between (tensor copies draw nothing), so a
    fresh AER after the same seed is that init exactly -- `aligned.
    control_check` compares it with the saved round -1 snapshot."""
    m = res["meta"]
    torch.manual_seed(int(m["seed"]))
    mdl = bridge.aer_class()(m["n_features"], m["window_size"], m["lstm_units"])
    mdl.eval()
    return mdl


def equivalence_check(fl_windows: dict, fl: dict, seed: int = 0,
                      rounds: int = 2) -> pd.DataFrame:
    """Does the lab loop reproduce the upstream loop? Trains both for a few
    rounds on the same windows (CPU) and compares weights and prototypes.

    Three pairs: upstream vs `LossTerms()`; and, when the installed
    `FPLTrainer` reads `proto_weight`, upstream `proto_weight=0` vs
    `LossTerms(proto_weight=0)`."""
    from .specs import LossTerms

    def train(fl_, lt):
        f = copy.deepcopy(fl_)
        f["training"]["device"] = "cpu"
        tr = SnapshotFPLTrainer(fl_windows, f, seed, rounds=rounds,
                                snapshot_rounds="none", loss_terms=lt)
        tr.train(rounds)
        return tr

    def compare(name, a, b):
        wa, wb = a.global_model.state_dict(), b.global_model.state_dict()
        dw = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
        pa = pd.DataFrame(a.local_proto_rows)
        pb = pd.DataFrame(b.local_proto_rows)
        fc = [c for c in pa.columns if c.startswith("f") and c[1:].isdigit()]
        dp = float(np.abs(pa[fc].to_numpy() - pb[fc].to_numpy()).max())
        return {"pair": name, "max_abs_weight_diff": dw,
                "max_abs_proto_diff": dp, "identical": dw == 0 and dp == 0}

    base = copy.deepcopy(fl)
    base["training"].pop("proto_weight", None)
    rows = [compare("upstream vs LossTerms()",
                    train(base, None), train(base, LossTerms()))]
    up0 = copy.deepcopy(base)
    up0["training"]["proto_weight"] = 0.0
    try:
        a = train(up0, None)
    except RuntimeError as exc:
        rows.append({"pair": "upstream pw=0 vs LossTerms(proto_weight=0)",
                     "max_abs_weight_diff": np.nan,
                     "max_abs_proto_diff": np.nan, "identical": False,
                     "note": str(exc)})
    else:
        rows.append(compare("upstream pw=0 vs LossTerms(proto_weight=0)",
                            a, train(base, LossTerms(proto_weight=0.0))))
    return pd.DataFrame(rows)
