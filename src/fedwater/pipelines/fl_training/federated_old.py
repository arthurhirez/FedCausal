"""In-process federated FPL training over AER autoencoders.

Protocol (faithful to BEPE / Huang et al. CVPR'23 "Rethinking FL with
Domain Shift: A Prototype View", adapted to time series):

Per communication round:
1. Each online client trains locally: AER triple-MSE plus — once global
   prototypes exist — the hierarchical prototype loss
   ``alpha * InfoNCE(z, cluster protos) + (1 - alpha) * MSE(z, mean proto)``.
2. Each client then extracts per-month prototypes as the MEAN latent per
   month label — computed in a clean **eval-mode pass with the final local
   weights** (correctness fix: the legacy code averaged latents collected
   mid-training across batches of the last epoch, i.e. under changing
   weights, taken from the wrong tensor — see ``aer.py``).
3. The server groups local prototypes by month, FINCH-clusters each group
   (cosine) into unbiased cluster prototypes, and FedAvg-aggregates weights.

Implementation notes
--------------------
* The InfoNCE is vectorized over the batch (the legacy per-instance Python
  loop also carried three bugs: division by batch size instead of matched
  count, a ``None`` crash when no batch label had a global prototype yet,
  and an ``.item()`` crash when exactly one instance matched).
* Everything is seeded: model init, batch shuffling, FINCH input order.
* All histories are emitted as tidy frames; the final-model latent
  trajectory table shares the ``feature_trajectories`` schema
  (district, kind, window, f0..fD), so the dependence battery's Tier-4
  machinery plugs into AER latent space with zero adaptation.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .aer import AER, aer_loss, split_window_targets
from .finch import FINCH


# --------------------------------------------------------------------------
# prototype machinery
# --------------------------------------------------------------------------
@torch.no_grad()
def extract_prototypes(model: AER, windows: torch.Tensor, labels: np.ndarray,
                       batch_size: int) -> dict[int, np.ndarray]:
    """Per-month mean latent from a clean eval-mode pass. {month: (D,)}"""
    model.eval()
    zs = []
    for i in range(0, len(windows), batch_size):
        zs.append(model.encode(windows[i:i + batch_size, 1:-1]))
    z = torch.cat(zs).cpu().numpy()
    return {int(m): z[labels == m].mean(axis=0) for m in np.unique(labels)}


def aggregate_prototypes(local_protos: dict[str, dict[int, np.ndarray]],
                         seed: int):
    """Server side: FINCH-cluster each month's client prototypes (cosine).

    Returns {month: (cluster_protos (C, D), mean_proto (D,))}. With a single
    contribution the cluster set is that prototype itself.
    """
    by_month: dict[int, list[np.ndarray]] = {}
    for client in sorted(local_protos):                     # deterministic order
        for month, proto in local_protos[client].items():
            by_month.setdefault(month, []).append(proto)

    global_protos = {}
    for month, protos in by_month.items():
        stack = np.stack(protos)
        if len(stack) > 1:
            c, _, _ = FINCH(stack, distance="cosine",
                            ensure_early_exit=False, verbose=False)
            assign = c[:, -1]
            clusters = np.stack([stack[assign == u].mean(axis=0)
                                 for u in np.unique(assign)])
        else:
            clusters = stack
        global_protos[month] = (clusters, clusters.mean(axis=0))
    return global_protos


def hierarchical_proto_loss(z: torch.Tensor, labels: torch.Tensor,
                            global_protos: dict, alpha: float,
                            temperature: float, device) -> torch.Tensor:
    """Vectorized FPL loss over the batch.

    InfoNCE: positives = cluster prototypes of the instance's own month,
    negatives = every other month's cluster prototypes; similarities are
    cosine / temperature. Regularizer: MSE to the own month's mean prototype.
    Only instances whose month has a global prototype contribute; if none
    do, returns 0 (grad-free) — the legacy crash case.
    """
    months = sorted(global_protos)
    proto_stack = torch.tensor(
        np.concatenate([global_protos[m][0] for m in months]),
        dtype=torch.float32, device=device)
    proto_labels = torch.tensor(
        np.concatenate([[m] * len(global_protos[m][0]) for m in months]),
        dtype=torch.int64, device=device)
    mean_stack = torch.tensor(
        np.stack([global_protos[m][1] for m in months]),
        dtype=torch.float32, device=device)
    month_to_row = {m: i for i, m in enumerate(months)}

    pos_mask = labels.unsqueeze(1) == proto_labels.unsqueeze(0)   # (B, P)
    matched = pos_mask.any(dim=1)
    if not bool(matched.any()):
        return torch.zeros((), device=device)

    zm, pm = z[matched], pos_mask[matched]
    sims = (torch.nn.functional.normalize(zm, dim=1)
            @ torch.nn.functional.normalize(proto_stack, dim=1).T) / temperature
    log_all = torch.logsumexp(sims, dim=1)
    log_pos = torch.logsumexp(sims.masked_fill(~pm, float("-inf")), dim=1)
    infonce = (log_all - log_pos).mean()

    rows = torch.tensor([month_to_row[int(m)] for m in labels[matched]],
                        dtype=torch.int64, device=device)
    reg = torch.nn.functional.mse_loss(zm, mean_stack[rows])

    return alpha * infonce + (1 - alpha) * reg


# --------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------
class FPLTrainer:
    def __init__(self, client_windows: dict, fl: dict, seed: int):
        torch.manual_seed(seed)
        self.cfg_m, self.cfg_t = fl["model"], fl["training"]
        self.seed = seed
        self.device = torch.device(self.cfg_t["device"])
        self.clients = sorted(client_windows)

        first = client_windows[self.clients[0]]
        n_features = first["windows"].shape[2]
        window_size = first["windows"].shape[1]

        self.data = {
            c: (torch.tensor(client_windows[c]["windows"], dtype=torch.float32,
                             device=self.device),
                torch.tensor(client_windows[c]["labels"], dtype=torch.int64,
                             device=self.device))
            for c in self.clients
        }
        # one global init, broadcast to all clients (legacy `ini`)
        self.global_model = AER(n_features, window_size,
                                self.cfg_m["lstm_units"]).to(self.device)
        self.models = {c: copy.deepcopy(self.global_model) for c in self.clients}
        self.optimizers = {
            c: torch.optim.Adam(self.models[c].parameters(),
                                lr=self.cfg_t["learning_rate"])
            for c in self.clients
        }
        self.global_protos: dict = {}
        self.rng = np.random.default_rng(seed)

        self.local_proto_rows: list[dict] = []
        self.global_proto_rows: list[dict] = []
        self.log_rows: list[dict] = []

    # ---------------------------------------------------------------- local
    def _local_update(self, client: str, round_idx: int):
        model, opt = self.models[client], self.optimizers[client]
        windows, labels = self.data[client]
        gen = torch.Generator().manual_seed(
            int(np.random.default_rng([self.seed, round_idx,
                                       self.clients.index(client)])
                .integers(2**31)))
        loader = DataLoader(TensorDataset(windows, labels),
                            batch_size=self.cfg_t["batch_size"],
                            shuffle=True, generator=gen)
        model.train()
        for epoch in range(self.cfg_t["local_epochs"]):
            s_tot = s_mse = s_proto = 0.0
            for wb, lb in loader:
                opt.zero_grad()
                x, ry_t, y_t, fy_t = split_window_targets(wb)
                ry, y, fy, z = model(x)
                loss_mse = aer_loss(ry, y, fy, ry_t, y_t, fy_t,
                                    self.cfg_m["reg_ratio"])
                loss_proto = (hierarchical_proto_loss(
                    z, lb, self.global_protos, self.cfg_t["proto_alpha"],
                    self.cfg_t["infonce_temperature"], self.device)
                    if self.global_protos else torch.zeros((), device=self.device))
                loss = loss_mse + loss_proto
                loss.backward()
                opt.step()
                s_tot += loss.item()
                s_mse += loss_mse.item()
                s_proto += float(loss_proto.detach())
            n = len(loader)
            self.log_rows.append(dict(round=round_idx, client=client,
                                      epoch=epoch, loss=s_tot / n,
                                      loss_mse=s_mse / n,
                                      loss_proto=s_proto / n))

        protos = extract_prototypes(model, windows, labels.cpu().numpy(),
                                    self.cfg_t["batch_size"])
        for m, p in protos.items():
            if not np.isfinite(p).all():
                raise AssertionError(f"Non-finite prototype: {client} month {m}")
            self.local_proto_rows.append(
                dict(round=round_idx, client=client, month=m,
                     **{f"f{i}": v for i, v in enumerate(p)}))
        return protos

    # --------------------------------------------------------------- server
    def _fedavg(self, online: list[str]):
        if self.cfg_t["averaging"] == "weight":
            sizes = np.array([len(self.data[c][0]) for c in online], float)
            freq = sizes / sizes.sum()
        else:
            freq = np.full(len(online), 1.0 / len(online))
        global_w = {k: sum(f * self.models[c].state_dict()[k]
                           for c, f in zip(online, freq))
                    for k in self.global_model.state_dict()}
        self.global_model.load_state_dict(global_w)
        for c in self.clients:
            self.models[c].load_state_dict(copy.deepcopy(global_w))

    # ---------------------------------------------------------------- train
    def train(self, rounds: int):
        n_online = max(1, int(round(self.cfg_t["participation"]
                                    * len(self.clients))))
        for r in range(rounds):
            online = sorted(self.rng.choice(self.clients, size=n_online,
                                            replace=False).tolist())
            local_protos = {c: self._local_update(c, r) for c in online}
            self.global_protos = aggregate_prototypes(local_protos, self.seed)
            for m, (clusters, _) in self.global_protos.items():
                for ci, p in enumerate(clusters):
                    self.global_proto_rows.append(
                        dict(round=r, month=m, cluster=ci,
                             **{f"f{i}": v for i, v in enumerate(p)}))
            self._fedavg(online)
        return self

    # -------------------------------------------------------------- exports
    @torch.no_grad()
    def latent_trajectories(self, client_windows: dict) -> pd.DataFrame:
        """Final-model latents per window — `feature_trajectories` schema
        (district, kind, window, f0..) so the dependence battery's Tier-4
        methods consume AER latent space directly."""
        frames = []
        for c in self.clients:
            model = self.models[c]
            model.eval()
            windows, labels = self.data[c]
            zs = []
            for i in range(0, len(windows), self.cfg_t["batch_size"]):
                zs.append(model.encode(windows[i:i + self.cfg_t["batch_size"],
                                               1:-1]))
            z = torch.cat(zs).cpu().numpy()
            df = pd.DataFrame(z, columns=[f"f{i}" for i in range(z.shape[1])])
            df.insert(0, "month", labels.cpu().numpy())
            df.insert(0, "window", client_windows[c]["window_start_step"])
            df.insert(0, "kind", "aer_latent")
            df.insert(0, "district", c)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
