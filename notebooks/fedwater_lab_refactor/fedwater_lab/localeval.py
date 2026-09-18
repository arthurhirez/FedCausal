"""Local vs global: what FedAvg costs each client.

After the last round, every client holds two models:

* **local**  its weights right after its last local update, before the final
  FedAvg. This is the model its shared prototypes were extracted from
  (`meta.client_weights == "local_pre_fedavg"`, `train.local_model`).
* **global** the FedAvg average, which every other POC_02 section decodes
  with.

`model_matrix` scores every model on every client's windows, which gives the
own-client diagonal and the cross-client off-diagonal. For model M on client
d's windows x_w, with reconstructions M(x_w):

loss   = aer_loss(M(x), x)  (training weighting, `reg_ratio`)
ratio  = loss / aer_loss(each window's own input mean)   (< 1: shape)
amp    = median_{w,i} std_t M(x_w)_i / std_t x_wi
r      = median_{w,i} corr_t(M(x_w)_i, x_wi); `r_pure` / `r_mixed` restrict
         i to that slot class (from `recon.channel_table`)

Reading:
* the **diagonal gap** (local minus global, on the own client) is the price
  of averaging;
* the **off-diagonal** asks whether client c's model can reconstruct client
  d at all. A column that only its own model decodes means channel i
  behaves differently there: sign, role, or level. Orientation should close
  the part that is sign.

Everything is in-sample: FL trains on every month, so there is no held-out
split here (POC_03 has one for the centralised ceiling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aligned as AL
from .worlds import World

__all__ = ["reconstruct", "model_matrix", "gap_table", "lg_row", "local_runs",
           "plot_matrix"]


def reconstruct(model, X: np.ndarray, batch: int = 512) -> np.ndarray:
    """Full W-step reconstructions (ry, y, fy stitched), like `train.decode`
    applied to the model's own encoding."""
    import torch
    from . import bridge
    from .train import model_device
    dev = model_device(model)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            wb = torch.as_tensor(np.array(X[i:i + batch], np.float32), device=dev)
            x, _, _, _ = bridge.split_window_targets(wb)
            ry, y, fy, _ = model(x)
            outs.append(torch.cat([ry.unsqueeze(1), y, fy.unsqueeze(1)], 1)
                        .cpu().numpy())
    return np.concatenate(outs)


def _aer(pred, X, r):
    def m(a, b):
        return float(np.mean((a - b) ** 2))
    return ((r / 2) * m(pred[:, 0], X[:, 0])
            + (1 - r) * m(pred[:, 1:-1], X[:, 1:-1])
            + (r / 2) * m(pred[:, -1], X[:, -1]))


def _med_amp_r(D, X, chans=None):
    if chans is not None:
        D, X = D[:, :, chans], X[:, :, chans]
    sx, sd = X.std(axis=1), D.std(axis=1)
    ok = sx > 1e-12
    amp = float(np.median(sd[ok] / sx[ok])) if ok.any() else np.nan
    xc = X - X.mean(axis=1, keepdims=True)
    dc = D - D.mean(axis=1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(axis=1) * (dc ** 2).sum(axis=1))
    good = den > 1e-12
    r = float(np.median((xc * dc).sum(axis=1)[good] / den[good])) \
        if good.any() else np.nan
    return amp, r


def model_matrix(world: World, out: dict, channels: pd.DataFrame | None = None,
                 store=None) -> pd.DataFrame:
    """Rows: (model, data client). `model` is "global" or a client name
    (its local weights)."""
    from . import train as T
    res, fw, fl = out["res"], out["fw"], out["fl"]
    r = float(fl["model"]["reg_ratio"])
    clients = sorted(fw)
    models = {"global": out["model"]}
    for c in clients:
        models[c] = T.local_model(res, world, c, store=store)
    cls = {}
    if channels is not None and len(channels):
        for c in clients:
            sub = channels[channels["client"] == c]
            cls[c] = {k: sub.loc[sub["slot_class"] == k, "channel"].astype(int)
                      .tolist() for k in ("pure", "mixed")}
    rows = []
    for d in clients:
        X = np.asarray(fw[d]["windows"], float)
        wmean = np.broadcast_to(X[:, 1:-1].mean(axis=1, keepdims=True), X.shape)
        base = _aer(wmean, X, r)
        for name, M in models.items():
            D = reconstruct(M, X)
            loss = _aer(D, X, r)
            amp, rr = _med_amp_r(D, X)
            row = {"model": name, "data": d,
                   "kind": ("global" if name == "global"
                            else "own" if name == d else "cross"),
                   "loss": loss, "ratio": loss / base, "amp": amp, "r": rr}
            for k in ("pure", "mixed"):
                idx = cls.get(d, {}).get(k)
                row[f"r_{k}"] = (_med_amp_r(D, X, idx)[1] if idx else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def gap_table(mm: pd.DataFrame) -> pd.DataFrame:
    """Per client: global vs own-local on its own windows, and the mean of
    the other clients' local models on them (cross)."""
    out = []
    for d, g in mm.groupby("data"):
        gl = g[g["kind"] == "global"].iloc[0]
        lo = g[g["kind"] == "own"].iloc[0]
        cr = g[g["kind"] == "cross"]
        row = {"client": d}
        for m in ("ratio", "amp", "r", "r_pure", "r_mixed"):
            row[f"{m}_global"] = gl[m]
            row[f"{m}_local"] = lo[m]
            row[f"{m}_cross"] = cr[m].mean()
        row["r_gap"] = lo["r"] - gl["r"]
        row["ratio_gap"] = gl["ratio"] - lo["ratio"]
        out.append(row)
    return pd.DataFrame(out)


def lg_row(mm: pd.DataFrame) -> dict:
    """Four numbers for the §3 sanity table."""
    g = mm[mm["kind"] == "global"]
    o = mm[mm["kind"] == "own"]
    c = mm[mm["kind"] == "cross"]
    return {"global_ratio": float(g["ratio"].mean()),
            "local_ratio": float(o["ratio"].mean()),
            "global_r": float(g["r"].mean()), "local_r": float(o["r"].mean()),
            "cross_r": float(c["r"].mean())}


def local_runs(world: World, out: dict, store=None) -> dict:
    """{client: AlignedRun of its LOCAL model on its own windows} (no
    calendar key), so `winrec` views apply unchanged."""
    from . import train as T
    runs = {}
    for c in sorted(out["fw"]):
        m = T.local_model(out["res"], world, c, store=store)
        fw1 = {c: out["fw"][c]}
        lat = AL.encode_windows(m, fw1)
        runs[c] = AL.build(world, {"latent_trajectories": lat}, fw1, out["fl"],
                           m, AL.Alignment(None, None, 1))
    return runs


def plot_matrix(ax, mm: pd.DataFrame, metric: str = "r", vmin=-1, vmax=1,
                cmap="RdBu_r", title: str = ""):
    """Heatmap: rows = model (global first, then local per client), columns
    = data client."""
    order = ["global"] + sorted(m for m in mm["model"].unique() if m != "global")
    piv = mm.pivot(index="model", columns="data", values=metric).reindex(order)
    im = ax.imshow(piv.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([c.replace("District_", "") for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([m if m == "global" else "local " + m.replace(
        "District_", "") for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set(xlabel="data (client windows)", ylabel="model")
    ax.set_title(title or metric, fontsize=9, loc="left")
    return im
