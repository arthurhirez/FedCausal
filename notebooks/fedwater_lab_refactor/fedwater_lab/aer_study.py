"""Focused AER study: can the upstream AER reconstruct ONE client's windows?

No federation, no prototypes, no FedAvg. The upstream `AER` and `aer_loss`
(borrowed through `bridge`) are trained centrally on one client's windows
with plain Adam, so the reconstruction question is separated from
everything the FL protocol adds. The windows are built by the same
`data.build_windows` path as the FL cells (same selection, transform,
scaling, geometry), for one client.

Split (purged, by month)
------------------------
Every `holdout_every`-th month (months m with m % k == k - 1) is held out.
A window covers native steps [start, start + W * agg), which usually
touches two months. **Validation** windows are those whose month LABEL is
held out. **Training** windows are those that touch NO held-out month at
all. Windows that are neither are dropped. With heavily overlapping
windows (small stride), a random split would put near-copies of every
validation window in training.

Scores (validation, every `eval_every` epochs)
----------------------------------------------
``val_loss``   `aer_loss` on validation windows (same weighting as training);
``base_wmean`` `aer_loss` of each validation window's own input mean at every
               step, i.e. the best flat line (fixed per config);
``base_const`` `aer_loss` of the per-channel training mean, a latent-free flat
               line;
``ratio``      val_loss / base_wmean; < 1 means some shape is reconstructed;
``val_amp``    median over (window, channel) of std_t(dec) / std_t(x);
``val_r``      median over (window, channel) of corr_t(dec, x).

`best` is the epoch with the lowest `val_loss`; its weights are kept and are
what `view` decodes.

Persistence: <root>/<world_hash>/<key>/{cfg.json, history.csv, model.pt,
summary.json}. `key` hashes the config and the world, so re-running a
config is a cache read.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from . import bridge
from .specs import CellSpec, Geometry, ModelCfg
from .worlds import World

__all__ = ["StudyCfg", "study_windows", "purged_split", "train_one", "budget",
           "run_study", "run_studies", "summary_table", "view",
           "plot_curves", "flat_baselines", "SLOT_SETS"]

SLOT_SETS = {"pure": ("pure",), "mixed": ("mixed",),
             "pure+mixed": ("pure", "mixed")}


@dataclass(frozen=True)
class StudyCfg:
    client: str
    slots: str = "pure+mixed"                   # key of SLOT_SETS
    kinds: tuple = ("flow",)
    geometry: Geometry = field(default_factory=lambda: Geometry(3, 84, 12))
    transform: str = "none"
    scaling: str = "minmax_ref"
    lstm_units: int = 30
    reg_ratio: float = 0.5
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    holdout_every: int = 5
    patience: int = 25                          # 0 = no early stopping
    eval_every: int = 5
    seed: int = 0
    device: str = "auto"

    def __post_init__(self):
        if self.slots not in SLOT_SETS:
            raise KeyError(f"slots must be one of {list(SLOT_SETS)}")
        if self.holdout_every < 2:
            raise ValueError("holdout_every must be >= 2")

    def with_(self, **kw) -> "StudyCfg":
        return replace(self, **kw)

    def spec(self) -> CellSpec:
        """The CellSpec whose window path this study uses (cap off)."""
        return CellSpec(kinds=tuple(self.kinds), classes=SLOT_SETS[self.slots],
                        transform=self.transform, scaling=self.scaling,
                        geometry=self.geometry,
                        model=ModelCfg(lstm_units=self.lstm_units,
                                       reg_ratio=self.reg_ratio),
                        cap=None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["geometry"] = asdict(self.geometry)
        d["geometry"]["feature_range"] = list(self.geometry.feature_range)
        d["kinds"] = list(self.kinds)
        d.pop("device")                          # not part of identity
        return d

    def key(self, world_hash: str) -> str:
        blob = json.dumps({"world": world_hash, "cfg": self.to_dict()},
                          sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def label(self) -> str:
        return (f"{self.client.replace('District_', '')}·{self.slots}·"
                f"{self.geometry.label()}·u{self.lstm_units}·lr{self.lr:g}"
                f"·e{self.epochs}")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def study_windows(world: World, cfg: StudyCfg, cache=None):
    """(windows dict for the one client, fl) via `data.build_windows`."""
    from . import data as D
    fw, _, fl = D.build_windows(world, cfg.spec(), clients=[cfg.client],
                                cache=cache)
    return fw[cfg.client], fl


def purged_split(world: World, d: dict, fl: dict, holdout_every: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(train_mask, val_mask) over the client's windows (see module doc)."""
    res_h = float(world.time["resolution_h"])
    spm = int(round(world.time["days_per_month"] * 24 / res_h))
    k = int(round(fl["preprocessing"]["interval_agg_h"] / res_h))
    W = int(fl["preprocessing"]["window_size"])
    start = np.asarray(d["window_start_step"]).astype(int)
    lab = np.asarray(d["labels"]).astype(int)
    first = start // spm
    last = (start + W * k - 1) // spm
    held = lambda m: (m % holdout_every) == holdout_every - 1   # noqa: E731
    val = held(lab)
    touches = np.zeros(len(start), bool)
    for j in range(len(start)):
        touches[j] = any(held(m) for m in range(first[j], last[j] + 1))
    train = ~touches
    return train, val


def budget(world: World, cfg: StudyCfg, steps: int, evals: int = 40,
           cache=None) -> StudyCfg:
    """`cfg` with `epochs` set so training runs ~`steps` optimiser steps
    (strides differ by an order of magnitude in windows per epoch, so
    equal EPOCHS would compare unequal budgets), and `eval_every` set for
    ~`evals` shape evaluations."""
    d, fl = study_windows(world, cfg, cache=cache)
    tr, _ = purged_split(world, d, fl, cfg.holdout_every)
    per_epoch = int(np.ceil(tr.sum() / cfg.batch_size))
    epochs = int(np.ceil(steps / max(per_epoch, 1)))
    return cfg.with_(epochs=epochs, eval_every=max(1, epochs // evals))


def _aer_np(pred, X, r):
    def m(a, b):
        return float(np.mean((a - b) ** 2))
    return ((r / 2) * m(pred[:, 0], X[:, 0])
            + (1 - r) * m(pred[:, 1:-1], X[:, 1:-1])
            + (r / 2) * m(pred[:, -1], X[:, -1]))


def flat_baselines(X_train, X_val, reg_ratio: float) -> dict:
    """AER loss of the two flat predictors on validation windows."""
    const = np.broadcast_to(X_train[:, 1:-1].mean(axis=(0, 1)), X_val.shape)
    wmean = np.broadcast_to(X_val[:, 1:-1].mean(axis=1, keepdims=True),
                            X_val.shape)
    return {"base_const": _aer_np(const, X_val, reg_ratio),
            "base_wmean": _aer_np(wmean, X_val, reg_ratio)}


def _shape_scores(D, X):
    sx, sd = X.std(axis=1), D.std(axis=1)
    ok = sx > 1e-12
    amp = float(np.median(sd[ok] / sx[ok])) if ok.any() else np.nan
    xc = X - X.mean(axis=1, keepdims=True)
    dc = D - D.mean(axis=1, keepdims=True)
    num = (xc * dc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1) * (dc ** 2).sum(axis=1))
    good = den > 1e-12
    r = float(np.median(num[good] / den[good])) if good.any() else np.nan
    return amp, r


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_one(world: World, cfg: StudyCfg, cache=None, verbose: bool = True
              ) -> dict:
    """Train one config. -> {cfg, history, best_epoch, state, summary,
    train_mask, val_mask, windows, fl}."""
    import torch
    from .train import resolve_device

    d, fl = study_windows(world, cfg, cache=cache)
    X = np.asarray(d["windows"], np.float32)
    tr_m, va_m = purged_split(world, d, fl, cfg.holdout_every)
    if tr_m.sum() < cfg.batch_size or va_m.sum() < 1:
        raise ValueError(f"split too small: train {tr_m.sum()}, val {va_m.sum()}")
    dev = torch.device(resolve_device(cfg.device))
    Xtr = torch.tensor(X[tr_m], device=dev)
    Xva = torch.tensor(X[va_m], device=dev)
    base = flat_baselines(X[tr_m], X[va_m], cfg.reg_ratio)

    torch.manual_seed(cfg.seed)
    model = bridge.aer_class()(X.shape[2], X.shape[1], cfg.lstm_units).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)

    def val_pass():
        model.eval()
        outs, loss = [], 0.0
        with torch.no_grad():
            for i in range(0, len(Xva), 512):
                wb = Xva[i:i + 512]
                x, ry_t, y_t, fy_t = bridge.split_window_targets(wb)
                ry, y, fy, _ = model(x)
                loss += bridge.aer_loss(ry, y, fy, ry_t, y_t, fy_t,
                                        cfg.reg_ratio).item() * len(wb)
                outs.append(torch.cat([ry.unsqueeze(1), y, fy.unsqueeze(1)], 1)
                            .cpu().numpy())
        return loss / len(Xva), np.concatenate(outs)

    hist, best, best_state, since = [], (np.inf, -1), None, 0
    t0 = time.time()
    n_batches = int(np.ceil(len(Xtr) / cfg.batch_size))
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(dev)
        tot = 0.0
        for b in range(n_batches):
            wb = Xtr[perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]]
            opt.zero_grad()
            x, ry_t, y_t, fy_t = bridge.split_window_targets(wb)
            ry, y, fy, _ = model(x)
            loss = bridge.aer_loss(ry, y, fy, ry_t, y_t, fy_t, cfg.reg_ratio)
            loss.backward()
            opt.step()
            tot += loss.item() * len(wb)
        vl, dec = val_pass()
        row = {"epoch": ep, "steps": (ep + 1) * n_batches,
               "train_loss": tot / len(Xtr), "val_loss": vl,
               "ratio": vl / base["base_wmean"], "val_amp": np.nan,
               "val_r": np.nan, "seconds": round(time.time() - t0, 1)}
        if ep % cfg.eval_every == 0 or ep == cfg.epochs - 1:
            row["val_amp"], row["val_r"] = _shape_scores(dec, X[va_m])
        hist.append(row)
        if vl < best[0] - 1e-7:
            best, since = (vl, ep), 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
        if verbose and (ep % cfg.eval_every == 0 or ep == cfg.epochs - 1):
            print(f"  ep {ep:>4}  train {row['train_loss']:.4f}  val {vl:.4f}  "
                  f"ratio {row['ratio']:.3f}  amp {row['val_amp']:.3f}  "
                  f"r {row['val_r']:.3f}  {row['seconds']:.0f}s")
        if cfg.patience and since >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {ep} (best {best[1]})")
            break

    model.load_state_dict(best_state)
    _, dec = val_pass()
    amp, r = _shape_scores(dec, X[va_m])
    H = pd.DataFrame(hist)
    summary = {"label": cfg.label(), "client": cfg.client, "slots": cfg.slots,
               "geometry": cfg.geometry.label(), "channels": X.shape[2],
               "n_train": int(tr_m.sum()), "n_val": int(va_m.sum()),
               "batches/epoch": n_batches, "epochs_run": len(H),
               "best_epoch": int(best[1]),
               "best_steps": int((best[1] + 1) * n_batches),
               "val_loss": float(best[0]), **base,
               "ratio": float(best[0] / base["base_wmean"]),
               "val_amp": amp, "val_r": r,
               "seconds": float(H["seconds"].iloc[-1]),
               "device": str(dev)}
    return {"cfg": cfg, "history": H, "best_epoch": int(best[1]),
            "state": best_state, "summary": summary,
            "train_mask": tr_m, "val_mask": va_m, "windows": d, "fl": fl}


def _model_from_state(cfg: StudyCfg, n_features: int, window: int, state):
    m = bridge.aer_class()(n_features, window, cfg.lstm_units)
    m.load_state_dict(state)
    m.eval()
    return m


def run_study(world: World, cfg: StudyCfg, root, overwrite: bool = False,
              cache=None, verbose: bool = True) -> dict:
    """`train_one` with a disk cache under `root`."""
    import torch
    root = pathlib.Path(root)
    d = root / world.sim_hash / cfg.key(world.sim_hash)
    if (d / "summary.json").exists() and not overwrite:
        if verbose:
            print(f"cached  {cfg.label()}  [{d.name}]")
        out = {"cfg": cfg,
               "history": pd.read_csv(d / "history.csv"),
               "summary": json.loads((d / "summary.json").read_text()),
               "state": torch.load(d / "model.pt", map_location="cpu",
                                   weights_only=False)}
        out["best_epoch"] = out["summary"]["best_epoch"]
        win, fl = study_windows(world, cfg, cache=cache)
        tr_m, va_m = purged_split(world, win, fl, cfg.holdout_every)
        out.update(windows=win, fl=fl, train_mask=tr_m, val_mask=va_m,
                   cache_hit=True, key=d.name)
        return out
    if verbose:
        print(f"train   {cfg.label()}  [{d.name}]")
    out = train_one(world, cfg, cache=cache, verbose=verbose)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cfg.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    out["history"].to_csv(d / "history.csv", index=False)
    (d / "summary.json").write_text(json.dumps(out["summary"], indent=2,
                                               default=str))
    torch.save(out["state"], d / "model.pt")
    out.update(cache_hit=False, key=d.name)
    return out


def run_studies(world: World, cfgs: dict, root, overwrite: bool = False,
                cache=None, verbose: bool = True) -> dict:
    """{name: run_study(...)}; failures are reported, not raised."""
    outs = {}
    for name, cfg in cfgs.items():
        try:
            outs[name] = run_study(world, cfg, root, overwrite=overwrite,
                                   cache=cache, verbose=verbose)
        except Exception as exc:
            print(f"FAILED  {name}: {type(exc).__name__}: {exc}")
    return outs


def summary_table(outs: dict) -> pd.DataFrame:
    cols = ["label", "channels", "n_train", "n_val", "batches/epoch",
            "epochs_run", "best_epoch", "best_steps", "val_loss",
            "base_wmean", "base_const", "ratio", "val_amp", "val_r", "seconds"]
    t = pd.DataFrame({k: v["summary"] for k, v in outs.items()}).T
    return t[[c for c in cols if c in t.columns]]


# --------------------------------------------------------------------------
# viewing: reuse the window-reconstruction tools
# --------------------------------------------------------------------------
def view(world: World, out: dict):
    """-> (run, windows dict {client: d}, split frame). `run` is an
    `aligned.AlignedRun` (no calendar key) of the best model on EVERY
    window of the client, so `winrec.window_scores / timeline /
    window_browser` apply unchanged; `split` marks each window train / val /
    dropped (join on `row`)."""
    from . import aligned as AL
    cfg, d = out["cfg"], out["windows"]
    X = np.asarray(d["windows"])
    model = _model_from_state(cfg, X.shape[2], X.shape[1], out["state"])
    fw = {cfg.client: d}
    lat = AL.encode_windows(model, fw)
    run = AL.build(world, {"latent_trajectories": lat}, fw, out["fl"], model,
                   AL.Alignment(None, None, 1))
    split = pd.DataFrame({"row": np.arange(len(X)),
                          "split": np.where(out["val_mask"], "val",
                                            np.where(out["train_mask"],
                                                     "train", "dropped"))})
    return run, fw, split


def plot_curves(axes, outs: dict, names=None):
    """Panel 0: val_loss / base_wmean per training step; panel 1: val_amp;
    panel 2: val_r. One colour per config."""
    from .figures import PALETTE
    names = names or list(outs)
    for j, n in enumerate(names):
        H = outs[n]["history"]
        col = PALETTE[j % len(PALETTE)]
        axes[0].plot(H["steps"], H["ratio"], color=col, label=n)
        e = H.dropna(subset=["val_amp"])
        axes[1].plot(e["steps"], e["val_amp"], color=col, marker=".", label=n)
        axes[2].plot(e["steps"], e["val_r"], color=col, marker=".", label=n)
    axes[0].axhline(1, color="k", lw=0.8, ls="--")
    axes[0].set(xscale="log", xlabel="optimiser steps",
                ylabel="val loss / best flat line")
    axes[1].set(xscale="log", xlabel="optimiser steps",
                ylabel="val amp (median)", ylim=(0, None))
    axes[2].set(xscale="log", xlabel="optimiser steps",
                ylabel="val r (median)", ylim=(-0.1, 1))
    axes[0].legend(fontsize=7)
    return axes
