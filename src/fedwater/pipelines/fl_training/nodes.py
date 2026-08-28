"""fl_training pipeline nodes — thin orchestration over federated.FPLTrainer."""
from __future__ import annotations

import pandas as pd


def effective_fl_seed(fl: dict, seed: int) -> int:
    """FL training seed: ``fl.training.seed`` when set (the experiments
    engine's replicate axis — same world, different training run), else the
    global sim seed, preserving the shipped single-seed behaviour. ``None``
    means unset; 0 is a valid seed."""
    fl_seed = (fl.get("training") or {}).get("seed")
    return int(seed if fl_seed is None else fl_seed)


def train_federated(fl_windows: dict, fl: dict, seed: int):
    """Run the FPL protocol. Returns models, tidy histories, and the
    latent-trajectory table (the dependence plug-in surface)."""
    from .federated import FPLTrainer  # torch import deferred to call time

    seed = effective_fl_seed(fl, seed)
    trainer = FPLTrainer(fl_windows, fl, seed).train(fl["training"]["rounds"])

    prototype_history = pd.DataFrame(trainer.local_proto_rows)
    global_prototype_history = pd.DataFrame(trainer.global_proto_rows)
    training_log = pd.DataFrame(trainer.log_rows)
    latent_trajectories = trainer.latent_trajectories(fl_windows)

    if not training_log["loss"].map(lambda v: pd.notna(v)).all():
        raise AssertionError("Non-finite training loss encountered.")

    models = {
        "global": trainer.global_model.state_dict(),
        **{c: trainer.models[c].state_dict() for c in trainer.clients},
        "meta": {"lstm_units": fl["model"]["lstm_units"],
                 "window_size": trainer.global_model.window_size,
                 "n_features": trainer.global_model.head.out_features,
                 "seed": seed},
    }
    return (models, prototype_history, global_prototype_history,
            latent_trajectories, training_log)
