"""Label factory: simulation-supervised datasets for learned detectors.

Each *world* is a full pipeline run (simulation -> FL -> dependence ->
attribution) under a randomized spec: seed, coupling variant/fraction, drift
origin district, drift seed node, drift target income. From every world we
extract:

* ``labeled_pairs``   — one row per district pair: dependence statistics
  from the oracle battery and the federated Level-T methods (features),
  with the physical truth (open boundaries, hydraulic distance) and the
  world spec (labels/metadata).
* ``labeled_clients`` — one row per district: drift-signal summaries,
  corrector outputs, C4 loop gain/weight, and the SIGNATURE features the
  analytic ladder identified as discriminative (outward-vs-inward Granger
  asymmetry); label = is the district the drift ORIGIN, and its onset.

Training discipline for consumers (documented here because it is the whole
point): always cross-validate GROUPED BY WORLD — rows within a world share
everything; and hold out entire coupling variants and, later, a second
network topology to test that the learned detector generalizes physics,
not simulator quirks.

Execution is delegated to :mod:`fedwater.experiments` (one engine for the
factory, D0 replication, and the sweeps): worlds are cached by CONTENT HASH
of their effective sim configuration — never by ordinal, so editing
``n_worlds`` or the spec generator can no longer silently reuse a stale
world — and every world/run leaves a ``manifest.json`` papertrail (resolved
config, package versions, timings). The extractors live in
``experiments.harvest`` (``pairs``/``clients`` groups) with feature names
unchanged, so AutoML consumers are unaffected.

Divergence from the engine's default failure policy, on purpose: a sweep
records failures and continues (infeasibility is data), but the factory
RAISES on any failed world — a supervised table silently missing worlds is
worse than no table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fedwater.experiments.engine import ExperimentEngine
from fedwater.experiments.spec import resolve_run, resolve_world

DISTRICTS = [f"District_{x}" for x in "ABCDE"]

_FACTORY_PIPELINES = ("fl", "dependence_detection", "drift_attribution")
_FACTORY_ORACLE_EXTRAS = {"n_surrogates": 20, "n_surrogates_expensive": 8,
                          "minirocket": {"window_h": 24, "stride_h": 6,
                                         "n_kernels": 1000, "pca_dims": 8}}
_FACTORY_RUN_SURROGATES = {"n_surrogates": 40, "n_surrogates_expensive": 15}


def build_world_specs(fl: dict, districts: dict, seed: int) -> pd.DataFrame:
    """Deterministic randomized specs. One row per world."""
    cfg = fl["label_factory"]
    rng = np.random.default_rng(seed)
    variants = cfg["coupling_variants"]  # e.g. [[baseline,0],[partial,.3],...]
    rows = []
    for w in range(cfg["n_worlds"]):
        variant, frac = variants[w % len(variants)]
        tgt = DISTRICTS[int(rng.integers(len(DISTRICTS)))]
        seed_node = str(rng.choice(districts["districts"][tgt]))
        rows.append(dict(
            world=w, world_seed=int(rng.integers(1, 2**16)),
            variant=variant, close_fraction=float(frac),
            drift_district=tgt, drift_seed_node=seed_node,
            drift_to_income=str(rng.choice(cfg["drift_incomes"])),
            anchor_scale=cfg["anchor_by_variant"].get(variant, 0.05),
        ))
    return pd.DataFrame(rows)


def _world_raw(spec, cfg) -> dict:
    """Engine world spec for one factory row (same reductions the factory
    always applied: short horizon, cheap oracle tiers, small MiniRocket)."""
    return {
        "sim_seed": int(spec.world_seed),
        "n_months": int(cfg["n_months"]),
        "anchor_scale": float(spec.anchor_scale),
        "coupling": {"variant": spec.variant,
                     "close_fraction": float(spec.close_fraction)},
        "drift": {"tgt_district": spec.drift_district,
                  "seed_node": str(spec.drift_seed_node),
                  "to_income": spec.drift_to_income},
        "oracle": {"tiers": cfg["oracle_tiers"], **_FACTORY_ORACLE_EXTRAS},
    }


def generate_labeled_worlds(world_specs: pd.DataFrame, fl: dict):
    """Run every world through the experiments engine (cache-or-run,
    hash-keyed); extract supervised tables from the harvested groups."""
    cfg = fl["label_factory"]
    project = Path.cwd()
    base_params = yaml.safe_load(
        (project / "conf/base/parameters.yml").read_text())
    engine = ExperimentEngine(project, root=Path(cfg["scratch_dir"]))
    run = resolve_run({"step_size": cfg["step_size"], "batch_size": 128,
                       "rounds": cfg["fl_rounds"],
                       **_FACTORY_RUN_SURROGATES},
                      base_params, pipelines=_FACTORY_PIPELINES)

    pairs, clients = [], []
    for spec in world_specs.itertuples(index=False):
        world = resolve_world(_world_raw(spec, cfg), base_params)
        built = engine.ensure_world(world)
        if built["status"] != "ok":
            raise RuntimeError(
                f"world {spec.world} ({world['sim_hash']}) failed simulation:"
                f" {built['status']} — see {built['dir']}/manifest.json")
        executed = engine.ensure_run("label_factory", world, run,
                                     harvest=("pairs", "clients"))
        if executed["status"] != "ok":
            raise RuntimeError(
                f"world {spec.world} ({executed['id']}) failed:"
                f" {executed['status']} — see {executed['dir']}/manifest.json")

        meta = {"world": spec.world, "variant": spec.variant,
                "close_fraction": spec.close_fraction}
        p = pd.read_parquet(executed["dir"] / "harvest/pairs.parquet")
        pairs.append(p.assign(**meta))
        c = pd.read_parquet(executed["dir"] / "harvest/clients.parquet")
        clients.append(c.assign(world=spec.world, variant=spec.variant))

    labeled_pairs = pd.concat(pairs, ignore_index=True)
    labeled_clients = pd.concat(clients, ignore_index=True)
    if labeled_clients.groupby("world")["label_is_origin"].sum().ne(1).any():
        raise AssertionError("Each world must have exactly one drift origin.")
    return labeled_pairs, labeled_clients
