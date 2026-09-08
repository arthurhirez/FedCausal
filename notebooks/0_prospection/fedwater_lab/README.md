# `fedwater_lab`

The unified sandbox for **sensor placement × time-series preprocessing × federated training ×
client similarity**. Replaces the overlapping halves of two notebooks:

| notebook | had | lacked |
|---|---|---|
| `aer_federated_sandbox_plot` | the prototype family (`local` / `global` / `post_fedavg`), saved weights, controlled probes, the projection browser | placement, transforms, the scaling axis, a deterministic run key |
| `PP_01__preprocessing_x_placement_latent` | placement × transform × scaling × geometry, deterministic cell caching, phase-wise similarity | the global/FINCH scopes, weights, model & training axes |

Each had what the other needed, plus two divergent copies of the trainer, the window builder,
the metrics and the store — which silently made their numbers incomparable.

## Install

Nothing to build. Put the package where `fedwater` is importable and add the repo to
`sys.path`:

```
<repo>/
  src/fedwater/            # unchanged
  notebooks/prospection/
    fedwater_lab/          # this package
    POC_00__fedwater_lab_end_to_end.ipynb
```

```python
import sys, pathlib
REPO = pathlib.Path.cwd().parents[1]          # from notebooks/prospection/
sys.path[:0] = [str(REPO / "src"), str(REPO / "notebooks" / "prospection")]

from fedwater_lab import load_world, Store, POC
from fedwater_lab.runner import run_cell

world = load_world(REPO / "data/09_experiments/worlds/e0f8c48c1006")
store = Store(REPO / "notebooks/prospection/lab_sandbox")
res   = run_cell(world, POC, store=store)     # cache read if the key exists
```

Requires numpy, pandas, torch, scikit-learn, scipy, pyyaml (+ matplotlib and ipywidgets for
the figures and browsers). kedro is *not* required: `bridge` falls back to loading
`fedwater`'s `nodes.py` files directly when the pipeline `__init__` modules cannot import it.

## Modules

| module | owns | depends on |
|---|---|---|
| `bridge` | the **only** import seam to `fedwater`; nothing upstream is re-implemented | — |
| `worlds` | world dir → params merge, ground truth, phases, drift progress | — |
| `specs` | the experiment axes as frozen dataclasses; `CellSpec.key()` is run identity | — |
| `data` | placement → transform → windows → scaling; bounded `WindowCache` | bridge, specs, worlds |
| `train` | one `SnapshotFPLTrainer`, `run_cell`, `compile_prototypes` | bridge, specs, worlds |
| `store` | two-layer persistence: immutable `runs/`, appendable `scores/` | specs, worlds |
| `metrics` | one definition of every latent / probe / drift metric, `check_run` | bridge, worlds |
| `cluster` | similarity + partition, single-slice API, shuffle nulls | bridge, worlds |
| `figures` | pure `df → Axes` | — |
| `widgets` | thin ipywidgets wiring over `figures` | data, figures |
| `runner` | the single entry point: cache-aware train + persist | data, specs, store |

## What changed behaviourally

* **One client source.** Every placement, `manual` included, is rebuilt from
  `pressures`/`flows` with `noise=False`. The shipped `07_model_output/clients/*.csv` carry
  `observed` (noisy) values, so using them for `manual` while rebuilding the others would
  confound the placement axis with a noise axis. Verified: rebuilt columns match the CSV
  exactly; values differ by ≈ the noise sigma.
* **`reference_months` comes from the world.** Defaults to `manifest.effective.scenario.
  drift.warmup_months` and is asserted not to reach past the first node switch.
* **Phases derived, not fixed.** `init = [0, first_switch)`, `final = [last_switch +
  ceil(ramp_days/30), n_months)`. On `e0f8c48c1006` that is months 0–5 and 25–29 — 11 of 30.
  `world.drift_progress()` gives the demand-weighted curve so the excluded transition band is
  a number rather than a convention.
* **Ground truth derived twice** (`consumption_map` string and `income_landuse_mapping`
  pairs) and asserted equal at load.
* **`shared` scaling added.** One MinMax on the *pooled* reference windows, so cross-client
  level survives. `ref_range_per_client(...).attrs["identical_bounds"]` distinguishes it from
  per-client `minmax_ref` — the check that would have caught the inert
  `preprocessing.scaling: "shared"` override.
* **Month-stratified snapshots.** Both notebooks sampled uniformly over the timeline, giving
  the six label-clean init months ~a fifth of the points.
* **`stage = pre|post` in one snapshot table**, so `post_fedavg` prototypes exist at every
  snapshot round rather than only the last, and `fedavg_gap` is a groupby.
* **`update_gram`** (5×5 cosine of the FedAvg update deltas, per round) computed at train
  time, which is why per-round weights are never retained.
* **Deterministic run key** over the normalised spec + world hash + resolved `sensor_hash`.
  Re-running is a cache read; a placement rule that silently changes its sensors becomes a new
  key rather than a corrupted hit.
* **Single-slice similarity API.** Every function asserts one `(scope, round, phase)`. Round
  explains the large majority of prototype variance, so pooling across it measures training
  progress, not client identity.

## De-duplication ledger

`SnapshotFPLTrainer` ×2 → 1 · `build_windows` ×2 → 1 · `latent_metrics` ×2 divergent → 1
superset · `_write_table`/`_read_table` ×2 → 1 · `save_analysis` ×2 (both inside the AER
notebook) → 1 · `cell_id_for` + `run_id_for` → `CellSpec.key()` · `_cos_matrix` +
`client_prototype_similarity` + `_deconfounded_prototypes` → `cluster.METHODS`.

## Retention policy

Measured on this world: `latents_by_round` **3.5 MB** per cell versus `fed_model` **0.9 MB**.
The storage dial is snapshot resolution, not weight retention.

* `prototype_history`, `global_prototype_history`, `update_gram` — **every round** (small, and
  what the similarity channels consume).
* `latents_by_round` — `snapshot_rounds="sparse"` → `{-1, 0, mid, final}`. Use `"all"` only
  for named deep-dive cells.
* `fed_model.pt` — final round only (`keep_weights="final"`). `"all_rounds"` exists and is
  never the default. Weights are a cache: `spec.json` + seed reproduces any run exactly.

## Scope of this pass

**Unification and structure only.** Explicitly *not* included: new similarity channels
(decoder reconstruction, second-moment/CKA, distributional distances, transfer error — the
slice API is shaped to accept them, `cluster.METHODS` is the registry), hydraulic decoupling,
gradual/progress-based scoring, cross-world pooling. Drift metrics are computed and stored but
never rank or gate a cell.

## Open items

* **Power.** 5 clients, 2 tokens → exhaustive-null p-floor 0.10 (3–2 split) or 0.20 (4–1).
  A 4–1 world can never be individually significant; that is arithmetic, not implementation.
* **Label-clean months.** 6 init / 5 final at `n_months=30`. Longer warm-up and post-drift
  stretches are the prerequisite for the init-state question.
* **`variance` placement** is excluded from `PLACEMENTS`: deterministic within a world,
  data-dependent across worlds (it ranks candidates on the world's own residual statistics,
  which include the drift). `sensor_hash` in the run key is what makes it safe to re-add.
* **Common-mode contamination** is visible as raw `proto_cos` ≈ 0.97 for every pair.
* **Noise determinism.** `train.assert_determinism` raises if `noise=True` without
  `PYTHONHASHSEED` set, because `sensing.add_measurement_noise` keys its RNG on
  `hash(sensor)`.
