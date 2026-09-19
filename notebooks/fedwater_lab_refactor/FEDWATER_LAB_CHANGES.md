# `fedwater_lab` — dynamic worlds, sensor selection, prototype reconstruction

Drop-in replacement for the modules in the package folder (the one holding
`fedwater_lab/`), plus the refactored `POC_01__fedwater_lab_end_to_end.ipynb`.

## Before you run it

* **`__init__.py` was not in the upload and is not shipped.** If it re-exports
  `PLACEMENTS` from `specs`, that import now fails: replace it with `CLASSES`
  (and add `recon` if it lists submodules). The notebook imports from the
  submodules directly, so it does not depend on `__init__`.
* **Worlds must come from the current pipeline** (they carry
  `data/03_primary/sensor_placement.csv` and a partition). Older worlds raise a
  clear error on load.
* **Every cached run is retrained.** `CellSpec.to_dict()` changed (selection
  fields; the three retention fields now round-trip), so new run keys differ
  from anything already in the lab store.
* In `LAB_DIR` (notebook §0), point at the folder that contains `fedwater_lab/`.

## Module changes

| module | change |
|---|---|
| `specs` | `placement` → `kinds` × `classes` (pure / mixed) + `include` / `exclude` (sensor ids). Set-valued fields are sorted before hashing (same selection, same key). `snapshot_rounds`, `snapshot_points`, `keep_weights` now round-trip through `spec.json` (they were silently dropped). `PLACEMENTS` → `CLASSES`. |
| `worlds` | Reads the world's own partition (`districts`, `partition_info`, `district_demand`) and `sensor_placement.csv`; `placement_source`, `districting_method`, `partition_id`, `network`. `list_worlds` / `pick_world` browse cached worlds from manifests only. The hardcoded `districts_graeme.yml` path and the `sensors` manifest reader are gone. |
| `data` | `sensor_table` / `dropped_table` (what the spec keeps or drops, and why). Client columns are **slot names**, as in the pipeline, so channel *i* is the same slot in every client. Noise now goes through the pipeline's own `add_measurement_noise`; the POC's noisy series equal the pipeline's `observed` values. The old placement-heuristics dependency is gone. |
| `store` | Records network, districting, partition id and placement source in `meta.json` and the index; index gains `classes`, `n_include`, `n_exclude`. |
| `bridge` | Adds `add_measurement_noise`, `load_globals`, `read_partition`, `partition_meta`. |
| `train` | Adds `reconstruction_model`, `decode` (the decoder half of `AER.forward`, standalone) and `reconstruct_prototypes`. |
| `recon` (new) | Torch-free: true and demand windows on identical window starts, channel table with label mixture, FINCH membership (nearest centroid by cosine), fidelity, own-vs-mixture retrieval (r², DTW), `room`, separability, phase summaries. |
| `figures` | `plot_reconstruction`, `plot_finch_month`, `plot_retrieval_summary`. |
| `widgets` | `preprocessing_browser` gains the selection controls and prints the selected sensors with their reasons; new `reconstruction_browser` (client × channel × month or phase mean, decoded sources, demand overlay, metrics for the selection) and `finch_browser`. |
| `metrics`, `cluster`, `runner` | unchanged |

## Notebook

§1 browse and pick a cached world by districting method; §1b choose sensors and
see why each is there; §2–6 as before (selection-aware); **§7 prototype
reconstruction**: a decode self-check, fidelity per phase, the interactive
month-by-month browser, FINCH centroids, phase-focused summaries, and mixture
retrieval; §8 grid with the selection as an axis.

## What was verified here, and what was not

Verified on real cached KY7 worlds (manual and fast_greedy partitions, dynamic
and manual sensors, a 20-month world with a `final` phase):

* world browsing and selection;
* frames, noise parity with the pipeline, and windows;
* store round-trip;
* both browsers, driven programmatically;
* all of `recon`;
* every notebook cell that does not need torch.

§7 ran with `train` stubbed (synthetic decodings in the real long format).

**Not executed: training and decoding.** Torch cannot be installed in this
sandbox. `decode` was checked line by line against `AER.forward`. The
notebook's §7 self-check cell asserts `decode(z) == forward(x)` on real windows
the first time you run it — if that cell passes, the rest of §7 is on solid
ground.

## A first reading from the data itself (true windows, 20-month KY7 world)

The only mixed flow sensor with room for the mixture (District_C `q_P-260`,
30% District_A) is explained **better by its own district** than by its label
mixture in `final` (gain −0.16, DTW agrees). Mixed sensors whose weights sit on
districts that never drifted have `room` ≈ 0, so the test is silent there by
construction. Worlds whose drifted district appears in many mixed sensors'
labels (the fast_greedy cut does) are the better test bed once they have a
settled `final` phase.
