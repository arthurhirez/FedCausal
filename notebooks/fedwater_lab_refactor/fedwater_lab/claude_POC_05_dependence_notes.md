# POC_05 — dependence and causality from prototypes: implementation notes

Companion to `POC_05__dependence_v1.ipynb`. Seven new modules in `fedwater_lab/`, all
torch-free except where they borrow the decoder, and none of them re-implements anything
upstream.

| module | stage | what it owns |
|---|---|---|
| `checks.py` | all | the sanity register: every check, its verdict, and the honest-n row |
| `precond.py` | 0 | phase/block census, chain budget, occupancy, channel balance, **storage-exchange gate**, adjacency (scoring only) |
| `traj.py` | 1–2 | the frozen ruler: decoded-prototype features, the raw-latent arm, ruler stability, recoverability, scaling discrimination |
| `transfer.py` | 3 | prequential transfer gain, capacity null, AR-R² diagnostic, rolling `G(b)`, onset test, the pure/mixed/all arms |
| `chain.py` | 4 | codebook, transition matrix, occupancy/JS, order and homogeneity tests, MI/TE with bias, two nulls |
| `partialdep.py` | 5 | partial correlation, shrinkage path, structural contrast, and the precondition that refuses |
| `grid.py` | — | index-free `scan`, comparability blocks, method-hashed `score_grid`, axis effects |
| `selfcheck.py` | — | every estimator run against **saved runs**; no synthetic data anywhere |

---

## 1. The three design decisions that were measured, not assumed

**Concurrent controls, not a lagged common factor.** A shared driver makes every client a noisy
measurement of the same latent `f[b]`. With lagged controls only, `f[b]` has to be extrapolated,
so *any* additional client improves the forecast and the gain becomes a statement about
measurement count. Conditioning on the other clients' **present** absorbs `f[b]`, leaving only
the source's past able to carry a mechanism. `TransferCfg(control="mean"|"each",
control_time="both")`.

**Do not subtract the cross-client mean.** It is what invariant 5 prescribes for dependence
statistics, and it is the wrong tool for a four-client panel: the sum-to-zero constraint forces a
spurious negative dependence between residuals. Control for the shared driver, do not remove it.
(The same degeneracy POC_04 hit when centring prototypes across clients.)

**Classes as replicates.** `transfer.panel` carries a replicate axis, so one model is fitted
pooled over the calendar classes of a block and scored on each. That is what makes `G(b)` on ~20
blocks estimable at all. It buys **estimation, not inference** — classes inside a month are not
independent — so every null permutes whole months with their classes attached
(`chain.month_block_null`, and `transfer.capacity_null` shuffles along the block axis only).

## 2. What the saved runs actually say (ky72, spectral, dynamic, drift = District_A)

Measured on the nine runs in `runs_POC_V2`, **raw-latent arm only** — the decoded arm needs the
decoder and the world, neither of which shipped with the export.

* **The protocol contract holds exactly.** The per-month mean of `latent_trajectories` equals the
  run's own `post_fedavg_full` prototype to `max |Δ| = 0`. The trajectory is the artifact the
  protocol exchanges, not something adjacent to it.
* **Calendar budget.** 70 months (init 0–11, transition 12–52, final 53–69), 14 weekly cycle
  classes at 12 h hops. At `block=3 months, bin_h=24`: 24 blocks (init 4 / transition 14 /
  final 6), 7 classes, minimum occupancy 4 windows per prototype. `block=2` gives 35 blocks at
  occupancy 4; `block=1` drops occupancy to ~2 and breaches the floor.
* **No predictive transfer in latent space.** Restricted AR-R² is 0.89–0.99 — a client's own past
  plus the others' present already explains nearly everything — and `G` runs −1.49 to +0.09 with
  no pair clearing its capacity null. This is a real negative on this arm, at this block size.
* **The occupancy fingerprint does separate the drifted client.** Init: every pair at similarity
  1.000. Transition: District_A's pairs fall to 0.935 while B/C/D stay at 1.000. Final: A's pairs
  at 0.852–0.867, the others still 0.999–1.000. Symbol occupancy shows A moving to 88% state 0 in
  final against 57% in init while the others barely move. That is idea 2 working, from the
  cheapest object available, with no reconstruction quality required.
* **Order 0 is strongly rejected** (G² = 163, df = 6, p ≈ 1e-32) and order 1 vs 2 is not
  (p = 0.32): there are dynamics, and order 1 is memory enough.
* **MI is mostly calendar.** MI(A,B) = 0.485 nats against a month-block null mean of 0.365
  (q95 0.418, p = 0.016). The excess is real but small; the raw number would have been read as
  40× the plug-in bias (0.012) and that reading would have been wrong.
* **TE is below its bias for every pair** (max TE ≈ 0.009 against bias 0.037 nats at n = 161).
  Nothing directional to report in symbol space at k = 3 on this horizon.
* **The partial-correlation precondition refuses.** Condition number 137 — near rank-1, the
  shared component dominates. The stage reports that instead of returning confident numbers.
* **The onset test has little power on this world.** The ramp covers 13 of 19 evaluated blocks,
  so `p_uniform` = 0.68: a peak inside the ramp is nearly uninformative when the ramp is 41 of 70
  months. A world with a shorter ramp (or a longer post-drift stretch) is what makes that test
  worth running.
* **Stationarity split has no power at `block=3`**: 7 transitions per half. It needs `block=1`
  for init, which breaches the occupancy floor — the trade-off is real and is printed rather
  than hidden.

## 3. Limits worth stating before the next run

* One world, one seed, six pairs. Every axis effect is a direction.
* The factor study over the nine runs sits in **one comparability block**, so the spread column
  is empty by construction; cross-block contrasts are refused by `grid.check_contrast`.
* The round −1 snapshot covers only the 12 commissioning months, so it is an init-phase control,
  not a full-horizon one. A proper untrained control is `aligned.control_run` and needs the model.
* The storage-exchange gate was never evaluated: it needs `wn_variant.pkl` and `flows.parquet`.
  Until it is, the window-scale lag arm stays closed by default rather than by evidence.

## 4. What is needed to run the decoded arm

The decoded arm is the one POC_04 showed carries the similarity ordering, and it cannot run from
the run export alone. It needs, for `sim_hash = f335b2cfad67`:

1. the world directory (`manifest.json` + `clone/`), or at minimum
   `data/02_intermediate/demand_series.parquet`, `data/03_primary/sensor_placement.csv`,
   `data/02_intermediate/flows.parquet` and `data/02_intermediate/wn_variant.pkl`;
2. the repo on the path, so `fedwater_lab.bridge` can resolve the AER class and `fed_model.pt`
   can be loaded.

With both, `§3` switches to `TJ.trajectory(...)` and `§4b` (pure / mixed / all) becomes live.
