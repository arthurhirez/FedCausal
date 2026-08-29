# fedwater — simulation

Kedro project generating physically validated water-distribution datasets for
federated concept-drift and dependence-detection research (ICMC-USP master's,
continuing the FAPESP/BEPE work at U.Porto).

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .
kedro run                                        # full 24-month baseline (~25 s)
kedro run --params "coupling.variant=isolated"   # independent-districts variant
pytest -q                                        # 86 tests, ~90 s
```

## Pipelines

| pipeline           | role                                                            |
|--------------------|-----------------------------------------------------------------|
| `network_prep`     | explicit hydraulic options, district partition guard, coupling variants (the *dependence dial*) |
| `urban_scenario`   | block portfolios (land use -> sector composition AND level, income -> residential intensity), drift diffusion schedule (**ground truth**) |
| `demand_synthesis` | sector-level behavioural model (residential peaks / commercial plateau / industrial 24-7) with *emergent* crowd smoothing (sqrt-N law), weekly + seasonal cycles, volume-exact by construction |
| `hydraulics`       | direct wntr/EPANET run; unit-base pattern convention makes volume bugs structurally impossible |
| `sensing`          | sensor extraction + measurement-noise layer -> per-client CSVs  |
| `sim_validation`   | physics checks as pipeline nodes: hard checks raise, report ships with every dataset |
| `dependence_oracle`| topological + centralized statistical dependence ground truth   |

## The demand model — land use

A junction is a city block. Two attributes describe it:

* **`income`** {low, medium, high} — the building-standards mix, hence the mean
  consumption of one **residential** plot. Residential-only: income enters the
  demand level only through the residential intensity term.
* **`land_use`** {residential, mixed, commercial, industrial} — the block's mix
  of sector plots, driving both level and shape.

| sector | daily | weekly | night floor | weather |
|---|---|---|---|---|
| residential | three use peaks | weekend morning shifted + damped | 0.15 | full |
| commercial | work-hours plateau 08-18 | near-closed Sat/Sun | 0.03 | weak |
| industrial | 24/7 with day lift 06-22 | mild dip | 0.85 | none |

A plateau is a boxcar convolved with the same Gaussian start-time jitter the
residential peaks use, so crowd smoothing stays *derived* from 1/sqrt(N) rather
than tuned, and plateau sectors inherit it for free. Industrial 24/7 draw
raises the minimum night flow (the classic DMA diagnostic) while commercial
pushes it toward zero — residential->commercial and residential->industrial are
opposite-signed, physically interpretable drifts.

Level: `volume = anchor * calibration * (mean_plot_intensity / reference) ** beta`.
`scenario.beta` is the level dial — `0.0` is shape-only and volume-neutral,
`1.0` is the full effect. It is a **world** axis, so gridding it re-simulates
(see the `beta_sweep` study).

Worlds name their scenario with a 5-token code in district order A-E, each
token = income initial `{L,M,H}` + land-use initial `{R,M,C,I}` ('M' is
disambiguated by position). The shipped map is `LR_LM_LC_LR_LR`.

## Design invariants

1. **Volume exactness**: every node-month integrates exactly to its target;
   EPANET must reproduce it (checked to 1e-3 rel, observed ~5e-6).
2. **Peak-feasible anchor**: `anchor_scale` is calibrated so the worst case
   (seasonal peak + fully drifted district) stays inside the ABNT NBR 12218
   pressure band. On the shipped map at `beta=0.35`: network peak 661 L/s
   against a capacity frontier near 1050-1200 L/s, min pressure 30.9 mca, max
   48.5, 100% of node-hours in band. Pressures carry real signal.
3. **Emergent smoothing**: aggregation over plots produces peak factors from
   the sqrt-N law; nothing is injected. Note the node peak factor is NOT
   monotone in commercial content (residential 2.16, commercial 2.05, mixed
   1.83) — a mixed block fills the midday trough between the residential
   peaks. Use the night/day ratio as the composition proxy.
4. **Shape survives the scaler; level does not**: per-client MinMax fitted on
   the commissioning months is affine, so it discards demand LEVEL and keeps
   weekly gates and night floors. A residential->commercial drift shifts the
   scaled weekly profile by 0.84 (1 - corr) at `beta=0` — with no level change
   at all — and identically at `beta=0.5`. Level is what threatens hydraulic
   feasibility; shape is what carries the regime information.
5. **Ground truth is a product**: drift schedule, boundary/closure table,
   topology features, and the centralized dependence oracle are catalog
   artifacts (`gt_*`), never mixed with training data.
6. **Reproducibility**: keyed RNGs — any single (node, month) is regenerable
   in isolation; same seed, same dataset, bit for bit.

## Dependence battery (the centralized oracle)

`dependence_oracle` scores every district pair with four method tiers, each
under circular-shift surrogate significance: **T1 linear** (Pearson, Spearman,
max-lagged xcorr + lead-lag, precision-matrix partial correlation), **T2
nonlinear** (distance correlation, kNN mutual information), **T3 directional**
(Granger F, Convergent Cross Mapping), **T4 representation space**
(MiniRocket feature trajectories on synchronized windows -> RV coefficient &
trajectory dCor — the FL-deployable tier: fixed kernels give every client the
same space with zero training/communication). `gt_structure_recovery` scores
each method's pair ranking against the physical topology; the dependence dial
(`coupling.variant`) sweeps ground truth from fully coupled to isolated, with
connectivity-preserving partial closures.

## The story notebook

`notebooks/simulator_story.ipynb` (executed, English, 16 exported figures in
`data/08_reporting/figures/`) walks the full narrative in six acts: the world,
the behavioural model, the physics, drift, dependence (dial sweep + battery
scoreboard + the exogenous-confound demonstration), and the federated client
view. It reads only catalog artifacts — no simulation logic is duplicated.

## Federated learning (the backbone)

Three pipelines refactor the BEPE FL stack (`kedro run --pipeline fl`):

* **fl_preprocessing** — client-local: 2h aggregation, MinMax scaling fitted
  on the reference/commissioning months ONLY (the legacy code fit on the full
  series — future leakage that also compressed the drift signal), vectorized
  windowing, majority month labels from the simulator's authoritative month
  column.
* **fl_training** — in-process FPL (Huang et al. CVPR'23) over AER
  autoencoders (Wong et al. 2022): per round, local AER triple-MSE +
  hierarchical prototype loss (alpha*InfoNCE + (1-alpha)*MSE-to-mean,
  vectorized), per-month prototypes extracted in a clean eval pass with final
  local weights, FINCH cosine clustering server-side, FedAvg. Correctness
  fixes vs legacy: latent = encoder bottleneck (was the decoder hidden
  state), eval-mode prototype extraction (was mid-training batches), InfoNCE
  bookkeeping (None-crash / wrong divisor / .item() on float), full seeding.
* **drift_detection** — delta_first / delta_roll cosine distances on the
  final-round prototype trajectory; threshold mu_ref + k*sigma_ref with
  persistence; evaluated against `gt_drift_schedule`.

Dependence plug-in surfaces: `latent_trajectories` shares the
`feature_trajectories` schema (district, kind, window, f0..), so the
dependence battery's Tier-4 methods run on AER latent space unchanged;
`prototype_history` exposes per-round per-client per-month prototypes.

**The motivating result, reproduced:** on the coupled baseline the detector
is contaminated by hydraulic spillover (drifted district ranks 2nd, 4 false
alarms, separation 1.08) while on the `isolated` variant it is clean (rank
1, 0 false alarms, separation 3.59). Closing that gap on the coupled network
is the dissertation's dependence-detection objective.

## Dependence detection & quantification (federated)

`kedro run --pipeline dependence_detection` measures inter-client dependence
at three communication budgets: **Level P** (prototype displacements — what
FPL already sends, zero marginal cost), **Level T** (per-window latents in
the shared FedAvg encoder space; month+phase+weekday residualized,
client-locally), **Level R** (the centralized oracle battery = upper bound).
Quantity: the RV coefficient ([0,1], shared co-inertia) with trajectory
dCor, PC1 lead-lag, and **partial RV** (conditioned on all other clients:
direct vs mediated coupling); circular-shift surrogates + BH-FDR q-values;
communication cost recorded per level. Validated on the dial: RV 0.64
(baseline) -> 0.40 (partial-50) -> 0.008 (isolated), significant pairs
10/10 -> 10/10 -> 0/10. Headline: **partial RV recovers the physical
boundary structure at AUROC 1.00**, above every centralized raw-data
method; Level P saturates — the budget axis is real. See
`notebooks/federated_story.ipynb`.

## Drift attribution — the corrector ladder (C0..C4)

`kedro run --pipeline drift_attribution`: C0 uncorrected, C1 peer z-score,
C2 median common-mode removal (+ the `drift_attribution` decomposition
delta_total = delta_common + delta_local), C2b leave-one-out median, C3
ridge peer-prediction (full / partial-RV-masked), C4 the Design-B loop
(rank-1-with-gains + client-sparse alternating decomposition, per-iteration
diagnostics in `loop_diagnostics`). All scored by the same harness.

**Finding (the chapter's pivot):** on planted worlds with exogenous or
gain-heterogeneous common modes, C2/C3/C4 restore the drifted client to
rank 1 and C4's weights identify the drifter (tests pin this). On the real
coupled network, NO analytic corrector succeeds — because the drifted
district's local signature and the network's common response are the same
physical event seen from two places: the sparse and low-rank components are
maximally COHERENT, violating the identifiability condition every
symmetric decomposition needs. What separates originator from receiver is
not displacement geometry but signatures (flow-vs-pressure composition,
direction of influence, timing) — the documented motivation for the
learned detector (label factory + C4-learned) and the episode-retrieval
layer (C5).

## Label factory + learned detectors

`kedro run --pipeline label_factory` generates randomized worlds (seed x
coupling x drift origin/target; cache-or-run, incremental) and extracts
`labeled_pairs` / `labeled_clients` with physical ground truth as labels.
`kedro run --pipeline automl` trains the learned dependence detector and
drift attributor under LEAVE-ONE-WORLD-OUT validation (small HistGB grid;
escalate the search only after this baseline is beaten). The 2-world demo
correctly reports chance AUROC — scale `n_worlds` (each ~3 min) before
drawing conclusions; hold out coupling variants and, later, a second
network topology.

## Personalization — drift + dependence -> clusters

`kedro run --pipeline personalization` fuses the two mechanisms into
client groups for personalized federated models. Domain similarity is built
in two deconfounded channels — drift-signature cosine (how domains move) and
common-mode-removed prototype similarity (the C2 median operator) — while
quantified DIRECT coupling (partial-RV, q-gated) enters as a *discount*, not
a similarity term: coupled pairs' apparent similarity is the least
trustworthy as domain evidence, so it is down-weighted (`s_effective =
s_fused * (1 - beta*coupling)`). This is the one-pass deconfounding; the
Design-B loop iterates it. `similarity_matrix` is the full audit
(s_drift, s_domain, s_fused, coupling, s_effective); `cluster_assignments`
routes each client to a personalization group (FINCH, the same routine the
FPL server uses). Server-side only — no new client communication.

## Provenance

Adapted from the BEPE research codebase (drift detection via federated
prototypes on this same Graeme network) with three correctness fixes over the
legacy simulation: the sum-to-1 pattern normalisation (delivered volumes ~24x
low), the hidden EPANET demand multiplier (0.2), and units-mixing in pattern
smoothing. See `conf/base/parameters.yml` for every physical assumption.

The `density` axis was replaced by land use. Density was volume-neutral by
construction (node volume was a pure function of income, and the month series
was renormalised onto it) and its hydraulic direction was backwards: higher
density widened peak-time dispersion, flattening the aggregate at constant
volume and so RAISING minimum pressure — a densification drift relieved the
network instead of stressing it. Land use drives level and shape together, and
puts the regime signature in the temporal channel the FL scaler provably
preserves. Every world hash changed; `data/09_experiments/worlds/` must be
re-simulated.

## Experiments engine

Declarative studies (seed replication, drift-origin sweeps, consumption-map
sweeps, the land-use level sweep, the label factory) run through one engine:
worlds are simulated once
and cached by content hash of their effective configuration; runs (FL
training + analysis) reuse cached worlds via hardlinks, so replication costs
FL only. Every world and run writes a `manifest.json` (resolved config,
package versions, timings, status) — the papertrail.

```bash
python -m fedwater.experiments list
python -m fedwater.experiments run d0_replication --n-jobs 4   # or --dry-run
python -m fedwater.experiments status d0_replication
```

Studies live in `conf/base/experiments.yml`; results land under
`data/09_experiments/<study>/` (`runs.parquet` index + per-group harvest
tables). Analysis retrieval:

```python
from fedwater.experiments import load_index, load_group
idx = load_index("d0_replication")
dep = load_group("d0_replication", "dependence_summary",
                 where={"level": "T", "method": "partial_rv"})
```