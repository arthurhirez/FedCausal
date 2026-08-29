# fedwater — pipeline architecture reference

*Repo: `arthurhirez/FedDependency` (package `fedwater`), Kedro 1.5, Python ≥3.10.
Verified by cloning the repo, reconstructing the missing raw inputs, and running
every pipeline end to end (see "Verification" at the bottom).*

---

## 1. What the project is

A Kedro project that **generates** a physically validated water-distribution dataset
(EPANET/wntr) with drift and dependence **ground truth as first-class artifacts**, then
runs a **federated learning** stack on it to study one question:

> On a hydraulically coupled network, a client's own drift signal is contaminated by its
> neighbours' drift arriving through shared pipes. Can inter-client dependence be
> detected and quantified *federatedly*, and then used to de-confound drift detection?

Everything is one repo with two halves that meet at one file format: the simulator writes
`data/07_model_output/clients/District_*.csv`, and the FL stack reads nothing else.

## 2. Ingestion — the two raw inputs

`data/**` is **gitignored**. Only these two files are true inputs; everything else in
`data/` is generated.

| Catalog entry | Path | Type | Content |
|---|---|---|---|
| `graeme_network` | `data/01_raw/Graeme.inp` | `fedwater.datasets.WntrNetworkDataset` (custom) | EPANET .inp — 113 junctions, 164 pipes, 1 reservoir (node `114`, head 55 m), units LPS, Hazen-Williams. Carries a hidden `DEMAND MULTIPLIER 0.2` that `network_prep` overrides. |
| `districts` | `data/01_raw/districts_graeme.yml` | `kedro_datasets.yaml.YAMLDataset` | `{districts: {District_A: [node ids…], …}}` — must **exactly partition** the 113 junctions; `validate_partition` raises otherwise. |

Everything a new notebook or paper implementation needs is downstream of these, so read
from the catalog, not from `01_raw`, unless you specifically need the network object.

Kedro's data-layer convention is used consistently:

```
01_raw          the two inputs above
02_intermediate wn_variant.pkl, portfolios_t0, demand_series, pressures, flows, demands_simulated
03_primary      gt_* ground truth, sensor_series, feature_trajectories, income_factors, landuse_factors, assignments_timeline
04_feature      fl_windows.pkl, fl_scalers
05_model_input  world_specs, labeled_pairs, labeled_clients   (label factory)
06_models       fl_models.pkl, automl_models.pkl
07_model_output clients/*.csv, prototype_history, latent_trajectories, drift_signals, corrected_drift_signals
08_reporting    every *_report / evaluation / scoreboard CSV + notebook figures
```

## 3. Pipelines and how to run them

`pipeline_registry.register_pipelines()` auto-discovers pipelines and builds
`__default__` as the **sum of the simulation-side pipelines only** — the FL-side ones are
explicitly excluded and run by name.

```bash
kedro run                                    # __default__: 22 nodes, ~2.5 min → validated world + oracle
kedro run --pipeline fl                      # fl_preprocessing + fl_training + drift_detection, ~6 min
kedro run --pipeline dependence_detection    # ~9 min
kedro run --pipeline drift_attribution       # seconds
kedro run --pipeline personalization         # seconds
kedro run --pipeline label_factory           # n_worlds × ~3 min (shells out to `kedro run` per world)
kedro run --pipeline automl                  # seconds
kedro run --params "coupling.variant=isolated"   # the dependence dial
pytest -q                                    # 86 tests
kedro run --pipeline experiments              # study runner (see conf/base/experiments.yml)
```

**Run order matters and is not enforced across invocations**: `fl` needs `__default__`'s
outputs, `dependence_detection` needs `fl`, `drift_attribution` needs both, and
`personalization` needs `fl` + `dependence_detection`.

### The simulation chain (`__default__`)

| # | Pipeline | Node(s) | In → Out |
|---|---|---|---|
| 1 | `network_prep` | `configure_network` | `graeme_network` → `wn_configured` *(memory)* — pins demand multiplier to 1.0, model `DD`, duration = n_months×days×24 h, 1 h timesteps |
| | | `validate_partition` | → `partition_report`; **raises** on overlap/missing/unknown nodes |
| | | `apply_coupling` | → `wn_variant`, `gt_boundaries` — **the dependence dial** |
| 2 | `urban_scenario` | `build_income_factors` | `params:buildings` → `income_factors`. `mean_unit_m3_month` **is** the residential plot intensity, which is what keeps income a residential-only attribute |
| | | `build_landuse_factors` | `income_factors` + `params:land_use` → `landuse_factors` — level factor per (income, land use) = (mean plot intensity / reference) ** `scenario.beta` |
| | | `build_portfolios` | → `portfolios_t0` — per node: **sector cohorts** (residential / commercial / industrial); anchored at `inp_base × anchor_scale`; global calibration scales **population**, not per-plot consumption, and is a diagnostic, never a gate |
| | | `build_drift_schedule` | → **`gt_drift_schedule`** — diffusion on the target district's subgraph from `seed_node`; carries `to_income` and `to_land_use` per row |
| | | `evolve_assignments` | → `assignments_timeline` (month × node × sector cohort) |
| 3 | `demand_synthesis` | `synthesize_demands` | → `demand_series_raw` *(memory)*: hourly L/s per node, **volume-exact by construction** |
| | | `apply_drift_ramp` | → `demand_series` — 42-day linear blend at each node's switch, **clamped** against the remaining horizon |
| 4 | `hydraulics` | `run_hydraulics` | `wn_variant`+`demand_series` → `pressures`, `flows`, `demands_simulated` (EPANET via wntr) |
| 5 | `sensing` | `extract_sensor_series` → `add_measurement_noise` → `package_client_datasets` | → `sensor_series`, **`client_datasets`** (PartitionedDataset, one CSV per district) |
| 6 | `sim_validation` | 4 checks + `compile_validation_report` | → `validation_report`; V1–V3 **raise**, V4–V7 warn |
| 7 | `dependence_oracle` | `topology_features`, `minirocket_trajectories`, `dependence_battery`, `structure_recovery` | → `gt_topology`, `feature_trajectories`, `gt_dependence_battery`, `gt_structure_recovery` |

Memory-only datasets (not in the catalog, gone after the run): `wn_configured`,
`demand_series_raw`, `sensor_series_true`, `report_mass`, `report_pressure`,
`report_consumption`, `report_peaks`.

### The federated chain

| Pipeline | Node | In → Out |
|---|---|---|
| `fl_preprocessing` | `preprocess_clients` | `client_datasets` → `fl_windows` (pickle), `fl_scalers`, `fl_prep_report` |
| `fl_training` | `train_federated` | `fl_windows` → `fl_models`, `prototype_history`, `global_prototype_history`, `latent_trajectories`, `fl_training_log` |
| `drift_detection` | `compute_drift_signals` → `evaluate_drift` | `prototype_history` + `gt_drift_schedule` → `drift_signals`, `drift_report` |
| `dependence_detection` | `residualize_latents` → `federated_dependence` → `evaluate_dependence` | `latent_trajectories`, `prototype_history`, `gt_topology`, `gt_dependence_battery` → `latent_trajectories_residualized`, `client_dependence`, `dependence_evaluation` |
| `drift_attribution` | `apply_correctors` → `evaluate_correctors` | `prototype_history`, `latent_trajectories_residualized`, `client_dependence` → `corrected_drift_signals`, `drift_attribution`, `loop_diagnostics`, `corrector_ladder_report` |
| `personalization` | `build_personalization_clusters` | `prototype_history`, `drift_signals`, `client_dependence` → `cluster_assignments`, `similarity_matrix` |
| `label_factory` | `build_world_specs` → `generate_labeled_worlds` | randomized worlds run as **subprocess `kedro run`** in `/tmp/fedwater_worlds/world_N` → `labeled_pairs`, `labeled_clients` |
| `automl` | `train_learned_detectors` | → `automl_models`, `automl_report` (HistGB, leave-one-world-out CV) |

## 4. Data shapes at 24 months (measured)

| Artifact | Shape | Notes |
|---|---|---|
| `demand_series` | 17 280 × 113 | 17 280 = 24 mo × 30 d × 24 h; `month` + 112 node columns (node `110` has zero base demand → no portfolio → no column) |
| `pressures` / `flows` | 17 280 × 115 / × 165 | mca / L/s |
| `sensor_series` | 432 000 × 7 | long: 17 280 × 25 sensors; `value` (true) and `observed` (noisy) |
| `client_datasets` | 5 × (17 280 × 7) | `timestamp, month, p_*, p_*, q_*, q_*, q_*` |
| `fl_windows[c]` | (7 614, 84, 5) float32 | 2 h aggregation → 8 640 steps → 84-step (7-day) windows, stride 1 |
| `prototype_history` | 1 800 × 63 | 15 rounds × 5 clients × 24 months, latent dim 60 = 2 × `lstm_units` |
| `latent_trajectories` | 38 070 × 64 | per-window latents, `feature_trajectories` schema |
| `gt_dependence_battery` | 240 × 9 | 2 kinds × 10 pairs × methods/directions |
| `client_dependence` | 60 × 10 | Level T (4 methods × 10 pairs) + Level P (2 × 10) |

## 5. The demand model — land use

A junction is a **city block**, not a building. Two attributes describe it:

- **`income`** ∈ {low, medium, high} — the building-standards mix, hence the mean
  consumption of one **residential** plot. Income is residential-only: it enters the
  demand level *only* through the residential intensity term.
- **`land_use`** ∈ {residential, mixed, commercial, industrial} — the block's mix of
  sector plots. Drives both the **level** (via a plot-intensity table) and the **shape**
  (via per-sector daily and weekly signatures).

Three sectors, each with its own signature:

| sector | daily | weekly | night floor | weather |
|---|---|---|---|---|
| residential | three Gaussian use peaks | weekend morning shifted + damped | 0.15 | full |
| commercial | work-hours plateau 08–18 | near-closed Sat/Sun (0.15) | 0.03 | weak |
| industrial | 24/7 with a day lift 06–22 | mild dip (0.90) | 0.85 | none |

A plateau is a boxcar convolved with the *same* Gaussian start-time jitter the residential
peaks use — a difference of two erfs — so crowd smoothing stays **derived** from `1/√N`
rather than tuned, and the plateau sectors inherit it for free.

**Level.** `volume = anchor × calibration × level_factor`, where
`level_factor = (mean_plot_intensity / reference) ** beta` and
`mean_plot_intensity = Σ_s plot_share[s] × intensity[s]`. `intensity[residential]` is not
free — it is `mean_unit_m3_month(income)` from the standards table. `scenario.beta` is the
level dial: `0.0` is shape-only and volume-neutral (reproducing the pre-refactor hydraulic
loading exactly), `1.0` is the full level effect. It is a **world** axis, so gridding it
re-simulates.

### Why this replaced the `density` axis

Density was **volume-neutral by construction** — node volume was a pure function of
income, and the month series was renormalised onto it — so it could only ever move the
shape. Worse, its hydraulic direction was backwards: higher density widened the peak-time
dispersion, *flattening* the aggregate at constant volume and therefore *raising* minimum
pressure. A densification drift relieved the network instead of stressing it. And because
`total_units` was back-solved from volume, densification could not add population at all.

Land use fixes all three, and adds the property the project actually needs (§6.4).

## 6. The physics and modelling invariants

Read `conf/base/parameters.yml` — every physical assumption lives there, commented.

1. **Volume exactness.** Each node-month series is rescaled so `Σ(L/s × 3600 s)` equals
   the target monthly volume exactly. `hydraulics` then sets every junction's
   `base_value = 0.001 m³/s` (1 L/s) and makes the *pattern* the L/s series, so EPANET
   reproduces it identically — the "1/24" and "×0.2" bug classes become structurally
   impossible. Measured V2 error 2.7e-6; V1 mass balance 2.4e-10.
2. **Peak-feasible anchor.** `hydraulics.anchor_scale = 0.05` with `beta = 0.35` on the
   shipped map `LR_LM_LC_LR_LR`: measured network peak **661 L/s** and mean 290 L/s
   against a capacity frontier at roughly **1050–1200 L/s** (where minimum pressure
   crosses the 5 mca floor) — about 60 % headroom. Measured min pressure **30.91 mca**,
   max **48.49 mca**, 100 % of node-hours inside the ABNT NBR 12218 band [10, 50].
   Overpressure is not the binding constraint and never was.
3. **Emergent crowd smoothing.** The `1/√N` law, not injection: a small cohort is noisier
   than a large one for every sector, plateau sectors included. Note that the resulting
   node peak factor is **not monotone** in commercial content — measured residential
   2.163, commercial 2.046, mixed 1.827. A ~40/60 residential/commercial blend fills the
   midday trough between the residential peaks and flattens the aggregate; push the
   commercial share high enough and it climbs back toward pure commercial's own value.
   Use the night/day ratio, not the peak factor, as the composition proxy.
4. **Shape survives the scaler; level does not.** This is the load-bearing invariant.
   The FL scaler is a per-client MinMax fitted on `fl.preprocessing.reference_months`, so
   it is affine — it destroys level and preserves a weekday/weekend gate or a night-floor
   change. Measured: a District_D residential→commercial drift shifts the MinMax-scaled
   weekly profile by **0.843** (1 − corr, before vs after) at **both** `beta = 0.0` and
   `beta = 0.5`. Identical, and necessarily so: raising `beta` multiplies the post-switch
   series by a constant, and Pearson correlation is invariant under affine transforms.
   Level is what threatens hydraulic feasibility; **shape is what carries the regime
   information**, and a low `beta` buys nearly all the legibility at almost none of the
   risk. Pinned by `test_shape_shift_survives_the_scaler_and_is_beta_independent`.
5. **Calibration is a diagnostic, never a gate.** It renormalises month-0 total volume
   onto the anchor total, rescaling every node by the same constant — hydraulically a
   no-op. Measured 0.888 on the shipped map. The pre-refactor code raised outside
   [0.5, 2.0], which short-circuited the feasibility search before it could reach the
   real boundary (the pressure floor). The number is recorded as V7; nothing raises.
6. **Ground truth is a product.** `gt_*` artifacts exist to evaluate methods, never to
   train them.
7. **Month-aware residualization is mandatory.** Every dependence statistic runs on
   residuals of the per-(month × hour × weekday) profile. Naive hour-of-day residuals
   make *isolated* districts look dependent (|r| 0.50 vs 0.045) because they share
   seasonality — the exogenous-driver confound.

### Consumption maps

Worlds name their scenario with a compact 5-token code in district order A–E, each token
= income initial + land-use initial: income over `{L, M, H}`, land use over
`{R, M, C, I}`. `M` is disambiguated by position (medium income vs mixed land use). The
shipped map is `LR_LM_LC_LR_LR`. A pre-refactor density-era map such as `LL_LM_LH_LL_LL`
now **fails loudly** in `decode_map`, because `L` is not a land-use initial.

## 7. Non-obvious implementation details

Each of these was found by a failure and is commented at its site in `src/`.

1. **Cohorts mix by VOLUME share, not plot count.** Sectors differ ~10× in intensity, so
   plot-weighting drowns the commercial signal under a residential majority.
2. **Each cohort is normalised over the WHOLE MONTH, not per day.** Per-day normalisation
   rescales every day to the same total and erases commercial's weekend collapse before
   it reaches the mix.
3. **Zero-volume nodes are dropped in the cohort builder** and again in the synthesiser.
   A zero-base-demand trunk junction (node `110`) otherwise yields `0/0` in the volume
   weighting; the NaN reaches the EPANET pattern and the solver dies with `Error 200`.
4. **Two node sets.** `pnodes` = every district junction (pressure is checked at trunk
   nodes too); `dnodes` = junctions that carry a demand column. Conflating them raises
   `KeyError: ['110'] not in index`.
5. **Cohorts below `land_use.min_plots` are dropped and the remainder renormalised**, so
   node volume is preserved exactly rather than leaking with the dropped cohort.
6. **Seasonality and the weather factor are volume-weighted across sectors** — industrial
   process water is weather-insensitive, residential is not.
7. **`drift.warmup_months` must be ≥ 1.** `apply_drift_ramp` blends against the month
   *before* the switch. Enforced in `build_drift_schedule` and in `validate_world`.
8. **`drift_ramp_days` is clamped** to `(n_months − last_switch_month) × days_per_month − 1`
   inside `apply_drift_ramp`. `growth_chance` and `max_neighbors_per_month` decide how
   late the front finishes, so any horizon change can silently re-break this.
9. **The drift front advances one BFS ring per month.** `max_neighbors_per_month` limits
   its width, not its depth, so a short horizon leaves the district partly converted.
10. **Weekends come from a global day counter** (`(month × days + d) % 7`), so the weekly
    cycle is continuous across month boundaries at any `days_per_month`. At 30 a month
    holds 8 or 9 weekend days depending on phase; volume is renormalised exactly, so this
    redistributes the monthly shape rather than leaking volume.
11. **Per-district sensitivity is real and is not explained by distance from the source.**
    The same drift at the same `beta` differed ~4× across districts in the stress
    ablation: District_B is the stable weak link, District_E is state-dependent (fine
    alone, fails once District_D is also commercial). District_E is the *farthest* from
    the source and among the safest — do not use hop count as the explanation.

## 8. The dependence dial

`coupling.variant` ∈ `baseline | partial | isolated` in `network_prep.apply_coupling`:

- **baseline** — Graeme as-is, single reservoir, all boundaries open (max coupling).
- **partial** — closes `close_fraction` of inter-district pipes, **connectivity-preserving**:
  a closure is kept only if every junction still reaches a source. From a single source
  this saturates (5/9 boundaries at `close_fraction=1.0`); actual closures are recorded
  in `gt_boundaries`.
- **isolated** — closes all boundaries **and** adds one reservoir per district.

## 9. Verification performed — PRE-REFACTOR (density era)

> **The numbers in this section describe the density-era simulator and no longer
> reproduce.** Every world hash changed with the land-use refactor, so
> `data/09_experiments/worlds/` must be re-simulated. The section is kept because the
> *method* — inverting the demand chain to recover base demands, and classifying nodes to
> reconstruct the partition — is still the way to audit a world, and because the peak-factor
> reconstruction trick is exactly what §6.3 now says not to rely on. Current measured
> values are in §6.1–6.5 and §10.

`data/01_raw` is not in the repo, so I reconstructed both inputs from the committed
`temp.inp` (an EPANET scratch file from a real baseline run, with the injected demand
patterns intact):

- **`Graeme.inp` base demands recovered exactly.** Inverting the demand-synthesis chain
  (pattern integral → monthly volume ÷ seasonal ÷ keyed-lognormal draw → anchor →
  `base = anchor / (anchor_scale × 86400 × 30)`) lands on a clean 2-decimal L/s grid to
  within 5e-7. This works because all five districts are income `low`, which makes the
  global calibration constant exactly `1 / income_factor(low)`.
- **District partition reconstructed** by classifying nodes by their month-0 peak factor
  (perfectly trimodal → B = medium ×29, C = high ×18) and identifying District D as the
  24 nodes whose volume triples over the horizon. All 10 configured sensor nodes land in
  their own district. Sizes: A 12, B 29, C 18, D 24, E 30.
- **The reconstruction reproduces the notebooks' numbers to 6 decimals** — validation
  report identical (V3 floor 11.942286, peak factors 2.281375 / 1.635668 / 1.455698),
  24 drifted nodes over months 2–19.
- **All 51 tests passed** at the time (now 86). Full run: `__default__` 151 s, `fl` 376 s,
  `dependence_detection` 518 s, `drift_attribution` and `personalization` seconds.
- **Headline results reproduced**: coupled baseline drift detection contaminated
  (4 false alarms, separation 1.06, drifted district ranks 3rd — README says 2nd);
  RV 10/10 significant pairs at Level T; no analytic corrector reaches rank 1
  (best C3 at rank 2) — the "coherence wall".

*Caveat: the A/E split of the low-density nodes is a graph-proximity guess, so the
reconstructed world is not bit-identical to the author's. Use the real
`data/01_raw` files.*

## 10. Findings to act on before building anything new

**Reproducibility hole — RESOLVED and pinned.** The fix landed and
`test_no_builtin_hash_seeding_in_pipelines` now fails the suite if builtin `hash((…))`
reappears anywhere under `src/fedwater/pipelines/`. Original diagnosis retained below,
since the failure mode is subtle enough to be worth recognising again.

**Reproducibility hole (verified, high impact).** `sensing.add_measurement_noise` keys its
RNG on `hash(sensor)` and `dependence_oracle`/`dependence_detection` key their surrogate
RNGs on `hash((district_a, district_b, kind))`. Python salts string/tuple hashing per
process unless `PYTHONHASHSEED` is set, so **the same seed produces a different client
dataset and different p-values on every run**. Measured: three runs of
`add_measurement_noise(seed=42)` on identical input gave observed-value sums 856968.38 /
856991.23 / 856983.29. This contradicts README invariant 5 ("same seed, same dataset, bit
for bit") and will silently break any before/after comparison against a paper baseline.
Fix: replace `hash(x)` with a stable digest (`zlib.crc32(str(x).encode())` or
`int.from_bytes(hashlib.blake2b(…, digest_size=4).digest())`).

**Fragile headline numbers.** `partial RV recovers the physical boundary structure at
AUROC 1.00` rests on **10 district pairs in one world**. On my reconstructed partition the
same code gives 0.64, and zero partial-RV pairs clear q < 0.05. Treat AUROC on 10 pairs as
a direction, not a result, and report it across the dial and multiple worlds.

**A claim its own output contradicts.** `simulator_story.ipynb` Act V says "on pressure …
only the MiniRocket representation space recovers structure", but the printed scoreboard
in the same cell gives `minirocket_rv` 0.48 and `minirocket_dcor` 0.48 on pressure — chance.
Worth re-checking before it reaches the dissertation.

**Silent degradation in `personalization`.** `_coupling_matrix` zeroes any pair with
`q_value ≥ coupling_q_threshold`. When no pair is significant (which happened in my run)
the coupling matrix is all-zero, `s_effective == s_fused`, and the entire dependence
channel is inert — with no warning. Also observed: `s_drift` was ≈0.99 for every pair
(all clients' `delta_first` trajectories share the common-mode shape), so the fused
similarity is driven almost entirely by `s_domain`. Both deserve an assertion.

**Kedro config merge is destructive.** `conf/local/parameters.yml` replaces whole
top-level keys rather than deep-merging. Any override must carry the **complete** block
(this is why `label_factory._world_override` and the notebook sweeps rebuild full blocks
from the loaded base params). Easy to get wrong when adding a paper variant.

**Repo hygiene.** `temp.inp` (8 MB, 323 k lines) and `temp.rpt` at the repo root are
EPANET scratch output from a wntr run and are explicitly whitelisted in `.gitignore`.
They are also, currently, the only copy of the network in the repo.

**Smaller items.** `apply_correctors`' signature says `-> tuple[DataFrame, DataFrame]` but
returns three. `label_factory.scratch_dir` is hardcoded `/tmp/...` (POSIX-only). The
`personalization` pipeline is absent from `pipeline_story.ipynb`. `automl`'s 2-world demo
correctly reports chance AUROC — do not read it as a result.

## 11. Where to plug a paper implementation in

The natural seams, in increasing order of intrusion:

1. **Read-only notebook off the catalog.** Load `client_datasets` (per-client CSVs),
   `sensor_series`, or `latent_trajectories` and score against `gt_drift_schedule` /
   `gt_topology` / `gt_dependence_battery`. Nothing upstream changes. This is what the
   three story notebooks do, and it matches the project brief ("read the ingestion data
   in the correct directory, implement the paper solution in a notebook").
2. **Reuse the evaluation harnesses.** `drift_detection.nodes.evaluate_drift` and
   `dependence_oracle.nodes.structure_recovery` are plain functions that take tidy frames
   — a new method scored through them is directly comparable to every existing rung.
3. **Tier-4 plug-in surface.** `latent_trajectories` deliberately shares the
   `feature_trajectories` schema (`district, kind, window, f0…fD`), so anything written
   against one runs on the other unchanged.
4. **New Kedro pipeline.** Add `src/fedwater/pipelines/<name>/{pipeline,nodes}.py` and
   catalog entries; `find_pipelines()` picks it up automatically. Add its name to the
   exclusion list in `pipeline_registry` if it should not join `__default__`.
