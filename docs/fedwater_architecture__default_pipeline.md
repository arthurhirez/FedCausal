kedro run                                    # __default__: 22 nodes, ~2.5 min → validated world + oracle


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
