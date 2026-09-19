"""End-to-end D-Town run: capacity -> assignment -> drift -> demands -> hydraulics.

Reference driver for the three modules. Every number this prints was measured on
D-Town; a divergence is a signal, not noise.
"""
import warnings
from pathlib import Path

import pandas as pd
import wntr
import yaml

import wdn_capacity as wc
import wdn_hydraulics as wh
import wdn_population as wp
from wdn_demand import ShapeParams

warnings.simplefilter("ignore")

NET = "Dtown.inp"
DISTRICTS_YML = "districts_dtown.yml"
OUT = Path("out")
OUT.mkdir(exist_ok=True)

# ---- inputs -------------------------------------------------------------
wn = wntr.network.WaterNetworkModel(NET)
districts = pd.Series({n: k for k, v in
                       yaml.safe_load(open(DISTRICTS_YML))["districts"].items()
                       for n in v}, name="district")
dm = wc.DemandModelConfig()

# ---- capacity -----------------------------------------------------------
# Dispatches on the network: storage -> dynamic, gravity -> steady state.
# D-Town: regime=storage, lambda*=1.068, capacity=282.0 L/s, no peak division.
cap = wc.measure_capacity(wn, wc.CapacityConfig())
print("capacity: regime=%s lambda*=%.3f %.1f L/s peak_division=%s"
      % (cap["regime"], cap["lambda_star"], cap["capacity_lps"],
         cap["apply_peak_division"]))

# ---- assignment ---------------------------------------------------------
# anchor='demand' makes month 0 reproduce the .inp spatial distribution.
assignment = wp.assign_landuse(wn, districts, wp.AssignmentConfig(seed=20), dm)
anchor = wp.anchor_population(assignment, cap["capacity_lps"], dm,
                              cap["apply_peak_division"])
print("P_max = %.0f PE" % anchor["population_capacity"])

# ---- scenario -----------------------------------------------------------
horizon = wp.HorizonConfig(n_months=40, days_per_month=30)
scenario = wp.ScenarioConfig(
    horizon=horizon,
    growth=wp.GrowthConfig(initial_capacity_fraction=0.40,
                           logistic_rate_per_month=0.045,
                           district_rate_spread=0.5, seed=20),
    noise=wp.NoiseConfig(sigma_district=0.06, sigma_common=0.03,
                         tau_h=48.0, seed=20),
    seasonal=True,
    drifts=(wp.DriftSpec("landuse_conversion",
                         districts=("District_A", "District_C"),
                         start_month=2, end_month=19, ramp_days=10.0,
                         magnitude=0.65, from_category="residential",
                         to_category="industrial"),),
)

# districts= is the FULL partition: the BFS needs the zero-demand junctions,
# which are District_A's articulation points.
schedule = wp.build_drift_schedule(wn, assignment, scenario.drifts, horizon,
                                   districts=districts)
traj = wp.synthesize_demands(wn, assignment, schedule, ShapeParams(), dm,
                             scenario, anchor["population_capacity"])
demand_series = traj["demand_series"]

# ---- pre-solve validation ----------------------------------------------
# Expect pearson_r ~0.9996 if the anchor worked; ~0 means it did not.
print(wp.check_anchor(demand_series, assignment, horizon).round(4)
      .to_string(index=False))
load = wp.validate_demands(demand_series, assignment, cap["capacity_lps"],
                           horizon)
if load.breach.any():
    raise SystemExit("mean demand exceeds capacity in months %s; retune before "
                     "solving" % load.month[load.breach].tolist())

# ---- hydraulics ---------------------------------------------------------
# Monolithic. ~3 GB peak and ~4 min at 40 months on this network. Under a
# memory ceiling pass block_months=10, overlap_days=28 and mask sim['seam_steps']
# -- blocking perturbs pump-switching phase and cannot be made exact.
run = wh.RunConfig(demand_model="PDD", required_pressure_m=15.0,
                   pressure_floor_m=15.0)
sim = wh.run_hydraulics(wn, demand_series, run)

validation = wh.validate_simulation(sim, wn, run)   # raises on V1/V2/V2b/V3
health = wh.health_report(sim, wn)
censoring = wh.censoring_report(sim)
print(validation.round(6).to_string(index=False))

# ---- persist ------------------------------------------------------------
assignment.to_csv(OUT / "assignment.csv")
schedule.to_csv(OUT / "gt_drift_schedule.csv", index=False)
traj["trajectory"].to_csv(OUT / "gt_trajectory.csv", index=False)
traj["noise"].to_csv(OUT / "gt_district_noise.csv", index=False)
load.to_csv(OUT / "load_validation.csv", index=False)
validation.to_csv(OUT / "validation_report.csv", index=False)
health.to_csv(OUT / "health_report.csv", index=False)
censoring.to_csv(OUT / "censoring_report.csv", index=False)
for k in ("pressures", "flows", "demands_delivered", "demands_requested",
          "tank_levels", "pump_status"):
    sim[k].to_parquet(OUT / ("%s.parquet" % k))
print("wrote %s" % OUT.resolve())
