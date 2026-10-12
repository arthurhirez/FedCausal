"""Checks on the worlds of one placement study, after `python -m fedwater.experiments run <study>`.

Usage (from the repo root):
    python check_placement_worlds.py 2b2256b1d4b7 e92927a56092 a1bede374cfc 9099527b8a62 c69aa8873093

Reads each world's clone outputs; nothing is simulated. Prints one block per check.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
ROOT = Path("data/09_experiments/worlds")
SWEEP = Path("notebooks/outputs/exact_sweep/selection_new.parquet")
hashes = sys.argv[1:]
assert hashes, "pass the world hashes"

W = {}
for h in hashes:
    c = ROOT / h / "clone"
    mf = json.loads((ROOT / h / "manifest.json").read_text())
    W[h] = {"clone": c, "flat": mf["world"], "status": mf["status"],
            "placement": pd.read_csv(c / "data/03_primary/sensor_placement.csv", dtype={"element": str}),
            "coverage": pd.read_csv(c / "data/08_reporting/sensor_placement/coverage.csv"),
            "verification": pd.read_csv(c / "data/08_reporting/sensor_placement/verification.csv"),
            "stacks": yaml.safe_load((c / "data/08_reporting/sensor_placement/stacks.yml").read_text()),
            "topology": pd.read_csv(c / "data/03_primary/gt_topology.csv")}

print("\n== 0. worlds")
print(pd.DataFrame([{"world": h, "status": w["status"], "network": w["flat"].get("network"),
                     "districting": w["flat"].get("districting"), "drift": w["flat"].get("drift_district"),
                     "n_months": w["flat"].get("n_months")} for h, w in W.items()]).to_string(index=False))

print("\n== 1. same sensors in every world of the partition (selection never reads the target world)")
key = ["district", "slot", "element"]
ref_h = hashes[0]
ref = W[ref_h]["placement"][key].reset_index(drop=True)
for h, w in W.items():
    same = w["placement"][key].reset_index(drop=True).equals(ref)
    print(f"  {h}: {'identical' if same else 'DIFFERS'} to {ref_h}")
print(ref.pivot(index="district", columns="slot", values="element").to_string())

print("\n== 2. same sensors as the sweep notebook (DEFAULT, same partition)")
if SWEEP.exists():
    nb = pd.read_parquet(SWEEP)
    f = W[ref_h]["flat"]
    nb = nb[(nb["network"] == f.get("network")) & (nb["partition"] == f.get("districting"))]
    nb = nb.rename(columns={"client": "district"})[key].sort_values(key[:2]).reset_index(drop=True)
    got = ref.sort_values(key[:2]).reset_index(drop=True)
    print("  identical" if nb.equals(got) else "  DIFFERS:\n" + nb.merge(
        got, on=key[:2], how="outer", suffixes=("_notebook", "_world")).to_string())
else:
    print(f"  {SWEEP} not found -- run the notebook first")

print("\n== 3. the stacks / exact layer the worlds used")
for h, w in W.items():
    s = w["stacks"]
    print(f"  {h}: selection {s['selection']['stack_hash']}  label {s['label']['stack_hash']}  "
          f"same_stack {s['same_stack']}  exact {Path(s['selection']['exact']).name}")

print("\n== 4. every slot filled by the rule (coverage)")
cov = W[ref_h]["coverage"]
cols = [c for c in ("district", "n_candidates", "n_transit_pool", "n_mixed_cand", "n_branch", "n_pool", "n_mixed",
                    "n_branch_fill", "short_branch", "short_transit", "transit_distance", "branch_jaccard") if c in cov]
print(cov[cols].round(3).to_string(index=False))

print("\n== 5. verification: the chosen set against each world's OWN drift (target world, sim seed != probe seed)")
rows = []
for h, w in W.items():
    v = w["verification"]
    st = v["status"].iloc[0] if "status" in v else "?"
    r = {"world": h, "drift": v["drift_district"].iloc[0] if "drift_district" in v else None, "status": st}
    if st == "ok":
        r["spearman(S_label, |g_obs|)"] = round(float(v["spearman_expected_vs_observed"].iloc[0]), 3)
        r["responded"] = int(v["responded"].sum())
        r["of"] = len(v)
        tr = v[v["slot_class"] == "transit"]
        r["transits responding"] = f"{int(tr['responded'].sum())}/{len(tr)}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
for h, w in W.items():
    v = w["verification"]
    if v["status"].iloc[0] == "ok":
        c = [x for x in ("district", "slot", "element", "filled_as", "expected_S_drift", "expected_dm_drift",
                         "observed_g", "responded") if x in v]
        print(f"\n  {h} (drift {v['drift_district'].iloc[0]})")
        print(v[c].round(3).to_string(index=False))

print("\n== 6. client CSVs: same columns in every client, flow slots only")
for h, w in W.items():
    cols = {p.stem: [c for c in pd.read_csv(p, nrows=1).columns if c not in ("timestamp", "month")]
            for p in sorted((w["clone"] / "data/07_model_output/clients").glob("*.csv"))}
    sets = {tuple(v) for v in cols.values()}
    print(f"  {h}: {len(cols)} clients, {'same columns' if len(sets) == 1 else 'COLUMNS DIFFER'}: "
          f"{list(next(iter(sets))) if len(sets) == 1 else sets}")

print("\n== 7. dependence oracle topology (pressure-gauge distance; pressure slots are 0)")
t = W[ref_h]["topology"]
print(f"  hydraulic_distance_m finite in {int(np.isfinite(t['hydraulic_distance_m']).sum())} of {len(t)} pairs")
