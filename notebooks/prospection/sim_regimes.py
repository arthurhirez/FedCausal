"""sim_regimes — EDA engine for regime dynamics around drift, across worlds.

Reads raw simulator artifacts directly (sensor_series + gt_* CSVs) per world,
the same file layout for "local" (data/07_model_output/...) and any cached
world (data/09_experiments/worlds/<sim_hash>/clone/data/...). No simulation
logic is duplicated: dependence statistics call straight into
``fedwater.pipelines.dependence_oracle`` (``district_signals``,
``dependence_battery``, ``structure_recovery``), just re-windowed by drift
phase instead of by month/dial.

Three questions, three groups of functions:
* level/shape   -- regime_trajectory, distribution_shift
* cohesion      -- phase_battery, phase_recovery, granger_direction
* negative control -- regime_similarity (structural equivalence, NOT cohesion;
  see FLC_033 regimes notes: a big off-diagonal statistic is evidence of a
  pipe, not of a shared regime)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# world discovery (verbatim convention from aer_sandbox.ipynb)
# ---------------------------------------------------------------------------
def find_repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "conf" / "base" / "parameters.yml").exists() and (cand / "src" / "fedwater").is_dir():
            return cand
    return None


def list_worlds(root: Path) -> dict:
    """world_id -> {client_dir, label, meta}. world_id is sim_hash, or 'local'."""
    worlds: dict[str, dict] = {}

    local_dir = root / "data" / "07_model_output" / "clients"
    if local_dir.exists():
        worlds["local"] = {"client_dir": local_dir, "label": "local (data/07_model_output)",
                           "meta": {}}

    exp_root = root / "data" / "09_experiments" / "worlds"
    for mpath in sorted(exp_root.glob("*/manifest.json")):
        try:
            man = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if man.get("status") != "ok":
            continue
        sim_hash = man.get("sim_hash", mpath.parent.name)
        cdir = mpath.parent / "clone" / "data" / "07_model_output" / "clients"
        if not cdir.exists():
            continue
        meta = man.get("world", {})
        tag = meta.get("drift_district") or meta.get("consumption_map") or "?"
        worlds[sim_hash] = {"client_dir": cdir, "meta": meta,
                            "label": f"{sim_hash}  ·  {meta.get('variant', '?')}  ·  {tag}"}
    return worlds


def filter_worlds(worlds: dict, **predicates) -> list[str]:
    """world_ids whose manifest meta matches every predicate exactly (a
    missing key excludes the world, so e.g. "local" never matches an
    n_months filter). Groups worlds for within-group comparison, e.g.
    `filter_worlds(WORLDS, n_months=12.0)` vs `n_months=24.0`."""
    return sorted(wid for wid, w in worlds.items()
                 if all(w.get("meta", {}).get(k) == v for k, v in predicates.items()))


def data_root(world: dict) -> Path:
    """`.../data/` for this world — two levels above its client_dir."""
    return world["client_dir"].parents[1]


# ---------------------------------------------------------------------------
# load one world: sensor_series + ground truth + drift window
# ---------------------------------------------------------------------------
def drift_window(gt_drift_schedule: pd.DataFrame) -> tuple[str, int, int]:
    """(drifting district, first drift month, last drift month) — the network-
    wide "during" span. Generalizes across worlds: whichever district the
    schedule names, not assumed to be District_D."""
    return (str(gt_drift_schedule["district"].iloc[0]),
            int(gt_drift_schedule["drift_month"].min()),
            int(gt_drift_schedule["drift_month"].max()))


def load_world(world: dict, time_cfg: dict) -> dict:
    """sensor_series + gt_drift_schedule + gt_topology + the monthly regime
    trajectory, all for one world. `time_cfg` is the base `parameters.yml`
    `time` block (resolution_h assumed constant across worlds)."""
    droot = data_root(world)
    ss = pd.read_parquet(droot / "03_primary/sensor_series.parquet")
    drift = pd.read_csv(droot / "03_primary/gt_drift_schedule.csv", dtype={"node": str})
    topo = pd.read_csv(droot / "03_primary/gt_topology.csv")
    district, m_min, m_max = drift_window(drift)
    steps_day = int(round(24 / time_cfg["resolution_h"]))
    traj = regime_trajectory(ss, steps_day)
    return dict(sensor_series=ss, drift=drift, topology=topo,
               drift_district=district, m_min=m_min, m_max=m_max, trajectory=traj)


def phase_slice(sensor_series: pd.DataFrame, m_min: int, m_max: int, phase: str) -> pd.DataFrame:
    """pre = before the drift window, during = the diffusion span itself
    (network-wide bounds — the only window a non-drifting district has),
    post = fully switched over. Same bounds applied to every district."""
    if phase == "pre":
        return sensor_series[sensor_series["month"] < m_min]
    if phase == "during":
        return sensor_series[(sensor_series["month"] >= m_min) & (sensor_series["month"] <= m_max)]
    if phase == "post":
        return sensor_series[sensor_series["month"] > m_max]
    raise ValueError(f"unknown phase {phase!r}")


# ---------------------------------------------------------------------------
# level / shape
# ---------------------------------------------------------------------------
def regime_trajectory(sensor_series: pd.DataFrame, steps_day: int) -> pd.DataFrame:
    """Monthly level/shape stats per (district, kind), from the *observed*
    signal — what a client actually has, sensors of the same kind pooled.
    peak_factor = mean-daily-profile max / mean (same definition as V6)."""
    rows = []
    for (d, k, m), g in sensor_series.groupby(["district", "kind", "month"]):
        prof = g.groupby(g["step"] % steps_day)["observed"].mean()
        pf = float(prof.max() / prof.mean()) if prof.mean() else np.nan
        rows.append(dict(district=d, kind=k, month=int(m),
                         mean=float(g["observed"].mean()),
                         std=float(g["observed"].std()), peak_factor=pf))
    return pd.DataFrame(rows).sort_values(["district", "kind", "month"], ignore_index=True)


def distribution_shift(sensor_series: pd.DataFrame, m_min: int, m_max: int) -> pd.DataFrame:
    """Pre vs post shift per (district, kind): KS statistic + Wasserstein
    distance (normalized by the pre-period std) on the raw *observed* signal.
    Deliberately NOT deseasonalized — a level shift (the regime question) is
    exactly what month-aware residualization would strip out (FLC_033
    finding). Caveat: pre/post also differ in season phase, a confound this
    first pass does not separate out."""
    from scipy.stats import ks_2samp, wasserstein_distance
    cols = ["district", "kind", "ks", "wasserstein_norm", "mean_pre", "mean_post"]
    rows = []
    for (d, k), g in sensor_series.groupby(["district", "kind"]):
        pre = g.loc[g["month"] < m_min, "observed"].to_numpy()
        post = g.loc[g["month"] > m_max, "observed"].to_numpy()
        if len(pre) < 10 or len(post) < 10:
            continue
        wd = wasserstein_distance(pre, post) / (pre.std() or 1.0)
        rows.append(dict(district=d, kind=k, ks=float(ks_2samp(pre, post).statistic),
                         wasserstein_norm=float(wd),
                         mean_pre=float(pre.mean()), mean_post=float(post.mean())))
    if not rows:
        # every group failed the >=10-points check -- typically m_min == 0 or
        # m_max sits at (or past) the last month, i.e. no real pre/post left.
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["kind", "wasserstein_norm"],
                                          ascending=[True, False], ignore_index=True)


# ---------------------------------------------------------------------------
# cohesion (dependence, re-windowed by phase)
# ---------------------------------------------------------------------------
CHEAP_ORACLE = dict(max_lag_h=6, tiers=[1, 2, 3], n_surrogates=30,
                    n_surrogates_expensive=15, subsample=1500, dcor_subsample=1000,
                    granger_lags=24,
                    ccm=dict(embed_dim=3, tau=2, lib_sizes=[200, 1200], n_neighbors=4))
# Tier 4 (MiniRocket) deliberately excluded here — it needs
# minirocket_trajectories precomputed; add it back if this EDA graduates.


def phase_battery(sensor_series: pd.DataFrame, m_min: int, m_max: int,
                  time_cfg: dict, seed: int = 0, oracle: dict | None = None) -> dict:
    """{phase: gt_dependence_battery-shaped frame}, tiers 1-3 only, computed
    independently on each phase's rows — the exact `dependence_battery` node,
    just called three times on three slices instead of once on everything.
    A phase with no rows (e.g. m_max sits on the world's last month, so
    "post" is empty) is skipped rather than handed to `dependence_battery`,
    which would otherwise hand back a columnless frame and break
    `structure_recovery` downstream."""
    from fedwater.pipelines.dependence_oracle.nodes import dependence_battery
    oracle = {**CHEAP_ORACLE, **(oracle or {})}
    out = {}
    for phase in ("pre", "during", "post"):
        sub = phase_slice(sensor_series, m_min, m_max, phase)
        if sub.empty:
            continue
        out[phase] = dependence_battery(sub, pd.DataFrame(), oracle, time_cfg, seed)
    return out


def phase_recovery(battery_by_phase: dict, gt_topology: pd.DataFrame) -> pd.DataFrame:
    """AUROC / Spearman-vs-proximity per phase, via the oracle's own scorer.
    Phases already absent from `battery_by_phase` (see `phase_battery`) are
    skipped; if that leaves nothing at all, returns an empty frame with the
    right columns instead of failing on `pd.concat([])`."""
    from fedwater.pipelines.dependence_oracle.nodes import structure_recovery
    cols = ["phase", "kind", "tier", "method", "auroc", "spearman_vs_proximity", "n_pairs"]
    out = []
    for phase, bat in battery_by_phase.items():
        if bat.empty:
            continue
        rec = structure_recovery(bat, gt_topology)
        rec.insert(0, "phase", phase)
        out.append(rec)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def granger_direction(battery_by_phase: dict, drift_district: str, kind: str = "flow") -> pd.DataFrame:
    """Granger-F outward (drifter -> other) vs inward (other -> drifter),
    per phase, generalized to whichever district actually drifted. Phases
    missing from `battery_by_phase` are skipped; if nothing at all comes
    out, or only one direction is ever present, both `inward`/`outward`
    columns are still guaranteed so the caller can index them safely."""
    rows = []
    for phase, bat in battery_by_phase.items():
        if bat.empty:
            continue
        g = bat[(bat["method"] == "granger_f") & (bat["kind"] == kind)]
        g = g[(g["district_a"] == drift_district) | (g["district_b"] == drift_district)]
        for _, r in g.iterrows():
            other = r["district_b"] if r["district_a"] == drift_district else r["district_a"]
            outward = r["direction"] == f"{drift_district}->{other}"
            rows.append(dict(phase=phase, other=other,
                             direction="outward" if outward else "inward",
                             statistic=float(r["statistic"])))
    if not rows:
        return pd.DataFrame(columns=["phase", "other", "inward", "outward"])
    piv = pd.DataFrame(rows).pivot_table(index=["phase", "other"], columns="direction",
                                         values="statistic").reset_index()
    for col in ("inward", "outward"):
        if col not in piv.columns:
            piv[col] = np.nan
    return piv


# ---------------------------------------------------------------------------
# negative control: structural equivalence vs cohesion
# ---------------------------------------------------------------------------
def regime_similarity(trajectory: pd.DataFrame, kind: str = "flow") -> pd.DataFrame:
    """Pairwise correlation of districts' monthly *level* trajectories — 'whose
    own regime behaves alike', not 'who's physically coupled'. Plot this next
    to a phase's cohesion matrix as the negative control from the FLC_033
    regimes notes: cohesion is about the off-diagonal (pipes), this is about
    the diagonal (own behaviour) — conflating the two was flagged as the
    central mistake to avoid."""
    piv = trajectory[trajectory["kind"] == kind].pivot(index="month", columns="district", values="mean")
    return piv.corr()


def pair_matrix(bat_or_corr, districts: list[str], method: str | None = None,
                kind: str | None = None) -> pd.DataFrame:
    """Symmetric |statistic| matrix for one (method, kind) out of a battery
    frame, OR pass a correlation frame (from `regime_similarity`) straight
    through — same plotting shape either way."""
    if method is None:
        return bat_or_corr.reindex(index=districts, columns=districts)
    Mx = pd.DataFrame(np.nan, index=districts, columns=districts)
    sel = bat_or_corr[(bat_or_corr["method"] == method) & (bat_or_corr["kind"] == kind)]
    for _, r in sel.iterrows():
        Mx.loc[r["district_a"], r["district_b"]] = abs(r["statistic"])
        Mx.loc[r["district_b"], r["district_a"]] = abs(r["statistic"])
    return Mx