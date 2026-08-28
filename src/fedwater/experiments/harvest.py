"""Harvest: fixed metric extractions from a finished run's ``data/`` tree,
plus the retrieval API analyses use to pull results back out.

Every group is a pure function ``data_dir -> tidy DataFrame`` over the
pipeline's reporting artifacts; the engine stores each group per run
(papertrail) and :meth:`ExperimentEngine.collect` concatenates them across
runs with ``(sim_hash, run_hash)`` keys and rebuilds the flat study index.

Analysis side:

    from fedwater.experiments import load_index, load_group
    idx = load_index("d0_replication")
    dep = load_group("d0_replication", "dependence_summary",
                     where={"level": "T", "method": "partial_rv"})

``load_group`` joins the study index columns (world/run spec, status) onto
the group rows, so a query like the one above comes back ready to
``groupby(["variant", "step_size"])`` — no manual bookkeeping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# group extractors
# ---------------------------------------------------------------------------


def harvest_validation(data: Path) -> pd.DataFrame:
    """The V-checks, verbatim (check, value, hard, passed)."""
    return pd.read_csv(data / "08_reporting/validation_report.csv")


def harvest_drift(data: Path) -> pd.DataFrame:
    """Uncorrected drift report, verbatim (per-client rows; the summary
    metrics — rank, false alarms, separation — ride on their usual rows)."""
    return pd.read_csv(data / "08_reporting/drift_report.csv")


def harvest_ladder(data: Path) -> pd.DataFrame:
    """Corrector ladder (C0..C4) report, verbatim, long over correctors."""
    return pd.read_csv(data / "08_reporting/corrector_ladder_report.csv")


def harvest_c4(data: Path) -> pd.DataFrame:
    """Final-iteration C4 loop state per client, with the drift origin from
    ground truth and the fingered client (argmin weight) marked. The
    across-run distribution of ``fingered_client`` is the attribution-
    stability measurand D0 exists to estimate."""
    diag = pd.read_csv(data / "08_reporting/loop_diagnostics.csv")
    last = diag[diag["iteration"] == diag["iteration"].max()].copy()
    gt = pd.read_csv(data / "03_primary/gt_drift_schedule.csv")
    origin = gt.sort_values("drift_month")["district"].iloc[0]
    last["is_origin"] = last["client"] == origin
    last["is_fingered"] = last["weight"] == last["weight"].min()
    return last.reset_index(drop=True)


def harvest_dependence_pairs(data: Path) -> pd.DataFrame:
    """Federated dependence statistics, verbatim (all levels/methods/pairs,
    with surrogate p and BH-FDR q values)."""
    return pd.read_csv(data / "08_reporting/client_dependence.csv")


def harvest_dependence_summary(data: Path) -> pd.DataFrame:
    """Per (level, method): magnitude, significance count, structure-recovery
    AUROC and the ranking **margin** (boundary-min minus non-boundary-max of
    |statistic|). The margin is the quantity the flagship AUROC actually
    rests on — the deep dive showed it lives at the 0.01–0.13 level and can
    go negative under a different training run, so it is harvested
    explicitly rather than left implicit in a perfect-looking 1.00."""
    fed = pd.read_csv(data / "08_reporting/client_dependence.csv")
    topo = pd.read_csv(data / "03_primary/gt_topology.csv")
    boundary = {tuple(sorted((a, b))): open_pipes > 0
                for a, b, open_pipes in zip(topo["district_a"],
                                            topo["district_b"],
                                            topo["open_boundary_pipes"])}
    rows = []
    for (level, method), g in fed.groupby(["level", "method"]):
        g = g.assign(
            abs_stat=g["statistic"].abs(),
            is_boundary=[boundary.get(tuple(sorted((a, b))))
                         for a, b in zip(g["district_a"], g["district_b"])])
        g = g.dropna(subset=["is_boundary"])
        b = g.loc[g["is_boundary"].astype(bool), "abs_stat"]
        nb = g.loc[~g["is_boundary"].astype(bool), "abs_stat"]
        rows.append({
            "level": level, "method": method, "n_pairs": len(g),
            "mean_abs_stat": g["abs_stat"].mean(),
            "n_sig_q05": int((g["q_value"] < 0.05).sum()),
            "auroc": _auroc(g["is_boundary"].astype(bool), g["abs_stat"]),
            "boundary_min": b.min() if len(b) else np.nan,
            "nonboundary_max": nb.max() if len(nb) else np.nan,
            "margin": (b.min() - nb.max()) if len(b) and len(nb) else np.nan,
        })
    out = pd.DataFrame(rows)
    eval_path = data / "08_reporting/dependence_evaluation.csv"
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        ev = ev[ev["evaluation"] == "structure_recovery"].copy()
        # The evaluation table keys rows by `kind`, not `level`, and methods
        # like rv exist under both kinds — merging on method alone would
        # duplicate summary rows. Map kind -> level and merge on both.
        ev["level"] = ev["kind"].map({"aer_latent": "T",
                                      "prototype_displacement": "P"})
        ev = ev.dropna(subset=["level"]).drop_duplicates(["level", "method"])
        out = out.merge(ev[["level", "method", "spearman_vs_proximity"]],
                        on=["level", "method"], how="left")
    return out


def _auroc(labels: pd.Series, scores: pd.Series) -> float:
    """Rank-based AUROC (Mann-Whitney), dependency-free on purpose."""
    pos, neg = scores[labels].to_numpy(), scores[~labels].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def harvest_pairs(data: Path) -> pd.DataFrame:
    """Label-factory pair table: oracle + federated statistics as features,
    physical connectivity as the label. Ported verbatim from the original
    factory extractor so downstream AutoML feature names are unchanged."""
    topo = pd.read_csv(data / "03_primary/gt_topology.csv")
    bat = pd.read_csv(data / "03_primary/gt_dependence_battery.csv")
    fed = pd.read_csv(data / "08_reporting/client_dependence.csv")

    feats = (bat.assign(abs_stat=bat["statistic"].abs())
             .groupby(["district_a", "district_b", "kind", "method"])
             ["abs_stat"].max().unstack(["kind", "method"]))
    feats.columns = [f"oracle_{k}_{m}" for k, m in feats.columns]
    ft = fed[fed["level"] == "T"].pivot_table(
        index=["district_a", "district_b"], columns="method",
        values="statistic", aggfunc="first")
    ft.columns = [f"fed_{c}" for c in ft.columns]

    out = topo.set_index(["district_a", "district_b"]) \
        .join(feats).join(ft).reset_index()
    out["label_connected"] = (out["open_boundary_pipes"] > 0).astype(int)
    return out


def harvest_clients(data: Path) -> pd.DataFrame:
    """Label-factory client table: drift-signal summaries, corrector means,
    C4 signatures; label = is the client the drift origin (+ onset month).
    Ported verbatim from the original factory extractor."""
    sig = pd.read_csv(data / "07_model_output/drift_signals.csv")
    ladder = pd.read_csv(data / "07_model_output/corrected_drift_signals.csv")
    diag = pd.read_csv(data / "08_reporting/loop_diagnostics.csv")
    gt = pd.read_csv(data / "03_primary/gt_drift_schedule.csv")

    base = sig.groupby("client").agg(
        delta_first_mean=("delta_first", "mean"),
        delta_first_max=("delta_first", "max"),
        delta_roll_max=("delta_roll", "max")).reset_index()
    lad = (ladder.groupby(["client", "corrector"])["delta_first"].mean()
           .unstack("corrector").add_prefix("mean_"))
    dfin = diag[diag["iteration"] == diag["iteration"].max()] \
        .set_index("client")[["weight", "gain", "sparse_score"]] \
        .add_prefix("c4_")
    out = base.set_index("client").join(lad).join(dfin).reset_index()
    out["label_is_origin"] = (out["client"] == gt["district"].iloc[0]).astype(int)
    out["label_onset_month"] = int(gt["drift_month"].min())
    return out


GROUPS = {
    "validation": harvest_validation,
    "drift": harvest_drift,
    "ladder": harvest_ladder,
    "c4": harvest_c4,
    "dependence_pairs": harvest_dependence_pairs,
    "dependence_summary": harvest_dependence_summary,
    "pairs": harvest_pairs,
    "clients": harvest_clients,
}
_ALIASES = {"dependence": ("dependence_pairs", "dependence_summary")}


def resolve_groups(names) -> list[str]:
    out: list[str] = []
    for n in names:
        for g in _ALIASES.get(n, (n,)):
            if g not in GROUPS:
                raise KeyError(f"Unknown harvest group '{g}'; available: "
                               f"{sorted(GROUPS)} + aliases {sorted(_ALIASES)}.")
            if g not in out:
                out.append(g)
    return out


# ---------------------------------------------------------------------------
# headline metrics for the flat study index
# ---------------------------------------------------------------------------
def _first_valid(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return s.iloc[0] if len(s) else np.nan


def headline(run_dir: Path, manifest: dict) -> dict:
    """Small, decision-relevant scalars for ``runs.parquet``; details stay
    in the group tables."""
    out: dict = {}
    h = run_dir / "harvest"
    try:
        dep = pd.read_parquet(h / "dependence_summary.parquet")
        t = dep[dep["level"] == "T"].set_index("method")
        for m in ("rv", "partial_rv"):
            if m in t.index:
                out[f"{m}_auroc_T"] = float(t.loc[m, "auroc"])
                out[f"{m}_mean_T"] = float(t.loc[m, "mean_abs_stat"])
                out[f"{m}_nsig_T"] = int(t.loc[m, "n_sig_q05"])
                out[f"{m}_margin_T"] = float(t.loc[m, "margin"])
    except (OSError, KeyError, ValueError):
        pass
    try:
        drift = pd.read_parquet(h / "drift.parquet")
        drifted = drift[drift["is_drifted"] == True]  # noqa: E712
        out["drift_rank"] = _first_valid(drifted, "rank")
        for col in ("false_alarms", "separation_ratio",
                    "detection_delay_months"):
            out[f"drift_{col}"] = _first_valid(drift, col)
    except (OSError, KeyError, ValueError):
        pass
    try:
        c4 = pd.read_parquet(h / "c4.parquet")
        fingered = c4[c4["is_fingered"]]
        out["c4_fingered_client"] = fingered["client"].iloc[0]
        out["c4_fingered_is_origin"] = bool(fingered["is_origin"].iloc[0])
    except (OSError, KeyError, IndexError, ValueError):
        pass
    return out


# ---------------------------------------------------------------------------
# retrieval API
# ---------------------------------------------------------------------------
def _study_root(study: str, project_root: Path | str = ".",
                root: Path | str | None = None) -> Path:
    if root is None:
        from .spec import load_studies
        root = load_studies(Path(project_root)).get("root",
                                                    "data/09_experiments")
    root = Path(root)
    if not root.is_absolute():
        root = Path(project_root) / root
    return root / study


def load_index(study: str, project_root: Path | str = ".",
               root: Path | str | None = None) -> pd.DataFrame:
    """The flat study index: one row per run — specs, status, headlines."""
    return pd.read_parquet(
        _study_root(study, project_root, root) / "runs.parquet")


def load_group(study: str, group: str, project_root: Path | str = ".",
               root: Path | str | None = None,
               where: dict | None = None,
               join_index: bool = True) -> pd.DataFrame:
    """One harvested group across all runs of a study, optionally joined
    with the study index and filtered by simple equality (``where``)."""
    base = _study_root(study, project_root, root)
    df = pd.read_parquet(base / "harvest" / f"{group}.parquet")
    if join_index:
        idx = load_index(study, project_root, root)
        overlap = [c for c in idx.columns
                   if c in df.columns and c not in ("sim_hash", "run_hash")]
        df = df.merge(idx.drop(columns=overlap),
                      on=["sim_hash", "run_hash"], how="left")
    for key, val in (where or {}).items():
        df = df[df[key].isin(val)] if isinstance(val, (list, tuple, set)) \
            else df[df[key] == val]
    return df.reset_index(drop=True)
