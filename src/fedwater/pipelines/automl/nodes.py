"""AutoML over the label factory's tables — learned detectors, honestly
validated.

Two models, one discipline:

* dependence detector: ``labeled_pairs`` -> P(pair physically connected),
  features = oracle + federated dependence statistics.
* drift attributor:    ``labeled_clients`` -> P(client is the drift ORIGIN),
  features = delta summaries, corrector outputs, C4 gain/weight signatures —
  the originator-vs-receiver distinction the analytic ladder could not make.

Validation is LEAVE-ONE-WORLD-OUT (rows within a world share everything;
random-row CV would be leakage). "AutoML" here is a small HistGB grid —
deliberately: the science is in the label factory and the grouped
validation, not the search; escalate the search space only after this
baseline is beaten. Models must ultimately beat the best analytic rung on
the SAME harness to earn deployment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {"world", "variant", "close_fraction", "district_a", "district_b",
        "client"}


def _xy(df: pd.DataFrame, label: str):
    feats = [c for c in df.columns
             if c not in META and not c.startswith("label_")
             and pd.api.types.is_numeric_dtype(df[c])]
    return df[feats].to_numpy(dtype=float), df[label].to_numpy(), feats


def _grouped_search(X, y, groups, grid: dict, seed: int):
    """Leave-one-world-out AUROC over a small HistGB grid."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    uworlds = np.unique(groups)
    if len(uworlds) < 2:
        raise ValueError("Need >= 2 worlds for leave-one-world-out CV.")
    results = []
    for mln in grid["max_leaf_nodes"]:
        for lr in grid["learning_rate"]:
            aucs = []
            for tr, te in LeaveOneGroupOut().split(X, y, groups):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                m = HistGradientBoostingClassifier(
                    max_leaf_nodes=mln, learning_rate=lr, random_state=seed)
                m.fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            if aucs:
                results.append(dict(max_leaf_nodes=mln, learning_rate=lr,
                                    cv_auroc=float(np.mean(aucs)),
                                    n_folds=len(aucs)))
    if not results:
        raise ValueError("No valid CV folds (single-class worlds?).")
    best = max(results, key=lambda r: r["cv_auroc"])
    final = HistGradientBoostingClassifier(
        max_leaf_nodes=best["max_leaf_nodes"],
        learning_rate=best["learning_rate"], random_state=seed)
    final.fit(X, y)
    return final, best, results


def train_learned_detectors(labeled_pairs: pd.DataFrame,
                            labeled_clients: pd.DataFrame,
                            fl: dict, seed: int):
    grid = fl["automl"]
    models, report_rows = {}, []

    for name, df, label in [
        ("dependence_detector", labeled_pairs, "label_connected"),
        ("drift_attributor", labeled_clients, "label_is_origin"),
    ]:
        X, y, feats = _xy(df, label)
        X = np.nan_to_num(X, nan=np.nan)  # HistGB handles NaN natively
        model, best, results = _grouped_search(
            X, y, df["world"].to_numpy(), grid, seed)
        models[name] = {"model": model, "features": feats}
        for r in results:
            report_rows.append(dict(model=name, **r,
                                    selected=r == best, n_rows=len(df),
                                    n_features=len(feats)))
    return models, pd.DataFrame(report_rows)
