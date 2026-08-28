"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)
    # __default__ is the simulation+oracle world; FL runs via --pipeline fl
    # (the composite would double-count its members' nodes inside a sum).
    pipelines["__default__"] = sum(
        p for name, p in pipelines.items()
        if name not in ("fl", "fl_preprocessing", "fl_training",
                        "drift_detection", "dependence_detection", "drift_attribution", "label_factory", "automl", "personalization"))
    return pipelines
