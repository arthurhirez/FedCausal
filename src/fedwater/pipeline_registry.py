"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Not part of the default (simulation + oracle) world build.
#   assessment  — a pre-flight over the NETWORK, not a world. Dozens of EPANET
#                 solves whose answer does not change between worlds, so it is
#                 run once per network by hand: `kedro run --pipeline assessment`.
#   the rest    — learning/analysis stages, run per-experiment by the engine.
_NON_DEFAULT = (
    "assessment",
    "fl", "fl_preprocessing", "fl_training",
    "drift_detection", "dependence_detection", "drift_attribution",
    "label_factory", "automl", "personalization",
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)
    # __default__ is the simulation+oracle world; FL runs via --pipeline fl
    # (the composite would double-count its members' nodes inside a sum).
    pipelines["__default__"] = sum(
        p for name, p in pipelines.items() if name not in _NON_DEFAULT)
    return pipelines
