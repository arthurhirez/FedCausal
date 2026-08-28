from kedro.pipeline import Pipeline, node

from .nodes import compute_drift_signals, evaluate_drift


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                compute_drift_signals,
                inputs=["prototype_history", "params:fl"],
                outputs="drift_signals",
                name="compute_drift_signals",
            ),
            node(
                evaluate_drift,
                inputs=["drift_signals", "gt_drift_schedule", "params:fl"],
                outputs="drift_report",
                name="evaluate_drift",
            ),
        ]
    )
