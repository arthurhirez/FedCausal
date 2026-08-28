from kedro.pipeline import Pipeline, node

from .nodes import apply_drift_ramp, synthesize_demands


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                synthesize_demands,
                inputs=["assignments_timeline", "params:patterns", "params:time",
                        "params:seed"],
                outputs="demand_series_raw",
                name="synthesize_demands",
            ),
            node(
                apply_drift_ramp,
                inputs=["demand_series_raw", "gt_drift_schedule", "params:patterns",
                        "params:time"],
                outputs="demand_series",
                name="apply_drift_ramp",
            ),
        ]
    )
