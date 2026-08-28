from kedro.pipeline import Pipeline, node

from .nodes import (
    build_drift_schedule,
    build_income_factors,
    build_portfolios,
    evolve_assignments,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                build_income_factors,
                inputs="params:buildings",
                outputs="income_factors",
                name="build_income_factors",
            ),
            node(
                build_portfolios,
                inputs=["wn_variant", "districts", "income_factors",
                        "params:scenario", "params:buildings",
                        "params:hydraulics", "params:seed"],
                outputs="portfolios_t0",
                name="build_portfolios",
            ),
            node(
                build_drift_schedule,
                inputs=["wn_variant", "districts", "params:scenario", "params:seed"],
                outputs="gt_drift_schedule",
                name="build_drift_schedule",
            ),
            node(
                evolve_assignments,
                inputs=["portfolios_t0", "gt_drift_schedule", "income_factors",
                        "params:buildings", "params:scenario", "params:seed"],
                outputs="assignments_timeline",
                name="evolve_assignments",
            ),
        ]
    )
