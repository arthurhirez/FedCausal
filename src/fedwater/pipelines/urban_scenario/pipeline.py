from kedro.pipeline import Pipeline, node

from .nodes import (
    build_drift_schedule,
    build_income_factors,
    build_landuse_factors,
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
                build_landuse_factors,
                inputs=["income_factors", "params:land_use", "params:scenario"],
                outputs="landuse_factors",
                name="build_landuse_factors",
            ),
            node(
                build_portfolios,
                inputs=["wn_variant", "districts", "landuse_factors",
                        "income_factors", "params:scenario", "params:land_use",
                        "params:hydraulics"],
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
                inputs=["portfolios_t0", "gt_drift_schedule", "landuse_factors",
                        "income_factors", "params:land_use", "params:scenario"],
                outputs="assignments_timeline",
                name="evolve_assignments",
            ),
        ]
    )
