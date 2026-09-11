from kedro.pipeline import Pipeline, node

from .nodes import (
    check_consumption_sanity,
    check_mass_balance,
    check_peak_factors,
    check_pressures,
    check_storage,
    compile_validation_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                check_mass_balance,
                inputs=["wn_variant", "demands_simulated", "demand_series",
                        "validation_cfg"],
                outputs="report_mass",
                name="check_mass_balance",
            ),
            node(
                check_pressures,
                inputs=["pressures", "demand_series", "validation_cfg"],
                outputs="report_pressure",
                name="check_pressures",
            ),
            node(
                check_storage,
                inputs=["pressures", "wn_variant", "validation_cfg"],
                outputs="report_storage",
                name="check_storage",
            ),
            node(
                check_consumption_sanity,
                inputs=["assignments_timeline", "income_factors",
                        "validation_cfg"],
                outputs="report_consumption",
                name="check_consumption_sanity",
            ),
            node(
                check_peak_factors,
                inputs=["demand_series", "assignments_timeline", "params:time",
                        "validation_cfg"],
                outputs="report_peaks",
                name="check_peak_factors",
            ),
            node(
                compile_validation_report,
                inputs=["report_mass", "report_pressure", "report_storage",
                        "report_consumption", "report_peaks"],
                outputs="validation_report",
                name="compile_validation_report",
            ),
        ]
    )
