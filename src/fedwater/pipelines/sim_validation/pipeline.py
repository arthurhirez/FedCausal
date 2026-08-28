from kedro.pipeline import Pipeline, node

from .nodes import (
    check_consumption_sanity,
    check_mass_balance,
    check_peak_factors,
    check_pressures,
    compile_validation_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                check_mass_balance,
                inputs=["demands_simulated", "demand_series", "params:validation"],
                outputs="report_mass",
                name="check_mass_balance",
            ),
            node(
                check_pressures,
                inputs=["pressures", "demand_series", "params:validation"],
                outputs="report_pressure",
                name="check_pressures",
            ),
            node(
                check_consumption_sanity,
                inputs=["assignments_timeline", "income_factors", "params:validation"],
                outputs="report_consumption",
                name="check_consumption_sanity",
            ),
            node(
                check_peak_factors,
                inputs=["demand_series", "assignments_timeline", "params:time",
                        "params:validation"],
                outputs="report_peaks",
                name="check_peak_factors",
            ),
            node(
                compile_validation_report,
                inputs=["report_mass", "report_pressure", "report_consumption",
                        "report_peaks"],
                outputs="validation_report",
                name="compile_validation_report",
            ),
        ]
    )
