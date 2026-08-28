from kedro.pipeline import Pipeline, node

from .nodes import run_hydraulics


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                run_hydraulics,
                inputs=["wn_variant", "demand_series"],
                outputs=["pressures", "flows", "demands_simulated"],
                name="run_hydraulics",
            ),
        ]
    )
