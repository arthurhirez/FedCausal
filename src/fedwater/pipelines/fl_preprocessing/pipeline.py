from kedro.pipeline import Pipeline, node

from .nodes import preprocess_clients


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                preprocess_clients,
                inputs=["client_datasets", "params:fl", "params:time"],
                outputs=["fl_windows", "fl_scalers", "fl_prep_report"],
                name="preprocess_clients",
            ),
        ]
    )
