from kedro.pipeline import Pipeline, node

from .nodes import train_learned_detectors


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(train_learned_detectors,
                 inputs=["labeled_pairs", "labeled_clients", "params:fl",
                         "params:seed"],
                 outputs=["automl_models", "automl_report"],
                 name="train_learned_detectors"),
        ]
    )
