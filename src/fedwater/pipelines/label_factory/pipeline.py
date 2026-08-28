from kedro.pipeline import Pipeline, node

from .nodes import build_world_specs, generate_labeled_worlds


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(build_world_specs,
                 inputs=["params:fl", "districts", "params:seed"],
                 outputs="world_specs", name="build_world_specs"),
            node(generate_labeled_worlds,
                 inputs=["world_specs", "params:fl"],
                 outputs=["labeled_pairs", "labeled_clients"],
                 name="generate_labeled_worlds"),
        ]
    )
