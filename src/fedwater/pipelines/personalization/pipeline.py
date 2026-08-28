from kedro.pipeline import Pipeline, node

from .nodes import build_personalization_clusters


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(build_personalization_clusters,
                 inputs=["prototype_history", "drift_signals",
                         "client_dependence", "params:fl", "params:seed"],
                 outputs=["cluster_assignments", "similarity_matrix"],
                 name="build_personalization_clusters"),
        ]
    )
