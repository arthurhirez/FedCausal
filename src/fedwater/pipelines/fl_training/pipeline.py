from kedro.pipeline import Pipeline, node

from .nodes import train_federated


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                train_federated,
                inputs=["fl_windows", "params:fl", "params:seed"],
                outputs=["fl_models", "prototype_history",
                         "global_prototype_history", "latent_trajectories",
                         "fl_training_log"],
                name="train_federated",
            ),
        ]
    )
