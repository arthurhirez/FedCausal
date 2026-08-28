from kedro.pipeline import Pipeline, node

from .nodes import evaluate_dependence, federated_dependence, residualize_latents


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                residualize_latents,
                inputs=["latent_trajectories", "params:fl", "params:time"],
                outputs="latent_trajectories_residualized",
                name="residualize_latents",
            ),
            node(
                federated_dependence,
                inputs=["latent_trajectories_residualized", "prototype_history",
                        "params:fl", "params:seed"],
                outputs="client_dependence",
                name="federated_dependence",
            ),
            node(
                evaluate_dependence,
                inputs=["client_dependence", "gt_topology",
                        "gt_dependence_battery"],
                outputs="dependence_evaluation",
                name="evaluate_dependence",
            ),
        ]
    )
