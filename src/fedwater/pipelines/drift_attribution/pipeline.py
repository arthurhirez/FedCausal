from kedro.pipeline import Pipeline, node

from .nodes import apply_correctors, evaluate_correctors


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                apply_correctors,
                inputs=["prototype_history", "latent_trajectories_residualized",
                        "client_dependence", "params:fl"],
                outputs=["corrected_drift_signals", "drift_attribution",
                         "loop_diagnostics"],
                name="apply_correctors",
            ),
            node(
                evaluate_correctors,
                inputs=["corrected_drift_signals", "gt_drift_schedule",
                        "params:fl"],
                outputs="corrector_ladder_report",
                name="evaluate_correctors",
            ),
        ]
    )
