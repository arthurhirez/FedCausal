from kedro.pipeline import Pipeline, node

from .nodes import apply_coupling, configure_network, validate_partition


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                configure_network,
                inputs=["graeme_network", "params:hydraulics", "params:time"],
                outputs="wn_configured",
                name="configure_network",
            ),
            node(
                validate_partition,
                inputs=["wn_configured", "districts"],
                outputs="partition_report",
                name="validate_partition",
            ),
            node(
                apply_coupling,
                inputs=["wn_configured", "districts", "params:coupling", "params:seed"],
                outputs=["wn_variant", "gt_boundaries"],
                name="apply_coupling",
            ),
        ]
    )
