from kedro.pipeline import Pipeline, node

from fedwater.networks.partition import validate_partition

from .nodes import apply_coupling, configure_network, resolve_network_parameters


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Must run FIRST: every downstream node reads the RESOLVED blocks
            # (hydraulics_cfg / scenario_cfg / validation_cfg), never
            # params:hydraulics et al. directly, so a network-scoped key can
            # never reach a node still holding its `null` placeholder.
            node(
                resolve_network_parameters,
                inputs=["network_profile", "params:hydraulics",
                        "params:scenario", "params:validation"],
                outputs=["hydraulics_cfg", "scenario_cfg", "validation_cfg",
                         "network_params_report"],
                name="resolve_network_parameters",
            ),
            node(
                configure_network,
                inputs=["network_inp", "network_profile", "hydraulics_cfg",
                        "params:time"],
                outputs=["wn_configured", "network_prep_report"],
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
                inputs=["wn_configured", "districts", "params:coupling",
                        "params:seed"],
                outputs=["wn_variant", "gt_boundaries"],
                name="apply_coupling",
            ),
        ]
    )
