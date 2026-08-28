from kedro.pipeline import Pipeline, node

from .nodes import (
    dependence_battery,
    minirocket_trajectories,
    structure_recovery,
    topology_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                topology_features,
                inputs=["wn_variant", "districts", "gt_boundaries", "params:sensors"],
                outputs="gt_topology",
                name="topology_features",
            ),
            node(
                minirocket_trajectories,
                inputs=["sensor_series", "params:oracle", "params:time", "params:seed"],
                outputs="feature_trajectories",
                name="minirocket_trajectories",
            ),
            node(
                dependence_battery,
                inputs=["sensor_series", "feature_trajectories", "params:oracle",
                        "params:time", "params:seed"],
                outputs="gt_dependence_battery",
                name="dependence_battery",
            ),
            node(
                structure_recovery,
                inputs=["gt_dependence_battery", "gt_topology"],
                outputs="gt_structure_recovery",
                name="structure_recovery",
            ),
        ]
    )
