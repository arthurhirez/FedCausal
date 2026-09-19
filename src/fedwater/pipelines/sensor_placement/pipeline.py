"""Sensor placement -- part of ``__default__``.

Sits between the simulation and ``sensing``: the sensors a world's clients
observe are chosen here, from probe stacks at the world's operating point
(see :mod:`fedwater.placement`), and handed downstream as
``sensor_placement`` (``data/03_primary/sensor_placement.csv``). With
``sensor_placement.source: manual`` the profile's hand-placed block is used
instead and no probe is simulated.
"""
from kedro.pipeline import Pipeline, node

from .nodes import ensure_probe_stacks, select_sensors, verify_sensors


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                ensure_probe_stacks,
                inputs=["network_inp", "network_profile", "districts",
                        "partition_manifest", "partition_meta",
                        "hydraulics_cfg", "scenario_cfg", "validation_cfg",
                        "params:time", "params:coupling", "gt_boundaries",
                        "params:land_use", "params:buildings",
                        "params:patterns", "params:sensor_placement"],
                outputs="placement_stacks",
                name="ensure_probe_stacks",
            ),
            node(
                select_sensors,
                inputs=["placement_stacks", "districts", "network_profile",
                        "partition_meta", "params:sensor_placement"],
                outputs=["sensor_placement", "placement_coverage",
                         "placement_channels", "placement_report_arms"],
                name="select_sensors",
            ),
            node(
                verify_sensors,
                inputs=["sensor_placement", "pressures", "flows",
                        "demand_series", "gt_drift_schedule", "districts",
                        "params:time", "params:patterns",
                        "params:sensor_placement"],
                outputs="placement_verification",
                name="verify_sensors",
            ),
        ]
    )
