"""Assessment pipeline — run this before building any world on a new network.

    kedro run --pipeline assessment

It reads whichever bundle ``conf/base/globals.yml`` selects and writes every
report under ``data/08_reporting/assessment/<network>/``. It is deliberately
NOT part of ``__default__``: the capacity sweep costs dozens of EPANET solves
and its answer is a property of the network, not of a world.
"""
from kedro.pipeline import Pipeline, node

from .nodes import (
    assess_capacity,
    assess_partition,
    compile_assessment_report,
    condition_and_tier,
    inventory_network,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                inventory_network,
                inputs=["network_inp", "network_profile", "districts"],
                outputs="network_inventory",
                name="inventory_network",
            ),
            node(
                assess_partition,
                inputs=["network_inp", "districts", "network_profile"],
                outputs=["assessment_partition", "assessment_crossing_links"],
                name="assess_partition",
            ),
            node(
                condition_and_tier,
                inputs=["network_inp", "network_profile", "params:assessment"],
                outputs=["conditioning_tier", "conditioning_attempts",
                         "conditioning_recipe"],
                name="condition_and_tier",
            ),
            node(
                assess_capacity,
                inputs=["network_inp", "network_profile", "params:assessment"],
                outputs=["capacity_summary", "capacity_curve",
                         "capacity_defects", "capacity_binding"],
                name="assess_capacity",
            ),
            node(
                compile_assessment_report,
                inputs=["network_inventory", "assessment_partition",
                        "assessment_crossing_links", "conditioning_tier",
                        "capacity_summary"],
                outputs="assessment_report",
                name="compile_assessment_report",
            ),
        ]
    )
