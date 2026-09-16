"""District attribution -- run once per network, outside ``__default__``.

    kedro run --pipeline districting

Builds every method listed in ``globals.districting.methods`` against the
selected network and writes each one to
``data/01_raw/<network>/partitions/<method>/`` (``districts.yml``,
``partition.yml``, ``boundary.csv``, ``sanity.csv``), plus the cross-method
comparison under ``data/08_reporting/districting/<network>/``.

It is not part of ``__default__`` for the same reason ``assessment`` is not:
its answer is a property of the network, not of a world. A world reads the
partition named by ``globals.districting.active``; the experiments engine
builds any partition a study needs before it builds worlds on it.
"""
from kedro.pipeline import Pipeline, node

from .nodes import build_partitions, partition_documents, partition_reports


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                build_partitions,
                inputs=["network_inp", "districts_manual", "network_profile",
                        "params:districting"],
                outputs="districting_results",
                name="build_partitions",
            ),
            node(
                partition_documents,
                inputs=["districting_results", "network_inp",
                        "districts_manual", "districts_manual_text",
                        "network_inp_text", "network_profile",
                        "params:districting"],
                outputs="partition_files",
                name="partition_documents",
            ),
            node(
                partition_reports,
                inputs=["districting_results", "network_inp",
                        "districts_manual", "network_profile",
                        "params:districting"],
                outputs=["partition_tables", "districting_summary",
                         "districting_sanity", "districting_order_sensitivity"],
                name="partition_reports",
            ),
        ]
    )
