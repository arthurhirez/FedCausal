"""District attribution for water distribution networks.

One partitioning stage, two families of method, one output contract.

The families
------------
* ``spectral`` -- partition the network's own graph under a *composite edge
  affinity* built from tunable factors (pipe resistance, elevation similarity,
  valve-cut preference, and optionally a measured hydraulic coupling matrix).
  Contiguity is guaranteed by construction, then repaired and verified.
* ``girvan_newman`` / ``fast_greedy`` / ``walktrap`` -- community detection on
  the bare topology, after Shekofteh, Yousefi-Khoshqalb & Piratla (2023), with
  the paper's external stopping condition (cut the dendrogram at ``k``).
* ``external`` -- adopt a partition that already exists (a ``districts.yml``,
  or a network's own published DMA tags) so it is scored on the same footing.

Whatever produces the labels, everything downstream is shared: naming, the
contiguity/balance/elevation report, boundary-pipe configuration and the
optional Phase-3 optimisation, the sanity suite, and the ``districts.yml``
writer. That is the point of this package -- the method is a parameter, not a
fork in the code.

The output contract
-------------------
``districts.yml``: junctions partitioned *exactly* under ``districts``, tanks
and reservoirs recorded under ``assets``. Downstream (demand portfolios, drift
diffusion, boundary-pipe accounting, and the sensor-placement stage that comes
next) reads that file and nothing else from here.

Layering
--------
This package depends on ``wntr``/``networkx``/``igraph``/``sklearn`` and on
nothing else in the project. In particular it does NOT import the dependency-
mapping modules: an interventional coupling matrix enters as an optional
``pd.DataFrame`` argument, never as an import. Dependency mapping consumes
districts, so the import arrow points that way and not back.
"""
from __future__ import annotations

from . import coupling, graph, hydraulics, metrics, partition, runner, weights, yaml_io
from .graph import (
    VALVE_TYPES_REGULATING,
    build_graph,
    check_elevation_informative,
    component_report,
    consumer_nodes,
    hydraulic_assets,
    source_nodes,
)
from .hydraulics import HydraulicCriteria
from .metrics import agreement, contiguity_report, full_report, summary_row
from .partition import METHODS, partition_labels
from .runner import DistrictingConfig, DistrictingResult, compare, run, run_all, sanity_check
from .weights import DistrictingWeights
from .yaml_io import (
    districts_mapping,
    find_bundle,
    read_districts_yml,
    validate_districts,
    write_districts_yml,
)

__all__ = [
    "DistrictingConfig", "DistrictingResult", "DistrictingWeights", "HydraulicCriteria",
    "METHODS", "VALVE_TYPES_REGULATING",
    "agreement", "build_graph", "check_elevation_informative", "compare",
    "component_report", "consumer_nodes", "contiguity_report", "districts_mapping",
    "find_bundle", "full_report", "hydraulic_assets", "partition_labels",
    "read_districts_yml", "run", "run_all", "sanity_check", "source_nodes",
    "summary_row", "validate_districts", "write_districts_yml",
    "coupling", "graph", "hydraulics", "metrics", "partition", "plots", "runner",
    "weights", "yaml_io",
]


def __getattr__(name):
    # `plots` pulls in matplotlib (and ipywidgets on use). The Kedro nodes never
    # draw, so it is imported on first access -- `dst.plots.plot_districts`
    # keeps working in a notebook exactly as before.
    if name == "plots":
        import importlib

        mod = importlib.import_module(".plots", __name__)
        globals()["plots"] = mod
        return mod
    raise AttributeError(name)
