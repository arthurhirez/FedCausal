"""fedwater.experiments — declarative studies over cached worlds x runs.

Public API::

    from fedwater.experiments import run_study, load_index, load_group

    run_study("d0_replication", n_jobs=4)         # or the module CLI:
    #   python -m fedwater.experiments run d0_replication --n-jobs 4

    idx = load_index("d0_replication")
    dep = load_group("d0_replication", "dependence_summary",
                     where={"level": "T", "method": "partial_rv"})

Studies are declared in ``conf/base/experiments.yml``. See ``spec.py`` for
the world/run factorization and ``engine.py`` for mechanics + papertrail.
"""
from .engine import ExperimentEngine, run_study
from .harvest import load_group, load_index
from .spec import (canonical_hash, decode_map, encode_map, expand_study,
                   resolve_run, resolve_world)

__all__ = ["ExperimentEngine", "run_study", "load_index", "load_group",
           "expand_study", "resolve_world", "resolve_run",
           "encode_map", "decode_map", "canonical_hash"]
