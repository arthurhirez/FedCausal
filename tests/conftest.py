"""Anchor test execution to the project root so relative data paths resolve
regardless of where pytest is invoked from."""
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def _run_from_project_root():
    prev = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield
    os.chdir(prev)


def _write_partitions(network: str, methods=("manual",)) -> None:
    """Build missing/stale partitions IN-PROCESS, writing exactly what the
    catalog's `partition_files` would (TextDataset: the text verbatim).

    Tests only -- a project run builds partitions with
    `kedro run --pipeline districting` (or the engine does).
    """
    import yaml
    import wntr

    from fedwater.config import load_base_params
    from fedwater.networks import partitions as pstore
    from fedwater.pipelines.districting import nodes as dn

    dp = dict(load_base_params(PROJECT_ROOT)["districting"])
    todo = [m for m in methods
            if pstore.status(PROJECT_ROOT, network, m, dp) != "current"]
    if not todo:
        return
    root = pstore.bundle_dir(PROJECT_ROOT, network)
    wn = wntr.network.WaterNetworkModel(str(root / "network.inp"))
    profile = yaml.safe_load((root / "profile.yml").read_text())
    manual_text = (root / "districts.yml").read_text()
    manual = yaml.safe_load(manual_text)
    dp["methods"] = list(todo)
    results = dn.build_partitions(wn, manual, profile, dp)
    files = dn.partition_documents(results, wn, manual, manual_text,
                                   (root / "network.inp").read_text(),
                                   profile, dp)
    for key, text in files.items():
        path = root / pstore.PARTITIONS_DIR / f"{key}.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture(scope="session")
def built_partitions(_run_from_project_root):
    """The manual partitions of the bundles the tests read."""
    for network in ("ky7", "graeme"):
        _write_partitions(network)
    return PROJECT_ROOT
