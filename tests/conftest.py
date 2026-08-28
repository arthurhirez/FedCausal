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
