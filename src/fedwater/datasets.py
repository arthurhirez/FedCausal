"""Custom Kedro datasets for water-network simulation artifacts."""
from __future__ import annotations

from pathlib import Path

import wntr
from kedro.io import AbstractDataset


class WntrNetworkDataset(AbstractDataset):
    """Load/save an EPANET ``.inp`` file as a ``wntr.network.WaterNetworkModel``.

    Making the network a catalog entry (rather than a path parameter) keeps
    lineage explicit: every pipeline that touches the network declares it.
    """

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def load(self) -> wntr.network.WaterNetworkModel:
        return wntr.network.WaterNetworkModel(str(self._filepath))

    def save(self, data: wntr.network.WaterNetworkModel) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        wntr.network.write_inpfile(data, str(self._filepath), version=2.2)

    def _describe(self) -> dict:
        return {"filepath": str(self._filepath)}
