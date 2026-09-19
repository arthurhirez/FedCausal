"""Districts: the junction partition, and what crosses between districts.

``districts.yml`` carries two blocks::

    districts:   # MUST partition the junction set exactly
      District_A: [J1, J2, ...]
    assets:      # tanks and reservoirs, RECORDED ONLY
      District_A: [T4]

The split is not cosmetic. When tanks sat inline in the district lists,
``build_portfolios`` crashed on them (a ``Tank`` has no
``demand_timeseries_list``) and ``_boundary_pipes`` classified every tank feed
as an inter-district boundary -- which the coupling variants then closed,
severing storage and recording it as dependence ground truth.
"""
from __future__ import annotations

import pandas as pd


def district_nodes(districts: dict) -> dict[str, list[str]]:
    """The junction partition. Always read districts through this."""
    return {d: [str(n) for n in nodes]
            for d, nodes in districts["districts"].items()}


def district_assets(districts: dict) -> dict[str, list[str]]:
    """Tanks and reservoirs assigned to a district. Nothing reads this yet."""
    return {d: [str(n) for n in nodes]
            for d, nodes in (districts.get("assets") or {}).items()}


def node_to_district(districts: dict) -> dict[str, str]:
    return {n: d for d, nodes in district_nodes(districts).items() for n in nodes}


def validate_partition(wn, districts: dict) -> pd.DataFrame:
    """Hard sanity: ``districts`` must exactly partition the junction set.

    Raises on overlap / missing / unknown nodes; returns a coverage report.
    A node listed under ``assets`` that is not a tank or reservoir is also an
    error -- it means a junction was moved out of the simulation domain.
    """
    nodes = district_nodes(districts)
    all_junctions = set(wn.junction_name_list)

    seen: set[str] = set()
    overlaps: set[str] = set()
    for names in nodes.values():
        overlaps |= seen & set(names)
        seen |= set(names)

    missing = all_junctions - seen
    unknown = seen - all_junctions
    if overlaps or missing or unknown:
        raise ValueError(
            f"District partition invalid — overlaps={sorted(overlaps)}, "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}. "
            "Tanks and reservoirs belong under `assets`, not `districts`."
        )

    storage = set(wn.tank_name_list) | set(wn.reservoir_name_list)
    assets = district_assets(districts)
    stray = {n for names in assets.values() for n in names} - storage
    if stray:
        raise ValueError(
            f"`assets` lists non-storage nodes {sorted(stray)}; only tanks and "
            "reservoirs go there. A junction listed under `assets` is silently "
            "dropped from the simulation domain."
        )

    report = pd.DataFrame(
        [{"district": d, "n_nodes": len(n), "n_assets": len(assets.get(d, []))}
         for d, n in nodes.items()]
    )
    report["total_nodes"] = len(all_junctions)
    return report


def district_links(wn, districts: dict) -> pd.DataFrame:
    """Every link whose endpoints sit in two different districts, by type.

    ``_boundary_pipes`` in ``network_prep`` walks pipes only, so a district
    pair joined through a pump or a valve is invisible to it and to the
    coupling variants -- an `isolated` world would not actually be isolated.
    This table exists to make that visible before it silently misleads.
    """
    n2d = node_to_district(districts)
    rows = []
    for name in wn.link_name_list:
        link = wn.get_link(name)
        da = n2d.get(link.start_node_name)
        db = n2d.get(link.end_node_name)
        if da and db and da != db:
            rows.append({"link": name, "link_type": link.link_type,
                         "district_a": min(da, db), "district_b": max(da, db)})
    return pd.DataFrame(rows, columns=["link", "link_type",
                                       "district_a", "district_b"])


def auto_seed_node(wn, districts: dict, district: str) -> str:
    """Deterministic drift seed: the district's largest-base-demand junction.

    Zero-demand trunk junctions are skipped (the node-110 class of bug the
    label factory exposed). Ties break on the sorted node name, so the answer
    does not depend on dict ordering.

    This is the rule ``experiments.spec.auto_seed_node`` already applied to
    ``.inp`` text; it lives here now so the engine and a plain ``kedro run``
    cannot drift apart.
    """
    carrying = {}
    for n in district_nodes(districts)[district]:
        node = wn.get_node(n)
        base = sum(float(ts.base_value or 0.0)
                   for ts in getattr(node, "demand_timeseries_list", []))
        if base > 0:
            carrying[n] = base
    if not carrying:
        raise ValueError(f"No demand-carrying junction found in {district}.")
    return max(sorted(carrying), key=carrying.get)
