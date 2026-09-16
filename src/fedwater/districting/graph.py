"""The network as a graph, plus who each node is.

One graph for both method families
----------------------------------
The spectral method wants edge features (resistance, valve/pump flags) to build
an affinity; the community-detection methods want bare topology. Those were two
separate constructions before, differing only in whether attributes were
attached -- which meant two places to get parallel-edge handling and self-loop
filtering right. There is one construction here: :func:`build_graph` attaches
the attributes, and the community methods simply ignore them (they are passed
``weight=None``). Identical vertex and edge sets by construction, so a
partition from either family is scored against exactly the same object.

Multi-edges (parallel pipes) collapse to the lowest-resistance parallel path,
which is the one that actually carries flow. Self-loops are dropped.

Node roles
----------
Three overlapping sets matter downstream and they are not interchangeable:

* **consumers** -- demand-bearing junctions that are not pump/valve scaffolding.
  These are the federated clients, and they are what balance is measured over.
* **sources** -- reservoirs and tanks. Type-1 DMA classification keys off these,
  and they are what ``districts.yml`` records under ``assets``.
* **everything else** -- zero-demand junctions inserted to give a pump or valve
  its two endpoints. They carry no demand and represent nobody, but they are
  load-bearing for *contiguity*: a district can be one connected piece whose
  connecting path runs through such a node, so they must be present in the
  graph handed to the partitioner even though they are excluded from balance.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

LPS_PER_M3S = 1000.0

#: Valves that actively set a downstream head or flow, as opposed to merely
#: throttling. Their presence is what makes instantaneous coupling directional.
VALVE_TYPES_REGULATING = ("PRV", "PSV", "PBV")

DEFAULT_LENGTH_M = 50.0
DEFAULT_DIAMETER_M = 0.15


# ==========================================================================
# node roles
# ==========================================================================

def base_demand_lps(wn, node: str) -> float:
    """Total base demand across every demand category on a node, in L/s."""
    n = wn.get_node(node)
    tsl = getattr(n, "demand_timeseries_list", None)
    if tsl is None:
        return 0.0
    return sum(float(ts.base_value or 0.0) for ts in tsl) * LPS_PER_M3S


def consumer_nodes(wn) -> list[str]:
    """Demand-bearing junctions that are not pump/valve scaffolding.

    Pump and valve links need two endpoints, so model builders insert zero-demand
    junctions on either side (``I-Pump-3``, ``O-RV-25`` in the KY corpus). Those
    are not consumers: they carry no demand, represent nobody, and must not
    become federated clients.

    A junction qualifies when it has positive base demand AND is not a degree-1
    or degree-2 node whose every incident link is a pump or a valve.
    """
    out = []
    for jname in wn.junction_name_list:
        if base_demand_lps(wn, jname) <= 0:
            continue
        links = list(wn.get_links_for_node(jname))
        types = {wn.get_link(l).link_type for l in links}
        if 0 < len(links) <= 2 and types and types.issubset({"Pump", "Valve"}):
            continue
        out.append(jname)
    return out


def source_nodes(wn) -> list[str]:
    """Reservoirs and tanks -- the "water sources" that define a Type 1 DMA."""
    return list(wn.reservoir_name_list) + list(wn.tank_name_list)


def hydraulic_assets(wn) -> pd.DataFrame:
    """Inventory of every non-pipe hydraulic component, with coordinates.

    Node assets (reservoirs, tanks) carry their own coordinates; link assets
    (pumps, valves) are given the midpoint of their endpoints, which is where a
    marker belongs on a network plot. ``kind`` is the coarse class and
    ``subtype`` the specific one (``PRV``, ``TCV``, ``HEAD`` pump, ...), because
    a regulating valve and a throttle valve are not the same thing to a
    districting decision even though both draw as "a valve".
    """
    coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    rows = []
    for name in wn.reservoir_name_list:
        rows.append({"name": name, "kind": "reservoir", "subtype": "reservoir",
                     "x": _x(coords.get(name), 0), "y": _x(coords.get(name), 1)})
    for name in wn.tank_name_list:
        t = wn.get_node(name)
        rows.append({"name": name, "kind": "tank",
                     "subtype": "tank (%.1f m diam)" % float(getattr(t, "diameter", np.nan) or np.nan),
                     "x": _x(coords.get(name), 0), "y": _x(coords.get(name), 1)})
    for name in wn.pump_name_list:
        p = wn.get_link(name)
        rows.append({"name": name, "kind": "pump",
                     "subtype": str(getattr(p, "pump_type", "pump")),
                     **_midpoint(coords, p)})
    for name in wn.valve_name_list:
        v = wn.get_link(name)
        vt = str(getattr(v, "valve_type", "") or "valve")
        rows.append({"name": name, "kind": "valve", "subtype": vt,
                     **_midpoint(coords, v)})
    return pd.DataFrame(rows, columns=["name", "kind", "subtype", "x", "y"])


def _x(coord, i):
    return float(coord[i]) if coord else np.nan


def _midpoint(coords, link) -> dict:
    a, b = coords.get(link.start_node_name), coords.get(link.end_node_name)
    if not a or not b:
        return {"x": np.nan, "y": np.nan}
    return {"x": 0.5 * (a[0] + b[0]), "y": 0.5 * (a[1] + b[1])}


# ==========================================================================
# the graph
# ==========================================================================

def build_graph(wn) -> nx.Graph:
    """Every node, every link, with the attributes both families need.

    Node attributes: ``elevation``, ``x``, ``y``, ``node_type``,
    ``base_demand_lps``, ``is_consumer``, ``is_source``.

    Edge attributes: ``link``, ``link_type``, ``length_m``, ``diameter_m``,
    ``resistance`` (``L / D**5``), ``is_valve``, ``is_regulating_valve``,
    ``is_pump``, ``is_closed``, ``closable``.

    ``closable`` marks pipes only: a pump or a control valve sitting on a
    district boundary cannot be swapped for a gate valve without changing what
    the network is, so non-pipe boundary links are held open and merely counted.
    """
    consumers = set(consumer_nodes(wn))
    sources = set(source_nodes(wn))

    g = nx.Graph()
    for name in wn.node_name_list:
        node = wn.get_node(name)
        coords = node.coordinates or (np.nan, np.nan)
        g.add_node(
            name,
            elevation=float(getattr(node, "elevation", np.nan) or np.nan),
            x=float(coords[0]), y=float(coords[1]),
            node_type=node.node_type,
            base_demand_lps=base_demand_lps(wn, name),
            is_consumer=name in consumers,
            is_source=name in sources,
        )

    for lname in wn.link_name_list:
        link = wn.get_link(lname)
        a, b = link.start_node_name, link.end_node_name
        if a == b or not g.has_node(a) or not g.has_node(b):
            continue                                  # self-loop or dangling
        length = float(getattr(link, "length", DEFAULT_LENGTH_M) or DEFAULT_LENGTH_M)
        diam = float(getattr(link, "diameter", DEFAULT_DIAMETER_M) or DEFAULT_DIAMETER_M)
        is_valve = link.link_type == "Valve"
        attrs = {
            "link": lname, "link_type": link.link_type,
            "length_m": length, "diameter_m": diam,
            "resistance": length / max(diam, 1e-3) ** 5,
            "is_valve": is_valve,
            "is_regulating_valve": is_valve and str(getattr(link, "valve_type", "")) in VALVE_TYPES_REGULATING,
            "is_pump": link.link_type == "Pump",
            "is_closed": str(getattr(link, "initial_status", "")).upper() == "CLOSED",
            "closable": link.link_type == "Pipe",
        }
        if g.has_edge(a, b):
            # parallel pipes: keep the lowest-resistance path, the one that carries flow
            if attrs["resistance"] < g[a][b]["resistance"]:
                g[a][b].update(attrs)
        else:
            g.add_edge(a, b, **attrs)
    return g


def check_elevation_informative(g: nx.Graph, min_span_m: float = 1.0) -> dict:
    """Is elevation actually usable on this network?

    Flat or single-value elevation (a transmission trunk modelled without real
    grade) makes the elevation factor inert: every similarity collapses to 1 and
    it contributes nothing regardless of its weight. Checked rather than
    assumed, because the failure is silent otherwise -- the weight slider would
    appear to do nothing and look like a bug.
    """
    el = np.array([d["elevation"] for _, d in g.nodes(data=True)
                   if d["node_type"] == "Junction" and np.isfinite(d["elevation"])])
    span = float(el.max() - el.min()) if el.size else np.nan
    return {"n_values": int(el.size), "span_m": span,
            "std_m": float(el.std()) if el.size else np.nan,
            "informative": bool(np.isfinite(span) and span >= min_span_m)}


# ==========================================================================
# connectivity policy -- one policy, both families
# ==========================================================================

def component_report(g: nx.Graph) -> dict:
    """Connected-component census of the whole network graph.

    The two method families used to disagree here: community detection raised on
    a disconnected graph while the spectral partitioner silently kept only the
    largest component, which left the dropped nodes with *no label at all* and
    would have quietly punched a hole in the junction partition that
    ``districts.yml`` is contractually required to cover exactly. One policy
    now, decided by the caller and always reported.
    """
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    return {"n_components": len(comps), "sizes": [len(c) for c in comps],
            "largest": set(comps[0]) if comps else set(),
            "connected": len(comps) <= 1}


def require_connected(g: nx.Graph, on_disconnected: str = "raise") -> tuple[nx.Graph, dict]:
    """Apply the connectivity policy and return the graph to partition.

    ``on_disconnected="raise"`` (default) refuses a disconnected network;
    ``"largest"`` restricts to the largest component and reports exactly which
    nodes were set aside, so they can be handled explicitly rather than
    vanishing.
    """
    rep = component_report(g)
    if rep["connected"]:
        rep["excluded"] = []
        return g, rep
    if on_disconnected == "largest":
        rep["excluded"] = sorted(set(g.nodes()) - rep["largest"])
        return g.subgraph(rep["largest"]).copy(), rep
    raise ValueError(
        "network graph is disconnected (%d components, sizes %s); "
        "partitioning assumes one network -- pass on_disconnected='largest' to "
        "partition the largest component and have the rest reported explicitly"
        % (rep["n_components"], rep["sizes"][:5]))
