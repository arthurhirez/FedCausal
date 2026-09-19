"""topology.py -- link endpoints, coordinates and base demands from the .inp.

Ported verbatim from the POC's ``bundles.topology``: the network views in
:mod:`fedwater.placement.figures` draw from persisted tables plus this, with no
wntr model in memory.
"""
from __future__ import annotations

import pathlib
import re

__all__ = ["topology"]


def topology(inp_path) -> dict:
    """Link endpoints, coordinates and base demands, straight from the .inp.

    Deliberately not via wntr. This runs BEFORE any world exists, so pulling
    in the solver's model object to answer "which nodes touch which links" is
    a dependency for nothing. The three sections read here are fixed-format
    and the parse is trivial; anything subtler is the solver's job.
    """
    text = pathlib.Path(inp_path).read_text()

    def section(name):
        m = re.search(r"\[" + name + r"\](.*?)(?=^\[|\Z)", text, re.S | re.M)
        if not m:
            return []
        out = []
        for line in m.group(1).splitlines():
            body = line.split(";")[0].strip()
            if body:
                out.append(body.split())
        return out

    links = {}
    for sec in ("PIPES", "PUMPS", "VALVES"):
        for f in section(sec):
            if len(f) >= 3:
                links[f[0]] = (f[1], f[2], sec.lower()[:-1])

    coords = {f[0]: (float(f[1]), float(f[2]))
              for f in section("COORDINATES") if len(f) >= 3}

    demand, elev = {}, {}
    for f in section("JUNCTIONS"):
        if len(f) >= 2:
            elev[f[0]] = float(f[1])
            demand[f[0]] = float(f[2]) if len(f) >= 3 else 0.0

    storage = ([f[0] for f in section("TANKS")]
               + [f[0] for f in section("RESERVOIRS")])
    return {"links": links, "coords": coords, "base_demand": demand,
            "elevation": elev, "storage": storage,
            "junctions": list(demand)}


def _graph(topo: dict):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(topo["junctions"])
    for u, v, _ in topo["links"].values():
        G.add_edge(u, v)
    return G

