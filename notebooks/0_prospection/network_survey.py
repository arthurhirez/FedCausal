"""EPANET corpus survey — static census + hydraulic feasibility assessment.

Design notes (read before changing anything)
--------------------------------------------
*Two readers, never one.* ``read_raw_sections`` scans the ``.inp`` text and counts
data lines per section; wntr builds the model. wntr is what we would *simulate*,
the raw scan is what is *in the file*. Every divergence becomes a row in the
``issues`` table rather than a silent loss. Sections wntr does not represent at
all (``[RULES]``, ``[EMITTERS]``, ``[SOURCES]``, ``[TAGS]`` in some versions) are
only visible through the raw scan.

*Type-aware attributes.* EPANET semantics differ per component: only junctions
carry demand and a demand pattern; reservoirs carry a *head* pattern; tanks carry
levels and possibly a volume curve; pumps carry speed/energy patterns. A single
flat element table would invent meaningless N/A cells and invite category errors,
so the element inventory is split into a thin common core (``elements``) and a
long per-type attribute table (``element_attrs``). The ``patterns`` table records
which *element types* reference each pattern, which is exactly where a
"consumption pattern on a reservoir" style error would show up.

*Capacity is a vector, not a number.* Static sums, storage, installed pumping and
a throughput min-cut are all reported side by side with the method that produced
them. The only defensible scalar answer -- the largest demand multiplier that
still respects a pressure floor -- requires solving repeatedly and lives in
tier 2 (``run_capacity``, off by default).

*Two smoke tiers by duration.* Most of this corpus ships ``Duration 0`` (a
steady-state snapshot) and some files carry zero controls. Extending such a file
to 24 h drains its tanks and produces massive negative pressures that say nothing
about the file's own validity. So ``as_authored`` (the file's own duration; a
single steady-state solve when that is 0) and ``extended`` (a forced 24 h run) are
reported as separate rows. Comparing them is the diagnosis.

Units: wntr converts everything to SI on load (m, m3/s, m/s). The native flow
unit from ``[OPTIONS] UNITS`` is reported alongside but never used for math.
"""
from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr

LPS_PER_M3S = 1000.0
SEC_PER_DAY = 86400.0
GRAVITY_KW = 9.81  # rho*g/1000 -> kW per (m3/s * m)

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class SurveyConfig:
    """Everything tunable. Defaults are the ones used for the corpus report."""

    # tier 1 -- static census: always on, no solver.
    emit_elements: bool = True          # per-element core + attribute tables
    emit_element_attrs: bool = True

    # tier 1b -- smoke test: cheap (< 1 s per network here), on by default.
    run_smoke: bool = True
    smoke_duration_h: int = 24
    smoke_timestep_s: int = 3600
    pressure_floor_m: float = 15.0      # feasibility threshold
    pressure_band_m: tuple[float, float] = (10.0, 50.0)  # ABNT NBR 12218 band

    # tier 2 -- capacity bisection: repeated solves, opt-in.
    run_capacity: bool = False
    capacity_multiplier_hi: float = 25.0
    capacity_tol: float = 0.02
    capacity_max_iter: int = 14

    # throughput proxy
    velocity_ceiling_ms: float = 1.5

    # graph metrics that scale badly
    max_nodes_for_diameter: int = 2000

    # naming
    artificial_prefix_patterns: tuple[str, ...] = (
        r"^[IO]-", r"^[IO]_",
    )


# --------------------------------------------------------------------------
# raw reader
# --------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^\s*\[([A-Za-z]+)\]")

# EPANET option/time keys that are two words; matched longest-first so that
# "DEMAND MULTIPLIER 0.45" does not parse as key="DEMAND", value="MULTIPLIER".
_MULTIWORD_KEYS = (
    "DEMAND MULTIPLIER", "EMITTER EXPONENT", "SPECIFIC GRAVITY",
    "MINIMUM PRESSURE", "REQUIRED PRESSURE", "PRESSURE EXPONENT",
    "HYDRAULIC TIMESTEP", "QUALITY TIMESTEP", "PATTERN TIMESTEP",
    "PATTERN START", "REPORT TIMESTEP", "REPORT START", "RULE TIMESTEP",
    "START CLOCKTIME", "DEMAND MODEL", "HEAD ERROR", "FLOW CHANGE",
    "GLOBAL BULK", "GLOBAL WALL", "BULK ORDER", "WALL ORDER", "TANK ORDER",
    "LIMITING POTENTIAL", "ROUGHNESS CORRELATION", "UNBALANCED",
)


def read_raw_sections(path: str | Path) -> dict:
    """Section-level scan of an ``.inp`` file, independent of wntr.

    Returns ``{SECTION: {"data": [[tokens...]], "raw": [str], "n_data": int,
    "n_comment": int, "occurrences": int}}`` plus a ``_meta`` entry with file
    facts (encoding surprises, line endings, section order, unknown sections).
    """
    path = Path(path)
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    try:
        text = data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        encoding = "latin-1 (not valid utf-8)"

    sections: dict[str, dict] = {}
    order: list[str] = []
    current = None
    n_preamble = 0

    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).upper()
            if current not in sections:
                sections[current] = {"data": [], "raw": [], "n_data": 0,
                                     "n_comment": 0, "occurrences": 0}
                order.append(current)
            sections[current]["occurrences"] += 1
            continue
        if current is None:
            if line.strip():
                n_preamble += 1
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            sections[current]["n_comment"] += 1
            continue
        payload = stripped.split(";", 1)[0].strip()
        if not payload:
            sections[current]["n_comment"] += 1
            continue
        sections[current]["n_data"] += 1
        sections[current]["raw"].append(payload)
        sections[current]["data"].append(payload.split())

    sections["_meta"] = {
        "encoding": encoding,
        "line_ending": "CRLF" if crlf else "LF",
        "n_lines": text.count("\n") + 1,
        "n_bytes": len(data),
        "section_order": order,
        "n_preamble_lines": n_preamble,
    }
    return sections


def parse_kv_lines(raw_lines: list[str]) -> list[tuple[str, str]]:
    """Parse an OPTIONS/TIMES-style block into (key, value) pairs."""
    out = []
    for line in raw_lines:
        norm = " ".join(line.split())
        upper = norm.upper()
        key = None
        for cand in _MULTIWORD_KEYS:
            if upper.startswith(cand):
                key, value = cand, norm[len(cand):].strip()
                break
        if key is None:
            parts = norm.split(None, 1)
            key = parts[0].upper()
            value = parts[1].strip() if len(parts) > 1 else ""
        out.append((key, value))
    return out


# --------------------------------------------------------------------------
# id / naming analysis
# --------------------------------------------------------------------------

def id_signature(name: str) -> str:
    """Collapse an ID to a shape: ``J-1`` -> ``A-#``, ``I-Pump-10`` -> ``A-A-#``."""
    sig = re.sub(r"[0-9]+", "#", str(name))
    sig = re.sub(r"[A-Za-z]+", "A", sig)
    return sig


def naming_profile(network: str, element_type: str, ids: list[str],
                   cfg: SurveyConfig) -> dict:
    """Charset, dominant ID shapes, numeric contiguity, samples."""
    if not ids:
        return {"network": network, "element_type": element_type, "n": 0}
    sigs = pd.Series([id_signature(i) for i in ids]).value_counts()
    charset = set("".join(str(i) for i in ids))
    nums = [int(m.group()) for i in ids
            if (m := re.search(r"\d+", str(i)))]
    contiguous = np.nan
    if nums:
        lo, hi = min(nums), max(nums)
        contiguous = len(set(nums)) == (hi - lo + 1)
    prefixes = pd.Series([re.match(r"^[^0-9]*", str(i)).group() for i in ids])
    return {
        "network": network,
        "element_type": element_type,
        "n": len(ids),
        "n_unique": len(set(ids)),
        "len_min": min(len(str(i)) for i in ids),
        "len_max": max(len(str(i)) for i in ids),
        "has_letters": any(c.isalpha() for c in charset),
        "has_digits": any(c.isdigit() for c in charset),
        "special_chars": "".join(sorted(c for c in charset
                                        if not c.isalnum())),
        "n_id_shapes": int(sigs.size),
        "top_shape": sigs.index[0],
        "top_shape_share": round(float(sigs.iloc[0] / len(ids)), 4),
        "shapes": " | ".join(f"{s}:{c}" for s, c in sigs.head(5).items()),
        "n_prefix_families": int(prefixes.nunique()),
        "top_prefixes": " | ".join(
            f"{p!r}:{c}" for p, c in prefixes.value_counts().head(5).items()),
        "numeric_contiguous": contiguous,
        "numeric_min": min(nums) if nums else np.nan,
        "numeric_max": max(nums) if nums else np.nan,
        "samples": ", ".join(str(i) for i in list(ids)[:5]),
    }


# --------------------------------------------------------------------------
# junction classification: consumer vs structural artefact
# --------------------------------------------------------------------------

def classify_junctions(wn, cfg: SurveyConfig) -> pd.DataFrame:
    """One row per junction, with the artificial-node verdict.

    Pumps and valves are *links* in EPANET, so they need two nodes. Model
    builders therefore insert zero-demand junctions on either side (the KY
    corpus names them ``I-Pump-3`` / ``O-RV-25``). Those nodes are not
    consumers: they must not receive demand, must not become federated clients,
    and must not enter per-junction demand statistics.

    Two independent verdicts are recorded so they can be cross-checked:
      * ``artificial_by_rule``  -- zero base demand AND every incident link is a
        pump or valve. Topological, name-agnostic.
      * ``artificial_by_name``  -- ID matches a configured prefix pattern.
    Disagreement between them is reported as an issue.
    """
    graph = wn.to_graph().to_undirected()
    name_res = [re.compile(p) for p in cfg.artificial_prefix_patterns]
    rows = []
    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        base = sum(float(ts.base_value or 0.0) for ts in j.demand_timeseries_list)
        patterns = [ts.pattern_name for ts in j.demand_timeseries_list
                    if ts.pattern_name]
        incident = list(graph.edges(jname, keys=True)) if graph.has_node(jname) else []
        types = set()
        for u, v, k in incident:
            try:
                types.add(wn.get_link(k).link_type)
            except KeyError:
                pass
        deg = len(incident)
        rows.append({
            "node": jname,
            "elevation_m": float(j.elevation),
            "base_demand_lps": base * LPS_PER_M3S,
            "n_demand_categories": len(j.demand_timeseries_list),
            "demand_patterns": ",".join(patterns),
            "has_demand_pattern": bool(patterns),
            "emitter_coefficient": float(j.emitter_coefficient or 0.0)
            if getattr(j, "emitter_coefficient", None) is not None else 0.0,
            "degree": deg,
            "incident_link_types": ",".join(sorted(types)),
            "tag": j.tag,
            # a node inserted only to terminate a pump/valve link: no demand,
            # touching such a link, and of degree <= 2 (the link plus at most
            # one pipe carrying on). The strict variant -- every incident link
            # is a pump/valve -- misses the common I-/O- pair, where the outer
            # side continues into a pipe.
            "artificial_by_rule": bool(
                abs(base) < 1e-12 and 0 < deg <= 2
                and bool(types & {"Pump", "Valve"})),
            "artificial_by_rule_strict": bool(
                abs(base) < 1e-12 and deg > 0
                and types.issubset({"Pump", "Valve"})),
            "artificial_by_name": any(r.search(jname) for r in name_res),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# element inventory
# --------------------------------------------------------------------------

def _pump_facts(wn, pump) -> dict:
    """Rated point / shutoff head for HEAD pumps, power for POWER pumps."""
    facts = {"pump_type": pump.pump_type, "power_kW": np.nan,
             "curve": None, "rated_flow_lps": np.nan, "rated_head_m": np.nan,
             "shutoff_head_m": np.nan, "n_curve_points": np.nan}
    if pump.pump_type.upper() == "POWER":
        facts["power_kW"] = float(pump.power) / 1000.0 if pump.power else np.nan
        return facts
    cname = pump.pump_curve_name
    facts["curve"] = cname
    if not cname:
        return facts
    pts = wn.get_curve(cname).points
    if not pts:
        return facts
    facts["n_curve_points"] = len(pts)
    flows = [p[0] for p in pts]
    heads = [p[1] for p in pts]
    facts["shutoff_head_m"] = float(max(heads))
    # design point: the single point of a 1-point curve, else the middle one
    idx = len(pts) // 2 if len(pts) > 1 else 0
    facts["rated_flow_lps"] = float(flows[idx]) * LPS_PER_M3S
    facts["rated_head_m"] = float(heads[idx])
    return facts


def element_tables(network: str, wn, junc: pd.DataFrame,
                   cfg: SurveyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(core, attrs). Core is one row per element; attrs is long per-type."""
    core, attrs = [], []

    def add_attr(etype, eid, attr, value):
        num = np.nan
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            num = float(value)
        attrs.append({"network": network, "element_type": etype, "id": eid,
                      "attribute": attr, "value": "" if value is None else str(value),
                      "value_num": num})

    coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    art = junc.set_index("node")[["artificial_by_rule", "artificial_by_name"]] \
        if not junc.empty else pd.DataFrame()

    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        row = art.loc[jname] if jname in art.index else None
        core.append({
            "network": network, "element_type": "junction", "subtype": None,
            "id": jname, "id_shape": id_signature(jname),
            "start_node": None, "end_node": None, "tag": j.tag,
            "x": coords[jname][0] if coords[jname] else np.nan,
            "y": coords[jname][1] if coords[jname] else np.nan,
            "is_artificial": bool(row["artificial_by_rule"]) if row is not None else None,
        })
        jr = junc.set_index("node").loc[jname] if jname in junc["node"].values else None
        add_attr("junction", jname, "elevation_m", float(j.elevation))
        if jr is not None:
            add_attr("junction", jname, "base_demand_lps", float(jr["base_demand_lps"]))
            add_attr("junction", jname, "n_demand_categories", int(jr["n_demand_categories"]))
            add_attr("junction", jname, "demand_patterns", jr["demand_patterns"])
            add_attr("junction", jname, "degree", int(jr["degree"]))

    for rname in wn.reservoir_name_list:
        r = wn.get_node(rname)
        core.append({"network": network, "element_type": "reservoir", "subtype": None,
                     "id": rname, "id_shape": id_signature(rname),
                     "start_node": None, "end_node": None, "tag": r.tag,
                     "x": coords[rname][0] if coords[rname] else np.nan,
                     "y": coords[rname][1] if coords[rname] else np.nan,
                     "is_artificial": False})
        add_attr("reservoir", rname, "base_head_m", float(r.base_head))
        add_attr("reservoir", rname, "head_pattern", r.head_pattern_name)

    for tname in wn.tank_name_list:
        t = wn.get_node(tname)
        if t.vol_curve_name:
            usable = np.nan  # curve-defined: integrate outside, flagged as issue
            try:
                usable = float(t.get_volume(t.max_level) - t.get_volume(t.min_level))
            except Exception:
                pass
        else:
            usable = np.pi / 4 * float(t.diameter) ** 2 * float(t.max_level - t.min_level)
        core.append({"network": network, "element_type": "tank", "subtype": None,
                     "id": tname, "id_shape": id_signature(tname),
                     "start_node": None, "end_node": None, "tag": t.tag,
                     "x": coords[tname][0] if coords[tname] else np.nan,
                     "y": coords[tname][1] if coords[tname] else np.nan,
                     "is_artificial": False})
        for k, v in [("elevation_m", float(t.elevation)),
                     ("init_level_m", float(t.init_level)),
                     ("min_level_m", float(t.min_level)),
                     ("max_level_m", float(t.max_level)),
                     ("diameter_m", float(t.diameter)),
                     ("min_vol_m3", float(t.min_vol or 0.0)),
                     ("vol_curve", t.vol_curve_name),
                     ("usable_volume_m3", usable)]:
            add_attr("tank", tname, k, v)

    for pname in wn.pipe_name_list:
        p = wn.get_link(pname)
        core.append({"network": network, "element_type": "pipe", "subtype": None,
                     "id": pname, "id_shape": id_signature(pname),
                     "start_node": p.start_node_name, "end_node": p.end_node_name,
                     "tag": p.tag, "x": np.nan, "y": np.nan, "is_artificial": False})
        for k, v in [("length_m", float(p.length)), ("diameter_m", float(p.diameter)),
                     ("roughness", float(p.roughness)),
                     ("minor_loss", float(p.minor_loss)),
                     ("initial_status", str(p.initial_status)),
                     ("check_valve", bool(getattr(p, "check_valve", False)))]:
            add_attr("pipe", pname, k, v)

    for pname in wn.pump_name_list:
        p = wn.get_link(pname)
        facts = _pump_facts(wn, p)
        core.append({"network": network, "element_type": "pump",
                     "subtype": facts["pump_type"], "id": pname,
                     "id_shape": id_signature(pname),
                     "start_node": p.start_node_name, "end_node": p.end_node_name,
                     "tag": p.tag, "x": np.nan, "y": np.nan, "is_artificial": False})
        for k, v in facts.items():
            add_attr("pump", pname, k, v)
        add_attr("pump", pname, "speed_pattern", p.speed_pattern_name)
        add_attr("pump", pname, "initial_status", str(p.initial_status))

    for vname in wn.valve_name_list:
        v = wn.get_link(vname)
        core.append({"network": network, "element_type": "valve",
                     "subtype": v.valve_type, "id": vname,
                     "id_shape": id_signature(vname),
                     "start_node": v.start_node_name, "end_node": v.end_node_name,
                     "tag": v.tag, "x": np.nan, "y": np.nan, "is_artificial": False})
        for k, val in [("valve_type", v.valve_type),
                       ("diameter_m", float(getattr(v, "diameter", np.nan) or np.nan)),
                       ("initial_setting", float(v.initial_setting)
                        if v.initial_setting is not None else np.nan),
                       ("minor_loss", float(getattr(v, "minor_loss", np.nan) or 0.0)),
                       ("initial_status", str(v.initial_status))]:
            add_attr("valve", vname, k, val)

    return pd.DataFrame(core), pd.DataFrame(attrs)


# --------------------------------------------------------------------------
# patterns: who references what
# --------------------------------------------------------------------------

def pattern_table(network: str, wn) -> pd.DataFrame:
    """One row per pattern, with the element types that reference it.

    This is where an EPANET category error would surface: a demand pattern is
    only meaningful on junctions, a head pattern only on reservoirs, a speed
    pattern only on pumps. The ``used_by_*`` columns make the actual usage
    explicit instead of assumed.
    """
    users: dict[str, dict[str, int]] = {}

    def bump(pat, kind):
        if not pat:
            return
        users.setdefault(pat, {}).setdefault(kind, 0)
        users[pat][kind] += 1

    for n in wn.junction_name_list:
        for ts in wn.get_node(n).demand_timeseries_list:
            bump(ts.pattern_name, "junction_demand")
    for n in wn.reservoir_name_list:
        bump(wn.get_node(n).head_pattern_name, "reservoir_head")
    for n in wn.tank_name_list:
        bump(getattr(wn.get_node(n), "pattern_name", None), "tank")
    for l in wn.pump_name_list:
        bump(wn.get_link(l).speed_pattern_name, "pump_speed")
        bump(getattr(wn.get_link(l), "energy_pattern", None), "pump_energy")

    rows = []
    ts_s = float(wn.options.time.pattern_timestep or 0)
    for pname in wn.pattern_name_list:
        mult = np.asarray(wn.get_pattern(pname).multipliers, dtype=float)
        u = users.get(pname, {})
        rows.append({
            "network": network, "pattern": pname,
            "length": int(mult.size),
            "pattern_timestep_s": ts_s,
            "span_h": mult.size * ts_s / 3600.0 if ts_s else np.nan,
            "min": float(mult.min()) if mult.size else np.nan,
            "max": float(mult.max()) if mult.size else np.nan,
            "mean": float(mult.mean()) if mult.size else np.nan,
            "peak_factor": float(mult.max() / mult.mean())
            if mult.size and mult.mean() else np.nan,
            "n_users_total": int(sum(u.values())),
            "used_by": " | ".join(f"{k}:{v}" for k, v in sorted(u.items())) or "UNUSED",
            "used_by_junction_demand": u.get("junction_demand", 0),
            "used_by_reservoir_head": u.get("reservoir_head", 0),
            "used_by_pump_speed": u.get("pump_speed", 0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------

def topology_table(network: str, wn, junc: pd.DataFrame,
                   cfg: SurveyConfig) -> tuple[dict, nx.Graph]:
    """Connectivity, dead ends, cut structure. Cut structure matters because
    articulation points and bridges constrain any later district partition."""
    g = wn.to_graph().to_undirected()
    simple = nx.Graph()
    simple.add_nodes_from(g.nodes())
    simple.add_edges_from((u, v) for u, v, _ in g.edges(keys=True))

    sources = list(wn.reservoir_name_list) + list(wn.tank_name_list)
    comps = list(nx.connected_components(simple))
    comps.sort(key=len, reverse=True)
    largest = simple.subgraph(comps[0]) if comps else simple

    unreachable = []
    if sources:
        reach = set()
        for s in sources:
            if simple.has_node(s):
                reach |= nx.node_connected_component(simple, s)
        unreachable = [n for n in wn.junction_name_list if n not in reach]

    degrees = np.array([d for _, d in simple.degree()])
    diameter = np.nan
    if largest.number_of_nodes() <= cfg.max_nodes_for_diameter:
        try:
            diameter = float(nx.diameter(largest))
        except Exception:
            pass

    row = {
        "network": network,
        "n_nodes": simple.number_of_nodes(),
        "n_edges": simple.number_of_edges(),
        "n_components": len(comps),
        "largest_component_frac": round(len(comps[0]) / simple.number_of_nodes(), 4)
        if comps and simple.number_of_nodes() else np.nan,
        "n_sources": len(sources),
        "n_reservoirs": len(wn.reservoir_name_list),
        "n_tanks": len(wn.tank_name_list),
        "n_junctions_unreachable_from_source": len(unreachable),
        "unreachable_sample": ", ".join(unreachable[:5]),
        "degree_mean": round(float(degrees.mean()), 3) if degrees.size else np.nan,
        "degree_max": int(degrees.max()) if degrees.size else 0,
        "n_dead_ends": int((degrees == 1).sum()),
        "dead_end_frac": round(float((degrees == 1).mean()), 4) if degrees.size else np.nan,
        "n_articulation_points": len(list(nx.articulation_points(largest))),
        "n_bridges": len(list(nx.bridges(largest))),
        "diameter_hops": diameter,
        "n_coords_missing": int(sum(
            1 for n in wn.node_name_list
            if not wn.get_node(n).coordinates
            or wn.get_node(n).coordinates == (0.0, 0.0))),
    }
    return row, simple


# --------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------

def _throughput_mincut(wn, cfg: SurveyConfig) -> float:
    """Upper bound on deliverable flow: min cut separating sources from demand
    nodes, with pipe capacity = area x velocity ceiling.

    A structural bound only -- it ignores head. Reported as such.
    """
    v = cfg.velocity_ceiling_ms
    G = nx.DiGraph()
    SRC, SNK = "__src__", "__snk__"

    def cap_of(link):
        d = getattr(link, "diameter", None)
        if d:
            return np.pi / 4 * float(d) ** 2 * v
        return 1e9

    for lname in wn.link_name_list:
        l = wn.get_link(lname)
        c = cap_of(l)
        for a, b in [(l.start_node_name, l.end_node_name),
                     (l.end_node_name, l.start_node_name)]:
            if G.has_edge(a, b):
                G[a][b]["capacity"] += c
            else:
                G.add_edge(a, b, capacity=c)

    for s in list(wn.reservoir_name_list) + list(wn.tank_name_list):
        G.add_edge(SRC, s, capacity=1e12)
    n_sinks = 0
    for n in wn.junction_name_list:
        j = wn.get_node(n)
        base = sum(float(ts.base_value or 0.0) for ts in j.demand_timeseries_list)
        if base > 0 and G.has_node(n):
            G.add_edge(n, SNK, capacity=1e12)
            n_sinks += 1
    if not G.has_node(SRC) or not n_sinks:
        return np.nan
    try:
        value, _ = nx.maximum_flow(G, SRC, SNK)
        return float(value) * LPS_PER_M3S
    except Exception:
        return np.nan


def capacity_table(network: str, wn, junc: pd.DataFrame,
                   elements: pd.DataFrame, attrs: pd.DataFrame,
                   cfg: SurveyConfig) -> pd.DataFrame:
    """The capacity vector. Each row names its own method and caveat."""
    rows = []

    def add(metric, value, unit, method, caveat=""):
        rows.append({"network": network, "metric": metric,
                     "value": float(value) if value is not None else np.nan,
                     "unit": unit, "method": method, "caveat": caveat})

    mult = float(wn.options.hydraulic.demand_multiplier or 1.0)
    base_lps = float(junc["base_demand_lps"].clip(lower=0).sum()) if not junc.empty else 0.0
    neg_lps = float(junc["base_demand_lps"].clip(upper=0).sum()) if not junc.empty else 0.0

    add("demand_base_sum", base_lps, "L/s",
        "sum of positive base demands over ALL demand categories")
    add("demand_base_sum_with_multiplier", base_lps * mult, "L/s",
        f"base sum x DEMAND MULTIPLIER ({mult})",
        "the multiplier is a global file-level scale and is easy to miss")
    add("demand_base_sum_cmd", base_lps * mult * SEC_PER_DAY / 1000.0, "m3/day",
        "base sum x multiplier x 86400", "comparable to the readme's CMD/MGD claim")
    if neg_lps < 0:
        add("demand_negative_sum", neg_lps, "L/s",
            "sum of negative base demands", "negative demand = injection, not consumption")

    # pattern-weighted demand envelope
    steps = None
    total_by_step = None
    for n in wn.junction_name_list:
        j = wn.get_node(n)
        for ts in j.demand_timeseries_list:
            b = float(ts.base_value or 0.0)
            if b == 0:
                continue
            if ts.pattern_name:
                m = np.asarray(wn.get_pattern(ts.pattern_name).multipliers, float)
            else:
                m = np.ones(1)
            if total_by_step is None:
                steps = m.size
                total_by_step = np.zeros(steps)
            if m.size != steps:  # ragged patterns: tile to the common length
                reps = int(np.ceil(steps / m.size))
                m = np.tile(m, reps)[:steps]
            total_by_step += b * m
    if total_by_step is not None:
        add("demand_pattern_peak", float(total_by_step.max()) * LPS_PER_M3S * mult,
            "L/s", "max over pattern steps of sum(base x pattern) x multiplier")
        add("demand_pattern_mean", float(total_by_step.mean()) * LPS_PER_M3S * mult,
            "L/s", "mean over pattern steps")
        add("demand_pattern_peak_factor",
            float(total_by_step.max() / total_by_step.mean())
            if total_by_step.mean() else np.nan, "-",
            "system peak factor from the patterns as shipped")

    # storage
    if not attrs.empty:
        usable = attrs[(attrs.element_type == "tank") &
                       (attrs.attribute == "usable_volume_m3")]["value_num"]
        vol = float(usable.sum()) if not usable.empty else 0.0
        add("storage_usable_volume", vol, "m3",
            "sum over tanks of pi/4 d^2 (max_level - min_level), or volume curve")
        if base_lps * mult > 0:
            add("storage_hours_of_mean_demand",
                vol / (base_lps * mult / 1000.0) / 3600.0, "h",
                "usable volume / mean demand",
                "how long storage alone covers demand")

    # supply
    heads = [float(wn.get_node(r).base_head) for r in wn.reservoir_name_list]
    if heads:
        add("reservoir_head_min", min(heads), "m", "[RESERVOIRS] Head")
        add("reservoir_head_max", max(heads), "m", "[RESERVOIRS] Head")

    if not attrs.empty:
        pa = attrs[attrs.element_type == "pump"]
        power = pa[pa.attribute == "power_kW"]["value_num"].dropna()
        rated = pa[pa.attribute == "rated_flow_lps"]["value_num"].dropna()
        shut = pa[pa.attribute == "shutoff_head_m"]["value_num"].dropna()
        if not power.empty:
            add("pump_installed_power", float(power.sum()), "kW",
                "sum of POWER-defined pumps",
                "POWER pumps have no curve, so no rated flow can be derived")
        if not rated.empty:
            add("pump_rated_flow_sum", float(rated.sum()), "L/s",
                "sum of design-point flows of HEAD-curve pumps",
                "design point = middle curve point; not a simultaneous-operation guarantee")
        if not shut.empty:
            add("pump_shutoff_head_max", float(shut.max()), "m",
                "max head over pump curves")

    # throughput
    add("throughput_mincut", _throughput_mincut(wn, cfg), "L/s",
        f"max-flow/min-cut, pipe capacity = area x {cfg.velocity_ceiling_ms} m/s",
        "structural bound; ignores head, pumps and valve settings")

    total_len = 0.0
    if not attrs.empty:
        pl = attrs[(attrs.element_type == "pipe") & (attrs.attribute == "length_m")]
        total_len = float(pl["value_num"].sum())
    add("pipe_length_total", total_len / 1000.0, "km", "sum of [PIPES] Length",
        "comparable to the readme's km/miles claim")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# hydraulic tiers
# --------------------------------------------------------------------------

def _run_once(wn, duration_s, timestep_s, cfg, consumers: set | None = None):
    """Solve and summarise. Returns (dict, results|None).

    The EPANET toolkit writes scratch ``temp.inp`` / ``temp.rpt`` files beside
    the process working directory. Left alone that litters the repo (the same
    hygiene problem ``fedwater_architecture.md`` records), so every solve runs
    inside a throwaway directory.
    """
    import copy
    import tempfile

    w = copy.deepcopy(wn)
    if duration_s is not None:
        w.options.time.duration = duration_s
        w.options.time.hydraulic_timestep = timestep_s
        w.options.time.report_timestep = timestep_s
    t0 = time.time()
    out = {"duration_h": (duration_s if duration_s is not None
                          else w.options.time.duration) / 3600.0,
           "converged": False, "runtime_s": np.nan, "error": ""}
    try:
        with warnings.catch_warnings(record=True) as caught, \
                tempfile.TemporaryDirectory() as scratch:
            warnings.simplefilter("always")
            cwd = os.getcwd()
            try:
                os.chdir(scratch)          # toolkit scratch files land here
                res = wntr.sim.EpanetSimulator(w).run_sim()
            finally:
                os.chdir(cwd)
        out["runtime_s"] = round(time.time() - t0, 3)
        out["converged"] = True
        out["warnings"] = " | ".join(sorted({str(c.message)[:120] for c in caught}))
    except Exception as exc:
        out["runtime_s"] = round(time.time() - t0, 3)
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out, None

    junctions = [n for n in w.junction_name_list if n in res.node["pressure"].columns]
    p = res.node["pressure"][junctions]
    lo, hi = cfg.pressure_band_m

    def stats(frame, suffix):
        """Pressure summary over a node subset.

        Reported twice: over ALL junctions, and over consumer junctions only.
        The difference matters -- in this corpus most sub-atmospheric readings
        sit on artificial pump-suction nodes (``I-Pump-*``), which are modelling
        scaffolding with no demand and no customer behind them. Judging a file
        infeasible on those is wrong; judging it feasible while real consumers
        starve would be worse. Hence both.
        """
        if frame.shape[1] == 0:
            return {}
        vals = frame.to_numpy()
        mins = frame.min(axis=0)
        return {
            f"n_nodes{suffix}": int(frame.shape[1]),
            f"n_node_hours{suffix}": int(vals.size),
            f"pressure_min_m{suffix}": float(np.nanmin(vals)),
            f"pressure_median_m{suffix}": float(np.nanmedian(vals)),
            f"pressure_max_m{suffix}": float(np.nanmax(vals)),
            f"n_negative_node_hours{suffix}": int((vals < 0).sum()),
            f"frac_negative{suffix}": round(float((vals < 0).mean()), 5),
            f"n_nodes_ever_negative{suffix}": int((mins < 0).sum()),
            f"frac_below_floor{suffix}": round(
                float((vals < cfg.pressure_floor_m).mean()), 5),
            f"frac_in_band{suffix}": round(
                float(((vals >= lo) & (vals <= hi)).mean()), 5),
            f"feasible_at_floor{suffix}": bool(
                np.nanmin(vals) >= cfg.pressure_floor_m),
            f"worst_nodes{suffix}": ", ".join(mins.sort_values().head(3).index),
        }

    out.update(stats(p, ""))
    if consumers is not None:
        cols = [c for c in junctions if c in consumers]
        out.update(stats(p[cols], "_consumer"))
    return out, res


def smoke_table(network: str, wn, cfg: SurveyConfig,
                consumers: set | None = None) -> pd.DataFrame:
    """Two tiers, deliberately separate rows.

    ``as_authored`` uses the file's own duration -- for a ``Duration 0`` file
    that is a single steady-state solve, which is the only run the author ever
    validated. ``extended`` forces ``smoke_duration_h``. A file that is fine in
    the first and catastrophic in the second is not broken: it has no operating
    controls, and extending it drained its tanks. That distinction is the whole
    reason both rows exist.
    """
    rows = []
    authored = float(wn.options.time.duration or 0.0)
    r, _ = _run_once(wn, None, None, cfg, consumers)
    r.update({"network": network, "tier": "as_authored",
              "steady_state": authored == 0.0})
    rows.append(r)

    # always emitted, even when the file's own duration already equals the smoke
    # duration, so the corpus-level table has the same two rows for every network
    r2, _ = _run_once(wn, cfg.smoke_duration_h * 3600, cfg.smoke_timestep_s,
                     cfg, consumers)
    r2.update({"network": network, "tier": "extended", "steady_state": False,
               "same_as_authored": authored == cfg.smoke_duration_h * 3600})
    rows.append(r2)
    return pd.DataFrame(rows)


def capacity_bisection(network: str, wn, cfg: SurveyConfig) -> pd.DataFrame:
    """Tier 2: largest global demand multiplier that keeps min pressure >= floor.

    This is the only honest scalar answer to "maximum capacity", and it costs
    ``capacity_max_iter`` solves. Off by default.
    """
    import copy

    def feasible(mult):
        w = copy.deepcopy(wn)
        w.options.hydraulic.demand_multiplier = mult
        out, _ = _run_once(w, cfg.smoke_duration_h * 3600, cfg.smoke_timestep_s, cfg)
        return out.get("converged") and out.get("pressure_min_m", -1e9) >= cfg.pressure_floor_m, out

    ok0, out0 = feasible(float(wn.options.hydraulic.demand_multiplier or 1.0))
    if not ok0:
        return pd.DataFrame([{
            "network": network, "max_multiplier": np.nan, "n_solves": 1,
            "status": "infeasible at the file's own multiplier",
            "pressure_min_m": out0.get("pressure_min_m", np.nan)}])

    lo, hi = 1.0, cfg.capacity_multiplier_hi
    ok_hi, _ = feasible(hi)
    n = 2
    if ok_hi:
        return pd.DataFrame([{
            "network": network, "max_multiplier": hi, "n_solves": n,
            "status": f"still feasible at hi={hi} -- bound not found",
            "pressure_min_m": np.nan}])
    for _ in range(cfg.capacity_max_iter):
        if (hi - lo) / max(lo, 1e-9) < cfg.capacity_tol:
            break
        mid = 0.5 * (lo + hi)
        ok, out = feasible(mid)
        n += 1
        if ok:
            lo = mid
        else:
            hi = mid
    base = float(wn.options.hydraulic.demand_multiplier or 1.0)
    junc_sum = sum(sum(float(ts.base_value or 0.0)
                       for ts in wn.get_node(j).demand_timeseries_list)
                   for j in wn.junction_name_list) * LPS_PER_M3S
    return pd.DataFrame([{
        "network": network, "max_multiplier": round(lo, 4), "n_solves": n,
        "status": "bracketed",
        "max_demand_lps": round(junc_sum * lo, 3),
        "headroom_vs_authored": round(lo / base, 3) if base else np.nan,
        "pressure_min_m": np.nan}])


# --------------------------------------------------------------------------
# the survey
# --------------------------------------------------------------------------

def survey_network(path: str | Path, cfg: SurveyConfig | None = None) -> dict:
    """Full survey of one ``.inp``. Returns a dict of DataFrames.

    Keys: inventory, options, sections, components, elements, element_attrs,
    naming, patterns, capacity, topology, groups, smoke, capacity_limit, issues.
    Never raises on a bad file: the failure lands in ``issues`` and the other
    tables come back empty.
    """
    cfg = cfg or SurveyConfig()
    path = Path(path)
    network = path.stem
    issues: list[dict] = []

    def issue(check, severity, message):
        issues.append({"network": network, "check": check,
                       "severity": severity, "message": str(message)[:400]})

    raw = read_raw_sections(path)
    meta = raw.pop("_meta")
    sections = pd.DataFrame([
        {"network": network, "section": s, "n_data": d["n_data"],
         "n_comment": d["n_comment"], "occurrences": d["occurrences"]}
        for s, d in raw.items()])

    if meta["encoding"] != "utf-8":
        issue("encoding", "warn", meta["encoding"])
    if meta["n_preamble_lines"]:
        issue("preamble", "warn",
              f"{meta['n_preamble_lines']} non-empty lines before the first section")
    for s, d in raw.items():
        if d["occurrences"] > 1:
            issue("duplicate_section", "info",
                  f"[{s}] appears {d['occurrences']} times")

    # ---- wntr load
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            wn = wntr.network.WaterNetworkModel(str(path))
        except Exception as exc:
            issue("wntr_load", "error", f"{type(exc).__name__}: {exc}")
            return {"inventory": pd.DataFrame([{"network": network,
                                                "wntr_ok": False}]),
                    "sections": sections,
                    "issues": pd.DataFrame(issues)}
    for c in {str(x.message)[:200] for x in caught}:
        issue("wntr_load_warning", "warn", c)

    # ---- raw vs wntr cross-check
    raw_counts = {s: d["n_data"] for s, d in raw.items()}
    wntr_counts = {
        "JUNCTIONS": len(wn.junction_name_list),
        "RESERVOIRS": len(wn.reservoir_name_list),
        "TANKS": len(wn.tank_name_list),
        "PIPES": len(wn.pipe_name_list),
        "PUMPS": len(wn.pump_name_list),
        "VALVES": len(wn.valve_name_list),
        "PATTERNS": len(wn.pattern_name_list),
        "CURVES": len(wn.curve_name_list),
    }
    for sec, n_wntr in wntr_counts.items():
        n_raw = raw_counts.get(sec, 0)
        if sec in ("PATTERNS", "CURVES"):
            continue  # multi-line per entity; count comparison is meaningless
        if n_raw != n_wntr:
            issue("raw_vs_wntr_count", "warn",
                  f"[{sec}] raw={n_raw} wntr={n_wntr}")
    for sec in ("RULES", "CONTROLS", "EMITTERS", "SOURCES", "QUALITY",
                "MIXING", "STATUS", "DEMANDS", "TAGS", "ENERGY"):
        n_raw = raw_counts.get(sec, 0)
        if n_raw:
            issue("section_present", "info", f"[{sec}] carries {n_raw} data lines")

    # ---- options / times, native and normalised
    opt_rows = []
    for sec in ("OPTIONS", "TIMES", "REPORT", "ENERGY", "REACTIONS"):
        for k, v in parse_kv_lines(raw.get(sec, {}).get("raw", [])):
            opt_rows.append({"network": network, "source": "raw",
                             "section": sec, "key": k, "value": v})
    h, t = wn.options.hydraulic, wn.options.time
    for k, v in list(vars(h).items()) + list(vars(t).items()):
        opt_rows.append({"network": network, "source": "wntr",
                         "section": "options", "key": k, "value": str(v)})
    options = pd.DataFrame(opt_rows)

    mult = float(h.demand_multiplier or 1.0)
    if abs(mult - 1.0) > 1e-9:
        issue("demand_multiplier", "warn",
              f"global DEMAND MULTIPLIER = {mult} -- scales every demand in the file")
    if float(t.duration or 0) == 0:
        issue("steady_state", "warn",
              "Duration 0: the file is a single-snapshot model, not a time series")
    if len(wn.control_name_list) == 0 and (wn.tank_name_list or wn.pump_name_list):
        issue("no_controls", "warn",
              "tanks/pumps present but zero controls -- an extended run has no "
              "operating rules and will drain storage")
    if t.pattern_timestep and t.hydraulic_timestep and \
            float(t.pattern_timestep) != float(t.hydraulic_timestep):
        issue("timestep_mismatch", "info",
              f"pattern_timestep={t.pattern_timestep}s != "
              f"hydraulic_timestep={t.hydraulic_timestep}s")

    # ---- junctions
    junc = classify_junctions(wn, cfg)
    n_art_rule = int(junc["artificial_by_rule"].sum()) if not junc.empty else 0
    n_art_name = int(junc["artificial_by_name"].sum()) if not junc.empty else 0
    if not junc.empty:
        disagree = junc[junc["artificial_by_rule"] != junc["artificial_by_name"]]
        if len(disagree):
            issue("artificial_node_disagreement", "info",
                  f"{len(disagree)} junctions where the topological rule and the "
                  f"name pattern disagree, e.g. {', '.join(disagree['node'].head(5))}")
        multi = junc[junc["n_demand_categories"] > 1]
        if len(multi):
            issue("multi_category_demand", "warn",
                  f"{len(multi)} junctions carry >1 demand category -- summing only "
                  f"the first would undercount")
        if (junc["base_demand_lps"] < 0).any():
            issue("negative_demand", "warn",
                  f"{int((junc['base_demand_lps'] < 0).sum())} junctions with "
                  "negative base demand (injection)")
        if junc["base_demand_lps"].abs().sum() == 0:
            issue("zero_demand", "error", "no junction carries any base demand")
        n_nopat = int((~junc["has_demand_pattern"] &
                       (junc["base_demand_lps"] > 0)).sum())
        if n_nopat:
            issue("demand_without_pattern", "info",
                  f"{n_nopat} demand-bearing junctions have no pattern (constant demand)")

    # ---- elements
    if cfg.emit_elements:
        elements, attrs = element_tables(network, wn, junc, cfg)
    else:
        elements, attrs = pd.DataFrame(), pd.DataFrame()
    if not cfg.emit_element_attrs:
        attrs_out = pd.DataFrame()
    else:
        attrs_out = attrs

    # id collisions across types
    if not elements.empty:
        dup = elements.groupby("id")["element_type"].nunique()
        clash = dup[dup > 1]
        if len(clash):
            issue("id_collision_across_types", "warn",
                  f"{len(clash)} IDs used by more than one element type: "
                  f"{', '.join(clash.index[:5])}")

    # ---- naming
    naming = pd.DataFrame([
        naming_profile(network, et, list(ids), cfg) for et, ids in [
            ("junction", wn.junction_name_list),
            ("reservoir", wn.reservoir_name_list),
            ("tank", wn.tank_name_list),
            ("pipe", wn.pipe_name_list),
            ("pump", wn.pump_name_list),
            ("valve", wn.valve_name_list),
            ("pattern", wn.pattern_name_list),
            ("curve", wn.curve_name_list),
        ] if ids])

    # ---- components
    comp_rows = [
        {"network": network, "element_type": "junction",
         "count": len(wn.junction_name_list),
         "n_demand_bearing": int((junc["base_demand_lps"] > 0).sum()) if not junc.empty else 0,
         "n_artificial_rule": n_art_rule, "n_artificial_name": n_art_name,
         "detail": ""},
        {"network": network, "element_type": "reservoir",
         "count": len(wn.reservoir_name_list), "detail": ""},
        {"network": network, "element_type": "tank",
         "count": len(wn.tank_name_list),
         "detail": f"{sum(1 for t in wn.tank_name_list if wn.get_node(t).vol_curve_name)} "
                   "with volume curve"},
        {"network": network, "element_type": "pipe",
         "count": len(wn.pipe_name_list),
         "detail": f"{sum(1 for p in wn.pipe_name_list if str(wn.get_link(p).initial_status).upper() == 'CLOSED')} "
                   "initially closed"},
        {"network": network, "element_type": "pump",
         "count": len(wn.pump_name_list),
         "detail": " | ".join(
             f"{k}:{v}" for k, v in pd.Series(
                 [wn.get_link(p).pump_type for p in wn.pump_name_list]
             ).value_counts().items()) if wn.pump_name_list else ""},
        {"network": network, "element_type": "valve",
         "count": len(wn.valve_name_list),
         "detail": " | ".join(
             f"{k}:{v}" for k, v in pd.Series(
                 [wn.get_link(v).valve_type for v in wn.valve_name_list]
             ).value_counts().items()) if wn.valve_name_list else ""},
        {"network": network, "element_type": "pattern",
         "count": len(wn.pattern_name_list), "detail": ""},
        {"network": network, "element_type": "curve",
         "count": len(wn.curve_name_list), "detail": ""},
        {"network": network, "element_type": "control",
         "count": len(wn.control_name_list),
         "detail": f"raw [RULES] lines: {raw_counts.get('RULES', 0)}"},
    ]
    components = pd.DataFrame(comp_rows)

    # ---- patterns
    patterns = pattern_table(network, wn)
    if not patterns.empty:
        unused = patterns[patterns["n_users_total"] == 0]
        if len(unused):
            issue("unused_pattern", "info",
                  f"{len(unused)} patterns referenced by nothing: "
                  f"{', '.join(unused['pattern'].head(5))}")
        lens = patterns["length"].unique()
        if len(lens) > 1:
            issue("ragged_patterns", "info",
                  f"pattern lengths differ: {sorted(lens)}")

    # ---- topology
    topo_row, simple = topology_table(network, wn, junc, cfg)
    topology = pd.DataFrame([topo_row])
    if topo_row["n_components"] > 1:
        issue("disconnected", "warn",
              f"{topo_row['n_components']} connected components; largest holds "
              f"{topo_row['largest_component_frac']:.1%} of nodes")
    if topo_row["n_junctions_unreachable_from_source"]:
        issue("unreachable_junctions", "error",
              f"{topo_row['n_junctions_unreachable_from_source']} junctions cannot "
              f"reach any source ({topo_row['unreachable_sample']})")
    if topo_row["n_coords_missing"]:
        issue("missing_coordinates", "info",
              f"{topo_row['n_coords_missing']} nodes without usable coordinates")

    # ---- pre-existing groupings (assessment only, NOT federated districts)
    tag_counts = {}
    for n in wn.node_name_list:
        tg = wn.get_node(n).tag
        if tg:
            tag_counts[tg] = tag_counts.get(tg, 0) + 1
    groups = pd.DataFrame([
        {"network": network, "group_source": "TAGS", "group": g, "n_nodes": c}
        for g, c in sorted(tag_counts.items())])
    if not groups.empty:
        covered = groups["n_nodes"].sum()
        issue("existing_groups", "info",
              f"[TAGS] defines {len(groups)} node groups covering {covered}/"
              f"{len(wn.node_name_list)} nodes -- a modeller's DMA labelling, "
              "not a federated partition")

    # ---- capacity
    capacity = capacity_table(network, wn, junc, elements, attrs, cfg)

    # ---- hydraulics
    consumers = set(junc.loc[(junc["base_demand_lps"] > 0) &
                             (~junc["artificial_by_rule"]), "node"]) \
        if not junc.empty else set()
    smoke = (smoke_table(network, wn, cfg, consumers)
             if cfg.run_smoke else pd.DataFrame())
    if not smoke.empty:
        for _, r in smoke.iterrows():
            if not r.get("converged"):
                issue(f"smoke_{r['tier']}", "error", r.get("error", "did not converge"))
            else:
                n_neg_c = r.get("n_negative_node_hours_consumer", 0) or 0
                if n_neg_c:
                    issue(f"smoke_{r['tier']}", "error",
                          f"{int(n_neg_c)} negative node-hours at CONSUMER junctions "
                          f"({r.get('frac_negative_consumer', float('nan')):.1%}), min "
                          f"{r.get('pressure_min_m_consumer', float('nan')):.1f} m, "
                          f"worst: {r.get('worst_nodes_consumer', '')}")
                elif r.get("n_negative_node_hours", 0):
                    issue(f"smoke_{r['tier']}", "warn",
                          f"{int(r['n_negative_node_hours'])} negative node-hours, but "
                          f"only at non-consumer nodes ({r['worst_nodes']}) -- modelling "
                          f"scaffolding, not unserved demand")
                elif not r.get("feasible_at_floor_consumer",
                               r.get("feasible_at_floor", True)):
                    issue(f"smoke_{r['tier']}", "warn",
                          f"min pressure {r['pressure_min_m']:.1f} m below floor "
                          f"{cfg.pressure_floor_m} m")
        if len(smoke) == 2 and all(smoke["converged"]):
            a = smoke[smoke.tier == "as_authored"].iloc[0]
            e = smoke[smoke.tier == "extended"].iloc[0]
            if (a.get("n_negative_node_hours_consumer", 0) == 0
                    and e.get("n_negative_node_hours_consumer", 0) > 0):
                issue("extension_artefact", "warn",
                      "feasible as authored but not when extended -- the file has no "
                      "operating controls for a time series, this is not evidence the "
                      "network itself is infeasible")

    capacity_limit = (capacity_bisection(network, wn, cfg)
                      if cfg.run_capacity else pd.DataFrame())

    # ---- inventory row
    pipe_len_km = float(capacity.loc[capacity.metric == "pipe_length_total",
                                     "value"].iloc[0]) if not capacity.empty else np.nan
    demand_cmd = capacity.loc[capacity.metric == "demand_base_sum_cmd", "value"]
    inv = {
        "network": network, "file": path.name, "wntr_ok": True,
        "wntr_version": wntr.__version__,
        "n_bytes": meta["n_bytes"], "line_ending": meta["line_ending"],
        "flow_units_native": h.inpfile_units,
        "headloss": h.headloss,
        "demand_model": h.demand_model,
        "demand_multiplier": mult,
        "emitter_exponent": h.emitter_exponent,
        "default_pattern": h.pattern,
        "duration_h": float(t.duration or 0) / 3600.0,
        "hydraulic_timestep_s": float(t.hydraulic_timestep or 0),
        "pattern_timestep_s": float(t.pattern_timestep or 0),
        "report_timestep_s": float(t.report_timestep or 0),
        "n_junctions": len(wn.junction_name_list),
        "n_junctions_demand_bearing": int((junc["base_demand_lps"] > 0).sum())
        if not junc.empty else 0,
        "n_junctions_artificial": n_art_rule,
        "n_reservoirs": len(wn.reservoir_name_list),
        "n_tanks": len(wn.tank_name_list),
        "n_pipes": len(wn.pipe_name_list),
        "n_pumps": len(wn.pump_name_list),
        "n_pumps_power_defined": sum(
            1 for p in wn.pump_name_list
            if wn.get_link(p).pump_type.upper() == "POWER"),
        "n_valves": len(wn.valve_name_list),
        "valve_types": " | ".join(sorted(
            {wn.get_link(v).valve_type for v in wn.valve_name_list})),
        "n_patterns": len(wn.pattern_name_list),
        "n_curves": len(wn.curve_name_list),
        "n_controls": len(wn.control_name_list),
        "n_rules_raw": raw_counts.get("RULES", 0),
        "pipe_length_km": round(pipe_len_km, 3),
        "demand_cmd": round(float(demand_cmd.iloc[0]), 1) if len(demand_cmd) else np.nan,
        "storage_m3": float(capacity.loc[capacity.metric == "storage_usable_volume",
                                         "value"].iloc[0])
        if (capacity.metric == "storage_usable_volume").any() else np.nan,
        "n_components": topo_row["n_components"],
        "n_dead_ends": topo_row["n_dead_ends"],
        "n_articulation_points": topo_row["n_articulation_points"],
        "n_bridges": topo_row["n_bridges"],
        "n_existing_groups": len(groups),
        "n_issues_error": sum(1 for i in issues if i["severity"] == "error"),
        "n_issues_warn": sum(1 for i in issues if i["severity"] == "warn"),
    }
    if not smoke.empty:
        for _, r in smoke.iterrows():
            inv[f"{r['tier']}_converged"] = bool(r["converged"])
            inv[f"{r['tier']}_pmin_m"] = r.get("pressure_min_m", np.nan)
            inv[f"{r['tier']}_pmin_m_consumer"] = r.get("pressure_min_m_consumer", np.nan)
            inv[f"{r['tier']}_frac_negative"] = r.get("frac_negative", np.nan)
            inv[f"{r['tier']}_frac_negative_consumer"] = r.get(
                "frac_negative_consumer", np.nan)
    if not capacity_limit.empty:
        inv["max_demand_multiplier"] = capacity_limit["max_multiplier"].iloc[0]

    return {
        "inventory": pd.DataFrame([inv]),
        "options": options,
        "sections": sections,
        "components": components,
        "elements": elements,
        "element_attrs": attrs_out,
        "naming": naming,
        "patterns": patterns,
        "capacity": capacity,
        "topology": topology,
        "groups": groups,
        "smoke": smoke,
        "capacity_limit": capacity_limit,
        "issues": pd.DataFrame(issues),
        "_wn": wn,
        "_junctions": junc,
        "_graph": simple,
    }


TABLE_NAMES = ("inventory", "options", "sections", "components", "elements",
               "element_attrs", "naming", "patterns", "capacity", "topology",
               "groups", "smoke", "capacity_limit", "issues")


def survey_corpus(paths, cfg: SurveyConfig | None = None,
                  verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Survey many files; a failure on one never stops the batch."""
    cfg = cfg or SurveyConfig()
    acc: dict[str, list[pd.DataFrame]] = {k: [] for k in TABLE_NAMES}
    for p in paths:
        t0 = time.time()
        try:
            out = survey_network(p, cfg)
        except Exception as exc:  # last-resort guard
            out = {"inventory": pd.DataFrame([{"network": Path(p).stem,
                                               "wntr_ok": False}]),
                   "issues": pd.DataFrame([{"network": Path(p).stem,
                                            "check": "survey_crash",
                                            "severity": "error",
                                            "message": f"{type(exc).__name__}: {exc}"}])}
        for k in TABLE_NAMES:
            df = out.get(k)
            if df is not None and len(df):
                acc[k].append(df)
        if verbose:
            n_err = len(out.get("issues", pd.DataFrame()).query(
                "severity == 'error'")) if len(out.get("issues", pd.DataFrame())) else 0
            print(f"  {Path(p).stem:12s} {time.time()-t0:5.1f}s  errors={n_err}")
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
            for k, v in acc.items()}


def write_tables(tables: dict[str, pd.DataFrame], outdir: str | Path,
                 prefix: str = "networks_") -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        if df is None or not len(df):
            continue
        p = outdir / f"{prefix}{name}.csv"
        df.to_csv(p, index=False)
        written.append(p)
    return written


# --------------------------------------------------------------------------
# readme claims and the diff
# --------------------------------------------------------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "no": 0,
}

# readme heading -> filename stem. Extend as the corpus grows.
README_ALIASES = {
    "balerma": "Balerma", "c-town": "C_Town", "ctown": "C_Town",
    "d-town": "D_Town", "dtown": "D_Town", "graeme": "Graeme",
    "graeme (zhi jiang)": "Graeme",
}


def _num(token: str):
    token = token.strip().lower().replace(",", "")
    if token in _WORD_NUM:
        return _WORD_NUM[token]
    try:
        return float(token)
    except ValueError:
        return np.nan


def parse_readme(path: str | Path) -> pd.DataFrame:
    """Extract structured claims from ``datasets_readme.txt``.

    The readme is a *claim*, not evidence: this table exists to be diffed
    against the measured census, not to be trusted.
    """
    text = Path(path).read_text(errors="replace")
    # entries start with a two-digit index and a name on its own line
    parts = re.split(r"\n(?=\d{2}\s+\S)", text)
    rows = []
    for part in parts[1:] if len(parts) > 1 else parts:
        head, _, body = part.partition("\n")
        m = re.match(r"(\d{2})\s+(.+?)\s*$", head)
        if not m:
            continue
        idx, name = m.group(1), m.group(2).strip()
        key = name.lower()
        stem = README_ALIASES.get(key)
        if stem is None:
            km = re.match(r"ky\s*(\d+)", key)
            stem = f"ky{km.group(1)}" if km else None
        blob = " ".join(body.split())
        row = {"readme_index": idx, "readme_name": name, "network": stem,
               "text": blob[:400]}
        if (m := re.search(r"total demand of ([\d.,]+)\s*(MGD|CMD|mgd|cmd)", blob)):
            val, unit = float(m.group(1).replace(",", "")), m.group(2).upper()
            row["claim_demand_value"] = val
            row["claim_demand_unit"] = unit
            row["claim_demand_cmd"] = val * 3785.41 if unit == "MGD" else val
        for field_, pat in [("reservoirs", r"([\w-]+)\s+reservoirs?\b"),
                            ("tanks", r"([\w-]+)\s+tanks?\b"),
                            ("pumps", r"([\w-]+)\s+pumps?\b")]:
            if (m := re.search(pat, blob)):
                row[f"claim_n_{field_}"] = _num(m.group(1))
        if (m := re.search(r"([\d.,]+)\s*(miles|kilometers|kilometres|km)\s+of pipe",
                           blob, re.I)):
            val = float(m.group(1).replace(",", ""))
            row["claim_pipe_km"] = val * 1.609344 if m.group(2).lower().startswith("mile") else val
        if (m := re.search(r"classified as ([\w\s-]+?) by Hwang", blob)):
            row["claim_class_hwang"] = m.group(1).strip()
        if (m := re.search(r"and ([\w-]+) by Hoagland", blob)):
            row["claim_class_hoagland"] = m.group(1).strip()
        rows.append(row)
    return pd.DataFrame(rows)


def compare_readme(inventory: pd.DataFrame, claims: pd.DataFrame,
                   rel_tol: float = 0.10) -> pd.DataFrame:
    """Long diff: one row per (network, field) with claim, measured, verdict."""
    pairs = [("claim_n_reservoirs", "n_reservoirs", "count"),
             ("claim_n_tanks", "n_tanks", "count"),
             ("claim_n_pumps", "n_pumps", "count"),
             ("claim_pipe_km", "pipe_length_km", "rel"),
             ("claim_demand_cmd", "demand_cmd", "rel")]
    inv = inventory.set_index("network")
    rows = []
    for _, c in claims.iterrows():
        net = c.get("network")
        if not net or net not in inv.index:
            rows.append({"network": net or c["readme_name"], "field": "-",
                         "claim": np.nan, "measured": np.nan,
                         "verdict": "no file surveyed"})
            continue
        m = inv.loc[net]
        for ck, mk, mode in pairs:
            claim, meas = c.get(ck, np.nan), m.get(mk, np.nan)
            if pd.isna(claim) and pd.isna(meas):
                continue
            if pd.isna(claim):
                verdict = "no claim"
            elif pd.isna(meas):
                verdict = "not measured"
            elif mode == "count":
                verdict = "match" if float(claim) == float(meas) else "MISMATCH"
            else:
                denom = max(abs(float(claim)), 1e-9)
                verdict = ("match" if abs(float(claim) - float(meas)) / denom <= rel_tol
                           else "MISMATCH")
            rows.append({"network": net, "field": mk,
                         "claim": claim, "measured": meas,
                         "rel_diff": (abs(float(claim) - float(meas)) /
                                      max(abs(float(claim)), 1e-9))
                         if not (pd.isna(claim) or pd.isna(meas)) else np.nan,
                         "verdict": verdict})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# plots (single network)
# --------------------------------------------------------------------------

def plot_network(out: dict, figsize=(13, 9)):
    """Four-panel deep dive for one surveyed network."""
    import matplotlib.pyplot as plt

    wn, junc = out["_wn"], out["_junctions"]
    net = out["inventory"]["network"].iloc[0]
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    # 1. layout, demand-bearing vs artificial
    a = ax[0, 0]
    for lname in wn.pipe_name_list:
        l = wn.get_link(lname)
        c1 = wn.get_node(l.start_node_name).coordinates
        c2 = wn.get_node(l.end_node_name).coordinates
        if c1 and c2:
            a.plot([c1[0], c2[0]], [c1[1], c2[1]], lw=0.4, color="0.75", zorder=1)
    if not junc.empty:
        pts = np.array([wn.get_node(n).coordinates or (np.nan, np.nan)
                        for n in junc["node"]], dtype=float)
        dem = junc["base_demand_lps"].to_numpy()
        art = junc["artificial_by_rule"].to_numpy()
        s = a.scatter(pts[~art, 0], pts[~art, 1], c=dem[~art], s=9,
                      cmap="viridis", zorder=3)
        a.scatter(pts[art, 0], pts[art, 1], marker="x", s=22, c="crimson",
                  zorder=4, label=f"artificial ({art.sum()})")
        plt.colorbar(s, ax=a, label="base demand (L/s)")
    for n in wn.reservoir_name_list:
        c = wn.get_node(n).coordinates
        if c:
            a.scatter(*c, marker="s", s=70, c="tab:blue", zorder=5)
    for n in wn.tank_name_list:
        c = wn.get_node(n).coordinates
        if c:
            a.scatter(*c, marker="^", s=70, c="tab:orange", zorder=5)
    a.set_title(f"{net}: layout (squares=reservoirs, triangles=tanks)")
    a.set_aspect("equal"); a.legend(loc="best", fontsize=7); a.axis("off")

    # 2. demand distribution
    a = ax[0, 1]
    d = junc.loc[junc["base_demand_lps"] > 0, "base_demand_lps"] if not junc.empty else pd.Series(dtype=float)
    if len(d):
        a.hist(d, bins=40, color="tab:green", alpha=0.8)
        a.set_yscale("log")
    a.set_title(f"base demand, demand-bearing junctions (n={len(d)})")
    a.set_xlabel("L/s"); a.set_ylabel("count (log)")

    # 3. patterns
    a = ax[1, 0]
    pats = out["patterns"]
    for _, r in pats.iterrows():
        m = np.asarray(wn.get_pattern(r["pattern"]).multipliers, float)
        a.plot(m, lw=1.2, label=f"{r['pattern']} ({r['used_by'].split(' | ')[0]})")
    a.set_title("patterns as shipped"); a.set_xlabel("step")
    if len(pats):
        a.legend(fontsize=7)

    # 4. smoke-test pressures
    a = ax[1, 1]
    sm = out["smoke"]
    if len(sm):
        txt = []
        for _, r in sm.iterrows():
            txt.append(
                f"{r['tier']}  duration={r['duration_h']:.0f} h  "
                f"converged={r['converged']}\n"
                f"  all junctions      pmin={r.get('pressure_min_m', float('nan')):8.1f} m"
                f"  neg={r.get('frac_negative', float('nan')):7.2%}\n"
                f"  consumers only     pmin="
                f"{r.get('pressure_min_m_consumer', float('nan')):8.1f} m"
                f"  neg={r.get('frac_negative_consumer', float('nan')):7.2%}"
                f"  in band={r.get('frac_in_band_consumer', float('nan')):7.2%}\n"
                f"  worst consumer: {r.get('worst_nodes_consumer', '-')}")
        a.text(0.0, 0.98, "\n\n".join(txt), va="top", family="monospace", fontsize=7.5)
    a.set_title("smoke test"); a.axis("off")

    fig.suptitle(f"{net} — network survey", fontsize=13)
    fig.tight_layout()
    return fig
