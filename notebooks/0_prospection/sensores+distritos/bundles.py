"""bundles.py -- a district partition, materialised as a network bundle.

Why a bundle and not an injected partition
------------------------------------------
`landuse_world.build` reads its partition from `data/01_raw/<network>/`, and
`network` is already part of `WorldCfg.identity()`. So the cheapest correct way
to sweep partitions is not to teach `build` a new argument -- it is to give
each partition its own network directory. Three things fall out of that:

* **no cache collision.** Two partitions produce two `network` strings, hence
  two `sim_hash` values, hence two world directories. Had the partition stayed
  outside the identity, all of them would have addressed the same world and
  the cache would have served the wrong one, silently.
* **content addressing for free.** The bundle name carries `stable_hash` of the
  partition itself, so editing a partition file renames its bundle and orphans
  the stale worlds instead of reusing them.
* **nothing upstream changes.** `landuse_sweep.cells` already sweeps networks.

The bundle is a DERIVED COPY of a real base bundle (`dtown`), not an invented
one: `profile.yml` is read from the base and only two blocks are rewritten.
That keeps every key the shipped `resolve_network_parameters` /
`configure_network` / `extract_sensor_series` nodes expect, without this module
having to know what they are.

What is rewritten, and why
--------------------------
`drift_seed_nodes`
    The base profile's seeds name nodes chosen for the base partition. Under a
    different cut those nodes land in the wrong district, which would seed a
    drift outside its target -- silently, since `build_drift_schedule` reads
    the profile only as a fallback. Recomputed here, per district.

`sensors`
    The shipped placement is keyed BY DISTRICT. The elements are still valid
    (same .inp) but their district assignment is not, so each gauge is re-homed
    under the new partition. This only affects the `shipped` annotation and the
    world's own client CSVs; the placement analysis reads raw `pressures` and
    `flows` and does not consult it.

The seed rule
-------------
GRAPH CENTRE of the district's induced subgraph, not the node with the largest
cross-district effect.

The first instinct -- seed where the drift will leak most -- does not survive
contact with the mechanism. `build_drift_schedule` diffuses a front from the
seed and `evolve_assignments` converts each node at its own `drift_month`, so
once every node of the district has a month inside the horizon the SETTLED
state is the whole district converted, whatever the seed was. The mixture
estimator measures init against settled. The terminal composition is therefore
seed-invariant and there is nothing to maximise.

What the seed does control is WHEN full conversion lands: a corner seed needs
as many months as its eccentricity, a central one needs the radius. Since
`mixture_probe.settled_window` raises when the post-drift tail is too short,
the binding constraint is conversion time, and minimising eccentricity is
exactly minimising it. A graph centre is interior by construction, so it also
avoids the boundary-seeding bias the first rule would have introduced.

The one case where the seed DOES change the terminal state is a district whose
induced subgraph is disconnected: the front cannot jump the gap, so coverage
comes out below 1. D-Town has this -- District_A is internally disconnected
when the walk traverses only consumer junctions, because N1-N11 are
articulation points. `seed_table` reports `coverage` per district for exactly
that reason; anything below 1.0 means the mixture is measuring a PARTIAL
conversion and the cell must be read with that in mind.
"""
from __future__ import annotations

import copy
import pathlib
import re
import shutil
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

import landuse_world as lw

__all__ = ["Bundle", "read_partition", "partition_id", "topology",
           "seed_table", "materialize", "prepare", "PIPELINE_METHODS",
           "REPORT_ONLY_METHODS"]

# The partitions the automated pipeline runs on. `external` is the ground-truth
# reference and stays selectable, but is not part of the usual grid.
PIPELINE_METHODS = ("spectral", "fast_greedy", "girvan_newman")

# Kept buildable and reportable, never shipped as a pipeline arm. `walktrap`
# puts 265 of 399 junctions and all seven tanks in one district, leaving three
# districts with no storage at all -- an edge case worth a row in the report
# and worth nothing as a client partition.
REPORT_ONLY_METHODS = ("walktrap",)

_NAME_RE = re.compile(r"^(?P<network>[^_]+)__(?P<method>.+?)__k(?P<k>\d+)__districts$")


# ==========================================================================
# partitions
# ==========================================================================
def read_partition(path) -> dict:
    """Read and VALIDATE one `input/*.yml` partition.

    The partition is the simulation domain: `validate_partition` upstream
    demands that it cover the junction set exactly. Failing here, on a file
    read, is a much better error than failing thirty seconds into a solve.
    """
    path = pathlib.Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if "districts" not in raw:
        raise ValueError(f"{path.name}: no `districts:` block")
    districts = {str(d): [str(n) for n in (ns or [])]
                 for d, ns in raw["districts"].items()}
    if not districts:
        raise ValueError(f"{path.name}: empty partition")

    seen, dup = set(), set()
    for ns in districts.values():
        for n in ns:
            (dup if n in seen else seen).add(n)
    if dup:
        raise ValueError(f"{path.name}: {len(dup)} node(s) in two districts, "
                         f"e.g. {sorted(dup)[:5]}")

    meta = _NAME_RE.match(path.stem)
    return {"districts": districts,
            "assets": {str(d): [str(n) for n in (ns or [])]
                       for d, ns in (raw.get("assets") or {}).items()},
            "path": path,
            "network": meta.group("network") if meta else "unknown",
            "method": meta.group("method") if meta else path.stem}


def partition_id(districts: dict) -> str:
    """8 hex chars, stable across processes, over the partition CONTENT.

    Uses `fedwater.hashing.stable_hash` -- crc32 of `repr`, not Python's
    salted `hash()`. The district ORDER is part of the content on purpose:
    `income_landuse_mapping` is positional, so a reordering is a different
    scenario and deserves a different address. Node order within a district is
    NOT, so it is sorted out of the key.
    """
    from fedwater.hashing import stable_hash
    canon = [(str(d), tuple(sorted(str(n) for n in ns)))
             for d, ns in districts.items()]
    return f"{stable_hash(canon):08x}"


# ==========================================================================
# topology, without wntr
# ==========================================================================
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


# ==========================================================================
# seeds
# ==========================================================================
def seed_table(districts: dict, topo: dict) -> pd.DataFrame:
    """One row per district: the seed node, and how far the front must travel.

    `coverage` is the share of the district reachable from the seed WITHOUT
    leaving the district. Below 1.0 the diffusion front cannot convert the
    whole district and the settled state genuinely depends on the seed -- see
    the module docstring.

    `eccentricity` is the number of hops from the seed to the furthest node it
    can reach, i.e. the lower bound on how many diffusion rounds full
    conversion needs. It is what sizes the horizon.
    """
    import networkx as nx
    G = _graph(topo)
    bd = topo["base_demand"]
    rows = []
    for d, nodes in districts.items():
        sub = G.subgraph([n for n in nodes if n in G])
        if not len(sub):
            rows.append({"district": d, "seed_node": None, "n_nodes": len(nodes),
                         "components": 0, "coverage": 0.0,
                         "eccentricity": None, "seed_base_demand": np.nan})
            continue
        comps = sorted(nx.connected_components(sub), key=len, reverse=True)
        big = sub.subgraph(comps[0])
        ecc = nx.eccentricity(big)
        lo = min(ecc.values())

        # Ties are common on a grid-like network, so break on base demand:
        # among equally central nodes, the one that actually consumes water
        # makes the first drift month a visible one.
        #
        # The widening matters more than the tie-break. D-Town's graph centres
        # are often trunk nodes with zero base demand (the N-series, and
        # junctions the .inp gives no consumer), and seeding the front on one
        # converts nothing in its first month. When no exact centre consumes,
        # accept one extra hop of eccentricity to get a node that does -- one
        # month of front travel bought against a first drift month that is
        # actually visible in the demand series.
        def best(radius):
            pool = [n for n, e in ecc.items() if e <= radius
                    and bd.get(n, 0.0) > 0]
            return sorted(pool, key=lambda n: (ecc[n], -bd.get(n, 0.0), n))[0] \
                if pool else None
        centre = best(lo) or best(lo + 1) or sorted(
            (n for n, e in ecc.items() if e == lo), key=str)[0]
        rows.append({"district": d, "seed_node": centre, "n_nodes": len(nodes),
                     "components": len(comps),
                     "coverage": round(len(comps[0]) / len(nodes), 4),
                     "eccentricity": int(lo),
                     "seed_base_demand": round(float(bd.get(centre, 0.0)), 4)})
    return pd.DataFrame(rows)


# ==========================================================================
# sensors, re-homed
# ==========================================================================
def _rehome_sensors(base_sensors: dict, districts: dict, topo: dict) -> dict:
    """The base bundle's gauges, reassigned to their district under `districts`.

    A flow link is homed to the district owning BOTH endpoints when it has
    one, else to whichever endpoint is a junction of some district. There is
    no flow series yet, so the delivers-into rule `signal_probe` uses cannot
    apply -- and does not need to: this block only annotates `shipped` and
    populates the world's own client CSVs, neither of which the placement
    analysis reads.
    """
    node2d = {n: d for d, ns in districts.items() for n in ns}
    out = {d: {"pressure": [], "flow": []} for d in districts}
    for kinds in (base_sensors or {}).values():
        if not isinstance(kinds, dict):
            continue
        for node in kinds.get("pressure") or []:
            d = node2d.get(str(node))
            if d:
                out[d]["pressure"].append(str(node))
        for link in kinds.get("flow") or []:
            u, v, _ = topo["links"].get(str(link), (None, None, None))
            du, dv = node2d.get(u), node2d.get(v)
            d = du if du == dv else (du or dv)
            if d:
                out[d]["flow"].append(str(link))

    # A district with no gauge of a kind gives a client CSV with no columns of
    # that kind, which `preprocess_clients` reads as a different feature
    # geometry per client. Backfill from the district's own elements so the
    # world is well formed; the real placement is chosen downstream anyway.
    bd, node2links = topo["base_demand"], {}
    for name, (u, v, _) in topo["links"].items():
        node2links.setdefault(u, []).append(name)
        node2links.setdefault(v, []).append(name)
    for d, nodes in districts.items():
        if not out[d]["pressure"]:
            pick = sorted(nodes, key=lambda n: (-bd.get(n, 0.0), n))[:1]
            out[d]["pressure"] = pick
        if not out[d]["flow"]:
            inside = {l for n in nodes for l in node2links.get(n, [])}
            out[d]["flow"] = sorted(inside)[:1]
    return out


# ==========================================================================
# the bundle
# ==========================================================================
@dataclass
class Bundle:
    """One materialised partition. `network` is what `WorldCfg` gets."""
    config_id: str
    method: str
    network: str
    path: pathlib.Path
    districts: dict
    assets: dict
    seeds: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: pathlib.Path | None = None

    @property
    def names(self) -> list[str]:
        return list(self.districts)

    @property
    def all_residential(self) -> str:
        return "_".join(["LR"] * len(self.districts))

    def summary(self) -> pd.Series:
        s = self.seeds
        return pd.Series({
            "config_id": self.config_id, "method": self.method,
            "network": self.network, "k": len(self.districts),
            "sizes": "/".join(str(len(v)) for v in self.districts.values()),
            "min_coverage": round(float(s["coverage"].min()), 3) if len(s) else np.nan,
            "max_eccentricity": (int(s["eccentricity"].max())
                                 if len(s) and s["eccentricity"].notna().any()
                                 else None),
            "districts_without_storage": sum(
                1 for d in self.districts if not self.assets.get(d)),
        })


def materialize(root, partition_path, base_network: str = "dtown",
                anchor_scale: float | None = None, overwrite: bool = True,
                verbose: bool = True) -> Bundle:
    """Write `data/01_raw/<bundle>/` for one partition and return its handle."""
    root = pathlib.Path(root)
    part = read_partition(partition_path)
    districts, assets = part["districts"], part["assets"]

    base = lw.load_bundle(root, base_network)
    topo = topology(base["inp"])

    missing = set(topo["junctions"]) - {n for ns in districts.values() for n in ns}
    extra = {n for ns in districts.values() for n in ns} - set(topo["junctions"])
    if missing or extra:
        raise ValueError(
            f"{pathlib.Path(partition_path).name}: partition does not cover the "
            f"junction set exactly ({len(missing)} missing, {len(extra)} unknown)")

    pid = partition_id(districts)
    network = f"{base_network}__{part['method']}__{pid}"
    out = root / "data" / "01_raw" / network
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists (overwrite=False)")
    out.mkdir(parents=True, exist_ok=True)

    # Every file this function writes has a name it already knows, so it
    # overwrites them in place rather than `rmtree`ing the directory first.
    # A materialised bundle is nothing else's to delete: `out` is named from
    # a content hash of the partition, so nothing but this function ever
    # targets it, and an rmtree-then-recreate loop over dozens of partitions
    # is exactly the delete/write pattern behavioural antivirus (Norton's
    # IDP.Generic among others) flags as ransomware-shaped. It is also not
    # crash-safe: a process killed between the rmtree and the writes leaves
    # an empty directory where a bundle used to be.
    shutil.copy(base["inp"], out / "network.inp")
    (out / "districts.yml").write_text(yaml.safe_dump(
        {"districts": districts, "assets": assets}, sort_keys=False))

    seeds = seed_table(districts, topo)
    profile = copy.deepcopy(base["profile"])
    seed_map = {r["district"]: r["seed_node"] for _, r in seeds.iterrows()
                if r["seed_node"]}
    # Mirror the base profile's container type where it has one: this module
    # cannot see `resolve_seed_node`, so it copies the shape it was given
    # rather than asserting one. The POC's first cell checks that the drift
    # actually landed in its target district.
    if isinstance(profile.get("drift_seed_nodes"), list):
        profile["drift_seed_nodes"] = [seed_map[d] for d in districts
                                       if d in seed_map]
    else:
        profile["drift_seed_nodes"] = seed_map
    profile["sensors"] = _rehome_sensors(profile.get("sensors"), districts, topo)
    if anchor_scale is not None:
        profile.setdefault("hydraulics", {})
        if isinstance(profile.get("hydraulics"), dict):
            profile["hydraulics"]["anchor_scale"] = float(anchor_scale)
    (out / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False))

    # `WorldCfg.resolved_anchor` raises for a network absent from this dict.
    # The sweep passes `anchor_scale` explicitly so it never gets there, but a
    # bare `lw.WorldCfg(network=<bundle>)` in a notebook would -- register it.
    lw.ANCHOR_SCALE[network] = float(
        anchor_scale if anchor_scale is not None
        else lw.ANCHOR_SCALE.get(base_network, 0.25))

    b = Bundle(config_id=f"{part['method']}__{pid}", method=part["method"],
               network=network, path=out, districts=districts, assets=assets,
               seeds=seeds, source=pathlib.Path(partition_path))
    if verbose:
        print(f"  {b.config_id:28s} -> {network}  "
              f"k={len(districts)} sizes={'/'.join(str(len(v)) for v in districts.values())}"
              f"  min coverage {seeds['coverage'].min():.2f}"
              f"  max ecc {seeds['eccentricity'].max()}")
    return b


def prepare(root, input_dir, methods=PIPELINE_METHODS, base_network="dtown",
            anchor_scale: float = 0.25, verbose: bool = True) -> list[Bundle]:
    """Materialise every partition in `input_dir` whose method is wanted.

    `methods=None` takes every file found -- use it for the report run that
    includes the edge cases, not for the pipeline grid.
    """
    input_dir = pathlib.Path(input_dir)
    files = sorted(input_dir.glob("*.yml")) + sorted(input_dir.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no partition files in {input_dir}")
    out = []
    if verbose:
        print(f"materialising bundles from {input_dir}")
    for f in files:
        part = read_partition(f)
        if part["network"] != base_network:   continue
        if methods is not None and part["method"] not in methods:   continue

        out.append(materialize(root, f, base_network=base_network,
                               anchor_scale=anchor_scale, verbose=verbose))
    if not out:
        raise ValueError(f"no partition in {input_dir} matched methods={methods}")
    return out
