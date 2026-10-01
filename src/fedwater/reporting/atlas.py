"""Navigate every simulated world of every study, without typing paths.

The experiments root already stores things along the lines of what they
SHARE, and the atlas navigates the same way::

    study      -> expands into worlds             (<study>/runs/*/manifest.json)
    partition  -> shared by every world cut by it (network + districting method)
    stack      -> shared by every world at one operating point
                  (probes/<network>/<method>/<hash>/, named by stacks.yml)
    world      -> what differs: drift target, seed (worlds/<sim_hash>/)

:class:`Atlas` scans the root ONCE into a tidy index -- one row per
(study, world) -- read from plain files (manifests, ``runs`` manifests, each
clone's ``stacks.yml`` pointer); no Kedro session is opened. The index is
cached in ``<root>/_atlas/index.parquet`` and rebuilt only when a manifest
it was built from changed (mtime/size signature).

Everything heavier is lazy and cached in memory, keyed by what it is shared
by: a world's tables by ``sim_hash``, a probe stack by its path (so the K
worlds of an every-district group load their shared stack once), a
partition's rebuilt districting results by network. :class:`Selector` is the
one navigation widget: study -> partition -> world, and every section shown
through it re-draws when the selection changes.
"""
from __future__ import annotations

import copy
import json
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

STACKS_REL = Path("data/08_reporting/sensor_placement/stacks.yml")
NOT_STUDIES = {"worlds", "probes", "partitions", "_atlas"}
INDEX_DIR = "_atlas"
NO_STUDY = "(no study)"


# ==========================================================================
# a world clone's catalog, without a Kedro session
# ==========================================================================
def world_catalog(clone):
    """``(DataCatalog, parameters, globals)`` of a project or world clone.

    Built from its own ``conf/`` (base + local, the engine's overrides) with
    every relative file path anchored at the clone, so it reads the clone's
    artifacts from any working directory. Needs no ``pyproject.toml`` and
    opens no session: a cached world is plain files.
    """
    from kedro.config import OmegaConfigLoader
    from kedro.io import DataCatalog

    clone = Path(clone).resolve()
    loader = OmegaConfigLoader(conf_source=str(clone / "conf"),
                               base_env="base", default_run_env="local")
    conf = copy.deepcopy(loader["catalog"])
    for entry in conf.values():
        for key in ("filepath", "path"):
            v = entry.get(key)
            if isinstance(v, str) and "://" not in v and not Path(v).is_absolute():
                entry[key] = str(clone / v)
    return DataCatalog.from_config(conf), loader["parameters"], loader["globals"]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def project_root(start) -> Path:
    """The nearest directory at or above ``start`` holding ``conf/base``."""
    start = Path(start).resolve()
    for p in (start, *start.parents):
        if (p / "conf" / "base").exists():
            return p
    raise FileNotFoundError(f"no conf/base at or above {start}")


# ==========================================================================
# the index
# ==========================================================================
def _sources(root: Path) -> list[Path]:
    files = sorted(root.glob("worlds/*/manifest.json"))
    files += sorted(root.glob(f"worlds/*/clone/{STACKS_REL.as_posix()}"))
    for study in sorted(p for p in root.iterdir()
                        if p.is_dir() and p.name not in NOT_STUDIES):
        files += sorted(study.glob("runs/*/manifest.json"))
    return files


def _signature(root: Path, files: list[Path]) -> str:
    h = sha256()
    for f in files:
        st = f.stat()
        h.update(f"{f.relative_to(root).as_posix()}|{st.st_mtime_ns}|"
                 f"{st.st_size}\n".encode())
    return h.hexdigest()


def _district_names(clone: Path, network, method, cache: dict) -> list:
    """District order of the world's partition (the map's token order)."""
    key = (network, method, str(clone))
    if key not in cache:
        doc = _read_yaml(clone / "data" / "01_raw" / str(network) / "partitions"
                         / str(method) / "districts.yml") if clone else {}
        cache[key] = list((doc.get("districts") or {}))
    return cache[key]


def _maps(world: dict, names: list) -> dict:
    """Initial map, final map (the drift district's token replaced by its
    target) and the drift change in words -- what the world IS, readable."""
    from fedwater.experiments.spec import _INCOME_INV, _LAND_USE_INV

    init = world.get("consumption_map")
    d = world.get("drift_district")
    to = (world.get("drift_to_income"), world.get("drift_to_land_use"))
    out = {"initial_map": init, "final_map": None, "drift_change": None,
           "n_districts": len(names) or None}
    if not init or d not in names:
        return out
    tokens = str(init).split("_")
    i = names.index(d)
    if i >= len(tokens):
        return out
    was = tokens[i]
    now = (_INCOME_INV.get(to[0], was[0]) if to[0] else was[0]) + \
        (_LAND_USE_INV.get(to[1], was[1]) if to[1] else was[1])
    tokens[i] = now
    from fedwater.experiments.spec import _INCOME, _LAND_USE
    words = lambda t: f"{_INCOME.get(t[0], t[0])} {_LAND_USE.get(t[1], t[1])}"
    out.update(final_map="_".join(tokens),
               drift_change=f"{words(was)} -> {words(now)}")
    return out


def build_index(root) -> pd.DataFrame:
    """One row per (study, world). A world no study's runs reference is kept
    under ``(no study)``; a world two studies share appears once in each."""
    root = Path(root).resolve()
    membership: dict[str, set] = {}
    for study in sorted(p for p in root.iterdir()
                        if p.is_dir() and p.name not in NOT_STUDIES):
        for m in study.glob("runs/*/manifest.json"):
            h = _read_json(m).get("sim_hash")
            if h:
                membership.setdefault(h, set()).add(study.name)

    rows, names = [], {}
    for mf in sorted(root.glob("worlds/*/manifest.json")):
        m = _read_json(mf)
        h = m.get("sim_hash") or mf.parent.name
        clone = mf.parent / "clone"
        world = m.get("world") or {}
        order = _district_names(clone, world.get("network"),
                                world.get("districting"), names)
        ptr = _read_yaml(clone / STACKS_REL) if (clone / STACKS_REL).exists() else {}
        sel, lab = ptr.get("selection") or {}, ptr.get("label") or {}
        base = {"sim_hash": h, "status": m.get("status"),
                **world, **_maps(world, order),
                "created_utc": m.get("created_utc"),
                "minutes": round(float(m.get("seconds") or 0) / 60, 1),
                "src_hash": m.get("src_hash"),
                "clone": str(clone) if clone.exists() else None,
                "selection_stack": sel.get("stack_hash"),
                "label_stack": lab.get("stack_hash"),
                "same_stack": ptr.get("same_stack"),
                "selection_path": sel.get("path"),
                "label_path": lab.get("path")}
        for study in sorted(membership.get(h, {NO_STUDY})):
            rows.append({"study": study, **base})
    cols = ["study", "sim_hash", "status", "network", "districting",
            "partition_id", "drift_district", "drift_change", "initial_map",
            "final_map", "sim_seed", "n_months"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    front = [c for c in cols if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    return df.sort_values(["study", "network", "districting",
                           "drift_district", "sim_seed"],
                          na_position="last").reset_index(drop=True)


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns holding MIXED python types (an int in one world, None
    or a string in another) become strings; None stays missing."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != object:
            continue
        kinds = {type(v) for v in out[c] if v is not None and v == v}
        if len(kinds) > 1 or kinds & {dict, list, tuple}:
            out[c] = out[c].map(lambda v: None if v is None or v != v else str(v))
    return out


# ==========================================================================
# lazy views
# ==========================================================================
class WorldView:
    """One world: its manifest, clone catalog, tables and stacks, all lazy."""

    def __init__(self, atlas: "Atlas", row: pd.Series):
        self.atlas, self.row = atlas, row
        self.sim_hash = row["sim_hash"]
        self.dir = atlas.root / "worlds" / self.sim_hash
        self.clone = Path(row["clone"]) if row.get("clone") else None
        self._tables: dict = {}
        self._catalog = None

    def __repr__(self):
        r = self.row
        return (f"<world {self.sim_hash} {r.get('network')}/"
                f"{r.get('districting')} drift={r.get('drift_district')} "
                f"seed={r.get('sim_seed')} {r.get('status')}>")

    # -- plain files -------------------------------------------------------
    @property
    def manifest(self) -> dict:
        return self._memo("manifest", lambda: _read_json(self.dir / "manifest.json"))

    @property
    def effective(self) -> dict:
        return self.manifest.get("effective") or {}

    @property
    def pointer(self) -> dict:
        return self._memo("pointer", lambda: _read_yaml(self.clone / STACKS_REL)
                          if self.clone else {})

    # -- the clone's catalog -----------------------------------------------
    @property
    def catalog(self):
        if self._catalog is None:
            if self.clone is None:
                raise FileNotFoundError(f"world {self.sim_hash} has no clone")
            self._catalog = world_catalog(self.clone)
        return self._catalog[0]

    @property
    def params(self) -> dict:
        self.catalog
        return self._catalog[1]

    def load(self, name: str):
        """A catalog dataset of this world, read once."""
        return self._memo(("ds", name), lambda: self.catalog.load(name))

    # -- shared objects ----------------------------------------------------
    def selection(self, with_probes: bool = False):
        path = self.pointer.get("selection", {}).get("path")
        return self.atlas.stack(path, with_probes) if path else None

    def label(self, with_probes: bool = False):
        if self.pointer.get("same_stack"):
            return self.selection(with_probes)
        path = self.pointer.get("label", {}).get("path")
        return self.atlas.stack(path, with_probes) if path else None

    @property
    def topology(self) -> dict:
        return self.atlas.topology(self.row["network"], self.clone)

    @property
    def network_model(self):
        return self.atlas.network_model(self.row["network"], self.clone)

    def _memo(self, key, fn):
        if key not in self._tables:
            self._tables[key] = fn()
        return self._tables[key]


class PartitionView:
    """Every stored partition of one network, rebuilt in memory once.

    The pipeline's own ``build_partitions`` and ``partition_reports`` run on
    the methods present on disk, and each rebuilt partition is checked
    against the stored one: the report never draws a partition that is not
    the one the worlds were built on.
    """

    def __init__(self, project: Path, network: str, methods=None):
        from fedwater.districting.yaml_io import load_network
        from fedwater.networks import partitions as pstore

        self.project, self.network = Path(project), network
        self.bundle = pstore.bundle_dir(self.project, network)
        _, params, _ = world_catalog(self.project)
        self.params = params["districting"]
        self.wn = load_network(self.bundle / "network.inp")
        self.profile = _read_yaml(self.bundle / "profile.yml")
        self.manual = _read_yaml(self.bundle / "districts.yml")
        on_disk = [m for m in pstore.METHODS
                   if (pstore.partition_dir(self.project, network, m)
                       / pstore.DISTRICTS_FILE).exists()]
        self.methods = [m for m in (methods or on_disk) if m in on_disk]
        self._results = self._reports = None

    @property
    def store(self) -> pd.DataFrame:
        from fedwater.networks import partitions as pstore
        rows = []
        for m in pstore.METHODS:
            man = pstore.read_manifest(self.project, self.network, m) or {}
            rows.append({"method": m,
                         "status": pstore.status(self.project, self.network, m,
                                                 self.params),
                         "partition_id": man.get("partition_id"),
                         "k": man.get("k"),
                         "sizes": "/".join(str(v) for v in
                                           (man.get("sizes") or {}).values()),
                         "boundary_links": man.get("n_boundary_links"),
                         "contiguous": man.get("all_contiguous"),
                         "built": man.get("created_utc")})
        return pd.DataFrame(rows)

    @property
    def results(self) -> dict:
        if self._results is None:
            from fedwater.networks import partitions as pstore
            from fedwater.pipelines.districting.nodes import build_partitions
            res = build_partitions(self.wn, self.manual, self.profile,
                                   {**self.params, "methods": self.methods})
            for m, r in res.items():
                doc = self.manual if m == pstore.MANUAL else r.districts_mapping(self.wn)
                stored = (pstore.read_manifest(self.project, self.network, m)
                          or {}).get("partition_id")
                if pstore.partition_id(doc) != stored:
                    raise ValueError(
                        f"{self.network}/{m}: rebuilt partition "
                        f"{pstore.partition_id(doc)} != stored {stored} -- the "
                        "stored partition is stale for the current parameters")
            self._results = res
        return self._results

    @property
    def reports(self) -> dict:
        """``summary``, ``sanity``, ``order`` -- the pipeline's own tables."""
        if self._reports is None:
            from fedwater.pipelines.districting.nodes import partition_reports
            _, summary, sanity, order = partition_reports(
                self.results, self.wn, self.manual, self.profile, self.params)
            self._reports = {"summary": summary, "sanity": sanity,
                             "order": order}
        return self._reports


# ==========================================================================
# the atlas
# ==========================================================================
class Atlas:
    """The index plus the shared caches. ``Atlas()`` finds the project from
    the working directory and reads ``<project>/data/09_experiments``."""

    def __init__(self, root=None, project=None, max_stacks: int = 8):
        self.project = project_root(project or Path.cwd())
        self.root = Path(root).resolve() if root else \
            self.project / "data" / "09_experiments"
        self.max_stacks = int(max_stacks)
        self._stacks: OrderedDict = OrderedDict()
        self._worlds: dict[str, WorldView] = {}
        self._partitions: dict = {}
        self._topo: dict = {}
        self._wn: dict = {}
        self.index = self._load_index()

    # -- index ---------------------------------------------------------------
    def _load_index(self, force: bool = False) -> pd.DataFrame:
        if not self.root.exists():
            return pd.DataFrame(columns=["study", "sim_hash", "status"])
        files = _sources(self.root)
        sig = _signature(self.root, files)
        cache = self.root / INDEX_DIR
        idx, stamp = cache / "index.parquet", cache / "signature.txt"
        if (not force and idx.exists() and stamp.exists()
                and stamp.read_text() == sig):
            return pd.read_parquet(idx)
        df = build_index(self.root)
        cache.mkdir(exist_ok=True)
        _parquet_safe(df).to_parquet(idx)
        stamp.write_text(sig)
        return pd.read_parquet(idx)

    def refresh(self) -> pd.DataFrame:
        """Re-read the index if anything on disk changed (cheap otherwise)."""
        self.index = self._load_index()
        return self.index

    def studies(self) -> list[str]:
        return sorted(self.index["study"].dropna().unique())

    def worlds(self, study: str | None = None, network: str | None = None,
               districting: str | None = None,
               ok_only: bool = False) -> pd.DataFrame:
        df = self.index
        for col, v in (("study", study), ("network", network),
                       ("districting", districting)):
            if v is not None and col in df:
                df = df[df[col] == v]
        if ok_only:
            df = df[df["status"] == "ok"]
        return df.reset_index(drop=True)

    def partitions(self, study: str | None = None) -> list[tuple[str, str]]:
        df = self.worlds(study)
        if df.empty or "network" not in df:
            return []
        pairs = df[["network", "districting"]].dropna().drop_duplicates()
        return [tuple(p) for p in pairs.itertuples(index=False)]

    def status(self, study: str | None = None) -> pd.DataFrame:
        """World counts per partition and status."""
        df = self.worlds(study)
        if df.empty:
            return df
        return (df.groupby(["study", "network", "districting", "status"],
                           dropna=False).size().rename("worlds")
                .reset_index())

    # -- lazy objects ----------------------------------------------------------
    def world(self, sim_hash: str) -> WorldView:
        if sim_hash not in self._worlds:
            hit = self.index[self.index["sim_hash"] == sim_hash]
            if hit.empty:
                raise KeyError(f"world {sim_hash} is not in the index "
                               f"({self.root}); refresh()?")
            self._worlds[sim_hash] = WorldView(self, hit.iloc[0])
        return self._worlds[sim_hash]

    def stack(self, path, with_probes: bool = False):
        """A probe stack, loaded once and shared by every world pointing at
        it. A stack loaded with its probes also serves requests without."""
        from fedwater.placement import store
        key = str(Path(path).resolve())
        for k in ((key, True),) if with_probes else ((key, True), (key, False)):
            if k in self._stacks:
                self._stacks.move_to_end(k)
                return self._stacks[k]
        st = store.load_stack(key, with_probes=with_probes)
        self._stacks[(key, with_probes)] = st
        while len(self._stacks) > self.max_stacks:
            self._stacks.popitem(last=False)
        return st

    def study_stacks(self, study: str, with_probes: bool = False) -> dict:
        """``{"<method>/selection:<hash>": stack, ...}`` for every distinct
        stack the study's ok worlds use -- the input ``sweep_browser`` takes."""
        out = {}
        for _, r in self.worlds(study, ok_only=True).iterrows():
            for role in ("selection", "label"):
                h, p = r.get(f"{role}_stack"), r.get(f"{role}_path")
                if not p or (role == "label" and bool(r.get("same_stack"))):
                    continue
                out.setdefault(f"{r['districting']}/{role}:{h}",
                               self.stack(p, with_probes))
        return out

    def partition(self, network: str) -> PartitionView:
        if network not in self._partitions:
            self._partitions[network] = PartitionView(self.project, network)
        return self._partitions[network]

    def topology(self, network: str, clone: Path | None = None) -> dict:
        if network not in self._topo:
            from fedwater.placement.topology import topology
            base = clone if clone is not None else self.project
            self._topo[network] = topology(base / "data" / "01_raw" / network
                                           / "network.inp")
        return self._topo[network]

    def network_model(self, network: str, clone: Path | None = None):
        if network not in self._wn:
            from fedwater.districting.yaml_io import load_network
            base = clone if clone is not None else self.project
            self._wn[network] = load_network(base / "data" / "01_raw" / network
                                             / "network.inp")
        return self._wn[network]

    def warm(self, study: str, with_probes: bool = True) -> int:
        """Pre-load every stack of a study (the slow part), once."""
        return len(self.study_stacks(study, with_probes=with_probes))

    def selector(self, **kw) -> "Selector":
        return Selector(self, **kw)


# ==========================================================================
# the navigation widget
# ==========================================================================
def _val(r, key, default="?"):
    v = r.get(key) if hasattr(r, "get") else None
    return default if v is None or (isinstance(v, float) and v != v) else v


def _world_label(r) -> str:
    """What the world IS, then how it was built, the hash last."""
    d = str(_val(r, "drift_district", "no drift")).replace("District_", "")
    last = _val(r, "drift_last_switch", None)
    horizon = f"{_val(r, 'n_months')} mo" + (f" (drift done m{int(float(last))})"
                                             if last not in (None, "None") else "")
    stack = str(_val(r, "selection_stack", ""))[:6]
    return (f"drift {d}: {_val(r, 'drift_change')} | "
            f"{_val(r, 'initial_map')} -> {_val(r, 'final_map')} | "
            f"seed {_val(r, 'sim_seed')} | {horizon} | "
            f"{_val(r, 'variant', 'baseline')} | stack {stack} | "
            f"{_val(r, 'status')} {r['sim_hash'][:6]}")


def _partition_label(atlas, study, network, method) -> str:
    ws = atlas.worlds(study, network, method)
    ok = int((ws["status"] == "ok").sum())
    pid = _val(ws.iloc[0], "partition_id", "?") if len(ws) else "?"
    k = _val(ws.iloc[0], "n_districts", "?") if len(ws) else "?"
    maps = sorted({str(m) for m in ws.get("initial_map", pd.Series(dtype=str)).dropna()})
    stacks = ws["selection_stack"].dropna().nunique() if "selection_stack" in ws else 0
    k = int(float(k)) if k not in ("?", None) else k
    return (f"{network} / {method} | k={k} | partition {pid} | "
            f"map {', '.join(maps) or '?'} | {ok}/{len(ws)} worlds ok | "
            f"{stacks} stack(s)")


def _study_label(atlas, study) -> str:
    ws = atlas.worlds(study)
    ok = int((ws["status"] == "ok").sum())
    nets = sorted(ws["network"].dropna().unique()) if "network" in ws else []
    meths = sorted(ws["districting"].dropna().unique()) if "districting" in ws else []
    return (f"{study} | {', '.join(map(str, nets))} | "
            f"{', '.join(map(str, meths))} | {ok}/{len(ws)} worlds ok")


class Selector:
    """study -> partition -> world. ``show(fn)`` renders ``fn(world)`` in its
    own output area and re-renders it on every world change;
    ``show(fn, level="study")`` renders ``fn(atlas, study)`` only when the
    STUDY changes (cross-world sections)."""

    def __init__(self, atlas: Atlas, study: str | None = None,
                 ok_only: bool = True):
        import ipywidgets as W

        self.atlas, self.ok_only = atlas, ok_only
        self._views: list = []
        studies = atlas.studies()
        if not studies:
            raise ValueError(f"no worlds under {atlas.root}")
        self.w_study = W.Dropdown(options=self._study_options(studies),
                                  description="study",
                                  value=study if study in studies else studies[-1],
                                  layout=W.Layout(width="900px"))
        self.w_part = W.Dropdown(description="partition",
                                 layout=W.Layout(width="900px"))
        self.w_world = W.Dropdown(description="world",
                                  layout=W.Layout(width="900px"))
        self.w_refresh = W.Button(description="refresh index",
                                  layout=W.Layout(width="140px"))
        self.w_study.observe(self._on_study, names="value")
        self.w_part.observe(self._on_part, names="value")
        self.w_world.observe(self._on_world, names="value")
        self.w_refresh.on_click(self._on_refresh)
        self.w_card = W.HTML()
        self.widget = W.VBox([W.HBox([self.w_study, self.w_refresh]),
                              self.w_part, self.w_world, self.w_card])
        self._fill_parts()
        self._card()

    # -- state ---------------------------------------------------------------
    @property
    def study(self) -> str:
        return self.w_study.value

    @property
    def world(self) -> WorldView | None:
        return self.atlas.world(self.w_world.value) if self.w_world.value else None

    def select(self, study=None, partition=None, sim_hash=None):
        """Programmatic navigation (also what the tests drive)."""
        if study is not None:
            self.w_study.value = study
        if partition is not None:
            self.w_part.value = tuple(partition)
        if sim_hash is not None:
            self.w_world.value = sim_hash
        return self.world

    # -- cascade ---------------------------------------------------------------
    def _study_options(self, studies):
        return [(_study_label(self.atlas, s), s) for s in studies]

    def _fill_parts(self):
        parts = self.atlas.partitions(self.study)
        self.w_part.options = [(_partition_label(self.atlas, self.study, n, m),
                                (n, m)) for n, m in parts]
        self.w_part.value = parts[0] if parts else None
        if not parts:
            self._fill_worlds()

    def _fill_worlds(self):
        part = self.w_part.value
        if part is None:
            self.w_world.options = []
            return
        df = self.atlas.worlds(self.study, *part, ok_only=self.ok_only)
        self.w_world.options = [(_world_label(r), r["sim_hash"])
                                for _, r in df.iterrows()]
        self.w_world.value = df["sim_hash"].iloc[0] if len(df) else None

    def _on_study(self, _):
        self._fill_parts()
        self._render("study")

    def _on_part(self, _):
        self._fill_worlds()

    def _on_world(self, _):
        self._card()
        self._render("world")

    def _card(self):
        """The selected world, spelled out, under the dropdowns."""
        w = self.world
        if w is None:
            self.w_card.value = "<i>no ok world for this selection</i>"
            return
        r = w.row
        cells = [("world", w.sim_hash), ("status", _val(r, "status")),
                 ("drift", f"{_val(r, 'drift_district')}: {_val(r, 'drift_change')}"),
                 ("seed node", _val(r, "drift_seed_node")),
                 ("map", f"{_val(r, 'initial_map')} &rarr; {_val(r, 'final_map')}"),
                 ("sim seed", _val(r, "sim_seed")),
                 ("horizon", f"{_val(r, 'n_months')} months x "
                             f"{_val(r, 'days_per_month')} d"
                             + (" (auto)" if str(_val(r, "n_months_auto", "")) == "True" else "")),
                 ("coupling", _val(r, "variant") + (
                     f" (close {_val(r, 'close_fraction')})"
                     if _val(r, "variant") == "partial" else "")),
                 ("beta / anchor", f"{_val(r, 'beta')} / {_val(r, 'anchor_scale')}"),
                 ("partition", f"{_val(r, 'network')}/{_val(r, 'districting')} "
                               f"({_val(r, 'partition_id')})"),
                 ("sensors", f"{_val(r, 'placement_source')}: "
                             f"{_val(r, 'n_flow_sensors')} flow, "
                             f"{_val(r, 'n_pressure_sensors')} pressure"),
                 ("stacks", f"selection {_val(r, 'selection_stack')}"
                            + ("" if str(_val(r, "same_stack", "")) == "True"
                               else f", label {_val(r, 'label_stack')}")),
                 ("studies", ", ".join(self.atlas.index.loc[
                     self.atlas.index["sim_hash"] == w.sim_hash, "study"]))]
        self.w_card.value = ("<table style='font-size:12px'>" + "".join(
            f"<tr><td style='padding-right:12px'><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in cells) + "</table>")

    def _on_refresh(self, _):
        keep = (self.study, self.w_part.value, self.w_world.value)
        self.atlas.refresh()
        self.w_study.options = self._study_options(self.atlas.studies())
        if keep[0] in self.atlas.studies():
            self.w_study.value = keep[0]
        self._fill_parts()
        if keep[1] in [v for _, v in self.w_part.options]:
            self.w_part.value = keep[1]
        if keep[2] in [v for _, v in self.w_world.options]:
            self.w_world.value = keep[2]
        self._render("study")
        self._render("world")

    # -- sections ---------------------------------------------------------------
    def show(self, fn, level: str = "world"):
        """Display ``fn``'s output here and keep it bound to the selection."""
        import ipywidgets as W
        from IPython.display import display

        if level not in ("world", "study"):
            raise ValueError("level must be 'world' or 'study'")
        out = W.Output()
        self._views.append((level, fn, out))
        display(out)
        self._draw(level, fn, out)
        return out

    def _render(self, level: str):
        for lv, fn, out in self._views:
            if lv == level:
                self._draw(lv, fn, out)

    def _draw(self, level, fn, out):
        import matplotlib.pyplot as plt
        from IPython.display import display

        out.clear_output(wait=True)
        with out:
            try:
                if level == "study":
                    fn(self.atlas, self.study)
                elif self.world is None:
                    print("no ok world for this selection")
                else:
                    fn(self.world)
                plt.show()
            except Exception as e:  # a section must not kill the page
                display(pd.DataFrame({"section failed": [f"{type(e).__name__}: {e}"]}))

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)
