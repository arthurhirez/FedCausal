"""Experiment engine: cached worlds x cheap runs, with a papertrail.

Layout under the experiments root (default ``data/09_experiments``)::

    worlds/<sim_hash>/                 # cached simulated worlds
        manifest.json                  # resolved sim config, status, versions
        clone/                         # project clone whose data/ holds the
                                       # simulation + oracle artifacts
    <study>/
        runs/<sim_hash>__<run_hash>/
            manifest.json              # full papertrail for this run
            harvest/<group>.parquet    # extracted metrics (always kept)
            retained/...               # optional heavy artifacts (opt-in)
        runs.parquet                   # flat index: specs + status + headlines
        harvest/<group>.parquet        # per-group tables across all runs

Mechanics preserved from the label factory (they encode hard-won lessons):
full-block ``conf/local`` overrides (Kedro merges destructively at the top
level), fresh-subprocess ``kedro run`` per stage (sidesteps the sys.path
bootstrap gotcha), and cache-or-run — now keyed by **content hash** of the
effective configuration instead of an ordinal, so editing a spec can never
silently reuse a stale world.

The selected network and partition travel separately from the parameter
override: they are written to ``conf/local/globals.yml``, which is what the
catalog interpolates into the bundle and partition filepaths. Both are part of
the world hash (the partition through its content id), so a world is never
reused across networks or cuts.

Two shared stores sit beside the worlds::

    partitions/<network>__<method>/    scratch clone + manifest of the last
                                       `--pipeline districting` build; the
                                       partition itself is copied into the
                                       PROJECT's data/01_raw/<network>/
                                       partitions/<method>/
    probes/<network>/<method>/<hash>/  sensor-placement probe stacks, shared by
                                       every world clone (globals.probe_store
                                       is written as this absolute path)

Partitions are built before a study is expanded (:meth:`prepare_study`),
because expanding reads them.

Failure policy (a): a run/world that fails is *recorded* (status + stdout
tail in its manifest) and the sweep continues — infeasibility is data.
Policy (b), the auto-anchor retry ladder, is implemented below but its
call site ships commented; see ``ensure_world``.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

from fedwater.config import load_base_params
from fedwater.networks import partitions as pstore

from . import harvest as harvest_mod
from .spec import expand_study, load_studies, study_partitions, with_anchor

_VALIDATION_MARKERS = ("sim_validation", "V1_", "V2_", "V3_", "V4_", "V5_",
                       "V6_", "pressure floor", "mass balance")
_VERSION_PKGS = ("numpy", "pandas", "scipy", "scikit-learn", "torch",
                 "wntr", "kedro", "aeon", "networkx", "pyarrow")

RETAINABLE = {
    "prototype_history": "data/07_model_output/prototype_history.parquet",
    "global_prototype_history":
        "data/07_model_output/global_prototype_history.parquet",
    "latent_trajectories": "data/07_model_output/latent_trajectories.parquet",
    "latent_trajectories_residualized":
        "data/07_model_output/latent_trajectories_residualized.parquet",
    "drift_signals": "data/07_model_output/drift_signals.csv",
    "corrected_drift_signals":
        "data/07_model_output/corrected_drift_signals.csv",
    "fl_training_log": "data/08_reporting/fl_training_log.csv",
}

# A world build is only `ok` if these exist — a zero exit code is a claim,
# not evidence (learned from D0: a world clone missing its dependence
# battery cost 40 downstream runs before anything complained).
WORLD_REQUIRED = (
    "data/03_primary/sensor_placement.csv",
    "data/03_primary/gt_topology.csv",
    "data/03_primary/gt_drift_schedule.csv",
    "data/03_primary/gt_dependence_battery.csv",
    "data/08_reporting/validation_report.csv",
    "data/07_model_output/clients",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in _VERSION_PKGS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               default=str))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _link_tree(src: Path, dst: Path) -> None:
    """Hardlink every file under ``src`` into ``dst`` (copy on cross-device).
    Pipelines never write into these linked inputs, and pandas/kedro saves
    replace files rather than mutating them, so the cached world is safe."""
    if not src.exists():
        return
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)


class ExperimentEngine:
    """One engine, many studies. See module docstring for the layout."""

    def __init__(self, project_root: Path | None = None,
                 root: Path | str | None = None):
        self.project = Path(project_root or Path.cwd()).resolve()
        root = Path(root) if root is not None else Path("data/09_experiments")
        self.root = root if root.is_absolute() else self.project / root
        self.env = {**os.environ, "KEDRO_LOGGING_CONFIG": ""}
        self._src_hash = self._hash_source()

    # -- source identity ---------------------------------------------------
    def _hash_source(self) -> str:
        h = sha256()
        for base in (self.project / "src", self.project / "conf" / "base"):
            for f in sorted(base.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    h.update(str(f.relative_to(self.project)).encode())
                    h.update(f.read_bytes())
        return h.hexdigest()[:12]

    # -- clone materialization ----------------------------------------------
    def _materialize_clone(self, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache",
                                        ".ruff_cache")
        shutil.copytree(self.project / "src", dest / "src", ignore=ignore)
        shutil.copytree(self.project / "conf" / "base", dest / "conf" / "base",
                        ignore=ignore)
        (dest / "conf" / "local").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.project / "pyproject.toml", dest / "pyproject.toml")
        shutil.copytree(self.project / "data" / "01_raw",
                        dest / "data" / "01_raw")

    def _globals(self, world: dict) -> dict:
        """The world's globals plus the ABSOLUTE probe store, so every clone
        reads and writes one set of probe stacks."""
        return {**(world.get("globals") or {}),
                "probe_store": str(self.root / "probes")}

    @staticmethod
    def _write_local(clone: Path, params: dict, globals_: dict | None) -> None:
        """Full-block conf/local overrides. `globals` selects the network
        bundle and partition the catalog interpolates (each top-level key is
        written whole: Kedro replaces them destructively); `parameters` is
        everything else."""
        local = clone / "conf" / "local"
        local.mkdir(parents=True, exist_ok=True)
        (local / "parameters.yml").write_text(
            yaml.safe_dump(params, sort_keys=False))
        if globals_:
            (local / "globals.yml").write_text(
                yaml.safe_dump(globals_, sort_keys=False))

    def _kedro(self, cwd: Path, pipeline: str | None = None):
        cmd = [sys.executable, "-m", "kedro", "run"]
        if pipeline:
            cmd += ["--pipeline", pipeline]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=cwd, env=self.env,
                           capture_output=True, text=True)
        return r, round(time.time() - t0, 1)

    @staticmethod
    def _tail(result, n: int = 4000) -> str:
        return (result.stdout + "\n" + result.stderr)[-n:]

    @staticmethod
    def _tasks_completed(result) -> str | None:
        """Kedro's own 'Completed N out of M tasks' line — stored on every
        manifest so a partial pipeline can never again hide behind exit 0."""
        import re
        hits = re.findall(r"Completed (\d+) out of (\d+) tasks",
                          result.stdout + result.stderr)
        return f"{hits[-1][0]}/{hits[-1][1]}" if hits else None

    # -- partitions -----------------------------------------------------------
    def ensure_partitions(self, pairs) -> dict:
        """Build every ``(network, method)`` partition that is missing or stale.

        Freshness is ``networks.partitions.status`` against the CURRENT base
        parameters and ``.inp``. A build runs ``--pipeline districting`` in a
        scratch clone that selects exactly that network and method, then
        copies ``partitions/<method>/`` into the project -- the project's own
        ``conf/local`` is never touched. A failed build raises: nothing can be
        expanded, let alone simulated, on a partition that does not exist.
        """
        districting = load_base_params(self.project)["districting"]
        out = {}
        for network, method in sorted(set(pairs)):
            state = pstore.status(self.project, network, method, districting)
            if state == "current":
                out[(network, method)] = "current"
                continue
            pdir = self.root / "partitions" / f"{network}__{method}"
            clone = pdir / "clone"
            print(f"[partitions] {network}/{method}: {state} -> building ...",
                  end="", flush=True)
            self._materialize_clone(clone)
            self._write_local(clone, {}, {
                "network": network,
                "districting": {"active": method, "methods": [method]}})
            result, seconds = self._kedro(clone, pipeline="districting")
            built = pstore.partition_dir(clone, network, method)
            ok = (result.returncode == 0
                  and pstore.status(clone, network, method, districting)
                  == "current")
            _write_json(pdir / "manifest.json", {
                "network": network, "method": method, "previous": state,
                "status": "ok" if ok else "failed", "seconds": seconds,
                "created_utc": _utcnow(), "src_hash": self._src_hash,
                "tasks_completed": self._tasks_completed(result),
                "stdout_tail": self._tail(result, 1500 if ok else 4000)})
            print(" ok" if ok else " FAILED", flush=True)
            if not ok:
                raise RuntimeError(
                    f"districting failed for {network}/{method}; see "
                    f"{pdir / 'manifest.json'}")
            target = pstore.partition_dir(self.project, network, method)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(built, target)
            shutil.rmtree(clone)
            out[(network, method)] = "built"
        return out

    def prepare_study(self, name: str) -> dict:
        """Build the partitions a study needs, expand it, and check that every
        dynamic world's probe schedule fits its horizon."""
        self.ensure_partitions(study_partitions(name, self.project))
        study_def = expand_study(name, self.project)
        self.preflight_placement(study_def)
        return study_def

    def preflight_placement(self, study_def: dict) -> None:
        """Plan every dynamic world's probe schedule before anything runs.

        ``excite.horizon_plan`` is pure planning (graph distances, no
        hydraulics), so a horizon that cannot convert and settle a partition --
        seed eccentricity, whole-year windows, the world's own front -- is
        reported for the whole study up front instead of failing world by
        world after their simulations.
        """
        import wntr

        from fedwater.placement import excite
        from .spec import bundle

        models, problems = {}, []
        for w in study_def["worlds"]:
            eff = w["effective"]
            sp = eff.get("sensor_placement") or {}
            if sp.get("source", "dynamic") != "dynamic":
                continue
            network, method = w["network"], w["flat"]["districting"]
            if network not in models:
                models[network] = wntr.network.WaterNetworkModel(
                    str(pstore.bundle_dir(self.project, network) / "network.inp"))
            b = bundle(self.project, network, method)
            world = {k: eff[k] for k in ("time", "scenario", "patterns")}
            try:
                excite.horizon_plan(models[network], b["districts"],
                                    b["partition"], sp["probe"], world)
            except ValueError as exc:
                problems.append(f"  {w['sim_hash']} {network}/{method}: {exc}")
        if problems:
            raise ValueError("probe schedule does not fit for "
                             f"{len(problems)} world(s):\n" + "\n".join(problems))

    # -- worlds --------------------------------------------------------------
    def ensure_world(self, world: dict) -> dict:
        """Simulate (or reuse) one world; returns ``{status, dir, cached}``."""
        wdir = self.root / "worlds" / world["sim_hash"]
        manifest = _read_json(wdir / "manifest.json")
        if manifest and manifest.get("sim_hash") == world["sim_hash"] \
                and manifest.get("status") == "ok":
            return {"status": "ok", "dir": wdir, "cached": True,
                    "sim_hash": world["sim_hash"]}

        clone = wdir / "clone"
        self._materialize_clone(clone)
        self._write_local(clone, world["override"], self._globals(world))
        result, seconds = self._kedro(clone)
        missing = [rel for rel in WORLD_REQUIRED
                   if not (clone / rel).exists()]
        if result.returncode == 0 and not missing:
            status, tail = "ok", self._tail(result, 1500)
        elif result.returncode == 0:
            status = "sim_incomplete:" + ",".join(
                Path(rel).name for rel in missing)
            tail = self._tail(result)
        elif any(m in result.stdout + result.stderr
                 for m in _VALIDATION_MARKERS):
            status, tail = "sim_validation_failed", self._tail(result)
        else:
            status, tail = "sim_failed", self._tail(result)

        _write_json(wdir / "manifest.json", {
            "sim_hash": world["sim_hash"], "status": status,
            "world": world["flat"], "override": world["override"],
            "effective": world["effective"],
            "seconds": seconds, "created_utc": _utcnow(),
            "tasks_completed": self._tasks_completed(result),
            "versions": _versions(), "src_hash": self._src_hash,
            "stdout_tail": tail,
        })

        # if status == "sim_validation_failed":
        #     # ---- Physics-constraint policy (b): auto-anchor ladder ----
        #     # Default policy (a) records infeasibility and moves on (the
        #     # sweep treats it as data). Uncomment to instead retry this
        #     # world at progressively reduced demand anchors until V-checks
        #     # pass. Caveat the manifest makes visible: the hydraulic
        #     # operating point then differs across worlds — a confound to
        #     # disclose in any cross-world comparison.
        #     retried = self._anchor_retry_ladder(world)
        #     if retried is not None:
        #         return retried

        return {"status": status, "dir": wdir, "cached": False,
                "sim_hash": world["sim_hash"]}

    def _anchor_retry_ladder(self, world: dict,
                             factors=(0.8, 0.6, 0.5, 0.4)) -> dict | None:
        """Policy (b): rebuild a validation-failed world at reduced
        ``anchor_scale`` (base x factor), returning the first that passes.
        Each retry is a *different* world (different hash, own manifest,
        flagged ``anchor_autocalibrated``); the failed original keeps its
        manifest, so the papertrail shows both. Returns None if the whole
        ladder fails."""
        base_anchor = float(world["flat"]["anchor_scale"])
        for factor in factors:
            retry = with_anchor(world, round(base_anchor * factor, 6))
            out = self.ensure_world(retry)
            if out["status"] == "ok":
                manifest_path = out["dir"] / "manifest.json"
                manifest = _read_json(manifest_path) or {}
                manifest["anchor_autocalibrated"] = {
                    "from": base_anchor, "factor": factor,
                    "original_sim_hash": world["sim_hash"]}
                _write_json(manifest_path, manifest)
                out["world"] = retry
                return out
        return None

    # -- runs -----------------------------------------------------------------
    def ensure_run(self, study: str, world: dict, run: dict,
                   harvest: tuple = (), retain: tuple = (),
                   keep_clone: bool = False) -> dict:
        """Execute (or reuse) one run of ``run`` on ``world`` for ``study``."""
        rid = f"{world['sim_hash']}__{run['run_hash']}"
        rdir = self.root / study / "runs" / rid
        manifest = _read_json(rdir / "manifest.json")
        if manifest and manifest.get("run_hash") == run["run_hash"] \
                and manifest.get("sim_hash") == world["sim_hash"] \
                and manifest.get("status") == "ok":
            return {"status": "ok", "dir": rdir, "cached": True, "id": rid}

        wdir = self.root / "worlds" / world["sim_hash"]
        wmanifest = _read_json(wdir / "manifest.json") or {}
        stage_seconds, status, tail = {}, "ok", None
        stage_tasks: dict = {}
        if wmanifest.get("status") != "ok":
            status = f"world_{wmanifest.get('status', 'missing')}"
        else:
            clone = rdir / "clone"
            self._materialize_clone(clone)
            for rel in ("data/03_primary", "data/07_model_output/clients",
                        "data/08_reporting"):
                # 08_reporting carries the sim-side reports (V-checks, oracle
                # structure recovery); run pipelines only ADD files there.
                _link_tree(wdir / "clone" / rel, clone / rel)
            self._write_local(clone, {**world["override"], "fl": run["fl"]},
                              self._globals(world))
            stage_tasks = {}
            for pipeline in run["pipelines"]:
                result, seconds = self._kedro(clone, pipeline)
                stage_seconds[pipeline] = seconds
                stage_tasks[pipeline] = self._tasks_completed(result)
                tail = self._tail(result, 1500)
                if result.returncode != 0:
                    status, tail = f"failed:{pipeline}", self._tail(result)
                    break

        harvested = []
        if status == "ok":
            data = rdir / "clone" / "data"
            (rdir / "harvest").mkdir(parents=True, exist_ok=True)
            for group in harvest_mod.resolve_groups(harvest):
                try:
                    df = harvest_mod.GROUPS[group](data)
                    df.to_parquet(rdir / "harvest" / f"{group}.parquet")
                    harvested.append(group)
                except Exception as exc:  # noqa: BLE001 — recorded, not fatal
                    status, tail = f"harvest_failed:{group}", repr(exc)
                    break
            (rdir / "retained").mkdir(parents=True, exist_ok=True)
            for name in retain:
                src = rdir / "clone" / RETAINABLE[name]
                if src.exists():
                    shutil.copy2(src, rdir / "retained" / src.name)

        _write_json(rdir / "manifest.json", {
            "study": study, "id": rid,
            "sim_hash": world["sim_hash"], "run_hash": run["run_hash"],
            "world": world["flat"], "run": run["flat"],
            "fl_effective": run["fl"], "pipelines": list(run["pipelines"]),
            "status": status, "stage_seconds": stage_seconds,
            "stage_tasks": stage_tasks,
            "harvested": harvested, "retained": list(retain),
            "created_utc": _utcnow(), "versions": _versions(),
            "src_hash": self._src_hash,
            "world_src_hash": wmanifest.get("src_hash"),
            "src_drift": wmanifest.get("src_hash") not in (None,
                                                           self._src_hash),
            "stdout_tail": tail,
        })
        if not keep_clone and (rdir / "clone").exists():
            shutil.rmtree(rdir / "clone")
        return {"status": status, "dir": rdir, "cached": False, "id": rid}

    # -- study orchestration ---------------------------------------------------
    def run_study(self, study_def: dict, n_jobs: int = 1,
                  limit: int | None = None, dry_run: bool = False
                  ) -> pd.DataFrame:
        name = study_def["name"]
        combos = [(w, r) for w in study_def["worlds"]
                  for r in study_def["runs"]][:limit]
        plan = pd.DataFrame(
            [{**w["flat"], **r["flat"], "sim_hash": w["sim_hash"],
              "run_hash": r["run_hash"]} for w, r in combos])
        if dry_run:
            return plan

        threads = max(1, (os.cpu_count() or 1) // max(1, n_jobs))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS"):
            self.env[var] = str(threads)

        worlds = {w["sim_hash"]: w for w, _ in combos}
        print(f"[{name}] {len(worlds)} world(s), {len(combos)} runs, "
              f"n_jobs={n_jobs}", flush=True)
        # Worlds build SERIALLY: there are few of them, they are heavy, and
        # concurrent full-pipeline builds were the one combination the D0
        # incident left untested. Runs parallelize below as before.
        for world in worlds.values():
            print(f"[{name}] world {world['sim_hash']} "
                  f"({world['flat'].get('variant')}) ...",
                  end="", flush=True)
            built = self.ensure_world(world)
            print(f" {built['status']}"
                  + (" (cached)" if built.get("cached") else ""), flush=True)
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            list(pool.map(
                lambda wr: self.ensure_run(
                    name, wr[0], wr[1], harvest=study_def["harvest"],
                    retain=study_def["retain"]),
                combos))
        return self.collect(name)

    def collect(self, study: str) -> pd.DataFrame:
        """Rebuild the study index + per-group tables from run manifests."""
        runs_dir = self.root / study / "runs"
        rows, groups = [], {}
        for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
            m = _read_json(manifest_path) or {}
            rdir = manifest_path.parent
            row = {"study": study, "sim_hash": m.get("sim_hash"),
                   "run_hash": m.get("run_hash"), "status": m.get("status"),
                   **{k: v for k, v in (m.get("world") or {}).items()},
                   **{k: v for k, v in (m.get("run") or {}).items()},
                   "seconds_total": sum((m.get("stage_seconds") or {}).values()),
                   "created_utc": m.get("created_utc"),
                   "src_hash": m.get("src_hash")}
            row.update(harvest_mod.headline(rdir, m))
            rows.append(row)
            for pq in (rdir / "harvest").glob("*.parquet"):
                df = pd.read_parquet(pq)
                df.insert(0, "run_hash", m.get("run_hash"))
                df.insert(0, "sim_hash", m.get("sim_hash"))
                groups.setdefault(pq.stem, []).append(df)

        index = pd.DataFrame(rows)
        out_dir = self.root / study
        out_dir.mkdir(parents=True, exist_ok=True)
        if not index.empty:
            index.to_parquet(out_dir / "runs.parquet")
        (out_dir / "harvest").mkdir(exist_ok=True)
        for group, parts in groups.items():
            pd.concat(parts, ignore_index=True).to_parquet(
                out_dir / "harvest" / f"{group}.parquet")
        return index


def run_study(name: str, project_root: Path | str = ".", n_jobs: int = 1,
              limit: int | None = None, dry_run: bool = False
              ) -> pd.DataFrame:
    """Module-level convenience: resolve + execute a named study from
    ``conf/base/experiments.yml``."""
    project_root = Path(project_root)
    root = load_studies(project_root).get("root", "data/09_experiments")
    engine = ExperimentEngine(project_root, root=root)
    study_def = engine.prepare_study(name)
    return engine.run_study(study_def, n_jobs=n_jobs, limit=limit,
                            dry_run=dry_run)
