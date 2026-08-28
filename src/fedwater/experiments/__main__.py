"""Module CLI: ``python -m fedwater.experiments <command>``.

Commands
--------
list                       Show studies declared in conf/base/experiments.yml.
run <study>                Execute a study (cache-or-run at every level).
    --n-jobs N             Parallel runs (subprocess-bound; BLAS threads are
                           capped per worker automatically).
    --limit K              Only the first K (world x run) combos — smoke runs.
    --dry-run              Print the resolved plan (specs + hashes), execute
                           nothing.
status <study>             Per-run status/timing table from the manifests.
collect <study>            Rebuild runs.parquet + per-group tables from the
                           run directories (normally automatic after `run`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .engine import ExperimentEngine, run_study
from .spec import expand_study, load_studies


def _print(df: pd.DataFrame, n: int | None = None) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 40,
                           "display.max_rows", n or 200):
        print(df.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fedwater.experiments",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=".",
                        help="fedwater project root (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")
    p_run = sub.add_parser("run")
    p_run.add_argument("study")
    p_run.add_argument("--n-jobs", type=int, default=1)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    for cmd in ("status", "collect"):
        p = sub.add_parser(cmd)
        p.add_argument("study")

    args = parser.parse_args(argv)
    project = Path(args.project_root)

    if args.command == "list":
        cfg = load_studies(project)
        rows = []
        for name, study in cfg.get("studies", {}).items():
            d = expand_study(name, project)
            rows.append({"study": name, "worlds": len(d["worlds"]),
                         "runs_per_world": len(d["runs"]),
                         "total_runs": len(d["worlds"]) * len(d["runs"]),
                         "description": (study.get("description", "")
                                         .strip().split("\n")[0][:70])})
        _print(pd.DataFrame(rows))
        return 0

    if args.command == "run":
        index = run_study(args.study, project, n_jobs=args.n_jobs,
                          limit=args.limit, dry_run=args.dry_run)
        if args.dry_run:
            print(f"-- dry run: {len(index)} (world x run) combos --")
            _print(index)
        else:
            summary = index.groupby("status").size().rename("runs")
            print(f"-- {args.study}: {len(index)} runs --")
            print(summary.to_string())
            print(f"index: {ExperimentEngine(project).root}/"
                  f"{args.study}/runs.parquet")
        return 0

    study_def = expand_study(args.study, project)
    engine = ExperimentEngine(project, root=study_def["root"])
    if args.command == "collect":
        index = engine.collect(args.study)
        print(f"collected {len(index)} runs.")
        return 0

    # status
    rows = []
    for m in sorted((engine.root / args.study / "runs").glob("*/manifest.json")):
        d = json.loads(m.read_text())
        rows.append({"id": d.get("id"), "status": d.get("status"),
                     "seconds": sum((d.get("stage_seconds") or {}).values()),
                     "fl_seed": (d.get("run") or {}).get("fl_seed"),
                     "variant": (d.get("world") or {}).get("variant"),
                     "created": d.get("created_utc")})
    if not rows:
        print(f"no runs recorded for '{args.study}'.")
        return 1
    _print(pd.DataFrame(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
