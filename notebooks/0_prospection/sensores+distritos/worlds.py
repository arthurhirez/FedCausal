"""World loading: a cached experiment world directory -> a `World` handle.

Entry point is a **path to the world directory** (the one that holds
`manifest.json` and `clone/`), never a repo-wide discovery scan. Nothing in
this module trains or scores; it answers "what is this world" so that
`data`, `train` and `cluster` never have to parse a manifest again.

Three jobs, in order of how much trouble they save downstream:

1. **Params merge.** `manifest.effective` is authoritative for everything the
   simulator was actually run with (`time`, `scenario`, `sensors`, `noise`),
   but it does NOT carry `fl` -- that lives in the clone's
   `conf/base/parameters.yml`. `params` is the deep merge, effective winning.
   Fields that the manifest is allowed to leave `None` (`resolution_h`,
   `n_months`, `days_per_month`) are coerced numeric here, once: an
   uncoerced `int(None)` kills every downstream cell in the world.

2. **Ground truth, derived two independent ways.** The regime token per
   district comes from `world.consumption_map` (a string) *and* from
   `scenario.income_landuse_mapping` (structured pairs). Both are computed and
   asserted to agree, so a silent format change surfaces here instead of as a
   quiet mislabelling in a score table.

3. **Phases derived from the drift schedule, not from a midpoint.** Nodes
   switch over many months and each switch carries a `drift_ramp_days` ramp,
   so the label-clean stretches are `[0, first_switch)` and
   `[last_switch + ramp_months, n_months)`. `drift_progress` exposes the
   demand-weighted switched fraction per month, so the excluded transition
   band is a visible column rather than an arbitrary convention.
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import yaml

__all__ = ["World", "load_world", "num", "deep_merge"]

# income / land-use -> the single letters used by `consumption_map`
_INCOME_LETTER = {"low": "L", "medium": "M", "high": "H"}
_LANDUSE_LETTER = {"industrial": "I", "residential": "R",
                   "commercial": "C", "mixed": "M"}


def num(x, default=None):
    """`float(x)` that survives None / '' / 'nan' -> `default`."""
    v = pd.to_numeric(x, errors="coerce")
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    return default if pd.isna(v) else float(v)


def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins. Neither input is mutated."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _district_names(n: int) -> list[str]:
    return [f"District_{chr(ord('A') + i)}" for i in range(n)]


@dataclass
class World:
    """One cached world. Cheap to construct; heavy tables load on access."""

    path: pathlib.Path
    manifest: dict
    params: dict
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ identity
    @property
    def sim_hash(self) -> str:
        return str(self.manifest.get("sim_hash", self.path.name))

    @property
    def variant(self) -> str:
        return str(self.meta.get("variant", "?"))

    @property
    def tag(self) -> str:
        return (f"{self.drift_district or '?'}__{self.consumption_map or '?'}"
                f"__{self.variant}")

    @property
    def clone(self) -> pathlib.Path:
        return self.path / "clone"

    @property
    def clients_dir(self) -> pathlib.Path:
        return self.clone / "data" / "07_model_output" / "clients"

    # ---------------------------------------------------------------- time
    @cached_property
    def time(self) -> dict:
        """`params.time` with every field numeric, recovered if absent."""
        t = dict(self.params.get("time", {}) or {})
        res = num(t.get("resolution_h"))
        if res is None or res <= 0:
            res = 24.0 / max(1.0, num(self.manifest.get("steps_day"), 24.0))
        dpm = num(t.get("days_per_month"), 30.0) or 30.0
        n = num(t.get("n_months"))
        if n is None or n <= 0:
            n = num((self.params.get("scenario") or {}).get("n_months"))
        if n is None or n <= 0:
            spm = max(1.0, (24.0 / res) * dpm)
            n = max(1.0, round(len(self.pressures) / spm))
        t.update(resolution_h=int(res) if float(res).is_integer() else res,
                 days_per_month=int(dpm), n_months=int(n))
        return t

    @property
    def steps_day(self) -> int:
        return int(round(24.0 / float(self.time["resolution_h"])))

    @property
    def n_months(self) -> int:
        return int(self.time["n_months"])

    @property
    def fl(self) -> dict:
        """`params.fl`, deep-copied, device left as configured."""
        return copy.deepcopy(self.params.get("fl", {}))

    # --------------------------------------------------------------- truth
    @property
    def consumption_map(self) -> str | None:
        cm = self.meta.get("consumption_map")
        return str(cm) if cm else None

    @property
    def scenario(self) -> dict:
        return dict(self.params.get("scenario", {}) or {})

    @property
    def drift(self) -> dict:
        return dict(self.scenario.get("drift", {}) or {})

    @property
    def drift_district(self) -> str | None:
        d = self.drift.get("tgt_district") or self.meta.get("drift_district")
        return str(d) if d else None

    @property
    def warmup_months(self) -> int | None:
        w = num(self.drift.get("warmup_months"))
        return None if w is None else int(w)

    @property
    def ramp_days(self) -> float:
        return num((self.params.get("patterns") or {}).get("drift_ramp_days"), 0.0)

    @property
    def ramp_months(self) -> int:
        dpm = float(self.time["days_per_month"])
        return int(math.ceil(self.ramp_days / dpm)) if self.ramp_days > 0 else 0

    def _regimes_from_mapping(self) -> dict[str, str]:
        """district -> token, from `scenario.income_landuse_mapping`."""
        mapping = self.scenario.get("income_landuse_mapping") or []
        out = {}
        for name, pair in zip(_district_names(len(mapping)), mapping):
            income, land_use = (list(pair) + [None, None])[:2]
            out[name] = (_INCOME_LETTER.get(str(income), "?")
                         + _LANDUSE_LETTER.get(str(land_use), "?"))
        return out

    def _regimes_from_consumption_map(self) -> dict[str, str]:
        """district -> token, from the `consumption_map` string."""
        cm = self.consumption_map
        if not cm:
            return {}
        toks = cm.split("_")
        return dict(zip(_district_names(len(toks)), toks))

    @cached_property
    def regime_sources(self) -> dict[str, dict[str, str]]:
        """Both derivations, kept separately for auditing."""
        return {"income_landuse_mapping": self._regimes_from_mapping(),
                "consumption_map": self._regimes_from_consumption_map()}

    def regimes(self, basis: str = "token", phase: str | None = None,
                strict: bool = True) -> dict[str, str]:
        """district -> regime label.

        basis
            `"token"` the 2-letter income+land-use code (default);
            `"land_use"` / `"income"` just that component. When income is
            uniform across districts -- which it is in the current world
            family -- `token` and `land_use` induce the same partition.
        phase
            `None`/`"init"` the commissioning map; `"final"` the same map with
            the drift district moved to its post-drift token. Anything else
            (a transition month) raises, because no clean label exists there.
        strict
            assert the two independent derivations agree.
        """
        both = self.regime_sources
        a, b = both["income_landuse_mapping"], both["consumption_map"]
        if strict and a and b and a != b:
            raise AssertionError(
                "ground-truth derivations disagree:\n"
                f"  income_landuse_mapping -> {a}\n"
                f"  consumption_map        -> {b}")
        toks = dict(a or b)
        if not toks:
            raise AssertionError(f"{self.sim_hash}: no regime ground truth")

        if phase in ("final",):
            tgt = self.drift_district
            if tgt in toks:
                toks[tgt] = (_INCOME_LETTER.get(str(self.drift.get("to_income")), "?")
                             + _LANDUSE_LETTER.get(str(self.drift.get("to_land_use")), "?"))
        elif phase not in (None, "init"):
            raise KeyError(f"no clean regime map for phase {phase!r} "
                           "(only 'init' and 'final' are label-clean)")

        if basis == "token":
            return toks
        idx = {"income": 0, "land_use": 1}.get(basis)
        if idx is None:
            raise KeyError(f"basis must be token|income|land_use, got {basis!r}")
        return {c: t[idx] if len(t) > idx else "?" for c, t in toks.items()}

    def partition(self, phase: str = "init", basis: str = "token") -> dict[str, int]:
        """Regime labels collapsed to integer group ids (stable by label)."""
        reg = self.regimes(basis=basis, phase=phase)
        order = {t: i for i, t in enumerate(sorted(set(reg.values())))}
        return {c: order[t] for c, t in reg.items()}

    # ------------------------------------------------------------- schedule
    @cached_property
    def drift_schedule(self) -> pd.DataFrame:
        """`gt_drift_schedule` (node, district, drift_month, ...) or empty."""
        f = self.clone / "data" / "03_primary" / "gt_drift_schedule.csv"
        if not f.exists():
            return pd.DataFrame(columns=["node", "district", "drift_month"])
        d = pd.read_csv(f)
        d.columns = [c.strip().lower() for c in d.columns]
        if "drift_month" not in d.columns:
            col = next((c for c in d.columns if "month" in c), None)
            if col:
                d = d.rename(columns={col: "drift_month"})
        d["node"] = d["node"].astype(str)
        d["drift_month"] = pd.to_numeric(d["drift_month"], errors="coerce")
        return d.dropna(subset=["drift_month"])

    @property
    def first_switch(self) -> int | None:
        d = self.drift_schedule
        return None if not len(d) else int(d["drift_month"].min())

    @property
    def last_switch(self) -> int | None:
        d = self.drift_schedule
        return None if not len(d) else int(d["drift_month"].max())

    def phases(self) -> dict[str, tuple[int, int]]:
        """Label-clean month ranges, half-open: {'init': (lo, hi), 'final': ...}.

        `init` ends at the first node switch (so it equals the simulator's
        warm-up when the schedule starts there); `final` begins once the last
        switch has finished ramping. Everything between is transition and
        belongs to neither.
        """
        n = self.n_months
        first = self.first_switch
        last = self.last_switch
        if first is None:
            w = self.warmup_months
            first = int(w) if w else n // 2
        if last is None:
            last = first
        init = (0, int(first))
        final = (min(int(last) + self.ramp_months, n), n)
        return {"init": init, "final": final}

    def phase_months(self, phase: str) -> list[int]:
        lo, hi = self.phases()[phase]
        return list(range(int(lo), int(hi)))

    def phase_of(self, month: int) -> str:
        ph = self.phases()
        for name, (lo, hi) in ph.items():
            if lo <= month < hi:
                return name
        return "transition"

    @cached_property
    def portfolios(self) -> pd.DataFrame | None:
        f = self.clone / "data" / "02_intermediate" / "portfolios_t0.parquet"
        return pd.read_parquet(f) if f.exists() else None

    def drift_progress(self) -> pd.DataFrame:
        """month -> switched fraction of the drift district (0..1).

        Demand-weighted by each node's `volume_m3_month` when
        `portfolios_t0` is available, else node-count weighted. Nothing in
        the POC consumes this; it exists so the transition band is a number.
        """
        sch, tgt = self.drift_schedule, self.drift_district
        months = np.arange(self.n_months)
        if not len(sch) or tgt is None:
            return pd.DataFrame({"month": months, "progress": 0.0})
        sub = sch[sch["district"] == tgt] if "district" in sch.columns else sch

        weight = pd.Series(1.0, index=sub["node"].astype(str))
        pf = self.portfolios
        if pf is not None and {"node", "volume_m3_month"} <= set(pf.columns):
            vol = (pf.assign(node=pf["node"].astype(str))
                     .groupby("node")["volume_m3_month"].sum())
            weight = weight.index.map(vol).to_series(index=weight.index).fillna(0.0)
            if weight.sum() <= 0:
                weight = pd.Series(1.0, index=sub["node"].astype(str))

        total = float(weight.sum()) or 1.0
        sw = pd.Series(sub["drift_month"].to_numpy(),
                       index=sub["node"].astype(str))
        prog = [float(weight[sw[sw <= m].index].sum()) / total for m in months]
        return pd.DataFrame({"month": months, "progress": prog})

    # ----------------------------------------------------------- placements
    @cached_property
    def sensors_manual(self) -> dict[str, dict[str, list[str]]]:
        """`manifest.effective.sensors`: the placement the world was run with.

        Shape: {district: {"pressure": [node, ...], "flow": [link, ...]}}.
        This is what the client CSVs contain, so `manual` needs no network
        object and no placement heuristics.
        """
        raw = self.params.get("sensors") or {}
        out = {}
        for d, kinds in raw.items():
            if not isinstance(kinds, dict):
                continue
            out[d] = {k: [str(x) for x in (v or [])]
                      for k, v in kinds.items() if k in ("pressure", "flow")}
        return out

    @property
    def districts_file(self) -> pathlib.Path:
        return self.clone / "data" / "01_raw" / "districts_graeme.yml"

    @cached_property
    def districts(self) -> dict[str, list[str]]:
        if not self.districts_file.exists():
            return {}
        raw = yaml.safe_load(self.districts_file.read_text()) or {}
        d = raw.get("districts", raw)
        return {k: [str(x) for x in v] for k, v in d.items()}

    @property
    def client_names(self) -> list[str]:
        if self.sensors_manual:
            return sorted(self.sensors_manual)
        if self.districts:
            return sorted(self.districts)
        return sorted(p.stem for p in self.clients_dir.glob("District_*.csv"))

    # -------------------------------------------------------- heavy tables
    @cached_property
    def pressures(self) -> pd.DataFrame:
        return pd.read_parquet(self.clone / "data" / "02_intermediate"
                               / "pressures.parquet")

    @cached_property
    def flows(self) -> pd.DataFrame:
        return pd.read_parquet(self.clone / "data" / "02_intermediate"
                               / "flows.parquet")

    def wn(self):
        """The wntr network (`wn_variant.pkl`). Imported lazily: only the
        topology-driven placements need it, and wntr need not be installed
        for the `manual` path."""
        import pickle
        with open(self.clone / "data" / "02_intermediate" / "wn_variant.pkl",
                  "rb") as fh:
            return pickle.load(fh)

    def artifacts(self) -> dict:
        """The `art`-shaped dict the placement POC's functions expect."""
        return {"wn": self.wn(), "districts": self.districts,
                "pressures": self.pressures, "flows": self.flows,
                "params": self.params, "steps_day": self.steps_day}

    # -------------------------------------------------------------- summary
    def summary(self) -> pd.Series:
        ph = self.phases()
        return pd.Series({
            "sim_hash": self.sim_hash, "tag": self.tag,
            "n_months": self.n_months, "resolution_h": self.time["resolution_h"],
            "steps_day": self.steps_day, "clients": len(self.client_names),
            "drift_district": self.drift_district,
            "warmup_months": self.warmup_months,
            "first_switch": self.first_switch, "last_switch": self.last_switch,
            "ramp_days": self.ramp_days,
            "init_months": f"{ph['init'][0]}..{ph['init'][1] - 1}",
            "final_months": f"{ph['final'][0]}..{ph['final'][1] - 1}",
            "regime_init": "_".join(self.regimes(phase="init")[c]
                                    for c in self.client_names),
            "regime_final": "_".join(self.regimes(phase="final")[c]
                                     for c in self.client_names),
        })


def load_world(path: str | pathlib.Path, base_params: dict | None = None) -> World:
    """Load a world from its directory (the one containing `manifest.json`).

    `base_params` overrides where the non-simulated blocks (notably `fl`) are
    read from; by default the world's own clone conf is used, which is what
    keeps a cached world self-describing.
    """
    p = pathlib.Path(path).expanduser().resolve()
    if p.name == "manifest.json":
        p = p.parent
    mf = p / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"no manifest.json in {p}")
    manifest = json.loads(mf.read_text())
    if manifest.get("status") not in (None, "ok"):
        raise RuntimeError(f"{p.name}: manifest status {manifest.get('status')!r}")

    if base_params is None:
        conf = p / "clone" / "conf" / "base" / "parameters.yml"
        base_params = yaml.safe_load(conf.read_text()) if conf.exists() else {}
    params = deep_merge(base_params or {}, manifest.get("effective") or {})

    return World(path=p, manifest=manifest, params=params,
                 meta=dict(manifest.get("world") or {}))
