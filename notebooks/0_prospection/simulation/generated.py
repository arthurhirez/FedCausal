"""Bridge from a *generator notebook's* output folder to the `World` surface.

This is deliberately **not** a subclass of `worlds.World` and does not touch
`worlds.py`: that module's contract (`manifest.json` + `clone/`) stays the
one for cached, pipeline-produced worlds. A generator notebook's output is a
different shape -- no manifest, no topology, curated sensors given directly
rather than resolved from `manifest.effective.sensors` -- so this is a
free-standing, duck-typed `GeneratedWorld` implementing exactly the surface
`data.py` / `train.py` / `cluster.py` / `metrics.py` / `store.py` actually
call on a `world` (verified by grep, not assumption): `sim_hash`, `tag`,
`path`, `time`, `steps_day`, `params`, `fl`, `pressures`, `flows`,
`sensors_manual`, `artifacts()`, `warmup_months`, `first_switch`,
`drift_district`, `phase_of()`, `phase_months()`, `phases()`, `regimes()`.
Because that surface matches, `data.build_windows`, `train.run_cell`,
`cluster.score_run`, `metrics.score_run`, `store.save_run`, `figures.*` and
`widgets.*` all run against a `GeneratedWorld` completely unmodified.

Two things are genuinely different here, not just missing, and are NOT
papered over:

1. **No topology.** `assignment.csv` maps node -> district; there is no
   pipe -> district mapping and no `.inp`/`wn_variant.pkl` in this drop, so
   only the *curated* sensor placement (given directly, matching the shape
   of `manifest.effective.sensors`) is usable. `artifacts()` raises rather
   than pretending a topology-driven placement is available.

2. **The drift is not "one district flips to one token."** `gt_drift_schedule`
   here is a single hop-spreading event from a seed node that can touch
   MULTIPLE districts at once, each node with its own continuous
   `switch_month` and a fractional `magnitude` (partial land-use
   conversion), not a discrete map string. So `drift_district` is honestly
   `None` here (everything that special-cases a single mover -- `_tag`'s
   `drift_status` column, `metrics.drift_metrics` -- degrades to a harmless
   no-op through their existing `is None` guards) and `regimes()`/`phases()`
   are instead derived from the REALIZED outcome in `gt_trajectory.csv`
   (population-weighted dominant land-use category per district per month),
   generalized to handle several districts moving on different timelines.
"""
from __future__ import annotations

import pathlib
from functools import cached_property

import numpy as np
import pandas as pd
import yaml

__all__ = ["GeneratedWorld", "load_generated", "load_sensors",
           "category_shares", "plot_category_shares", "DEFAULT_FL"]

_CATS = ("residential", "commercial", "industrial")
_LETTER = {"residential": "R", "commercial": "C", "industrial": "I"}

# The subset of `fl` a generated world needs: `preprocessing`/`model`/
# `training` are overwritten by `CellSpec.as_fl`; `dependence`/
# `personalization`/`drift` are carried through untouched for anything
# downstream that reads them. Values are the pipeline's own defaults
# (`conf/base/parameters.yml`), not invented.
DEFAULT_FL = {
    "preprocessing": {"interval_agg_h": 2, "window_size": 84, "step_size": 1,
                      "reference_months": 2, "label_threshold": 0.75,
                      "feature_range": [-1.0, 1.0]},
    "model": {"lstm_units": 30, "reg_ratio": 0.5},
    "training": {"rounds": 15, "local_epochs": 1, "learning_rate": 1e-3,
                "batch_size": 64, "participation": 1.0, "averaging": "equal",
                "proto_alpha": 0.2, "infonce_temperature": 0.02,
                "device": "cpu"},
    "dependence": {"max_lag_windows": 6, "n_surrogates": 200,
                  "n_surrogates_expensive": 50},
    "personalization": {"domain_weight": 0.5, "coupling_penalty": 0.5,
                        "coupling_q_threshold": 0.05},
    "drift": {"reference_months": 6, "k_sigma": 3.0, "persistence": 2},
}


def load_sensors(spec) -> dict[str, dict[str, list[str]]]:
    """A curated sensor map from a dict, a YAML path, or YAML text.

    Accepts either the bare top-level mapping (`District_A: {pressure: [...],
    flow: [...]}`, as pasted) or the same thing nested under a `sensors:` key.
    Normalises id lists to `str` and drops any key that isn't
    `pressure`/`flow`, matching `worlds.World.sensors_manual`'s shape exactly
    so nothing downstream needs to know the difference.
    """
    if isinstance(spec, dict):
        raw = spec
    else:
        text = (pathlib.Path(spec).read_text()
                if pathlib.Path(str(spec)).exists() else str(spec))
        raw = yaml.safe_load(text)
    raw = raw.get("sensors", raw)
    out = {}
    for d, kinds in raw.items():
        if not isinstance(kinds, dict):
            continue
        out[str(d)] = {k: [str(x) for x in (kinds.get(k) or [])]
                       for k in ("pressure", "flow")}
    if not out:
        raise ValueError(f"no district -> {{pressure, flow}} mapping found "
                         f"in {spec!r}")
    return out


def category_shares(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted (district, month) -> category shares + dominant.

    The truth table this whole adapter is built on: `gt_trajectory.csv`'s
    per-node `residential/commercial/industrial` shares, aggregated to the
    district level by node population. `dominant` is the argmax category's
    letter -- a coarser, discrete readout of the same continuous shares this
    function also returns, so both are available to the notebook.
    """
    rows = []
    for (d, m), g in trajectory.groupby(["district", "month"]):
        wt = g["population"].clip(lower=1e-9)
        shares = {c: float(np.average(g[c], weights=wt)) for c in _CATS}
        rows.append({"district": d, "month": int(m), **shares,
                     "dominant": _LETTER[max(shares, key=shares.get)]})
    return pd.DataFrame(rows).sort_values(["district", "month"]
                                          ).reset_index(drop=True)


def _phases_from_categories(cat: pd.DataFrame, n_months: int,
                            buffer_months: int = 0):
    """Generalizes `World.phases()` to several districts on different
    timelines: `init` ends at the EARLIEST district's first departure from
    its own month-0 category; `final` begins after the LATEST district
    settles into its own final-month category. A district that never changes
    contributes no constraint (it is label-clean everywhere)."""
    first, last = {}, {}
    for d, g in cat.groupby("district"):
        g = g.sort_values("month")
        dom, months = g["dominant"].to_numpy(), g["month"].to_numpy()
        moved0 = months[dom != dom[0]]
        movedF = months[dom != dom[-1]]
        first[d] = int(moved0.min()) if len(moved0) else n_months
        last[d] = int(movedF.max()) if len(movedF) else -1

    first_switch = min(first.values())
    last_switch = max(last.values())
    if last_switch < 0:                       # nothing ever changes
        init, final = (0, n_months), (n_months, n_months)
    else:
        init = (0, first_switch)
        final = (min(last_switch + 1 + buffer_months, n_months), n_months)
    return init, final, first_switch, last_switch, first, last


class GeneratedWorld:
    """Duck-typed `World` over a generator notebook's output folder.

    Not constructed directly -- use `load_generated`. Every property here
    exists because something in `data`/`train`/`cluster`/`metrics`/`store`
    calls it on a `world`; nothing extra was added speculatively.
    """

    def __init__(self, path, sensors: dict, resolution_h: float = 1.0,
                days_per_month: float = 30.0, n_months: int | None = None,
                seed: int = 42, buffer_months: int = 0,
                fl_overrides: dict | None = None, tag: str | None = None):
        self.path = pathlib.Path(path)
        self._sensors = sensors
        self._resolution_h = float(resolution_h)
        self._days_per_month = float(days_per_month)
        self._n_months_override = n_months
        self._buffer_months = int(buffer_months)
        self._seed = int(seed)
        self._fl_overrides = fl_overrides or {}
        self._tag = tag or self.path.name

    # ------------------------------------------------------------- identity
    @property
    def sim_hash(self) -> str:
        return self.path.name

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def params(self) -> dict:
        return {"seed": self._seed}

    @property
    def fl(self) -> dict:
        import copy
        from worlds import deep_merge
        return deep_merge(copy.deepcopy(DEFAULT_FL), self._fl_overrides)

    # ---------------------------------------------------------------- time
    @cached_property
    def time(self) -> dict:
        n_steps = len(self.pressures)
        steps_day = round(24.0 / self._resolution_h)
        per_month = steps_day * self._days_per_month
        n_months = self._n_months_override
        if n_months is None:
            if n_steps % per_month:
                raise AssertionError(
                    f"{n_steps} steps does not divide evenly by "
                    f"{per_month:.0f} steps/month (resolution_h="
                    f"{self._resolution_h}, days_per_month="
                    f"{self._days_per_month}); pass n_months= explicitly or "
                    f"correct resolution_h/days_per_month")
            n_months = int(n_steps // per_month)
        return {"resolution_h": self._resolution_h,
               "days_per_month": self._days_per_month,
               "n_months": int(n_months)}

    @property
    def steps_day(self) -> int:
        return round(24.0 / self.time["resolution_h"])

    @property
    def n_months(self) -> int:
        return int(self.time["n_months"])

    # --------------------------------------------------------------- series
    @cached_property
    def pressures(self) -> pd.DataFrame:
        return self._with_month(pd.read_parquet(self.path / "pressures.parquet"))

    @cached_property
    def flows(self) -> pd.DataFrame:
        return self._with_month(pd.read_parquet(self.path / "flows.parquet"))

    def _with_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """Neither table carries a `month` column here (unlike cached
        worlds); it is `step // steps_per_month`, added once and cached."""
        steps_day = round(24.0 / self._resolution_h)
        per_month = steps_day * self._days_per_month
        month = (np.arange(len(df)) // per_month).astype(int)
        return df.assign(month=month)

    # ----------------------------------------------------------- placement
    @property
    def sensors_manual(self) -> dict:
        return self._sensors

    @property
    def client_names(self) -> list[str]:
        return sorted(self._sensors)

    def artifacts(self):
        raise NotImplementedError(
            f"{self.sim_hash}: no network topology in this generator drop "
            "(no .inp / wn_variant.pkl), so only placement='manual' "
            "(the curated sensors passed to load_generated) is available")

    # --------------------------------------------------------------- truth
    @cached_property
    def trajectory(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "gt_trajectory.csv")

    @cached_property
    def category_table(self) -> pd.DataFrame:
        return category_shares(self.trajectory)

    @cached_property
    def _phase_calc(self):
        return _phases_from_categories(self.category_table, self.n_months,
                                       self._buffer_months)

    def phases(self) -> dict[str, tuple[int, int]]:
        init, final, *_ = self._phase_calc
        return {"init": init, "final": final}

    @property
    def first_switch(self) -> int | None:
        return self._phase_calc[2]

    @property
    def last_switch(self) -> int | None:
        return self._phase_calc[3]

    @property
    def switch_by_district(self) -> tuple[dict, dict]:
        """(first-change month, last-change month) per district -- the
        per-district detail `phases()` collapses to one global boundary."""
        return self._phase_calc[4], self._phase_calc[5]

    @property
    def warmup_months(self) -> int | None:
        return self.first_switch

    @property
    def drift_district(self) -> None:
        """Honestly `None`: the drift here is multi-district and fractional,
        not one client flipping to one token. See the module docstring for
        what this degrades to downstream."""
        return None

    def phase_months(self, phase: str) -> list[int]:
        lo, hi = self.phases()[phase]
        return list(range(int(lo), int(hi)))

    def phase_of(self, month: int) -> str:
        for name, (lo, hi) in self.phases().items():
            if lo <= month < hi:
                return name
        return "transition"

    def regimes(self, phase: str | None = "init", basis: str = "token"
               ) -> dict[str, str]:
        """district -> dominant land-use letter, majority vote over the
        phase's months. `basis` is accepted for interface parity with
        `worlds.World.regimes` (there is no separate income axis here, so
        anything other than "token" just returns the same letter)."""
        phase = phase or "init"
        if phase not in ("init", "final"):
            raise KeyError(f"no clean regime map for phase {phase!r} "
                           "(only 'init' and 'final' are label-clean)")
        months = self.phase_months(phase)
        cat = self.category_table[self.category_table["month"].isin(months)]
        if not len(cat):
            raise KeyError(f"phase {phase!r} has no months in range "
                           f"{self.phases()[phase]}")
        return {d: g["dominant"].mode().iloc[0]
               for d, g in cat.groupby("district") if d in self._sensors}

    def partition(self, phase: str = "init", basis: str = "token"
                 ) -> dict[str, int]:
        reg = self.regimes(phase=phase, basis=basis)
        order = {t: i for i, t in enumerate(sorted(set(reg.values())))}
        return {c: order[t] for c, t in reg.items()}

    def drifted_districts(self) -> list[str]:
        """Districts whose dominant category differs between init and
        final -- the multi-district analogue of a single `drift_district`."""
        a, b = self.regimes(phase="init"), self.regimes(phase="final")
        return sorted(c for c in a if a.get(c) != b.get(c))

    # -------------------------------------------------------------- summary
    def summary(self) -> pd.Series:
        ph = self.phases()
        return pd.Series({
            "sim_hash": self.sim_hash, "tag": self.tag,
            "n_months": self.n_months, "resolution_h": self.time["resolution_h"],
            "steps_day": self.steps_day, "clients": len(self.client_names),
            "drifted_districts": ",".join(self.drifted_districts()) or "(none)",
            "first_switch": self.first_switch, "last_switch": self.last_switch,
            "init_months": f"{ph['init'][0]}..{ph['init'][1] - 1}",
            "final_months": f"{ph['final'][0]}..{ph['final'][1] - 1}"
                            if ph["final"][1] > ph["final"][0] else "(empty)",
            "regime_init": self.regimes(phase="init"),
            "regime_final": self.regimes(phase="final"),
        })


def load_generated(path, sensors, **kw) -> GeneratedWorld:
    """`load_world`'s counterpart for a generator notebook's output folder.

    `sensors` is a dict, a YAML path, or YAML text -- see `load_sensors`.
    `**kw` forwards to `GeneratedWorld` (`resolution_h`, `days_per_month`,
    `n_months`, `seed`, `buffer_months`, `fl_overrides`, `tag`).
    """
    p = pathlib.Path(path).expanduser().resolve()
    missing = [f for f in ("pressures.parquet", "flows.parquet",
                           "gt_trajectory.csv")
              if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"{p}: missing {missing}")
    return GeneratedWorld(p, load_sensors(sensors), **kw)


def plot_category_shares(ax, world: GeneratedWorld, category: str = "industrial"):
    """One curve per district: population-weighted share of `category`.

    The multi-district analogue of `figures.plot_drift_progress`, which
    assumes a single mover -- not a fit here, since this generator's drift
    can touch several districts on independent timelines. Districts that
    never move still get a (flat) line, so "nothing happened here" is as
    visible as "something happened there".
    """
    import figures as F
    cat = world.category_table
    cmap = F.colour_map(world.client_names)
    for d, g in cat.groupby("district"):
        if d not in cmap:
            continue
        g = g.sort_values("month")
        ax.plot(g["month"], g[category], marker="o", ms=3, lw=1.3,
                color=cmap[d], label=d)
    ph = world.phases()
    ax.axvspan(*ph["init"], color="#55A868", alpha=.12, lw=0, label="init")
    if ph["final"][1] > ph["final"][0]:
        ax.axvspan(*ph["final"], color="#4C72B0", alpha=.12, lw=0, label="final")
    ax.set_xlabel("month")
    ax.set_ylabel(f"{category} share")
    ax.set_ylim(-.02, 1.02)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(f"{world.tag} · moved: {', '.join(world.drifted_districts()) or '(none)'}",
                fontsize=9, loc="left")
    return ax
