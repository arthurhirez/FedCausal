"""signal_probe.py -- which sensors actually carry a district's land-use shape.

The question
------------
A land-use drift changes a district's demand SHAPE. Sensors do not see demand;
they see pressure and flow, through a coupled hydraulic network. This module
ranks every candidate gauge (every junction for pressure, every link for flow)
by how well its weekly signature reproduces its own district's demand
signature, before and after the drift.

Three things make the naive version useless, and each has a countermeasure
here.

1. **Common mode.** Every sensor on a closed network correlates with the
   network's aggregate diurnal cycle, so raw correlation against own-district
   demand is high nearly everywhere and ranks nothing. Every correlation is
   therefore reported twice: `r_raw`, and `r_res` after projecting the network
   aggregate weekly profile out of both series. `r_res` is the number that
   discriminates.

2. **Sign.** Pressure falls when demand rises, and a link's flow sign is the
   arbitrary orientation EPANET gave it. Both are aligned before anything is
   measured: pressure enters as `-p`, flow as `sign(mean q) * q`. The
   alignment applied is recorded per candidate, so nothing is silently
   flipped.

3. **"Resembles" is not "responds".** A sensor can track the district's shape
   perfectly in both phases and not move at all when the drift lands --
   typically because it is dominated by transit flow. `resemblance` (does it
   look like the district) and `response` (did it move the way the district's
   demand moved) are scored separately; a prototype needs both.

Scale-free features are ADDITIVE CONTRASTS on the z-scored weekly profile
(`night - day`, `weekend - weekday`), not ratios. A ratio needs a positive
series, which pressure is not once aligned; a contrast is defined for every
channel and carries the same information.

Nothing here writes to a world or changes the drift engine. It reads a built
world and returns tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

__all__ = ["Probe", "from_world", "from_lab_world", "weekly_profile",
           "candidates", "battery", "phase_delta", "discriminative_gap",
           "rank", "select", "cluster", "template_fit",
           "zscore", "sensors_yaml"]

EPS = 1e-12

# column naming, fixed by `sensing.extract_sensor_series`
_PREFIX = {"pressure": "p_", "flow": "q_"}

# the consumption-map codec, so a regime is one comparable token
_INC_L = {"low": "L", "medium": "M", "high": "H"}
_LU_L = {"residential": "R", "mixed": "M", "commercial": "C", "industrial": "I"}


# ==========================================================================
# small numerics
# ==========================================================================
def zscore(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Zero mean, unit sd along `axis`. Constant rows come back as zeros."""
    a = np.asarray(a, dtype=float)
    mu = a.mean(axis=axis, keepdims=True)
    sd = a.std(axis=axis, keepdims=True)
    return np.where(sd > EPS, (a - mu) / np.where(sd > EPS, sd, 1.0), 0.0)


def _corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n, T) x (m, T) -> (n, m) Pearson r, both already finite."""
    Az, Bz = zscore(A, -1), zscore(B, -1)
    return (Az @ Bz.T) / Az.shape[-1]


def _project_out(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Remove the component of every row of `A` along `u` (the common mode)."""
    u = np.asarray(u, dtype=float).ravel()
    n = np.linalg.norm(u)
    if n < EPS:
        return np.asarray(A, dtype=float)
    u = u / n
    A = np.asarray(A, dtype=float)
    return A - np.outer(A @ u, u)


def weekly_profile(values: np.ndarray, steps_day: int, step0: int = 0,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """Mean 7-day profile of a contiguous series, shape (..., 7 * steps_day).

    `values` is (n_steps,) or (n_steps, k) over a CONTIGUOUS slice of the
    simulation starting at absolute step `step0`. The weekday of a step is
    `(step // steps_day) % 7`, which is the same global day counter
    `synthesize_demands` uses, so the weekend days here are the weekend days
    the demand model built.
    """
    v = np.asarray(values, dtype=float)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    n_days = v.shape[0] // steps_day
    if n_days < 7:
        raise ValueError(f"need >= 7 whole days, got {n_days}")
    v = v[:n_days * steps_day].reshape(n_days, steps_day, v.shape[1])
    day0 = step0 // steps_day
    wd = (np.arange(n_days) + day0) % 7

    out = np.empty((7, steps_day, v.shape[2]))
    for w in range(7):
        m = wd == w
        out[w] = v[m].mean(axis=0) if m.any() else v.mean(axis=0)
    prof = out.reshape(7 * steps_day, -1).T          # (k, 7*steps_day)
    return prof[0] if squeeze else prof


# ==========================================================================
# context
# ==========================================================================
@dataclass
class Probe:
    """Everything the battery needs, from one built world."""

    districts: dict                      # district -> [junction, ...]
    assets: dict                         # district -> [tank/reservoir, ...]
    demand: pd.DataFrame                 # step x node, + 'month'
    pressures: pd.DataFrame              # step x node, + 'month'
    flows: pd.DataFrame                  # step x link, + 'month'
    schedule: pd.DataFrame               # gt_drift_schedule
    time: dict                           # n_months / days_per_month / resolution_h
    wn: object = None                    # wntr model; only topology needs it
    sensors: dict = field(default_factory=dict)      # the shipped placement
    params: dict = field(default_factory=dict)       # for canonical templates
    mapping: dict = field(default_factory=dict)      # district -> (income, land_use)
    drift: dict = field(default_factory=dict)        # tgt_district / to_* 
    label: str = "world"

    # ------------------------------------------------------------ geometry
    @property
    def steps_day(self) -> int:
        return int(round(24 / float(self.time["resolution_h"])))

    @property
    def steps_month(self) -> int:
        return self.steps_day * int(self.time["days_per_month"])

    @property
    def names(self) -> list[str]:
        return list(self.districts)

    @cached_property
    def node2district(self) -> dict:
        return {n: d for d, ns in self.districts.items() for n in ns}

    @cached_property
    def asset2district(self) -> dict:
        return {n: d for d, ns in (self.assets or {}).items() for n in ns}

    def phases(self) -> dict:
        """Label-clean month ranges, half-open, matching `fedwater_lab.World`.

        `init` runs to the first node switch; `final` starts once the last
        switch has finished ramping. Months between are transition and belong
        to neither -- scoring inside them mixes two regimes in one profile.
        """
        n = int(self.time["n_months"])
        dpm = int(self.time["days_per_month"])
        sch = self.schedule
        if sch is None or not len(sch):
            return {"init": (0, n // 2), "final": (n // 2, n)}
        first, last = int(sch["drift_month"].min()), int(sch["drift_month"].max())
        ramp_days = float((self.params.get("patterns") or {}).get("drift_ramp_days", 0))
        ramp_months = int(np.ceil(ramp_days / dpm)) if ramp_days > 0 else 0
        return {"init": (0, first), "final": (min(last + ramp_months, n - 1), n)}

    def regimes(self, phase: str = "init") -> dict:
        """district -> 2-letter regime token, with the drift applied in `final`.

        Districts that share a token are NOT separable by shape -- the demand
        model gives them the same signature by construction. Every
        "which district does this look like" statistic is therefore reported
        twice: once against districts (`hit`) and once against REGIMES
        (`regime_hit`), and only the second is a claim about recoverability.
        """
        tok = {d: _INC_L.get(str(i), "?") + _LU_L.get(str(lu), "?")
               for d, (i, lu) in (self.mapping or {}).items()}
        tgt = (self.drift or {}).get("tgt_district")
        if phase == "final" and tgt in tok:
            tok[tgt] = (_INC_L.get(str(self.drift.get("to_income")), "?")
                        + _LU_L.get(str(self.drift.get("to_land_use")), "?"))
        return tok

    def month_slice(self, phase: str) -> tuple[int, int]:
        lo, hi = self.phases()[phase]
        if hi - lo < 1:
            raise ValueError(f"phase {phase!r} is empty ({lo}..{hi}); lengthen "
                             "the horizon or lower warmup_months")
        return lo * self.steps_month, hi * self.steps_month

    # ------------------------------------------------------------- columns
    @cached_property
    def demand_nodes(self) -> list[str]:
        """Junctions that carry a demand column (zero-base-demand ones do not)."""
        return [c for c in self.demand.columns
                if c != "month" and c in self.node2district]

    @cached_property
    def district_demand(self) -> pd.DataFrame:
        """step x district, L/s. The ground-truth signal we want to recover."""
        out = {}
        for d, nodes in self.districts.items():
            cols = [n for n in nodes if n in self.demand.columns]
            out[d] = self.demand[cols].sum(axis=1).to_numpy() if cols \
                else np.zeros(len(self.demand))
        return pd.DataFrame(out, index=self.demand.index)

    # ------------------------------------------------------- construction
    @classmethod
    def from_world(cls, world: dict) -> "Probe":
        """From `landuse_world.build`'s return value."""
        p = world["params"]
        dy = world["districts"]
        mapping = dict(zip(list(dy["districts"]),
                           [tuple(x) for x in p["scenario"]["income_landuse_mapping"]]))
        return cls(districts=dy["districts"], assets=dy.get("assets") or {},
                   demand=world["demand"], pressures=world["pressures"],
                   flows=world["flows"], schedule=world["schedule"],
                   time=p["time"], wn=world["wn"],
                   sensors=world["profile"].get("sensors") or {},
                   params=p, mapping=mapping, drift=p["scenario"]["drift"],
                   label=f"{world['cfg'].network}:{world['sim_hash']}")


def from_world(world: dict) -> Probe:
    return Probe.from_world(world)


def from_lab_world(w) -> Probe:
    """From a cached `fedwater_lab.World` (reads its clone tree)."""
    clone = w.clone
    demand = pd.read_parquet(clone / "data" / "02_intermediate" / "demand_series.parquet")
    mapping = dict(zip(sorted(w.districts),
                       [tuple(x) for x in (w.scenario.get("income_landuse_mapping") or [])]))
    import yaml
    dy = yaml.safe_load((clone / "data" / "01_raw" / "districts.yml").read_text())
    return Probe(districts=dy["districts"], assets=dy.get("assets") or {},
                 demand=demand, pressures=w.pressures, flows=w.flows,
                 schedule=w.drift_schedule, time=w.time, wn=w.wn(),
                 sensors=w.sensors_manual, params=w.params, mapping=mapping,
                 drift=w.drift, label=w.tag)


# ==========================================================================
# candidate universe and topological annotation
# ==========================================================================
def candidates(P: Probe, kinds=("pressure", "flow"), subtree: bool = True,
               consumers_only: bool = True) -> pd.DataFrame:
    """Every gauge that could be placed, annotated a priori.

    Columns
    -------
    id, kind, district      the candidate and the district it would report to
    role                    internal | boundary | external | storage_feed
    shipped                 is it in the bundle's `sensors:` block
    elev / base_demand      (pressure) the node's own properties
    mean, cv                the aligned series' level and variability
    served_own              (flow, `subtree=True`) share of the demand fed
                            THROUGH this link that belongs to `district`

    `consumers_only` drops junctions that carry no demand -- trunk nodes and
    pump suction/discharge nodes. They have a pressure and no customer behind
    it, so a gauge there measures the network, not a client (KY7's I-Pump-1
    sits at -8.6 mca by design). Pass False to include them deliberately.

    `served_own` is the a-priori prediction of the empirical ranking: a link
    whose downstream subtree is 90% District_A demand should look like
    District_A. Where the two disagree, the disagreement is the finding --
    usually a loop that the directed-subtree approximation over-counts, or a
    pump/valve crossing that makes the direction meaningless.
    """
    rows = []
    if "pressure" in kinds:
        rows += _pressure_candidates(P, consumers_only=consumers_only)
    if "flow" in kinds:
        rows += _flow_candidates(P, subtree=subtree)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no candidates; check the district partition")
    ship = {_PREFIX[k] + str(i)
            for d, cfg in (P.sensors or {}).items()
            for k in ("pressure", "flow") for i in (cfg.get(k) or [])}
    df["shipped"] = df["id"].map(lambda x: x in ship)
    return df.sort_values(["kind", "district", "id"]).reset_index(drop=True)


def _pressure_candidates(P: Probe, consumers_only: bool = True) -> list[dict]:
    out = []
    for node, district in P.node2district.items():
        if node not in P.pressures.columns:
            continue
        # Demand is read from the SERIES, not from the model: `run_hydraulics`
        # rewrites every junction's base_value to 1 L/s, so a wn that has been
        # through it reports the same base for a trunk node and a consumer.
        base = float(P.demand[node].mean()) if node in P.demand.columns else 0.0
        if consumers_only and not base > 0:
            continue
        elev = np.nan
        if P.wn is not None:
            try:
                elev = float(getattr(P.wn.get_node(node), "elevation", np.nan))
            except Exception:
                pass
        out.append({"id": _PREFIX["pressure"] + node, "element": node,
                    "kind": "pressure",
                    "district": district, "role": "internal",
                    "elev": elev, "base_demand": base, "served_own": np.nan})
    return out


def _link_endpoints(P: Probe):
    if P.wn is None:
        raise ValueError("flow candidates need the wntr model (Probe.wn)")
    out = {}
    for name in P.wn.link_name_list:
        link = P.wn.get_link(name)
        out[name] = (link.start_node_name, link.end_node_name,
                     type(link).__name__.lower())
    return out


def _flow_candidates(P: Probe, subtree: bool = True) -> list[dict]:
    ends = _link_endpoints(P)
    served = _served_shares(P, ends) if subtree else {}
    mean_q = P.flows.drop(columns=["month"], errors="ignore").mean()
    out = []
    for name, (u, v, ltype) in ends.items():
        if name not in P.flows.columns:
            continue
        du = P.node2district.get(u) or P.asset2district.get(u)
        dv = P.node2district.get(v) or P.asset2district.get(v)
        if du is None and dv is None:
            role, district = "external", None
        elif du == dv:
            role, district = "internal", du
        elif du is None or dv is None:
            role, district = "external", du or dv
        else:
            # A crossing is assigned to the district it DELIVERS INTO (mean
            # flow direction), not to whichever endpoint the .inp wrote first.
            # It still scores against every district through `r_res_best`.
            role = "boundary"
            district = dv if float(mean_q.get(name, 0.0)) >= 0 else du
        # a link touching a district's own storage is internal to it
        if role == "boundary" and (u in P.asset2district or v in P.asset2district):
            role = "storage_feed"
        if district is None:
            continue
        sh = served.get(name, {})
        out.append({"id": _PREFIX["flow"] + name, "element": name, "kind": "flow",
                    "district": district, "role": role, "link_type": ltype,
                    "elev": np.nan, "base_demand": np.nan,
                    "served_own": float(sh.get(district, np.nan)) if sh else np.nan,
                    "served_top": (max(sh, key=sh.get) if sh else None),
                    "from_district": du, "to_district": dv})
    return out


def _served_shares(P: Probe, ends: dict) -> dict:
    """For each link: the district mix of the demand fed THROUGH it.

    Direction is the sign of the link's mean flow. The downstream set is
    everything reachable from the downstream endpoint once the link itself is
    cut. On a tree this is exact; on a looped network it OVER-counts, because
    a node fed by two paths is charged to both. It is an annotation, never a
    score -- the empirical battery is what ranks.
    """
    import networkx as nx

    mean_q = P.flows.drop(columns=["month"], errors="ignore").mean()
    base = {}
    for d, nodes in P.districts.items():
        for n in nodes:
            base[n] = float(P.demand[n].mean()) if n in P.demand.columns else 0.0

    # MultiGraph, not Graph: parallel pipes between the same pair are common
    # (D-Town has several), and in a simple graph cutting one would cut both.
    G = nx.MultiGraph()
    for name, (u, v, _) in ends.items():
        if name in P.flows.columns:
            G.add_edge(u, v, key=name)

    out = {}
    for name, (u, v, _) in ends.items():
        if name not in P.flows.columns or not G.has_edge(u, v, key=name):
            continue
        q = float(mean_q.get(name, 0.0))
        down = v if q >= 0 else u
        G.remove_edge(u, v, key=name)
        try:
            reach = nx.node_connected_component(G, down)
        except Exception:
            reach = set()
        finally:
            G.add_edge(u, v, key=name)
        tot, mix = 0.0, {}
        for n in reach:
            b = base.get(n, 0.0)
            if b > 0:
                d = P.node2district[n]
                mix[d] = mix.get(d, 0.0) + b
                tot += b
        out[name] = {d: s / tot for d, s in mix.items()} if tot > 0 else {}
    return out


# ==========================================================================
# aligned weekly profiles
# ==========================================================================
def _series_matrix(P: Probe, cand: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """(n_cand, n_steps) raw series over a step window, IN CANDIDATE ORDER.

    Order matters more than it looks: every table downstream is keyed by row
    position against `cand`, so a reordering here silently mislabels every
    score. The two kinds are filled by boolean mask into a pre-allocated
    array rather than stacked, which makes the order structural.
    """
    kinds = cand["kind"].to_numpy()
    elems = cand["element"].to_numpy()
    out = np.empty((len(cand), hi - lo), dtype=float)
    for kind, src in (("pressure", P.pressures), ("flow", P.flows)):
        m = kinds == kind
        if m.any():
            out[m] = src.iloc[lo:hi][list(elems[m])].to_numpy(float).T
    return out


def profiles(P: Probe, cand: pd.DataFrame | None, phase: str) -> dict:
    """Everything the battery reads, for one phase.

    Returns
    -------
    sensor  (n_cand, 7*steps_day) ALIGNED weekly profile, raw units
    demand  (n_districts, 7*steps_day) district demand weekly profile
    total   (7*steps_day,) network aggregate demand -- the common mode
    sign    (n_cand,) the alignment applied (+1 pressure is negated, flow is
            oriented by the sign of its mean)
    """
    lo, hi = P.month_slice(phase)
    sd = P.steps_day

    dd = P.district_demand.iloc[lo:hi]
    demand = np.atleast_2d(weekly_profile(dd.to_numpy(float), sd, step0=lo))
    total = weekly_profile(dd.to_numpy(float).sum(axis=1), sd, step0=lo)

    if cand is None or not len(cand):
        return {"sensor": np.empty((0, 7 * sd)), "demand": demand,
                "total": total, "sign": np.empty(0),
                "districts": list(dd.columns), "phase": phase,
                "window": (lo, hi)}

    raw = _series_matrix(P, cand, lo, hi)
    kinds = cand["kind"].to_numpy()
    sign = np.where(kinds == "pressure", -1.0, 1.0)
    flow = kinds == "flow"
    sign = np.where(flow & (raw.mean(axis=1) < 0), -1.0, sign)
    sensor = weekly_profile((raw * sign[:, None]).T, sd, step0=lo)

    return {"sensor": sensor, "demand": demand, "total": total, "sign": sign,
            "districts": list(dd.columns), "phase": phase, "window": (lo, hi)}


# ==========================================================================
# the battery
# ==========================================================================
def battery(P: Probe, cand: pd.DataFrame | None = None,
            max_lag_h: float = 3.0) -> pd.DataFrame:
    """One row per (candidate, phase). The ranking table.

    Columns
    -------
    r_raw_own / r_res_own   correlation with own-district demand, before and
                            after the network common mode is projected out
    r_res_best / best_district / margin
                            which district this sensor actually looks like,
                            and by how much it beats the runner-up
    lag_h / r_lag           best circular lag and the correlation there; a
                            non-zero lag is storage or transit delay, and on a
                            tankless network it should be 0 everywhere
    c_night / c_weekend     additive contrasts on the z-scored weekly profile
    d_night / d_weekend     |contrast - own district's contrast|
    """
    cand = candidates(P) if cand is None else cand
    max_lag = int(round(max_lag_h / float(P.time["resolution_h"])))
    rows = []
    per_phase = {}

    for phase in ("init", "final"):
        pr = profiles(P, cand, phase)
        per_phase[phase] = pr
        S, D = pr["sensor"], pr["demand"]
        names = pr["districts"]
        own = np.array([names.index(d) if d in names else -1 for d in cand["district"]])

        r_raw = _corr_rows(S, D)
        Sr, Dr = _project_out(zscore(S), zscore(pr["total"])), \
            _project_out(zscore(D), zscore(pr["total"]))
        r_res = _corr_rows(Sr, Dr)

        best = np.argmax(r_res, axis=1)
        srt = np.sort(r_res, axis=1)
        margin = srt[:, -1] - srt[:, -2] if r_res.shape[1] > 1 else srt[:, -1]
        # mean correlation against every district that is NOT this candidate's
        # own -- the baseline the own-district number has to beat.
        n_d = r_res.shape[1]
        other = np.full(len(cand), np.nan)
        for i, o in enumerate(own):
            if o >= 0 and n_d > 1:
                other[i] = (r_res[i].sum() - r_res[i, o]) / (n_d - 1)

        # Regime view. Comparing against a district that carries the SAME
        # regime token is not a test of anything -- the demand model gave the
        # two the same signature -- so the peer set here is districts whose
        # token differs.
        reg = P.regimes(phase)
        tok = np.array([reg.get(d, "?") for d in names])
        own_tok = np.array([reg.get(d, "?") for d in cand["district"]])
        best_tok = tok[best]
        margin_reg = np.full(len(cand), np.nan)
        other_diff = np.full(len(cand), np.nan)
        for i, o in enumerate(own):
            if o < 0:
                continue
            diff = tok != own_tok[i]
            if diff.any():
                margin_reg[i] = r_res[i, o] - r_res[i, diff].max()
                other_diff[i] = r_res[i, diff].mean()

        lag, r_lag = _best_lag(S, D, own, max_lag)
        cs = _contrasts(S, P.steps_day)
        cd = _contrasts(D, P.steps_day)

        for i in range(len(cand)):
            o = own[i]
            rows.append({
                "id": cand["id"].iat[i], "kind": cand["kind"].iat[i],
                "district": cand["district"].iat[i], "role": cand["role"].iat[i],
                "shipped": bool(cand["shipped"].iat[i]), "phase": phase,
                "sign": float(pr["sign"][i]),
                "r_raw_own": float(r_raw[i, o]) if o >= 0 else np.nan,
                "r_res_own": float(r_res[i, o]) if o >= 0 else np.nan,
                "r_res_other": float(other[i]),
                "r_res_other_regime": float(other_diff[i]),
                "own_regime": str(own_tok[i]), "best_regime": str(best_tok[i]),
                "regime_hit": bool(own_tok[i] == best_tok[i]),
                "margin_regime": float(margin_reg[i]),
                "r_res_best": float(r_res[i, best[i]]),
                "best_district": names[best[i]],
                "margin": float(margin[i]),
                "lag_h": float(lag[i] * float(P.time["resolution_h"])),
                "r_lag": float(r_lag[i]),
                "c_night": float(cs[i, 0]), "c_weekend": float(cs[i, 1]),
                "d_night": abs(float(cs[i, 0] - cd[o, 0])) if o >= 0 else np.nan,
                "d_weekend": abs(float(cs[i, 1] - cd[o, 1])) if o >= 0 else np.nan,
            })

    df = pd.DataFrame(rows)
    resp = _response(per_phase, cand)
    df = df.merge(resp, on="id", how="left")
    return df


def _best_lag(S: np.ndarray, D: np.ndarray, own: np.ndarray, max_lag: int):
    """Circular cross-correlation of each sensor against its own district.

    The week wraps, so a circular shift is the right null: it moves the
    profile in time without inventing or deleting samples.
    """
    n = len(S)
    lag = np.zeros(n, dtype=int)
    best = np.zeros(n)
    Sz, Dz = zscore(S), zscore(D)
    T = Sz.shape[1]
    for i in range(n):
        o = own[i]
        if o < 0:
            best[i] = np.nan
            continue
        d = Dz[o]
        rs = [float((np.roll(Sz[i], k) @ d) / T) for k in range(-max_lag, max_lag + 1)]
        j = int(np.argmax(rs))
        lag[i], best[i] = j - max_lag, rs[j]
    return lag, best


def _contrasts(W: np.ndarray, steps_day: int) -> np.ndarray:
    """(n, 2): night-day and weekend-weekday contrasts on the z-scored week."""
    W = np.atleast_2d(W)
    Z = zscore(W).reshape(len(W), 7, steps_day)
    h = (np.arange(steps_day) + 0.5) * (24.0 / steps_day)
    night = Z[:, :, h < 5].mean(axis=(1, 2))
    day = Z[:, :, (h >= 9) & (h < 17)].mean(axis=(1, 2))
    wknd = Z[:, 5:].mean(axis=(1, 2))
    wkdy = Z[:, :5].mean(axis=(1, 2))
    return np.stack([night - day, wknd - wkdy], axis=1)


def _response(per_phase: dict, cand: pd.DataFrame) -> pd.DataFrame:
    """How the sensor MOVED across the drift, against how its district moved.

    Both deltas are taken on z-scored weekly profiles, so this is a pure
    change of temporal signature: a sensor that only changed level scores 0.
    """
    a, b = per_phase["init"], per_phase["final"]
    names = a["districts"]
    dS = zscore(b["sensor"]) - zscore(a["sensor"])
    dD = zscore(b["demand"]) - zscore(a["demand"])
    own = np.array([names.index(d) if d in names else -1 for d in cand["district"]])
    r, mag = np.full(len(cand), np.nan), np.linalg.norm(dS, axis=1)
    for i in range(len(cand)):
        if own[i] < 0:
            continue
        x, y = dS[i], dD[own[i]]
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        r[i] = float(x @ y / (nx * ny)) if nx > EPS and ny > EPS else 0.0
    return pd.DataFrame({"id": cand["id"].to_numpy(), "response": r,
                         "response_mag": mag})


# ==========================================================================
# read-outs
# ==========================================================================
def discriminative_gap(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean r_res against own district minus mean against the rest, per kind.

    The single number that says whether a channel can tell districts apart at
    all -- on Graeme this came out ~0.004 for pressure against ~0.23 for flow,
    which is why pressure-only placements never separated clients. Reproduce
    it on any new network before trusting a placement built there.

    `hit_rate` is the complementary categorical view: how often the district a
    candidate most resembles is in fact its own.
    """
    cols = ["r_res_own", "r_res_other", "r_res_other_regime", "margin",
            "margin_regime", "response"]
    g = scores.groupby(["phase", "kind"])[cols].mean().reset_index()
    hit = (scores.assign(hit=scores["best_district"] == scores["district"])
           .groupby(["phase", "kind"])[["hit", "regime_hit"]]
           .agg(["mean"]).droplevel(1, axis=1).reset_index())
    n = scores.groupby(["phase", "kind"]).size().rename("n").reset_index()
    g = g.merge(hit, on=["phase", "kind"]).merge(n, on=["phase", "kind"])
    g["gap"] = g["r_res_own"] - g["r_res_other"]
    g["gap_regime"] = g["r_res_own"] - g["r_res_other_regime"]
    return g[["phase", "kind", "n", "r_res_own", "r_res_other_regime",
              "gap_regime", "gap", "margin_regime", "hit", "regime_hit",
              "response"]].rename(columns={"hit": "hit_rate",
                                           "regime_hit": "regime_hit_rate"})


def rank(scores: pd.DataFrame, phase: str = "init", kind: str | None = None,
         by: str = "r_res_own", top: int = 5,
         require_hit: bool = True) -> pd.DataFrame:
    """Top candidates per (district, kind).

    `require_hit` keeps only candidates whose best-matching district IS their
    own -- a sensor that looks more like the neighbour is not a sensor for
    this client, whatever its own-district correlation.
    """
    d = scores[scores["phase"] == phase]
    if kind:
        d = d[d["kind"] == kind]
    if require_hit:
        d = d[d["best_district"] == d["district"]]
    cols = ["id", "kind", "district", "own_regime", "role", "shipped",
            "r_raw_own", "r_res_own", "margin_regime", "regime_hit", "lag_h",
            "c_weekend", "response"]
    return (d.sort_values(by, ascending=False)
            .groupby(["district", "kind"], group_keys=False)
            .head(top)[cols]
            .sort_values(["district", "kind", by], ascending=[True, True, False])
            .reset_index(drop=True))


def select(scores: pd.DataFrame, phase: str = "init", n_pressure: int = 2,
           n_flow: int = 3, min_response: float = 0.0,
           require_hit: bool = True) -> dict:
    """A `sensors:` block: the top candidates that both resemble AND respond.

    `min_response` is the gate that separates this from a pure similarity
    ranking. Raise it to demand that the gauge actually moved with the drift.
    """
    full = scores[scores["phase"] == phase]
    d = full
    if require_hit:
        d = d[d["best_district"] == d["district"]]
    d = d[d["response"].fillna(-1) >= min_response]

    out, relaxed = {}, []
    for district in sorted(full["district"].dropna().unique()):
        pick = {}
        for kind, n in (("pressure", n_pressure), ("flow", n_flow)):
            sub = d[(d["district"] == district) & (d["kind"] == kind)]
            if not len(sub):
                # An empty channel would produce a client CSV with no columns
                # of that kind, which `preprocess_clients` reads as a different
                # feature geometry per client. Fall back to the unfiltered pool
                # and SAY SO rather than shipping a ragged placement.
                sub = full[(full["district"] == district) & (full["kind"] == kind)]
                if len(sub):
                    relaxed.append(f"{district}/{kind}")
            sub = sub.sort_values("r_res_own", ascending=False)
            pick[kind] = [i[len(_PREFIX[kind]):] for i in sub["id"].head(n)]
        out[district] = pick
    if relaxed:
        print("select: no candidate passed the filters for "
              + ", ".join(relaxed) + " -- fell back to the best available")
    return out


def sensors_yaml(sel: dict) -> str:
    """The selection, in the exact shape a bundle's `profile.yml` wants."""
    lines = ["sensors:"]
    for d in sorted(sel):
        p = ", ".join(f'"{x}"' for x in sel[d].get("pressure", []))
        q = ", ".join(f'"{x}"' for x in sel[d].get("flow", []))
        lines.append(f"  {d}: {{pressure: [{p}], flow: [{q}]}}")
    return "\n".join(lines)


def compare_to_shipped(scores: pd.DataFrame, phase: str = "init") -> pd.DataFrame:
    """Where the bundle's own placement sits in the candidate ranking.

    A percentile, not a pass/fail: the shipped rule (largest base demand,
    highest elevation, largest mean |flow|) was never chosen to maximise shape
    resemblance, so a low percentile is information about the rule, not a bug.
    """
    d = scores[scores["phase"] == phase].copy()
    d["pct"] = d.groupby(["district", "kind"])["r_res_own"].rank(pct=True)
    s = d[d["shipped"]]
    return s[["id", "kind", "district", "role", "r_res_own", "margin_regime",
              "regime_hit", "response", "pct"]].sort_values(
                  ["district", "kind"]).reset_index(drop=True)


def phase_delta(P: Probe) -> pd.DataFrame:
    """Per district: what the drift did to demand, and what the sensors saw.

    The three columns that matter together. `d_demand_%` is the level move,
    `shape_shift` is the part that survives a per-client MinMax scaler
    (1 - corr between the z-scored weekly profiles of the two phases), and the
    pressure/flow columns say whether either reached an observable. A large
    `shape_shift` with a flat `d_pressure` is a district whose drift is real
    and invisible from its gauges.
    """
    cand = candidates(P)
    a, b = profiles(P, None, "init"), profiles(P, None, "final")
    la, ha = P.month_slice("init")
    lb, hb = P.month_slice("final")
    reg_i, reg_f = P.regimes("init"), P.regimes("final")

    rows = []
    for j, d in enumerate(a["districts"]):
        za, zb = zscore(a["demand"][j]), zscore(b["demand"][j])
        pcols = [c for c in P.districts[d] if c in P.pressures.columns]
        links = [l for l in cand.loc[(cand["district"] == d)
                                     & (cand["kind"] == "flow")
                                     & (cand["role"].isin(["boundary", "storage_feed"])),
                                     "element"] if l in P.flows.columns]

        def win(df, cols, lo, hi, absolute=False):
            if not cols:
                return np.nan
            v = df.iloc[lo:hi][cols]
            v = v.abs() if absolute else v
            return float(v.to_numpy().mean())

        dem_a = float(P.district_demand[d].iloc[la:ha].mean())
        dem_b = float(P.district_demand[d].iloc[lb:hb].mean())
        pa, pb = win(P.pressures, pcols, la, ha), win(P.pressures, pcols, lb, hb)
        fa = win(P.flows, links, la, ha, True)
        fb = win(P.flows, links, lb, hb, True)
        rows.append({
            "district": d, "regime": f"{reg_i.get(d)}->{reg_f.get(d)}",
            "drifted": d == P.drift.get("tgt_district"),
            "demand_init": dem_a, "demand_final": dem_b,
            "d_demand_%": 100 * (dem_b / dem_a - 1) if dem_a else np.nan,
            "shape_shift": float(1 - np.corrcoef(za, zb)[0, 1]),
            "pressure_init": pa, "pressure_final": pb, "d_pressure": pb - pa,
            "boundary_flow_init": fa, "boundary_flow_final": fb,
            "d_boundary_%": 100 * (fb / fa - 1) if fa else np.nan,
        })
    return pd.DataFrame(rows).round(3)


# ==========================================================================
# clustering: do the sensors carry district identity at all?
# ==========================================================================
def cluster(P: Probe, cand: pd.DataFrame, phase: str = "init",
            kind: str | None = None, residualize: bool = True,
            n_clusters: int | None = None) -> dict:
    """Agglomerative clustering on correlation distance between weekly shapes.

    If the sensors of a district cluster together, a federated prototype built
    from them can carry district identity. If they do not, no amount of FL
    machinery recovers it -- the information is not in the channel.

    `residualize=True` removes the network common mode first, which is the
    honest test: without it the clustering mostly recovers "everything looks
    like the aggregate".
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    sel = cand if kind is None else cand[cand["kind"] == kind]
    sel = sel.reset_index(drop=True)
    pr = profiles(P, sel, phase)
    W = zscore(pr["sensor"])
    if residualize:
        W = _project_out(W, zscore(pr["total"]))
    R = _corr_rows(W, W)
    Dm = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(Dm, 0.0)

    labels = sel["district"].to_numpy()
    reg = P.regimes(phase)
    rlabels = np.array([reg.get(d, "?") for d in labels])
    k = n_clusters or len(set(labels))
    assign = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                     linkage="average").fit_predict(Dm)
    # Two ARIs on purpose. Against DISTRICTS is the placement question ("can a
    # gauge be traced to its client"); against REGIMES is the thesis question
    # ("is the land-use signature legible at all"). Districts that share a
    # regime cannot be separated by shape, so a low district-ARI with a high
    # regime-ARI is the model working, not failing.
    sil = float(silhouette_score(Dm, labels, metric="precomputed")) \
        if len(set(labels)) > 1 else np.nan
    return {"D": pd.DataFrame(Dm, index=sel["id"], columns=sel["id"]),
            "assign": pd.Series(assign, index=sel["id"], name="cluster"),
            "labels": pd.Series(labels, index=sel["id"], name="district"),
            "regimes": pd.Series(rlabels, index=sel["id"], name="regime"),
            "ari": float(adjusted_rand_score(labels, assign)),
            "ari_regime": float(adjusted_rand_score(rlabels, assign)),
            "silhouette_true_labels": sil, "phase": phase, "kind": kind or "all",
            "cand": sel}


# ==========================================================================
# canonical land-use template -- what the district SHOULD look like
# ==========================================================================
def canonical_profile(P: Probe, income: str, land_use: str,
                      plots: float = 500.0, seed: int = 0) -> np.ndarray:
    """The pure sector signature for one (income, land_use), as a weekly profile.

    Built with the SHIPPED `sector_day_shape` and the same volume-share mixing
    `synthesize_demands` uses, at a large plot count so crowd noise is
    negligible. This is the target a drift is aiming at; comparing it with the
    district's realised demand profile says how much of the assigned land use
    actually reached the water.
    """
    from bridge import nodes_module
    DS = nodes_module("demand_synthesis")
    US = nodes_module("urban_scenario")

    lu = P.params["land_use"]
    sectors, mix = lu["sectors"], lu["mix"][land_use]
    inc = US.build_income_factors(P.params["buildings"])
    intensity = US.plot_intensity(inc, income, lu)

    vol = {s: w * intensity[s] for s, w in mix.items()}
    tot = sum(vol.values())
    sd = P.steps_day
    t = (np.arange(sd) + 0.5) * (24.0 / sd)
    rng = np.random.default_rng(seed)

    series = np.zeros(7 * sd)
    for s, v in vol.items():
        coh = np.concatenate([
            DS.sector_day_shape(t, sectors[s], plots * mix[s], rng, d >= 5,
                                P.params["patterns"])
            for d in range(7)])
        series += (v / tot) * (coh / coh.mean())
    return series


def template_fit(P: Probe) -> pd.DataFrame:
    """Per district and phase: realised demand profile vs canonical template.

    The regime the district was ASSIGNED, and (for the drift target in the
    final phase) the regime it was moved TO. A low correlation here means the
    land-use map is not reaching the demand series, and no sensor ranking
    downstream can fix that.
    """
    rows = []
    for phase in ("init", "final"):
        pr = profiles(P, None, phase)
        for j, d in enumerate(pr["districts"]):
            inc, lu = P.mapping.get(d, (None, None))
            if phase == "final" and d == P.drift.get("tgt_district"):
                inc, lu = P.drift.get("to_income", inc), P.drift.get("to_land_use", lu)
            if inc is None:
                continue
            tmpl = canonical_profile(P, inc, lu)
            r = float(_corr_rows(pr["demand"][j][None, :], tmpl[None, :])[0, 0])
            rows.append({"district": d, "phase": phase, "regime": f"{inc}/{lu}",
                         "r_template": r})
    return pd.DataFrame(rows)


# ==========================================================================
# figures
# ==========================================================================
def plot_district_response(P: Probe, axes=None):
    """Demand, pressure and boundary inflow per district, month by month.

    The link the notebook's old version was missing: a change in demand is
    only useful if it shows up in the channels the clients actually observe.
    """
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    month = P.demand["month"].to_numpy()
    tgt = P.drift.get("tgt_district")
    ph = P.phases()

    dd = P.district_demand.assign(month=month)
    dem = dd.groupby("month").mean()

    cand = candidates(P)
    pres = {}
    inflow = {}
    for d in P.names:
        cols = [c for c in P.districts[d] if c in P.pressures.columns]
        pres[d] = P.pressures[cols].mean(axis=1).groupby(month).mean() if cols else None
        links = cand[(cand["district"] == d) & (cand["kind"] == "flow")
                     & (cand["role"].isin(["boundary", "storage_feed"]))]["element"]
        links = [l for l in links if l in P.flows.columns]
        inflow[d] = (P.flows[links].abs().sum(axis=1).groupby(month).mean()
                     if links else None)

    for ax, data, ttl in ((axes[0], dem, "district demand (L/s)"),
                          (axes[1], pd.DataFrame(pres), "mean district pressure (mca)"),
                          (axes[2], pd.DataFrame(inflow),
                           "boundary |flow| into district (L/s)")):
        for d in P.names:
            if d not in data or data[d] is None:
                continue
            hot = d == tgt
            ax.plot(data.index, data[d], marker="o", ms=3, label=d,
                    lw=2.2 if hot else 1.0, alpha=1.0 if hot else 0.45)
        ax.axvspan(ph["init"][0], ph["init"][1] - 1, color="tab:blue", alpha=.07)
        ax.axvspan(ph["final"][0], ph["final"][1] - 1, color="tab:orange", alpha=.07)
        ax.set(xlabel="month", title=ttl)
    axes[0].legend(fontsize=6, ncol=2)
    return axes


def plot_weekly(P: Probe, district: str, scores: pd.DataFrame | None = None,
                kind: str = "flow", n: int = 2, axes=None):
    """District demand week, init vs final, with the best and worst gauges."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 3.2), sharey=True)
    cand = candidates(P)
    j = P.names.index(district)
    picks = []
    if scores is not None:
        s = scores[(scores["phase"] == "init") & (scores["district"] == district)
                   & (scores["kind"] == kind)].sort_values("r_res_own")
        picks = list(s["id"].tail(n)) + list(s["id"].head(1))

    for ax, phase in zip(axes, ("init", "final")):
        pr = profiles(P, cand, phase)
        x = np.arange(pr["demand"].shape[1]) / P.steps_day
        ax.plot(x, zscore(pr["demand"][j]), lw=2.4, color="k", label="demand")
        ax.plot(x, zscore(pr["total"]), lw=1.0, color="grey", ls="--",
                label="network total")
        for pid in picks:
            i = int(np.where(cand["id"].to_numpy() == pid)[0][0])
            ax.plot(x, zscore(pr["sensor"][i]), lw=1.2, alpha=.85, label=pid)
        for d in (5, 6):
            ax.axvspan(d, d + 1, color="grey", alpha=.12, zorder=0)
        ax.set(xlabel="day of week", title=f"{district} -- {phase}")
    axes[0].set_ylabel("z-scored weekly profile")
    axes[1].legend(fontsize=6, ncol=2)
    return axes


def plot_selection_map(scores: pd.DataFrame, phase: str = "init", ax=None):
    """Resemblance x response. The upper right is where a prototype comes from."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.4))
    d = scores[scores["phase"] == phase]
    for kind, m in (("pressure", "o"), ("flow", "^")):
        s = d[d["kind"] == kind]
        hit = s["best_district"] == s["district"]
        ax.scatter(s.loc[~hit, "r_res_own"], s.loc[~hit, "response"], s=12,
                   marker=m, alpha=.25, c="lightgrey")
        ax.scatter(s.loc[hit, "r_res_own"], s.loc[hit, "response"], s=16,
                   marker=m, alpha=.7, label=f"{kind} (own-district best)")
    sh = d[d["shipped"]]
    ax.scatter(sh["r_res_own"], sh["response"], s=90, facecolors="none",
               edgecolors="crimson", lw=1.4, label="shipped placement")
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel="resemblance  r_res(own district)",
           ylabel="response  corr(dsensor, ddemand)",
           title=f"sensor selection map -- {phase}")
    ax.legend(fontsize=7)
    return ax


def plot_cluster(res: dict, ax=None):
    """Correlation-distance matrix, ordered by true district. ARI in the title."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    order = res["labels"].sort_values().index
    M = res["D"].loc[order, order].to_numpy()
    ax.imshow(M, cmap="viridis_r", vmin=0, vmax=2)
    lab = res["labels"].loc[order].to_numpy()
    edges = np.where(lab[1:] != lab[:-1])[0] + 0.5
    for e in edges:
        ax.axhline(e, color="w", lw=.8)
        ax.axvline(e, color="w", lw=.8)
    ax.set(xticks=[], yticks=[],
           title=f"{res['kind']} shape distance, {res['phase']} -- ARI "
                 f"{res['ari']:.2f} (district) / {res['ari_regime']:.2f} (regime)")
    return ax


def plot_served_vs_measured(scores: pd.DataFrame, cand: pd.DataFrame,
                            phase: str = "init", ax=None):
    """Topological prediction against the empirical ranking, for flow links.

    Agreement means the ranking is explainable from the network alone -- which
    would make placement transferable to a network with no simulation yet.
    Disagreement is where the hydraulics are doing something the topology does
    not say.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))
    d = (scores[(scores["phase"] == phase) & (scores["kind"] == "flow")]
         .merge(cand[["id", "served_own", "role"]], on="id", how="left",
                suffixes=("", "_c")))
    d = d.dropna(subset=["served_own"])
    for role, grp in d.groupby("role"):
        ax.scatter(grp["served_own"], grp["r_res_own"], s=14, alpha=.6, label=role)
    if len(d) > 2:
        r = float(np.corrcoef(d["served_own"], d["r_res_own"].fillna(0))[0, 1])
        ax.set_title(f"served share vs measured resemblance (r={r:.2f})")
    ax.set(xlabel="share of downstream demand owned by the district",
           ylabel="r_res(own district)")
    ax.legend(fontsize=7)
    return ax
